"""
Heartbeat Monitor for Distributed 3DGS Pipeline
Detects dead workers and requeues their in-flight jobs
"""

import time
import threading
import json
import logging
from typing import Dict, List, Set, Optional

import redis
from rq import Queue
from rq.job import Job


class HeartbeatMonitor(threading.Thread):
    """Monitors worker heartbeats and handles dead worker recovery."""
    
    def __init__(self, redis_conn: redis.Redis, db_manager, check_interval: int = 60):
        """
        Initialize heartbeat monitor.
        
        Args:
            redis_conn: Redis connection
            db_manager: Database manager instance
            check_interval: Check interval in seconds (default: 60)
        """
        super().__init__(daemon=True)
        self.redis_conn = redis_conn
        self.db_manager = db_manager
        self.check_interval = check_interval
        self.stop_event = threading.Event()
        self.logger = logging.getLogger('heartbeat_monitor')
        
        self.logger.info("Heartbeat monitor initialized")
    
    def stop(self):
        """Stop the heartbeat monitor."""
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=5)
        self.logger.info("Heartbeat monitor stopped")
    
    def get_registered_workers(self) -> Dict[str, Dict]:
        """Get all registered workers."""
        try:
            workers_data = self.redis_conn.hgetall("workers:registry")
            workers = {}
            
            for worker_key, worker_json in workers_data.items():
                try:
                    worker_info = json.loads(worker_json)
                    workers[worker_key] = worker_info
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid worker data for {worker_key}")
            
            return workers
            
        except Exception as e:
            self.logger.error(f"Failed to get registered workers: {e}")
            return {}
    
    def get_worker_heartbeat(self, worker_key: str) -> Optional[Dict]:
        """Get heartbeat data for a worker."""
        try:
            heartbeat_key = f"workers:heartbeat:{worker_key}"
            heartbeat_data = self.redis_conn.get(heartbeat_key)
            
            if heartbeat_data:
                return json.loads(heartbeat_data)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get heartbeat for {worker_key}: {e}")
            return None
    
    def detect_dead_workers(self) -> Set[str]:
        """Detect workers with expired heartbeats."""
        dead_workers = set()
        registered_workers = self.get_registered_workers()
        
        for worker_key, worker_info in registered_workers.items():
            heartbeat = self.get_worker_heartbeat(worker_key)
            
            if not heartbeat:
                # No heartbeat at all - worker is dead
                dead_workers.add(worker_key)
                self.logger.warning(f"Dead worker detected (no heartbeat): {worker_key}")
                continue
            
            # Check if heartbeat TTL expired (Redis automatically expires keys)
            # If we can't find the heartbeat key, it's expired
            if not self.redis_conn.exists(f"workers:heartbeat:{worker_key}"):
                dead_workers.add(worker_key)
                self.logger.warning(f"Dead worker detected (expired heartbeat): {worker_key}")
        
        return dead_workers
    
    def get_chunk_executing_worker(self, chunk_id: str, rq_job_id: str) -> Optional[str]:
        """Get the worker currently executing a chunk."""
        try:
            job = Job.fetch(rq_job_id, connection=self.redis_conn)
            
            if job.is_started and job.worker_name:
                return job.worker_name
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get executing worker for {chunk_id}: {e}")
            return None
    
    def cancel_and_requeue_chunk(self, chunk_id: str, rq_job_id: str, dead_worker: str):
        """Cancel a job and requeue it."""
        try:
            # Fetch the job
            job = Job.fetch(rq_job_id, connection=self.redis_conn)
            
            # Cancel the job
            job.cancel()
            self.logger.info(f"Cancelled job {rq_job_id} for chunk {chunk_id}")
            
            # Reset chunk status to PENDING
            chunk_data = self.db_manager.get_chunk_status(chunk_id)
            if chunk_data:
                # Prepare retry config (same attempt, don't increment)
                retry_config = {
                    "attempt": chunk_data.get('attempt', 0),
                    "max_gaussians": chunk_data.get('max_gaussians', 200000),
                    "densify_interval": chunk_data.get('densify_interval', 100),
                    "num_iters": chunk_data.get('num_iters', 3000)
                }
                
                # Update database
                self.db_manager.increment_chunk_attempt(chunk_id, retry_config)
                
                # Re-enqueue the job
                from master_orchestrator import MasterOrchestrator
                orchestrator = MasterOrchestrator.__new__(MasterOrchestrator)
                orchestrator.redis_conn = self.redis_conn
                orchestrator.queue_name = "3dgs_chunks"  # Default queue name
                
                new_job_id = orchestrator.enqueue_chunk(chunk_id, retry_config)
                
                self.logger.info(f"Requeued chunk {chunk_id} as job {new_job_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to requeue chunk {chunk_id}: {e}")
    
    def cleanup_dead_worker(self, worker_key: str):
        """Clean up dead worker from registry."""
        try:
            # Remove from registry
            self.redis_conn.hdel("workers:registry", worker_key)
            
            # Remove any remaining heartbeat key
            self.redis_conn.delete(f"workers:heartbeat:{worker_key}")
            
            # Remove health status
            self.redis_conn.delete(f"worker_health:{worker_key}")
            
            self.logger.info(f"Cleaned up dead worker: {worker_key}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup dead worker {worker_key}: {e}")
    
    def recover_orphaned_jobs(self, dead_workers: Set[str]):
        """Recover jobs from dead workers."""
        processing_chunks = self.db_manager.get_processing_chunks()
        
        for chunk in processing_chunks:
            chunk_id = chunk['chunk_id']
            rq_job_id = chunk.get('rq_job_id')
            
            if not rq_job_id:
                continue
            
            # Get the worker executing this chunk
            executing_worker = self.get_chunk_executing_worker(chunk_id, rq_job_id)
            
            if executing_worker and executing_worker in dead_workers:
                self.logger.warning(f"Requeuing chunk {chunk_id} from dead worker {executing_worker}")
                self.cancel_and_requeue_chunk(chunk_id, rq_job_id, executing_worker)
    
    def run(self):
        """Main monitoring loop."""
        self.logger.info(f"Heartbeat monitor started (check interval: {self.check_interval}s)")
        
        while not self.stop_event.is_set():
            try:
                # Detect dead workers
                dead_workers = self.detect_dead_workers()
                
                if dead_workers:
                    self.logger.warning(f"Found {len(dead_workers)} dead workers: {dead_workers}")
                    
                    # Recover orphaned jobs
                    self.recover_orphaned_jobs(dead_workers)
                    
                    # Clean up dead workers
                    for worker_key in dead_workers:
                        self.cleanup_dead_worker(worker_key)
                
                # Sleep until next check
                self.stop_event.wait(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                # Continue monitoring despite errors
                self.stop_event.wait(10)  # Short wait on error
        
        self.logger.info("Heartbeat monitor loop ended")


class WorkerHealthChecker(threading.Thread):
    """Periodically checks worker health status."""
    
    def __init__(self, redis_conn: redis.Redis, check_interval: int = 300):
        """
        Initialize worker health checker.
        
        Args:
            redis_conn: Redis connection
            check_interval: Check interval in seconds (default: 300 = 5 minutes)
        """
        super().__init__(daemon=True)
        self.redis_conn = redis_conn
        self.check_interval = check_interval
        self.stop_event = threading.Event()
        self.logger = logging.getLogger('worker_health_checker')
    
    def stop(self):
        """Stop the health checker."""
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=5)
        self.logger.info("Worker health checker stopped")
    
    def check_all_workers(self):
        """Check health of all registered workers."""
        try:
            # Get all registered workers
            workers_data = self.redis_conn.hgetall("workers:registry")
            
            for worker_key, worker_json in workers_data.items():
                try:
                    worker_info = json.loads(worker_json)
                    
                    # Trigger health check task
                    from rq import Queue
                    queue = Queue("health_checks", connection=self.redis_conn)
                    
                    queue.enqueue(
                        "rq_tasks.worker_health_check",
                        worker_info,
                        job_timeout=30,
                        ttl=60,
                        result_ttl=300
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to trigger health check for {worker_key}: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to check workers: {e}")
    
    def run(self):
        """Main health checking loop."""
        self.logger.info(f"Worker health checker started (check interval: {self.check_interval}s)")
        
        while not self.stop_event.is_set():
            try:
                self.check_all_workers()
                
                # Sleep until next check
                self.stop_event.wait(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Worker health checker error: {e}")
                self.stop_event.wait(30)  # Short wait on error
        
        self.logger.info("Worker health checker loop ended")


if __name__ == "__main__":
    # Test heartbeat monitor
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        import redis
        from master_orchestrator import DatabaseManager
        
        # Test connection
        redis_conn = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_conn.ping()
        
        # Test database manager
        db_manager = DatabaseManager("test_state.db")
        
        # Create and start monitor
        monitor = HeartbeatMonitor(redis_conn, db_manager, check_interval=10)
        monitor.start()
        
        print("Heartbeat monitor test running for 30 seconds...")
        time.sleep(30)
        
        monitor.stop()
        print("Test completed")
        
    except Exception as e:
        print(f"Test failed: {e}")
