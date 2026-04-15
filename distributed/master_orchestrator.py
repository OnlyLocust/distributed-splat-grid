"""
Master Orchestrator for Distributed 3DGS Pipeline
Runs on the master node, enqueues chunk jobs to Redis, monitors progress,
and handles OOM retries and final stitching
"""

import os
import sys
import json
import time
import sqlite3
import argparse
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Redis Queue imports
import redis
from rq import Queue, Connection
from rq.job import Job

# Local imports
from shared_storage import load_config, get_storage, read_json, write_json
from heartbeat_monitor import HeartbeatMonitor

# Import existing pipeline components
sys.path.append(str(Path(__file__).parent.parent))
from splitter import ChunkSplitter
from optimizer import CameraOptimizer


class DatabaseManager:
    """Manages SQLite state database on master node."""
    
    def __init__(self, db_path: str):
        """Initialize database manager."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema with RQ job support."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'PENDING',
                    attempt INTEGER DEFAULT 0,
                    batch_size INTEGER DEFAULT 1,
                    max_gaussians INTEGER DEFAULT 200000,
                    densify_interval INTEGER DEFAULT 100,
                    num_iters INTEGER DEFAULT 3000,
                    error_message TEXT,
                    rq_job_id TEXT,
                    enqueued_at TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
            conn.commit()
    
    def add_chunk(self, chunk_id: str, metadata: Dict[str, Any]):
        """Add chunk to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO chunks 
                (chunk_id, status, point_count, attempt, batch_size, max_gaussians, 
                 densify_interval, num_iters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (chunk_id, 'PENDING', metadata.get('point_count', 0), 0, 1,
                  200000, 100, 3000))
            conn.commit()
    
    def get_chunk_status(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk status from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM chunks WHERE chunk_id = ?', (chunk_id,))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None
    
    def get_pending_chunks(self) -> List[Dict[str, Any]]:
        """Get all pending chunks."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM chunks WHERE status = "PENDING" ORDER BY chunk_id')
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_processing_chunks(self) -> List[Dict[str, Any]]:
        """Get all processing chunks."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM chunks WHERE status = "PROCESSING" ORDER BY chunk_id')
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def update_chunk_status(self, chunk_id: str, status: str, **kwargs):
        """Update chunk status."""
        with sqlite3.connect(self.db_path) as conn:
            if status == 'PROCESSING':
                conn.execute('''
                    UPDATE chunks SET status = ?, started_at = CURRENT_TIMESTAMP 
                    WHERE chunk_id = ?
                ''', (status, chunk_id))
            elif status == 'COMPLETED':
                conn.execute('''
                    UPDATE chunks SET status = ?, completed_at = CURRENT_TIMESTAMP 
                    WHERE chunk_id = ?
                ''', (status, chunk_id))
            elif status == 'FAILED':
                conn.execute('''
                    UPDATE chunks SET status = ?, error_message = ? 
                    WHERE chunk_id = ?
                ''', (status, kwargs.get('error_message'), chunk_id))
            else:
                conn.execute('UPDATE chunks SET status = ? WHERE chunk_id = ?', 
                           (status, chunk_id))
            
            # Update additional fields
            for key, value in kwargs.items():
                if key in ['rq_job_id', 'attempt', 'max_gaussians', 'densify_interval', 'num_iters']:
                    conn.execute(f'UPDATE chunks SET {key} = ? WHERE chunk_id = ?', 
                               (value, chunk_id))
            
            conn.commit()
    
    def increment_chunk_attempt(self, chunk_id: str, new_config: Dict[str, Any]):
        """Increment chunk attempt and update config."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE chunks SET 
                    status = 'PENDING',
                    attempt = ?,
                    max_gaussians = ?,
                    densify_interval = ?,
                    num_iters = ?,
                    rq_job_id = NULL
                WHERE chunk_id = ?
            ''', (new_config['attempt'], new_config['max_gaussians'], 
                  new_config['densify_interval'], new_config['num_iters'], chunk_id))
            conn.commit()
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM chunks ORDER BY chunk_id')
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def reset_processing_chunks(self):
        """Reset PROCESSING chunks to PENDING (crash recovery)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('UPDATE chunks SET status = "PENDING", rq_job_id = NULL WHERE status = "PROCESSING"')
            conn.commit()


class MasterOrchestrator:
    """Master orchestrator for distributed 3DGS pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize master orchestrator."""
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Setup components
        self.storage = get_storage(self.config["storage"])
        self.db_manager = DatabaseManager(str(Path(config_path).parent / "state.db"))
        
        # Redis connection
        self.redis_conn = self.connect_redis()
        
        # RQ queue
        self.queue_name = self.config.get("redis", {}).get("queue_name", "3dgs_chunks")
        
        # Heartbeat monitor
        self.heartbeat_monitor = None
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info("Master orchestrator initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO
        
        self.logger = logging.getLogger('master_orchestrator')
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = Path(self.config_path).parent / "pipeline.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def connect_redis(self) -> redis.Redis:
        """Connect to Redis server."""
        redis_config = self.config.get("redis", {})
        
        try:
            redis_conn = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                password=redis_config.get("password"),
                decode_responses=True
            )
            
            # Test connection
            redis_conn.ping()
            self.logger.info(f"Connected to Redis: {redis_config.get('host')}:{redis_config.get('port')}")
            
            return redis_conn
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def run_splitter_if_needed(self, colmap_dir: str, skip_split: bool = False):
        """Run splitter if tasks don't exist."""
        if skip_split:
            self.logger.info("Skipping splitter step")
            return
        
        # Check if tasks already exist
        if self.storage.exists("grid_manifest.json"):
            self.logger.info("Tasks already exist, skipping splitter")
            return
        
        self.logger.info("Running splitter...")
        splitter = ChunkSplitter()
        splitter.split_points(colmap_dir, "tasks", dry_run=False)
        
        # Upload tasks to shared storage if needed
        if self.config["storage"]["backend"] == "s3":
            tasks_dir = Path("tasks")
            if tasks_dir.exists():
                for file_path in tasks_dir.rglob('*'):
                    if file_path.is_file():
                        rel_path = str(file_path.relative_to("tasks"))
                        self.storage.upload_from_local(str(file_path), f"tasks/{rel_path}")
    
    def run_optimizer_if_needed(self, colmap_dir: str, skip_cull: bool = False):
        """Run optimizer if cameras don't exist."""
        if skip_cull:
            self.logger.info("Skipping optimizer step")
            return
        
        # Check if cameras already exist
        manifest = read_json(self.storage, "grid_manifest.json")
        cameras_exist = all(
            self.storage.exists(f"tasks/{chunk['chunk_id']}/cameras.json")
            for chunk in manifest
        )
        
        if cameras_exist:
            self.logger.info("Cameras already exist, skipping optimizer")
            return
        
        self.logger.info("Running optimizer...")
        optimizer = CameraOptimizer()
        optimizer.optimize_all_chunks(colmap_dir, "tasks", "images", dry_run=False)
        
        # Upload cameras to shared storage if needed
        if self.config["storage"]["backend"] == "s3":
            tasks_dir = Path("tasks")
            for chunk_dir in tasks_dir.iterdir():
                if chunk_dir.is_dir() and chunk_dir.name.startswith('chunk_'):
                    cameras_file = chunk_dir / "cameras.json"
                    if cameras_file.exists():
                        rel_path = str(cameras_file.relative_to("tasks"))
                        self.storage.upload_from_local(str(cameras_file), f"tasks/{rel_path}")
    
    def initialize_chunks_from_manifest(self):
        """Initialize chunks in database from manifest."""
        manifest = read_json(self.storage, "grid_manifest.json")
        
        for chunk_info in manifest:
            chunk_id = chunk_info['chunk_id']
            
            # Check if chunk already exists
            existing = self.db_manager.get_chunk_status(chunk_id)
            if existing is None:
                # Add new chunk
                self.db_manager.add_chunk(chunk_id, chunk_info)
        
        self.logger.info(f"Initialized {len(manifest)} chunks in database")
    
    def enqueue_chunk(self, chunk_id: str, chunk_config: Dict[str, Any]) -> str:
        """Enqueue a chunk job to Redis."""
        with Connection(self.redis_conn):
            queue = Queue(self.queue_name, connection=self.redis_conn)
            
            # Prepare task config
            task_config = self.config.copy()
            task_config["worker"] = chunk_config
            
            # Enqueue job
            job = queue.enqueue(
                "rq_tasks.execute_chunk_training",
                chunk_id=chunk_id,
                config=task_config,
                job_timeout=3600,           # 1 hour timeout
                result_ttl=86400,           # Keep result for 24h
                failure_ttl=86400,
                meta={"attempt": chunk_config.get("attempt", 0)}
            )
            
            # Update database
            self.db_manager.update_chunk_status(
                chunk_id, 
                'PROCESSING', 
                rq_job_id=job.id,
                attempt=chunk_config.get("attempt", 0)
            )
            
            self.logger.info(f"Enqueued {chunk_id} as job {job.id}")
            return job.id
    
    def enqueue_all_pending_chunks(self):
        """Enqueue all pending chunks in priority order."""
        pending_chunks = self.db_manager.get_pending_chunks()
        
        if not pending_chunks:
            self.logger.info("No pending chunks to enqueue")
            return
        
        # Sort by point count (smallest first)
        pending_chunks.sort(key=lambda x: x.get('point_count', 0))
        
        self.logger.info(f"Enqueuing {len(pending_chunks)} pending chunks")
        
        for chunk in pending_chunks:
            chunk_id = chunk['chunk_id']
            
            # Prepare chunk config
            chunk_config = {
                "attempt": chunk.get('attempt', 0),
                "max_gaussians": chunk.get('max_gaussians', 200000),
                "densify_interval": chunk.get('densify_interval', 100),
                "num_iters": chunk.get('num_iters', 3000)
            }
            
            self.enqueue_chunk(chunk_id, chunk_config)
    
    def handle_oom_retry(self, chunk_id: str, chunk_data: Dict[str, Any]):
        """Handle OOM retry for a chunk."""
        attempt = chunk_data.get('attempt', 0) + 1
        
        if attempt >= 3:
            self.logger.error(f"Chunk {chunk_id} exceeded max retries (3)")
            self.db_manager.update_chunk_status(
                chunk_id, 'FAILED', 
                error_message="Max OOM retries exceeded"
            )
            return False
        
        # Update config for retry
        new_config = {
            "attempt": attempt,
            "max_gaussians": chunk_data.get('max_gaussians', 200000) // 2,
            "densify_interval": chunk_data.get('densify_interval', 100) * 2,
            "num_iters": chunk_data.get('num_iters', 3000)
        }
        
        self.logger.warning(f"Retrying {chunk_id} with attempt {attempt}: {new_config}")
        
        # Update database and re-enqueue
        self.db_manager.increment_chunk_attempt(chunk_id, new_config)
        self.enqueue_chunk(chunk_id, new_config)
        
        return True
    
    def monitor_progress(self):
        """Monitor job progress and handle results."""
        processing_chunks = self.db_manager.get_processing_chunks()
        
        for chunk in processing_chunks:
            chunk_id = chunk['chunk_id']
            rq_job_id = chunk.get('rq_job_id')
            
            if not rq_job_id:
                continue
            
            try:
                # Fetch job from Redis
                job = Job.fetch(rq_job_id, connection=self.redis_conn)
                
                if job.is_finished:
                    result = job.result
                    
                    if result.get("status") == "COMPLETED":
                        self.db_manager.update_chunk_status(chunk_id, 'COMPLETED')
                        self.logger.info(f"Chunk {chunk_id} completed: {result.get('num_gaussians')} Gaussians")
                        
                    elif result.get("status") == "OOM":
                        self.handle_oom_retry(chunk_id, chunk)
                        
                    else:  # FAILED
                        error_msg = result.get("error", "Unknown error")
                        self.db_manager.update_chunk_status(chunk_id, 'FAILED', error_message=error_msg)
                        self.logger.error(f"Chunk {chunk_id} failed: {error_msg}")
                
                elif job.is_failed:
                    # RQ-level failure
                    error_msg = job.exc_info or "RQ job failed"
                    self.db_manager.update_chunk_status(chunk_id, 'FAILED', error_message=error_msg)
                    self.logger.error(f"Chunk {chunk_id} RQ job failed: {error_msg}")
                
            except Exception as e:
                self.logger.error(f"Error monitoring {chunk_id}: {e}")
    
    def check_completion(self) -> Tuple[int, int]:
        """Check if all chunks are completed or failed."""
        all_chunks = self.db_manager.get_all_chunks()
        
        completed = sum(1 for chunk in all_chunks if chunk['status'] == 'COMPLETED')
        failed = sum(1 for chunk in all_chunks if chunk['status'] == 'FAILED')
        total = len(all_chunks)
        
        return completed, failed, total
    
    def run_stitcher_if_needed(self):
        """Run stitcher if all chunks are completed."""
        completed, failed, total = self.check_completion()
        
        if completed + failed < total:
            return
        
        self.logger.info("All chunks completed, running stitcher...")
        
        # Run stitcher
        from stitcher import GaussianStitcher
        
        stitcher = GaussianStitcher()
        
        # Determine results path
        if self.config["storage"]["backend"] == "nfs":
            results_dir = self.storage.get_full_path("results")
        else:
            # For S3, download all PLY files locally first
            results_dir = "/tmp/stitcher_results"
            os.makedirs(results_dir, exist_ok=True)
            
            ply_files = self.storage.list_prefix("results/")
            for ply_file in ply_files:
                if ply_file.endswith('.ply'):
                    local_path = f"{results_dir}/{Path(ply_file).name}"
                    self.storage.download_to_local(ply_file, local_path)
        
        output_path = "final_output.ply"
        if self.config["storage"]["backend"] == "nfs":
            output_path = self.storage.get_full_path(output_path)
        
        try:
            stats = stitcher.stitch_chunks(results_dir, output_path)
            
            # Upload final result if using S3
            if self.config["storage"]["backend"] == "s3":
                self.storage.upload_from_local(output_path, "final_output.ply")
            
            self.logger.info(f"Stitching completed: {stats}")
            
        except Exception as e:
            self.logger.error(f"Stitching failed: {e}")
    
    def run(self, colmap_dir: str, skip_split: bool = False, skip_cull: bool = False, resume: bool = False):
        """Run the complete distributed pipeline."""
        self.logger.info("Starting distributed 3DGS pipeline")
        
        try:
            # Step 1: Run splitter if needed
            self.run_splitter_if_needed(colmap_dir, skip_split)
            
            # Step 2: Run optimizer if needed
            self.run_optimizer_if_needed(colmap_dir, skip_cull)
            
            # Step 3: Initialize chunks from manifest
            self.initialize_chunks_from_manifest()
            
            # Step 4: Reset processing chunks if not resuming
            if not resume:
                self.db_manager.reset_processing_chunks()
            
            # Step 5: Start heartbeat monitor
            self.heartbeat_monitor = HeartbeatMonitor(self.redis_conn, self.db_manager)
            self.heartbeat_monitor.start()
            
            # Step 6: Enqueue all pending chunks
            self.enqueue_all_pending_chunks()
            
            # Step 7: Monitor progress
            self.logger.info("Monitoring progress...")
            
            while True:
                self.monitor_progress()
                
                completed, failed, total = self.check_completion()
                
                # Print progress
                print(f"\rProgress: {completed}/{total} completed, {failed} failed", end="", flush=True)
                
                # Check if all done
                if completed + failed >= total:
                    break
                
                time.sleep(10)  # Monitor every 10 seconds
            
            print()  # New line after progress
            
            # Step 8: Run stitcher
            self.run_stitcher_if_needed()
            
            # Final summary
            completed, failed, total = self.check_completion()
            
            print(f"\n=== Pipeline Summary ===")
            print(f"Total chunks: {total}")
            print(f"Completed: {completed}")
            print(f"Failed: {failed}")
            
            if failed > 0:
                self.logger.error(f"Pipeline completed with {failed} failed chunks")
                sys.exit(1)
            else:
                self.logger.info("Pipeline completed successfully")
                sys.exit(0)
        
        except KeyboardInterrupt:
            self.logger.info("Pipeline interrupted by user")
            sys.exit(130)
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            sys.exit(1)
        
        finally:
            # Stop heartbeat monitor
            if self.heartbeat_monitor:
                self.heartbeat_monitor.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Distributed 3DGS Master Orchestrator")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--colmap_dir", required=True, help="COLMAP sparse directory")
    parser.add_argument("--skip_split", action="store_true", help="Skip splitter step")
    parser.add_argument("--skip_cull", action="store_true", help="Skip optimizer step")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    
    args = parser.parse_args()
    
    # Validate config file
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Create and run orchestrator
    orchestrator = MasterOrchestrator(args.config)
    orchestrator.run(
        colmap_dir=args.colmap_dir,
        skip_split=args.skip_split,
        skip_cull=args.skip_cull,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
