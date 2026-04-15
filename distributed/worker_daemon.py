"""
Worker Daemon for Distributed 3DGS Pipeline
A persistent process that runs on each worker machine, connects to Redis,
and continuously dequeues and executes chunk training jobs
"""

import os
import sys
import json
import socket
import time
import signal
import threading
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Redis Queue imports
import redis
from rq import Worker, Queue, Connection
from rq.timeouts import JobTimeoutException
from rq.job import Job

# Local imports
from shared_storage import load_config
from rq_tasks import execute_chunk_training, worker_health_check, cleanup_worker_temp


class WorkerDaemon:
    """Persistent worker daemon for distributed 3DGS pipeline."""
    
    def __init__(self, config_path: str, gpu_id: int = 0):
        """
        Initialize worker daemon.
        
        Args:
            config_path: Path to configuration file
            gpu_id: GPU ID to use for this worker
        """
        self.config_path = config_path
        self.gpu_id = gpu_id
        self.hostname = socket.gethostname()
        self.stop_requested = False
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup GPU environment
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Setup logging
        self.setup_logging()
        
        # Connect to Redis
        self.redis_conn = self.connect_redis()
        
        # Setup RQ worker
        self.rq_worker = None
        
        # Heartbeat thread
        self.heartbeat_thread = None
        self.heartbeat_stop_event = threading.Event()
        
        self.logger.info(f"Worker daemon initialized: {self.hostname} GPU {gpu_id}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup root logger
        self.logger = logging.getLogger(f'worker_{self.hostname}_{self.gpu_id}')
        self.logger.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if shared storage available)
        try:
            storage_config = self.config.get("storage", {})
            if storage_config.get("backend") == "nfs":
                log_dir = Path(storage_config.get("base_path")) / "worker_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / f"{self.hostname}_gpu{self.gpu_id}.log"
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.warning(f"Failed to setup file logging: {e}")
    
    def connect_redis(self) -> redis.Redis:
        """Connect to Redis server."""
        redis_config = self.config.get("redis", {})
        
        try:
            redis_conn = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                password=redis_config.get("password"),
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10
            )
            
            # Test connection
            redis_conn.ping()
            self.logger.info(f"Connected to Redis: {redis_config.get('host')}:{redis_config.get('port')}")
            
            return redis_conn
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def register_worker(self):
        """Register worker with Redis."""
        worker_info = {
            "hostname": self.hostname,
            "gpu_id": str(self.gpu_id),
            "status": "IDLE",
            "registered_at": time.time(),
            "pid": os.getpid(),
            "config": {
                "max_gaussians": self.config.get("worker", {}).get("max_gaussians", 200000),
                "num_iters": self.config.get("worker", {}).get("num_iters", 3000)
            }
        }
        
        self.redis_conn.hset(
            "workers:registry",
            f"{self.hostname}:{self.gpu_id}",
            json.dumps(worker_info)
        )
        
        self.logger.info(f"Worker registered: {self.hostname}:{self.gpu_id}")
    
    def unregister_worker(self):
        """Unregister worker from Redis."""
        self.redis_conn.hdel("workers:registry", f"{self.hostname}:{self.gpu_id}")
        self.redis_conn.delete(f"workers:heartbeat:{self.hostname}:{self.gpu_id}")
        
        self.logger.info(f"Worker unregistered: {self.hostname}:{self.gpu_id}")
    
    def heartbeat_worker(self):
        """Send heartbeat to Redis every 30 seconds."""
        while not self.heartbeat_stop_event.is_set():
            try:
                heartbeat_data = {
                    "hostname": self.hostname,
                    "gpu_id": str(self.gpu_id),
                    "timestamp": time.time(),
                    "status": "IDLE" if self.rq_worker and self.rq_worker.state == "idle" else "BUSY"
                }
                
                self.redis_conn.setex(
                    f"workers:heartbeat:{self.hostname}:{self.gpu_id}",
                    90,  # 90 seconds TTL
                    json.dumps(heartbeat_data)
                )
                
                # Update worker status in registry
                self.redis_conn.hset(
                    "workers:registry",
                    f"{self.hostname}:{self.gpu_id}",
                    json.dumps({
                        "hostname": self.hostname,
                        "gpu_id": str(self.gpu_id),
                        "status": heartbeat_data["status"],
                        "last_heartbeat": heartbeat_data["timestamp"]
                    })
                )
                
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
            
            # Sleep for 30 seconds or until stop requested
            self.heartbeat_stop_event.wait(30)
        
        self.logger.info("Heartbeat thread stopped")
    
    def start_heartbeat(self):
        """Start heartbeat thread."""
        self.heartbeat_thread = threading.Thread(
            target=self.heartbeat_worker,
            daemon=True
        )
        self.heartbeat_thread.start()
        self.logger.info("Heartbeat thread started")
    
    def stop_heartbeat(self):
        """Stop heartbeat thread."""
        if self.heartbeat_thread:
            self.heartbeat_stop_event.set()
            self.heartbeat_thread.join(timeout=5)
    
    def setup_rq_worker(self):
        """Setup RQ worker."""
        queue_name = self.config.get("redis", {}).get("queue_name", "3dgs_chunks")
        
        with Connection(self.redis_conn):
            self.rq_worker = Worker(
                [Queue(queue_name)],
                connection=self.redis_conn,
                name=f"{self.hostname}:{self.gpu_id}",
                default_job_timeout=3600  # 1 hour timeout
            )
        
        self.logger.info(f"RQ worker setup for queue: {queue_name}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_requested = True
        
        if self.rq_worker:
            self.rq_worker.stop()
    
    def run_health_check(self):
        """Run periodic health checks."""
        try:
            health_result = worker_health_check({
                "hostname": self.hostname,
                "gpu_id": str(self.gpu_id)
            })
            
            # Publish health status
            self.redis_conn.setex(
                f"worker_health:{self.hostname}:{self.gpu_id}",
                300,  # 5 minutes TTL
                json.dumps(health_result)
            )
            
            if health_result["status"] != "healthy":
                self.logger.warning(f"Health check failed: {health_result.get('error')}")
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
    
    def run(self):
        """Main worker daemon loop."""
        try:
            # Register worker
            self.register_worker()
            
            # Start heartbeat
            self.start_heartbeat()
            
            # Setup RQ worker
            self.setup_rq_worker()
            
            # Setup signal handlers
            signal.signal(signal.SIGTERM, self.signal_handler)
            signal.signal(signal.SIGINT, self.signal_handler)
            
            self.logger.info("Worker daemon started successfully")
            
            # Run initial health check
            self.run_health_check()
            
            # Start RQ worker (blocking call)
            with Connection(self.redis_conn):
                self.rq_worker.work()
        
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        
        except Exception as e:
            self.logger.error(f"Worker daemon error: {e}")
        
        finally:
            # Cleanup
            self.logger.info("Shutting down worker daemon...")
            
            self.stop_heartbeat()
            self.unregister_worker()
            
            # Cleanup temporary files
            try:
                cleanup_result = cleanup_worker_temp()
                self.logger.info(f"Cleanup result: {cleanup_result}")
            except Exception as e:
                self.logger.error(f"Cleanup failed: {e}")
            
            self.logger.info("Worker daemon shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="3DGS Worker Daemon")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Validate config file
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("Error: CUDA not available on this machine")
            sys.exit(1)
        
        if args.gpu >= torch.cuda.device_count():
            print(f"Error: GPU {args.gpu} not available. Available GPUs: {torch.cuda.device_count()}")
            sys.exit(1)
        
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        
    except ImportError:
        print("Error: PyTorch not installed")
        sys.exit(1)
    
    # Create and run worker daemon
    daemon = WorkerDaemon(args.config, args.gpu)
    
    try:
        daemon.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
