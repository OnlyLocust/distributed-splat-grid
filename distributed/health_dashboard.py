"""
Live Health Dashboard for Distributed 3DGS Pipeline
Real-time terminal dashboard showing job status across all workers
"""

import os
import sys
import time
import json
import argparse
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import redis
from rq import Queue
from rq.job import Job

# Try to import rich for fancy display
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich library not available. Using plain text display.")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from shared_storage import load_config


class HealthDashboard:
    """Live health dashboard for distributed 3DGS pipeline."""
    
    def __init__(self, config_path: str, refresh_interval: int = 5):
        """
        Initialize health dashboard.
        
        Args:
            config_path: Path to configuration file
            refresh_interval: Refresh interval in seconds
        """
        self.config = load_config(config_path)
        self.refresh_interval = refresh_interval
        
        # Connect to Redis
        self.redis_conn = self.connect_redis()
        
        # Setup console
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        self.running = True
    
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
            
            redis_conn.ping()
            return redis_conn
            
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            raise
    
    def get_workers_status(self) -> List[Dict[str, Any]]:
        """Get status of all workers."""
        workers = []
        
        try:
            # Get registered workers
            workers_data = self.redis_conn.hgetall("workers:registry")
            
            for worker_key, worker_json in workers_data.items():
                try:
                    worker_info = json.loads(worker_json)
                    
                    # Get heartbeat
                    heartbeat_key = f"workers:heartbeat:{worker_key}"
                    heartbeat_data = self.redis_conn.get(heartbeat_key)
                    
                    if heartbeat_data:
                        heartbeat = json.loads(heartbeat_data)
                        worker_info.update(heartbeat)
                        worker_info['alive'] = True
                    else:
                        worker_info['alive'] = False
                        worker_info['status'] = 'DEAD'
                    
                    # Get health status
                    health_key = f"worker_health:{worker_key}"
                    health_data = self.redis_conn.get(health_key)
                    
                    if health_data:
                        health = json.loads(health_data)
                        worker_info['health'] = health
                    else:
                        worker_info['health'] = None
                    
                    workers.append(worker_info)
                    
                except json.JSONDecodeError:
                    continue
            
        except Exception as e:
            print(f"Error getting workers status: {e}")
        
        return workers
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        try:
            queue_name = self.config.get("redis", {}).get("queue_name", "3dgs_chunks")
            queue = Queue(queue_name, connection=self.redis_conn)
            
            # Get job counts
            pending = len(queue)
            started = len(queue.started_job_registry)
            finished = len(queue.finished_job_registry)
            failed = len(queue.failed_job_registry)
            
            return {
                'pending': pending,
                'processing': started,
                'completed': finished,
                'failed': failed,
                'total': pending + started + finished + failed
            }
            
        except Exception as e:
            print(f"Error getting queue stats: {e}")
            return {'pending': 0, 'processing': 0, 'completed': 0, 'failed': 0, 'total': 0}
    
    def get_recently_completed(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently completed jobs."""
        completed = []
        
        try:
            queue_name = self.config.get("redis", {}).get("queue_name", "3dgs_chunks")
            queue = Queue(queue_name, connection=self.redis_conn)
            
            # Get recent finished jobs
            finished_job_ids = queue.finished_job_registry.get_job_ids(limit)
            
            for job_id in finished_job_ids:
                try:
                    job = Job.fetch(job_id, connection=self.redis_conn)
                    
                    if job.result and isinstance(job.result, dict):
                        result = job.result
                        completed.append({
                            'chunk_id': result.get('chunk_id', 'unknown'),
                            'num_gaussians': result.get('num_gaussians', 0),
                            'worker_hostname': result.get('worker_hostname', 'unknown'),
                            'training_time': result.get('training_time', 0),
                            'completed_at': job.ended_at
                        })
                        
                except Exception:
                    continue
            
        except Exception as e:
            print(f"Error getting recent completions: {e}")
        
        return completed[:limit]
    
    def get_progress_info(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get progress information for a chunk."""
        try:
            progress_key = f"progress:{chunk_id}"
            progress_data = self.redis_conn.get(progress_key)
            
            if progress_data:
                return json.loads(progress_data)
            
        except Exception:
            pass
        
        return None
    
    def format_time(self, seconds: float) -> str:
        """Format time in human readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def calculate_eta(self, queue_stats: Dict[str, int], completed: List[Dict[str, Any]]) -> str:
        """Calculate estimated time remaining."""
        pending = queue_stats['pending']
        processing = queue_stats['processing']
        
        if pending == 0:
            return "Done"
        
        # Calculate average time per chunk from recent completions
        if completed:
            avg_time = sum(job['training_time'] for job in completed) / len(completed)
            active_workers = len([w for w in self.get_workers_status() if w.get('status') == 'BUSY'])
            
            if active_workers > 0:
                eta_seconds = (pending / active_workers) * avg_time
                return f"~{self.format_time(eta_seconds)}"
        
        return "Unknown"
    
    def display_rich_dashboard(self):
        """Display dashboard using rich library."""
        while self.running:
            try:
                # Clear screen
                self.console.clear()
                
                # Create layout
                layout = Layout()
                layout.split_column(
                    Layout(name="header", size=3),
                    Layout(name="main"),
                    Layout(name="footer", size=3)
                )
                
                layout["main"].split_row(
                    Layout(name="workers", ratio=1),
                    Layout(name="jobs", ratio=1)
                )
                
                # Header
                header_text = Text("=== 3DGS Distributed Training Dashboard ===", style="bold blue")
                header_text.append(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
                layout["header"].update(Panel(header_text))
                
                # Workers section
                workers = self.get_workers_status()
                workers_table = Table(title="WORKERS")
                workers_table.add_column("Hostname", style="cyan")
                workers_table.add_column("GPU", style="green")
                workers_table.add_column("Status", style="yellow")
                workers_table.add_column("Current Task", style="magenta")
                workers_table.add_column("Progress", style="blue")
                
                for worker in workers:
                    status = worker.get('status', 'UNKNOWN')
                    status_style = "green" if status == 'IDLE' else "yellow" if status == 'BUSY' else "red"
                    
                    current_task = ""
                    progress_bar = ""
                    
                    if status == 'BUSY' and worker.get('hostname'):
                        # Try to get current task from job registry
                        try:
                            queue_name = self.config.get("redis", {}).get("queue_name", "3dgs_chunks")
                            queue = Queue(queue_name, connection=self.redis_conn)
                            started_jobs = queue.started_job_registry.get_job_ids()
                            
                            for job_id in started_jobs:
                                job = Job.fetch(job_id, connection=self.redis_conn)
                                if job.worker_name == f"{worker['hostname']}:{worker.get('gpu_id', '0')}":
                                    current_task = job.meta.get('chunk_id', 'unknown')
                                    
                                    # Get progress
                                    progress_info = self.get_progress_info(current_task)
                                    if progress_info:
                                        percentage = progress_info.get('percentage', 0)
                                        progress_bar = f"[{percentage:.0f}%]"
                                    break
                        except:
                            pass
                    
                    workers_table.add_row(
                        worker.get('hostname', 'unknown'),
                        str(worker.get('gpu_id', '0')),
                        Text(status, style=status_style),
                        current_task,
                        progress_bar
                    )
                
                layout["workers"].update(Panel(workers_table))
                
                # Jobs section
                queue_stats = self.get_queue_stats()
                recent_completed = self.get_recently_completed()
                
                # Queue stats
                queue_text = Text()
                queue_text.append(f"Pending: {queue_stats['pending']}   ", style="yellow")
                queue_text.append(f"Processing: {queue_stats['processing']}   ", style="blue")
                queue_text.append(f"Completed: {queue_stats['completed']}   ", style="green")
                queue_text.append(f"Failed: {queue_stats['failed']}   ", style="red")
                queue_text.append(f"Total: {queue_stats['total']}", style="bold")
                
                # ETA
                eta = self.calculate_eta(queue_stats, recent_completed)
                queue_text.append(f"\nETA: {eta}", style="cyan")
                
                layout["jobs"].update(Panel(queue_text, title="QUEUE"))
                
                # Footer with recent completions
                if recent_completed:
                    recent_text = Text("RECENTLY COMPLETED:\n", style="bold")
                    for job in recent_completed[:5]:
                        recent_text.append(
                            f"{job['chunk_id']}   {job['num_gaussians']:,} Gaussians   "
                            f"{self.format_time(job['training_time'])}   {job['worker_hostname']}\n",
                            style="dim"
                        )
                    
                    layout["footer"].update(Panel(recent_text))
                else:
                    layout["footer"].update(Panel(Text("No recent completions", style="dim")))
                
                # Print layout
                self.console.print(layout)
                
                # Sleep
                time.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"Dashboard error: {e}")
                time.sleep(self.refresh_interval)
    
    def display_plain_dashboard(self):
        """Display dashboard using plain text."""
        while self.running:
            try:
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Header
                print("=== 3DGS Distributed Training Dashboard ===")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()
                
                # Workers
                workers = self.get_workers_status()
                print(f"WORKERS ({len(workers)} online):")
                print("-" * 80)
                print(f"{'Hostname':<20} {'GPU':<6} {'Status':<12} {'Current Task':<15} {'Progress'}")
                print("-" * 80)
                
                for worker in workers:
                    status = worker.get('status', 'UNKNOWN')
                    current_task = ""
                    progress = ""
                    
                    if status == 'BUSY':
                        try:
                            queue_name = self.config.get("redis", {}).get("queue_name", "3dgs_chunks")
                            queue = Queue(queue_name, connection=self.redis_conn)
                            started_jobs = queue.started_job_registry.get_job_ids()
                            
                            for job_id in started_jobs:
                                job = Job.fetch(job_id, connection=self.redis_conn)
                                if job.worker_name == f"{worker['hostname']}:{worker.get('gpu_id', '0')}":
                                    current_task = job.meta.get('chunk_id', 'unknown')
                                    
                                    progress_info = self.get_progress_info(current_task)
                                    if progress_info:
                                        progress = f"{progress_info.get('percentage', 0):.0f}%"
                                    break
                        except:
                            pass
                    
                    print(f"{worker.get('hostname', 'unknown'):<20} {worker.get('gpu_id', '0'):<6} "
                          f"{status:<12} {current_task:<15} {progress}")
                
                print()
                
                # Queue stats
                queue_stats = self.get_queue_stats()
                recent_completed = self.get_recently_completed()
                
                print("QUEUE:")
                print("-" * 40)
                print(f"Pending: {queue_stats['pending']}")
                print(f"Processing: {queue_stats['processing']}")
                print(f"Completed: {queue_stats['completed']}")
                print(f"Failed: {queue_stats['failed']}")
                print(f"Total: {queue_stats['total']}")
                
                eta = self.calculate_eta(queue_stats, recent_completed)
                print(f"ETA: {eta}")
                
                print()
                
                # Recent completions
                if recent_completed:
                    print("RECENTLY COMPLETED:")
                    print("-" * 60)
                    for job in recent_completed[:5]:
                        print(f"{job['chunk_id']}   {job['num_gaussians']:,} Gaussians   "
                              f"{self.format_time(job['training_time'])}   {job['worker_hostname']}")
                
                # Sleep
                time.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"Dashboard error: {e}")
                time.sleep(self.refresh_interval)
    
    def run(self):
        """Run the dashboard."""
        print(f"Starting 3DGS Health Dashboard (refresh every {self.refresh_interval}s)")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            if RICH_AVAILABLE:
                self.display_rich_dashboard()
            else:
                self.display_plain_dashboard()
        
        except KeyboardInterrupt:
            pass
        finally:
            print("\nDashboard stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="3DGS Health Dashboard")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    # Validate config file
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Create and run dashboard
    dashboard = HealthDashboard(args.config, args.refresh)
    dashboard.run()


if __name__ == "__main__":
    main()
