"""
Redis Queue Task Functions for Distributed 3DGS Pipeline
Defines the task functions that workers execute remotely
"""

import os
import json
import socket
import traceback
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Import worker training logic
import sys
sys.path.append(str(Path(__file__).parent.parent))
from worker import GaussianWorker
from shared_storage import get_storage, read_json, write_json


def execute_chunk_training(chunk_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ task function that executes chunk training on a worker machine.
    
    This function runs entirely on the worker machine and returns a result dict
    that RQ stores in Redis. It handles both NFS and S3 storage backends.
    
    Args:
        chunk_id: Identifier of the chunk to train (e.g., "chunk_2_3")
        config: Configuration dictionary containing storage and worker settings
        
    Returns:
        Result dictionary with training status and metadata
    """
    # Initialize result with default values
    result = {
        "chunk_id": chunk_id,
        "status": "FAILED",
        "num_gaussians": 0,
        "error": None,
        "worker_hostname": socket.gethostname(),
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "training_time": 0,
        "output_path": None
    }
    
    # Get storage backend
    storage = get_storage(config["storage"])
    
    # Setup local directories for training
    local_tmp = None
    output_ply_path = None
    
    try:
        import time
        start_time = time.time()
        
        # For S3 backend: download chunk files to local /tmp before training
        if config["storage"]["backend"] == "s3":
            local_tmp = f"/tmp/3dgs_chunks/{chunk_id}"
            os.makedirs(local_tmp, exist_ok=True)
            
            # Download chunk data files
            chunk_files = [
                f"tasks/{chunk_id}/points.npz",
                f"tasks/{chunk_id}/cameras.json", 
                f"tasks/{chunk_id}/metadata.json"
            ]
            
            for rel_path in chunk_files:
                local_path = f"{local_tmp}/{Path(rel_path).name}"
                storage.download_to_local(rel_path, local_path)
            
            # Download only the images listed in cameras.json
            cameras = read_json(storage, f"tasks/{chunk_id}/cameras.json")
            for cam in cameras:
                img_rel = cam["image_path"]  # e.g., "images/frame_00042.jpg"
                img_local = f"{local_tmp}/{Path(img_rel).name}"
                storage.download_to_local(img_rel, img_local)
            
            chunk_dir = local_tmp
            output_dir = f"/tmp/3dgs_results/{chunk_id}"
            os.makedirs(output_dir, exist_ok=True)
            output_ply_path = f"{output_dir}/{chunk_id}.ply"
            
        else:
            # NFS backend: use direct paths
            chunk_dir = storage.get_full_path(f"tasks/{chunk_id}")
            output_dir = storage.get_full_path("results")
            output_ply_path = f"{output_dir}/{chunk_id}.ply"
        
        # Initialize worker and train chunk
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        worker = GaussianWorker(device)
        
        # Prepare worker configuration
        worker_config = config["worker"].copy()
        worker_config.update({
            'num_iterations': worker_config.get('num_iters', 3000),
            'lr_positions': 1.6e-4,
            'lr_opacities': 1e-2,
            'lr_scales': 1e-3,
            'lr_rotations': 1e-3,
            'lr_colors': 5e-3,
            'densify_interval': worker_config.get('densify_interval', 100),
            'output_dir': output_dir
        })
        
        # Execute training
        training_result = worker.train_chunk(chunk_dir, worker_config)
        
        # For S3: upload result .ply back to shared storage
        if config["storage"]["backend"] == "s3" and training_result.get('status') == 'completed':
            storage.upload_from_local(output_ply_path, f"results/{chunk_id}.ply")
        
        # Count Gaussians in output .ply for reporting
        if training_result.get('status') == 'completed':
            try:
                # Try to read PLY to count Gaussians
                try:
                    from plyfile import PlyData
                    ply = PlyData.read(output_ply_path)
                    num_gaussians = len(ply.elements[0].data)
                except ImportError:
                    # Fallback: estimate from worker result
                    num_gaussians = training_result.get('num_gaussians', 0)
                
                result["num_gaussians"] = num_gaussians
                result["status"] = "COMPLETED"
                result["output_path"] = f"results/{chunk_id}.ply"
                
            except Exception as e:
                result["status"] = "FAILED"
                result["error"] = f"Failed to process output PLY: {e}"
        
        else:
            # Training failed
            result["status"] = training_result.get('status', 'FAILED')
            result["error"] = training_result.get('error', 'Unknown training error')
        
        result["training_time"] = time.time() - start_time
        
    except torch.cuda.OutOfMemoryError as e:
        result["status"] = "OOM"
        result["error"] = f"CUDA OOM: {str(e)}"
        
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = f"Unexpected error: {traceback.format_exc()}"
        
    finally:
        # Cleanup GPU memory
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        
        # Cleanup temporary directories for S3 backend
        if local_tmp and os.path.exists(local_tmp):
            try:
                shutil.rmtree(local_tmp)
            except:
                pass
    
    return result


def publish_progress(chunk_id: str, iteration: int, total_iterations: int, 
                    config: Dict[str, Any]) -> None:
    """
    Publish training progress to Redis for monitoring.
    
    Args:
        chunk_id: Chunk identifier
        iteration: Current iteration number
        total_iterations: Total number of iterations
        config: Configuration dictionary
    """
    try:
        import redis
        
        redis_config = config.get("redis", {})
        redis_conn = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            password=redis_config.get("password"),
            decode_responses=True
        )
        
        # Publish progress with TTL of 2 minutes
        progress_data = {
            "chunk_id": chunk_id,
            "iteration": iteration,
            "total_iterations": total_iterations,
            "percentage": (iteration / total_iterations) * 100,
            "timestamp": time.time()
        }
        
        redis_conn.setex(
            f"progress:{chunk_id}", 
            120,  # 2 minutes TTL
            json.dumps(progress_data)
        )
        
        # Also publish to a channel for real-time updates
        redis_conn.publish(
            "progress_updates",
            json.dumps(progress_data)
        )
        
    except Exception as e:
        # Don't let progress publishing errors interrupt training
        print(f"Warning: Failed to publish progress: {e}")


# Utility function to validate chunk before training
def validate_chunk(chunk_id: str, storage) -> bool:
    """
    Validate that all required files exist for a chunk.
    
    Args:
        chunk_id: Chunk identifier
        storage: Storage backend instance
        
    Returns:
        True if all required files exist, False otherwise
    """
    required_files = [
        f"tasks/{chunk_id}/points.npz",
        f"tasks/{chunk_id}/cameras.json",
        f"tasks/{chunk_id}/metadata.json"
    ]
    
    for file_path in required_files:
        if not storage.exists(file_path):
            print(f"Missing required file: {file_path}")
            return False
    
    # Validate cameras.json contains valid image paths
    try:
        cameras = read_json(storage, f"tasks/{chunk_id}/cameras.json")
        for cam in cameras:
            img_path = cam.get("image_path")
            if img_path and not storage.exists(img_path):
                print(f"Missing image file: {img_path}")
                return False
    except Exception as e:
        print(f"Failed to validate cameras.json: {e}")
        return False
    
    return True


# Worker health check task
def worker_health_check(worker_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Health check task for workers.
    
    Args:
        worker_info: Worker information dictionary
        
    Returns:
        Health check result
    """
    result = {
        "hostname": worker_info.get("hostname", "unknown"),
        "gpu_id": worker_info.get("gpu_id", "unknown"),
        "status": "healthy",
        "timestamp": time.time(),
        "gpu_memory_used": 0,
        "gpu_memory_total": 0,
        "cpu_usage": 0,
        "error": None
    }
    
    try:
        # Check GPU status
        import torch
        if torch.cuda.is_available():
            gpu_id = int(worker_info.get("gpu_id", "0"))
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            gpu_memory_used = torch.cuda.memory_allocated(gpu_id)
            
            result["gpu_memory_total"] = gpu_memory
            result["gpu_memory_used"] = gpu_memory_used
        else:
            result["status"] = "no_gpu"
            result["error"] = "CUDA not available"
        
        # Check CPU usage (basic)
        import psutil
        result["cpu_usage"] = psutil.cpu_percent()
        
    except Exception as e:
        result["status"] = "unhealthy"
        result["error"] = str(e)
    
    return result


# Task to cleanup temporary files
def cleanup_worker_temp() -> Dict[str, Any]:
    """
    Cleanup temporary files on worker machine.
    
    Returns:
        Cleanup result
    """
    result = {
        "status": "completed",
        "files_removed": 0,
        "space_freed": 0,
        "error": None
    }
    
    try:
        temp_dir = Path("/tmp/3dgs_chunks")
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                if item.is_dir():
                    # Calculate directory size
                    total_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    shutil.rmtree(item)
                    result["files_removed"] += 1
                    result["space_freed"] += total_size
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
    
    return result


if __name__ == "__main__":
    # Test function locally
    import time
    
    test_config = {
        "storage": {
            "backend": "nfs",
            "base_path": "/tmp/test_storage"
        },
        "worker": {
            "max_gaussians": 200000,
            "num_iters": 100,
            "densify_interval": 50
        },
        "redis": {
            "host": "localhost",
            "port": 6379
        }
    }
    
    # Create test storage
    os.makedirs("/tmp/test_storage", exist_ok=True)
    
    print("Testing RQ task functions...")
    
    # Test health check
    health_result = worker_health_check({
        "hostname": "test-host",
        "gpu_id": "0"
    })
    print(f"Health check result: {health_result}")
    
    # Test cleanup
    cleanup_result = cleanup_worker_temp()
    print(f"Cleanup result: {cleanup_result}")
    
    print("RQ task tests completed!")
