"""
VRAM Guard Utilities
Memory management utilities for 4GB GPU constraint
"""

import torch
import gc
import psutil
import os
from typing import Dict, Optional, Tuple
import warnings


class VRAMGuard:
    """Memory management utilities for GPU training on 4GB VRAM."""
    
    def __init__(self, device: torch.device):
        """
        Initialize VRAM guard.
        
        Args:
            device: PyTorch device (should be CUDA)
        """
        self.device = device
        self.total_vram_gb = self._get_total_vram()
        self.safety_margin_gb = 0.5  # Keep 500MB free
        self.max_allocatable_gb = self.total_vram_gb - self.safety_margin_gb
        
        print(f"VRAM Guard initialized: {self.total_vram_gb:.1f}GB total, {self.max_allocatable_gb:.1f}GB allocatable")
    
    def _get_total_vram(self) -> float:
        """Get total VRAM in GB."""
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU memory tracking")
            return psutil.virtual_memory().total / (1024**3)
        
        return torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
    
    def get_vram_usage(self) -> Dict[str, float]:
        """
        Get current VRAM usage statistics.
        
        Returns:
            Dict with usage stats in GB
        """
        if not torch.cuda.is_available():
            mem = psutil.virtual_memory()
            return {
                'allocated': 0.0,
                'cached': 0.0,
                'free': mem.available / (1024**3),
                'total': mem.total / (1024**3)
            }
        
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        cached = torch.cuda.memory_reserved(self.device) / (1024**3)
        total = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        free = total - cached
        
        return {
            'allocated': allocated,
            'cached': cached,
            'free': free,
            'total': total
        }
    
    def check_memory_safety(self, required_gb: float) -> bool:
        """
        Check if there's enough memory for allocation.
        
        Args:
            required_gb: Required memory in GB
            
        Returns:
            True if safe to allocate, False otherwise
        """
        usage = self.get_vram_usage()
        available = usage['free']
        
        safe = available >= required_gb
        
        if not safe:
            print(f"Memory check failed: need {required_gb:.2f}GB, have {available:.2f}GB free")
        
        return safe
    
    def aggressive_cleanup(self) -> None:
        """Perform aggressive memory cleanup."""
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force Python garbage collection
        gc.collect()
        
        # Additional cleanup for PyTorch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print("Performed aggressive memory cleanup")
    
    def estimate_gaussian_memory(self, num_gaussians: int, sh_degree: int = 1) -> float:
        """
        Estimate memory usage for Gaussian parameters.
        
        Args:
            num_gaussians: Number of Gaussians
            sh_degree: Spherical harmonics degree
            
        Returns:
            Estimated memory usage in GB
        """
        # Base parameters
        positions = num_gaussians * 3 * 4  # float32
        scales = num_gaussians * 3 * 4     # float32
        rotations = num_gaussians * 4 * 4   # float32
        opacities = num_gaussians * 1 * 4   # float32
        
        # SH coefficients (degree 1 = 3 coefficients, degree 3 = 16 coefficients)
        sh_coeffs_per_gaussian = 3 if sh_degree == 1 else 16
        sh_coeffs = num_gaussians * sh_coeffs_per_gaussian * 4  # float32
        
        # Add gradients (same size as parameters)
        total_params = positions + scales + rotations + opacities + sh_coeffs
        with_gradients = total_params * 2
        
        # Add optimizer states (Adam uses 2x parameter size for moments)
        with_optimizer = with_gradients * 3
        
        # Add some overhead for intermediate tensors
        overhead = with_optimizer * 0.5
        
        total_bytes = with_optimizer + overhead
        return total_bytes / (1024**3)
    
    def estimate_training_memory(self, 
                               num_gaussians: int, 
                               image_size: Tuple[int, int],
                               batch_size: int = 1) -> float:
        """
        Estimate total training memory usage.
        
        Args:
            num_gaussians: Number of Gaussians
            image_size: Image dimensions (height, width)
            batch_size: Number of images processed simultaneously
            
        Returns:
            Estimated memory usage in GB
        """
        # Gaussian parameters
        gaussian_memory = self.estimate_gaussian_memory(num_gaussians, sh_degree=1)
        
        # Image memory
        h, w = image_size
        image_memory = batch_size * h * w * 3 * 4  # RGB float32
        
        # Rendering buffers (approximate)
        render_memory = batch_size * h * w * 4 * 4  # RGBA float32
        
        # Intermediate tensors during rendering
        intermediate_memory = render_memory * 2
        
        total_gb = (gaussian_memory + image_memory + render_memory + intermediate_memory) / (1024**3)
        
        return total_gb
    
    def get_safe_max_gaussians(self, 
                              image_size: Tuple[int, int],
                              target_memory_gb: Optional[float] = None) -> int:
        """
        Calculate safe maximum number of Gaussians for given image size.
        
        Args:
            image_size: Image dimensions (height, width)
            target_memory_gb: Target memory limit (defaults to max allocatable)
            
        Returns:
            Safe maximum number of Gaussians
        """
        if target_memory_gb is None:
            target_memory_gb = self.max_allocatable_gb
        
        # Binary search for safe number of Gaussians
        low, high = 1000, 500000  # Start with reasonable bounds
        
        while low < high:
            mid = (low + high + 1) // 2
            estimated_memory = self.estimate_training_memory(mid, image_size)
            
            if estimated_memory <= target_memory_gb:
                low = mid
            else:
                high = mid - 1
        
        safe_max = low
        print(f"Safe max Gaussians for {image_size}: {safe_max:,} (estimated {self.estimate_training_memory(safe_max, image_size):.2f}GB)")
        
        return safe_max
    
    def monitor_memory_usage(self, description: str = "") -> None:
        """
        Print current memory usage with optional description.
        
        Args:
            description: Optional description for the log
        """
        usage = self.get_vram_usage()
        
        print(f"Memory{description}: "
              f"{usage['allocated']:.2f}GB allocated, "
              f"{usage['cached']:.2f}GB cached, "
              f"{usage['free']:.2f}GB free, "
              f"{usage['total']:.2f}GB total")
    
    def safe_tensor_creation(self, 
                            shape: Tuple[int, ...], 
                            dtype: torch.dtype = torch.float32,
                            description: str = "tensor") -> torch.Tensor:
        """
        Safely create a tensor with memory check.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            description: Description for logging
            
        Returns:
            Created tensor
            
        Raises:
            RuntimeError: If not enough memory
        """
        # Estimate memory requirement
        num_elements = torch.prod(torch.tensor(shape)).item()
        bytes_per_element = 4 if dtype == torch.float32 else 2  # float16
        required_gb = (num_elements * bytes_per_element) / (1024**3)
        
        if not self.check_memory_safety(required_gb):
            self.aggressive_cleanup()
            
            if not self.check_memory_safety(required_gb):
                raise RuntimeError(f"Cannot create {description}: need {required_gb:.2f}GB, not enough VRAM")
        
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        print(f"Created {description}: {shape} ({required_gb:.2f}GB)")
        
        return tensor


class ChunkOOMError(Exception):
    """Custom exception for chunk out-of-memory errors."""
    pass


def setup_memory_efficient_training(device: torch.device) -> VRAMGuard:
    """
    Setup memory-efficient training configuration.
    
    Args:
        device: PyTorch device
        
    Returns:
        Configured VRAMGuard instance
    """
    # Enable memory efficient settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Enable gradient checkpointing if available
    try:
        from torch.utils.checkpoint import checkpoint
        print("Gradient checkpointing available")
    except ImportError:
        print("Gradient checkpointing not available")
    
    guard = VRAMGuard(device)
    guard.monitor_memory_usage(" (initial)")
    
    return guard


def test_vram_guard():
    """Test VRAM guard functionality."""
    print("Testing VRAM Guard...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    guard = VRAMGuard(device)
    
    # Test memory estimation
    gaussians = 100000
    image_size = (800, 600)
    
    gaussian_memory = guard.estimate_gaussian_memory(gaussians)
    training_memory = guard.estimate_training_memory(gaussians, image_size)
    max_gaussians = guard.get_safe_max_gaussians(image_size)
    
    print(f"Memory estimates for {gaussians:,} Gaussians:")
    print(f"  Gaussian parameters: {gaussian_memory:.2f}GB")
    print(f"  Total training: {training_memory:.2f}GB")
    print(f"  Safe max Gaussians: {max_gaussians:,}")
    
    # Test safe tensor creation
    try:
        tensor = guard.safe_tensor_creation((1000, 1000), description="test tensor")
        print(f"Successfully created test tensor: {tensor.shape}")
        del tensor
    except RuntimeError as e:
        print(f"Tensor creation failed: {e}")
    
    # Test cleanup
    guard.aggressive_cleanup()
    guard.monitor_memory_usage(" (after cleanup)")
    
    print("VRAM Guard test completed!")


if __name__ == "__main__":
    test_vram_guard()
