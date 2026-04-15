"""
PLY File Writer/Reader Utilities
Handles standard PLY format for 3D Gaussian Splatting outputs
Compatible with SuperSplat / Luma / Polycam
"""

import struct
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys


class PLYWriter:
    """Writer for 3D Gaussian Splatting PLY files."""
    
    # PLY property order for compatibility
    PROPERTY_ORDER = [
        'x', 'y', 'z',                    # Position (float32)
        'nx', 'ny', 'nz',                 # Normal (float32, set to 0)
        'f_dc_0', 'f_dc_1', 'f_dc_2',    # SH DC coefficients (float32)
        'opacity',                        # Opacity (float32, pre-sigmoid)
        'scale_0', 'scale_1', 'scale_2',  # Scale (float32, log-space)
        'rot_0', 'rot_1', 'rot_2', 'rot_3' # Rotation quaternion (float32)
    ]
    
    def __init__(self):
        """Initialize PLY writer."""
        self.header_template = """ply
format binary_little_endian 1.0
element vertex {num_vertices}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    
    def write_gaussians(self, 
                       positions: np.ndarray,      # (N, 3) float32
                       sh_dc: np.ndarray,          # (N, 3) float32  
                       opacities: np.ndarray,      # (N, 1) float32
                       scales: np.ndarray,         # (N, 3) float32
                       rotations: np.ndarray,      # (N, 4) float32
                       output_path: str) -> None:
        """
        Write Gaussian data to PLY file.
        
        Args:
            positions: XYZ coordinates (N, 3)
            sh_dc: SH DC coefficients (N, 3) 
            opacities: Opacity values (N, 1) pre-sigmoid
            scales: Log-space scales (N, 3)
            rotations: Quaternion rotations (N, 4)
            output_path: Output PLY file path
        """
        num_gaussians = len(positions)
        print(f"Writing {num_gaussians} Gaussians to {output_path}")
        
        # Validate input shapes
        assert positions.shape == (num_gaussians, 3), f"Expected positions ({num_gaussians}, 3), got {positions.shape}"
        assert sh_dc.shape == (num_gaussians, 3), f"Expected sh_dc ({num_gaussians}, 3), got {sh_dc.shape}"
        assert opacities.shape == (num_gaussians, 1), f"Expected opacities ({num_gaussians}, 1), got {opacities.shape}"
        assert scales.shape == (num_gaussians, 3), f"Expected scales ({num_gaussians}, 3), got {scales.shape}"
        assert rotations.shape == (num_gaussians, 4), f"Expected rotations ({num_gaussians}, 4), got {rotations.shape}"
        
        # Ensure float32 dtype
        positions = positions.astype(np.float32)
        sh_dc = sh_dc.astype(np.float32)
        opacities = opacities.astype(np.float32)
        scales = scales.astype(np.float32)
        rotations = rotations.astype(np.float32)
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            # Write header
            header = self.header_template.format(num_vertices=num_gaussians)
            f.write(header.encode('ascii'))
            
            # Write binary data
            for i in range(num_gaussians):
                # Position (x, y, z)
                f.write(struct.pack('<fff', positions[i, 0], positions[i, 1], positions[i, 2]))
                
                # Normal (nx, ny, nz) - set to 0
                f.write(struct.pack('<fff', 0.0, 0.0, 0.0))
                
                # SH DC coefficients (f_dc_0, f_dc_1, f_dc_2)
                f.write(struct.pack('<fff', sh_dc[i, 0], sh_dc[i, 1], sh_dc[i, 2]))
                
                # Opacity
                f.write(struct.pack('<f', opacities[i, 0]))
                
                # Scale (scale_0, scale_1, scale_2)
                f.write(struct.pack('<fff', scales[i, 0], scales[i, 1], scales[i, 2]))
                
                # Rotation (rot_0, rot_1, rot_2, rot_3)
                f.write(struct.pack('<ffff', rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]))
        
        # Verify file size
        file_size = Path(output_path).stat().st_size
        expected_size = num_gaussians * 59 + len(header)  # 59 bytes per Gaussian + header
        print(f"PLY file written: {file_size} bytes (expected: {expected_size} bytes)")


class PLYReader:
    """Reader for 3D Gaussian Splatting PLY files."""
    
    def __init__(self):
        """Initialize PLY reader."""
        pass
    
    def read_gaussians(self, ply_path: str) -> Dict[str, np.ndarray]:
        """
        Read Gaussian data from PLY file.
        
        Args:
            ply_path: Input PLY file path
            
        Returns:
            Dict with keys: positions, sh_dc, opacities, scales, rotations
        """
        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
        
        print(f"Reading PLY file: {ply_path}")
        
        with open(ply_path, 'rb') as f:
            # Read header
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            
            # Parse header to get vertex count
            num_vertices = None
            for line in header_lines:
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[2])
                    break
            
            if num_vertices is None:
                raise ValueError("Could not find vertex count in PLY header")
            
            print(f"Reading {num_vertices} Gaussians...")
            
            # Read binary data
            positions = np.zeros((num_vertices, 3), dtype=np.float32)
            sh_dc = np.zeros((num_vertices, 3), dtype=np.float32)
            opacities = np.zeros((num_vertices, 1), dtype=np.float32)
            scales = np.zeros((num_vertices, 3), dtype=np.float32)
            rotations = np.zeros((num_vertices, 4), dtype=np.float32)
            
            for i in range(num_vertices):
                # Position (x, y, z)
                positions[i] = struct.unpack('<fff', f.read(12))
                
                # Normal (nx, ny, nz) - skip
                f.read(12)
                
                # SH DC coefficients (f_dc_0, f_dc_1, f_dc_2)
                sh_dc[i] = struct.unpack('<fff', f.read(12))
                
                # Opacity
                opacities[i] = struct.unpack('<f', f.read(4))
                
                # Scale (scale_0, scale_1, scale_2)
                scales[i] = struct.unpack('<fff', f.read(12))
                
                # Rotation (rot_0, rot_1, rot_2, rot_3)
                rotations[i] = struct.unpack('<ffff', f.read(16))
        
        print(f"Successfully read {num_vertices} Gaussians")
        
        return {
            'positions': positions,
            'sh_dc': sh_dc,
            'opacities': opacities,
            'scales': scales,
            'rotations': rotations
        }
    
    def validate_ply(self, ply_path: str) -> bool:
        """
        Validate PLY file format.
        
        Args:
            ply_path: PLY file path
            
        Returns:
            True if valid, False otherwise
        """
        try:
            data = self.read_gaussians(ply_path)
            
            # Check shapes
            expected_shapes = {
                'positions': (None, 3),
                'sh_dc': (None, 3),
                'opacities': (None, 1),
                'scales': (None, 3),
                'rotations': (None, 4)
            }
            
            num_gaussians = len(data['positions'])
            
            for key, expected_shape in expected_shapes.items():
                actual_shape = data[key].shape
                if actual_shape[0] != num_gaussians:
                    print(f"Validation failed: {key} has {actual_shape[0]} vertices, expected {num_gaussians}")
                    return False
                if actual_shape[1:] != expected_shape[1:]:
                    print(f"Validation failed: {key} shape {actual_shape} doesn't match expected {expected_shape}")
                    return False
            
            print(f"PLY validation passed: {num_gaussians} Gaussians")
            return True
            
        except Exception as e:
            print(f"PLY validation failed: {e}")
            return False


def estimate_vram_usage(num_gaussians: int) -> float:
    """
    Estimate VRAM usage for loading Gaussian data.
    
    Args:
        num_gaussians: Number of Gaussians
        
    Returns:
        Estimated VRAM usage in GB
    """
    # Each Gaussian uses approximately 59 bytes in PLY format
    # In memory with torch tensors, it's roughly 4x due to float32 and gradients
    bytes_per_gaussian = 59 * 4
    total_bytes = num_gaussians * bytes_per_gaussian
    gb_usage = total_bytes / (1024**3)
    return gb_usage


def test_ply_io():
    """Test PLY read/write functionality."""
    print("Testing PLY I/O...")
    
    # Create test data
    num_test = 1000
    positions = np.random.randn(num_test, 3).astype(np.float32)
    sh_dc = np.random.randn(num_test, 3).astype(np.float32) * 0.5
    opacities = np.random.rand(num_test, 1).astype(np.float32)
    scales = np.random.rand(num_test, 3).astype(np.float32) * 0.1
    rotations = np.random.randn(num_test, 4).astype(np.float32)
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)  # Normalize
    
    # Write test PLY
    test_path = "test_gaussians.ply"
    writer = PLYWriter()
    writer.write_gaussians(positions, sh_dc, opacities, scales, rotations, test_path)
    
    # Read test PLY
    reader = PLYReader()
    data = reader.read_gaussians(test_path)
    
    # Validate
    is_valid = reader.validate_ply(test_path)
    
    # Compare data
    np.testing.assert_allclose(data['positions'], positions, rtol=1e-5)
    np.testing.assert_allclose(data['sh_dc'], sh_dc, rtol=1e-5)
    np.testing.assert_allclose(data['opacities'], opacities, rtol=1e-5)
    np.testing.assert_allclose(data['scales'], scales, rtol=1e-5)
    np.testing.assert_allclose(data['rotations'], rotations, rtol=1e-5)
    
    print(f"PLY I/O test passed! Valid: {is_valid}")
    
    # Clean up
    Path(test_path).unlink(missing_ok=True)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_ply_io()
    else:
        print("Usage: python ply_writer.py --test")
