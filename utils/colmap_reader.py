"""
COLMAP Binary File Reader
Utility module for parsing COLMAP binary format files (points3D.bin, cameras.bin, images.bin)
"""

import struct
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class COLMAPReader:
    """Reader for COLMAP binary files following the official COLMAP binary format specification."""
    
    def __init__(self, colmap_dir: str):
        """
        Initialize COLMAP reader.
        
        Args:
            colmap_dir: Path to COLMAP sparse directory (e.g., "./sparse/0")
        """
        self.colmap_dir = Path(colmap_dir)
        
    def read_points3d(self) -> Dict[int, np.ndarray]:
        """
        Read points3D.bin file.
        
        Returns:
            Dict mapping point_id -> numpy array [x, y, z, r, g, b, error]
        """
        points_file = self.colmap_dir / "points3D.bin"
        if not points_file.exists():
            raise FileNotFoundError(f"points3D.bin not found at {points_file}")
            
        points = {}
        
        with open(points_file, 'rb') as f:
            # Read number of points
            num_points = struct.unpack('<Q', f.read(8))[0]
            print(f"Reading {num_points} 3D points...")
            
            for _ in range(num_points):
                # Read point_id (8 bytes)
                point_id = struct.unpack('<Q', f.read(8))[0]
                
                # Read XYZ coordinates (3 * 8 bytes = 24 bytes)
                x, y, z = struct.unpack('<ddd', f.read(24))
                
                # Read RGB color (3 * 1 byte = 3 bytes)
                r, g, b = struct.unpack('<BBB', f.read(3))
                
                # Read error (8 bytes)
                error = struct.unpack('<d', f.read(8))[0]
                
                # Read track length (8 bytes)
                track_length = struct.unpack('<Q', f.read(8))[0]
                
                # Skip track data (track_length * 8 bytes)
                f.seek(track_length * 8, 1)
                
                # Store point data
                points[point_id] = np.array([x, y, z, r, g, b, error])
                
        print(f"Successfully read {len(points)} 3D points")
        return points
    
    def read_cameras(self) -> Dict[int, Dict]:
        """
        Read cameras.bin file.
        
        Returns:
            Dict mapping camera_id -> camera parameters dict
        """
        cameras_file = self.colmap_dir / "cameras.bin"
        if not cameras_file.exists():
            raise FileNotFoundError(f"cameras.bin not found at {cameras_file}")
            
        cameras = {}
        
        with open(cameras_file, 'rb') as f:
            # Read number of cameras
            num_cameras = struct.unpack('<Q', f.read(8))[0]
            print(f"Reading {num_cameras} cameras...")
            
            for _ in range(num_cameras):
                # Read camera_id (4 bytes)
                camera_id = struct.unpack('<I', f.read(4))[0]
                
                # Read model (1 byte)
                model = struct.unpack('<B', f.read(1))[0]
                
                # Read width and height (4 bytes each)
                width = struct.unpack('<I', f.read(4))[0]
                height = struct.unpack('<I', f.read(4))[0]
                
                # Read number of parameters (4 bytes)
                num_params = struct.unpack('<I', f.read(4))[0]
                
                # Read parameters (8 bytes each)
                params = []
                for _ in range(num_params):
                    param = struct.unpack('<d', f.read(8))[0]
                    params.append(param)
                
                # Store camera data
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }
                
        print(f"Successfully read {len(cameras)} cameras")
        return cameras
    
    def read_images(self) -> Dict[int, Dict]:
        """
        Read images.bin file.
        
        Returns:
            Dict mapping image_id -> image data dict
        """
        images_file = self.colmap_dir / "images.bin"
        if not images_file.exists():
            raise FileNotFoundError(f"images.bin not found at {images_file}")
            
        images = {}
        
        with open(images_file, 'rb') as f:
            # Read number of images
            num_images = struct.unpack('<Q', f.read(8))[0]
            print(f"Reading {num_images} images...")
            
            for _ in range(num_images):
                # Read image_id (4 bytes)
                image_id = struct.unpack('<I', f.read(4))[0]
                
                # Read quaternion qvec (4 * 8 bytes = 32 bytes)
                qw, qx, qy, qz = struct.unpack('<dddd', f.read(32))
                
                # Read translation tvec (3 * 8 bytes = 24 bytes)
                tx, ty, tz = struct.unpack('<ddd', f.read(24))
                
                # Read camera_id (4 bytes)
                camera_id = struct.unpack('<I', f.read(4))[0]
                
                # Read image name (null-terminated string)
                name = b''
                while True:
                    char = f.read(1)
                    if char == b'\0':
                        break
                    name += char
                image_name = name.decode('utf-8')
                
                # Read number of 2D points (8 bytes)
                num_points2d = struct.unpack('<Q', f.read(8))[0]
                
                # Skip 2D points data (num_points2d * 8 bytes)
                f.seek(num_points2d * 8, 1)
                
                # Store image data
                images[image_id] = {
                    'qvec': np.array([qw, qx, qy, qz]),
                    'tvec': np.array([tx, ty, tz]),
                    'camera_id': camera_id,
                    'name': image_name
                }
                
        print(f"Successfully read {len(images)} images")
        return images
    
    def quaternion_to_rotation_matrix(self, qvec: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            qvec: Quaternion [qw, qx, qy, qz]
            
        Returns:
            3x3 rotation matrix
        """
        qw, qx, qy, qz = qvec
        
        # Normalize quaternion
        q_norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if q_norm > 0:
            qw, qx, qy, qz = qw / q_norm, qx / q_norm, qy / q_norm, qz / q_norm
        
        # Compute rotation matrix
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R
    
    def get_camera_world_position(self, qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Get camera world position from quaternion and translation.
        
        Args:
            qvec: Quaternion [qw, qx, qy, qz]
            tvec: Translation vector [tx, ty, tz]
            
        Returns:
            Camera world position [x, y, z]
        """
        R = self.quaternion_to_rotation_matrix(qvec)
        camera_pos = -R.T @ tvec
        return camera_pos
    
    def get_pinhole_intrinsics(self, camera_params: List[float]) -> Tuple[float, float, float, float]:
        """
        Extract pinhole camera intrinsics from parameters.
        
        Args:
            camera_params: Camera parameters list
            
        Returns:
            Tuple of (fx, fy, cx, cy)
        """
        if len(camera_params) >= 4:
            fx, fy, cx, cy = camera_params[:4]
            return fx, fy, cx, cy
        else:
            raise ValueError("Insufficient parameters for pinhole camera model")


def test_colmap_reader(colmap_dir: str):
    """Test function to verify COLMAP reader functionality."""
    print(f"Testing COLMAP reader with directory: {colmap_dir}")
    
    try:
        reader = COLMAPReader(colmap_dir)
        
        # Test reading points
        points = reader.read_points3d()
        if points:
            sample_point = list(points.values())[0]
            print(f"Sample point: {sample_point}")
        
        # Test reading cameras
        cameras = reader.read_cameras()
        if cameras:
            sample_camera = list(cameras.values())[0]
            print(f"Sample camera: {sample_camera}")
        
        # Test reading images
        images = reader.read_images()
        if images:
            sample_image = list(images.values())[0]
            print(f"Sample image: {sample_image}")
            
        print("COLMAP reader test completed successfully!")
        return True
        
    except Exception as e:
        print(f"COLMAP reader test failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        colmap_dir = sys.argv[1]
        test_colmap_reader(colmap_dir)
    else:
        print("Usage: python colmap_reader.py <colmap_sparse_dir>")
