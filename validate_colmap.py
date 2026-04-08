import os
import argparse
import open3d as o3d
import numpy as np

# Attempt to use gsplat's colmap utilities if available.
try:
    from gsplat.utils.colmap import read_cameras_binary, read_images_binary, read_points3D_binary
except ImportError:
    # Minimal fallback parser in case import path differs in gsplat 1.3.0
    import struct
    def read_images_binary(path):
        images = {}
        with open(path, "rb") as fid:
            num_images = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_images):
                fid.read(4) # image_id
                fid.read(56) # qvec (32b) + tvec (24b)
                fid.read(4) # camera_id
                name_bytes = bytearray()
                while True:
                    char = fid.read(1)
                    if char == b"\x00" or char == b"":
                        break
                    name_bytes.extend(char)
                name = name_bytes.decode("utf-8", errors="replace")
                num_points2D = struct.unpack("<Q", fid.read(8))[0]
                fid.read(num_points2D * 24)
                images[name] = True
        return images

    def read_points3D_binary(path):
        points3D = {}
        with open(path, "rb") as fid:
            num_points = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_points):
                pt_id = struct.unpack("<Q", fid.read(8))[0]
                xyz = struct.unpack("<3d", fid.read(24))
                rgb = struct.unpack("<3B", fid.read(3))
                fid.read(8) # error
                track_len = struct.unpack("<Q", fid.read(8))[0]
                fid.read(track_len * 8)
                points3D[pt_id] = {'xyz': xyz, 'rgb': rgb}
        return points3D

def validate(data_dir):
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    cameras_path = os.path.join(sparse_dir, "cameras.bin")
    images_path = os.path.join(sparse_dir, "images.bin")
    points_path = os.path.join(sparse_dir, "points3D.bin")

    assert os.path.exists(cameras_path), f"Missing {cameras_path}"
    assert os.path.exists(images_path), f"Missing {images_path}"
    assert os.path.exists(points_path), f"Missing {points_path}"

    images = read_images_binary(images_path)
    points3D = read_points3D_binary(points_path)

    num_cameras = len(images)
    num_points = len(points3D)

    print(f"Registered Cameras: {num_cameras}")
    print(f"Sparse Points: {num_points}")

    assert num_cameras >= 40, f"Expected at least 40 cameras, found {num_cameras}"
    assert num_points >= 500, f"Expected at least 500 points, found {num_points}"

    print("Pass: COLMAP assertions succeeded! Visualizing...")

    # Visualization
    xyzs = np.array([pt['xyz'] for pt in points3D.values()])
    rgbs = np.array([pt['rgb'] for pt in points3D.values()]) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzs)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)

    o3d.visualization.draw_geometries([pcd], window_name="COLMAP Validation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    args = parser.parse_args()
    validate(args.data_dir)
