import os
import struct
from typing import Dict, Tuple, Any

import numpy as np


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def read_cameras_binary(path: str) -> Dict[int, Dict[str, Any]]:
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<i", fid.read(4))[0]
            model_id = struct.unpack("<i", fid.read(4))[0]
            width = struct.unpack("<Q", fid.read(8))[0]
            height = struct.unpack("<Q", fid.read(8))[0]
            model_name, num_params = CAMERA_MODELS[model_id]
            params = struct.unpack("<" + "d" * num_params, fid.read(8 * num_params))
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model_id": model_id,
                "model_name": model_name,
                "width": int(width),
                "height": int(height),
                "params": np.array(params, dtype=np.float64),
            }
    return cameras


def read_images_binary(path: str) -> Dict[int, Dict[str, Any]]:
    images = {}
    with open(path, "rb") as fid:
        num_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", fid.read(4))[0]
            qvec = np.array(struct.unpack("<4d", fid.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", fid.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<i", fid.read(4))[0]

            name_bytes = bytearray()
            while True:
                c = fid.read(1)
                if c in (b"\x00", b""):
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8", errors="replace")

            num_points2d = struct.unpack("<Q", fid.read(8))[0]
            fid.read(num_points2d * 24)

            images[image_id] = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images


def read_points3d_binary(path: str) -> Dict[int, Dict[str, Any]]:
    points = {}
    with open(path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            pt_id = struct.unpack("<Q", fid.read(8))[0]
            xyz = np.array(struct.unpack("<3d", fid.read(24)), dtype=np.float64)
            rgb = np.array(struct.unpack("<3B", fid.read(3)), dtype=np.uint8)
            fid.read(8)  # reprojection error
            track_len = struct.unpack("<Q", fid.read(8))[0]
            fid.read(track_len * 8)
            points[int(pt_id)] = {"xyz": xyz, "rgb": rgb}
    return points


def intrinsics_from_camera(camera: Dict[str, Any]) -> Tuple[float, float, float, float]:
    model = camera["model_name"]
    p = camera["params"]
    if model == "SIMPLE_PINHOLE":
        fx = fy = float(p[0])
        cx, cy = float(p[1]), float(p[2])
    elif model == "PINHOLE":
        fx, fy, cx, cy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    elif model in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = float(p[0])
        cx, cy = float(p[1]), float(p[2])
    elif model in ("RADIAL", "RADIAL_FISHEYE", "FOV", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
        fx, fy = float(p[0]), float(p[1])
        cx, cy = float(p[2]), float(p[3])
    else:
        raise ValueError(f"Unsupported camera model for V1: {model}")
    return fx, fy, cx, cy


def load_colmap_scene(data_dir: str) -> Dict[str, Any]:
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    cameras_path = os.path.join(sparse_dir, "cameras.bin")
    images_path = os.path.join(sparse_dir, "images.bin")
    points_path = os.path.join(sparse_dir, "points3D.bin")
    for p in (cameras_path, images_path, points_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing COLMAP file: {p}")

    return {
        "cameras": read_cameras_binary(cameras_path),
        "images": read_images_binary(images_path),
        "points3D": read_points3d_binary(points_path),
    }
