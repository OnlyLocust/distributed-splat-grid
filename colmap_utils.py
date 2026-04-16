"""
colmap_utils.py
===============
Shared COLMAP binary parsers for Splat-Grid master and worker nodes.
Provides: read_cameras_binary, read_images_binary, read_points3D_binary,
          load_colmap_data, compute_global_bbox.
"""

from __future__ import annotations

import logging
import os
import struct

import numpy as np

log = logging.getLogger(__name__)

# Correct parameter counts per COLMAP model id
# https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
_COLMAP_MODEL_NUM_PARAMS = {
    0:  1,   # SIMPLE_PINHOLE        : f
    1:  4,   # PINHOLE               : fx fy cx cy
    2:  4,   # SIMPLE_RADIAL         : f cx cy k1
    3:  5,   # RADIAL                : f cx cy k1 k2
    4:  8,   # OPENCV                : fx fy cx cy k1 k2 p1 p2
    5:  8,   # OPENCV_FISHEYE        : fx fy cx cy k1 k2 k3 k4
    6:  12,  # FULL_OPENCV           : fx fy cx cy k1..k6 p1 p2
    7:  5,   # FOV                   : fx fy cx cy omega
    8:  4,   # SIMPLE_RADIAL_FISHEYE : f cx cy k
    9:  5,   # RADIAL_FISHEYE        : f cx cy k1 k2
    10: 12,  # THIN_PRISM_FISHEYE    : fx fy cx cy ...
}


# ──────────────────────────────────────────────────────────────────────────────
# Binary parsers
# ──────────────────────────────────────────────────────────────────────────────

def read_cameras_binary(path: str) -> dict:
    """Parse COLMAP cameras.bin → {cam_id: {width, height, fx, fy, cx, cy}}"""
    cameras = {}
    with open(path, "rb") as f:
        (num_cameras,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_cameras):
            (cam_id,)   = struct.unpack("<I", f.read(4))
            (model_id,) = struct.unpack("<I", f.read(4))
            (width,)    = struct.unpack("<Q", f.read(8))
            (height,)   = struct.unpack("<Q", f.read(8))

            n_params = _COLMAP_MODEL_NUM_PARAMS.get(model_id, 4)
            params   = struct.unpack(f"<{n_params}d", f.read(8 * n_params))

            if model_id == 0:
                fx = fy = params[0]
                cx, cy  = width / 2.0, height / 2.0
            elif model_id == 1:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model_id in (2, 8):
                fx = fy = params[0]
                cx, cy  = params[1], params[2]
            elif model_id in (3, 9):
                fx = fy = params[0]
                cx, cy  = params[1], params[2]
            else:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]

            cameras[cam_id] = {
                "width": width, "height": height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }
    return cameras


def read_images_binary(path: str) -> dict:
    """Parse COLMAP images.bin → {image_id: {name, cam_id, R[3x3], t[3]}}"""
    images = {}
    with open(path, "rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            (img_id,) = struct.unpack("<I", f.read(4))
            qvec      = struct.unpack("<4d", f.read(32))    # qw qx qy qz
            tvec      = struct.unpack("<3d", f.read(24))
            (cam_id,) = struct.unpack("<I", f.read(4))

            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")

            (num_pts2d,) = struct.unpack("<Q", f.read(8))
            f.read(num_pts2d * 24)   # skip 2-D observations

            qw, qx, qy, qz = qvec
            R = np.array([
                [1-2*(qy*qy+qz*qz),  2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw),    1-2*(qx*qx+qz*qz),  2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
            ], dtype=np.float32)

            images[img_id] = {
                "name": name, "cam_id": cam_id,
                "R": R, "t": np.array(tvec, dtype=np.float32),
            }
    return images


def read_points3D_binary(path: str) -> dict:
    """Parse COLMAP points3D.bin → {pt_id: {xyz, rgb}}"""
    points3D = {}
    with open(path, "rb") as fid:
        (num_points,) = struct.unpack("<Q", fid.read(8))
        for _ in range(num_points):
            (pt_id,)     = struct.unpack("<Q", fid.read(8))
            xyz          = struct.unpack("<3d", fid.read(24))
            rgb          = struct.unpack("<3B", fid.read(3))
            fid.read(8)                         # reprojection error (float64)
            (track_len,) = struct.unpack("<Q",  fid.read(8))
            fid.read(track_len * 8)             # track entries
            points3D[pt_id] = {"xyz": xyz, "rgb": rgb}
    return points3D


# ──────────────────────────────────────────────────────────────────────────────
# High-level loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_colmap_data(data_dir: str) -> tuple[dict, dict, dict | None]:
    """
    Load cameras, image poses, and (optionally) sparse 3-D points.

    Returns
    -------
    cameras   : {cam_id: {width, height, fx, fy, cx, cy}}
    images    : {image_id: {name, cam_id, R, t}}
    points3D  : {pt_id: {xyz, rgb}} or None
    """
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    cam_path   = os.path.join(sparse_dir, "cameras.bin")
    img_path   = os.path.join(sparse_dir, "images.bin")
    pts_path   = os.path.join(sparse_dir, "points3D.bin")

    if not os.path.exists(cam_path):
        raise FileNotFoundError(f"cameras.bin not found: {cam_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"images.bin not found: {img_path}")

    log.info("Loading cameras.bin …")
    cameras = read_cameras_binary(cam_path)
    log.info(f"  {len(cameras)} cameras.")

    log.info("Loading images.bin …")
    images  = read_images_binary(img_path)
    log.info(f"  {len(images)} poses.")

    points3D = None
    if os.path.exists(pts_path):
        log.info("Loading points3D.bin …")
        points3D = read_points3D_binary(pts_path)
        log.info(f"  {len(points3D)} sparse points.")
    else:
        log.warning("points3D.bin not found — bbox will be estimated from camera centers.")

    return cameras, images, points3D


def compute_global_bbox(
    points3D: dict | None,
    images: dict | None = None,
    padding: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a world-space bounding box from sparse 3-D points.
    Falls back to camera centres if points3D is None.
    Adds a `padding` fraction on each side.

    Returns
    -------
    bbox_min : np.ndarray[3] float32
    bbox_max : np.ndarray[3] float32
    """
    if points3D and len(points3D) > 0:
        xyzs = np.array([pt["xyz"] for pt in points3D.values()], dtype=np.float64)
    elif images and len(images) > 0:
        log.warning("No sparse points — estimating bbox from camera centres.")
        xyzs = np.array([-(m["R"].T @ m["t"]) for m in images.values()],
                        dtype=np.float64)
    else:
        log.warning("No geometry data — using default bbox [-5, -5, -5] to [5, 5, 5].")
        return (np.array([-5., -5., -5.], dtype=np.float32),
                np.array([ 5.,  5.,  5.], dtype=np.float32))

    # Remove extreme outliers using IQR clip (robust against noisy COLMAP outputs)
    q1   = np.percentile(xyzs, 5,  axis=0)
    q99  = np.percentile(xyzs, 95, axis=0)
    mask = np.all((xyzs >= q1) & (xyzs <= q99), axis=1)
    xyzs = xyzs[mask] if mask.sum() > 10 else xyzs

    raw_min = xyzs.min(axis=0).astype(np.float32)
    raw_max = xyzs.max(axis=0).astype(np.float32)
    extent  = raw_max - raw_min
    pad     = extent * padding

    return (raw_min - pad).astype(np.float32), (raw_max + pad).astype(np.float32)
