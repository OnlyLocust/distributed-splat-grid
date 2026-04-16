"""
splat_grid.colmap_utils
=======================
Binary COLMAP parsers shared by both master.py and worker.py.

Extracted from train_single_node.py (V1 build) with no functional changes.
Camera model parameter counts are sourced from:
  https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
"""

import os
import struct
import logging

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera model parameter counts
# ---------------------------------------------------------------------------

_COLMAP_MODEL_NUM_PARAMS = {
    0:  1,   # SIMPLE_PINHOLE          : f
    1:  4,   # PINHOLE                 : fx fy cx cy
    2:  4,   # SIMPLE_RADIAL           : f cx cy k1
    3:  5,   # RADIAL                  : f cx cy k1 k2
    4:  8,   # OPENCV                  : fx fy cx cy k1 k2 p1 p2
    5:  8,   # OPENCV_FISHEYE          : fx fy cx cy k1 k2 k3 k4
    6:  12,  # FULL_OPENCV             : fx fy cx cy k1..k6 p1 p2
    7:  5,   # FOV                     : fx fy cx cy omega
    8:  4,   # SIMPLE_RADIAL_FISHEYE   : f cx cy k
    9:  5,   # RADIAL_FISHEYE          : f cx cy k1 k2
    10: 12,  # THIN_PRISM_FISHEYE      : fx fy cx cy ...
}


# ---------------------------------------------------------------------------
# Binary parsers
# ---------------------------------------------------------------------------

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

            if model_id == 0:            # SIMPLE_PINHOLE: f
                fx = fy = params[0]
                cx, cy  = width / 2.0, height / 2.0
            elif model_id == 1:          # PINHOLE: fx fy cx cy
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model_id in (2, 8):     # SIMPLE_RADIAL / SIMPLE_RADIAL_FISHEYE
                fx = fy = params[0]
                cx, cy  = params[1], params[2]
            elif model_id in (3, 9):     # RADIAL / RADIAL_FISHEYE
                fx = fy = params[0]
                cx, cy  = params[1], params[2]
            else:                        # OPENCV / FISHEYE / FOV / FULL / THIN_PRISM
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]

            cameras[cam_id] = {
                "width": width, "height": height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }
    return cameras


def read_images_binary(path: str) -> dict:
    """Parse COLMAP images.bin → {image_id: {name, cam_id, R[3×3], t[3]}}"""
    images = {}
    with open(path, "rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            (img_id,) = struct.unpack("<I", f.read(4))
            qvec      = struct.unpack("<4d", f.read(32))  # qw qx qy qz
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
            f.read(num_pts2d * 24)  # skip 2-D observations

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
    """Parse COLMAP points3D.bin → {pt_id: {xyz[3], rgb[3]}}"""
    points3D = {}
    with open(path, "rb") as fid:
        (num_points,) = struct.unpack("<Q", fid.read(8))
        for _ in range(num_points):
            (pt_id,)     = struct.unpack("<Q", fid.read(8))
            xyz          = struct.unpack("<3d", fid.read(24))
            rgb          = struct.unpack("<3B", fid.read(3))
            fid.read(8)                           # reprojection error (float64)
            (track_len,) = struct.unpack("<Q",   fid.read(8))
            fid.read(track_len * 8)               # track entries
            points3D[pt_id] = {"xyz": xyz, "rgb": rgb}
    return points3D


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------

def load_colmap_data(data_dir: str) -> tuple:
    """
    Load cameras, poses, and optional sparse points from COLMAP binary files.

    Returns
    -------
    cameras   : dict  {cam_id → camera params}
    images    : dict  {image_id → pose + name}
    points3D  : dict  {pt_id → xyz + rgb}  or  None
    """
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    cam_path   = os.path.join(sparse_dir, "cameras.bin")
    img_path   = os.path.join(sparse_dir, "images.bin")
    pts_path   = os.path.join(sparse_dir, "points3D.bin")

    if not os.path.exists(cam_path):
        raise FileNotFoundError(f"cameras.bin not found: {cam_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"images.bin not found: {img_path}")

    log.info("Loading cameras...")
    cameras = read_cameras_binary(cam_path)
    log.info(f"  {len(cameras)} cameras loaded.")

    log.info("Loading image poses...")
    images = read_images_binary(img_path)
    log.info(f"  {len(images)} poses loaded.")

    points3D = None
    if os.path.exists(pts_path):
        log.info("Loading sparse points...")
        points3D = read_points3D_binary(pts_path)
        log.info(f"  {len(points3D)} sparse points loaded.")
    else:
        log.warning("points3D.bin not found — random Gaussian init will be used on workers.")

    return cameras, images, points3D


def load_images_for_worker(
    local_image_dir: str,
    image_metas: dict,
    image_names: list,
    downscale: int = 4,
) -> tuple:
    """
    Load a subset of ground-truth images from disk (worker-side).

    Parameters
    ----------
    local_image_dir : str   Path to the directory containing downloaded images.
    image_metas     : dict  Full {image_id → meta} from read_images_binary.
    image_names     : list  Filenames the master assigned to this task.
    downscale       : int   Resolution divisor (V1 default: 4 → ¼ resolution).

    Returns
    -------
    gt_images : {image_id → Tensor[H, W, 3] float32, CPU}
    hw        : {image_id → (H, W)}
    """
    import torch

    gt_images: dict = {}
    hw:        dict = {}

    # Build reverse name→id map
    name_to_id = {meta["name"]: img_id for img_id, meta in image_metas.items()}

    log.info(f"Loading {len(image_names)} images from {local_image_dir} (↓{downscale}×)...")
    for name in image_names:
        img_id = name_to_id.get(name)
        if img_id is None:
            log.warning(f"  No COLMAP entry for image '{name}', skipping.")
            continue
        fpath = os.path.join(local_image_dir, name)
        if not os.path.exists(fpath):
            log.warning(f"  Image not found on disk: {fpath}")
            continue
        try:
            img = Image.open(fpath).convert("RGB")
            if downscale > 1:
                img = img.resize(
                    (img.width // downscale, img.height // downscale),
                    Image.LANCZOS,
                )
            arr               = np.array(img, dtype=np.float32) / 255.0
            gt_images[img_id] = torch.from_numpy(arr)   # [H, W, 3]
            hw[img_id]        = (arr.shape[0], arr.shape[1])
        except Exception as exc:
            log.warning(f"  Failed to load {fpath}: {exc}")

    log.info(f"  {len(gt_images)} images loaded successfully.")
    return gt_images, hw
