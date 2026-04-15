
import os
import math
import argparse
import struct
import random
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    rasterization    = None
    GSPLAT_AVAILABLE = False
    log.warning("gsplat not installed. Run: pip install gsplat")

try:
    from pytorch_msssim import ssim as ssim_fn
    SSIM_AVAILABLE = True
    log.info("pytorch_msssim available — SSIM loss enabled.")
except ImportError:
    ssim_fn        = None
    SSIM_AVAILABLE = False
    log.info("pytorch_msssim not found — L1 loss only.  pip install pytorch-msssim")

# ── Constants ─────────────────────────────────────────────────────────────────
SH_DEGREE   = 3
SH_COEFFS   = (SH_DEGREE + 1) ** 2    # 16
SH_C0       = 0.28209479177387814      # 1 / (2*sqrt(pi))
LAMBDA_SSIM = 0.2
SEED        = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# HELPER — safe optimizer-state operations (0-dim tensor safe)
# =============================================================================

def _slice_optim_state(v, mask):
    """
    THE ROOT-CAUSE FIX for the IndexError crash.

    In PyTorch >= 2.0, Adam stores its step counter as a 0-dimensional
    scalar Tensor (shape = ()).  The original code did:

        if isinstance(v, torch.Tensor) and v.shape[0] == old_p.shape[0]:

    Calling v.shape[0] on a 0-dim Tensor raises:
        IndexError: tuple index out of range

    This helper checks v.ndim > 0 BEFORE accessing v.shape[0], so
    scalar tensors (and any other non-indexable values) are passed
    through unchanged.  This fix is applied in both prune_gaussians
    and densify_gaussians.
    """
    if isinstance(v, torch.Tensor) and v.ndim > 0:
        return v[mask]
    return v   # 0-dim step tensor, int, float — return as-is


def _pad_optim_state(v, n_old: int, n_pad: int, device):
    """
    Keep the first n_old rows (survivors) and append n_pad zero rows.
    Also 0-dim safe: if v is a scalar tensor it is returned unchanged.
    """
    if (isinstance(v, torch.Tensor)
            and v.ndim > 0
            and v.shape[0] == n_old):
        pad = torch.zeros(n_pad, *v.shape[1:], device=device, dtype=v.dtype)
        return torch.cat([v, pad], dim=0)
    return v


# =============================================================================
# 1. ARGUMENT PARSING
# =============================================================================

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D Gaussian Splatting trainer")
    p.add_argument("--data_dir",               type=str,   default="data")
    p.add_argument("--output",                 type=str,   default="output.ply")
    p.add_argument("--iterations",             type=int,   default=30_000)
    p.add_argument("--lr_pos",                 type=float, default=1.6e-4)
    p.add_argument("--lr_color",               type=float, default=2.5e-3)
    p.add_argument("--lr_opacity",             type=float, default=0.05)
    p.add_argument("--lr_scale",               type=float, default=5e-3)
    p.add_argument("--lr_rot",                 type=float, default=1e-3)
    p.add_argument("--bbox",                   type=str,   default="-5,-5,-5,5,5,5")
    p.add_argument("--densify_start",          type=int,   default=500)
    p.add_argument("--densify_every",          type=int,   default=100)
    p.add_argument("--densify_end",            type=int,   default=15_000)
    p.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    p.add_argument("--opacity_reset_interval", type=int,   default=3_000)
    p.add_argument("--checkpoint_every",       type=int,   default=5_000)
    p.add_argument("--max_gaussians",          type=int,   default=100_000)
    p.add_argument("--image_downscale",        type=int,   default=2)
    p.add_argument("--opacity_threshold",      type=float, default=0.01)
    p.add_argument("--lambda_ssim",            type=float, default=LAMBDA_SSIM)
    p.add_argument("--sh_degree",              type=int,   default=SH_DEGREE,
                   choices=[0, 1, 2, 3])
    p.add_argument("--seed",                   type=int,   default=SEED)
    return p.parse_args()


# =============================================================================
# 2. BOUNDING BOX
# =============================================================================

def parse_bbox(bbox_str: str, device: torch.device):
    """'-5,-5,-5,5,5,5' -> (min_tensor[3], max_tensor[3]) on the correct device."""
    parts = list(map(float, bbox_str.split(",")))
    if len(parts) != 6:
        raise ValueError(f"bbox needs 6 comma-separated floats, got: {bbox_str!r}")
    return (
        torch.tensor(parts[:3], dtype=torch.float32, device=device),
        torch.tensor(parts[3:], dtype=torch.float32, device=device),
    )


# =============================================================================
# 3. COLMAP BINARY PARSERS
# =============================================================================

# Correct parameter counts — source:
# https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
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


def read_cameras_binary(path: str) -> dict:
    """Parse COLMAP cameras.bin -> {cam_id: {width, height, fx, fy, cx, cy}}"""
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
            elif model_id in (2, 8):     # SIMPLE_RADIAL / SIMPLE_RADIAL_FISHEYE: f cx cy ...
                fx = fy = params[0]
                cx, cy  = params[1], params[2]
            elif model_id in (3, 9):     # RADIAL / RADIAL_FISHEYE: f cx cy ...
                fx = fy = params[0]
                cx, cy  = params[1], params[2]
            else:                        # OPENCV / FISHEYE / FOV / FULL / THIN_PRISM: fx fy cx cy ...
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]

            cameras[cam_id] = {
                "width": width, "height": height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }
    return cameras


def read_images_binary(path: str) -> dict:
    """Parse COLMAP images.bin -> {image_id: {name, cam_id, R[3x3], t[3]}}"""
    images = {}
    with open(path, "rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            (img_id,) = struct.unpack("<I", f.read(4))
            qvec      = struct.unpack("<4d", f.read(32))   # qw qx qy qz
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
    """Parse COLMAP points3D.bin -> {pt_id: {xyz, rgb}}"""
    points3D = {}
    with open(path, "rb") as fid:
        (num_points,) = struct.unpack("<Q", fid.read(8))
        for _ in range(num_points):
            (pt_id,)     = struct.unpack("<Q", fid.read(8))
            xyz          = struct.unpack("<3d", fid.read(24))
            rgb          = struct.unpack("<3B", fid.read(3))
            fid.read(8)                          # reprojection error (float64)
            (track_len,) = struct.unpack("<Q",   fid.read(8))
            fid.read(track_len * 8)              # track entries
            points3D[pt_id] = {"xyz": xyz, "rgb": rgb}
    return points3D


def load_colmap_data(data_dir: str) -> tuple:
    """Load cameras, poses, and optional sparse points from COLMAP binary files."""
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
    log.info(f"  {len(cameras)} cameras.")

    log.info("Loading image poses...")
    images = read_images_binary(img_path)
    log.info(f"  {len(images)} poses.")

    points3D = None
    if os.path.exists(pts_path):
        log.info("Loading sparse points...")
        points3D = read_points3D_binary(pts_path)
        log.info(f"  {len(points3D)} sparse points.")
    else:
        log.warning("points3D.bin not found — random initialisation will be used.")

    return cameras, images, points3D


# =============================================================================
# 4. IMAGE LOADER
# =============================================================================

def load_images(data_dir: str, image_metas: dict, downscale: int = 2) -> tuple:
    """
    Load ground-truth images.
    Returns:
        gt_images : {image_id -> Tensor[H, W, 3] float32, CPU}
        hw        : {image_id -> (H, W)}
    """
    img_dir   = os.path.join(data_dir, "images")
    gt_images = {}
    hw        = {}

    log.info(f"Loading images from {img_dir}  (downscale x{downscale}) ...")
    for img_id, meta in sorted(image_metas.items()):
        fpath = os.path.join(img_dir, meta["name"])
        if not os.path.exists(fpath):
            log.warning(f"  Image not found, skipping: {fpath}")
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

    log.info(f"  {len(gt_images)} images loaded.")
    return gt_images, hw


# =============================================================================
# 5. GAUSSIAN INITIALISATION
# =============================================================================

def initialize_from_colmap(data_dir: str, device: torch.device,
                            sh_coeffs: int = SH_COEFFS):
    """
    Initialise Gaussians from COLMAP sparse points.
    Returns (means, sh_colors, opacities, scales, quats) or None.
    """
    points_path = os.path.join(data_dir, "sparse", "0", "points3D.bin")
    if not os.path.exists(points_path):
        log.warning("points3D.bin not found — falling back to random init.")
        return None

    log.info(f"Initialising from: {points_path}")
    points3D   = read_points3D_binary(points_path)
    num_points = len(points3D)
    log.info(f"  {num_points} sparse points.")

    xyzs = np.array([pt["xyz"] for pt in points3D.values()], dtype=np.float32)
    rgbs = np.array([pt["rgb"] for pt in points3D.values()], dtype=np.float32) / 255.0

    means     = torch.tensor(xyzs, device=device)
    sh_colors = torch.zeros((num_points, sh_coeffs, 3), device=device)
    sh_colors[:, 0, :] = (torch.tensor(rgbs, device=device) - 0.5) / SH_C0

    opacities   = torch.full((num_points, 1),  -2.2, device=device)
    scales      = torch.full((num_points, 3),  -3.0, device=device)
    quats       = torch.zeros((num_points, 4), device=device)
    quats[:, 0] = 1.0

    return means, sh_colors, opacities, scales, quats


# =============================================================================
# 6. CAMERA UTILITIES
# =============================================================================

def build_viewmat(R: np.ndarray, t: np.ndarray,
                  device: torch.device) -> torch.Tensor:
    """4x4 world-to-camera matrix (OpenCV convention) for gsplat."""
    W2C         = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R
    W2C[:3,  3] = t
    return torch.tensor(W2C, dtype=torch.float32, device=device)


def build_K(cam: dict, downscale: int, device: torch.device) -> torch.Tensor:
    """3x3 intrinsic matrix adjusted for the downscaled resolution."""
    s = 1.0 / downscale
    return torch.tensor(
        [[cam["fx"]*s,  0.,          cam["cx"]*s],
         [0.,           cam["fy"]*s, cam["cy"]*s],
         [0.,           0.,          1.]],
        dtype=torch.float32, device=device,
    )


# =============================================================================
# 7. RENDERER
# =============================================================================

def render_scene(
    means, sh_colors, opacities, scales, quats,
    viewmat, K, H, W, device, sh_degree=SH_DEGREE
) -> torch.Tensor:
    """
    Render Gaussians with gsplat + spherical harmonics.
    Returns [H, W, 3] float32, NaN/Inf-safe.
    Scales are clamped before exp() to prevent overflow.
    """
    if not GSPLAT_AVAILABLE:
        raise RuntimeError("gsplat is not installed. Run: pip install gsplat")

    act_opacities = torch.sigmoid(opacities).squeeze(-1)
    act_scales    = torch.exp(scales.clamp(-10.0, 10.0))   # prevent inf
    act_quats     = F.normalize(quats, dim=-1)

    render_colors, _alphas, _meta = rasterization(
        means      = means,
        quats      = act_quats,
        scales     = act_scales,
        opacities  = act_opacities,
        colors     = sh_colors.unsqueeze(0),   # [1, N, K, 3]
        viewmats   = viewmat.unsqueeze(0),     # [1, 4, 4]
        Ks         = K.unsqueeze(0),           # [1, 3, 3]
        width      = W,
        height     = H,
        sh_degree  = sh_degree,
        near_plane = 0.01,
        far_plane  = 1e10,
        render_mode= "RGB",
    )

    rendered = render_colors[0]   # [H, W, 3]

    # Guard against NaN/Inf from degenerate Gaussians
    if not torch.isfinite(rendered).all():
        log.warning("Non-finite values in render — replacing with 0.")
        rendered = torch.nan_to_num(rendered, nan=0.0, posinf=1.0, neginf=0.0)

    return rendered


# =============================================================================
# 8. PLY EXPORT
# =============================================================================

def write_ply(path: str, means, sh_colors, opacities, scales, quats) -> None:
    """Export to a 3DGS-compatible PLY file."""
    N        = means.shape[0]
    means_np = means.detach().cpu().float().numpy()
    sh_np    = sh_colors.detach().cpu().float().numpy()   # [N, K, 3]
    ops_np   = opacities.detach().cpu().float().numpy().reshape(-1, 1)
    sc_np    = scales.detach().cpu().float().numpy()
    qt_np    = quats.detach().cpu().float().numpy()
    nm_np    = np.zeros((N, 3), dtype=np.float32)

    f_dc     = sh_np[:, 0, :]
    f_rest   = sh_np[:, 1:, :].reshape(N, -1)
    n_rest   = f_rest.shape[1]

    actual_sh_degree = int(round(math.sqrt(sh_np.shape[1]))) - 1

    with open(path, "wb") as fout:
        fout.write(b"ply\nformat binary_little_endian 1.0\n")
        fout.write(f"element vertex {N}\n".encode())
        fout.write(b"property float x\nproperty float y\nproperty float z\n")
        fout.write(b"property float nx\nproperty float ny\nproperty float nz\n")
        fout.write(b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n")
        for i in range(n_rest):
            fout.write(f"property float f_rest_{i}\n".encode())
        fout.write(b"property float opacity\n")
        fout.write(b"property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
        fout.write(b"property float rot_0\nproperty float rot_1\n"
                   b"property float rot_2\nproperty float rot_3\n")
        fout.write(b"end_header\n")

        data = np.hstack(
            [means_np, nm_np, f_dc, f_rest, ops_np, sc_np, qt_np]
        ).astype(np.float32)
        fout.write(data.tobytes())

    log.info(f"[Export] {N} Gaussians (SH degree {actual_sh_degree}) -> {path}")


# =============================================================================
# 9. PRUNING
# =============================================================================

def prune_gaussians(
    means, sh_colors, opacities, scales, quats,
    optimizer, bbox_min, bbox_max,
    opacity_threshold: float = 0.01,
):
    """
    Remove low-opacity / out-of-bbox Gaussians and patch the Adam state.

    Uses _slice_optim_state() which safely handles the 0-dim 'step' tensor
    that Adam stores in PyTorch >= 2.0 (the source of the IndexError crash).
    """
    with torch.no_grad():
        act_op  = torch.sigmoid(opacities.squeeze(-1))
        in_bbox = ((means >= bbox_min) & (means <= bbox_max)).all(dim=-1)
        keep    = (act_op > opacity_threshold) & in_bbox

    n_removed = int((~keep).sum().item())
    if n_removed == 0:
        return means, sh_colors, opacities, scales, quats

    param_map = {
        "xyz":     (means,     means[keep].detach()),
        "color":   (sh_colors, sh_colors[keep].detach()),
        "opacity": (opacities, opacities[keep].detach()),
        "scale":   (scales,    scales[keep].detach()),
        "rot":     (quats,     quats[keep].detach()),
    }

    new_tensors = {}
    for group in optimizer.param_groups:
        name            = group["name"]
        old_p, new_data = param_map[name]
        new_p           = new_data.requires_grad_(True)
        new_tensors[name] = new_p

        old_state = optimizer.state.pop(old_p, {})
        new_state = {}
        for k, v in old_state.items():
            # _slice_optim_state checks v.ndim > 0 before v.shape[0]
            # This prevents IndexError on Adam's 0-dim 'step' tensor
            new_state[k] = _slice_optim_state(v, keep)
        optimizer.state[new_p] = new_state
        group["params"] = [new_p]

    log.info(f"  [Prune] Removed {n_removed} -> {int(keep.sum().item())} remain.")
    return (new_tensors["xyz"],    new_tensors["color"],
            new_tensors["opacity"], new_tensors["scale"],
            new_tensors["rot"])


# =============================================================================
# 10. DENSIFICATION
# =============================================================================

def densify_gaussians(
    means, sh_colors, opacities, scales, quats,
    optimizer, grad_accum, grad_count,
    grad_threshold: float, max_gaussians: int,
):
    """
    Split Gaussians with high average position-gradient and remove parents.
    Uses 0-dim-safe state helpers throughout.
    """
    device = means.device
    N      = means.shape[0]

    # Defensive shape guard
    if grad_accum.shape[0] != N or grad_count.shape[0] != N:
        log.warning("grad_accum/grad_count shape mismatch — resetting.")
        grad_accum = torch.zeros(N, device=device)
        grad_count = torch.zeros(N, device=device, dtype=torch.long)

    avg_grad          = torch.zeros(N, device=device)
    counted           = grad_count > 0
    avg_grad[counted] = grad_accum[counted] / grad_count[counted].float()

    to_split  = avg_grad > grad_threshold
    keep_mask = ~to_split
    n_split   = int(to_split.sum().item())
    n_keep    = int(keep_mask.sum().item())

    if n_split == 0 or N >= max_gaussians:
        return (means, sh_colors, opacities, scales, quats,
                torch.zeros(N, device=device),
                torch.zeros(N, device=device, dtype=torch.long))

    # Build 2 children per parent
    s_means      = means[to_split]
    s_scales_exp = torch.exp(scales[to_split].clamp(-10.0, 10.0))
    noise        = (torch.randn_like(s_means)
                    * s_scales_exp.mean(dim=-1, keepdim=True) * 0.3)

    child_means  = torch.cat([s_means + noise, s_means - noise], dim=0)
    child_colors = sh_colors[to_split].repeat(2, 1, 1)
    child_ops    = (opacities[to_split] - 0.5).clamp(-10.0, 5.0).repeat(2, 1)
    child_scales = (scales[to_split] - math.log(2)).repeat(2, 1)
    child_quats  = quats[to_split].repeat(2, 1)

    n_children  = child_means.shape[0]
    n_available = max(0, max_gaussians - n_keep)   # prevent negative slice
    if n_children > n_available:
        n_children   = n_available
        child_means  = child_means[:n_children]
        child_colors = child_colors[:n_children]
        child_ops    = child_ops[:n_children]
        child_scales = child_scales[:n_children]
        child_quats  = child_quats[:n_children]

    if n_children == 0:
        return (means, sh_colors, opacities, scales, quats,
                torch.zeros(N, device=device),
                torch.zeros(N, device=device, dtype=torch.long))

    def cat_kept_and_new(orig, new_data):
        return torch.cat(
            [orig[keep_mask].detach(), new_data.detach()], dim=0
        ).requires_grad_(True)

    new_means     = cat_kept_and_new(means,     child_means)
    new_colors    = cat_kept_and_new(sh_colors, child_colors)
    new_opacities = cat_kept_and_new(opacities, child_ops)
    new_scales    = cat_kept_and_new(scales,    child_scales)
    new_quats     = cat_kept_and_new(quats,     child_quats)
    N_new         = new_means.shape[0]

    param_map = {
        "xyz":     (means,     new_means),
        "color":   (sh_colors, new_colors),
        "opacity": (opacities, new_opacities),
        "scale":   (scales,    new_scales),
        "rot":     (quats,     new_quats),
    }

    for group in optimizer.param_groups:
        name         = group["name"]
        old_p, new_p = param_map[name]
        old_state    = optimizer.state.pop(old_p, {})
        new_state    = {}
        for k, v in old_state.items():
            # 0-dim safe: keep survivors, zero-pad for children
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == N:
                kept = v[keep_mask]
                pad  = torch.zeros(n_children, *v.shape[1:],
                                   device=device, dtype=v.dtype)
                new_state[k] = torch.cat([kept, pad], dim=0)
            else:
                new_state[k] = v   # 0-dim step or scalar — keep as-is
        optimizer.state[new_p] = new_state
        group["params"]        = [new_p]

    log.info(f"  [Densify] Split {n_split} -> +{n_children} children | total {N_new}")

    return (new_means, new_colors, new_opacities, new_scales, new_quats,
            torch.zeros(N_new, device=device),
            torch.zeros(N_new, device=device, dtype=torch.long))


# =============================================================================
# 11. LOSS
# =============================================================================

def compute_loss(rendered, gt, rendered_bchw, gt_bchw,
                 lambda_ssim: float) -> tuple:
    """
    Combined L1 + optional SSIM loss.
    Returns (total_loss, l1_loss, ssim_val_or_None).
    ssim_val is always defined so logging never raises NameError.
    """
    l1_loss  = F.l1_loss(rendered, gt)
    ssim_val = None

    if SSIM_AVAILABLE and lambda_ssim > 0.0 and ssim_fn is not None:
        score    = ssim_fn(rendered_bchw, gt_bchw,
                           data_range=1.0, size_average=True)
        ssim_val = float(score.item())
        loss     = l1_loss + lambda_ssim * (1.0 - score)
    else:
        loss = l1_loss

    return loss, l1_loss, ssim_val


# =============================================================================
# 12. MAIN
# =============================================================================

def main() -> None:
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    max_sh_degree = args.sh_degree
    sh_coeffs     = (max_sh_degree + 1) ** 2
    lambda_ssim   = args.lambda_ssim

    log.info(f"Max SH degree={max_sh_degree} ({sh_coeffs} coeffs/channel) | "
             f"lambda_ssim={lambda_ssim}")

    bbox_min, bbox_max = parse_bbox(args.bbox, device)

    # ── Load COLMAP data ──────────────────────────────────────────────────────
    cameras, image_metas, _ = load_colmap_data(args.data_dir)
    gt_images, hw           = load_images(args.data_dir, image_metas,
                                          args.image_downscale)
    image_ids = sorted(gt_images.keys())

    if not image_ids:
        raise RuntimeError("No images loaded — check data/images/")

    # ── Initialise Gaussians ──────────────────────────────────────────────────
    log.info("Initialising Gaussians...")
    colmap_data = initialize_from_colmap(args.data_dir, device, sh_coeffs)

    if colmap_data is not None:
        means, sh_colors, opacities, scales, quats = colmap_data
    else:
        num_init           = 5_000
        means              = torch.rand((num_init, 3), device=device) * 2 - 1
        sh_colors          = torch.zeros((num_init, sh_coeffs, 3), device=device)
        sh_colors[:, 0, :] = (torch.rand((num_init, 3), device=device) - 0.5) / SH_C0
        opacities          = torch.full((num_init, 1),  -2.2, device=device)
        scales             = torch.full((num_init, 3),  -3.0, device=device)
        quats              = torch.zeros((num_init, 4), device=device)
        quats[:, 0]        = 1.0

    # Shape assertions
    N = means.shape[0]
    assert sh_colors.shape == (N, sh_coeffs, 3), "sh_colors shape error"
    assert opacities.shape == (N, 1),            "opacities shape error"
    assert scales.shape    == (N, 3),            "scales shape error"
    assert quats.shape     == (N, 4),            "quats shape error"
    log.info(f"  {N} Gaussians initialised.")

    means.requires_grad_(True)
    sh_colors.requires_grad_(True)
    opacities.requires_grad_(True)
    scales.requires_grad_(True)
    quats.requires_grad_(True)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = optim.Adam([
        {"params": [means],     "lr": args.lr_pos,     "name": "xyz"},
        {"params": [sh_colors], "lr": args.lr_color,   "name": "color"},
        {"params": [opacities], "lr": args.lr_opacity, "name": "opacity"},
        {"params": [scales],    "lr": args.lr_scale,   "name": "scale"},
        {"params": [quats],     "lr": args.lr_rot,     "name": "rot"},
    ])

    grad_accum = torch.zeros(N, device=device)
    grad_count = torch.zeros(N, device=device, dtype=torch.long)

    os.makedirs("val_renders", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    log.info(f"Training: {args.iterations} iters | {N} Gaussians | {device}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for step in range(1, args.iterations + 1):
        optimizer.zero_grad()

        # SH degree scheduler: 0 -> max_sh_degree, +1 every 1000 steps
        current_sh_degree = min(max_sh_degree, (step - 1) // 1000)

        # Sample one random training view
        img_id  = image_ids[torch.randint(len(image_ids), (1,)).item()]
        meta    = image_metas[img_id]
        cam     = cameras[meta["cam_id"]]
        H, W    = hw[img_id]
        viewmat = build_viewmat(meta["R"], meta["t"], device)
        K       = build_K(cam, args.image_downscale, device)
        gt      = gt_images[img_id].to(device)   # [H, W, 3]

        rendered = render_scene(
            means, sh_colors, opacities, scales, quats,
            viewmat, K, H, W, device,
            sh_degree=current_sh_degree,
        )

        rendered_bchw = rendered.permute(2, 0, 1).unsqueeze(0).contiguous()
        gt_bchw       = gt.permute(2, 0, 1).unsqueeze(0).contiguous()

        loss, l1_loss, ssim_val = compute_loss(
            rendered, gt, rendered_bchw, gt_bchw, lambda_ssim
        )

        loss.backward()

        # Accumulate position gradients — only count where gradient is nonzero
        if means.grad is not None:
            g    = means.grad.detach().norm(dim=-1)
            size = min(g.shape[0], grad_accum.shape[0])
            grad_accum[:size] += g[:size]
            grad_count[:size] += (g[:size] > 0).long()

        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % 500 == 0:
            vram_gb  = (torch.cuda.memory_allocated() / 1e9
                        if device.type == "cuda" else 0.0)
            ssim_str = (f" SSIM={ssim_val:.4f}" if ssim_val is not None else "")
            log.info(
                f"Iter {step:6d} | Loss {loss.item():.5f} "
                f"(L1={l1_loss.item():.5f}{ssim_str}) | "
                f"SH={current_sh_degree} | "
                f"Gaussians={means.shape[0]:6d} | VRAM={vram_gb:.2f}GB"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % args.checkpoint_every == 0:
            ckpt_path = f"checkpoints/ckpt_{step:07d}.pt"
            torch.save({
                "step":           step,
                "sh_degree":      max_sh_degree,
                "current_sh_deg": current_sh_degree,
                "means":          means.detach().cpu(),
                "sh_colors":      sh_colors.detach().cpu(),
                "opacities":      opacities.detach().cpu(),
                "scales":         scales.detach().cpu(),
                "quats":          quats.detach().cpu(),
                "optimizer":      optimizer.state_dict(),
            }, ckpt_path)
            log.info(f"  [Checkpoint] -> {ckpt_path}")

        # ── Opacity reset ─────────────────────────────────────────────────────
        if step % args.opacity_reset_interval == 0:
            with torch.no_grad():
                opacities.fill_(-2.2)
            # Clear Adam momentum so the reset is not immediately undone
            for group in optimizer.param_groups:
                if group["name"] == "opacity":
                    for p in group["params"]:
                        st = optimizer.state.get(p, {})
                        if "exp_avg"    in st: st["exp_avg"].zero_()
                        if "exp_avg_sq" in st: st["exp_avg_sq"].zero_()
                    break
            log.info(f"  [Opacity] Reset at step {step}")

        # ── Validation render ─────────────────────────────────────────────────
        if step % 1000 == 0:
            val_id = image_ids[0]
            vH, vW = hw[val_id]
            vmat   = build_viewmat(image_metas[val_id]["R"],
                                   image_metas[val_id]["t"], device)
            vK     = build_K(cameras[image_metas[val_id]["cam_id"]],
                             args.image_downscale, device)
            with torch.no_grad():
                val_img = render_scene(
                    means, sh_colors, opacities, scales, quats,
                    vmat, vK, vH, vW, device,
                    sh_degree=current_sh_degree,
                )
            arr = (val_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(f"val_renders/step_{step:07d}.png")

        # ── Densification + Pruning ────────────────────────────────────────────
        if (args.densify_start <= step <= args.densify_end
                and step % args.densify_every == 0):

            means, sh_colors, opacities, scales, quats = prune_gaussians(
                means, sh_colors, opacities, scales, quats,
                optimizer, bbox_min, bbox_max,
                opacity_threshold=args.opacity_threshold,
            )

            # Reset grad buffers to match post-prune population
            N_pruned   = means.shape[0]
            grad_accum = torch.zeros(N_pruned, device=device)
            grad_count = torch.zeros(N_pruned, device=device, dtype=torch.long)

            if means.shape[0] < args.max_gaussians:
                (means, sh_colors, opacities, scales, quats,
                 grad_accum, grad_count) = densify_gaussians(
                    means, sh_colors, opacities, scales, quats,
                    optimizer, grad_accum, grad_count,
                    args.densify_grad_threshold, args.max_gaussians,
                )
            else:
                N_cur      = means.shape[0]
                grad_accum = torch.zeros(N_cur, device=device)
                grad_count = torch.zeros(N_cur, device=device, dtype=torch.long)

    # ── Final PLY export ──────────────────────────────────────────────────────
    log.info("Training complete. Exporting PLY...")
    write_ply(args.output, means, sh_colors, opacities, scales, quats)

    val_id = image_ids[0]
    vH, vW = hw[val_id]
    vmat   = build_viewmat(image_metas[val_id]["R"],
                           image_metas[val_id]["t"], device)
    vK     = build_K(cameras[image_metas[val_id]["cam_id"]],
                     args.image_downscale, device)
    with torch.no_grad():
        final_img = render_scene(
            means, sh_colors, opacities, scales, quats,
            vmat, vK, vH, vW, device,
            sh_degree=max_sh_degree,
        )
    arr = (final_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save("val_renders/final.png")
    log.info("Done. Check output.ply and val_renders/final.png")


if __name__ == "__main__":
    main()