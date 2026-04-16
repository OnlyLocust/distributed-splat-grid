"""
Splat-Grid — Worker Node (v3 — SQLite-aware, Fault Tolerant)
=============================================================
Run this on each participating worker machine.

Usage:
    python worker.py --master http://192.168.1.10:8765 --iterations 700

Flow
----
1. POST /join          → receive unique worker_id
2. GET  /get_task      → receive a voxel task descriptor
3. Download images from master, initialise Gaussians, train with gsplat
4. POST /heartbeat     every 30 s during training (background thread)
5. POST /submit_result → upload finished .ply chunk
6. Repeat until no tasks remain

Memory-Safety Constraints (V1 rules — do NOT relax without OOM testing)
------------------------------------------------------------------------
  - IMAGE_DOWNSCALE  = 4  (1/4 resolution per axis)
  - DEFAULT_ITERATIONS = 700
  - SH_DEGREE = 0  (DC-only, keeps color tensor tiny)
  - torch.cuda.empty_cache() every CACHE_CLEAR_EVERY iterations
  - MAX_GAUSSIANS hard cap at 50 000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────────────

_log_handler = logging.FileHandler("worker.log", encoding="utf-8")
_log_handler.setFormatter(logging.Formatter(
    "%(asctime)s [WORKER][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [WORKER][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logging.basicConfig(level=logging.INFO, handlers=[_log_handler, _console_handler])
log = logging.getLogger(__name__)

# ── Optional: gsplat rasterizer ───────────────────────────────────────────────
try:
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    rasterization    = None
    GSPLAT_AVAILABLE = False
    log.warning("gsplat not installed — run: pip install gsplat")

# ── Memory-safety constants ───────────────────────────────────────────────────
SH_DEGREE          = 0
SH_COEFFS          = (SH_DEGREE + 1) ** 2   # = 1
SH_C0              = 0.28209479177387814
IMAGE_DOWNSCALE    = 4
DEFAULT_ITERATIONS = 700
MAX_GAUSSIANS      = 50_000
DENSIFY_START      = 100
DENSIFY_EVERY      = 50
DENSIFY_END        = 400
DENSIFY_GRAD_THRES = 0.0002
OPACITY_THRESHOLD  = 0.01
OPACITY_RESET_INT  = 200
CACHE_CLEAR_EVERY  = 50
HEARTBEAT_INTERVAL = 30   # seconds between heartbeat POSTs
SEED               = 42


# ── Seed utility ──────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Optimiser state helpers ───────────────────────────────────────────────────

def _slice_optim_state(v, mask):
    if isinstance(v, torch.Tensor) and v.ndim > 0:
        return v[mask]
    return v


def _pad_optim_state(v, n_old: int, n_pad: int, device):
    if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == n_old:
        pad = torch.zeros(n_pad, *v.shape[1:], device=device, dtype=v.dtype)
        return torch.cat([v, pad], dim=0)
    return v


# ── Master HTTP client ────────────────────────────────────────────────────────

class MasterClient:
    def __init__(self, base_url: str, timeout: int = 120):
        self.base    = base_url.rstrip("/")
        self.timeout = timeout
        self.sess    = requests.Session()
        self.sess.headers.update({"User-Agent": "SplatGrid-Worker/3.0"})
        self.worker_id: str | None = None

    # ── Registration ──────────────────────────────────────────────────────────

    def join(self) -> str:
        """Register with the master; stores and returns worker_id."""
        for attempt in range(1, 6):
            try:
                r = self.sess.get(f"{self.base}/join", timeout=self.timeout)
                r.raise_for_status()
                wid = r.json()["worker_id"]
                self.worker_id = wid
                log.info(f"[Join] Registered as worker {wid[:8]}…")
                return wid
            except requests.exceptions.RequestException as exc:
                log.warning(f"[Join] Attempt {attempt}/5 failed: {exc}")
                time.sleep(5)
        raise ConnectionError("Could not connect to master after 5 attempts.")

    # ── Task request ──────────────────────────────────────────────────────────

    def request_task(self) -> dict | None:
        """
        Returns a task dict, or None when the queue is truly exhausted.
        - HTTP 404: no PENDING tasks (may still be IN_PROGRESS)
        - HTTP 429: fair-share cap; back off and retry transparently
        """
        for _attempt in range(10):
            try:
                r = self.sess.get(
                    f"{self.base}/get_task",
                    params={"worker_id": self.worker_id},
                    timeout=self.timeout,
                )
                if r.status_code == 404:
                    return None          # no PENDING tasks right now
                if r.status_code == 429:
                    # Fair-share cap: back off silently, then let the
                    # caller decide whether to keep waiting.
                    log.info("[request_task] Fair-share cap — backing off 15s …")
                    time.sleep(15)
                    return None          # caller will retry via its own loop
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as exc:
                log.error(f"[request_task] Network error: {exc}")
                time.sleep(5)
        return None

    def get_queue_status(self) -> dict:
        """Query /status to see how many tasks are PENDING vs IN_PROGRESS."""
        try:
            r = self.sess.get(f"{self.base}/status", timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException:
            return {}

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    def send_heartbeat(self, task_id: str) -> bool:
        try:
            r = self.sess.post(
                f"{self.base}/heartbeat",
                json={"worker_id": self.worker_id, "task_id": task_id},
                timeout=30,
            )
            r.raise_for_status()
            return True
        except requests.exceptions.RequestException as exc:
            log.warning(f"[heartbeat] Failed: {exc}")
            return False

    # ── File downloads ────────────────────────────────────────────────────────

    def download_image(self, image_name: str, dest: Path) -> bool:
        try:
            r = self.sess.get(
                f"{self.base}/image/{image_name}",
                timeout=self.timeout, stream=True,
            )
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            return True
        except requests.exceptions.RequestException as exc:
            log.warning(f"[download_image] {image_name}: {exc}")
            return False

    def download_sparse_file(self, filename: str, dest: Path) -> bool:
        try:
            r = self.sess.get(
                f"{self.base}/sparse/{filename}",
                timeout=self.timeout, stream=True,
            )
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            return True
        except requests.exceptions.RequestException as exc:
            log.warning(f"[download_sparse] {filename}: {exc}")
            return False

    # ── Result upload ─────────────────────────────────────────────────────────

    def submit_ply(self, task_id: str, ply_path: Path) -> bool:
        try:
            with open(ply_path, "rb") as f:
                r = self.sess.post(
                    f"{self.base}/submit_result/{task_id}",
                    params={"worker_id": self.worker_id},
                    files={"file": (ply_path.name, f, "application/octet-stream")},
                    timeout=self.timeout * 5,
                )
            r.raise_for_status()
            log.info(
                f"[submit_ply] {ply_path.name} uploaded "
                f"({ply_path.stat().st_size / 1024:.1f} KB) → {r.json()}"
            )
            return True
        except requests.exceptions.RequestException as exc:
            log.error(f"[submit_ply] Failed: {exc}")
            return False


# ── Heartbeat background thread ───────────────────────────────────────────────

class HeartbeatThread(threading.Thread):
    """Sends a heartbeat to the master every HEARTBEAT_INTERVAL seconds."""

    def __init__(self, client: MasterClient, task_id: str):
        super().__init__(daemon=True)
        self.client   = client
        self.task_id  = task_id
        self._stop_ev = threading.Event()

    def stop(self) -> None:
        self._stop_ev.set()

    def run(self) -> None:
        log.info(f"[HeartbeatThread] Started for task {self.task_id[:8]}…")
        while not self._stop_ev.wait(timeout=HEARTBEAT_INTERVAL):
            ok = self.client.send_heartbeat(self.task_id)
            if not ok:
                log.warning(f"[HeartbeatThread] Heartbeat missed for task {self.task_id[:8]}…")
        log.info(f"[HeartbeatThread] Stopped for task {self.task_id[:8]}…")


# ── Image loading ─────────────────────────────────────────────────────────────

def load_images_from_task(
    task: dict,
    client: MasterClient,
    local_img_dir: Path,
    downscale: int = IMAGE_DOWNSCALE,
) -> tuple[dict, dict]:
    """
    Download (if missing) and load all task images.
    Returns: gt_images {img_id → Tensor[H,W,3]}, hw {img_id → (H, W)}
    """
    gt_images: dict = {}
    hw: dict        = {}

    for img_meta in task["images"]:
        img_id   = img_meta["image_id"]
        img_name = img_meta["name"]
        dest     = local_img_dir / img_name

        if not dest.exists():
            ok = client.download_image(img_name, dest)
            if not ok:
                log.warning(f"  Skipping {img_name} — download failed.")
                continue

        try:
            img = Image.open(dest).convert("RGB")
            if downscale > 1:
                img = img.resize(
                    (img.width // downscale, img.height // downscale),
                    Image.LANCZOS,
                )
            arr              = np.array(img, dtype=np.float32) / 255.0
            gt_images[img_id] = torch.from_numpy(arr)   # [H, W, 3]
            hw[img_id]        = (arr.shape[0], arr.shape[1])
        except Exception as exc:
            log.warning(f"  Could not load {dest}: {exc}")

    log.info(f"  {len(gt_images)} images loaded (downscale ×{downscale})")
    return gt_images, hw


# ── Gaussian initialisation ───────────────────────────────────────────────────

def initialize_gaussians(
    task: dict,
    local_sparse_dir: Path,
    client: MasterClient,
    device: torch.device,
    sh_coeffs: int = SH_COEFFS,
):
    """
    Seed Gaussians from sparse COLMAP points that fall inside the voxel.
    Falls back to uniform random if no matching points exist.
    """
    pts_path = local_sparse_dir / "points3D.bin"
    if not pts_path.exists():
        if not client.download_sparse_file("points3D.bin", pts_path):
            log.warning("Could not download points3D.bin — using random init.")
            pts_path = None

    bbox_min = np.array(task["bbox_min"], dtype=np.float32)
    bbox_max = np.array(task["bbox_max"], dtype=np.float32)

    xyzs, rgbs = [], []
    if pts_path and pts_path.exists():
        from colmap_utils import read_points3D_binary
        pts = read_points3D_binary(str(pts_path))
        for pt in pts.values():
            xyz = np.array(pt["xyz"], dtype=np.float32)
            if np.all(xyz >= bbox_min) and np.all(xyz <= bbox_max):
                xyzs.append(xyz)
                rgbs.append(np.array(pt["rgb"], dtype=np.float32) / 255.0)

    if xyzs:
        xyzs_arr  = np.array(xyzs, dtype=np.float32)
        rgbs_arr  = np.array(rgbs, dtype=np.float32)
        means     = torch.tensor(xyzs_arr, device=device)
        sh_colors = torch.zeros((len(xyzs), sh_coeffs, 3), device=device)
        sh_colors[:, 0, :] = (torch.tensor(rgbs_arr, device=device) - 0.5) / SH_C0
        log.info(f"  Seeded {len(xyzs)} Gaussians from sparse points in voxel.")
    else:
        num_init = 2_000
        log.info(f"  No sparse points in bbox — random init with {num_init} Gaussians.")
        means     = (
            torch.rand((num_init, 3), device=device)
            * torch.tensor(bbox_max - bbox_min, device=device)
            + torch.tensor(bbox_min, device=device)
        )
        sh_colors = torch.zeros((num_init, sh_coeffs, 3), device=device)
        sh_colors[:, 0, :] = (torch.rand((num_init, 3), device=device) - 0.5) / SH_C0

    N           = means.shape[0]
    opacities   = torch.full((N, 1),  -2.2, device=device)
    scales      = torch.full((N, 3),  -3.0, device=device)
    quats       = torch.zeros((N, 4), device=device)
    quats[:, 0] = 1.0
    return means, sh_colors, opacities, scales, quats


# ── Camera utilities ──────────────────────────────────────────────────────────

def build_viewmat(R_list: list, t_list: list, device: torch.device) -> torch.Tensor:
    R   = np.array(R_list, dtype=np.float32)
    t   = np.array(t_list, dtype=np.float32)
    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R
    W2C[:3, 3]  = t
    return torch.tensor(W2C, dtype=torch.float32, device=device)


def build_K(cam: dict, downscale: int, device: torch.device) -> torch.Tensor:
    s = 1.0 / downscale
    return torch.tensor(
        [
            [cam["fx"] * s, 0.0,           cam["cx"] * s],
            [0.0,           cam["fy"] * s, cam["cy"] * s],
            [0.0,           0.0,           1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


# ── Renderer ──────────────────────────────────────────────────────────────────

def render_scene(
    means, sh_colors, opacities, scales, quats,
    viewmat, K, H, W, device,
    sh_degree: int = SH_DEGREE,
) -> torch.Tensor:
    if not GSPLAT_AVAILABLE:
        raise RuntimeError("gsplat is not installed. Run: pip install gsplat")

    act_opacities = torch.sigmoid(opacities).squeeze(-1)
    act_scales    = torch.exp(scales.clamp(-10.0, 10.0))
    act_quats     = F.normalize(quats, dim=-1)

    render_colors, _alphas, _meta = rasterization(
        means      = means,
        quats      = act_quats,
        scales     = act_scales,
        opacities  = act_opacities,
        colors     = sh_colors.unsqueeze(0),   # [1, N, K, 3]
        viewmats   = viewmat.unsqueeze(0),      # [1, 4, 4]
        Ks         = K.unsqueeze(0),            # [1, 3, 3]
        width      = W,
        height     = H,
        sh_degree  = sh_degree,
        near_plane = 0.01,
        far_plane  = 1e10,
        render_mode= "RGB",
    )

    rendered = render_colors[0]   # [H, W, 3]
    if not torch.isfinite(rendered).all():
        rendered = torch.nan_to_num(rendered, nan=0.0, posinf=1.0, neginf=0.0)
    return rendered


# ── Pruning ───────────────────────────────────────────────────────────────────

def prune_gaussians(
    means, sh_colors, opacities, scales, quats,
    optimizer,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    opacity_threshold: float = OPACITY_THRESHOLD,
):
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
        new_state = {k: _slice_optim_state(v, keep) for k, v in old_state.items()}
        optimizer.state[new_p] = new_state
        group["params"]        = [new_p]

    log.info(f"  [Prune] -{n_removed} → {int(keep.sum())} remain")
    return (
        new_tensors["xyz"],     new_tensors["color"],
        new_tensors["opacity"], new_tensors["scale"], new_tensors["rot"],
    )


# ── Densification ─────────────────────────────────────────────────────────────

def densify_gaussians(
    means, sh_colors, opacities, scales, quats,
    optimizer, grad_accum, grad_count,
    grad_threshold: float, max_gaussians: int,
):
    device = means.device
    N      = means.shape[0]

    if grad_accum.shape[0] != N or grad_count.shape[0] != N:
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
        return (
            means, sh_colors, opacities, scales, quats,
            torch.zeros(N, device=device),
            torch.zeros(N, device=device, dtype=torch.long),
        )

    s_means      = means[to_split]
    s_scales_exp = torch.exp(scales[to_split].clamp(-10.0, 10.0))
    noise        = (
        torch.randn_like(s_means)
        * s_scales_exp.mean(dim=-1, keepdim=True) * 0.3
    )

    child_means  = torch.cat([s_means + noise, s_means - noise], dim=0)
    child_colors = sh_colors[to_split].repeat(2, 1, 1)
    child_ops    = (opacities[to_split] - 0.5).clamp(-10.0, 5.0).repeat(2, 1)
    child_scales = (scales[to_split] - math.log(2)).repeat(2, 1)
    child_quats  = quats[to_split].repeat(2, 1)

    n_children  = child_means.shape[0]
    n_available = max(0, max_gaussians - n_keep)
    if n_children > n_available:
        n_children   = n_available
        child_means  = child_means[:n_children]
        child_colors = child_colors[:n_children]
        child_ops    = child_ops[:n_children]
        child_scales = child_scales[:n_children]
        child_quats  = child_quats[:n_children]

    if n_children == 0:
        return (
            means, sh_colors, opacities, scales, quats,
            torch.zeros(N, device=device),
            torch.zeros(N, device=device, dtype=torch.long),
        )

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
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == N:
                kept = v[keep_mask]
                pad  = torch.zeros(n_children, *v.shape[1:], device=device, dtype=v.dtype)
                new_state[k] = torch.cat([kept, pad], dim=0)
            else:
                new_state[k] = v
        optimizer.state[new_p] = new_state
        group["params"]        = [new_p]

    log.info(f"  [Densify] split {n_split} → +{n_children} children | total {N_new}")
    return (
        new_means, new_colors, new_opacities, new_scales, new_quats,
        torch.zeros(N_new, device=device),
        torch.zeros(N_new, device=device, dtype=torch.long),
    )


# ── PLY export ────────────────────────────────────────────────────────────────

def write_ply(path: str, means, sh_colors, opacities, scales, quats) -> None:
    N        = means.shape[0]
    means_np = means.detach().cpu().float().numpy()
    sh_np    = sh_colors.detach().cpu().float().numpy()     # [N, K, 3]
    ops_np   = opacities.detach().cpu().float().numpy().reshape(-1, 1)
    sc_np    = scales.detach().cpu().float().numpy()
    qt_np    = quats.detach().cpu().float().numpy()
    nm_np    = np.zeros((N, 3), dtype=np.float32)

    f_dc   = sh_np[:, 0, :]
    f_rest = sh_np[:, 1:, :].reshape(N, -1)
    n_rest = f_rest.shape[1]

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
        data = np.hstack([means_np, nm_np, f_dc, f_rest, ops_np, sc_np, qt_np]).astype(np.float32)
        fout.write(data.tobytes())

    log.info(f"[PLY] {N} Gaussians → {path}")


# ── Core training loop ────────────────────────────────────────────────────────

def train_chunk(
    task: dict,
    gt_images: dict,
    hw: dict,
    device: torch.device,
    means, sh_colors, opacities, scales, quats,
    lr_pos:       float = 1.6e-4,
    lr_color:     float = 2.5e-3,
    lr_opacity:   float = 0.05,
    lr_scale:     float = 5e-3,
    lr_rot:       float = 1e-3,
    iterations:   int   = DEFAULT_ITERATIONS,
    max_gaussians:int   = MAX_GAUSSIANS,
) -> tuple:
    """Memory-safe training loop for one voxel chunk."""
    bbox_min  = torch.tensor(task["bbox_min"], dtype=torch.float32, device=device)
    bbox_max  = torch.tensor(task["bbox_max"], dtype=torch.float32, device=device)
    cameras   = task["cameras"]
    img_metas = {m["image_id"]: m for m in task["images"]}
    image_ids = sorted(gt_images.keys())

    if not image_ids:
        log.warning("No images for this chunk — returning initial Gaussians.")
        return means, sh_colors, opacities, scales, quats

    means.requires_grad_(True)
    sh_colors.requires_grad_(True)
    opacities.requires_grad_(True)
    scales.requires_grad_(True)
    quats.requires_grad_(True)

    optimizer = optim.Adam([
        {"params": [means],     "lr": lr_pos,     "name": "xyz"},
        {"params": [sh_colors], "lr": lr_color,   "name": "color"},
        {"params": [opacities], "lr": lr_opacity, "name": "opacity"},
        {"params": [scales],    "lr": lr_scale,   "name": "scale"},
        {"params": [quats],     "lr": lr_rot,     "name": "rot"},
    ])

    N          = means.shape[0]
    grad_accum = torch.zeros(N, device=device)
    grad_count = torch.zeros(N, device=device, dtype=torch.long)

    for step in range(1, iterations + 1):
        optimizer.zero_grad()

        img_id = image_ids[torch.randint(len(image_ids), (1,)).item()]
        meta   = img_metas[img_id]
        cam_id = meta["cam_id"]
        cam    = cameras.get(str(cam_id)) or cameras.get(cam_id)
        if cam is None:
            continue
        H, W    = hw[img_id]
        viewmat = build_viewmat(meta["R"], meta["t"], device)
        K       = build_K(cam, IMAGE_DOWNSCALE, device)
        gt      = gt_images[img_id].to(device)   # [H, W, 3]

        rendered = render_scene(
            means, sh_colors, opacities, scales, quats,
            viewmat, K, H, W, device, sh_degree=SH_DEGREE,
        )

        loss = F.l1_loss(rendered, gt)
        loss.backward()

        if means.grad is not None:
            g    = means.grad.detach().norm(dim=-1)
            size = min(g.shape[0], grad_accum.shape[0])
            grad_accum[:size] += g[:size]
            grad_count[:size] += (g[:size] > 0).long()

        optimizer.step()

        # Periodic VRAM flush
        if step % CACHE_CLEAR_EVERY == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if step % 100 == 0:
            vram = (torch.cuda.memory_allocated() / 1e9
                    if device.type == "cuda" else 0.0)
            log.info(
                f"  Step {step:4d}/{iterations} | "
                f"Loss={loss.item():.5f} | "
                f"N={means.shape[0]} | "
                f"VRAM={vram:.2f} GB"
            )

        # Opacity reset
        if step % OPACITY_RESET_INT == 0:
            with torch.no_grad():
                opacities.fill_(-2.2)
            for group in optimizer.param_groups:
                if group["name"] == "opacity":
                    for p in group["params"]:
                        st = optimizer.state.get(p, {})
                        if "exp_avg"    in st: st["exp_avg"].zero_()
                        if "exp_avg_sq" in st: st["exp_avg_sq"].zero_()
                    break

        # Prune + Densify
        if DENSIFY_START <= step <= DENSIFY_END and step % DENSIFY_EVERY == 0:
            means, sh_colors, opacities, scales, quats = prune_gaussians(
                means, sh_colors, opacities, scales, quats,
                optimizer, bbox_min, bbox_max,
            )
            N_pruned   = means.shape[0]
            grad_accum = torch.zeros(N_pruned, device=device)
            grad_count = torch.zeros(N_pruned, device=device, dtype=torch.long)

            if means.shape[0] < max_gaussians:
                (means, sh_colors, opacities, scales, quats,
                 grad_accum, grad_count) = densify_gaussians(
                    means, sh_colors, opacities, scales, quats,
                    optimizer, grad_accum, grad_count,
                    DENSIFY_GRAD_THRES, max_gaussians,
                )
            else:
                N_cur      = means.shape[0]
                grad_accum = torch.zeros(N_cur, device=device)
                grad_count = torch.zeros(N_cur, device=device, dtype=torch.long)

    # Final VRAM cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return means, sh_colors, opacities, scales, quats


# ── Worker main loop ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Splat-Grid Worker Node v3")
    p.add_argument("--master",      type=str, required=True,
                   help="Master node URL, e.g. http://192.168.1.10:8765")
    p.add_argument("--work_dir",    type=str, default="worker_workdir",
                   help="Local scratch directory for images and output PLYs")
    p.add_argument("--iterations",  type=int, default=DEFAULT_ITERATIONS,
                   help=f"Training iterations per chunk (default: {DEFAULT_ITERATIONS})")
    p.add_argument("--max_tasks",   type=int, default=0,
                   help="Max chunks to process (0 = unlimited — drain queue)")
    p.add_argument("--retry_wait",  type=int, default=30,
                   help="Seconds to wait when master has no tasks ready")
    p.add_argument("--max_retries", type=int, default=20,
                   help="Consecutive 'no task' responses before worker exits "
                        "(default 20 × 30s = 10 min, longer than the 300s watchdog)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    client = MasterClient(args.master)
    client.join()   # Register and get worker_id — raises on failure

    work_dir = Path(args.work_dir)
    img_dir  = work_dir / "images"
    spr_dir  = work_dir / "sparse" / "0"
    out_dir  = work_dir / "output"
    for d in [img_dir, spr_dir, out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tasks_done = 0
    retries    = 0

    while True:
        if args.max_tasks > 0 and tasks_done >= args.max_tasks:
            log.info(f"Reached max_tasks={args.max_tasks}. Exiting.")
            break

        log.info(f"Requesting task from master ({args.master}) …")
        task = client.request_task()

        if task is None:
            # Smart retry: distinguish "tasks still IN_PROGRESS (need to wait
            # for watchdog)" from "queue truly empty (can exit)".
            status      = client.get_queue_status()
            in_progress = status.get("IN_PROGRESS", 0)
            completed   = status.get("COMPLETED",   0)
            total       = status.get("total",        0)

            if total > 0 and completed >= total:
                log.info("All tasks COMPLETED — worker exiting cleanly.")
                break

            if in_progress > 0:
                # Some tasks are still being processed (or stale IN_PROGRESS
                # waiting for the 300s watchdog to reset them).
                # Reset retries so we don't exit before reclaim happens.
                log.info(
                    f"No PENDING task but {in_progress} task(s) still "
                    f"IN_PROGRESS — waiting {args.retry_wait}s for watchdog …"
                )
                retries = 0
                time.sleep(args.retry_wait)
                continue

            retries += 1
            if retries >= args.max_retries:
                log.info("No more tasks available — all chunks likely done. Exiting.")
                break
            log.info(
                f"No task available — waiting {args.retry_wait}s "
                f"(retry {retries}/{args.max_retries})"
            )
            time.sleep(args.retry_wait)
            continue

        retries   = 0
        task_id   = task["task_id"]
        chunk_id  = task["chunk_id"]
        voxel_idx = task["voxel_idx"]
        log.info(
            f"Got task {task_id[:8]}… chunk={chunk_id} voxel={voxel_idx} "
            f"| {len(task['images'])} images"
        )

        # Start heartbeat thread
        hb_thread = HeartbeatThread(client, task_id)
        hb_thread.start()

        try:
            # 1. Download images
            gt_images, hw = load_images_from_task(task, client, img_dir)
            if not gt_images:
                # Voxel has no usable training views (cameras outside bbox).
                # MUST still submit a result so the task reaches COMPLETED;
                # silently skipping leaves it IN_PROGRESS forever, which
                # blocks other workers from getting tasks.
                log.warning(
                    f"[task {task_id[:8]}] No images loaded — "
                    "submitting empty PLY to mark task COMPLETED."
                )
                ply_name = (
                    f"chunk_{task_id[:8]}_"
                    f"{voxel_idx[0]}_{voxel_idx[1]}_{voxel_idx[2]}.ply"
                )
                ply_path = out_dir / ply_name
                with open(ply_path, "wb") as _f:
                    _f.write(b"ply\nformat binary_little_endian 1.0\n")
                    _f.write(b"element vertex 0\n")
                    _f.write(
                        b"property float x\nproperty float y\n"
                        b"property float z\nend_header\n"
                    )
                hb_thread.stop()
                hb_thread.join(timeout=5)
                client.submit_ply(task_id, ply_path)
                tasks_done += 1
                log.info(f"[task {task_id[:8]}] Empty-PLY submitted (0-image voxel).")
                continue

            # 2. Initialise Gaussians
            means, sh_colors, opacities, scales, quats = initialize_gaussians(
                task, spr_dir, client, device,
            )

            # 3. Train
            log.info(
                f"Training {args.iterations} iters on {means.shape[0]} Gaussians …"
            )
            means, sh_colors, opacities, scales, quats = train_chunk(
                task, gt_images, hw, device,
                means, sh_colors, opacities, scales, quats,
                iterations=args.iterations,
            )

            # 4. Export PLY
            ply_name = (
                f"chunk_{task_id[:8]}_"
                f"{voxel_idx[0]}_{voxel_idx[1]}_{voxel_idx[2]}.ply"
            )
            ply_path = out_dir / ply_name
            write_ply(str(ply_path), means, sh_colors, opacities, scales, quats)

        finally:
            # Always stop heartbeat (even if training raised an exception)
            hb_thread.stop()
            hb_thread.join(timeout=5)

        # 5. Upload PLY
        ok = client.submit_ply(task_id, ply_path)
        if not ok:
            log.error(
                f"Upload failed for task {task_id[:8]}. PLY kept at {ply_path}."
            )

        tasks_done += 1
        log.info(f"Task {task_id[:8]} complete. Total done this session: {tasks_done}")

        # Release GPU memory between chunks
        del means, sh_colors, opacities, scales, quats, gt_images
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    log.info("Worker exiting.")


if __name__ == "__main__":
    main()
