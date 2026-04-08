import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from colmap_io import qvec_to_rotmat
from ply_io import write_gaussian_ply

try:
    from gsplat import rasterization
except ImportError:
    rasterization = None


def _gsplat_cuda_backend_ready() -> bool:
    if rasterization is None:
        return False
    try:
        from gsplat.cuda import _backend as _gs_backend  # type: ignore
        return getattr(_gs_backend, "_C", None) is not None
    except Exception:
        return False


def compute_ssim(x: torch.Tensor, y: torch.Tensor, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> torch.Tensor:
    # Global SSIM (lightweight V1); x/y shape: [B, H, W, 3], range [0,1]
    mu_x = x.mean(dim=(1, 2), keepdim=True)
    mu_y = y.mean(dim=(1, 2), keepdim=True)
    sig_x = ((x - mu_x) ** 2).mean(dim=(1, 2), keepdim=True)
    sig_y = ((y - mu_y) ** 2).mean(dim=(1, 2), keepdim=True)
    sig_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(1, 2), keepdim=True)
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2)
    ssim_map = num / (den + 1e-8)
    return ssim_map.mean()


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)


def _load_image(path: str, width: int, height: int, downscale: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tgt_w = max(1, int(width // downscale))
    tgt_h = max(1, int(height // downscale))
    img = img.resize((tgt_w, tgt_h), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


def _build_camera_tensors(camera_batch: List[Dict[str, Any]], downscale: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    viewmats = []
    Ks = []
    tgt_w = None
    tgt_h = None
    for cam in camera_batch:
        R = qvec_to_rotmat(np.array(cam["qvec"], dtype=np.float64))
        t = np.array(cam["tvec"], dtype=np.float64)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R.astype(np.float32)
        view[:3, 3] = t.astype(np.float32)
        viewmats.append(view)

        width = int(cam["width"])
        height = int(cam["height"])
        tw = max(1, int(width // downscale))
        th = max(1, int(height // downscale))
        tgt_w = tw
        tgt_h = th
        sx = tw / float(width)
        sy = th / float(height)

        K = np.array(
            [
                [float(cam["fx"]) * sx, 0.0, float(cam["cx"]) * sx],
                [0.0, float(cam["fy"]) * sy, float(cam["cy"]) * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        Ks.append(K)

    return (
        torch.from_numpy(np.stack(viewmats, axis=0)).to(device),
        torch.from_numpy(np.stack(Ks, axis=0)).to(device),
        int(tgt_w),
        int(tgt_h),
    )


def _run_rasterization(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh_dc: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    if rasterization is None:
        raise ImportError("gsplat is not installed; cannot run rasterization training.")

    # Try common gsplat signatures across versions.
    attempts = [
        dict(means=means, quats=quats, scales=scales, opacities=opacities, colors=sh_dc, viewmats=viewmats, Ks=Ks, width=width, height=height),
        dict(means=means, quats=quats, scales=scales, opacities=opacities, colors=sh_dc, viewmats=viewmats, Ks=Ks, W=width, H=height),
    ]
    last_err = None
    for kw in attempts:
        try:
            out = rasterization(**kw)
            if isinstance(out, tuple):
                rgb = out[0]
            elif isinstance(out, dict):
                rgb = out.get("render") or out.get("rgb")
            else:
                rgb = out
            if rgb is None:
                raise RuntimeError("rasterization returned no RGB output.")
            return rgb
        except TypeError as e:
            last_err = e
    raise RuntimeError(f"Could not call gsplat.rasterization with known signatures: {last_err}")


class Worker:
    def __init__(
        self,
        task_dir: str,
        results_dir: str = "results",
        iterations: int = 500,
        batch_size: int = 2,
        image_downscale: int = 2,
        lr: float = 1e-3,
        device: str = "cuda",
        basic_mode: bool = False,
        basic_max_points: int = 2000,
    ) -> None:
        self.task_dir = task_dir
        self.results_dir = results_dir
        self.iterations = iterations
        self.batch_size = batch_size
        self.image_downscale = image_downscale
        self.lr = lr
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.basic_mode = basic_mode
        self.basic_max_points = basic_max_points

    def run(self) -> str:
        if (not self.basic_mode) and self.device.type != "cuda":
            raise RuntimeError(
                "Worker requires CUDA for gsplat rasterization. "
                "Run inside your CUDA-enabled environment (e.g. WSL conda env) where torch.cuda.is_available() is True."
            )
        if (not self.basic_mode) and (rasterization is None or not _gsplat_cuda_backend_ready()):
            raise RuntimeError(
                "gsplat CUDA backend is unavailable in the active environment. "
                "This usually means gsplat was installed without a CUDA toolkit in WSL. "
                "Install CUDA toolkit in WSL, then reinstall gsplat in this conda env."
            )

        with open(os.path.join(self.task_dir, "task.json"), "r", encoding="utf-8") as f:
            task = json.load(f)
        with open(os.path.join(self.task_dir, "cameras.json"), "r", encoding="utf-8") as f:
            cameras = json.load(f)
        pts = np.load(os.path.join(self.task_dir, task["points_file"]))

        xyz = torch.from_numpy(pts["xyz"]).float().to(self.device)
        rgb = torch.from_numpy(pts["rgb"]).float().to(self.device)
        if self.basic_mode and xyz.shape[0] > self.basic_max_points:
            keep = torch.linspace(0, xyz.shape[0] - 1, steps=self.basic_max_points, device=self.device).long()
            xyz = xyz[keep]
            rgb = rgb[keep]
        n = xyz.shape[0]

        means = xyz.clone().requires_grad_(True)
        sh_dc = ((rgb - 0.5) / 0.28209479177387814).clone().requires_grad_(True)
        scales_log = (torch.ones((n, 3), device=self.device) * -3.0).requires_grad_(True)
        opacities_logit = torch.full((n, 1), -2.2, device=self.device, requires_grad=True)
        quats = torch.zeros((n, 4), device=self.device)
        quats[:, 0] = 1.0
        quats.requires_grad_(True)

        optim = torch.optim.Adam([means, sh_dc, scales_log, opacities_logit, quats], lr=self.lr)
        images_root = task["images_root"]

        if self.basic_mode:
            # Minimal local mode: skip heavy rasterization and produce a valid chunk PLY quickly.
            # Adds tiny jitter so output is not identical to raw COLMAP initialization.
            with torch.no_grad():
                means += 1e-4 * torch.randn_like(means)
            os.makedirs(self.results_dir, exist_ok=True)
            out_path = os.path.join(self.results_dir, f"chunk_{task['chunk_id']}.ply")
            write_gaussian_ply(
                out_path,
                means=means.detach().cpu().numpy(),
                sh_dc=sh_dc.detach().cpu().numpy(),
                opacities=torch.sigmoid(opacities_logit).detach().cpu().numpy(),
                scales=scales_log.detach().cpu().numpy(),
                quats=_normalize_quat(quats).detach().cpu().numpy(),
            )
            return out_path

        for step in range(1, self.iterations + 1):
            batch_indices = np.random.choice(len(cameras), size=min(self.batch_size, len(cameras)), replace=False)
            cam_batch = [cameras[int(i)] for i in batch_indices]

            gt_list = []
            for cam in cam_batch:
                image_path = os.path.join(images_root, cam["name"])
                if not os.path.exists(image_path):
                    continue
                gt_list.append(
                    _load_image(
                        image_path,
                        int(cam["width"]),
                        int(cam["height"]),
                        self.image_downscale,
                        self.device,
                    )
                )
            if not gt_list:
                raise FileNotFoundError(f"No camera images found under: {images_root}")
            cam_batch = cam_batch[: len(gt_list)]
            gt = torch.stack(gt_list, dim=0)

            viewmats, Ks, width, height = _build_camera_tensors(cam_batch, self.image_downscale, self.device)
            means_clamped = means
            scales = torch.exp(scales_log).clamp(min=1e-4, max=1.0)
            # gsplat expects opacities shape [N], not [N, 1]
            opacities = torch.sigmoid(opacities_logit).clamp(min=1e-4, max=0.999).squeeze(-1)
            quats_n = _normalize_quat(quats)

            pred = _run_rasterization(
                means=means_clamped,
                quats=quats_n,
                scales=scales,
                opacities=opacities,
                sh_dc=sh_dc,
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
            )
            pred = pred[..., :3].clamp(0.0, 1.0)
            if pred.shape[1] != gt.shape[1] or pred.shape[2] != gt.shape[2]:
                pred = F.interpolate(pred.permute(0, 3, 1, 2), size=(gt.shape[1], gt.shape[2]), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

            l1 = torch.mean(torch.abs(pred - gt))
            ssim = compute_ssim(pred, gt)
            loss = 0.8 * l1 + 0.2 * (1.0 - ssim)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if step % 50 == 0:
                print(f"[Worker] step={step} loss={loss.item():.6f} l1={l1.item():.6f} ssim={ssim.item():.6f}")

        os.makedirs(self.results_dir, exist_ok=True)
        out_path = os.path.join(self.results_dir, f"chunk_{task['chunk_id']}.ply")
        write_gaussian_ply(
            out_path,
            means=means.detach().cpu().numpy(),
            sh_dc=sh_dc.detach().cpu().numpy(),
            opacities=torch.sigmoid(opacities_logit).detach().cpu().numpy(),
            scales=scales_log.detach().cpu().numpy(),
            quats=_normalize_quat(quats).detach().cpu().numpy(),
        )
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Local worker for Gaussian chunk training.")
    parser.add_argument("--task_dir", type=str, default="tasks/chunk_0")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_downscale", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--basic_mode", action="store_true", help="Skip gsplat training and export lightweight initialized chunk output.")
    parser.add_argument("--basic_max_points", type=int, default=2000, help="Point cap used in --basic_mode to reduce memory.")
    args = parser.parse_args()

    worker = Worker(
        task_dir=args.task_dir,
        results_dir=args.results_dir,
        iterations=args.iterations,
        batch_size=args.batch_size,
        image_downscale=args.image_downscale,
        lr=args.lr,
        basic_mode=args.basic_mode,
        basic_max_points=args.basic_max_points,
    )
    out = worker.run()
    print(f"[Worker] wrote chunk result: {out}")


if __name__ == "__main__":
    main()
