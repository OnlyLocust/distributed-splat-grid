# 3D Gaussian Splatting — Data Flow

This document summarizes the data flow used by the `gaussian-splatting` codebase (for a presentation).

## Overview

- **Inputs:** COLMAP SfM outputs + calibrated images (poses, intrinsics), optional depth priors and exposure info.
- **Initialization:** COLMAP point cloud → `GaussianModel.create_from_pcd()` → per-point Gaussians (position, SH-based color features, scales, rotations, opacity).
- **Core loop:** For each training iteration: select camera → render via `gaussian_renderer.render()` → rasterize/composite → compute loss → backprop → optimizer step → (periodic) densify/prune.
- **Outputs:** Checkpointed Gaussian model (PLY / model folder) for real-time viewing or offline rendering.

## Step-by-step data flow

1) Dataset & cameras

  - Camera intrinsics/extrinsics and images are loaded from the COLMAP dataset. The camera object (used by the renderer) provides `world_view_transform`, `full_proj_transform`, FoV and image dimensions.

2) Gaussian model representation

  - Each Gaussian stores:
    - 3D center `xyz` (optimizable tensor)
    - SH color features (`features_dc`, `features_rest`) or precomputed RGB
    - anisotropic scale (`scaling`) and rotation (`rotation`) → 3D covariance
    - `opacity` and per-image exposure parameters
    - bookkeeping (e.g., `max_radii2D`, gradient accumulators)
  - Created via `GaussianModel.create_from_pcd()` or `load_ply()` (see `scene/gaussian_model.py`).

3) Rendering (forward pass)

  - `gaussian_renderer.render(viewpoint_camera, pc, pipe, bg_color, ...)`:
    - Builds `GaussianRasterizationSettings` (image dims, tanFOV, transforms, SH degree, etc.).
    - Prepares inputs: `means3D` (pc.get_xyz), `opacities`, either (`scales` + `rotations`) or `cov3D_precomp`, and either SH coefficients or precomputed RGB.
    - Calls `GaussianRasterizer` (from submodule `diff-gaussian-rasterization`) which:
      - Projects anisotropic Gaussians to screen space → computes per-Gaussian 2D radii, depth, screen positions.
      - Performs visibility-aware compositing (anisotropic EWA / splatting), optionally converts SH→RGB on GPU.
      - Returns `rendered_image`, `radii`, `depth_image`.
    - `render()` applies exposure compensation if requested and returns the render plus `visibility_filter` and `viewspace_points` used for gradient bookkeeping.

4) Loss & backprop

  - Loss compares `rendered_image` to ground-truth (RGB L2/L1, optional depth priors, LPIPS, regularizers).
  - Gradients flow back through the rasterizer into Gaussian parameters: positions, SH features (or colors), opacity, scaling, rotation, and exposure.
  - Optimizer updates parameters. Supported optimizers: standard `Adam` and `SparseGaussianAdam` (when available).

5) Adaptive topology (densify / split / prune)

  - Per-Gaussian gradient statistics are accumulated (via `add_densification_stats()` using viewspace gradients).
  - Periodically `densify_and_prune()` runs:
    - Clone or split Gaussians where gradients are high (increase local density/detail).
    - Prune Gaussians with low opacity or low importance.
    - When topology changes, optimizer parameter groups are updated (`cat_tensors_to_optimizer`, `_prune_optimizer`) so training continues seamlessly.

6) Checkpoint & view

  - Save model snapshots as PLY / model folders (positions, features, scales, rotations, opacity, exposures).
  - View with the SIBR viewer (OpenGL viewer) or run `render.py` for final renderings.

## Implementation details / notes

- SH vs precomputed color: SH→RGB conversion can be done in Python (`pipe.convert_SHs_python`) or inside the rasterizer (faster GPU path).
- Covariance: either computed from `scaling` + `rotation` in rasterizer or precomputed in Python and passed as `cov3D_precomp`.
- Exposure: per-image 3x4 affine transforms can be optimized during training and applied in `render()` when `use_trained_exp` is enabled.
- Rasterizer: the differentiable, visibility-aware anisotropic Gaussian rasterizer is provided by the `diff-gaussian-rasterization` submodule.

## One-slide summary (speaker notes ready)

- Input: images + COLMAP → initialize Gaussians (`xyz`, SH color, `scaling`, `rotation`, `opacity`).
- Loop: pick camera → rasterize Gaussians → composite → compute loss → backprop → update Gaussians.
- Adapt topology: accumulate gradients → split/clone/prune Gaussians → continue training.
- Output: compact, real-time renderable Gaussian model → view with SIBR viewer or `render.py`.

## Diagram (Mermaid)

```mermaid
flowchart LR
  A[Input images + COLMAP SfM outputs] --> B[Init point cloud]
  B --> C[GaussianModel init\n(`xyz`, SH features, scale, rotation, opacity)]
  C --> D[Training loop (iter)]
  D --> E[Render(viewpoint_camera, GaussianModel, pipe)]
  E --> F[GaussianRasterizer\n(project Gaussians → screen, compute radii & depth)]
  F --> G[Visibility-aware compositing\n(EWA / anisotropic splatting)]
  G --> H[Rendered image, depth, radii]
  H --> I[Loss computation\n(RGB, depth priors, LPIPS, regs)]
  I --> J[Backprop through rasterizer\n(gradients to `xyz`, features, scale, rot, opacity, exposure)]
  J --> K[Optimizer step\n(`Adam` or `SparseGaussianAdam`)]
  K --> C
  D --> L[Densify / Split / Prune\n(accumulate gradients → decide topology changes)]
  L --> C
  K --> M[Checkpoint / Export (PLY, model dir)]
  M --> N[Viewer / Real-time renderer (SIBR or GL viewer)]
  subgraph optional
    O1[Precompute cov3D → `cov3D_precomp`] --> F
    O2[SH→RGB conv (Python or rasterizer)] --> F
    O3[Per-image exposure params] --> G
  end
  style A fill:#f8f9fa,stroke:#333,stroke-width:1px
  style C fill:#e6f7ff,stroke:#333,stroke-width:1px
  style F fill:#fff1b8,stroke:#333,stroke-width:1px
  style L fill:#ffd6e7,stroke:#333,stroke-width:1px
```

---

If you want, I can export this diagram as a PNG slide and add a short speaker-notes section per bullet.
