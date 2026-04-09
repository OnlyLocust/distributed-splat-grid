# Distributed Gaussian Splat Rendering - Architecture

## High-Level Overview

This project implements a chunked Gaussian Splatting pipeline for COLMAP scenes. The core idea is:

1. Parse a COLMAP reconstruction (`cameras.bin`, `images.bin`, `points3D.bin`).
2. Split the global sparse point cloud into spatial chunks.
3. Train or initialize one Gaussian set per chunk.
4. Export each chunk to a Gaussian-compatible PLY.
5. Stitch all chunk PLY files into one final output.

The "distributed" aspect comes from chunk-level independence: each chunk can be processed by an isolated worker process (potentially on separate machines/GPUs). In the current repository, the provided shell pipeline runs chunk workers sequentially, but the file/task design supports parallelization by launching multiple `worker.py` jobs against different `tasks/chunk_*` directories.

---

## Directory Structure

Simplified view of critical project areas:

```text
distributed-gaussian-splat-rendering/
├─ main.py
├─ splitter.py
├─ worker.py
├─ stitcher.py
├─ colmap_io.py
├─ ply_io.py
├─ run_full_pipeline.sh
├─ setup.sh
├─ requirements.txt
├─ PYTHON_RUNBOOK.md
├─ data/
│  ├─ images/
│  └─ sparse/0/
│     ├─ cameras.bin
│     ├─ images.bin
│     └─ points3D.bin
├─ tasks/
│  └─ chunk_<id>/
│     ├─ task.json
│     ├─ cameras.json
│     └─ points.npz
└─ results/
   └─ chunk_<id>.ply
```

Role of each major directory:

- `data/`: Input dataset root. Must include COLMAP sparse reconstruction files and source RGB images.
- `tasks/`: Splitter output. Each `chunk_<id>` is a self-contained worker input package.
- `results/`: Worker output PLY files per chunk. Stitcher consumes everything ending in `.ply` from this directory.

---

## Core Components & Tech Stack

### Libraries and Frameworks

- **PyTorch** (`torch`, `torchvision`): tensor operations, optimization loop, CUDA device execution.
- **gsplat**: differentiable Gaussian rasterization backend used in full training mode.
- **CUDA**: required for full gsplat training path (`worker.py` enforces CUDA availability).
- **NumPy**: serialization, geometry arrays, chunk partitioning.
- **Pillow**: image loading and resizing for supervision frames.
- **Open3D** (auxiliary): validation and visualization scripts (`validate_colmap.py`, `smoke_test.py`).

### Initialization Points

- Dependency/version declarations: `requirements.txt`.
- Runtime checks:
  - `worker.py` validates CUDA availability for non-basic mode.
  - `worker.py` checks gsplat CUDA backend readiness.
- Data model initialization in `worker.py`:
  - `means` from COLMAP xyz.
  - `sh_dc` initialized from RGB (DC SH coefficients).
  - `scales_log`, `opacities_logit`, `quats` initialized as trainable Gaussian parameters.

---

## System Architecture

### Main Entry Points

- `main.py`: lightweight coordinator pipeline (`Splitter -> Worker -> Stitcher`) for local single-worker simulation.
- `run_full_pipeline.sh`: practical orchestration script for all chunks (`splitter.py`, then looped `worker.py`, then `stitcher.py`).
- `splitter.py`: task generation from COLMAP scene.
- `worker.py`: per-chunk training/export execution unit.
- `stitcher.py`: global merge of per-chunk PLY files.

### Coordinator vs Worker

`main.py` acts as a manager, but with an important constraint: it only runs the first chunk returned by the splitter. This is useful for sanity checks and development iteration, not full multi-chunk production.

`worker.py` is the true chunk compute engine. Given one `tasks/chunk_<id>`, it:

1. Loads task metadata and camera/point payloads.
2. Builds trainable Gaussian tensors.
3. Runs either:
   - **basic mode**: lightweight jitter + export, or
   - **full mode**: iterative gsplat render supervision with L1 + SSIM loss.
4. Writes `results/chunk_<id>.ply`.

---

## Data Flow / Execution Pipeline

Typical end-to-end run:

1. **Ingestion (`splitter.py`)**
   - `load_colmap_scene()` reads COLMAP binaries from `data/sparse/0`.
   - Intrinsics are normalized into camera records via `intrinsics_from_camera()`.

2. **Chunking (`splitter.py`)**
   - Global point cloud is partitioned along X (`grid_x x 1 x 1`).
   - For each chunk:
     - Save points to `points.npz`.
     - Save camera metadata to `cameras.json`.
     - Save task metadata to `task.json`.

3. **Chunk Processing (`worker.py`)**
   - Load one chunk package.
   - Initialize Gaussian parameters.
   - **Full mode path**:
     - Sample camera batches.
     - Load corresponding images from `task["images_root"]`.
     - Build view and intrinsic matrices.
     - Rasterize via `gsplat.rasterization(...)`.
     - Compute `0.8 * L1 + 0.2 * (1 - SSIM)`.
     - Backprop + Adam update.
   - Export chunk Gaussian PLY.

4. **Global Merge (`stitcher.py`)**
   - Enumerate all `.ply` files in `results/`.
   - Read each with `read_gaussian_ply()`.
   - Concatenate arrays (`means`, `sh_dc`, `opacities`, `scales`, `quats`).
   - Write final stitched PLY.

5. **Operational Wrapper (`run_full_pipeline.sh`)**
   - Stage 1: split.
   - Stage 2: iterate chunk IDs and run workers.
   - Stage 3: stitch to `output.ply`.

---

## State Management & Concurrency

### Task Distribution

- Distribution unit = one `tasks/chunk_<id>` directory.
- Point ownership is deterministic from X-range binning in the splitter.
- Each point index is assigned to exactly one chunk.
- Camera records are currently duplicated to every chunk for simplicity.

### Parallel Execution Model

- The provided shell script processes chunks serially (`for i in ...; worker.py`).
- The architecture is still parallelizable because chunk inputs/outputs are isolated:
  - Input isolation: `tasks/chunk_<id>/...`
  - Output isolation: `results/chunk_<id>.ply`
- Full mode uses GPU-parallel operations internally through PyTorch + gsplat CUDA kernels.

### Collision and Consistency Controls

- File-level isolation by chunk ID prevents output overwrites between different chunks.
- Split boundary handling avoids duplicate point assignment across neighboring chunks.
- `stitcher.py` merges all `.ply` files present in `results/`; this means stale files from prior runs can contaminate outputs if not cleaned.

Recommended operational hygiene before a new full run:

- Clear or version the `results/` directory.
- Keep `tasks/` tied to a specific `grid_x` and source scene snapshot.

---

## Notes for Future Evolution

- Replace sequential shell loop with a scheduler/queue for true multi-node execution.
- Add run manifests (job ID, chunk count, timestamp, dataset hash) to prevent mixing old/new outputs.
- Consider camera culling per chunk to reduce unnecessary image supervision work.
- Add a strict "expected chunk count" check before stitching.
