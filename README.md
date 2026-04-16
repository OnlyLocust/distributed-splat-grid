# Splat-Grid: Distributed 3D Gaussian Splatting Engine

**Splat-Grid** distributes the 3D Gaussian Splatting (3DGS) training process across multiple
consumer laptops to prevent Out-of-Memory (OOM) errors and speed up large-scene reconstruction.
It partitions the 3-D bounding box of a COLMAP dataset into voxel chunks and assigns each
chunk to a worker node over a LAN.

```
┌─────────────────────────────────────┐        LAN
│          MASTER NODE                │◄──────────────┐
│  • Reads COLMAP data                │               │
│  • Partitions scene into voxels     │  GET /task    │
│  • Serves images & COLMAP files     │──────────────►│
│  • Receives finished .ply chunks    │  POST /result │
│  • Stitches final output PLY        │◄──────────────│
└─────────────────────────────────────┘       │       │
                                         WORKER 1   WORKER 2 …
```

---

## Repository Layout

```
gaussian-splatting/
├── master.py              ← Master node (FastAPI server)
├── worker.py              ← Worker node (HTTP client + trainer)
├── splat_grid/
│   ├── colmap_utils.py    ← Shared COLMAP binary parsers
│   └── ply_utils.py       ← PLY writer + multi-file stitcher
├── setup.sh               ← Dependency setup script (Linux/WSL2)
├── validate_colmap.py     ← Utility: verify a COLMAP dataset
└── results/               ← Created at runtime by master
```

---

## Hardware Requirements

| Node   | GPU VRAM | Notes |
|--------|----------|-------|
| Master | None required | CPU-only is fine; master only parses data and serves files |
| Worker | ≥ 4 GB VRAM | Tested on RTX 3050 (4 GB). Memory-safe constraints applied by default |

---

## Dependencies

### Master Node

```
python >= 3.10
fastapi
uvicorn[standard]
numpy
Pillow
```

### Worker Node

```
python >= 3.10
torch >= 2.0   (CUDA build matching your driver)
gsplat         (pip install gsplat)
requests
numpy
Pillow
pytorch-msssim  (optional — enables SSIM loss; falls back to L1 without it)
```

> **Note:** The worker does **not** need the compiled `diff-gaussian-rasterization` submodule.
> It uses the `gsplat` package (`gsplat.rasterization`) for rendering.

### Quick install

```bash
# On the MASTER node
pip install fastapi "uvicorn[standard]" numpy Pillow

# On each WORKER node (example for CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install gsplat requests numpy Pillow pytorch-msssim
```

Or use the provided `setup.sh`:

```bash
bash setup.sh master   # installs master dependencies
bash setup.sh worker   # installs worker dependencies
```

---

## Dataset Format (COLMAP)

Your dataset must follow the standard COLMAP layout:

```
<data_dir>/
├── images/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

To prepare your own images, use COLMAP or the included `convert.py` helper.

---

## Running the Distributed Pipeline

### Step 1 — Start the Master Node

On the machine that holds the dataset:

```bash
python master.py \
  --data_dir /path/to/colmap_dataset \
  --output   stitched_output.ply \
  --grid     2 \
  --host     0.0.0.0 \
  --port     8000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | *(required)* | Root of the COLMAP dataset |
| `--output` | `stitched_output.ply` | Path for the merged output PLY |
| `--grid` | `2` | Divides the scene into `N×N×N` voxels. `--grid 2` → 8 tasks, `--grid 3` → up to 27 tasks |
| `--host` | `0.0.0.0` | Bind address (use `0.0.0.0` to accept LAN connections) |
| `--port` | `8000` | TCP port |

Check the master is ready:
```bash
curl http://localhost:8000/status
# {"ready":true,"total":8,"pending":8,"in_flight":0,"completed":0,"stitched":false}
```

### Step 2 — Start Worker Nodes

On **each** worker laptop (can be on the same machine for testing):

```bash
python worker.py \
  --master http://<MASTER_IP>:8000 \
  --iterations 750 \
  --downscale  4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--master` | *(required)* | Full URL of the master node |
| `--work_dir` | *(system temp)* | Local directory for downloaded data and temp PLYs |
| `--iterations` | `750` | Training iterations per chunk (500–1000 recommended for 4 GB VRAM) |
| `--downscale` | `4` | Image resolution divisor (4 → ¼ resolution, ~16× fewer pixels) |
| `--max_tasks` | `0` | Max tasks before exiting. `0` = run until queue empty |
| `--retry_delay` | `5.0` | Seconds between retries when master is initialising |

Workers will:
1. Download only the COLMAP data and images for their assigned voxel
2. Train a Gaussian model on that spatial chunk
3. Upload the `.ply` result back to the master
4. Request the next task, repeating until the queue is empty

### Step 3 — Collect the Output

Once all tasks are complete, the master automatically stitches the `.ply` chunks and writes:

```
stitched_output.ply
```

This file is a standard 3DGS-compatible PLY viewable with
[SuperSplat](https://playcanvas.com/supersplat/editor),
[Polycam](https://poly.cam), or the SIBR viewer.

---

## Memory-Safety Constraints (Worker)

These defaults are tuned for **4 GB VRAM (RTX 3050)**. Change only if you have more VRAM:

| Parameter | Default | Why |
|-----------|---------|-----|
| `--downscale 4` | 4× | Reduces pixel count by 16×, biggest single VRAM saving |
| `--iterations 750` | 750 | Bounded training time per chunk; full quality needs ~30 000 globally |
| SH degree | 0 (hardcoded) | 1 coeff/channel instead of 16; reduces Gaussian tensor size 16× |
| Max Gaussians | 50 000/chunk | Prevents unbounded growth from densification |
| `torch.cuda.empty_cache()` | Every 50 steps + after densify | Releases PyTorch cache fragments back to CUDA |

---

## Validating a COLMAP Dataset

```bash
python validate_colmap.py --data_dir /path/to/colmap_dataset
```

Checks that cameras.bin / images.bin / points3D.bin are present and meet minimum thresholds
(≥ 40 cameras, ≥ 500 sparse points).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Worker: `gsplat not installed` | `pip install gsplat` |
| Worker: CUDA out-of-memory | Increase `--downscale` (try `8`) and/or reduce `--iterations` (try `500`) |
| Worker: `Cannot connect to master` | Check firewall — port 8000 must be open on the master. Use `--host 0.0.0.0` on master |
| Master: `cameras.bin not found` | Pass the dataset root to `--data_dir`, not the `sparse/0/` subdirectory |
| PLY looks empty | The voxel may have had no sparse points. Try `--grid 1` (single task) to verify the dataset loads correctly |
| Workers finish but no stitched file | Wait a few seconds — stitching runs in a background thread. Check master log output |
