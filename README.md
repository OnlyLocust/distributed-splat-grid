# Splat-Grid: Distributed 3D Gaussian Splatting

## 1. Project Overview
Splat-Grid is a fault-tolerant, distributed 3D Gaussian Splatting engine designed to bypass single-GPU memory limits. It partitions COLMAP 3D scene data into voxel-based chunks across multiple consumer laptops (Workers), managed by a central Orchestrator (Master). This approach enables the rendering of large-scale radiance fields without requiring high-end GPUs with massive VRAM, by utilizing the collective power of a local network swarm.

## 2. Prerequisites
To run the Master or Worker nodes, ensure your system meets the following requirements:
- **Operating System:** Windows or Linux (WSL2 supported)
- **Environment Management:** Conda (recommended) or `venv`
- **Hardware:** NVIDIA GPU with CUDA support (RTX 3050 or better recommended for Workers)
- **Core Software:**
  - Python 3.8+
  - PyTorch (compiled with CUDA support to match your system)
  - `gsplat` (for rasterization)
  - COLMAP dataset (undistorted `images/` and `sparse/0/` directory containing `.bin` files)

## 3. Installation & Setup
Follow these steps on every machine participating in the swarm (Master and all Workers):

1. **Clone the repository:**
   ```bash
   git clone https://github.com/graphdeco-inria/gaussian-splatting.git
   cd gaussian-splatting
   ```

2. **Run setup scripts or create Conda environment:**
   ```bash
   # Run the provided setup script
   bash setup.sh
   
   # OR create a clean Conda environment manually:
   conda create -n splatgrid python=3.10 -y
   conda activate splatgrid
   ```

3. **Install the dependencies:**
   Make sure you install PyTorch with the correct CUDA version for your system before running the requirements installation.
   ```bash
   # Adjust CUDA version as needed (e.g., cu118, cu121)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements_worker.txt
   ```

## 4. How to Run the Master Node
The Master Node sits at the center of the architecture. It calculates the scene's 3D bounding box from COLMAP data, creates the task assignments, and orchestrates the distributed stitching.

To start the Master server, run:
```bash
python master.py --data_dir /path/to/colmap/data --host 0.0.0.0 --port 8765 --grid 2x2x2
```
- `--data_dir`: Must point to the root of your COLMAP output (the folder containing the `images/` and `sparse/` subdirectories).
- `--host` / `--port`: Determines where the Master listens for Worker connections (use `0.0.0.0` to allow external connections).
- `--grid`: Adjusts how many chunks the bounding box is split into (e.g., `2x2x2` = 8 chunks).

*Note:* When launched, the Master initializes a persistent local database (`splat_state.db` using SQLite) to track task assignments and handle fault tolerance across the network.

## 5. How to Run a Worker Node (Joining the Swarm)
Worker Nodes act as the computational workhorses. A teammate only needs a fresh laptop and to run the Worker script to immediately join the active swarm.

To start a Worker and connect it to the Master, run:
```bash
python worker.py --master http://<MASTER_IP>:8765 --iterations 700
```
- `--master`: The full URL pointing to the Master Node's IP address and Port (e.g., `http://192.168.1.100:8765`).
- `--iterations`: (Optional) Specifies how many training steps to perform per chunk (default is 700).

**Memory-Safe Constraints (Enforced):**
To ensure standard consumer laptops do not crash from Out-Of-Memory (OOM) errors during heavy training chunks, Workers strictly enforce several constraints:
- **Downscaling:** Scene images requested from the Master are safely downscaled to 1/4 resolution by default.
- **Spherical Harmonics:** SH degree is forced to 0 (DC-only color) reducing memory footprints dramatically.
- **Max Iterations & Gaussians:** Hard caps on iterations and max Gaussians (e.g., 50,000) are enforced.
- **Cache Clearing:** Memory is explicitly cleared and released periodically (`torch.cuda.empty_cache()`).

## 6. System Architecture (High-Level)
Splat-Grid uses a stateless-worker paradigm to guarantee fault tolerance and high availability across varying hardware:

1. **Master Initialization:** The Master reads the `.bin` data to calculate the dataset's global 3D bounding box, splits it globally into voxel chunks according to the grid spec, and populates the SQLite `tasks` table with `PENDING` chunks.
2. **Worker Request:** A Worker connects (`GET /join`) to obtain a unique ID and requests work (`GET /get_task`). The Master checks the queue and assigns a `PENDING` chunk, changing its status to `IN_PROGRESS`.
3. **Training & Heartbeats:** The Worker downloads the required partial image set and `.bin` data, then initiates a local `gsplat` training loop on the dedicated chunk. Concurrently, it sends a `POST /heartbeat` every 30 seconds to the Master.
   - *Fault Tolerance:* If the Master watchdog loop detects no heartbeat has been received from an `IN_PROGRESS` task for 5 minutes, it assumes the Worker crashed and reverts the assigned chunk back to `PENDING` for another node to take over.
4. **Result Upload & Stitching:** Upon completing a chunk, the Worker uploads the computed `.ply` to `POST /submit_result/{id}`. The Master stores it and updates the specific task to `COMPLETED`. When all chunks hit `COMPLETED`, the Master stitches them all securely into the final `.ply` output geometry.
