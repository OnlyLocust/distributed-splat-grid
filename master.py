"""
master.py — Splat-Grid Master Node
===================================
Hosts the FastAPI server that:
  1. Reads COLMAP data at startup and partitions the 3-D bounding box into a
     grid of voxel tasks.
  2. Hands tasks to Worker nodes on demand  (GET /task).
  3. Serves raw data files so workers can download what they need
     (GET /data/images/<name>, GET /data/colmap/<name>).
  4. Receives finished .ply chunks from workers (POST /result/<task_id>).
  5. Stitches all chunks into a single stitched_output.ply when complete.

Usage
-----
  python master.py --data_dir /path/to/colmap_dataset \\
                   --output   stitched_output.ply    \\
                   --grid     2                      \\
                   --host     0.0.0.0                \\
                   --port     8000

Then on each worker machine:
  python worker.py --master http://<MASTER_IP>:8000
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from splat_grid.colmap_utils import load_colmap_data
from splat_grid.ply_utils import stitch_ply_files

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------

app = FastAPI(title="Splat-Grid Master", version="2.0.0")

_data_dir:       str              = ""
_output_path:    str              = "stitched_output.ply"
_results_dir:    str              = "results"

_task_queue:     deque            = deque()          # task_id strings yet to be assigned
_all_tasks:      dict             = {}               # task_id → task_dict
_assigned:       set              = set()            # task_ids given to workers
_completed:      dict             = {}               # task_id → local .ply path
_lock:           threading.Lock  = threading.Lock()
_ready:          bool             = False            # True once startup finishes


# ---------------------------------------------------------------------------
# CLI argument parsing  (called from __main__)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Splat-Grid Master Node")
    p.add_argument("--data_dir", required=True,
                   help="Root of COLMAP dataset (contains images/ and sparse/0/).")
    p.add_argument("--output",   default="stitched_output.ply",
                   help="Path for the final merged PLY (default: stitched_output.ply).")
    p.add_argument("--grid",     type=int, default=2,
                   help="Grid resolution N: scene divided into N×N×N voxels (default: 2).")
    p.add_argument("--host",     default="0.0.0.0",
                   help="Host to bind the API server (default: 0.0.0.0).")
    p.add_argument("--port",     type=int, default=8000,
                   help="Port for the API server (default: 8000).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scene partitioning
# ---------------------------------------------------------------------------

def _compute_global_bbox(points3D: dict) -> tuple:
    """Return (bbox_min, bbox_max) as float32 numpy arrays [3]."""
    if not points3D:
        log.warning("No sparse points — using default bbox [-5, -5, -5] → [5, 5, 5].")
        return np.array([-5.0, -5.0, -5.0], np.float32), \
               np.array([ 5.0,  5.0,  5.0], np.float32)

    xyzs    = np.array([pt["xyz"] for pt in points3D.values()], dtype=np.float32)
    bbox_min = xyzs.min(axis=0)
    bbox_max = xyzs.max(axis=0)

    # Add a small margin so boundary points fall cleanly inside
    margin   = (bbox_max - bbox_min) * 0.02 + 1e-4
    return bbox_min - margin, bbox_max + margin


def _partition_scene(
    cameras: dict,
    images:  dict,
    points3D: Optional[dict],
    grid_n:  int,
) -> dict:
    """
    Partition the scene bounding box into grid_n^3 voxels.

    Each voxel that contains at least one sparse point becomes a task.
    All image names are assigned to every task — workers filter Gaussians to
    their voxel bbox during training, so every camera is a valid supervisor.

    Returns
    -------
    tasks : {task_id: {task_id, bbox_min, bbox_max, image_names, n_points}}
    """
    bbox_min, bbox_max = _compute_global_bbox(points3D)
    cell_size = (bbox_max - bbox_min) / grid_n

    log.info(f"Global bbox: {bbox_min} → {bbox_max}")
    log.info(f"Partition:   {grid_n}³ grid  ({grid_n**3} max cells)")

    # Bin each sparse point into its voxel
    if points3D:
        xyzs       = np.array([pt["xyz"] for pt in points3D.values()], dtype=np.float32)
        indices    = np.floor((xyzs - bbox_min) / cell_size).astype(int)
        indices    = np.clip(indices, 0, grid_n - 1)
        voxel_counts: dict = {}
        for idx in map(tuple, indices):
            voxel_counts[idx] = voxel_counts.get(idx, 0) + 1
    else:
        # No sparse points — create a single task covering the full scene
        voxel_counts = {(0, 0, 0): 0}
        grid_n = 1

    # Collect all image filenames
    all_image_names = sorted({meta["name"] for meta in images.values()})

    tasks = {}
    for (ix, iy, iz), n_pts in voxel_counts.items():
        v_min = bbox_min + np.array([ix, iy, iz], np.float32) * cell_size
        v_max = v_min + cell_size
        tid   = f"voxel_{ix}_{iy}_{iz}"
        tasks[tid] = {
            "task_id":     tid,
            "bbox_min":    [float(x) for x in v_min],
            "bbox_max":    [float(x) for x in v_max],
            "image_names": all_image_names,
            "n_points":    int(n_pts),
            "status":      "pending",
        }

    log.info(f"Tasks created: {len(tasks)}")
    return tasks


# ---------------------------------------------------------------------------
# Startup  (called once before serving requests)
# ---------------------------------------------------------------------------

def build_task_queue(data_dir: str, grid_n: int, output_path: str, results_dir: str):
    global _data_dir, _output_path, _results_dir, _ready

    _data_dir    = data_dir
    _output_path = output_path
    _results_dir = results_dir

    os.makedirs(results_dir, exist_ok=True)

    log.info("=== Splat-Grid Master: loading COLMAP data ===")
    cameras, images, points3D = load_colmap_data(data_dir)

    tasks = _partition_scene(cameras, images, points3D, grid_n)

    with _lock:
        _all_tasks.update(tasks)
        for tid in tasks:
            _task_queue.append(tid)

    _ready = True
    log.info(f"=== Ready — {len(_task_queue)} tasks queued ===")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
def status():
    """Return a progress summary."""
    with _lock:
        total     = len(_all_tasks)
        done      = len(_completed)
        pending   = len(_task_queue)
        in_flight = len(_assigned) - done  # assigned but not yet complete
    return {
        "ready":     _ready,
        "total":     total,
        "pending":   pending,
        "in_flight": max(in_flight, 0),
        "completed": done,
        "stitched":  os.path.exists(_output_path),
    }


@app.get("/task")
def get_task():
    """
    Pop and return the next available task.

    HTTP 200  — task JSON payload.
    HTTP 204  — queue empty (all tasks assigned or done).
    HTTP 503  — master still initialising.
    """
    log.info("[Task] Worker requested a task.")
    
    if not _ready:
        log.warning("[Task] Master is still initialising. Returning 503.")
        raise HTTPException(status_code=503, detail="Master is still initialising.")

    with _lock:
        if not _task_queue:
            log.info("[Task] Queue is empty. Returning 204.")
            return JSONResponse(status_code=204, content=None)

        tid  = _task_queue.popleft()
        task = dict(_all_tasks[tid])          # shallow copy
        task["status"] = "assigned"
        _all_tasks[tid]["status"] = "assigned"
        _assigned.add(tid)

    log.info(f"[Task] Assigned '{tid}' ({len(task['image_names'])} images)")
    return JSONResponse(content=task)


@app.get("/data/colmap/{filename}")
def serve_colmap_file(filename: str):
    """Serve cameras.bin / images.bin / points3D.bin to workers."""
    allowed = {"cameras.bin", "images.bin", "points3D.bin"}
    if filename not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown COLMAP file: {filename}")

    path = os.path.join(_data_dir, "sparse", "0", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{filename} not found on master.")

    return FileResponse(path, media_type="application/octet-stream", filename=filename)


@app.get("/data/images/{image_name:path}")
def serve_image(image_name: str):
    """Serve an individual training image to a worker."""
    # Sanitise path traversal
    safe_name = Path(image_name).name
    fpath     = os.path.join(_data_dir, "images", safe_name)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Image not found: {safe_name}")

    return FileResponse(fpath)


@app.post("/result/{task_id}")
async def receive_result(task_id: str, request: Request):
    """
    Accept a finished .ply chunk from a worker.
    Triggers stitching once all tasks are complete.
    """
    log.info(f"[Result] Start receiving result for task: {task_id}")
    
    if task_id not in _all_tasks:
        log.error(f"[Result] Unknown task_id: {task_id}")
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

    # Save the uploaded PLY directly from the stream
    dest = os.path.join(_results_dir, f"{task_id}.ply")
    bytes_received = 0
    with open(dest, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            bytes_received += len(chunk)

    with _lock:
        _completed[task_id] = dest
        _all_tasks[task_id]["status"] = "completed"
        n_done  = len(_completed)
        n_total = len(_all_tasks)

    log.info(f"[Result] Received '{task_id}' ({bytes_received:,} bytes) — {n_done}/{n_total} complete.")

    # Trigger stitching when all chunks are in
    if n_done == n_total:
        log.info("[Stitch] All chunks received — stitching final PLY...")
        threading.Thread(target=_stitch, daemon=True).start()

    return JSONResponse(content={"status": "ok", "task_id": task_id, "completed": n_done, "total": n_total})


def _stitch():
    """Background thread: merge all received PLY chunks into one file."""
    with _lock:
        ply_paths = list(_completed.values())

    try:
        total = stitch_ply_files(ply_paths, _output_path)
        log.info(f"[Stitch] ✓ {total:,} Gaussians written → {_output_path}")
    except Exception as exc:
        log.error(f"[Stitch] FAILED: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Run startup in a background thread so uvicorn can begin accepting
    # connections immediately (workers will get 503 until ready=True).
    threading.Thread(
        target=build_task_queue,
        args=(args.data_dir, args.grid, args.output, "results"),
        daemon=True,
    ).start()

    log.info(f"Starting Splat-Grid Master on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
