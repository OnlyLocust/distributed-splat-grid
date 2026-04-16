"""
Splat-Grid — Master Node (v3 — SQLite + Fault Tolerant)
========================================================
Run this on the machine that holds the COLMAP dataset.

Usage:
    python master.py --data_dir ./data --host 0.0.0.0 --port 8765 --grid 2x2x2

Endpoints
---------
  GET  /join                   Register a new worker → {worker_id}
  GET  /get_task               Assign a PENDING task → task descriptor JSON
  POST /heartbeat              Worker keeps task alive → {status: "ok"}
  POST /submit_result/{id}     Upload finished .ply chunk
  GET  /image/{image_name}     Serve a raw image file to a worker
  GET  /sparse/{filename}      Serve cameras/images/points3D .bin files
  GET  /status                 Overall progress summary
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.responses import FileResponse, JSONResponse

# ── Logging ───────────────────────────────────────────────────────────────────

_log_handler = logging.FileHandler("master.log", encoding="utf-8")
_log_handler.setFormatter(logging.Formatter(
    "%(asctime)s [MASTER][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [MASTER][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logging.basicConfig(level=logging.INFO, handlers=[_log_handler, _console_handler])
log = logging.getLogger(__name__)

# ── COLMAP helpers ────────────────────────────────────────────────────────────
from colmap_utils import load_colmap_data, compute_global_bbox

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Splat-Grid Master", version="3.0.0")

# ── Configuration (set at startup) ───────────────────────────────────────────
_DB_PATH:       str  = "splat_state.db"
_DATA_DIR:      str  = ""
_OUTPUT_DIR:    Path = Path("results")
_NUM_TASKS:     int  = 0
_CAMERAS:       dict = {}
_IMAGE_METAS:   dict = {}
_STITCH_DONE:   bool = False

# Heartbeat timeout — tasks silent for this many seconds revert to PENDING
HEARTBEAT_TIMEOUT_S: int = 300   # 5 minutes

# ── Database helpers ──────────────────────────────────────────────────────────

@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    """Yield a WAL-mode SQLite connection; always commit-or-rollback."""
    conn = sqlite3.connect(_DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> float:
    return time.time()


# ── Startup: partition COLMAP data into voxel grid tasks ─────────────────────

def _parse_grid(spec: str) -> tuple[int, int, int]:
    """Parse '2x2x2' or '2' → (gx, gy, gz)."""
    parts = spec.lower().split("x")
    if len(parts) == 1:
        g = int(parts[0])
        return g, g, g
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    raise ValueError(f"grid spec must be 'N' or 'NxNxN', got {spec!r}")


def _camera_center(meta: dict) -> np.ndarray:
    """World-space camera centre: C = -R^T @ t."""
    return -(meta["R"].T @ meta["t"])


def _images_in_voxel(
    image_metas: dict,
    vmin: np.ndarray,
    vmax: np.ndarray,
    margin_factor: float = 1.50,
) -> list[int]:
    """
    Return image IDs whose camera centres fall within voxel + margin.
    margin_factor=1.50 expands each side by 150% of the voxel dimension,
    providing wide overlap so cameras near voxel edges are always captured.
    Fallback: if still zero matches, assign every image — an empty voxel
    is always worse than one that trains on all available views.
    """
    margin = (vmax - vmin) * margin_factor
    lo, hi = vmin - margin, vmax + margin
    matched = [
        img_id for img_id, meta in image_metas.items()
        if np.all(_camera_center(meta) >= lo) and np.all(_camera_center(meta) <= hi)
    ]
    if not matched:
        log.warning(
            f"  [_images_in_voxel] No cameras in expanded voxel "
            f"(margin_factor={margin_factor}) — assigning ALL {len(image_metas)} images as fallback."
        )
        matched = list(image_metas.keys())
    return matched


def init_master(data_dir: str, grid_spec: str, output_dir: str) -> None:
    """Load COLMAP, compute bbox, splits into voxels, populate DB."""
    global _DATA_DIR, _OUTPUT_DIR, _NUM_TASKS, _CAMERAS, _IMAGE_METAS

    _DATA_DIR   = data_dir
    _OUTPUT_DIR = Path(output_dir)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading COLMAP data from: {data_dir}")
    cameras, image_metas, points3D = load_colmap_data(data_dir)
    _CAMERAS     = cameras
    _IMAGE_METAS = image_metas

    bbox_min, bbox_max = compute_global_bbox(points3D, image_metas)
    log.info(f"Global bbox — min={bbox_min.tolist()}  max={bbox_max.tolist()}")

    gx, gy, gz = _parse_grid(grid_spec)
    log.info(f"Partitioning into {gx}×{gy}×{gz} = {gx*gy*gz} voxels")

    step = (bbox_max - bbox_min) / np.array([gx, gy, gz], dtype=np.float32)
    now  = _now()

    with _db() as conn:
        # Clear any stale tasks from a previous run
        conn.execute("DELETE FROM tasks;")

        count = 0
        for ix in range(gx):
            for iy in range(gy):
                for iz in range(gz):
                    vmin = bbox_min + step * np.array([ix, iy, iz], dtype=np.float32)
                    vmax = vmin + step
                    voxel_img_ids = _images_in_voxel(image_metas, vmin, vmax)
                    task_id  = str(uuid.uuid4())
                    chunk_id = f"{ix}_{iy}_{iz}"
                    conn.execute(
                        """INSERT INTO tasks
                           (id, chunk_id, voxel_idx, bbox_min, bbox_max,
                            image_ids, status, created_at, updated_at)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            task_id,
                            chunk_id,
                            json.dumps([ix, iy, iz]),
                            json.dumps(vmin.tolist()),
                            json.dumps(vmax.tolist()),
                            json.dumps(voxel_img_ids),
                            "PENDING",
                            now,
                            now,
                        ),
                    )
                    count += 1
                    log.info(
                        f"  Voxel [{ix},{iy},{iz}] bbox={vmin.tolist()} → "
                        f"{vmax.tolist()} | {len(voxel_img_ids)} images"
                    )

    _NUM_TASKS = count
    log.info(f"Master ready — {_NUM_TASKS} tasks in database (all PENDING).")


# ── Fault-tolerance watchdog ──────────────────────────────────────────────────

def _watchdog_loop(interval_s: int = 60) -> None:
    """
    Background thread — runs every `interval_s` seconds.
    Any task that is IN_PROGRESS but whose last_heartbeat is older than
    HEARTBEAT_TIMEOUT_S is reset to PENDING so another worker can claim it.
    """
    log.info("Watchdog started (heartbeat timeout = "
             f"{HEARTBEAT_TIMEOUT_S}s, check every {interval_s}s).")
    while True:
        time.sleep(interval_s)
        cutoff = _now() - HEARTBEAT_TIMEOUT_S
        try:
            with _db() as conn:
                cur = conn.execute(
                    """UPDATE tasks
                       SET status='PENDING', assigned_worker=NULL,
                           last_heartbeat=NULL, updated_at=?
                       WHERE status='IN_PROGRESS'
                         AND (last_heartbeat IS NULL OR last_heartbeat < ?)
                       RETURNING id, chunk_id, assigned_worker""",
                    (_now(), cutoff),
                )
                rows = cur.fetchall()
            for row in rows:
                log.warning(
                    f"[Watchdog] Task {row['id'][:8]}… (chunk {row['chunk_id']}) "
                    f"timed out — reset to PENDING."
                )
        except Exception as exc:
            log.error(f"[Watchdog] DB error: {exc}")


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/join", summary="Register a new worker; returns a unique worker_id")
def join():
    worker_id = str(uuid.uuid4())
    now = _now()
    with _db() as conn:
        conn.execute(
            "INSERT INTO workers (id, registered_at, last_seen) VALUES (?,?,?)",
            (worker_id, now, now),
        )
    log.info(f"[Join] New worker registered: {worker_id[:8]}…")
    return {"worker_id": worker_id}


@app.get("/get_task", summary="Assign the next PENDING task to a worker")
def get_task(worker_id: str):
    """
    Query param: `worker_id` (obtained from /join).
    Returns a full task descriptor or 404 if queue is exhausted.
    """
    now = _now()
    with _db() as conn:
        # Verify worker exists
        w = conn.execute("SELECT id FROM workers WHERE id=?", (worker_id,)).fetchone()
        if w is None:
            raise HTTPException(status_code=401,
                                detail="Unknown worker_id — call /join first.")

        # Atomically claim one PENDING task
        row = conn.execute(
            """SELECT * FROM tasks WHERE status='PENDING'
               ORDER BY created_at ASC LIMIT 1"""
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="No pending tasks available.")

        task_id = row["id"]
        conn.execute(
            """UPDATE tasks
               SET status='IN_PROGRESS', assigned_worker=?,
                   last_heartbeat=?, updated_at=?
               WHERE id=?""",
            (worker_id, now, now, task_id),
        )
        # Update worker's last_seen
        conn.execute("UPDATE workers SET last_seen=? WHERE id=?", (now, worker_id))

    # Build image + camera manifests from in-memory metadata
    image_ids  = json.loads(row["image_ids"])
    voxel_idx  = json.loads(row["voxel_idx"])
    bbox_min   = json.loads(row["bbox_min"])
    bbox_max   = json.loads(row["bbox_max"])

    img_manifest = []
    cam_manifest: dict = {}
    for img_id in image_ids:
        meta = _IMAGE_METAS.get(img_id)
        if meta is None:
            continue
        img_manifest.append({
            "image_id": img_id,
            "name":     meta["name"],
            "cam_id":   meta["cam_id"],
            "R":        meta["R"].tolist(),
            "t":        meta["t"].tolist(),
        })
        cam_id = meta["cam_id"]
        if cam_id not in cam_manifest:
            cam = _CAMERAS.get(cam_id, {})
            cam_manifest[cam_id] = cam

    log.info(
        f"[get_task] Dispatched {task_id[:8]}… → worker {worker_id[:8]}… | "
        f"voxel={voxel_idx} | {len(img_manifest)} images"
    )

    return {
        "task_id":   task_id,
        "chunk_id":  row["chunk_id"],
        "voxel_idx": voxel_idx,
        "bbox_min":  bbox_min,
        "bbox_max":  bbox_max,
        "cameras":   cam_manifest,
        "images":    img_manifest,
    }


@app.post("/heartbeat", summary="Worker signals it is still alive for its task")
def heartbeat(payload: dict = Body(...)):
    """
    Expected body: {"worker_id": "...", "task_id": "..."}
    """
    worker_id = payload.get("worker_id")
    task_id   = payload.get("task_id")
    if not worker_id or not task_id:
        raise HTTPException(status_code=422,
                            detail="Body must contain worker_id and task_id.")

    now = _now()
    with _db() as conn:
        cur = conn.execute(
            """UPDATE tasks
               SET last_heartbeat=?, updated_at=?
               WHERE id=? AND assigned_worker=? AND status='IN_PROGRESS'""",
            (now, now, task_id, worker_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=409,
                                detail="Task not IN_PROGRESS or wrong worker_id.")
        conn.execute("UPDATE workers SET last_seen=? WHERE id=?", (now, worker_id))

    log.info(f"[Heartbeat] worker {worker_id[:8]}… | task {task_id[:8]}…")
    return {"status": "ok", "timestamp": now}


@app.post("/submit_result/{task_id}",
          summary="Worker uploads the finished .ply for a completed task")
async def submit_result(task_id: str, worker_id: str, file: UploadFile = File(...)):
    """
    Query param: `worker_id`
    Multipart: `file` — the compiled .ply chunk
    """
    # Validate task belongs to this worker and is still in-progress
    with _db() as conn:
        row = conn.execute(
            "SELECT status, assigned_worker FROM tasks WHERE id=?", (task_id,)
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
        if row["status"] == "COMPLETED":
            raise HTTPException(status_code=409, detail="Result already received.")
        if row["assigned_worker"] != worker_id:
            raise HTTPException(status_code=403, detail="Task not assigned to this worker.")

    contents = await file.read()
    out_path  = _OUTPUT_DIR / f"chunk_{task_id[:8]}.ply"
    out_path.write_bytes(contents)

    now = _now()
    with _db() as conn:
        conn.execute(
            """UPDATE tasks
               SET status='COMPLETED', result_path=?, updated_at=?
               WHERE id=?""",
            (str(out_path), now, task_id),
        )
        conn.execute("UPDATE workers SET last_seen=? WHERE id=?", (now, worker_id))

    # Count completions
    with _db() as conn:
        n_done = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='COMPLETED'"
        ).fetchone()[0]

    log.info(
        f"[submit_result] task {task_id[:8]}… COMPLETED by worker {worker_id[:8]}… "
        f"({len(contents)/1024:.1f} KB) [{n_done}/{_NUM_TASKS}]"
    )

    if n_done == _NUM_TASKS:
        log.info("All chunks received — triggering stitcher.")
        threading.Thread(target=_stitch_all, daemon=True).start()

    return {"status": "ok", "received_bytes": len(contents)}


@app.get("/image/{image_name:path}", summary="Serve a raw image to a worker")
def get_image(image_name: str):
    img_path = Path(_DATA_DIR) / "images" / image_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_name}")
    return FileResponse(str(img_path))


@app.get("/sparse/{filename}", summary="Serve a COLMAP sparse binary to a worker")
def get_sparse_file(filename: str):
    allowed = {"cameras.bin", "images.bin", "points3D.bin"}
    if filename not in allowed:
        raise HTTPException(status_code=400, detail="Not a valid sparse file name.")
    fpath = Path(_DATA_DIR) / "sparse" / "0" / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(str(fpath))


@app.get("/status", summary="Overall task-queue progress summary")
def get_status():
    with _db() as conn:
        rows = conn.execute(
            """SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"""
        ).fetchall()
    counts = {r["status"]: r["n"] for r in rows}
    return {
        "total":       _NUM_TASKS,
        "PENDING":     counts.get("PENDING",     0),
        "IN_PROGRESS": counts.get("IN_PROGRESS", 0),
        "COMPLETED":   counts.get("COMPLETED",   0),
        "FAILED":      counts.get("FAILED",      0),
        "stitched":    _STITCH_DONE,
    }


# ── Stitcher ──────────────────────────────────────────────────────────────────

def _stitch_all() -> None:
    global _STITCH_DONE
    log.info("Stitching all chunks into stitched_output.ply …")

    with _db() as conn:
        rows = conn.execute(
            "SELECT result_path FROM tasks WHERE status='COMPLETED'"
        ).fetchall()

    all_vertices: list[np.ndarray] = []
    n_props_global: int = 0

    for row in rows:
        ply_path = Path(row["result_path"])
        if not ply_path.exists():
            log.warning(f"[Stitch] Missing file: {ply_path}")
            continue
        data = ply_path.read_bytes()
        try:
            verts, _, n_props = _parse_ply(data)
            if verts is not None:
                all_vertices.append(verts)
                n_props_global = n_props
        except Exception as exc:
            log.warning(f"[Stitch] Could not parse {ply_path}: {exc}")

    if not all_vertices:
        log.error("[Stitch] No valid PLY chunks — aborted.")
        return

    merged   = np.concatenate(all_vertices, axis=0)
    stitched = _OUTPUT_DIR / "stitched_output.ply"
    _write_ply_raw(stitched, merged, n_props_global)
    _STITCH_DONE = True
    log.info(f"[Stitch] {merged.shape[0]} total Gaussians → {stitched}")


def _parse_ply(data: bytes) -> tuple[np.ndarray | None, bytes, int]:
    """
    Minimal binary PLY parser — extracts the vertex data as a float32 array.
    Returns (vertex_array, header_bytes, n_float_props).
    """
    end_tag = b"end_header\n"
    idx = data.find(end_tag)
    if idx == -1:
        end_tag = b"end_header\r\n"
        idx = data.find(end_tag)
        if idx == -1:
            return None, b"", 0
    header_end = idx + len(end_tag)
    header = data[:header_end]
    body   = data[header_end:]

    header_str = header.decode("utf-8", errors="replace")
    n_vertices = n_props = 0
    for line in header_str.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
        elif line.startswith("property float"):
            n_props += 1

    if n_vertices == 0 or n_props == 0:
        return None, header, 0

    expected = n_vertices * n_props * 4
    if len(body) < expected:
        log.warning(f"[Stitch] PLY body short: {len(body)} < {expected}")
        n_vertices = len(body) // (n_props * 4)

    arr = np.frombuffer(body[: n_vertices * n_props * 4], dtype=np.float32)
    return arr.reshape(n_vertices, n_props), header, n_props


def _write_ply_raw(path: Path, data: np.ndarray, n_props: int) -> None:
    N = data.shape[0]
    standard_props = [
        b"property float x\n",
        b"property float y\n",
        b"property float z\n",
        b"property float nx\n",
        b"property float ny\n",
        b"property float nz\n",
        b"property float f_dc_0\n",
        b"property float f_dc_1\n",
        b"property float f_dc_2\n",
    ]
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        for prop in standard_props[:n_props]:
            f.write(prop)
        for i in range(9, n_props):
            f.write(f"property float prop_{i}\n".encode())
        f.write(b"end_header\n")
        f.write(data.astype(np.float32).tobytes())


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Splat-Grid Master Node v3")
    p.add_argument("--data_dir",   type=str, default="data",
                   help="Root COLMAP dataset directory (contains sparse/ and images/)")
    p.add_argument("--output_dir", type=str, default="results",
                   help="Directory for chunk PLYs and stitched output")
    p.add_argument("--host",       type=str, default="0.0.0.0",
                   help="Bind host (0.0.0.0 = all interfaces)")
    p.add_argument("--port",       type=int, default=8765,
                   help="TCP port")
    p.add_argument("--grid",       type=str, default="2x2x2",
                   help="Voxel grid spec, e.g. '2x2x2' or '3x3x3'")
    p.add_argument("--db",         type=str, default="splat_state.db",
                   help="Path to the SQLite state database")
    p.add_argument("--reset_db",   action="store_true",
                   help="Drop and recreate the task database on startup")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    _DB_PATH = args.db

    # Initialise (or reset) the DB schema
    from init_db import init_db
    init_db(_DB_PATH, reset=args.reset_db)

    # Partition dataset and populate task queue
    init_master(args.data_dir, args.grid, args.output_dir)

    # Start heartbeat watchdog in background
    wdog = threading.Thread(target=_watchdog_loop, args=(60,), daemon=True)
    wdog.start()

    log.info(f"FastAPI server starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
