import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np

from colmap_io import load_colmap_scene, intrinsics_from_camera


class Splitter:
    """
    V1 head-node splitter.
    Uses 1x1x1 partitioning, so one chunk contains all points/cameras.
    """

    def __init__(self, data_dir: str, tasks_dir: str = "tasks", grid_shape: Tuple[int, int, int] = (1, 1, 1)) -> None:
        self.data_dir = data_dir
        self.tasks_dir = tasks_dir
        self.grid_shape = grid_shape

    def _compute_bbox(self, points_xyz: np.ndarray) -> Dict[str, Any]:
        bb_min = points_xyz.min(axis=0)
        bb_max = points_xyz.max(axis=0)
        return {"min": bb_min.tolist(), "max": bb_max.tolist()}

    def _split_along_x(self, xyz: np.ndarray, bb_min: np.ndarray, bb_max: np.ndarray) -> List[np.ndarray]:
        gx, gy, gz = self.grid_shape
        if (gy, gz) != (1, 1):
            raise ValueError("V1 splitter currently supports only gx x 1 x 1 partitioning.")
        if gx < 1:
            raise ValueError("grid x dimension must be >= 1")

        if gx == 1:
            return [np.arange(xyz.shape[0])]

        edges = np.linspace(bb_min[0], bb_max[0], num=gx + 1)
        chunk_indices = []
        xvals = xyz[:, 0]
        for i in range(gx):
            left = edges[i]
            right = edges[i + 1]
            if i == gx - 1:
                mask = (xvals >= left) & (xvals <= right)
            else:
                mask = (xvals >= left) & (xvals < right)
            chunk_indices.append(np.where(mask)[0])
        return chunk_indices

    def run(self) -> str:
        scene = load_colmap_scene(self.data_dir)
        points3d = scene["points3D"]
        images = scene["images"]
        cameras = scene["cameras"]

        point_ids = np.array(list(points3d.keys()), dtype=np.int64)
        xyz = np.stack([points3d[int(pid)]["xyz"] for pid in point_ids], axis=0).astype(np.float32)
        rgb = np.stack([points3d[int(pid)]["rgb"] for pid in point_ids], axis=0).astype(np.float32) / 255.0
        bb_min = xyz.min(axis=0)
        bb_max = xyz.max(axis=0)
        bbox = {"min": bb_min.tolist(), "max": bb_max.tolist()}

        camera_records = []
        for _, image in images.items():
            cam = cameras[image["camera_id"]]
            fx, fy, cx, cy = intrinsics_from_camera(cam)
            camera_records.append(
                {
                    "image_id": int(image["image_id"]),
                    "camera_id": int(image["camera_id"]),
                    "name": image["name"],
                    "width": int(cam["width"]),
                    "height": int(cam["height"]),
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "qvec": image["qvec"].tolist(),
                    "tvec": image["tvec"].tolist(),
                }
            )

        chunk_indices = self._split_along_x(xyz, bb_min, bb_max)
        first_chunk_dir = ""
        for chunk_id, idxs in enumerate(chunk_indices):
            chunk_dir = os.path.join(self.tasks_dir, f"chunk_{chunk_id}")
            os.makedirs(chunk_dir, exist_ok=True)
            if chunk_id == 0:
                first_chunk_dir = chunk_dir

            chunk_xyz = xyz[idxs]
            chunk_rgb = rgb[idxs]
            chunk_ids = point_ids[idxs]
            if chunk_xyz.shape[0] == 0:
                chunk_bbox = bbox
            else:
                chunk_bbox = self._compute_bbox(chunk_xyz)

            task_meta = {
                "chunk_id": chunk_id,
                "grid_shape": [self.grid_shape[0], self.grid_shape[1], self.grid_shape[2]],
                "bbox": chunk_bbox,
                "point_count": int(chunk_xyz.shape[0]),
                "camera_count": int(len(camera_records)),
                "images_root": os.path.join(self.data_dir, "images"),
                "points_file": "points.npz",
                "cameras_file": "cameras.json",
            }

            np.savez_compressed(
                os.path.join(chunk_dir, "points.npz"),
                point_ids=chunk_ids,
                xyz=chunk_xyz,
                rgb=chunk_rgb,
            )
            with open(os.path.join(chunk_dir, "cameras.json"), "w", encoding="utf-8") as f:
                json.dump(camera_records, f, indent=2)
            with open(os.path.join(chunk_dir, "task.json"), "w", encoding="utf-8") as f:
                json.dump(task_meta, f, indent=2)

        return first_chunk_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create local chunk tasks from COLMAP scene.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--tasks_dir", type=str, default="tasks")
    parser.add_argument("--grid_x", type=int, default=1, help="Number of chunks along X axis (V1 supports Xx1x1).")
    args = parser.parse_args()

    splitter = Splitter(data_dir=args.data_dir, tasks_dir=args.tasks_dir, grid_shape=(args.grid_x, 1, 1))
    first = splitter.run()
    print(f"[Splitter] created tasks in: {args.tasks_dir} (first chunk: {first})")


if __name__ == "__main__":
    main()
