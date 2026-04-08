import struct
from typing import Dict

import numpy as np


def write_gaussian_ply(path: str, means: np.ndarray, sh_dc: np.ndarray, opacities: np.ndarray, scales: np.ndarray, quats: np.ndarray) -> None:
    vertex_count = means.shape[0]
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {vertex_count}\n".encode("utf-8"))
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
        f.write(b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n")
        f.write(b"property float opacity\n")
        f.write(b"property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
        f.write(b"property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
        f.write(b"end_header\n")

        normals = np.zeros((vertex_count, 3), dtype=np.float32)
        for i in range(vertex_count):
            f.write(
                struct.pack(
                    "<17f",
                    *means[i].astype(np.float32),
                    *normals[i],
                    *sh_dc[i].astype(np.float32),
                    float(opacities[i]),
                    *scales[i].astype(np.float32),
                    *quats[i].astype(np.float32),
                )
            )


def read_gaussian_ply(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("utf-8").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        vertex_count = None
        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                break
        if vertex_count is None:
            raise ValueError(f"Could not parse vertex count from PLY header: {path}")

        raw = f.read(vertex_count * 17 * 4)
        vals = np.array(struct.unpack("<" + "f" * (vertex_count * 17), raw), dtype=np.float32).reshape(vertex_count, 17)

    return {
        "means": vals[:, 0:3],
        "sh_dc": vals[:, 6:9],
        "opacities": vals[:, 9:10],
        "scales": vals[:, 10:13],
        "quats": vals[:, 13:17],
    }
