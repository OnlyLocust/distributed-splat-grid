"""
splat_grid.ply_utils
====================
PLY file writer and multi-file stitcher for the Splat-Grid pipeline.

write_ply  — used by worker.py to export a trained chunk.
stitch_ply — used by master.py to merge all chunk PLYs into the final output.

The format written is the 3DGS-compatible binary-little-endian PLY used by
viewers like SuperSplat and the INRIA SIBR viewer.
"""

import math
import logging
import struct
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Writer (worker-side)
# ---------------------------------------------------------------------------

def write_ply(
    path: str,
    means,       # torch.Tensor [N, 3]
    sh_colors,   # torch.Tensor [N, K, 3]
    opacities,   # torch.Tensor [N, 1]
    scales,      # torch.Tensor [N, 3]
    quats,       # torch.Tensor [N, 4]
) -> None:
    """
    Export Gaussians to a 3DGS-compatible binary PLY file.

    Properties written (in order):
      x y z  nx ny nz  f_dc_0 f_dc_1 f_dc_2  f_rest_*  opacity
      scale_0 scale_1 scale_2  rot_0 rot_1 rot_2 rot_3
    """
    N        = means.shape[0]
    means_np = means.detach().cpu().float().numpy()
    sh_np    = sh_colors.detach().cpu().float().numpy()   # [N, K, 3]
    ops_np   = opacities.detach().cpu().float().numpy().reshape(-1, 1)
    sc_np    = scales.detach().cpu().float().numpy()
    qt_np    = quats.detach().cpu().float().numpy()
    nm_np    = np.zeros((N, 3), dtype=np.float32)

    f_dc    = sh_np[:, 0, :]                              # [N, 3]
    f_rest  = sh_np[:, 1:, :].reshape(N, -1)             # [N, (K-1)*3]
    n_rest  = f_rest.shape[1]

    actual_sh_degree = int(round(math.sqrt(sh_np.shape[1]))) - 1

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

        data = np.hstack(
            [means_np, nm_np, f_dc, f_rest, ops_np, sc_np, qt_np]
        ).astype(np.float32)
        fout.write(data.tobytes())

    log.info(f"[PLY] {N} Gaussians (SH deg {actual_sh_degree}) → {path}")


# ---------------------------------------------------------------------------
# Stitcher (master-side)
# ---------------------------------------------------------------------------

def _parse_ply_header(path: str) -> tuple[int, int]:
    """
    Read a binary_little_endian PLY header and return
    (vertex_count, header_byte_length).
    Raises ValueError for unsupported formats.
    """
    header_bytes = bytearray()
    with open(path, "rb") as f:
        # Read until 'end_header\n'
        while True:
            line = f.readline()
            header_bytes.extend(line)
            if line.strip() == b"end_header":
                break
            if len(header_bytes) > 8192:
                raise ValueError(f"PLY header too large in {path}")

    header_text = header_bytes.decode("ascii", errors="ignore")
    if "format binary_little_endian" not in header_text:
        raise ValueError(f"Unsupported PLY format in {path} (expected binary_little_endian)")

    m = re.search(r"element vertex\s+(\d+)", header_text)
    if not m:
        raise ValueError(f"Could not find 'element vertex' count in PLY header: {path}")

    vertex_count   = int(m.group(1))
    header_length  = len(header_bytes)
    return vertex_count, header_length


def _bytes_per_vertex(path: str) -> int:
    """
    Compute vertex byte stride from the PLY header's property list.
    Every property is assumed 'float' (4 bytes), which matches write_ply().
    """
    header_bytes = bytearray()
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            header_bytes.extend(line)
            if line.strip() == b"end_header":
                break

    header_text = header_bytes.decode("ascii", errors="ignore")
    n_props = header_text.count("property float ")
    if n_props == 0:
        raise ValueError(f"No 'property float' entries found in {path}")
    return n_props * 4   # each float32 = 4 bytes


def stitch_ply_files(input_paths: list[str], output_path: str) -> int:
    """
    Concatenate multiple 3DGS PLY files (same property layout) into one.

    All input files must have been written by write_ply() (identical property
    list). The vertex counts are summed and the binary data sections are
    concatenated directly — no re-encoding.

    Returns the total number of vertices in the output file.
    """
    if not input_paths:
        raise ValueError("stitch_ply_files: input_paths is empty")

    log.info(f"[Stitch] Merging {len(input_paths)} PLY files → {output_path}")

    # Use the first file as the header template
    ref_path = input_paths[0]
    total_vertices = 0
    chunks: list[bytes] = []

    bpv = _bytes_per_vertex(ref_path)

    for fpath in input_paths:
        vcnt, hlen = _parse_ply_header(fpath)
        bpv_i      = _bytes_per_vertex(fpath)

        if bpv_i != bpv:
            raise ValueError(
                f"PLY property stride mismatch: {ref_path} has {bpv}B/vertex "
                f"but {fpath} has {bpv_i}B/vertex. Cannot stitch."
            )

        with open(fpath, "rb") as f:
            f.seek(hlen)
            data = f.read(vcnt * bpv)
            if len(data) != vcnt * bpv:
                log.warning(
                    f"  {fpath}: expected {vcnt * bpv} bytes of vertex data, "
                    f"got {len(data)}. File may be truncated."
                )
        chunks.append(data)
        total_vertices += vcnt
        log.info(f"  + {Path(fpath).name}: {vcnt:,} vertices")

    # Build new header with updated count
    ref_vc, ref_hlen = _parse_ply_header(ref_path)
    with open(ref_path, "rb") as f:
        old_header_bytes = f.read(ref_hlen)

    old_header_text = old_header_bytes.decode("ascii", errors="ignore")
    new_header_text = re.sub(
        r"(element vertex\s+)\d+",
        f"element vertex {total_vertices}",
        old_header_text,
    )
    new_header_bytes = new_header_text.encode("ascii")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fout:
        fout.write(new_header_bytes)
        for chunk in chunks:
            fout.write(chunk)

    log.info(f"[Stitch] Done — {total_vertices:,} total Gaussians → {output_path}")
    return total_vertices
