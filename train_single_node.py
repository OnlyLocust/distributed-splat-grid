import os
import math
import argparse
import time
import struct
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

try:
    from gsplat import rasterization
except ImportError:
    rasterization = None

def get_args():
    parser = argparse.ArgumentParser(description="Gaussian Splatting 'Hello World' for RTX 3050")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="output.ply")
    parser.add_argument("--iterations", type=int, default=70000)
    parser.add_argument("--lr_pos", type=float, default=1.6e-4)
    parser.add_argument("--lr_color", type=float, default=2.5e-3)
    parser.add_argument("--lr_opacity", type=float, default=0.05)
    parser.add_argument("--lr_scale", type=float, default=5e-3)
    parser.add_argument("--lr_rot", type=float, default=1e-3)
    parser.add_argument("--bbox", type=str, default="-1,-1,-1,1,1,1", help="min_x,min_y,min_z,max_x,max_y,max_z")
    parser.add_argument("--densify_start", type=int, default=500)
    parser.add_argument("--densify_every", type=int, default=100)
    parser.add_argument("--densify_end", type=int, default=15000)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)
    parser.add_argument("--checkpoint_every", type=int, default=5000)
    parser.add_argument("--max_gaussians", type=int, default=120000, help="Hard cap for 4GB VRAM")
    parser.add_argument("--image_downscale", type=int, default=2)
    return parser.parse_args()

def parse_bbox(bbox_str):
    parts = list(map(float, bbox_str.split(",")))
    return torch.tensor(parts[:3], device="cuda"), torch.tensor(parts[3:], device="cuda")

def read_points3D_binary(path):
    points3D = {}
    with open(path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            pt_id = struct.unpack("<Q", fid.read(8))[0]
            xyz = struct.unpack("<3d", fid.read(24))
            rgb = struct.unpack("<3B", fid.read(3))
            fid.read(8) # error
            track_len = struct.unpack("<Q", fid.read(8))[0]
            fid.read(track_len * 8)
            points3D[pt_id] = {'xyz': xyz, 'rgb': rgb}
    return points3D

def initialize_from_colmap(data_dir, device):
    points_path = os.path.join(data_dir, "sparse", "0", "points3D.bin")
    if not os.path.exists(points_path):
        print(f"Warning: {points_path} not found. Falling back to random initialization.")
        return None
    
    print(f"Loading sparse points from {points_path}...")
    points3D = read_points3D_binary(points_path)
    num_points = len(points3D)
    print(f"Loaded {num_points} points.")

    xyzs = np.array([pt['xyz'] for pt in points3D.values()])
    rgbs = np.array([pt['rgb'] for pt in points3D.values()]) / 255.0

    means = torch.tensor(xyzs, dtype=torch.float32, device=device)
    colors = torch.tensor(rgbs, dtype=torch.float32, device=device)
    opacities = torch.ones((num_points, 1), device=device) * 0.1
    scales = torch.ones((num_points, 3), device=device) * -3.0 # log scale
    quats = torch.zeros((num_points, 4), device=device)
    quats[:, 0] = 1.0 # identity

    return means, colors, opacities, scales, quats

def write_ply(path, means, colors, opacities, scales, quats):
    vertex_count = means.shape[0]
    SH_C0 = 0.28209479177387814
    
    means_np = means.detach().cpu().numpy()
    # Convert RGB [0, 1] to SH DC coefficients
    colors_np = (colors.detach().cpu().numpy() - 0.5) / SH_C0
    opacities_np = opacities.detach().cpu().numpy()
    scales_np = scales.detach().cpu().numpy()
    quats_np = quats.detach().cpu().numpy()
    
    # Normal vectors (nx, ny, nz) are usually 0 in Gaussian Splatting PLY
    normals_np = np.zeros((vertex_count, 3), dtype=np.float32)
    
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

        for i in range(vertex_count):
            # Pos (3f), Normals (3f), f_dc (3f), Opacity (1f), Scale (3f), Rotation (4f)
            # Total: 3+3+3+1+3+4 = 17 floats
            f.write(struct.pack("<17f", 
                *means_np[i], 
                *normals_np[i], 
                *colors_np[i], 
                float(opacities_np[i].item()), 
                *scales_np[i], 
                *quats_np[i]
            ))

def main():
    args = get_args()
    device = torch.device("cuda")
    bbox_min, bbox_max = parse_bbox(args.bbox)

    # Initialize Gaussians from COLMAP or fallback to random
    print("Initializing Gaussians...")
    colmap_data = initialize_from_colmap(args.data_dir, device)
    
    if colmap_data:
        means, colors, opacities, scales, quats = colmap_data
    else:
        num_initial = 5000 
        means = torch.rand((num_initial, 3), device=device) * 2 - 1
        colors = torch.rand((num_initial, 3), device=device)
        opacities = torch.ones((num_initial, 1), device=device) * 0.1
        scales = torch.ones((num_initial, 3), device=device) * -3.0 
        quats = torch.zeros((num_initial, 4), device=device)
        quats[:, 0] = 1.0 # identity

    means.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    scales.requires_grad = True
    quats.requires_grad = True

    optimizer = optim.Adam([
        {'params': [means], 'lr': args.lr_pos, "name": "xyz"},
        {'params': [colors], 'lr': args.lr_color, "name": "color"},
        {'params': [opacities], 'lr': args.lr_opacity, "name": "opacity"},
        {'params': [scales], 'lr': args.lr_scale, "name": "scale"},
        {'params': [quats], 'lr': args.lr_rot, "name": "rot"}
    ])

    print("Starting training loop...")
    for step in range(1, args.iterations + 1):
        optimizer.zero_grad()
        
        # Dummy loss for layout
        loss = torch.sum(means ** 2) * 0.001
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            vram_gb = torch.cuda.memory_allocated() / 1e9
            print(f"Iter: {step} | Loss: {loss.item():.4f} | Gaussians: {means.shape[0]} | VRAM: {vram_gb:.2f} GB")

        if step % args.checkpoint_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            # torch.save(...) # Dummy save

        # Pruning check based on max gaussians and bbox
        if step > args.densify_start and step < args.densify_end and step % args.densify_every == 0:
            in_bbox = ((means >= bbox_min) & (means <= bbox_max)).all(dim=-1)
            valid = (opacities.squeeze(-1) > 0.005) & in_bbox
            # If means count > max_gaussians, skip splitting
            if means.shape[0] < args.max_gaussians:
                # Do split logic
                pass
            
    print("Training finished. Exporting PLY...")
    write_ply(args.output, means, colors, opacities, scales, quats)
    print("Export complete.")

if __name__ == "__main__":
    main()
