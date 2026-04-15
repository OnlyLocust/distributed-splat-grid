# Distributed 3D Gaussian Splatting Pipeline

A complete distributed training pipeline for 3D Gaussian Splatting optimized for 4GB GPU constraints (NVIDIA RTX 3050). This pipeline processes large COLMAP reconstructions by splitting them into spatial chunks, training them independently, and merging the results.

## Features

- **4GB VRAM Optimized**: Designed specifically for laptops with limited GPU memory
- **Distributed Processing**: Splits large scenes into manageable chunks
- **Fault Tolerance**: Automatic retry with reduced resources on OOM errors
- **Camera Culling**: Intelligent camera selection to reduce training workload
- **Deduplication**: Spatial hashing to remove duplicate Gaussians from overlapping chunks
- **Progress Tracking**: SQLite database for state management and crash recovery

## System Requirements

- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (tested on RTX 3050)
- PyTorch 2.x
- gsplat library
- COLMAP reconstruction data

## Installation

1. Clone or download this pipeline
2. Install dependencies:

```bash
pip install torch torchvision
pip install gsplat
pip install numpy scipy pillow
pip install plyfile  # Optional, for PLY validation
```

## Project Structure

```
3dgs_pipeline/
|-- splitter.py          # Data ingestion & spatial partitioning
|-- optimizer.py         # Camera frustum culling 
|-- worker.py           # Rendering & optimization per chunk
|-- orchestrator.py     # Distributed systems management
|-- stitcher.py         # Output assembly & deduplication
|-- utils/
|   |-- colmap_reader.py    # COLMAP binary parser
|   |-- ply_writer.py       # PLY file I/O utilities
|   |-- vram_guard.py       # Memory management utilities
|-- tasks/               # Created by splitter + optimizer
|-- results/             # Created by worker via orchestrator
|-- state.db            # Created by orchestrator
|-- pipeline.log         # Training log
```

## Pipeline Execution

The pipeline follows a 4-step process. Execute in this order:

### Step 1: Split COLMAP data into chunks

```bash
python splitter.py --input ./sparse/0 --output ./tasks
```

Options:
- `--target_points`: Target points per chunk (default: 50000)
- `--halo_margin`: Halo margin percentage (default: 0.05)
- `--dry_run`: Print what would happen without writing files

### Step 2: Optimize cameras (frustum culling)

```bash
python optimizer.py --colmap_dir ./sparse/0 --tasks_dir ./tasks --images_dir ./images
```

Options:
- `--margin`: Frustum culling margin percentage (default: 0.1)
- `--dry_run`: Print what would happen without writing files

### Step 3: Train all chunks (orchestrated)

```bash
python orchestrator.py --tasks_dir ./tasks --images_dir ./images --output_dir ./results
```

Options:
- `--device`: PyTorch device (default: cuda)
- The orchestrator automatically calls worker.py for each chunk

### Step 4: Stitch final output

```bash
python stitcher.py --results_dir ./results --output ./final_output.ply
```

Options:
- `--precision`: Decimal precision for spatial hashing (default: 3)
- `--validate`: Validate output PLY file after stitching

## Individual Component Usage

### Training a Single Chunk

For debugging or testing individual chunks:

```bash
python worker.py --chunk_dir ./tasks/chunk_0_0 --output_dir ./results --num_iterations 500
```

### Testing Components

Test utility modules:

```bash
# Test COLMAP reader
python utils/colmap_reader.py ./sparse/0

# Test PLY I/O
python utils/ply_writer.py --test

# Test VRAM guard
python utils/vram_guard.py
```

## 4GB VRAM Optimization Rules

This pipeline follows strict memory management rules:

1. **Chunk Size**: Target 50,000 points per chunk (not 100,000)
2. **SH Degree**: Only degree 1 spherical harmonics (degree 3 uses 16× memory)
3. **Memory Types**: 
   - SH coefficients: float16
   - Positions/scales/quaternions/opacities: float32
4. **Image Loading**: Load one image per iteration, resize to max 800px
5. **Cleanup**: Call `torch.cuda.empty_cache()` every 100 iterations
6. **Gaussian Cap**: Hard limit of 200,000 Gaussians per chunk
7. **Sequential Processing**: No parallel training on single GPU
8. **OOM Recovery**: Automatic retry with reduced resources

## Configuration

### Training Parameters

Default training configuration (can be modified in orchestrator.py):

```python
config = {
    'num_iterations': 3000,
    'lr_positions': 1.6e-4,
    'lr_opacities': 1e-2,
    'lr_scales': 1e-3,
    'lr_rotations': 1e-3,
    'lr_colors': 5e-3,
    'densify_interval': 100,
    'max_gaussians': 200000
}
```

### Adaptive Density Control

- **Clone**: High gradient + small scale
- **Split**: High gradient + large scale  
- **Prune**: Opacity < 0.005
- **Frequency**: Every 100 iterations (500-4000)
- **Safety**: Skip if >200,000 Gaussians

### Retry Strategy

On OOM errors, the pipeline automatically:
1. **Attempt 1**: Halve max_gaussians (100,000), double densify_interval (200)
2. **Attempt 2**: Halve max_gaussians (50,000), double densify_interval (400)
3. **Attempt 3**: Mark as FAILED and continue

## Output Format

The final `final_output.ply` follows the standard 3DGS format:

```
ply
format binary_little_endian 1.0
element vertex N
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
```

Compatible with:
- SuperSplat
- Luma
- Polycam
- Other 3DGS viewers

## Monitoring and Logging

- **Progress**: Live progress updates during training
- **Database**: SQLite state tracking in `state.db`
- **Logs**: Detailed training log in `pipeline.log`
- **Statistics**: Final summary with compression ratios

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Pipeline automatically retries with reduced resources
   - Check `pipeline.log` for OOM events
   - Consider reducing `--target_points` in splitter

2. **COLMAP Format Errors**:
   - Ensure COLMAP binary files are in sparse/0/ directory
   - Verify files: points3D.bin, cameras.bin, images.bin

3. **Missing Images**:
   - Check that image paths in cameras.json match actual files
   - Verify images directory structure

4. **gsplat Import Error**:
   - Install with: `pip install gsplat`
   - Pipeline falls back to dummy rendering if gsplat unavailable

### Performance Tips

- **SSD Storage**: Use SSD for faster I/O with many small files
- **Memory Monitoring**: Watch GPU memory with `nvidia-smi`
- **Chunk Size**: Adjust `--target_points` based on your scene complexity
- **Image Resolution**: Lower resolution images train faster

## Example Workflow

```bash
# 1. Split scene into chunks
python splitter.py --input ./my_scene/sparse/0 --output ./tasks

# 2. Filter cameras per chunk
python optimizer.py --colmap_dir ./my_scene/sparse/0 --tasks_dir ./tasks --images_dir ./my_scene/images

# 3. Train all chunks
python orchestrator.py --tasks_dir ./tasks --images_dir ./my_scene/images --output_dir ./results

# 4. Merge results
python stitcher.py --results_dir ./results --output ./final_scene.ply
```

## License

This implementation follows the 3D Gaussian Splatting research paper methodology. Ensure compliance with original paper licensing when using for commercial purposes.

## Contributing

This pipeline is optimized for 4GB GPUs. For higher-end GPUs, you may want to:
- Increase `target_points_per_chunk` in splitter.py
- Enable SH degree 3 in worker.py
- Increase `max_gaussians` limits
- Enable parallel processing in orchestrator.py
