# Gaussian Splatting Pipeline Execution Guide

This document provides the necessary commands and configuration steps to execute the Gaussian Splatting training pipeline on a Windows machine using **WSL2** with an **NVIDIA RTX 3050 (4GB VRAM)**.

## 1. Environment Setup

The project uses WSL2 (Ubuntu) and Conda for environment isolation. Ensure you have WSL2 installed and configured with NVIDIA drivers on Windows.

### Initial Setup
Run the provided setup script to install Miniconda, create the environment, and install dependencies.

```bash
# From within your WSL2 terminal
chmod +x setup.sh
./setup.sh
```

### Manual Environment Activation
If the environment is already created, activate it using:

```bash
conda activate gaussian-splat
```

## 2. Version & Hardware Verification

Before running the full pipeline, verify that PyTorch can access your GPU and that all library versions are correct for the RTX 3050.

```bash
python smoke_test.py
```

> [!IMPORTANT]
> The smoke test specifically checks for an **RTX 3050**. If you are using a different GPU, you may need to update the assertion in `smoke_test.py`.

## 3. Data Preparation (COLMAP)

The pipeline expects images to be placed in `data/images`. 

### Run COLMAP Structure-from-Motion (SfM)
If you haven't processed your images yet, use COLMAP to generate the sparse reconstruction within WSL2:

```bash
# 1. Feature extraction
colmap feature_extractor --database_path data/database.db --image_path data/images

# 2. Feature matching
colmap exhaustive_matcher --database_path data/database.db

# 3. Sparse reconstruction
mkdir -p data/sparse
colmap mapper --database_path data/database.db --image_path data/images --output_path data/sparse
```

### Validate COLMAP Output
Run the validation script to ensure the reconstruction is sufficient for training (minimum 40 cameras and 500 points).

```bash
python validate_colmap.py --data_dir data
```

## 4. Pipeline Execution (Training)

The training script is optimized for the **4GB VRAM** limit of the RTX 3050. It includes a hard cap on the number of Gaussians and automatic image downscaling.

```bash
python train_single_node.py \
    --data_dir data \
    --iterations 30000 \
    --max_gaussians 120000 \
    --image_downscale 2 \
    --output output.ply
```

### Optimization Parameters for RTX 3050:
- `--max_gaussians 120000`: Prevents "Out of Memory" errors by capping the model size.
- `--image_downscale 2`: Reduces VRAM usage by downscaling input images during training.
- `--iterations 30000`: Standard training length; can be reduced to 7000 for quick tests.

## 5. Configuration & Module Integration

### Environment Variables (.env)
The project includes a `.env` file for integration with external modules (e.g., Supabase for storage/tracking, Anedya for telemetry). 

Ensure your `.env` is properly configured:
- `VITE_SUPABASE_URL`: Your Supabase instance URL.
- `VITE_ANEDYA_API_KEY`: API key for Anedya integration.
- `VITE_USE_MOCKS`: Set to `true` to use mock data for testing.

### Dependency Manifest
The versions below are verified for compatibility with CUDA 12.1 and the RTX 3050 architecture.

| Dependency | Version | Note |
| :--- | :--- | :--- |
| Python | 3.10 | Required for `gsplat` 1.3.0 |
| PyTorch | 2.2.0+cu121 | Match with CUDA 12.1 for performance |
| gsplat | 1.3.0 | Core splatting engine |
| COLMAP | Latest (apt) | Required for SfM |

## 6. Troubleshooting

- **VRAM Out of Memory**: If you encounter OOM errors, try increasing `--image_downscale` to 4 or decreasing `--max_gaussians` to 80,000.
- **CUDA Errors**: Ensure your Windows host has the latest NVIDIA drivers. WSL2 uses the host's GPU drivers automatically.
- **Missing `gsplat` modules**: If you see an `ImportError`, ensure you have compiled/installed `gsplat` correctly. `setup.sh` handles this via pip but might require build tools (`gcc`, `g++`).
