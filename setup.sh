#!/usr/bin/env bash
# setup.sh — Splat-Grid dependency installer
# Usage:
#   bash setup.sh master   → install master-node dependencies
#   bash setup.sh worker   → install worker-node dependencies
#   bash setup.sh both     → install everything (for dev / single-machine testing)
#
# Assumes Python 3.10+ and pip are available in the current environment.
# For Conda users, activate your environment first:
#   conda activate gaussian_splatting && bash setup.sh worker

set -euo pipefail

ROLE="${1:-both}"

echo "╔══════════════════════════════════════════╗"
echo "║     Splat-Grid Dependency Installer      ║"
echo "╚══════════════════════════════════════════╝"
echo "Role: ${ROLE}"
echo ""

# ---------------------------------------------------------------------------
# Master dependencies
# ---------------------------------------------------------------------------
install_master() {
    echo "── Installing MASTER dependencies ──"
    pip install --upgrade pip
    pip install \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "numpy>=1.24" \
        "Pillow>=9.0"
    echo "✓  Master dependencies installed."
}

# ---------------------------------------------------------------------------
# Worker dependencies
# ---------------------------------------------------------------------------
install_worker() {
    echo "── Installing WORKER dependencies ──"
    pip install --upgrade pip

    # Core
    pip install \
        "numpy>=1.24" \
        "Pillow>=9.0" \
        "requests>=2.28"

    # gsplat (GPU rasterizer — requires CUDA toolkit to be present)
    echo ""
    echo "Installing gsplat (this may take a few minutes, CUDA compilation required)..."
    pip install gsplat

    # Optional SSIM loss — gracefully ignored by worker if missing
    echo ""
    echo "Installing pytorch-msssim (optional — improves training quality)..."
    pip install pytorch-msssim || echo "  ⚠  pytorch-msssim install failed — L1-only loss will be used."

    echo ""
    echo "NOTE: PyTorch must be installed separately with the correct CUDA version."
    echo "Example for CUDA 12.1:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    echo "Find your CUDA version with: nvcc --version  OR  nvidia-smi"
    echo "Wheel index: https://download.pytorch.org/whl/torch_stable.html"
    echo ""
    echo "✓  Worker dependencies installed."
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${ROLE}" in
    master)
        install_master
        ;;
    worker)
        install_worker
        ;;
    both)
        install_master
        echo ""
        install_worker
        ;;
    *)
        echo "Unknown role '${ROLE}'. Use: master | worker | both"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════"
echo "Setup complete for role: ${ROLE}"
echo ""
echo "MASTER launch command:"
echo "  python master.py --data_dir /path/to/colmap_data --grid 2 --port 8000"
echo ""
echo "WORKER launch command:"
echo "  python worker.py --master http://<MASTER_IP>:8000 --iterations 750 --downscale 4"
echo "═══════════════════════════════════════════"
