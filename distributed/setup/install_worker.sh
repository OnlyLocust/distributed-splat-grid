#!/bin/bash
# Worker Node Installation Script for Distributed 3DGS Pipeline
# Run this on each worker node

set -e

echo "=== Installing 3DGS Worker Node Dependencies ==="

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. CUDA may not be properly installed."
    echo "Please ensure CUDA drivers are installed before continuing."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check GPU information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
    echo
fi

# Install Python dependencies
echo "Installing Python packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install gsplat numpy scipy pillow plyfile rich boto3 redis rq

# Do NOT install Redis server on workers - they only need the client
echo "Redis client installed (server not required on workers)"

# Create shared storage mount point
echo "Creating shared storage mount point..."
SHARED_DIR="/mnt/3dgs_share"
sudo mkdir -p $SHARED_DIR

# Create temporary directories for worker processing
echo "Creating temporary directories..."
sudo mkdir -p /tmp/3dgs_chunks
sudo mkdir -p /tmp/3dgs_results
sudo chmod 755 /tmp/3dgs_chunks /tmp/3dgs_results

# Create worker log directory
echo "Creating worker log directory..."
sudo mkdir -p $SHARED_DIR/worker_logs
sudo chmod 755 $SHARED_DIR/worker_logs

# Install NFS client
echo "Installing NFS client..."
sudo apt-get install -y nfs-common

echo ""
echo "=== Worker Node Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy config.json from the master node:"
echo "   scp user@master_ip:/path/to/config.json ."
echo ""
echo "2. Mount shared storage (see nfs_setup.md for detailed instructions):"
echo "   sudo mount -t nfs master_ip:/mnt/3dgs_share $SHARED_DIR"
echo ""
echo "3. Test the setup:"
echo "   ls $SHARED_DIR"
echo ""
echo "4. Start worker daemon (one per GPU):"
echo "   CUDA_VISIBLE_DEVICES=0 python distributed/worker_daemon.py --config config.json --gpu 0"
echo ""
echo "5. For multiple GPUs on one machine, run multiple daemons:"
echo "   CUDA_VISIBLE_DEVICES=0 python distributed/worker_daemon.py --config config.json --gpu 0 &"
echo "   CUDA_VISIBLE_DEVICES=1 python distributed/worker_daemon.py --config config.json --gpu 1 &"
echo ""
echo "6. Monitor worker status:"
echo "   python distributed/health_dashboard.py --config config.json"
