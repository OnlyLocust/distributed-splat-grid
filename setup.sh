#!/bin/bash
set -e
mkdir -p ~/projects/gaussian-splat/data/images
cd ~/projects/gaussian-splat

cp /mnt/d/Development/Projects/DC50/distributed-gaussian-splat-rendering/requirements.txt requirements.txt
cp /mnt/d/Development/Projects/DC50/distributed-gaussian-splat-rendering/smoke_test.py smoke_test.py
cp /mnt/d/Development/Projects/DC50/distributed-gaussian-splat-rendering/validate_colmap.py validate_colmap.py
cp /mnt/d/Development/Projects/DC50/distributed-gaussian-splat-rendering/train_single_node.py train_single_node.py

if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
fi

export PATH="$HOME/miniconda3/bin:$PATH"

if ! conda env list | grep -q 'gaussian-splat'; then
    echo "Creating conda environment..."
    conda create -n gaussian-splat python=3.10 -y
fi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gaussian-splat

echo "Installing torch with CUDA 12.1..."
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
echo "Installing other dependencies..."
pip install gsplat==1.3.0 open3d Pillow numpy tqdm

echo "Installing COLMAP..."
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt install -y colmap || true

echo "Setup complete."
