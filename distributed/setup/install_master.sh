#!/bin/bash
# Master Node Installation Script for Distributed 3DGS Pipeline
# Run this on the master node once

set -e

echo "=== Installing 3DGS Master Node Dependencies ==="

# Update system packages
echo "Updating system packages..."
sudo apt-get update

# Install Python dependencies
echo "Installing Python packages..."
pip install redis rq rq-dashboard rich boto3 plyfile

# Install Redis server
echo "Installing Redis server..."
sudo apt-get install -y redis-server

# Configure Redis for network access
echo "Configuring Redis for network access..."

# Backup original config
sudo cp /etc/redis/redis.conf /etc/redis/redis.conf.backup

# Configure Redis to bind to all interfaces and set password
sudo sed -i 's/^bind 127.0.0.1/bind 0.0.0.0/' /etc/redis/redis.conf

# Set a secure password (replace with your own)
REDIS_PASSWORD="your_redis_password_here"
sudo sed -i "s/^# requirepass .*/requirepass $REDIS_PASSWORD/" /etc/redis/redis.conf

# Enable Redis service
sudo systemctl enable redis-server
sudo systemctl restart redis-server

# Test Redis connection
echo "Testing Redis connection..."
if redis-cli -a $REDIS_PASSWORD ping > /dev/null 2>&1; then
    echo "Redis is running successfully!"
else
    echo "Redis connection failed. Please check the configuration."
    exit 1
fi

# Create shared storage directory
echo "Creating shared storage directory..."
SHARED_DIR="/mnt/3dgs_share"
sudo mkdir -p $SHARED_DIR
sudo chmod 755 $SHARED_DIR

# Create configuration template
echo "Creating configuration template..."
cat > config.json << EOF
{
  "storage": {
    "backend": "nfs",
    "base_path": "/mnt/3dgs_share"
  },
  "redis": {
    "host": "$(hostname -I | awk '{print $1}')",
    "port": 6379,
    "password": "$REDIS_PASSWORD",
    "queue_name": "3dgs_chunks"
  },
  "worker": {
    "max_gaussians": 200000,
    "num_iters": 3000,
    "densify_interval": 100,
    "max_oom_retries": 3
  }
}
EOF

echo "Configuration template created: config.json"
echo "Please update the Redis host IP if needed and distribute this config to all workers."

# Install NFS server if using NFS storage
echo "Installing NFS server..."
sudo apt-get install -y nfs-kernel-server

# Configure NFS exports
echo "Configuring NFS exports..."
echo "$SHARED_DIR *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports

# Export NFS shares
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server

echo "=== Master Node Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Update config.json with your preferred settings"
echo "2. Set up NFS shares on worker nodes (see nfs_setup.md)"
echo "3. Distribute config.json to all worker nodes"
echo "4. Start worker daemons on each worker node"
echo "5. Run: python distributed/master_orchestrator.py --config config.json --colmap_dir /path/to/colmap"
