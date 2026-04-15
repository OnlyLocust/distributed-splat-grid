# Distributed 3D Gaussian Splatting Pipeline

A distributed version of the 3D Gaussian Splatting pipeline that scales across multiple machines with RTX 3050 GPUs. This architecture uses Redis Queue for job dispatch and shared storage (NFS or S3) for data coordination.

## Architecture Overview

```
MASTER NODE                          WORKER NODES
+-----------------+                 +-----------------+
| splitter.py     |                 | worker_daemon.py |
| optimizer.py    |                 | (GPU training)  |
| orchestrator.py |                 |                 |
| stitcher.py     |                 |                 |
| Redis Server    | <--- Redis ---> | Redis Client    |
| Shared Storage  | <--- NFS/S3 ---> | Shared Storage  |
+-----------------+                 +-----------------+
```

## Key Features

- **Scalable Training**: Add/remove workers dynamically
- **Fault Tolerance**: Automatic job recovery from dead workers
- **4GB VRAM Optimized**: Each worker handles 50K point chunks
- **Progress Monitoring**: Real-time dashboard with live updates
- **Storage Flexibility**: NFS for LAN, S3/MinIO for cloud
- **No Code Changes**: Core training logic unchanged

## System Requirements

### Master Node
- Python 3.10+
- Redis server
- NFS server or S3 access
- 8GB+ RAM recommended
- Network storage

### Worker Nodes
- Python 3.10+
- CUDA GPU with 4GB+ VRAM (RTX 3050+)
- Redis client
- NFS client or S3 access
- Network access to master

## Quick Start

### 1. Master Node Setup

```bash
# Install dependencies
bash distributed/setup/install_master.sh

# Update configuration
cp config.json.example config.json
# Edit config.json with your settings

# Set up shared storage (choose one)
# Option A: NFS (see distributed/setup/nfs_setup.md)
# Option B: S3/MinIO (see distributed/setup/nfs_setup.md)
```

### 2. Worker Node Setup

```bash
# Install dependencies on each worker
bash distributed/setup/install_worker.sh

# Copy configuration from master
scp user@master_ip:/path/to/config.json .

# Mount shared storage (NFS only)
sudo mount -t nfs master_ip:/mnt/3dgs_share /mnt/3dgs_share

# Add to fstab for automatic mounting
echo "master_ip:/mnt/3dgs_share  /mnt/3dgs_share  nfs  defaults,_netdev  0  0" | sudo tee -a /etc/fstab
```

### 3. Start Workers

```bash
# Start one worker daemon per GPU
CUDA_VISIBLE_DEVICES=0 python distributed/worker_daemon.py --config config.json --gpu 0

# For multiple GPUs on one machine:
CUDA_VISIBLE_DEVICES=0 python distributed/worker_daemon.py --config config.json --gpu 0 &
CUDA_VISIBLE_DEVICES=1 python distributed/worker_daemon.py --config config.json --gpu 1 &
```

### 4. Run Pipeline

```bash
# On master node, run the complete pipeline
python distributed/master_orchestrator.py --config config.json --colmap_dir ./sparse/0

# Monitor progress
python distributed/health_dashboard.py --config config.json
```

## Configuration

### Configuration File (config.json)

```json
{
  "storage": {
    "backend": "nfs",  // "nfs" or "s3"
    "base_path": "/mnt/3dgs_share"  // NFS only
  },
  "redis": {
    "host": "192.168.1.100",
    "port": 6379,
    "password": "your_redis_password",
    "queue_name": "3dgs_chunks"
  },
  "worker": {
    "max_gaussians": 200000,
    "num_iters": 3000,
    "densify_interval": 100,
    "max_oom_retries": 3
  }
}
```

### S3 Configuration Example

```json
{
  "storage": {
    "backend": "s3",
    "bucket": "my-3dgs-bucket",
    "prefix": "training/",
    "region": "us-east-1",
    "endpoint_url": "https://s3.amazonaws.com",
    "aws_access_key_id": "your_access_key",
    "aws_secret_access_key": "your_secret_key"
  }
}
```

## Pipeline Stages

### Stage 1: Data Splitting (Master)
```bash
python splitter.py --input ./sparse/0 --output /mnt/3dgs_share/tasks
```
- Splits COLMAP data into 50K point chunks
- Creates overlapping halo margins
- Generates grid manifest

### Stage 2: Camera Culling (Master)
```bash
python optimizer.py --colmap_dir ./sparse/0 --tasks_dir /mnt/3dgs_share/tasks --images_dir /mnt/3dgs_share/images
```
- Filters cameras per chunk using frustum culling
- Reduces training workload by 30-70%
- Saves camera lists per chunk

### Stage 3: Distributed Training (Master + Workers)
```bash
python distributed/master_orchestrator.py --config config.json --colmap_dir ./sparse/0
```
- Enqueues chunk jobs to Redis
- Workers process chunks independently
- Automatic OOM recovery with reduced settings

### Stage 4: Result Stitching (Master)
```bash
python stitcher.py --results_dir /mnt/3dgs_share/results --output /mnt/3dgs_share/final_output.ply
```
- Merges all chunk PLY files
- Deduplicates overlapping Gaussians
- Creates final 3DGS model

## Monitoring

### Health Dashboard
```bash
python distributed/health_dashboard.py --config config.json
```

Shows:
- **Worker Status**: Online/offline, current task, progress
- **Queue Stats**: Pending, processing, completed, failed jobs
- **Recent Completions**: Latest finished chunks with performance
- **ETA**: Estimated completion time

### Worker Logs
```bash
# View worker logs on shared storage
tail -f /mnt/3dgs_share/worker_logs/hostname_gpu0.log
```

### Redis Monitoring
```bash
# Check queue status
redis-cli -a your_password llen 3dgs_chunks

# Monitor worker heartbeats
redis-cli -a your_password keys "workers:heartbeat:*"
```

## Fault Tolerance

### Automatic Recovery
- **Dead Workers**: Heartbeat monitor detects and requeues jobs
- **OOM Errors**: Automatic retry with 50% resources, max 3 attempts
- **Network Issues**: Workers reconnect and resume processing
- **Master Crash**: Resume with `--resume` flag

### Manual Recovery
```bash
# Resume interrupted pipeline
python distributed/master_orchestrator.py --config config.json --colmap_dir ./sparse/0 --resume

# Requeue failed chunks manually
redis-cli -a your_password del workers:heartbeat:dead_worker_hostname
```

## Performance Optimization

### Network Optimization
- Use 1Gbps+ network for NFS
- Place master and workers on same switch if possible
- Monitor network bandwidth during training

### Storage Optimization
- Use SSD storage on master node
- Enable NFS caching on workers
- Consider S3 for remote workers

### GPU Utilization
- One worker daemon per GPU
- Monitor GPU memory usage
- Adjust chunk size if OOM occurs frequently

## Scaling Guidelines

### Adding Workers
1. Install worker dependencies
2. Mount shared storage
3. Copy configuration file
4. Start worker daemon(s)
5. Workers automatically pick up next jobs

### Cluster Size
- **Small**: 2-4 workers for testing
- **Medium**: 8-16 workers for production
- **Large**: 32+ workers for massive scenes

### Bottlenecks
- **Network**: Slow storage access
- **Master**: CPU/memory for orchestration
- **Storage**: I/O bandwidth for large datasets

## Troubleshooting

### Common Issues

**Workers can't connect to Redis**
```bash
# Test Redis connection from worker
redis-cli -h master_ip -p 6379 -a your_password ping

# Check firewall on master
sudo ufw status
```

**NFS mount fails**
```bash
# Check NFS server exports
sudo showmount -e master_ip

# Test mount manually
sudo mount -t nfs master_ip:/mnt/3dgs_share /mnt/3dgs_share -v
```

**Workers show OOM errors**
- Reduce `max_gaussians` in config
- Check GPU memory with `nvidia-smi`
- Monitor memory usage in logs

**Jobs stuck in queue**
- Check worker heartbeats
- Restart dead workers
- Manual requeue with orchestrator --resume

### Debug Mode
```bash
# Enable verbose logging
python distributed/worker_daemon.py --config config.json --gpu 0 --verbose

# Test single chunk
python worker.py --chunk_dir /mnt/3dgs_share/tasks/chunk_0_0 --output_dir /tmp/test --num_iterations 100
```

## Security Considerations

### Network Security
- Use firewall to restrict Redis and NFS access
- Consider VPN for remote workers
- Monitor network traffic

### Access Control
- Use strong Redis passwords
- Implement IAM policies for S3
- Regular security updates

### Data Protection
- Backup shared storage regularly
- Enable S3 versioning if using S3
- Monitor access logs

## Migration from Single-Node

To migrate from the single-node pipeline:

1. **Backup Data**
   ```bash
   cp -r tasks/ results/ /backup/
   ```

2. **Setup Distributed Environment**
   - Install Redis and shared storage
   - Deploy configuration to all nodes

3. **Transfer Data**
   ```bash
   # Copy to shared storage
   cp -r tasks/ results/ /mnt/3dgs_share/
   ```

4. **Start Distributed Pipeline**
   - Workers will process existing chunks
   - Use `--resume` to skip completed work

## Advanced Usage

### Custom Storage Backends
Implement `StorageBackend` class in `shared_storage.py` for custom storage solutions.

### Custom Task Types
Add new task functions to `rq_tasks.py` for specialized processing.

### Performance Monitoring
Integrate with Prometheus/Grafana for metrics collection.

## Support

### Logs and Debugging
- Master logs: `pipeline.log`
- Worker logs: `worker_logs/hostname_gpuX.log`
- Redis logs: `/var/log/redis/redis-server.log`

### Performance Tuning
- Monitor GPU utilization
- Track network bandwidth
- Analyze job completion times

### Community
- Report issues on GitHub
- Share configurations and performance results
- Contribute improvements and features

## License

This distributed implementation maintains the same license as the original 3DGS pipeline. Ensure compliance with all dependencies when deploying in production environments.
