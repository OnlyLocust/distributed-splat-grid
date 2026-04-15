# NFS Shared Storage Setup Guide

This guide covers setting up NFS shared storage for the distributed 3DGS pipeline.

## Option A: NFS Setup (Recommended for Local Network)

### ON MASTER NODE (Server)

1. **Install NFS Server**
   ```bash
   sudo apt-get update
   sudo apt-get install -y nfs-kernel-server
   ```

2. **Create Shared Directory**
   ```bash
   sudo mkdir -p /mnt/3dgs_share
   sudo chmod 755 /mnt/3dgs_share
   ```

3. **Configure NFS Exports**
   
   Edit `/etc/exports` and add:
   ```
   /mnt/3dgs_share *(rw,sync,no_subtree_check,no_root_squash)
   ```

4. **Export and Start NFS Service**
   ```bash
   sudo exportfs -ra
   sudo systemctl restart nfs-kernel-server
   sudo systemctl enable nfs-kernel-server
   ```

5. **Verify NFS Server**
   ```bash
   sudo showmount -e localhost
   ```

### ON EACH WORKER NODE (Client)

1. **Install NFS Client**
   ```bash
   sudo apt-get install -y nfs-common
   ```

2. **Create Mount Point**
   ```bash
   sudo mkdir -p /mnt/3dgs_share
   ```

3. **Mount NFS Share**
   ```bash
   sudo mount -t nfs <MASTER_IP>:/mnt/3dgs_share /mnt/3dgs_share
   ```

4. **Add to fstab for Automatic Mount**
   
   Edit `/etc/fstab` and add:
   ```
   <MASTER_IP>:/mnt/3dgs_share  /mnt/3dgs_share  nfs  defaults,_netdev  0  0
   ```

5. **Test Mount**
   ```bash
   ls /mnt/3dgs_share
   ```

### Network Configuration

1. **Firewall Settings** (if enabled)
   ```bash
   # Allow NFS ports
   sudo ufw allow 2049
   sudo ufw allow 111
   sudo ufw allow from <WORKER_IP> to any port 1024:65535 proto tcp
   ```

2. **Network Performance** (optional but recommended)
   
   On master node, add to `/etc/sysctl.conf`:
   ```
   # NFS performance tuning
   net.core.rmem_max = 16777216
   net.core.wmem_max = 16777216
   net.ipv4.tcp_rmem = 4096 87380 16777216
   net.ipv4.tcp_wmem = 4096 65536 16777216
   ```
   
   Apply changes:
   ```bash
   sudo sysctl -p
   ```

## Option B: S3/MinIO Setup (Alternative for Cloud/WAN)

### Using MinIO (Self-Hosted S3)

1. **Install Docker** (if not already installed)
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

2. **Run MinIO Container**
   ```bash
   docker run -d \
     --name minio \
     -p 9000:9000 \
     -p 9001:9001 \
     -v /mnt/minio_data:/data \
     -e MINIO_ROOT_USER=minioadmin \
     -e MINIO_ROOT_PASSWORD=minioadmin123 \
     minio/minio server /data --console-address ":9001"
   ```

3. **Create Bucket**
   ```bash
   # Install MinIO client
   curl https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc
   chmod +x /usr/local/bin/mc
   
   # Configure and create bucket
   mc alias set local http://<MASTER_IP>:9000 minioadmin minioadmin123
   mc mb local/3dgs
   ```

4. **Update Configuration**
   
   Update `config.json` to use S3 backend:
   ```json
   {
     "storage": {
       "backend": "s3",
       "bucket": "3dgs",
       "prefix": "",
       "region": "us-east-1",
       "endpoint_url": "http://<MASTER_IP>:9000",
       "aws_access_key_id": "minioadmin",
       "aws_secret_access_key": "minioadmin123"
     }
   }
   ```

### Using AWS S3

1. **Create S3 Bucket**
   ```bash
   aws s3 mb s3://your-3dgs-bucket --region us-east-1
   ```

2. **Configure IAM User**
   - Create IAM user with S3 access
   - Generate access keys
   - Apply bucket policy for cross-account access if needed

3. **Update Configuration**
   ```json
   {
     "storage": {
       "backend": "s3",
       "bucket": "your-3dgs-bucket",
       "prefix": "3dgs/",
       "region": "us-east-1",
       "aws_access_key_id": "YOUR_ACCESS_KEY",
       "aws_secret_access_key": "YOUR_SECRET_KEY"
     }
   }
   ```

## Performance Considerations

### NFS Performance

1. **Network Requirements**
   - Minimum 1Gbps network recommended
   - Low latency (<5ms) for best performance
   - Consider dedicated network switch for large clusters

2. **Storage Performance**
   - Use SSD storage on master node for better I/O
   - Ensure sufficient RAM for NFS caching
   - Monitor network bandwidth during training

3. **Common Issues**
   - Slow image loading over NFS
   - Stale file handles (restart NFS service)
   - Permission issues (check UID/GID consistency)

### S3 Performance

1. **Network Requirements**
   - Internet connection with good upload speed
   - Consider CDN for multi-region deployments

2. **Cost Optimization**
   - Use lifecycle policies for old data
   - Enable S3 transfer acceleration
   - Monitor data transfer costs

3. **Worker Configuration**
   - Workers download chunks locally to `/tmp`
   - Automatic cleanup of temporary files
   - Parallel downloads for multiple images

## Security Considerations

### NFS Security

1. **Network Security**
   - Use firewall to restrict NFS access
   - Consider VPN for remote workers
   - Monitor NFS logs for suspicious activity

2. **File Permissions**
   - Ensure consistent UID/GID across nodes
   - Use specific export options instead of `*`
   - Regular permission audits

### S3 Security

1. **Access Control**
   - Use IAM roles instead of access keys when possible
   - Rotate access keys regularly
   - Apply least privilege principle

2. **Data Protection**
   - Enable S3 versioning
   - Use S3 encryption
   - Regular backups to different regions

## Troubleshooting

### Common NFS Issues

1. **Mount Fails**
   ```bash
   # Check NFS server status
   sudo systemctl status nfs-kernel-server
   
   # Check exports
   sudo showmount -e localhost
   
   # Test connectivity
   telnet <MASTER_IP> 2049
   ```

2. **Permission Denied**
   ```bash
   # Check file permissions
   ls -la /mnt/3dgs_share
   
   # Check NFS export options
   sudo exportfs -v
   ```

3. **Slow Performance**
   ```bash
   # Check network latency
   ping <MASTER_IP>
   
   # Monitor NFS stats
   nfsstat -c
   ```

### Common S3 Issues

1. **Connection Timeout**
   - Check endpoint URL and region
   - Verify network connectivity
   - Review firewall settings

2. **Access Denied**
   - Verify credentials
   - Check bucket policies
   - Review IAM permissions

## Testing the Setup

### NFS Test
```bash
# On master
echo "test" > /mnt/3dgs_share/test_file

# On worker
cat /mnt/3dgs_share/test_file
```

### S3 Test
```bash
# Install awscli
pip install awscli

# Test upload/download
aws s3 cp test.txt s3://your-bucket/test.txt
aws s3 cp s3://your-bucket/test.txt downloaded_test.txt
```

## Migration Between Storage Backends

To switch from NFS to S3 or vice versa:

1. **Stop all workers**
2. **Export current data**
3. **Update configuration**
4. **Import data to new backend**
5. **Restart workers**

The pipeline automatically handles the storage backend based on configuration, so no code changes are required.
