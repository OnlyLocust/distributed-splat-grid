"""
Unified File I/O Abstraction for Distributed 3DGS Pipeline
Supports both NFS and S3 backends with identical interface
"""

import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import logging

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("Warning: boto3 not available. S3 backend will not work.")


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def read(self, relative_path: str) -> bytes:
        """Read file content from storage."""
        pass
    
    @abstractmethod
    def write(self, relative_path: str, data: bytes) -> None:
        """Write file content to storage."""
        pass
    
    @abstractmethod
    def exists(self, relative_path: str) -> bool:
        """Check if file exists in storage."""
        pass
    
    @abstractmethod
    def list_prefix(self, prefix: str) -> List[str]:
        """List all files with given prefix."""
        pass
    
    @abstractmethod
    def get_full_path(self, relative_path: str) -> str:
        """Get full local path for file access."""
        pass
    
    @abstractmethod
    def delete(self, relative_path: str) -> None:
        """Delete file from storage."""
        pass


class NFSStorage(StorageBackend):
    """NFS storage backend for local network shared storage."""
    
    def __init__(self, base_path: str):
        """
        Initialize NFS storage.
        
        Args:
            base_path: Base path for shared storage (e.g., "/mnt/3dgs_share")
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"NFS base path does not exist: {base_path}")
        
        logging.info(f"NFS storage initialized with base path: {base_path}")
    
    def read(self, relative_path: str) -> bytes:
        """Read file content from NFS storage."""
        full_path = self.base_path / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")
        
        with open(full_path, 'rb') as f:
            return f.read()
    
    def write(self, relative_path: str, data: bytes) -> None:
        """Write file content to NFS storage."""
        full_path = self.base_path / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'wb') as f:
            f.write(data)
    
    def exists(self, relative_path: str) -> bool:
        """Check if file exists in NFS storage."""
        full_path = self.base_path / relative_path
        return full_path.exists()
    
    def list_prefix(self, prefix: str) -> List[str]:
        """List all files with given prefix in NFS storage."""
        search_path = self.base_path / prefix
        if not search_path.exists():
            return []
        
        if search_path.is_file():
            return [prefix]
        
        # Recursively find all files
        files = []
        for file_path in search_path.rglob('*'):
            if file_path.is_file():
                relative = file_path.relative_to(self.base_path)
                files.append(str(relative))
        
        return sorted(files)
    
    def get_full_path(self, relative_path: str) -> str:
        """Get full local path for file access."""
        return str(self.base_path / relative_path)
    
    def delete(self, relative_path: str) -> None:
        """Delete file from NFS storage."""
        full_path = self.base_path / relative_path
        if full_path.exists():
            full_path.unlink()


class S3Storage(StorageBackend):
    """S3 storage backend for cloud object storage."""
    
    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1", 
                 endpoint_url: Optional[str] = None, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None):
        """
        Initialize S3 storage.
        
        Args:
            bucket: S3 bucket name
            prefix: Prefix within bucket (e.g., "3dgs/")
            region: AWS region
            endpoint_url: Custom endpoint URL (for MinIO, etc.)
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 storage backend")
        
        self.bucket = bucket
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''
        self.region = region
        
        # Initialize S3 client
        s3_config = {'region_name': region}
        if endpoint_url:
            s3_config['endpoint_url'] = endpoint_url
        if aws_access_key_id and aws_secret_access_key:
            s3_config['aws_access_key_id'] = aws_access_key_id
            s3_config['aws_secret_access_key'] = aws_secret_access_key
        
        try:
            self.s3_client = boto3.client('s3', **s3_config)
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket)
            logging.info(f"S3 storage initialized: bucket={bucket}, prefix={prefix}")
        except (ClientError, NoCredentialsError) as e:
            raise ConnectionError(f"Failed to connect to S3: {e}")
    
    def _get_s3_key(self, relative_path: str) -> str:
        """Get S3 key for relative path."""
        return self.prefix + relative_path
    
    def read(self, relative_path: str) -> bytes:
        """Read file content from S3 storage."""
        s3_key = self._get_s3_key(relative_path)
        
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"File not found in S3: {relative_path}")
            raise
    
    def write(self, relative_path: str, data: bytes) -> None:
        """Write file content to S3 storage."""
        s3_key = self._get_s3_key(relative_path)
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=data
            )
        except ClientError as e:
            raise IOError(f"Failed to write to S3: {e}")
    
    def exists(self, relative_path: str) -> bool:
        """Check if file exists in S3 storage."""
        s3_key = self._get_s3_key(relative_path)
        
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    def list_prefix(self, prefix: str) -> List[str]:
        """List all files with given prefix in S3 storage."""
        s3_prefix = self._get_s3_key(prefix)
        files = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Remove prefix to get relative path
                        relative_key = obj['Key'][len(self.prefix):]
                        files.append(relative_key)
        except ClientError as e:
            logging.warning(f"Failed to list S3 prefix {prefix}: {e}")
        
        return sorted(files)
    
    def get_full_path(self, relative_path: str) -> str:
        """
        Get full local path for file access.
        
        Note: S3 backend requires downloading files locally before use.
        Use download_to_local() method first.
        """
        raise NotImplementedError(
            "S3 backend requires workers to download files locally before use. "
            "Call download_to_local(rel_path, local_path) first."
        )
    
    def download_to_local(self, relative_path: str, local_path: str) -> None:
        """Download file from S3 to local path."""
        s3_key = self._get_s3_key(relative_path)
        
        # Create local directory if needed
        local_file = Path(local_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.s3_client.download_file(self.bucket, s3_key, local_path)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"File not found in S3: {relative_path}")
            raise IOError(f"Failed to download from S3: {e}")
    
    def upload_from_local(self, local_path: str, relative_path: str) -> None:
        """Upload local file to S3."""
        s3_key = self._get_s3_key(relative_path)
        
        if not Path(local_path).exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        try:
            self.s3_client.upload_file(local_path, self.bucket, s3_key)
        except ClientError as e:
            raise IOError(f"Failed to upload to S3: {e}")
    
    def delete(self, relative_path: str) -> None:
        """Delete file from S3 storage."""
        s3_key = self._get_s3_key(relative_path)
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
        except ClientError as e:
            logging.warning(f"Failed to delete S3 file {relative_path}: {e}")


def get_storage(config: Dict[str, Any]) -> StorageBackend:
    """
    Factory function to create storage backend from config.
    
    Args:
        config: Configuration dictionary with storage settings
        
    Returns:
        StorageBackend instance
    """
    backend_type = config.get("backend", "nfs").lower()
    
    if backend_type == "nfs":
        base_path = config.get("base_path")
        if not base_path:
            raise ValueError("NFS backend requires 'base_path' in config")
        return NFSStorage(base_path)
    
    elif backend_type == "s3":
        bucket = config.get("bucket")
        if not bucket:
            raise ValueError("S3 backend requires 'bucket' in config")
        
        return S3Storage(
            bucket=bucket,
            prefix=config.get("prefix", ""),
            region=config.get("region", "us-east-1"),
            endpoint_url=config.get("endpoint_url"),
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key")
        )
    
    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


# Utility functions for common operations
def read_json(storage: StorageBackend, relative_path: str) -> Dict[str, Any]:
    """Read JSON file from storage."""
    data = storage.read(relative_path)
    return json.loads(data.decode('utf-8'))


def write_json(storage: StorageBackend, relative_path: str, data: Dict[str, Any]) -> None:
    """Write JSON file to storage."""
    json_data = json.dumps(data, indent=2)
    storage.write(relative_path, json_data.encode('utf-8'))


def read_numpy(storage: StorageBackend, relative_path: str) -> 'np.ndarray':
    """Read numpy array from storage."""
    import numpy as np
    data = storage.read(relative_path)
    return np.load(data)


def write_numpy(storage: StorageBackend, relative_path: str, array: 'np.ndarray') -> None:
    """Write numpy array to storage."""
    import numpy as np
    import io
    
    # Save to buffer
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=array)
    buffer.seek(0)
    
    storage.write(relative_path, buffer.getvalue())


if __name__ == "__main__":
    # Test storage backends
    import tempfile
    
    # Test NFS backend
    with tempfile.TemporaryDirectory() as tmp_dir:
        nfs_config = {"backend": "nfs", "base_path": tmp_dir}
        storage = get_storage(nfs_config)
        
        # Test write/read
        test_data = b"Hello, NFS!"
        storage.write("test.txt", test_data)
        assert storage.read("test.txt") == test_data
        assert storage.exists("test.txt")
        
        # Test JSON
        test_json = {"key": "value", "number": 42}
        write_json(storage, "test.json", test_json)
        loaded_json = read_json(storage, "test.json")
        assert loaded_json == test_json
        
        print("NFS storage test passed!")
    
    # Test S3 backend (if credentials available)
    try:
        s3_config = {
            "backend": "s3",
            "bucket": "test-bucket",
            "prefix": "test/",
            "region": "us-east-1"
        }
        storage = get_storage(s3_config)
        print("S3 storage test passed!")
    except Exception as e:
        print(f"S3 storage test skipped: {e}")
