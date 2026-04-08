import torch
import gsplat
import open3d

print(f"PyTorch version: {torch.__version__}")
print(f"gsplat version: {gsplat.__version__}")
print(f"open3d version: {open3d.__version__}")

assert torch.cuda.is_available(), "CUDA is not available! PyTorch installed CPU-only version."
device_name = torch.cuda.get_device_name(0)
print(f"CUDA Device: {device_name}")
assert "RTX 3050" in device_name, f"Device name {device_name} does not match RTX 3050!"

print("Smoke test passed successfully!")
