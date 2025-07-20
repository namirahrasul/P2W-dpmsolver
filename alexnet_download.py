import os
import torch
import torchvision.models as models
from torchvision.models import AlexNet_Weights
import shutil
from lpips import LPIPS
import importlib.util

# Set cache directory
cache_dir = "/kaggle/working/p2w/.cache/torch/"
os.environ["TORCH_HOME"] = cache_dir
os.makedirs(os.path.join(cache_dir, "hub/checkpoints"), exist_ok=True)
# Download AlexNet weights
print("Pre-downloading AlexNet weights...")
alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
print("AlexNet weights downloaded successfully!")

# Check for weights file
weights_path = os.path.join(cache_dir, "hub/checkpoints/alexnet-owt-7be5be79.pth")
print(f"Checking for weights at: {weights_path}")
if os.path.exists(weights_path):
    print(f"Weights file found at: {weights_path}")
else:
    print("Warning: Weights file not found. Saving manually...")
    torch.save(alexnet.state_dict(), weights_path)
    if os.path.exists(weights_path):
        print(f"Weights saved to: {weights_path}")
    else:
        print("Failed to save weights.")
# Preload LPIPS to ensure weights are available (bundled in package)
print("Pre-downloading LPIPS weights...")
lpips_model = LPIPS(net="alex")
print("LPIPS weights downloaded successfully!")

# Dynamically get LPIPS weights path from installed package (alternative to __file__)
lpips_spec = importlib.util.find_spec("lpips")
if lpips_spec is not None:
    module_dir = os.path.dirname(lpips_spec.origin)
    package_weights_dir = os.path.join(module_dir, "weights", "v0.1")
    default_lpips_path = os.path.join(package_weights_dir, "alex.pth")
    print(f"LPIPS source weights path: {default_lpips_path}")
else:
    print("Error: lpips module not found.")
    default_lpips_path = None

# Copy LPIPS weights to custom persistent path
lpips_cache_dir = "/kaggle/working/p2w/.cache/lpips/"
os.makedirs(lpips_cache_dir, exist_ok=True)
custom_lpips_path = os.path.join(lpips_cache_dir, "alex.pth")  # Corrected filename to match default
if default_lpips_path and os.path.exists(default_lpips_path):
    shutil.copy(default_lpips_path, custom_lpips_path)
    print(f"LPIPS weights copied to: {custom_lpips_path}")
else:
    print("Warning: LPIPS weights not found in package path.")
