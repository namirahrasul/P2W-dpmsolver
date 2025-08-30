import subprocess
import sys
#download mpi with conda install -n p2wd mpi4py openmpi -y
def install(package_args):
    """Run a pip install command with support for extra arguments."""
    # Base command
    command = [sys.executable, "-m", "pip", "install"]
    # Split package_args into a list if it contains spaces (e.g., for --extra-index-url)
    args = package_args.split()
    # Extend command with all arguments
    command.extend(args)
    try:
        subprocess.check_call(command)
        print(f"Successfully installed: {' '.join(args)}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {' '.join(args)}: {e}")
        sys.exit(1)

# Install PyTorch and torchvision together to enforce consistency
install("torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121")

# Install other packages
packages = [
    "PyYAML",
    "tqdm",
    "scipy",
    "pytorch_fid",
    "blobfile",
    "numpy<2",
    "tensorboard",
    "lmdb",
    "pillow",
    "lpips",
    "torchmetrics", 
    "click",
    "psutil",
    "requests",
    "imageio",
    "imageio-ffmpeg",
    "pyspng",
    "omegaconf",
    "pytorch_lightning",  # Pin to a version compatible with PyTorch 1.13.1
    "einops",
    "taming-transformers",
    "transformers==4.28.0",
    "safetensors==0.4.3"
]

for package in packages:
    install(package)

# Verify key installations
# print("Verifying installations...")
# try:
#     import torch
#     import torchvision
#     print(f"Python version: {sys.version}")
#     print(f"PyTorch version: {torch.__version__}")
#     print(f"Torchvision version: {torchvision.__version__}")
#     print(f"CUDA available: {torch.cuda.is_available()}")
# except ImportError as e:
#     print(f"Verification failed: {e}")
#     sys.exit(1)'''
