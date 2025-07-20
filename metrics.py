# === Metric Imports ===
from lpips import LPIPS
import torch
import torch.nn.functional as F
from math import log10
from torchmetrics.functional.image import structural_similarity_index_measure
from PIL import Image
from torchvision import transforms
import os
import sys
import io
import subprocess
os.environ["TORCH_HOME"] = "/kaggle/working/p2w/.cache/torch/"

# === Install FID if not installed ===
#!pip install -q pytorch-fid

# === Metric Calculation Functions ===
def check_range(img):
    assert torch.min(img) >= -1.0, "Minimum pixel value is less than -1.0"
    assert torch.max(img) <= 1.0, "Maximum pixel value is greater than 1.0"
def calculate_lpips(img1, img2):
    check_range(img1)
    check_range(img2)
    lpips_model = LPIPS(net="alex", model_path="/kaggle/working/p2w/.cache/lpips/alex.pth").to(img2.device)
    return lpips_model(img1.to(img2.device), img2).item()
def calculate_ssim(img1, img2):
    check_range(img1)
    check_range(img2)
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    return structural_similarity_index_measure(img1.to(img2.device), img2)
def calculate_psnr(img1, img2):
    check_range(img1)
    check_range(img2)
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    mse = F.mse_loss(img1, img2)
    return 20 * log10(1.0 / torch.sqrt(mse))
def calculate_l1(img1, img2):
    check_range(img1)
    check_range(img2)
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    return F.l1_loss(img1, img2).item()
def normalize_to_neg1_1(t):
    return t * 2 - 1
def avg(lst): return sum(lst) / len(lst) if lst else 0
# === Set Paths ===
gt_dir = '/kaggle/working/p2w/experiments/celebahq/400_200_time_uniform_type-dpmsolver/image_samples/results/celeba/thick/gtImg'
out_dir = '/kaggle/working/p2w/experiments/celebahq/400_200_time_uniform_type-dpmsolver/image_samples/results/celeba/thick/outImg'
output_path = "/kaggle/working/inpainting_metrics_report.txt"
# === Redirect Print to File ===
f = open(output_path, "w")
sys.stdout = io.TextIOWrapper(open(output_path, 'wb'))
# === Preprocessing ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])
lpips_model = LPIPS(net="alex", model_path="/kaggle/working/p2w/.cache/lpips/alex.pth")
lpips_scores, ssim_scores, psnr_scores, l1_scores = [], [], [], []
filenames = sorted([
    f for f in os.listdir(out_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])
print(f" Found {len(filenames)} image pairs to evaluate.\n")
for fname in filenames:
    path_out = os.path.join(out_dir, fname)
    path_gt = os.path.join(gt_dir, fname)
    if not os.path.exists(path_gt):
        print(f" Missing ground truth for: {fname}, skipping.")
        continue
    try:
        img_out = normalize_to_neg1_1(transform(Image.open(path_out).convert("RGB")).unsqueeze(0))
        img_gt = normalize_to_neg1_1(transform(Image.open(path_gt).convert("RGB")).unsqueeze(0))
        lpips_val = lpips_model(img_out.to(img_gt.device), img_gt.to(img_gt.device)).item()
        ssim_val = calculate_ssim(img_out, img_gt)
        psnr_val = calculate_psnr(img_out, img_gt)
        l1_val = calculate_l1(img_out, img_gt)
        lpips_scores.append(lpips_val)
        ssim_scores.append(ssim_val.item() if torch.is_tensor(ssim_val) else ssim_val)
        psnr_scores.append(psnr_val)
        l1_scores.append(l1_val)
    except Exception as e:
        print(f" Error on {fname}: {e}")
# === Print Final Results ===
print("\n Overall Evaluation Results:")
print(f"LPIPS : {avg(lpips_scores):.4f} ↓ (lower is better)")
print(f"SSIM  : {avg(ssim_scores):.4f} ↑ (higher is better)")
print(f"PSNR  : {avg(psnr_scores):.2f} dB ↑ (higher is better)")
print(f"L1    : {avg(l1_scores):.4f} ↓ (lower is better)")
# === FID Calculation (CLI) ===
print("\n Running FID Calculation using pytorch-fid...")
try:
    result = subprocess.run(
        ["python", "-m", "pytorch_fid", gt_dir, out_dir],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        fid_output = result.stdout.strip()
        print(f"FID   : {fid_output.splitlines()[-1]} ↓ (lower is better)")
    else:
        print(" FID calculation failed:", result.stderr.strip())
except Exception as e:
    print(f" FID subprocess error: {e}")
# === Restore Output ===
sys.stdout = sys.__stdout__
print(f"\n Report saved to: {output_path}")
