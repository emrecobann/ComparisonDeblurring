import os
from glob import glob
from natsort import natsorted
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from skimage.io import imread
import torch
import lpips
from DISTS_pytorch import DISTS

# Directories
dataset_dir = "dataset"
output_dirs = [
    "lakdnet_outputs",
    "restormer_outputs",
    "nafnet_outputs",
    "maxim_outputs",
]


# Initialize LPIPS model
lpips_fn = lpips.LPIPS(net="alex")  # You can use 'alex', 'vgg', or 'squeeze'

# Initialize DISTS model
dists_fn = DISTS()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips_fn.to(device)
dists_fn = dists_fn.to(device)


def calculate_metrics(gt_img, pred_img):
    psnr = compute_psnr(gt_img, pred_img, data_range=gt_img.max() - gt_img.min())
    ssim = compute_ssim(
        gt_img,
        pred_img,
        win_size=7,
        channel_axis=-1,
        data_range=gt_img.max() - gt_img.min(),
    )

    # Convert images to tensor and normalize to [-1, 1] for LPIPS
    gt_img_tensor = (
        torch.tensor(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    )
    pred_img_tensor = (
        torch.tensor(pred_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    )

    # Convert images to tensor and normalize to [0, 1] for DISTS
    gt_img_dists = torch.tensor(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    pred_img_dists = (
        torch.tensor(pred_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )

    # Move tensors to the appropriate device
    gt_img_tensor = gt_img_tensor.to(device)
    pred_img_tensor = pred_img_tensor.to(device)
    gt_img_dists = gt_img_dists.to(device)
    pred_img_dists = pred_img_dists.to(device)

    lpips_value = lpips_fn(gt_img_tensor, pred_img_tensor).item()
    dists_value = dists_fn(gt_img_dists, pred_img_dists).item()

    return psnr, ssim, lpips_value, dists_value


def process_model_outputs(output_dir, gt_files):
    psnr_values = []
    ssim_values = []
    lpips_values = []
    dists_values = []

    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        pred_path = os.path.join(output_dir, filename)

        if not os.path.exists(pred_path):
            print(f"Output file {pred_path} not found, skipping...")
            continue

        gt_img = imread(gt_path)
        pred_img = imread(pred_path)

        psnr, ssim, lpips_value, dists_value = calculate_metrics(gt_img, pred_img)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips_value)
        dists_values.append(dists_value)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    avg_dists = np.mean(dists_values)
    return avg_psnr, avg_ssim, avg_lpips, avg_dists


def main():
    gt_files = natsorted(glob(os.path.join(dataset_dir, "*.png")))
    assert len(gt_files) > 0, "No ground truth images found in the dataset folder."

    results = {}
    for output_dir in output_dirs:
        avg_psnr, avg_ssim, avg_lpips, avg_dists = process_model_outputs(
            output_dir, gt_files
        )
        results[output_dir] = {
            "PSNR": avg_psnr,
            "SSIM": avg_ssim,
            "LPIPS": avg_lpips,
            "DISTS": avg_dists,
        }
        print(
            f"{output_dir} - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}, DISTS: {avg_dists:.4f}"
        )


if __name__ == "__main__":
    main()
