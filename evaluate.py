import os
from glob import glob
from natsort import natsorted
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from skimage.io import imread

# Directories
dataset_dir = "dataset"
output_dirs = [
    "lakdnet_outputs",
    "restormer_outputs",
    "nafnet_outputs",
    "maxim_outputs",
]


def calculate_metrics(gt_img, pred_img):
    psnr = compute_psnr(gt_img, pred_img, data_range=gt_img.max() - gt_img.min())
    ssim = compute_ssim(
        gt_img,
        pred_img,
        win_size=7,
        channel_axis=-1,
        data_range=gt_img.max() - gt_img.min(),
    )
    return psnr, ssim


def process_model_outputs(output_dir, gt_files):
    psnr_values = []
    ssim_values = []

    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        pred_path = os.path.join(output_dir, filename)

        if not os.path.exists(pred_path):
            print(f"Output file {pred_path} not found, skipping...")
            continue

        gt_img = imread(gt_path)
        pred_img = imread(pred_path)

        psnr, ssim = calculate_metrics(gt_img, pred_img)
        psnr_values.append(psnr)
        ssim_values.append(ssim)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    return avg_psnr, avg_ssim


def main():
    gt_files = natsorted(glob(os.path.join(dataset_dir, "*.png")))
    assert len(gt_files) > 0, "No ground truth images found in the dataset folder."

    results = {}
    for output_dir in output_dirs:
        avg_psnr, avg_ssim = process_model_outputs(output_dir, gt_files)
        results[output_dir] = {"PSNR": avg_psnr, "SSIM": avg_ssim}
        print(f"{output_dir} - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    main()
