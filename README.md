# Single-Image Blind Motion Deblurring: Model Comparison and Analysis

## Overview
This repository contains the code, data, and results for the project "Single-Image Blind Motion Deblurring: Model Comparison and Analysis." The project evaluates four state-of-the-art motion deblurring models using various evaluation metrics over multiple datasets.

## Models Evaluated
1. **MAXIM**: Multi-Axis MLP for Image Processing.
2. **Restormer**: Efficient Transformer for High-Resolution Image Restoration.
3. **NAFNet**: Simple Baselines for Image Restoration.
4. **LaKDNet**: Revisiting Image Deblurring with an Efficient ConvNet.

## Reach the Pre-Trained Models
- [MAXIM](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Deblurring/GoPro)
- [Restormer](https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth)
- [NAFNet](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view)
- [LaKDNet](https://lakdnet.mpi-inf.mpg.de/)

## File Organization
Below is the organization of files within this repository:
- **dataset/**: Images to be deblurred
- **NAFNet (To be Cloned)**: This directory contains the code for the NAFNet model. To obtain it, clone the [NAFNet repository](https://github.com/megvii-research/NAFNet).
- **NafNetModel/**: This directory contains the pre-trained model file for the NAFNet model.
  - `NAFNet-GoPro-width64.pth`: Pretrained weights file for the NAFNet model.
- **Restormer (To be Cloned)**: This directory contains the code for the Restormer Model. To obtain it, clone the [Restormer repository](https://github.com/swz30/Restormer).
- **RestormerModel/**:
  - `motion_deblurring.pth`: Restormer pre-trained weights.
- **maxim (To be Cloned)**: This directory contains the code for the MAXIM Model. To obtain it, clone the [MAXIM repository](https://github.com/google-research/maxim).
- **maxim_model/**:
  - `adobe.npz`: MAXIM pre-trained weights.
- **LaKDNet (To be Cloned)**: This directory contains the code for the LaKDNet Model. To obtain it, clone the [LaKDNet repository](https://github.com/lingyanruan/LaKDNet).
  - **ckpts/Motion/train_on_gopro_s/train_on_gopro_s.pth**: Pretrained model.

- **exp_maxim.py**
- **exp_restormer.py**
- **exp_nafnet.py**
- **expt_lakdnet.py**
- **evaluate.py**: Calculate PSNR, SSIM, LPIPS, and DISTS.
- **FinalReport.pdf**: Read our final report for further information
