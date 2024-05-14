## PreTrained Models

- [Maxim](https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/Deblurring/GoPro)
- [Restormer](https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth)
- [NafNet](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view)

## File Organization

Below is the organization of files within this repository:

- **dataset**: Images to be deblurred
- **NAFNet (To be Cloned)**: This directory contains the code for the NAFNet model. To obtain it, clone the [NAFNet repository](https://github.com/megvii-research/NAFNet).
- **NafNetModel**: This directory contains the pre-trained model file for the NAFNet model.
  - `NAFNet-GoPro-width64.pth`: Pretrained weights file for the NAFNet model.
- **Restormer (To be Cloned)** : This directory contains the code for Restormer Model. To obtain it, clone the [Restormer repository](https://github.com/swz30/Restormer)
- **RestormerModel**:
  - `motion_deblurring.pth`: Restormer pre-trained weights
- **maxim (To be Cloned)**: This directory contains the code for Maxim Model. To obtain it, clone the [Maxim repository](https://github.com/google-research/maxim)
- **maxim_model**:
  - `adobe.npz`: Maxim pre-trained weights
- **exp_maxim.py**
- **exp_restormer.py**
- **exp_nafnet**
