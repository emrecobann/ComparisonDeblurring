import os
import cv2
import torch
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from runpy import run_path
from natsort import natsorted
import torch.nn.functional as F
from skimage import img_as_ubyte
import torchvision.transforms.functional as TF

## Prepare Model
def get_weights_and_parameters(parameters):
    weights = os.path.join(
            "RestormerModel", "pretrained_models", "motion_deblurring.pth"
    )

    return weights, parameters


# Get model weights and parameters
parameters = {
    "inp_channels": 3,
    "out_channels": 3,
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    "ffn_expansion_factor": 2.66,
    "bias": False,
    "LayerNorm_type": "WithBias",
    "dual_pixel_task": False,
}

# Load the Restormer model as pre-trained 
weights, parameters = get_weights_and_parameters(parameters)

load_arch = run_path(os.path.join("Restormer","basicsr", "models", "archs", "restormer_arch.py"))
model = load_arch["Restormer"](**parameters)
model.cuda()

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint["params"])
model.eval()

# Create a directory for dataset
input_dir = "dataset"
out_dir = "restormer_outputs"
os.makedirs(out_dir, exist_ok=True)
extensions = ["jpg", "JPG", "png", "PNG", "jpeg", "JPEG", "bmp", "BMP"]
files = natsorted(glob(os.path.join(input_dir, "*")))

img_multiple_of = 8

# Run inference over the dataset
with torch.no_grad():
    for filepath in tqdm(files):
        # print(file_)
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        input_ = (
            torch.from_numpy(img)
            .float()
            .div(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .cuda()
        )

        # Pad the input if not_multiple_of 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, (
            (w + img_multiple_of) // img_multiple_of
        ) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), "reflect")

        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:, :, :h, :w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        filename = os.path.split(filepath)[-1]
        cv2.imwrite(
            os.path.join(out_dir, filename), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        )
