import os
import torch
import torchvision.utils as vutils
from LAKDNET.LaKDNet.util.util import read_image, crop_image
from pathlib import Path
from glob import glob
from natsort import natsorted
from LAKDNET.LaKDNet.models.LaKDNet import LaKDNet
import yaml
from tqdm import tqdm

# Hardcoded paths
dataset_dir = "dataset"
output_dir = "lakdnet_outputs"
config_path = "LAKDNET/LaKDNet/options/test_configs.yml"

# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)["Motion"]


# Select the first available network configuration and weight for simplicity
net_config_key = config["net_configs"][0]
net_weight_key = config["test_status"][0]


# Network configuration
net_config = config[net_config_key]
net_weight = config["weight"][net_weight_key]


# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Initialize the network
network = LaKDNet(**net_config).cuda()
network.load_state_dict(torch.load(net_weight))


def process_images(input_file_path_list, result_dir, network):
    # Process each image
    for input_path in tqdm(input_file_path_list):
        # Read and preprocess input image
        input_img = read_image(input_path, 255.0)
        input_img = torch.FloatTensor(input_img.transpose(0, 3, 1, 2).copy()).cuda()
        input_img, h, w = crop_image(input_img, 8, True)

        # Inference
        with torch.no_grad():
            output = network(input_img)

        # Crop and save output
        output = output[:, :, :h, :w]
        output_path = os.path.join(result_dir, os.path.basename(input_path))
        vutils.save_image(output, output_path, nrow=1, padding=0, normalize=False)


# Get input file paths
input_file_path_list = natsorted(glob(os.path.join(dataset_dir, "*.png")))
assert len(input_file_path_list) > 0, "No input images found in the dataset folder."

# Process and save images
process_images(input_file_path_list, output_dir, network)
