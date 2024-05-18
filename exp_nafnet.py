import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from NAFNet.basicsr.models import create_model
from NAFNet.basicsr.utils.options import parse
from NAFNet.basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite


def imread(img_path):
    """
    Read the image
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def img2tensor(img, bgr2rgb=False, float32=True):
    """
    Convert image into tensors
    """
    img = img.astype(np.float32) / 255.0
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)


def single_image_inference(model, img, save_path):
    """
    Make inference over one image
    """
    model.feed_data(data={"lq": img.unsqueeze(dim=0)})

    if model.opt["val"].get("grids", False):
        model.grids()

    model.test()

    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals["result"]])
    imwrite(sr_img, save_path)

# Create a path for get pre-trained model
opt_path = "NAFNET/options/test/GoPro/NAFNet-width64.yml"
opt = parse(opt_path, is_train=False)
opt["dist"] = False
NAFNet = create_model(opt)

# Create a path for dataset
input_path = "dataset/"
output_path = "nafnet_output/"

def process_images_in_folder(model, input_folder, output_folder):
    """
    Use single image function consecutively over the input images to apply deblurring over dataset.
    """

    os.makedirs(output_folder, exist_ok=True)

    image_files = os.listdir(input_folder)

    # Iterate over each image file
    for image_file in image_files:
        # Check if the file is an image
        if image_file.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            # Construct paths
            input_path = os.path.join(input_folder, image_file)
            output_path = os.path.join(output_folder, image_file)

            # Read input image
            img_input = imread(input_path)
            inp = img2tensor(img_input)

            # Perform inference
            single_image_inference(model, inp, output_path)


# Call this function with your NAFNet model and paths
process_images_in_folder(NAFNet, input_path, output_path)
