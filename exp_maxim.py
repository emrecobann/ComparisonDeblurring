from PIL import Image
import matplotlib.pyplot as plt
import collections
import importlib
import io
import os
import math
import requests
from tqdm import tqdm
import gdown  # to download weights from Drive
import flax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf
from jax.experimental import jax2tf


_MODEL_FILENAME = "maxim"

_MODEL_VARIANT_DICT = {
    "Denoising": "S-3",
    "Deblurring": "S-3",
    "Deraining": "S-2",
    "Dehazing": "S-2",
    "Enhancement": "S-2",
}

_MODEL_CONFIGS = {
    "variant": "",
    "dropout_rate": 0.0,
    "num_outputs": 3,
    "use_bias": True,
    "num_supervision_scales": 3,
}


class DummyFlags:
    def __init__(
        self,
        ckpt_path: str,
        task: str,
        input_dir: str = "dataset",
        output_dir: str = "maxim_output",
        has_target: bool = False,
        save_images: bool = True,
        geometric_ensemble: bool = False,
    ):

        self.ckpt_path = ckpt_path
        self.task = task
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.has_target = has_target
        self.save_images = save_images
        self.geometric_ensemble = geometric_ensemble


def recover_tree(keys, values):
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def build_model(task="Deblurring"):
    model_mod = importlib.import_module(f"maxim.models.{_MODEL_FILENAME}")
    model_configs = ml_collections.ConfigDict(_MODEL_CONFIGS)

    model_configs.variant = _MODEL_VARIANT_DICT[task]

    model = model_mod.Model(**model_configs)
    return model


def make_shape_even(image):
    """Pad the image to have even shapes."""
    height, width = image.shape[0], image.shape[1]
    padh = 1 if height % 2 != 0 else 0
    padw = 1 if width % 2 != 0 else 0
    image = jnp.pad(image, [(0, padh), (0, padw), (0, 0)], mode="reflect")
    return image


def mod_padding_symmetric(image, factor=64):
    """Padding the image to be divided by factor."""
    height, width = image.shape[0], image.shape[1]
    height_pad, width_pad = ((height + factor) // factor) * factor, (
        (width + factor) // factor
    ) * factor
    padh = height_pad - height if height % factor != 0 else 0
    padw = width_pad - width if width % factor != 0 else 0
    image = jnp.pad(
        image, [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)], mode="reflect"
    )
    return image


def pre_process(input_file):
    """
    Pre-process the image before sending to the model
    """
    input_img = np.asarray(Image.open(input_file).convert("RGB"), np.float32) / 255.0
    # Padding images to have even shapes
    height, width = input_img.shape[0], input_img.shape[1]
    input_img = make_shape_even(input_img)
    height_even, width_even = input_img.shape[0], input_img.shape[1]

    # padding images to be multiplies of 64
    input_img = mod_padding_symmetric(input_img, factor=64)
    input_img = np.expand_dims(input_img, axis=0)

    return input_img, height, width, height_even, width_even


def predict(input_img):
    # handle multi-stage outputs, obtain the last scale output of last stage
    return model.apply({"params": flax.core.freeze(params)}, input_img)


def post_process(preds, height, width, height_even, width_even):
    """
    Post process the image coming out from prediction
    """
    if isinstance(preds, list):
        preds = preds[-1]
        if isinstance(preds, list):
            preds = preds[-1]

    # De-ensemble by averaging inferenced results.
    preds = np.array(preds[0], np.float32)

    # unpad images to get the original resolution
    new_height, new_width = preds.shape[0], preds.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width
    preds = preds[h_start:h_end, w_start:w_end, :]
    return np.array((np.clip(preds, 0.0, 1.0) * 255.0).astype(jnp.uint8))


def get_params(ckpt_path):
    """Get params checkpoint."""

    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        data = f.read()
    values = np.load(io.BytesIO(data))
    params = recover_tree(*zip(*values.items()))
    params = params["opt"]["target"]

    return params


MODEL_PATH = "maxim_model/adobe.npz"
FLAGS = DummyFlags(ckpt_path=MODEL_PATH, task="Deblurring")

params = get_params(FLAGS.ckpt_path)  # Parse the config

model = build_model()  # Build Model


# Define the dataset folder and the output folder
dataset_folder = "dataset"
output_folder = "maxim_output"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(dataset_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Construct the input and output paths
        input_path = os.path.join(dataset_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Pre-process the input image
        input_img, height, width, height_even, width_even = pre_process(input_path)

        # Perform prediction
        preds = predict(input_img)

        # Post-process the prediction
        output_img = post_process(preds, height, width, height_even, width_even)

        # Save the output image
        output_img_pil = Image.fromarray(output_img)
        output_img_pil.save(output_path)

        print(f"Saved deblurred image: {output_path}")
