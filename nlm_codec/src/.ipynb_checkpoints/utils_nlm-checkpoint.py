import os
import numpy as np
from PIL import Image

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0

def save_image(arr, path):
    arr8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def apply_nlm_rgb(image_float, patch_size=5, patch_distance=6, h=0.1):
    den = denoise_nl_means(
        image_float,
        h=h,
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=-1,
        fast_mode=True
    )
    return den