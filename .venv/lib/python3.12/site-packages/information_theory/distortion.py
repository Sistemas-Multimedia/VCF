'''Distortion computation.'''

from .information import average_energy
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

def MSE(noisy, GT):
    error_signal = noisy.astype(np.float64) - GT
    return average_energy(error_signal)

def RMSE(noisy, GT):
    return math.sqrt(MSE(noisy, GT))

def SSIM_color(noisy, GT, channel_axis=2):
    # This is not the average SSIM between the RGB channels.
    return ssim(noisy, GT, data_range=GT.max() - GT.min(), full=False, channel_axis=channel_axis)

def SSIM_grayscale(noisy, GT):
    return ssim(noisy, GT, data_range=GT.max() - GT.min(), full=False)

def SSIM(noisy, GT):
    return SSIM_grayscale(noisy, GT)

def PSNR(noisy, GT):
    max_GT = np.max(GT).astype(np.float64)
    assert max_GT > 0, f"max_GT={max_GT}"
    return 10*math.log((max_GT*max_GT)/MSE(noisy, GT))

def avg_PSNR(noisy, GT, N_channels=3):
    avg = 0
    for i in range(N_channels):
        p = PSNR(noisy[..., i], GT[..., i])
        print(p)
        avg += p
    avg /= 3
    return avg

def avg_SSIM(noisy, GT, N_channels=3):
    avg = 0
    for i in range(N_channels):
        p = SSIM(noisy[..., i], GT[..., i])
        print(p)
        avg += p
    avg /= 3
    return avg
