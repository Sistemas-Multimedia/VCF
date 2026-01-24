import os
import numpy as np
from utils_nlm import load_image
from skimage.metrics import peak_signal_noise_ratio as psnr

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FRAMES_DIR = os.path.join(ROOT, "frames")
    RESULTS_DIR = os.path.join(ROOT, "results")

    orig_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")])
    nlm_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith("_nlm.png")])

    if not orig_files or not nlm_files:
        print("No hay suficientes imágenes para comparar.")
        return

    for orig, nlm in zip(orig_files, nlm_files):
        img_o = load_image(os.path.join(FRAMES_DIR, orig))
        img_n = load_image(os.path.join(RESULTS_DIR, nlm))

        p = psnr(img_o, img_n)
        print(f"{orig} → {nlm} | PSNR = {p:.2f} dB")

    print("\nPara comparación visual ejecuta:")
    print("src/python compare_nlm.py")

if __name__ == "__main__":
    main()