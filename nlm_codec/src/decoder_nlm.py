import os
import numpy as np
from utils_nlm import load_image, save_image

def nlm_fast(image, sigma_s=2.0, sigma_r=0.1):
    """
    Filtro bilateral (aproximación rápida de NLM)
    - sigma_s: suavizado espacial
    - sigma_r: suavizado por intensidad
    """
    H, W, C = image.shape
    out = np.zeros_like(image)

    # kernel espacial 5x5
    k = 5
    pad = k // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    # kernel gaussiano espacial
    y, x = np.mgrid[-pad:pad+1, -pad:pad+1]
    spatial = np.exp(-(x*x + y*y) / (2 * sigma_s * sigma_s))

    for c in range(C):
        channel = padded[:, :, c]
        out_c = np.zeros((H, W))

        for i in range(H):
            for j in range(W):
                region = channel[i:i+k, j:j+k]
                diff = region - channel[i+pad, j+pad]
                range_w = np.exp(-(diff*diff) / (2 * sigma_r * sigma_r))

                weights = spatial * range_w
                out_c[i, j] = np.sum(weights * region) / np.sum(weights)

        out[:, :, c] = out_c

    return out


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    COMP_DIR = os.path.join(ROOT, "compressed")
    RESULTS_DIR = os.path.join(ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    comp_files = sorted([f for f in os.listdir(COMP_DIR) if f.endswith(".npz")])
    if not comp_files:
        print("No hay archivos comprimidos en nlm_codec/compressed/")
        return

    for f in comp_files:
        data = np.load(os.path.join(COMP_DIR, f), allow_pickle=True)
        img = data["image"]

        den = nlm_fast(img)

        out_path = os.path.join(RESULTS_DIR, f.replace(".npz", "_nlm.png"))
        save_image(den, out_path)
        print(f"NLM aplicado: {out_path}")

if __name__ == "__main__":
    main()