import os
import numpy as np
from utils_nlm import load_image, save_image

def concat_side_by_side(img1, img2):
    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]
    out = np.zeros((h, w, 3), dtype=np.float32)

    out[:img1.shape[0], :img1.shape[1]] = img1
    out[:img2.shape[0], img1.shape[1]:] = img2
    return out

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FRAMES_DIR = os.path.join(ROOT, "frames")
    RESULTS_DIR = os.path.join(ROOT, "results")

    orig_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")])
    nlm_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith("_nlm.png")])

    if not orig_files or not nlm_files:
        print("No hay imágenes suficientes para comparar.")
        return

    for orig, nlm in zip(orig_files, nlm_files):
        img_o = load_image(os.path.join(FRAMES_DIR, orig))
        img_n = load_image(os.path.join(RESULTS_DIR, nlm))

        comp = concat_side_by_side(img_o, img_n)

        out_path = os.path.join(RESULTS_DIR, f"compare_{orig.replace('.png','')}.png")
        save_image(comp, out_path)
        print(f"Comparación generada: {out_path}")

if __name__ == "__main__":
    main()