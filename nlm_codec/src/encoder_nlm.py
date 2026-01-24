import os
import numpy as np
from utils_nlm import load_image

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FRAMES_DIR = os.path.join(ROOT, "frames")
    COMP_DIR = os.path.join(ROOT, "compressed")
    os.makedirs(COMP_DIR, exist_ok=True)

    files = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))])
    if not files:
        print("No hay imágenes en nlm_codec/frames/")
        return

    for f in files:
        img = load_image(os.path.join(FRAMES_DIR, f))
        out_path = os.path.join(COMP_DIR, f.replace(".png", ".npz"))
        np.savez(out_path, image=img)
        print(f"Imagen comprimida (simulada): {out_path}")

if __name__ == "__main__":
    main()