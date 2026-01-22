import argparse
import numpy as np
from PIL import Image

def read_image_rgb(path: str) -> np.ndarray:
    """Lee imagen y convierte a float32 en rango [0,1]"""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [0,1]
    return arr

def save_image_rgb(path: str, img01: np.ndarray) -> None:
    """Guarda imagen desde rango [0,1] a uint8"""
    img01 = np.clip(img01, 0.0, 1.0)
    out = (img01 * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(path)

def main():
    parser = argparse.ArgumentParser(description="BM3D Denoising para imágenes RGB")
    parser.add_argument("--in", dest="inp", required=True, help="Ruta de imagen de entrada (jpg/png/...)")
    parser.add_argument("--out", dest="out", required=True, help="Ruta de salida (png recomendado)")
    parser.add_argument("--sigma", type=float, required=True, help="Sigma del ruido AWGN en escala 0..255 (ej: 25)")
    args = parser.parse_args()

    # 1) Cargar imagen
    print(f"Cargando imagen: {args.inp}")
    x = read_image_rgb(args.inp)
    print(f"  Tamaño: {x.shape}")

    # 2) Ejecutar BM3D
    #    Esta librería espera sigma en escala [0,1] si la imagen está en [0,1]
    sigma01 = args.sigma / 255.0

    try:
        from bm3d import bm3d_rgb
    except ImportError:
        raise SystemExit(
            "No tienes instalada la librería bm3d.\n"
            "Instala con:\n"
            "  pip install bm3d\n"
        )

    print(f"Aplicando BM3D RGB (σ={args.sigma})...")
    # bm3d_rgb aplica BM3D en color
    y_hat = bm3d_rgb(x, sigma_psd=sigma01)

    # 3) Guardar
    save_image_rgb(args.out, y_hat)
    print(f"✓ Imagen guardada en: {args.out}")

if __name__ == "__main__":
    main()
