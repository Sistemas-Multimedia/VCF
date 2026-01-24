"""
tests_ipp.py

Script de pruebas para el códec IPP:

- Carga los frames originales desde ../frames/
- Carga los frames reconstruidos desde ../results/
- Calcula:
  - PSNR por frame
  - PSNR medio
  - Relación de compresión (CR)
  - Tiempo de codificación (leído del .npz)
"""

import os
import numpy as np
from PIL import Image
from temporal_codec_utils import (
    load_frames_from_folder,
    compute_psnr
)


def load_reconstructed_frames(results_dir):
    """
    Carga los frames reconstruidos desde la carpeta results_dir.

    Args:
        results_dir (str): ruta a la carpeta con frames reconstruidos.

    Returns:
        list[np.ndarray]: lista de arrays (H, W, 3) uint8.
    """
    files = sorted([
        f for f in os.listdir(results_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])
    frames = []
    for fname in files:
        path = os.path.join(results_dir, fname)
        img = Image.open(path).convert("RGB")
        frames.append(np.array(img, dtype=np.uint8))
    return frames, files


def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FRAMES_DIR = os.path.join(ROOT_DIR, "frames")
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    COMPRESSED_DIR = os.path.join(ROOT_DIR, "compressed")
    compressed_file = os.path.join(COMPRESSED_DIR, "video_ipp_compressed.npz")

    if not os.path.exists(compressed_file):
        print("No se encontró el archivo comprimido:", compressed_file)
        return

    print("Cargando frames originales desde:", FRAMES_DIR)
    orig_frames, orig_files = load_frames_from_folder(FRAMES_DIR)

    print("Cargando frames reconstruidos desde:", RESULTS_DIR)
    recon_frames, recon_files = load_reconstructed_frames(RESULTS_DIR)

    if len(orig_frames) != len(recon_frames):
        print("Aviso: número de frames originales y reconstruidos no coincide.")
        print("Originales:", len(orig_frames), "Reconstruidos:", len(recon_frames))

    # Cargar datos del archivo comprimido
    data = np.load(compressed_file, allow_pickle=True)
    encode_time = float(data["encode_time"])
    q_step = int(data["q_step"])

    # Calcular PSNR por frame
    psnr_values = []
    num_frames = min(len(orig_frames), len(recon_frames))

    for i in range(num_frames):
        psnr_val = compute_psnr(orig_frames[i], recon_frames[i])
        psnr_values.append(psnr_val)
        print(f"Frame {i:03d} PSNR = {psnr_val:.2f} dB")

    if psnr_values:
        mean_psnr = np.mean(psnr_values)
    else:
        mean_psnr = 0.0

    print(f"\nPSNR medio sobre {num_frames} frames: {mean_psnr:.2f} dB")

    # Relación de compresión (CR)
    total_original_bytes = sum(fr.nbytes for fr in orig_frames)
    compressed_size_bytes = os.path.getsize(compressed_file)

    if compressed_size_bytes > 0:
        cr = (total_original_bytes * 1.0) / compressed_size_bytes
    else:
        cr = float("inf")

    print(f"Tamaño total original: {total_original_bytes} bytes")
    print(f"Tamaño archivo comprimido: {compressed_size_bytes} bytes")
    print(f"Relación de compresión (CR) ≈ {cr:.2f}")

    print(f"Tiempo de codificación (guardado en .npz): {encode_time:.3f} s")
    print(f"Parámetro de cuantización q_step: {q_step}")


if __name__ == "__main__":
    main()