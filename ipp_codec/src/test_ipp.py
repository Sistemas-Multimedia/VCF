import os, numpy as np
from temporal_codec_utils import load_frames_from_folder, compute_psnr

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FRAMES_DIR = os.path.join(ROOT, "frames")
    RESULTS_DIR = os.path.join(ROOT, "results")
    COMP_DIR = os.path.join(ROOT, "compressed")
    comp_files = sorted([f for f in os.listdir(COMP_DIR) if f.startswith("gop_") and f.endswith(".npz")])

    orig_frames, orig_names = load_frames_from_folder(FRAMES_DIR)


    recon_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".png")])
    recon_frames = [np.array(__import__("PIL").Image.open(os.path.join(RESULTS_DIR,f)).convert("RGB"), dtype=np.uint8) for f in recon_files]

    n = min(len(orig_frames), len(recon_frames))
    psnrs = []
    for i in range(n):
        p = compute_psnr(orig_frames[i], recon_frames[i])
        psnrs.append(p)
        print(f"Frame {i:03d} PSNR = {p:.2f} dB")

    mean_psnr = float(np.mean([v for v in psnrs if np.isfinite(v)])) if psnrs else 0.0
    print(f"\nPSNR medio (sin contar inf) = {mean_psnr:.2f} dB")

    #tamaño comprimido total
    total_orig = sum(fr.nbytes for fr in orig_frames)
    total_comp = sum(os.path.getsize(os.path.join(COMP_DIR,f)) for f in comp_files) if comp_files else 1
    cr = (total_orig * 1.0) / total_comp
    print(f"Tamaño original: {total_orig} bytes")
    print(f"Tamaño comprimido (sum gops): {total_comp} bytes")
    print(f"CR ≈ {cr:.2f}")

if __name__ == "__main__":
    main()