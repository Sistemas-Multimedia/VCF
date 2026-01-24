import os
import numpy as np
from PIL import Image

def load_frames_from_folder(frames_dir):
    files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
    frames = []
    for f in files:
        img = Image.open(os.path.join(frames_dir, f)).convert("RGB")
        frames.append(np.array(img, dtype=np.uint8))
    return frames, files

def save_frames_to_folder(frames, out_dir, prefix="recon_"):
    os.makedirs(out_dir, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr.astype(np.uint8)).save(os.path.join(out_dir, f"{prefix}{i:03d}.png"))

def compute_psnr(orig, recon, data_range=255.0):
    orig_f = orig.astype(np.float32)
    recon_f = recon.astype(np.float32)
    mse = np.mean((orig_f - recon_f)**2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(data_range / np.sqrt(mse))

def quantize_residual(residual, q_step):
    return np.round(residual / float(q_step)).astype(np.int16)

def dequantize_residual(res_q, q_step):
    return (res_q.astype(np.int16) * int(q_step)).astype(np.int16)

def compute_residual(target, predicted):
    return target.astype(np.int16) - predicted.astype(np.int16)

def reconstruct_from_residual(predicted, residual):
    rec = predicted.astype(np.int16) + residual
    return np.clip(rec, 0, 255).astype(np.uint8)

# -----------------------------
#   MOTION ESTIMATION 
# -----------------------------

def block_motion_estimation_mv(ref, target, block_size=16, search_range=2):
    H, W, _ = ref.shape
    H_blocks = H // block_size
    W_blocks = W // block_size
    mv = np.zeros((H_blocks, W_blocks, 2), dtype=np.int16)

    ref_f = ref.astype(np.float32)
    tgt_f = target.astype(np.float32)

    for by in range(H_blocks):
        for bx in range(W_blocks):
            y0 = by * block_size
            x0 = bx * block_size
            block_t = tgt_f[y0:y0+block_size, x0:x0+block_size, :]

            best_err = np.inf
            best_dy = 0
            best_dx = 0

            for dy in range(-search_range, search_range+1):
                for dx in range(-search_range, search_range+1):
                    yy = y0 + dy
                    xx = x0 + dx
                    if yy < 0 or xx < 0 or yy + block_size > H or xx + block_size > W:
                        continue
                    block_r = ref_f[yy:yy+block_size, xx:xx+block_size, :]
                    err = np.mean((block_t - block_r)**2)
                    if err < best_err:
                        best_err = err
                        best_dy = dy
                        best_dx = dx

            mv[by, bx, 0] = best_dy
            mv[by, bx, 1] = best_dx

    return mv

def apply_motion_vectors_mv(ref, motion_vectors, block_size=16):
    H, W, _ = ref.shape
    H_blocks, W_blocks, _ = motion_vectors.shape
    ref_f = ref.astype(np.float32)
    pred = np.zeros_like(ref_f)

    for by in range(H_blocks):
        for bx in range(W_blocks):
            y0 = by * block_size
            x0 = bx * block_size
            dy, dx = motion_vectors[by, bx]
            yy = y0 + dy
            xx = x0 + dx
            block_ref = ref_f[yy:yy+block_size, xx:xx+block_size, :]
            pred[y0:y0+block_size, x0:x0+block_size, :] = block_ref

    return np.clip(pred, 0, 255).astype(np.uint8)