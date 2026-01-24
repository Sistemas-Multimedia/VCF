# Simula la descompresión de I-frame y usa esa reconstrucción como referencia
# Guarda la I-frame reconstruida en el .npz para comparación

import os, time, numpy as np
from temporal_codec_utils import (
    load_frames_from_folder,
    quantize_residual,
    compute_residual,
    block_motion_estimation_mv,
    apply_motion_vectors_mv
)

GOP_SIZE = 16
q_step = 16


USE_MV = True
MV_BLOCK = 16
MV_RANGE = 2

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FRAMES_DIR = os.path.join(ROOT, "frames")
    COMP_DIR = os.path.join(ROOT, "compressed")
    os.makedirs(COMP_DIR, exist_ok=True)

    frames, filenames = load_frames_from_folder(FRAMES_DIR)
    if not frames:
        print("No hay frames en frames/.")
        return

    num_frames = len(frames)
    num_gops = (num_frames + GOP_SIZE - 1) // GOP_SIZE

    t0 = time.time()

    for g in range(num_gops):
        start = g * GOP_SIZE
        end = min(start + GOP_SIZE, num_frames)
        gop = frames[start:end]
        gop_names = filenames[start:end]

        # I-frame
        i_frame = gop[0]
        prev_rec = i_frame.copy()  # simulación de decodificación

        motion_vectors_list = []
        residuals_q = []

        for t in range(1, len(gop)):
            curr = gop[t]

            # Predicción
            if USE_MV:
                mv = block_motion_estimation_mv(prev_rec, curr, block_size=MV_BLOCK, search_range=MV_RANGE)
                pred = apply_motion_vectors_mv(prev_rec, mv, block_size=MV_BLOCK)
            else:
                mv = None
                pred = prev_rec

            # Residual
            residual = compute_residual(curr, pred)
            residual_q = quantize_residual(residual, q_step)

            motion_vectors_list.append(mv)
            residuals_q.append(residual_q)

            # Simular decodificador
            prev_rec = np.clip(pred.astype(np.int16) + (residual_q.astype(np.int16) * q_step), 0, 255).astype(np.uint8)

        out_path = os.path.join(COMP_DIR, f"gop_{g:03d}.npz")
        np.savez(
            out_path,
            i_frame=i_frame,
            i_frame_recon_sim=frames[start],  # referencia reconstruida simulada
            motion_vectors=np.array(motion_vectors_list, dtype=object),
            residuals_q=np.array(residuals_q, dtype=object),
            q_step=q_step,
            filenames=np.array(gop_names, dtype=object),
            use_mv=USE_MV
        )

        print(f"GOP {g} guardado en {out_path}")

    t1 = time.time()
    print("Codificación completa en {:.2f} s".format(t1 - t0))

if __name__ == "__main__":
    main()