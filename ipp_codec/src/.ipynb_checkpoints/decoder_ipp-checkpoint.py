import os, numpy as np, time
from temporal_codec_utils import (
    dequantize_residual,
    reconstruct_from_residual,
    save_frames_to_folder,
    apply_motion_vectors_mv
)

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    COMP_DIR = os.path.join(ROOT, "compressed")
    RESULTS_DIR = os.path.join(ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    gop_files = sorted([f for f in os.listdir(COMP_DIR) if f.startswith("gop_") and f.endswith(".npz")])
    all_recon = []

    t0 = time.time()
    for gf in gop_files:
        data = np.load(os.path.join(COMP_DIR, gf), allow_pickle=True)

        i_frame = data["i_frame"]
        residuals_q = data["residuals_q"]
        q_step = int(data["q_step"])

        # motion vectors pueden existir o ser None
        if "motion_vectors" in data:
            motion_vectors_arr = data["motion_vectors"]
        else:
            motion_vectors_arr = [None] * len(residuals_q)

        prev_rec = i_frame.copy()
        recon_frames = [prev_rec]

        for idx, res_q in enumerate(residuals_q):
            mv = motion_vectors_arr[idx]

            if mv is None:
                pred = prev_rec
            else:
                pred = apply_motion_vectors_mv(prev_rec, mv, block_size=16)

            residual = dequantize_residual(res_q, q_step)
            rec = reconstruct_from_residual(pred, residual)

            recon_frames.append(rec)
            prev_rec = rec

        gop_index = gf.split("_")[1].split(".")[0]
        save_frames_to_folder(recon_frames, RESULTS_DIR, prefix=f"gop{gop_index}_")
        all_recon.extend(recon_frames)
        print(f"GOP {gf} decodificado, frames: {len(recon_frames)}")

    t1 = time.time()
    print("Decodificación completa en {:.2f} s".format(t1 - t0))

if __name__ == "__main__":
    main()