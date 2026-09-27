'''IPP... coding: runs an intra-predictive 2D image codec for each frame of a video sequence.'''

import sys
import io
import os
import logging
import numpy as np
import cv2 as cv
import av
from PIL import Image
import importlib
import re
import pickle
import glob
with open("/tmp/description.txt", "w") as f:  # Used by parser.py
    f.write(__doc__)
import parser
import main
import entropy_video_coding as EVC

# ------------------------------------------------------------
# Encoder / Decoder arguments
# ------------------------------------------------------------
parser.parser_encode.add_argument(
    "-T", "--transform", type=str,
    help=f"2D-transform, default: {EVC.DEFAULT_TRANSFORM}",
    default=EVC.DEFAULT_TRANSFORM
)
parser.parser_encode.add_argument(
    "-N", "--number_of_frames", type=parser.int_or_str,
    help=f"Number of frames to encode (default: {EVC.N_FRAMES})",
    default=f"{EVC.N_FRAMES}"
)
parser.parser_encode.add_argument(
    "--intra_period", type=int,
    help="Insert an intra frame every intra_period frames (default: 32)",
    default=32
)
parser.parser_encode.add_argument(
    "--block_size", type=int,
    help="Block size for motion compensation (default: 16)",
    default=16
)
parser.parser_encode.add_argument(
    "--rd_optimization", action="store_true",
    help="Enable Rate-Distortion optimization for block types (default: False)"
)
parser.parser_encode.add_argument(
    "--visualize_prediction", action="store_true",
    help="Save prediction images and I/P block maps (default: False)"
)
parser.parser_encode.add_argument(
    "--lamb", type=float,
    help="Lambda parameter for RD optimization (default: 0.5)",
    default=0.5
)

parser.parser_decode.add_argument(
    "-T", "--transform", type=str,
    help=f"2D-transform, default: {EVC.DEFAULT_TRANSFORM}",
    default=EVC.DEFAULT_TRANSFORM
)
parser.parser_decode.add_argument(
    "-N", "--number_of_frames", type=parser.int_or_str,
    help=f"Number of frames to decode (default: {EVC.N_FRAMES})",
    default=f"{EVC.N_FRAMES}"
)
parser.parser_decode.add_argument(
    "--intra_period", type=int,
    help="Insert an intra frame every intra_period frames (default: 32)",
    default=32
)
parser.parser_decode.add_argument(
    "--block_size", type=int,
    help="Block size for motion compensation (default: 16)",
    default=16
)
parser.parser_decode.add_argument(
    "--rd_optimization", action="store_true",
    help="Enable Rate-Distortion optimization for block types (default: False)"
)

args = parser.parser.parse_known_args()[0]

# ------------------------------------------------------------
# Import 2D transform codec
# ------------------------------------------------------------
if __debug__ and args.debug:
    print(f"IPP: Importing {args.transform}")

try:
    transform = importlib.import_module(args.transform)
except ImportError as e:
    print(f"Error: Could not find {args.transform} module ({e})")
    print(f"Make sure '2D-{args.transform}.py' is in the same directory as IPP.py")
    sys.exit(1)

# ------------------------------------------------------------
# IPP CoDec
# ------------------------------------------------------------
class CoDec(EVC.CoDec):
    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        self.transform_codec = transform.CoDec(args)
        logging.info(f"Using {args.transform} codec for IPP")
        logging.info(f"Intra period = {args.intra_period}")
        self.block_size = args.block_size
        self.mv_blocks = []
        self.total_mse = 0.0
        self.total_pixels_per_frame = 0

    def _save_mv_chunk(self, current_img_idx):
        """Saves the current chunk of motion vectors to a separate file (one per GOP)."""
        chunk_idx = current_img_idx // self.args.intra_period
        mv_file = f"{EVC.ENCODE_OUTPUT_PREFIX}_mv_chunk_{chunk_idx:04d}.pkl"

        # Calculate which vectors belong to this chunk
        start = chunk_idx * self.args.intra_period
        chunk_data = self.mv_blocks[start : current_img_idx + 1]

        with open(mv_file, 'wb') as f:
            pickle.dump(chunk_data, f)
        logging.info(f"Saved MV chunk to {mv_file}")

    # --------------------------------------------------------
    # Encoding
    # --------------------------------------------------------
    def encode(self):
        logging.debug("trace")
        fn = self.args.original
        logging.info(f"Encoding {fn}")
        container = av.open(fn)
        img_counter = 0
        exit_flag = False
        prev_recon = None

        for packet in container.demux():
            if __debug__:
                self.total_input_size += packet.size

            for frame in packet.decode():
                img = frame.to_image().convert("RGB")
                img_np = np.array(img, dtype=np.int16)
                raw_fn = f"/tmp/original_%04d.png" % img_counter
                code_fn = f"{EVC.ENCODE_OUTPUT_PREFIX}_%04d" % img_counter
                recon_fn = f"/tmp/recon_%04d.png" % img_counter
                res_fn = f"/tmp/residual_%04d.png" % img_counter
                img.save(raw_fn)

                if img_counter == 0 or (img_counter % self.args.intra_period) == 0:
                    # Intra frame
                    logging.info("Encoding intra frame")
                    O_bytes = self.transform_codec.encode_fn(raw_fn, code_fn)
                    self.transform_codec.decode_fn(code_fn, recon_fn)
                    recon = np.array(Image.open(recon_fn), dtype=np.int16)
                    self.mv_blocks.append(None)
                else:
                    # Inter frame con Motion Compensation + RD Optimization
                    logging.info("Encoding inter frame with RD optimization")
                    prev_gray = cv.cvtColor(np.clip(prev_recon, 0, 255).astype(np.uint8), cv.COLOR_RGB2GRAY)
                    curr_gray = cv.cvtColor(np.clip(img_np, 0, 255).astype(np.uint8), cv.COLOR_RGB2GRAY)
                    flow = cv.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                    H, W = img_np.shape[:2]
                    b_h, b_w = self.block_size, self.block_size
                    mv_frame = []
                    predicted = np.zeros_like(img_np, dtype=np.float32)

                    for y in range(0, H, b_h):
                        row = []
                        for x in range(0, W, b_w):
                            y0, y1 = y, min(y+b_h, H)
                            x0, x1 = x, min(x+b_w, W)

                            # Estimate motion vector and generate inter-prediction block
                            block_flow = flow[y0:y1, x0:x1]
                            avg_vector = block_flow.reshape(-1, 2).mean(axis=0)
                            dx, dy = avg_vector
                            ys = np.clip(np.arange(y0, y1) + dy, 0, H-1).astype(np.int32)
                            xs = np.clip(np.arange(x0, x1) + dx, 0, W-1).astype(np.int32)
                            p_block = np.zeros((y1-y0, x1-x0, 3), dtype=np.float32)
                            for c in range(3):
                                p_block[:,:,c] = prev_recon[np.ix_(ys, xs, [c])].reshape(y1-y0, x1-x0)

                            # RD Optimization: Decide between Intra (I) or Inter (P)
                            use_intra = False
                            if self.args.rd_optimization:
                                original_block = img_np[y0:y1, x0:x1]

                                # --- INTER (P) Evaluation ---
                                dist_inter = np.mean((original_block - p_block)**2)
                                # Estimate R: residual entropy + motion vector bits
                                rate_inter = np.log2(np.var(original_block - p_block) + 1) + 2 
                                j_inter = dist_inter + self.args.lamb * rate_inter

                                # --- INTRA (I) Evaluation ---
                                # No prediction used (baseline 128)
                                dist_intra = np.mean((original_block - 128)**2)
                                # Estimate R: original block entropy + intra overhead
                                rate_intra = np.log2(np.var(original_block) + 1) + 4 
                                j_intra = dist_intra + self.args.lamb * rate_intra

                                if j_intra < j_inter:
                                    use_intra = True
                            if use_intra:
                                # Encode as Intra block (no motion compensation)
                                predicted[y0:y1, x0:x1] = 128
                                row.append(None)
                            else:
                                # Encode as Inter block using motion vector
                                predicted[y0:y1, x0:x1] = p_block
                                row.append(avg_vector)
                        mv_frame.append(row)

                    self.mv_blocks.append(mv_frame)

                    # Visual instrumentation
                    if self.args.visualize_prediction:
                        # 1. Save the inter-frame prediction signal (P[n])
                        pred_img_save = np.clip(predicted, 0, 255).astype(np.uint8)
                        Image.fromarray(pred_img_save).save(f"/tmp/pred_{img_counter:04d}.png")
                        # 2. Generate I/P block decision map
                        # Create a segmentation map where dark blocks represent Intra mode 
                        # and light blocks represent Inter mode (motion compensated)
                        block_map = np.zeros((H, W), dtype=np.uint8)
                        for by, row in enumerate(mv_frame):
                            for bx, vec in enumerate(row):
                                y0, x0 = by * b_h, bx * b_w
                                y1, x1 = min(y0 + b_h, H), min(x0 + b_w, W)
                                # Decision based on RD optimization: None = Intra, vector = Inter
                                val = 50 if vec is None else 200
                                block_map[y0:y1, x0:x1] = val
                        map_fn = f"/tmp/map_{img_counter:04d}.png"
                        Image.fromarray(block_map).save(map_fn)
                        logging.info(f"Saved prediction and block map for frame {img_counter}")

                    residual = img_np - predicted
                    residual = np.clip(residual + 128, 0, 255).astype(np.uint8)
                    residual_img = Image.fromarray(residual, mode="RGB")
                    residual_img.save(res_fn)
                    O_bytes = self.transform_codec.encode_fn(res_fn, code_fn)
                    self.transform_codec.decode_fn(code_fn, recon_fn)
                    residual_dec = np.array(Image.open(recon_fn), dtype=np.int16)
                    recon = predicted + (residual_dec - 128)

                frame_mse = np.mean((img_np.astype(np.float32) - recon.astype(np.float32))**2)
                self.total_mse += frame_mse
                if self.total_pixels_per_frame == 0:
                    self.total_pixels_per_frame = img_np.shape[0] * img_np.shape[1]

                self.total_output_size += O_bytes
                prev_recon = recon

                # Save chunk if we reached the end of an intra_period or the last frame
                if (img_counter + 1) % self.args.intra_period == 0 or (img_counter + 1) == self.args.number_of_frames:
                    self._save_mv_chunk(img_counter)

                img_counter += 1
                logging.info(f"img_counter = {img_counter} / {args.number_of_frames}")
                if img_counter >= args.number_of_frames:
                    exit_flag = True
                    break
            if exit_flag:
                break

        self.N_frames = img_counter

        # J = R + D
        # 1. Calculate Rate (R): Bits per pixel including motion vectors
        mv_files = glob.glob(f"{EVC.ENCODE_OUTPUT_PREFIX}_mv_chunk_*.pkl")
        total_mv_bytes = sum(os.path.getsize(f) for f in mv_files)

        total_bits = (self.total_output_size + total_mv_bytes) * 8
        total_pixels_video = self.total_pixels_per_frame * self.N_frames
        R = total_bits / total_pixels_video

        # 2. Calculate Distortion (D): Root Mean Squared Error
        avg_mse = self.total_mse / self.N_frames
        D = np.sqrt(avg_mse)

        # 3. Efficiency Metric J
        J = D + (self.args.lamb * R)

        print("\n" + "="*40)
        print(" FINAL PERFORMANCE METRICS (J = D + λR)")
        print("="*40)
        print(f"Lambda (λ):     {self.args.lamb:.4f}")
        print(f"Rate (R):       {R:.4f} bits/pixel")
        print(f"Distortion (D): {D:.4f} (RMSE)")
        print(f"Efficiency (J): {J:.4f}")
        print("="*40 + "\n")
        print(f"Frames processed:     {self.N_frames}")
        print(f"Total Residual Bytes: {self.total_output_size}")
        print(f"Total MV Bytes:       {total_mv_bytes}")

        self.height, self.width = img_np.shape[:2]
        self.N_channels = img_np.shape[2]

    # --------------------------------------------------------
    # Decoding
    # --------------------------------------------------------
    def decode(self):
        logging.debug("trace")
        prev_recon = None
        current_chunk_idx = -1
        current_mv_chunk = []

        for img_counter in range(self.args.number_of_frames):
            # Dynamic loading: Load the correct MV chunk for the current GOP
            chunk_idx = img_counter // self.args.intra_period
            if chunk_idx != current_chunk_idx:
                mv_file = f"{EVC.ENCODE_OUTPUT_PREFIX}_mv_chunk_{chunk_idx:04d}.pkl"
                with open(mv_file, 'rb') as f:
                    current_mv_chunk = pickle.load(f)
                current_chunk_idx = chunk_idx
                logging.info(f"Loaded MV chunk {mv_file} for frame {img_counter}")

            code_fn = f"{EVC.ENCODE_OUTPUT_PREFIX}_%04d" % img_counter
            out_fn = f"{EVC.DECODE_OUTPUT_PREFIX}_%04d.png" % img_counter
            logging.info(f"Decoding frame {code_fn} into {out_fn}")
            self.transform_codec.decode_fn(code_fn, out_fn)
            img_np = np.array(Image.open(out_fn), dtype=np.int16)

            mv_frame = current_mv_chunk[img_counter % self.args.intra_period]
            if mv_frame is None:
                recon = img_np
            else:
                H, W = img_np.shape[:2]
                b_h, b_w = self.block_size, self.block_size
                predicted = np.zeros_like(img_np, dtype=np.float32)
                for by, row in enumerate(mv_frame):
                    for bx, vec in enumerate(row):
                        y0, x0 = by * b_h, bx * b_w
                        y1, x1 = min(y0 + b_h, H), min(x0 + b_w, W)
                        if vec is None:
                            # Reconstruct Intra block
                            predicted[y0:y1, x0:x1] = 128
                        else:
                            # Reconstruct Inter block using stored motion vector
                            dx, dy = vec
                            ys = np.clip(np.arange(y0, y1) + dy, 0, H-1).astype(np.int32)
                            xs = np.clip(np.arange(x0, x1) + dx, 0, W-1).astype(np.int32)
                            for c in range(img_np.shape[2]):
                                predicted[y0:y1, x0:x1, c] = prev_recon[np.ix_(ys, xs, [c])].reshape(y1-y0, x1-x0)

                # Combine prediction with decoded residual
                recon = predicted + (img_np - 128)

            recon = np.clip(recon, 0, 255).astype(np.uint8)
            Image.fromarray(recon, mode="RGB").save(out_fn)
            prev_recon = recon

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
