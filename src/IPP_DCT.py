'''IPP hybrid video coding using motion compensation and DCT.'''

import os
import sys
import json
import tempfile
import fractions
from typing import List, Tuple
import importlib
import av
import cv2
import numpy as np
import imageio.v3 as iio
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from scipy.fftpack import dct, idct

# NOTE: Space transform and quantizer are now dynamically imported below
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TMP_DIR = os.path.join(_SCRIPT_DIR, "/tmp")
os.makedirs(_TMP_DIR, exist_ok=True)

_desc_path = os.path.join(_TMP_DIR, "description.txt")
with open(_desc_path, "w") as f:
    f.write(__doc__)

# ⚠️ ALSO create it in /tmp for parser.py compatibility
_tmp_desc_path = "/tmp/description.txt"
with open(_tmp_desc_path, "w") as f:
    f.write(__doc__)

import main
import parser as vcf_parser

# Defaults
DEFAULT_INPUT = "http://www.hpca.ual.es/~vruiz/videos/mobile_352x288x30x420x300.mp4"
DEFAULT_OUTPUT = "./ipp_encoded"
DEFAULT_N_FRAMES = 30
DEFAULT_GOP = 10
DEFAULT_BLOCK_ME = 16
DEFAULT_SEARCH = 8
DEFAULT_SPACE_TRANSFORM = "2D-DCT"

# Parse space transform BEFORE importing parent codec to avoid conflicts
import argparse
import sys
_temp_parser = argparse.ArgumentParser(add_help=False)
# 2. DEFINE THE --st ARGUMENT
# By default, DEFAULT_SPACE_TRANSFORM is "2D-DCT".
# If the user runs the script without --st, space_transform_name will be "2D-DCT".
_temp_parser.add_argument("--st", dest="space_transform", type=str, default=DEFAULT_SPACE_TRANSFORM)

# 3. READ THE COMMAND LINE
# This looks at what the user typed (like --st 2D-DCT).
_temp_args, _ = _temp_parser.parse_known_args()

# 4. STORE THE FILE NAME IN A VARIABLE
# If the user didn't specify --st, space_transform_name = "2D-DCT"
space_transform_name = _temp_args.space_transform

# --- HACKS FOR COMPATIBILITY ---
# (Removes help flags and ensures 'encode' is present so the imported
# module doesn't crash while parsing its own arguments)
_help_flags = [x for x in sys.argv if x in ('-h', '--help')]
for x in _help_flags: sys.argv.remove(x)
_added_encode = False
if 'encode' not in sys.argv and 'decode' not in sys.argv:
    sys.argv.append('encode')
    _added_encode = True

# 5. THE DYNAMIC IMPORT (THE "MAGIC" LINE)
# This is where the inheritance starts.
# It takes the string inside 'space_transform_name' (e.g., "2D-DCT")
# and loads that file into a variable called 'ST'.
try:
    # This is equivalent to saying: import 2D-DCT as ST
    ST = importlib.import_module(space_transform_name)
finally:
    if _added_encode:
        sys.argv.remove('encode')

# Restore help flags
sys.argv.extend(_help_flags)

# IPP-specific (temporal) encoder arguments
# Try to add inputs/outputs, ignoring if they already exist
try:
    vcf_parser.parser_encode.add_argument("-i", "--input", type=str, default=DEFAULT_INPUT,
        help=f"Input video (default: {DEFAULT_INPUT})")
except Exception:
    pass

try:
    vcf_parser.parser_encode.add_argument("-O", "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output prefix (default: {DEFAULT_OUTPUT})")
except Exception:
    pass

vcf_parser.parser_encode.add_argument("-N", "--number_of_frames", type=int, default=DEFAULT_N_FRAMES,
    help=f"Number of frames to encode (default: {DEFAULT_N_FRAMES})")
vcf_parser.parser_encode.add_argument("-G", "--gop_size", type=int, default=DEFAULT_GOP,
    help=f"GOP size for IPP pattern (default: {DEFAULT_GOP})")
vcf_parser.parser_encode.add_argument("-M", "--block_size_ME", type=int, default=DEFAULT_BLOCK_ME,
    help=f"Motion estimation block size (default: {DEFAULT_BLOCK_ME})")
vcf_parser.parser_encode.add_argument("-S", "--search_range", type=int, default=DEFAULT_SEARCH,
    help=f"Search range in pixels (default: {DEFAULT_SEARCH})")
vcf_parser.parser_encode.add_argument("--fast", action="store_true",
    help="Use fast motion estimation (3-step search)")
vcf_parser.parser_encode.add_argument("--threads", type=int, default=0,
    help="Number of threads (0=auto, default: 0)")
vcf_parser.parser_encode.add_argument("-R", "--rdo_lambda", type=float, default=0.0,
    help="RDO lambda for IPP block mode decision (default: 0.0)")

# IPP-specific (temporal) decoder arguments
try:
    vcf_parser.parser_decode.add_argument("-i", "--input", type=str, default=DEFAULT_OUTPUT,
        help=f"Input prefix (default: {DEFAULT_OUTPUT})")
except Exception:
    pass

try:
    vcf_parser.parser_decode.add_argument("-O", "--output", type=str, default="./ipp_decoded",
        help="Output prefix (default: ./ipp_decoded)")
except Exception:
    pass
vcf_parser.parser_decode.add_argument("-N", "--number_of_frames", type=int, default=DEFAULT_N_FRAMES,
    help=f"Number of frames (default: {DEFAULT_N_FRAMES})")
vcf_parser.parser_decode.add_argument("-G", "--gop_size", type=int,
    help="GOP size for IPP pattern")
vcf_parser.parser_decode.add_argument("-M", "--block_size_ME", type=int,
    help="Motion estimation block size")
vcf_parser.parser_decode.add_argument("-S", "--search_range", type=int,
    help="Search range in pixels")


def _resolve_prefix(prefix: str) -> str:
    """
    Resolves a relative file prefix string to an absolute path within the temporary directory.
    If the prefix starts with './', it's treated as relative to _TMP_DIR.
    Otherwise, if it's not absolute, it's joined with _TMP_DIR.
    """
    if prefix.startswith("./"):
        return os.path.join(_TMP_DIR, prefix[2:])
    if not os.path.isabs(prefix):
        return os.path.join(_TMP_DIR, prefix)
    return prefix


def _ensure_dir(prefix: str):
    """
    Ensures that the directory component of the given file prefix exists.
    Creates the directory if it doesn't already exist.
    """
    d = os.path.dirname(prefix)
    if d:
        os.makedirs(d, exist_ok=True)

def _three_step_search(ref_frame, curr_block, i, j, bs, initial_sr=8):
    """
    Implementation of the Three-Step Search (TSS) algorithm for fast motion estimation.
    It reduces the number of search points significantly compared to full search by
    iteratively refining the search center with decreasing step sizes.
    """
    h, w = ref_frame.shape[:2]
    step_size = initial_sr // 2
    center_x, center_y = j, i
    best_mv = (0, 0)
    min_sad = float('inf')

    # Initial check at center
    if (center_y >= 0 and center_y + bs <= h and
        center_x >= 0 and center_x + bs <= w):
        ref_block = ref_frame[center_y:center_y+bs, center_x:center_x+bs]
        min_sad = np.sum(np.abs(curr_block.astype(np.int16) - ref_block.astype(np.int16)))

    while step_size >= 1:
        improved = False
        # Check 8 surrounding points
        for dy in [-step_size, 0, step_size]:
            for dx in [-step_size, 0, step_size]:
                if dy == 0 and dx == 0:
                    continue

                ref_y = center_y + dy
                ref_x = center_x + dx

                if (ref_y >= 0 and ref_y + bs <= h and
                    ref_x >= 0 and ref_x + bs <= w):
                    ref_block = ref_frame[ref_y:ref_y+bs, ref_x:ref_x+bs]
                    sad = np.sum(np.abs(curr_block.astype(np.int16) - ref_block.astype(np.int16)))

                    if sad < min_sad:
                        min_sad = sad
                        best_mv = (ref_x - j, ref_y - i)
                        center_x, center_y = ref_x, ref_y
                        improved = True

        if not improved:
            step_size //= 2
        else:
            step_size = max(1, step_size // 2)

    return best_mv


def _process_block_row(args):
    """
    Helper function to process a single row of blocks for motion estimation.
    Designed for parallel execution using multiprocessing or a thread pool.
    Calculates motion vectors for each block in the row.
    """
    ref_frame, curr_frame, i, bs, sr, w, use_fast = args
    row_mvs = []

    for j in range(0, w - bs + 1, bs):
        curr_block = curr_frame[i:i+bs, j:j+bs]

        if use_fast:
            best_mv = _three_step_search(ref_frame, curr_block, i, j, bs, sr)
        else:
            # Full search (optimized)
            min_sad = float('inf')
            best_mv = (0, 0)
            h = ref_frame.shape[0]

            for dy in range(-sr, sr + 1):
                ref_y = i + dy
                if ref_y < 0 or ref_y + bs > h:
                    continue

                for dx in range(-sr, sr + 1):
                    ref_x = j + dx
                    if ref_x < 0 or ref_x + bs > w:
                        continue

                    ref_block = ref_frame[ref_y:ref_y+bs, ref_x:ref_x+bs]
                    sad = np.sum(np.abs(curr_block.astype(np.int16) - ref_block.astype(np.int16)))

                    if sad < min_sad:
                        min_sad = sad
                        best_mv = (dx, dy)

        row_mvs.append(best_mv)

    return i, row_mvs

class IPP:
    def __init__(self, codec, block_size=16, search_range=8, use_fast=False, num_threads=0, rdo_lambda=0.0):
        self.codec = codec # The parent codec (ST.CoDec) instance
        self.block_size = block_size
        self.search_range = search_range
        self.use_fast = use_fast
        self.num_threads = num_threads
        self.rdo_lambda = rdo_lambda

    def dct_2d(self, block: np.ndarray) -> np.ndarray:
        """Applies 2D-DCT to a block."""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct_2d(self, block: np.ndarray) -> np.ndarray:
        """Applies inverse 2D-DCT to a block."""
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def quantize(self, block: np.ndarray, qss: int = 32) -> np.ndarray:
        """Quantizes a DCT block using simple uniform quantization."""
        return np.round(block / qss).astype(np.int16)

    def dequantize(self, indices: np.ndarray, qss: int = 32) -> np.ndarray:
        """Dequantizes indices back to DCT coefficients."""
        return indices.astype(np.float32) * qss

    def get_rate(self, data: np.ndarray, is_intra: bool = False) -> float:
        """Estime le coût en bits des coefficients quantifiés."""
        if data.size == 0:
            return 0.0

        non_zero = np.sum(np.abs(data) > 0)
        magnitude_sum = np.sum(np.abs(data))

        if is_intra:
            # ✅ INTRA : coûte plus cher mais pas 10× plus cher
            base_rate = non_zero * 3.0      # Réduit de 8.0 → 3.0
            magnitude_cost = magnitude_sum * 0.3  # Réduit de 1.0 → 0.3
            header_cost = 8.0  # Réduit de 16.0 → 8.0
        else:
            # INTER : léger
            base_rate = non_zero * 2.0
            magnitude_cost = magnitude_sum * 0.2
            header_cost = 4.0  # MV cost (réduit de 8.0 → 4.0)

        return base_rate + magnitude_cost + header_cost

    def rdo_block_decision(self, curr_block: np.ndarray, ref_block: np.ndarray, qss: int = None):
        """
        Décision RDO : INTRA vs INTER en utilisant DCT + Quantization
        """
        if qss is None:
            qss = getattr(self.codec.args, 'QSS', 32)

        # ====================================================================
        # INTER Mode : encoder le résiduel
        # ====================================================================
        residual_inter = curr_block.astype(float) - ref_block.astype(float)
        dct_inter = self.dct_2d(residual_inter)
        q_indices_inter = self.quantize(dct_inter, qss)
        recon_dct_inter = self.dequantize(q_indices_inter, qss)
        recon_residual_inter = self.idct_2d(recon_dct_inter)
        recon_block_inter = ref_block.astype(float) + recon_residual_inter

        distortion_inter = np.mean((curr_block.astype(float) - recon_block_inter) ** 2)
        rate_inter = self.get_rate(q_indices_inter, is_intra=False)
        cost_inter = distortion_inter + self.rdo_lambda * rate_inter

        # ====================================================================
        # INTRA Mode : encoder le bloc directement
        # ====================================================================
        dct_intra = self.dct_2d(curr_block.astype(float))
        q_indices_intra = self.quantize(dct_intra, qss)
        recon_dct_intra = self.dequantize(q_indices_intra, qss)
        recon_block_intra = self.idct_2d(recon_dct_intra)

        distortion_intra = np.mean((curr_block.astype(float) - recon_block_intra) ** 2)
        rate_intra = self.get_rate(q_indices_intra, is_intra=True)
        cost_intra = distortion_intra + self.rdo_lambda * rate_intra

        # ✅ DEBUG: Log quelques blocs
        if np.random.rand() < 0.001:  # 0.1% des blocs
            print(f"\n🔍 DEBUG RDO (λ={self.rdo_lambda}, QSS={qss}):")
            print(f"  INTER: D={distortion_inter:.2f}, R={rate_inter:.2f}, Cost={cost_inter:.2f}")
            print(f"  INTRA: D={distortion_intra:.2f}, R={rate_intra:.2f}, Cost={cost_intra:.2f}")
            print(f"  q_inter non-zero: {np.sum(np.abs(q_indices_inter) > 0)}/{q_indices_inter.size}")
            print(f"  q_intra non-zero: {np.sum(np.abs(q_indices_intra) > 0)}/{q_indices_intra.size}")
            print(f"  → Decision: {'INTER' if cost_inter <= cost_intra else 'INTRA'}")

        # ====================================================================
        # Décision
        # ====================================================================
        if cost_inter <= cost_intra:
            return "P", residual_inter, distortion_inter
        else:
            return "I", curr_block.astype(float), distortion_intra

    def block_matching(self, ref_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        h, w = ref_frame.shape[:2]
        bs = self.block_size
        sr = self.search_range

        # Convert to grayscale if color
        if len(ref_frame.shape) == 3:
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        else:
            ref_gray = ref_frame
            curr_gray = curr_frame

        mv_field = np.zeros((h // bs, w // bs, 2), dtype=np.float32)

        # Prepare arguments for parallel processing
        row_args = [
            (ref_gray, curr_gray, i, bs, sr, w, self.use_fast)
            for i in range(0, h - bs + 1, bs)
        ]

        # Use thread pool for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.num_threads if self.num_threads > 0 else None) as executor:
            # Note: _process_block_row must be available in module scope
            results = list(executor.map(_process_block_row, row_args))

        # Assemble results
        for i, row_mvs in results:
            row_idx = i // bs
            for col_idx, mv in enumerate(row_mvs):
                mv_field[row_idx, col_idx] = mv

        return mv_field

    def motion_compensate(self, frame: np.ndarray, mv_field: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        bs = self.block_size
        compensated = np.zeros_like(frame)

        for i in range(0, h - bs + 1, bs):
            for j in range(0, w - bs + 1, bs):
                mv = mv_field[i // bs, j // bs]
                ref_y = int(i + mv[1])
                ref_x = int(j + mv[0])

                if (ref_y >= 0 and ref_y + bs <= h and
                    ref_x >= 0 and ref_x + bs <= w):
                    compensated[i:i+bs, j:j+bs] = frame[ref_y:ref_y+bs, ref_x:ref_x+bs]
                else:
                    compensated[i:i+bs, j:j+bs] = frame[i:i+bs, j:j+bs]

        return compensated

    def temporal_filter(self, frames: List[np.ndarray], gop_size: int):

      # Applies temporal coding using an IPP structure:
      # splits frames into GOPs, encodes the first frame as I-frame,
      # then encodes following frames as P-frames using motion estimation,
      # motion compensation, and optional RDO for intra/inter block decisions.

        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for IPP structure")

        I_infos = []
        P_infos = []
        mv_infos = []
        reconstructed_frames = []

        # Assuming first frame is I-frame for the first GOP
        # But we loop GOPs.

        for gop_idx in range(0, len(frames), gop_size):
            logging.info(f"Processing GOP {gop_idx // gop_size + 1}")

            # --- I-Frame ---
            I_frame_orig = frames[gop_idx]

            # Delegate encoding of I-frame to parent codec (Transform + Quant + Entropy)
            recon_I, bits_I = self.codec.encode_decode_proxy(
                I_frame_orig,
                frame_type="I",
                seq_idx=len(I_infos)
            )

            I_infos.append({"bits": bits_I, "idx": gop_idx})
            reconstructed_frames.append(recon_I)
            ref_frame = recon_I

            # --- P-Frames ---
            for p_idx in range(1, min(gop_size, len(frames) - gop_idx)):
                curr_idx = gop_idx + p_idx
                curr_frame = frames[curr_idx]

                # ME / MC
                mv_field = self.block_matching(ref_frame, curr_frame)
                compensated = self.motion_compensate(ref_frame, mv_field)

                # Check if RDO is enabled
                if self.rdo_lambda > 0:
                    # Block-level RDO mode decision
                    h, w = curr_frame.shape[:2]
                    bs = self.block_size
                    mode_map = np.zeros((h // bs, w // bs), dtype=np.uint8)  # 0 for P, 1 for I

                    # Create frame to encode with mixed modes
                    frame_to_encode = np.zeros_like(curr_frame, dtype=np.float32)

                    for i in range(0, h - bs + 1, bs):
                        for j in range(0, w - bs + 1, bs):
                            # ✅ RDO decision based on LUMINANCE (grayscale) only
                            if len(curr_frame.shape) == 3:
                                # Convert block to grayscale for RDO decision
                                curr_block_gray = cv2.cvtColor(
                                    curr_frame[i:i+bs, j:j+bs],
                                    cv2.COLOR_RGB2GRAY
                                )
                                comp_block_gray = cv2.cvtColor(
                                    compensated[i:i+bs, j:j+bs],
                                    cv2.COLOR_RGB2GRAY
                                )
                            else:
                                curr_block_gray = curr_frame[i:i+bs, j:j+bs]
                                comp_block_gray = compensated[i:i+bs, j:j+bs]

                            # ✅ RDO decision (ONE TIME per block)
                            qss = getattr(self.codec.args, 'QSS', 32)
                            mode, _, _ = self.rdo_block_decision(curr_block_gray, comp_block_gray, qss)

                            # Store mode
                            mode_map[i // bs, j // bs] = 1 if mode == "I" else 0

                            # ✅ Now apply the decision to ALL channels
                            for chan in range(curr_frame.shape[2] if len(curr_frame.shape) == 3 else 1):
                                if len(curr_frame.shape) == 3:
                                    curr_block = curr_frame[i:i+bs, j:j+bs, chan]
                                    comp_block = compensated[i:i+bs, j:j+bs, chan]
                                else:
                                    curr_block = curr_frame[i:i+bs, j:j+bs]
                                    comp_block = compensated[i:i+bs, j:j+bs]

                                # Prepare data to encode based on mode
                                if mode == "P":
                                    # Residual
                                    residual_data = curr_block.astype(float) - comp_block.astype(float)
                                else:
                                    # For Intra, encode as residual from zero (shifted)
                                    residual_data = curr_block.astype(float) - 128.0

                                if len(curr_frame.shape) == 3:
                                    frame_to_encode[i:i+bs, j:j+bs, chan] = residual_data
                                else:
                                    frame_to_encode[i:i+bs, j:j+bs] = residual_data
                    num_intra = np.sum(mode_map == 1)
                    num_inter = np.sum(mode_map == 0)
                    total_blocks = mode_map.size

                    print(f"\n📊 Frame {curr_idx} (λ={self.rdo_lambda}):")
                    print(f"   INTRA: {num_intra}/{total_blocks} ({100*num_intra/total_blocks:.1f}%)")
                    print(f"   INTER: {num_inter}/{total_blocks} ({100*num_inter/total_blocks:.1f}%)")
                    # ✅✅✅ FIN DES LOGS ✅✅✅

                    # Shift to 0-255 range for encoding
                    frame_to_encode_shifted = np.clip(frame_to_encode + 128, 0, 255).astype(np.uint8)

                    # Encode the mixed-mode frame
                    recon_shifted, bits_P = self.codec.encode_decode_proxy(
                        frame_to_encode_shifted,
                        frame_type="P",
                        seq_idx=len(P_infos)
                    )

                    # Reconstruct P-frame with mode awareness
                    recon_data = recon_shifted.astype(np.float32) - 128
                    recon_P = np.zeros_like(curr_frame, dtype=np.uint8)

                    for i in range(0, h - bs + 1, bs):
                        for j in range(0, w - bs + 1, bs):
                            mode = "I" if mode_map[i // bs, j // bs] == 1 else "P"

                            if len(curr_frame.shape) == 3:
                                for chan in range(curr_frame.shape[2]):
                                    if mode == "P":
                                        recon_block = compensated[i:i+bs, j:j+bs, chan].astype(np.float32) + recon_data[i:i+bs, j:j+bs, chan]
                                    else:
                                        recon_block = recon_data[i:i+bs, j:j+bs, chan] + 128.0
                                    recon_P[i:i+bs, j:j+bs, chan] = np.clip(recon_block, 0, 255)
                            else:
                                if mode == "P":
                                    recon_block = compensated[i:i+bs, j:j+bs].astype(np.float32) + recon_data[i:i+bs, j:j+bs]
                                else:
                                    recon_block = recon_data[i:i+bs, j:j+bs] + 128.0
                                recon_P[i:i+bs, j:j+bs] = np.clip(recon_block, 0, 255)

                    # Store mode map with motion vectors
                    mv_infos.append({"mv": mv_field, "modes": mode_map})

                    # Log RDO statistics
                    num_intra = np.sum(mode_map == 1)
                    num_inter = np.sum(mode_map == 0)
                    total_blocks = mode_map.size
                    logging.info(f"  RDO (λ={self.rdo_lambda}): {num_intra}/{total_blocks} I-blocks ({100*num_intra/total_blocks:.1f}%), {num_inter}/{total_blocks} P-blocks ({100*num_inter/total_blocks:.1f}%)")


                else:
                    # Original behavior: no RDO, always use P-mode
                    # Residual Calculation
                    residual = curr_frame.astype(np.float32) - compensated.astype(np.float32)

                    # Shift residual to be positive (0..255) for standard image codec
                    residual_shifted = np.clip(residual + 128, 0, 255).astype(np.uint8)

                    # Delegate encoding of Residual (Transform + Quant + Entropy)
                    recon_res_shifted, bits_P = self.codec.encode_decode_proxy(
                        residual_shifted,
                        frame_type="P",
                        seq_idx=len(P_infos)
                    )

                    # Reconstruct P-frame
                    recon_res = recon_res_shifted.astype(np.float32) - 128
                    recon_P = np.clip(compensated + recon_res, 0, 255).astype(np.uint8)

                    # Store only motion vectors (no mode map)
                    mv_infos.append(mv_field)

                # Update Reference
                ref_frame = recon_P

                P_infos.append({"bits": bits_P})
                reconstructed_frames.append(recon_P)

        return I_infos, P_infos, mv_infos, reconstructed_frames


class CoDec(ST.CoDec):
    def __init__(self, args):
        super().__init__(args)
        self.gop_size = getattr(args, 'gop_size', 10)
        self.block_size_ME = getattr(args, 'block_size_ME', 16)
        self.search_range = getattr(args, 'search_range', 8)
        self.use_fast = getattr(args, 'fast', False)
        self.num_threads = getattr(args, 'threads', 0)

        # Output prefix handling
        self.prefix = _resolve_prefix(args.output) if args.output else None

    def bye(self):
        if hasattr(self, 'total_bits') and hasattr(self, 'N_frames') and self.N_frames > 0:
            bpp = self.total_bits / (self.N_frames * self.width * self.height)
            logging.info(f"Output bit-rate = {bpp:.4f} bits/pixel")

    def encode_decode_proxy(self, img, frame_type, seq_idx):
        # Temp file names
        # Make them dependent on prefix to avoid collisions
        tmp_in = f"{self.prefix}_{frame_type}_{seq_idx}_tmp_in.png"

        # Output filename base for the codec
        # Parent codec usually appends extensions, so we give a base.
        # But we need to know the exact output file to measure size.
        # 2D-DCT writes multiple files.
        enc_out_base = f"{self.prefix}_{frame_type}_{seq_idx}_enc"

        # Save temp input
        iio.imwrite(tmp_in, img)

        # Encode using PARENT's method (inherited)
        # This uses self.encode_fn provided by 2D-DCT/2D-DWT
        # logic: encode_fn(in_fn, out_fn) -> output_size
        size = self.encode_fn(tmp_in, enc_out_base)

        # Decode using PARENT's method to get reconstruction
        tmp_rec = f"{self.prefix}_{frame_type}_{seq_idx}_tmp_rec.png"
        self.decode_fn(enc_out_base, tmp_rec)

        # Read reconstruction
        recon = iio.imread(tmp_rec)

        # Clean up temp inputs/outputs if desired
        # keeping enc_out_base files as they are the payload!
        if os.path.exists(tmp_in): os.remove(tmp_in)
        if os.path.exists(tmp_rec): os.remove(tmp_rec)

        return recon, size

    def encode(self):
        logging.debug("trace")
        fn = self.args.input # origina
        logging.info(f"Encoding {fn}")

        # Read Video
        container = av.open(fn)
        stream = container.streams.video[0]

        frames = []
        for packet in container.demux(stream):
            for frame in packet.decode():
                img = np.array(frame.to_image().convert("RGB"))
                frames.append(img)
                if len(frames) >= self.args.number_of_frames: break
            if len(frames) >= self.args.number_of_frames: break
        container.close()

        self.N_frames = len(frames)
        self.height, self.width = frames[0].shape[:2]

        logging.info(f"Video: {self.width}x{self.height}, {self.N_frames} frames")

        # Save original frames for evaluation (Required by validation script)
        orig_prefix = f"{self.prefix}_O"
        for idx, img in enumerate(frames):
            iio.imwrite(f"{orig_prefix}_{idx:04d}.png", img)

        # Initialize IPP with SELF as codec
        ipp = IPP(self, self.block_size_ME, self.search_range, self.use_fast, self.num_threads,
                  rdo_lambda=getattr(self.args, 'rdo_lambda', 0.0))

        # Run Temporal Filter (which calls back to self.encode_decode_proxy)
        I_infos, P_infos, mv_infos, recon_frames = ipp.temporal_filter(frames, self.gop_size)

        # Save Motion Vectors
        mv_path = f"{self.prefix}_mv.npz"
        np.savez_compressed(mv_path, mv=np.array(mv_infos, dtype=object))

        # Metadata
        total_bits = sum(i['bits'] for i in I_infos) + sum(p['bits'] for p in P_infos)
        # Add MV size approximation or actual file size
        mv_size = os.path.getsize(mv_path) * 8
        total_bits += mv_size
        self.total_bits = total_bits

        meta = {
            "n_frames": self.N_frames,
            "width": self.width,
            "height": self.height,
            "gop_size": self.gop_size,
            "total_bits": total_bits,
            "I_info": I_infos,
            "P_info": P_infos,
            "mv_file": f"{os.path.basename(self.prefix)}_mv.npz",
            "base_prefix": os.path.basename(self.prefix)
        }

        with open(f"{self.prefix}_meta.json", "w") as f:
            json.dump(meta, f, indent=4)

    def decode(self):
        logging.info("Decoding with Modular IPP structure")
        # Load meta
        # For decode, args.input is the prefix (or we derive it?)
        # args.input is typically output of encoder (e.g. prefix)
        # But here we used -i for input prefix and -O for output prefix in decoder.

        in_prefix = _resolve_prefix(self.args.input)

        with open(f"{in_prefix}_meta.json", "r") as f:
            meta = json.load(f)

        mv_path = f"{os.path.dirname(in_prefix)}/{meta['mv_file']}"
        mvs = np.load(mv_path, allow_pickle=True)['mv']

        recon_frames = []
        mv_idx = 0
        P_idx = 0
        I_idx = 0
        gop_size = meta['gop_size']
        N_frames = meta['n_frames']

        out_prefix = _resolve_prefix(self.args.output)
        _ensure_dir(out_prefix)

        for gop_idx in range(0, N_frames, gop_size):
            # Decode I-frame
            if I_idx >= len(meta['I_info']): break

            enc_base = f"{in_prefix}_I_{I_idx}_enc"
            tmp_rec = f"{out_prefix}_tmp_I_{I_idx}.png"

            # Use PARENT decode_fn
            self.decode_fn(enc_base, tmp_rec)

            ref_frame = iio.imread(tmp_rec)
            recon_frames.append(ref_frame)
            I_idx += 1

            # Decode P-frames
            for p in range(1, min(gop_size, N_frames - gop_idx)):
                # Decode Residual
                enc_base_P = f"{in_prefix}_P_{P_idx}_enc"
                tmp_rec_P = f"{out_prefix}_tmp_P_{P_idx}.png"

                self.decode_fn(enc_base_P, tmp_rec_P)

                recon_res_shifted = iio.imread(tmp_rec_P).astype(np.float32)
                recon_res = recon_res_shifted - 128

                # Get MV and check for mode map
                current_mv_data = mvs[mv_idx]

                # Check if this is a dict with mode map (RDO enabled) or just array (no RDO)
                if isinstance(current_mv_data, dict):
                    current_mv = current_mv_data['mv']
                    mode_map = current_mv_data.get('modes', None)
                else:
                    current_mv = current_mv_data
                    mode_map = None

                # MC
                ipp = IPP(self, self.block_size_ME) # Helper for motion compensate
                pred = ipp.motion_compensate(ref_frame, current_mv)

                # Reconstruct with mode awareness
                if mode_map is not None:
                    # RDO was used: apply mode-specific reconstruction
                    h, w = ref_frame.shape[:2]
                    bs = self.block_size_ME
                    recon_P = np.zeros_like(ref_frame, dtype=np.uint8)

                    for i in range(0, h - bs + 1, bs):
                        for j in range(0, w - bs + 1, bs):
                            mode = "I" if mode_map[i // bs, j // bs] == 1 else "P"

                            if len(ref_frame.shape) == 3:
                                for chan in range(ref_frame.shape[2]):
                                    if mode == "P":
                                        recon_block = pred[i:i+bs, j:j+bs, chan].astype(np.float32) + recon_res[i:i+bs, j:j+bs, chan]
                                    else:
                                        recon_block = recon_res[i:i+bs, j:j+bs, chan] + 128.0
                                    recon_P[i:i+bs, j:j+bs, chan] = np.clip(recon_block, 0, 255)
                            else:
                                if mode == "P":
                                    recon_block = pred[i:i+bs, j:j+bs].astype(np.float32) + recon_res[i:i+bs, j:j+bs]
                                else:
                                    recon_block = recon_res[i:i+bs, j:j+bs] + 128.0
                                recon_P[i:i+bs, j:j+bs] = np.clip(recon_block, 0, 255)
                else:
                    # No RDO: standard P-frame reconstruction
                    recon_P = np.clip(pred + recon_res, 0, 255).astype(np.uint8)

                recon_frames.append(recon_P)
                ref_frame = recon_P

                mv_idx += 1
                P_idx += 1

                # Cleanup
                if os.path.exists(tmp_rec_P): os.remove(tmp_rec_P)

            if os.path.exists(tmp_rec): os.remove(tmp_rec)

        # Save Decoded Video? Or frames?
        print(f"\n🔍 DEBUG DECODE - Début de la sauvegarde")
        print(f"  Nombre de frames reconstruites : {len(recon_frames)}")

        # Sauvegarder les frames PNG individuelles
        for idx, img in enumerate(recon_frames):
            try:
                if img is not None:
                    iio.imwrite(f"{out_prefix}_{idx:04d}.png", img)
                else:
                    print(f"⚠️  Frame {idx} est None, ignorée pour PNG")
            except Exception as e:
                print(f"❌ Erreur sauvegarde PNG frame {idx}: {e}")

        # Also save as MP4 video for convenience
        print(f"\n🔍 DEBUG DECODE - Tentative création MP4")

        try:
            if len(recon_frames) == 0:
                logging.warning("No frames to save as MP4 (recon_frames is empty)")
            else:
                print(f"  Validation des frames...")
                # Ensure all frames are valid numpy arrays with same dtype
                valid_frames = []
                for idx, frame in enumerate(recon_frames):
                    if frame is None:
                        print(f"  ⚠️  Frame {idx} est None")
                        continue

                    if not isinstance(frame, np.ndarray):
                        print(f"  ⚠️  Frame {idx} n'est pas un ndarray: {type(frame)}")
                        continue

                    # Debug info pour les premières frames
                    if idx < 3:
                        print(f"  Frame {idx}: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")

                    # Convert to uint8 if not already
                    if frame.dtype != np.uint8:
                        print(f"  ⚠️  Frame {idx} n'est pas uint8: {frame.dtype}, conversion...")
                        frame = np.clip(frame, 0, 255).astype(np.uint8)

                    valid_frames.append(frame)

                print(f"  Frames valides : {len(valid_frames)}/{len(recon_frames)}")

                if len(valid_frames) > 0:
                    print(f"  Empilement des frames...")
                    # Stack frames into (N, H, W, C) array
                    video_array = np.stack(valid_frames)
                    print(f"  Video array shape: {video_array.shape}, dtype: {video_array.dtype}")

                    print(f"  Écriture du MP4...")
                    # Save using pyav plugin (default for imageio video)
                    iio.imwrite(f"{out_prefix}.mp4", video_array, fps=30, codec='libx264', plugin='pyav')
                    logging.info(f"Saved decoded video to {out_prefix}.mp4 ({len(valid_frames)} frames)")
                    print(f"✓ MP4 créé avec succès")
                else:
                    logging.warning("No valid frames to save as MP4")
                    print(f"❌ Aucune frame valide pour créer le MP4")

        except Exception as e:
            logging.warning(f"Could not save decoded video to MP4: {e}")
            print(f"\n❌ ERREUR DÉTAILLÉE lors de la création du MP4:")
            print(f"   Type d'erreur: {type(e).__name__}")
            print(f"   Message: {e}")
            import traceback
            print(f"   Traceback:")
            traceback.print_exc()

if __name__ == "__main__":
    main.main(vcf_parser.parser, logging, CoDec)
