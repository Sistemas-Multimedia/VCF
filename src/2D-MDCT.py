'''Exploiting spatial redundancy with the Modified Discrete Cosine Transform (MDCT).

The MDCT uses overlapping windows (2N samples with 50% overlap) to eliminate
blocking artifacts inherent in block-based transforms like DCT. Also known as
Modulated Lapped Transform (MLT) in Malvar's work.

Signal extension modes (symmetric/reflect) are used at image boundaries to
minimize border artifacts, especially important for large block sizes.

Reference: H.S. Malvar, "Signal Processing with Lapped Transforms", Artech House, 1992
'''

import numpy as np
import logging
import struct
import importlib
import cv2
import os
import glob
from scipy.fftpack import dct, idct
from PIL import Image

# Import resources from DCT implementation (reused for block processing)
from DCT2D.block_DCT import analyze_image as space_analyze
from DCT2D.block_DCT import synthesize_image as space_synthesize
from DCT2D.block_DCT import get_subbands, get_blocks
from color_transforms.YCoCg import from_RGB, to_RGB
from information_theory import distortion
import main

# Write description before importing parser (parser.py tries to read it)
os.makedirs("/tmp", exist_ok=True)
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)

# Load local parser safely
import importlib.util
import sys
parser_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parser.py")
spec = importlib.util.spec_from_file_location("local_parser", parser_path)
local_parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_parser)

# Inject local parser into sys.modules for other modules
sys.modules['parser'] = local_parser

# Default settings
default_block_size = 8
default_CT = "YCoCg"
perceptual_quantization = False
disable_subbands = False

# Add parser arguments
local_parser.parser_encode.add_argument("-B", "--block_size_MDCT", type=local_parser.int_or_str, 
                                        help=f"Block size (default: {default_block_size})", 
                                        default=default_block_size)
local_parser.parser_encode.add_argument("-t", "--color_transform", type=local_parser.int_or_str, 
                                        help=f"Color transform (default: \"{default_CT}\")", 
                                        default=default_CT)
local_parser.parser_encode.add_argument("-p", "--perceptual_quantization", action='store_true', 
                                        help=f"Use perceptual quantization (default: \"{perceptual_quantization}\")", 
                                        default=perceptual_quantization)
local_parser.parser_encode.add_argument("-x", "--disable_subbands", action='store_true', 
                                        help=f"Disable the coefficients reordering in subbands (default: \"{disable_subbands}\")", 
                                        default=disable_subbands)

local_parser.parser_decode.add_argument("-B", "--block_size_MDCT", type=local_parser.int_or_str, 
                                        help=f"Block size (default: {default_block_size})", 
                                        default=default_block_size)
local_parser.parser_decode.add_argument("-t", "--color_transform", type=local_parser.int_or_str, 
                                        help=f"Color transform (default: \"{default_CT}\")", 
                                        default=default_CT)
local_parser.parser_decode.add_argument("-p", "--perceptual_quantization", action='store_true', 
                                        help=f"Use perceptual dequantization (default: \"{perceptual_quantization}\")", 
                                        default=perceptual_quantization)
local_parser.parser_decode.add_argument("-x", "--disable_subbands", action='store_true', 
                                        help=f"Disable the coefficients reordering in subbands (default: \"{disable_subbands}\")", 
                                        default=disable_subbands)

args = local_parser.parser.parse_known_args()[0]
CT = importlib.import_module(args.color_transform)

# =============================================================================
# MLT/MDCT Implementation (Malvar's Lapped Transform)
# =============================================================================

def create_mdct_window(N):
    """
    Create the sine window for MDCT (Malvar's standard window).

    The sine window satisfies the Princen-Bradley condition for
    perfect reconstruction: w[n]^2 + w[n+N]^2 = 1

    Parameters:
        N (int): Block size (window is 2N samples).

    Returns:
        np.ndarray: Window of length 2N.
    """
    n = np.arange(2 * N)
    window = np.sin(np.pi * (n + 0.5) / (2 * N))
    return window


def mdct_1d(x, N):
    """
    Compute the Modified Discrete Cosine Transform (MDCT) - Vectorized.

    MDCT maps 2N input samples to N output coefficients.
    Uses the Type-IV DCT basis with time-domain aliasing.

    Parameters:
        x (np.ndarray): Input signal (length 2N, windowed).
        N (int): Block size (output length).

    Returns:
        np.ndarray: MDCT coefficients (length N).
    """
    n0 = (N + 1) / 2
    # Vectorized computation: create matrices for n and k
    n = np.arange(2 * N)
    k = np.arange(N)
    # Compute cosine matrix (N x 2N)
    cos_matrix = np.cos(np.pi / N * (n + n0) * (k[:, np.newaxis] + 0.5))
    # Apply matrix multiplication: dot product of cosine matrix with signal
    X = np.dot(cos_matrix, x)
    return X


def imdct_1d(X, N):
    """
    Compute the Inverse MDCT - Vectorized.

    IMDCT maps N coefficients to 2N output samples.
    Perfect reconstruction requires overlap-add with adjacent blocks.

    Parameters:
        X (np.ndarray): MDCT coefficients (length N).
        N (int): Block size.

    Returns:
        np.ndarray: Reconstructed signal (length 2N).
    """
    n0 = (N + 1) / 2
    # Vectorized computation: create matrices for n and k
    n = np.arange(2 * N)
    k = np.arange(N)
    # Compute cosine matrix (2N x N)
    cos_matrix = np.cos(np.pi / N * (n[:, np.newaxis] + n0) * (k + 0.5))
    # Apply matrix multiplication: dot product of cosine matrix with coefficients
    x = np.dot(cos_matrix, X) * 2 / N
    return x


def mdct_analyze_1d(signal, N, extension_mode='symmetric'):
    """
    Apply MDCT analysis to a 1D signal.

    Divides signal into overlapping blocks, applies window, computes MDCT.
    Uses signal extension at boundaries to minimize border artifacts.

    Parameters:
        signal (np.ndarray): Input 1D signal.
        N (int): Block size.
        extension_mode (str): Signal extension mode for boundaries.
            Options: 'symmetric', 'reflect', 'periodic', 'constant', 'zero'
            Default: 'symmetric' (best for natural images)

    Returns:
        np.ndarray: MDCT coefficients (same length as input).
    """
    L = len(signal)
    # Extend signal using specified mode to minimize border artifacts
    # This is crucial for large block sizes
    if extension_mode == 'symmetric':
        # Symmetric padding: ... x2 x1 | x1 x2 ... xn | xn xn-1 ...
        padded = np.pad(signal, (N, N), mode='symmetric')
    elif extension_mode == 'reflect':
        # Reflect padding: ... x3 x2 | x1 x2 ... xn | xn-1 xn-2 ...
        padded = np.pad(signal, (N, N), mode='reflect')
    elif extension_mode == 'periodic':
        # Periodic padding: ... xn-1 xn | x1 x2 ... xn | x1 x2 ...
        padded = np.pad(signal, (N, N), mode='wrap')
    elif extension_mode == 'constant':
        # Constant padding: border values replicated
        padded = np.pad(signal, (N, N), mode='edge')
    else:
        # Zero padding (original behavior, causes border artifacts)
        padded = np.zeros(L + 2 * N)
        padded[N:N + L] = signal

    window = create_mdct_window(N)
    num_blocks = L // N
    coeffs = np.zeros(L)

    for b in range(num_blocks):
        start = b * N
        # Extract 2N samples with 50% overlap
        block = padded[start:start + 2 * N].copy()
        # Apply analysis window
        block *= window
        # Compute MDCT
        X = mdct_1d(block, N)
        # Store N coefficients
        coeffs[b * N:(b + 1) * N] = X

    return coeffs


def mdct_synthesize_1d(coeffs, N, extension_mode='symmetric'):
    """
    Apply MDCT synthesis to reconstruct a 1D signal.

    Computes IMDCT for each block, applies window, overlap-adds.
    The extension_mode parameter should match the one used in analysis.

    Parameters:
        coeffs (np.ndarray): MDCT coefficients.
        N (int): Block size.
        extension_mode (str): Signal extension mode (should match analysis).

    Returns:
        np.ndarray: Reconstructed 1D signal.
    """
    L = len(coeffs)
    window = create_mdct_window(N)
    num_blocks = L // N

    # Output buffer with padding for overlap-add
    output = np.zeros(L + 2 * N)

    for b in range(num_blocks):
        # Get N coefficients for this block
        X = coeffs[b * N:(b + 1) * N]
        # Compute IMDCT (produces 2N samples)
        block = imdct_1d(X, N)
        # Apply synthesis window
        block *= window
        # Overlap-add
        start = b * N
        output[start:start + 2 * N] += block

    # Extract the valid portion
    return output[N:N + L]


def mdct_analyze_2d(img, N, extension_mode='symmetric'):
    """
    Apply 2D MDCT analysis (separable: rows then columns).

    Uses signal extension at image boundaries to minimize border artifacts,
    which is especially important for large block sizes.

    Parameters:
        img (np.ndarray): Input image (H x W x C).
        N (int): Block size.
        extension_mode (str): Signal extension mode ('symmetric', 'reflect', etc.)

    Returns:
        np.ndarray: MDCT coefficients.
    """
    h, w, c = img.shape
    output = img.astype(np.float64).copy()

    # Apply to rows
    for ch in range(c):
        for row in range(h):
            output[row, :, ch] = mdct_analyze_1d(output[row, :, ch], N, extension_mode)

    # Apply to columns
    for ch in range(c):
        for col in range(w):
            output[:, col, ch] = mdct_analyze_1d(output[:, col, ch], N, extension_mode)

    return output.astype(np.float32)


def mdct_synthesize_2d(coeffs, N, extension_mode='symmetric'):
    """
    Apply 2D MDCT synthesis (separable: columns then rows).

    The extension_mode should match the one used in mdct_analyze_2d.

    Parameters:
        coeffs (np.ndarray): MDCT coefficients (H x W x C).
        N (int): Block size.
        extension_mode (str): Signal extension mode (should match analysis).

    Returns:
        np.ndarray: Reconstructed image.
    """
    h, w, c = coeffs.shape
    output = coeffs.astype(np.float64).copy()

    # Reverse order: columns first
    for ch in range(c):
        for col in range(w):
            output[:, col, ch] = mdct_synthesize_1d(output[:, col, ch], N, extension_mode)

    # Then rows
    for ch in range(c):
        for row in range(h):
            output[row, :, ch] = mdct_synthesize_1d(output[row, :, ch], N, extension_mode)

    return output.astype(np.float32)


def calculate_rmse(image1_path, image2_path):
    """
    Calculate Root Mean Square Error (RMSE) between two images.
    """
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    if arr1.shape != arr2.shape:
        logging.warning(f"Image dimensions do not match: {arr1.shape} vs {arr2.shape}")
        return None, None

    mse = np.mean((arr1 - arr2) ** 2)
    rmse = np.sqrt(mse)
    return rmse, arr1.shape


def get_file_size(file_path):
    """
    Get file size in bytes.
    """
    try:
        return os.path.getsize(file_path)
    except FileNotFoundError:
        return 0


def calculate_J(original_path, codestream_prefix, decoded_path):
    """
    Calculate the Rate/Distortion efficiency (J = R + D).

    Parameters:
        original_path (str): Path to original image
        codestream_prefix (str): Prefix for codestream files (without extension)
        decoded_path (str): Path to decoded image

    Returns:
        float: J value (R + D), or None if calculation fails
    """
    # Calculate RMSE
    rmse, shape = calculate_rmse(original_path, decoded_path)
    if rmse is None:
        return None

    # Calculate codestream size in bytes
    codestream_pattern = codestream_prefix + '*'
    codestream_files = glob.glob(codestream_pattern)
    codestream_bytes = sum(get_file_size(f) for f in codestream_files)

    # Calculate bits per pixel
    if shape is None:
        return None
    number_of_pixels = shape[0] * shape[1]
    codestream_bpp = (codestream_bytes * 8) / number_of_pixels

    # Calculate J
    J = codestream_bpp + rmse

    logging.info(f"=== Rate/Distortion Efficiency ===")
    logging.info(f"Original: {original_path}")
    logging.info(f"Codestream files: {codestream_files}")
    logging.info(f"Codestream size: {codestream_bytes} bytes ({codestream_bpp:.2f} bits/pixel)")
    logging.info(f"Decoded: {decoded_path}")
    logging.info(f"Image shape: {shape}")
    logging.info(f"Distortion (RMSE): {rmse:.2f}")
    logging.info(f"J = R + D = {codestream_bpp:.2f} + {rmse:.2f} = {J:.2f}")

    return J


# =============================================================================
# MDCT CoDec Class - Uses MDCT (Malvar's Lapped Transform)
# =============================================================================

class CoDec(CT.CoDec):
    """
    Codec using MDCT (Modified Discrete Cosine Transform).

    This implements Malvar's Modulated Lapped Transform, which is the
    theoretically correct way to eliminate blocking artifacts. The MDCT:
    - Uses 50% overlapping windows
    - Provides perfect reconstruction (without quantization)
    - Is critically sampled (same number of coefficients as samples)
    - Is used in MP3, AAC, Vorbis, and other successful codecs
    """

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        self.args = args
        self.block_size = args.block_size_MDCT
        self.use_mdct = True
        # MDCT scaling factor: Empirically determined to match DCT coefficient range
        # Different strategies for different quantizers:
        # - deadzone: Works well with moderate scaling
        # - LloydMax: Needs precise DCT-equivalent scaling
        # For LloydMax, we need to match DCT's statistical distribution exactly
        if args.quantizer == "LloydMax":
            # For LloydMax, use a more aggressive scaling to match DCT range
            # Empirically: DCT coeffs are in ~[-1000, 1000], MDCT in ~[-350, 1676]
            # Ratio: MDCT/DCT ≈ 1.5-2x, so scale by block_size/1.5
            self.mdct_scale_factor = float(self.block_size) / 1.5
        elif self.block_size <= 8:
            self.mdct_scale_factor = float(self.block_size) / 2.0
        elif self.block_size >= 32:
            self.mdct_scale_factor = float(self.block_size) / 4.0
        else:
            # Linear interpolation between 8 and 32
            t = (self.block_size - 8) / (32 - 8)
            scale_8 = 8.0 / 2.0
            scale_32 = 32.0 / 4.0
            self.mdct_scale_factor = scale_8 + t * (scale_32 - scale_8)

        logging.info(f"MDCT: block_size = {self.block_size}, scale_factor = {self.mdct_scale_factor:.2f}, quantizer = {args.quantizer}")
        logging.debug(f"block_size = {self.block_size}")
        logging.debug(f"MDCT scale factor = {self.mdct_scale_factor:.2f}")

        if args.perceptual_quantization:
            # JPEG standard quantization matrices
            self.Y_QSSs = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
                                    [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
                                    [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
                                    [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]]).astype(np.float32)
            self.C_QSSs = np.array([[17,18,24,47,99,99,99,99],[18,21,26,66,99,99,99,99],
                                    [24,26,56,99,99,99,99,99],[47,66,99,99,99,99,99,99],
                                    [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],
                                    [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99]]).astype(np.float32)
            inter = cv2.INTER_AREA if self.block_size < 8 else cv2.INTER_LINEAR
            self.C_QSSs = cv2.resize(self.C_QSSs, (self.block_size, self.block_size), interpolation=inter).astype(np.float32)
            self.Y_QSSs = cv2.resize(self.Y_QSSs, (self.block_size, self.block_size), interpolation=inter).astype(np.float32)

            self.Y_QSSs_max = np.max(self.Y_QSSs) if np.max(self.Y_QSSs) > 0 else 1.0
            self.C_QSSs_max = np.max(self.C_QSSs) if np.max(self.C_QSSs) > 0 else 1.0

        self.offset = 128 if args.quantizer == "deadzone" else 0

    def pad_and_center_to_multiple_of_block_size(self, img):
        """
        Pad image to multiple of block size, centering the original content.
        Uses symmetric extension to minimize border artifacts.
        Extended padding (2*block_size) ensures MDCT has enough context at borders.
        """
        logging.debug("trace")
        if img.ndim != 3:
            raise ValueError("Input image must be a 3D array.")
        self.original_shape = img.shape
        h, w, c = img.shape

        # Calculate target size: multiple of block_size, plus extra for MDCT overlap
        # Add block_size on each side to ensure proper MDCT context at boundaries
        extra_pad = self.block_size
        th = ((h + 2*extra_pad + self.block_size - 1) // self.block_size) * self.block_size
        tw = ((w + 2*extra_pad + self.block_size - 1) // self.block_size) * self.block_size

        ph, pw = th - h, tw - w
        top, left = ph // 2, pw // 2

        # Use symmetric padding to reduce border artifacts
        # This mirrors the signal at boundaries, providing smooth continuation
        padded_img = np.pad(img, ((top, ph-top), (left, pw-left), (0,0)), mode='symmetric')

        # Store padding info for removal
        self.pad_top = top
        self.pad_left = left
        self.padded_shape = padded_img.shape

        return padded_img

    def remove_padding(self, img):
        """
        Remove padding from image to restore original dimensions.
        """
        logging.debug("trace")
        if img.ndim != 3:
            raise ValueError("Input must be 3D.")
        if self.original_shape is None:
            raise ValueError("Original shape not set.")
        oh, ow, _ = self.original_shape

        # Use stored padding info if available, otherwise calculate
        if hasattr(self, 'pad_top') and hasattr(self, 'pad_left'):
            top, left = self.pad_top, self.pad_left
        else:
            ph, pw, _ = img.shape
            top, left = (ph - oh)//2, (pw - ow)//2

        return img[top:top+oh, left:left+ow, :]

    def encode_fn(self, in_fn, out_fn):
        """
        Encode image using MDCT (Malvar's Lapped Transform).
        """
        logging.debug("trace")
        # Force INFO logging for diagnostics
        logging.getLogger().setLevel(logging.INFO)
        img = self.encode_read_fn(in_fn).astype(np.float32)
        self.original_shape = img.shape
        img = self.pad_and_center_to_multiple_of_block_size(img)

        # Save original shape and padding info
        with open(f"{out_fn}_shape.bin", "wb") as f:
            f.write(struct.pack("iii", *self.original_shape))
            f.write(struct.pack("ii", self.pad_top, self.pad_left))

        # Follow DCT pattern: subtract offset BEFORE transform
        img -= self.offset

        # Color transform: RGB → YCoCg
        ct_img = from_RGB(img)

        # MDCT analysis
        if self.use_mdct:
            mdct_coeffs = mdct_analyze_2d(ct_img, self.block_size)
            # Normalize MDCT coefficients to match DCT range
            mdct_coeffs /= self.mdct_scale_factor
        else:
            mdct_coeffs = space_analyze(ct_img, self.block_size, self.block_size)

        # DEBUG: Print coefficient statistics
        logging.info(f"MDCT coeffs after transform: min={np.min(mdct_coeffs):.2f}, max={np.max(mdct_coeffs):.2f}, mean={np.mean(mdct_coeffs):.2f}, std={np.std(mdct_coeffs):.2f}")

        # Perceptual quantization scaling BEFORE subbands
        if self.args.perceptual_quantization:
            logging.debug(f"Using perceptual quantization with block_size = {self.block_size}")
            mdct_coeffs = mdct_coeffs.astype(np.float32)
            blocks_in_y = int(ct_img.shape[0]/self.block_size)
            blocks_in_x = int(ct_img.shape[1]/self.block_size)
            for by in range(blocks_in_y):
                for bx in range(blocks_in_x):
                    y_start = by*self.block_size
                    y_end = (by+1)*self.block_size
                    x_start = bx*self.block_size
                    x_end = (bx+1)*self.block_size
                    # Multiply by (QSSs/max) like DCT
                    mdct_coeffs[y_start:y_end, x_start:x_end, 0] *= (self.Y_QSSs/self.Y_QSSs_max)
                    mdct_coeffs[y_start:y_end, x_start:x_end, 1] *= (self.C_QSSs/self.C_QSSs_max)
                    mdct_coeffs[y_start:y_end, x_start:x_end, 2] *= (self.C_QSSs/self.C_QSSs_max)

            # DEBUG: Print after perceptual scaling
            logging.info(f"MDCT coeffs after perceptual: min={np.min(mdct_coeffs):.2f}, max={np.max(mdct_coeffs):.2f}, mean={np.mean(mdct_coeffs):.2f}, std={np.std(mdct_coeffs):.2f}")

        # Coefficients reordering in subbands
        if self.args.disable_subbands:
            decom_img = mdct_coeffs
        else:
            decom_img = get_subbands(mdct_coeffs, self.block_size, self.block_size)

        # DEBUG: Print before quantization
        logging.info(f"Before quantization: min={np.min(decom_img):.2f}, max={np.max(decom_img):.2f}, mean={np.mean(decom_img):.2f}, std={np.std(decom_img):.2f}")

        # Quantization and compression
        decom_k = self.quantize_decom(decom_img)

        # DEBUG: Print quantization output
        logging.info(f"After quantization: min={np.min(decom_k):.2f}, max={np.max(decom_k):.2f}, mean={np.mean(decom_k):.2f}")

        # Handle different quantizers appropriately
        if self.args.quantizer == "LloydMax":
            # LloydMax returns indices in [0, QSS-1], which fits in uint8 if QSS <= 256
            # No offset needed, just ensure proper type
            if self.args.QSS <= 256:
                decom_k = np.clip(decom_k, 0, 255).astype(np.uint8)
            else:
                # For QSS > 256, use uint16
                decom_k = np.clip(decom_k, 0, 65535).astype(np.uint16)
        else:
            # Deadzone quantizer: add offset and clip to [0, 255]
            decom_k += self.offset
            decom_k = np.clip(decom_k, 0, 255).astype(np.uint8)

        decom_k = self.compress(decom_k)
        output_size = self.encode_write_fn(decom_k, out_fn)
        return output_size

    def encode(self, in_fn="/tmp/original.png", out_fn="/tmp/encoded"):
        return self.encode_fn(in_fn, out_fn)

    def decode_fn(self, in_fn, out_fn):
        """
        Decode code-stream using MDCT (Malvar's Lapped Transform).
        """
        logging.debug("trace")
        decom_k = self.decode_read_fn(in_fn)

        # Read original shape and padding info
        with open(f"{in_fn}_shape.bin", "rb") as f:
            self.original_shape = struct.unpack("iii", f.read(12))
            try:
                self.pad_top, self.pad_left = struct.unpack("ii", f.read(8))
            except struct.error:
                # Fallback for old format without padding info
                self.pad_top = None
                self.pad_left = None

        # Decompress and dequantize
        decom_k = self.decompress(decom_k)

        # Handle different quantizers appropriately in decode
        if self.args.quantizer == "LloydMax":
            # LloydMax: indices are already in correct range, no offset to remove
            decom_k = decom_k.astype(np.int16)
        else:
            # Deadzone: remove offset
            decom_k = decom_k.astype(np.int16)
            decom_k -= self.offset

        # DEBUG: Print after decompression
        logging.info(f"After decompression: min={np.min(decom_k):.2f}, max={np.max(decom_k):.2f}, mean={np.mean(decom_k):.2f}")

        decom_y = self.dequantize_decom(decom_k)

        # Reconstruct blocks from subbands
        if self.args.disable_subbands:
            mdct_coeffs = decom_y
        else:
            mdct_coeffs = get_blocks(decom_y, self.block_size, self.block_size)

        # Perceptual dequantization
        if self.args.perceptual_quantization:
            logging.debug(f"Using perceptual dequantization with block_size = {self.block_size}")
            mdct_coeffs = mdct_coeffs.astype(np.float32)
            blocks_in_y = int(mdct_coeffs.shape[0]/self.block_size)
            blocks_in_x = int(mdct_coeffs.shape[1]/self.block_size)
            for by in range(blocks_in_y):
                for bx in range(blocks_in_x):
                    y_start = by*self.block_size
                    y_end = (by+1)*self.block_size
                    x_start = bx*self.block_size
                    x_end = (bx+1)*self.block_size

                    mdct_coeffs[y_start:y_end, x_start:x_end, 0] /= (self.Y_QSSs/self.Y_QSSs_max)
                    mdct_coeffs[y_start:y_end, x_start:x_end, 1] /= (self.C_QSSs/self.C_QSSs_max)
                    mdct_coeffs[y_start:y_end, x_start:x_end, 2] /= (self.C_QSSs/self.C_QSSs_max)

        # IMDCT synthesis
        if self.use_mdct:
            ct_y = mdct_synthesize_2d(mdct_coeffs, self.block_size)
            # Denormalize to restore original scale
            ct_y *= self.mdct_scale_factor
        else:
            ct_y = space_synthesize(mdct_coeffs, self.block_size, self.block_size)

        # Inverse color transform: YCoCg → RGB
        y = to_RGB(ct_y)

        # Remove padding and add offset back
        y = self.remove_padding(y)
        y += self.offset
        y = np.clip(y, 0, 255).astype(np.uint8)
        output_size = self.decode_write_fn(y, out_fn)

        # Calculate and log J (Rate/Distortion Efficiency)
        J = calculate_J("/tmp/original.png", in_fn, out_fn)
        if J is not None:
            logging.info(f"Rate/Distortion Efficiency: J = {J:.2f}")

        return output_size

    def decode(self, in_fn="/tmp/encoded", out_fn="/tmp/decoded.png"):
        return self.decode_fn(in_fn, out_fn)

    def quantize_decom(self, decom):
        logging.debug("trace")
        result = self.quantize(decom)
        # DEBUG: Check quantization output
        logging.info(f"quantize_decom output: min={np.min(result):.2f}, max={np.max(result):.2f}, dtype={result.dtype}")
        if self.args.quantizer == "LloydMax":
            logging.info(f"LloydMax quantization - QSS={self.args.QSS}, expected range=[0, {self.args.QSS-1}]")
            unique_vals = np.unique(result)
            logging.info(f"Unique quantized values: {len(unique_vals)} values in range [{np.min(unique_vals)}, {np.max(unique_vals)}]")
        return result

    def dequantize_decom(self, decom_k):
        logging.debug("trace")
        return self.dequantize(decom_k)

if __name__ == "__main__":
    main.main(local_parser.parser, logging, CoDec)
