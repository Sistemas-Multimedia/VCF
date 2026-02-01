"""
MCTF: Motion Compensated Temporal Filtering codec.

Este módulo implementa un codec de video basado en filtrado temporal
con compensación de movimiento usando lifting scheme.

Estructura GOP: IBB... (I-frame seguido de B-frames, SIN P-frames)
- Frame I: codificado independientemente (intra)
- Frames B: predichos bidireccionalmente

Referencias:
    - Ohm, J.R. (1994). "Three-dimensional subband coding with motion compensation"
    - Pesquet-Popescu, B., Bottreau, V. (2001). "Three-dimensional lifting schemes"
    - González-Ruiz, V. "MCTF" https://github.com/vicente-gonzalez-ruiz/motion_compensated_temporal_filtering
"""

import sys
import os
import logging
import numpy as np
import cv2
import av
from PIL import Image
import importlib
import pickle
import tempfile

import main
import platform_utils as pu

# Inicialización multiplataforma
TMP_DIR = pu.get_vcf_temp_dir()
pu.ensure_description_file(__doc__)

import parser
import entropy_video_coding as EVC

from motion_estimation import block_matching_bidirectional
from motion_compensation import motion_compensate
from temporal_filtering import temporal_filter_lifting, inverse_temporal_filter_lifting

# Parámetros por defecto
DEFAULT_GOP_SIZE = 16
DEFAULT_TEMPORAL_LEVELS = 4
DEFAULT_BLOCK_SIZE = 16
DEFAULT_SEARCH_RANGE = 16
DEFAULT_WAVELET_TYPE = '5/3'

# Parser para encoder - usamos --video_input para evitar conflicto con -o de entropy_image_coding
parser.parser_encode.add_argument("-V", "--video_input", type=parser.int_or_str,
    help=f"Input video (default: {EVC.ENCODE_INPUT})",
    default=EVC.ENCODE_INPUT)
parser.parser_encode.add_argument("-O", "--video_output", type=parser.int_or_str,
    help=f"Output prefix (default: {EVC.ENCODE_OUTPUT_PREFIX})",
    default=EVC.ENCODE_OUTPUT_PREFIX)
parser.parser_encode.add_argument("-T", "--transform", type=str,
    help=f"2D-transform (default: {EVC.DEFAULT_TRANSFORM})",
    default=EVC.DEFAULT_TRANSFORM)
parser.parser_encode.add_argument("-N", "--number_of_frames", type=parser.int_or_str,
    help=f"Number of frames to encode (default: {EVC.N_FRAMES})",
    default=f"{EVC.N_FRAMES}")
parser.parser_encode.add_argument("--gop_size", type=int,
    help=f"GOP size (default: {DEFAULT_GOP_SIZE})",
    default=DEFAULT_GOP_SIZE)
parser.parser_encode.add_argument("--temporal_levels", type=int,
    help=f"Temporal decomposition levels (default: {DEFAULT_TEMPORAL_LEVELS})",
    default=DEFAULT_TEMPORAL_LEVELS)
parser.parser_encode.add_argument("--block_size", type=int,
    help=f"Block size for motion estimation (default: {DEFAULT_BLOCK_SIZE})",
    default=DEFAULT_BLOCK_SIZE)
parser.parser_encode.add_argument("--search_range", type=int,
    help=f"Search range for motion estimation (default: {DEFAULT_SEARCH_RANGE})",
    default=DEFAULT_SEARCH_RANGE)
parser.parser_encode.add_argument("--wavelet_type", type=str,
    help=f"Temporal wavelet type: haar, 5/3, 9/7 (default: {DEFAULT_WAVELET_TYPE})",
    default=DEFAULT_WAVELET_TYPE)

# Parser para decoder - usamos --video_input para evitar conflicto
parser.parser_decode.add_argument("-V", "--video_input", type=parser.int_or_str,
    help=f"Input MCTF stream prefix (default: {EVC.ENCODE_OUTPUT_PREFIX})",
    default=EVC.ENCODE_OUTPUT_PREFIX)
parser.parser_decode.add_argument("-O", "--video_output", type=parser.int_or_str,
    help=f"Output prefix (default: {EVC.DECODE_OUTPUT_PREFIX})",
    default=EVC.DECODE_OUTPUT_PREFIX)
parser.parser_decode.add_argument("-T", "--transform", type=str,
    help=f"2D-transform (default: {EVC.DEFAULT_TRANSFORM})",
    default=EVC.DEFAULT_TRANSFORM)
parser.parser_decode.add_argument("-N", "--number_of_frames", type=parser.int_or_str,
    help=f"Number of frames to decode (default: {EVC.N_FRAMES})",
    default=f"{EVC.N_FRAMES}")
parser.parser_decode.add_argument("--gop_size", type=int,
    help=f"GOP size (default: {DEFAULT_GOP_SIZE})",
    default=DEFAULT_GOP_SIZE)
parser.parser_decode.add_argument("--temporal_levels", type=int,
    help=f"Temporal decomposition levels (default: {DEFAULT_TEMPORAL_LEVELS})",
    default=DEFAULT_TEMPORAL_LEVELS)
parser.parser_decode.add_argument("--block_size", type=int,
    help=f"Block size for motion estimation (default: {DEFAULT_BLOCK_SIZE})",
    default=DEFAULT_BLOCK_SIZE)
parser.parser_decode.add_argument("--search_range", type=int,
    help=f"Search range for motion estimation (default: {DEFAULT_SEARCH_RANGE})",
    default=DEFAULT_SEARCH_RANGE)
parser.parser_decode.add_argument("--wavelet_type", type=str,
    help=f"Temporal wavelet type: haar, 5/3, 9/7 (default: {DEFAULT_WAVELET_TYPE})",
    default=DEFAULT_WAVELET_TYPE)

args = parser.parser.parse_known_args()[0]

# Importar transformada espacial
if __debug__:
    if args.debug:
        print(f"MCTF: Importing {args.transform}")

try:
    transform = importlib.import_module(args.transform)
except ImportError as e:
    print(f"Error: Could not find {args.transform} module ({e})")
    sys.exit(1)


class CoDec(EVC.CoDec):
    """
    Codec MCTF (Motion Compensated Temporal Filtering).
    
    Implementa compresión de video usando:
    1. Estimación de movimiento bidireccional
    2. Filtrado temporal con lifting scheme (Predict + Update)
    3. Transformada espacial 2D (DCT o DWT)
    4. Cuantización y codificación entrópica
    """

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        
        # Configuración MCTF
        self.gop_size = args.gop_size
        self.temporal_levels = args.temporal_levels
        self.block_size = args.block_size
        self.search_range = args.search_range
        self.wavelet_type = args.wavelet_type
        
        # Codec de transformada espacial
        self.transform_codec = transform.CoDec(args)
        
        logging.info(f"MCTF Codec initialized:")
        logging.info(f"  GOP size: {self.gop_size}")
        logging.info(f"  Temporal levels: {self.temporal_levels}")
        logging.info(f"  Block size: {self.block_size}")
        logging.info(f"  Search range: {self.search_range}")
        logging.info(f"  Wavelet type: {self.wavelet_type}")
        logging.info(f"  Spatial transform: {args.transform}")

    def bye(self):
        """Sobrescribe bye() para usar video_input/video_output."""
        logging.debug("trace")
        if __debug__:
            if self.encoding:
                BPP = (self.total_output_size*8)/(self.N_frames*self.width*self.height)
                logging.info(f"Output bit-rate = {BPP} bits/pixel")
                # Guardar metadatos
                with open(f"{self.args.video_output}.txt", 'w') as f:
                    f.write(f"{self.args.video_input}\n")
                    f.write(f"{self.N_frames}\n")
                    f.write(f"{self.height}\n")
                    f.write(f"{self.width}\n")
                    f.write(f"{BPP}\n")
            else:
                # Leer metadatos y calcular distorsión
                with open(f"{self.args.video_input}.txt", 'r') as f:
                    original_file = f.readline().strip()
                    logging.info(f"original_file = {original_file}")
                    N_frames = int(f.readline().strip())
                    logging.info(f"N_frames = {N_frames}")
                    height = f.readline().strip()
                    logging.info(f"video height = {height} pixels")
                    width = f.readline().strip()
                    logging.info(f"video width = {width} pixels")
                    BPP = float(f.readline().strip())
                    logging.info(f"BPP = {BPP}")

    def encode(self):
        """
        Codifica un video usando MCTF.

        Proceso:
        1. Lee frames del video de entrada
        2. Agrupa frames en GOPs
        3. Para cada GOP:
           a. Estima movimiento bidireccional
           b. Aplica filtrado temporal (lifting)
           c. Codifica frames L y H con transformada espacial
        """
        logging.debug("trace")
        fn = self.args.video_input
        logging.info(f"MCTF Encoding {fn}")

        # Leer video y extraer frames
        container = av.open(fn)
        frames = []
        img_counter = 0

        for packet in container.demux():
            if __debug__:
                self.total_input_size += packet.size
            for frame in packet.decode():
                img = frame.to_image()
                img_array = np.array(img.convert('L'))  # Grayscale para MCTF
                frames.append(img_array)

                # Guardar original para comparación
                if __debug__:
                    img_fn = os.path.join(TMP_DIR, f"original_{img_counter:04d}.png")
                    img.save(img_fn)

                img_counter += 1
                if img_counter >= self.args.number_of_frames:
                    break
            if img_counter >= self.args.number_of_frames:
                break

        self.N_frames = len(frames)
        self.height, self.width = frames[0].shape
        logging.info(f"Read {self.N_frames} frames of size {self.width}x{self.height}")

        # Procesar GOPs
        gop_data_list = []
        frame_idx = 0
        gop_counter = 0

        while frame_idx < self.N_frames:
            gop_end = min(frame_idx + self.gop_size, self.N_frames)
            gop_frames = frames[frame_idx:gop_end]

            logging.info(f"Processing GOP {gop_counter}: frames {frame_idx}-{gop_end-1}")

            # Codificar GOP
            gop_data = self._encode_gop(gop_frames, gop_counter)
            gop_data_list.append(gop_data)

            frame_idx = gop_end
            gop_counter += 1

        # Guardar stream codificado
        self._write_mctf_stream(gop_data_list)

        logging.info(f"MCTF encoding complete: {gop_counter} GOPs")

    def _encode_gop(self, gop_frames, gop_idx):
        """Codifica un GOP usando MCTF."""
        n_frames = len(gop_frames)

        if n_frames < 2:
            # GOP muy pequeño: codificar como intra
            return self._encode_intra_only(gop_frames, gop_idx)

        # === Paso 1: Estimación de movimiento ===
        logging.info(f"  Motion estimation for {n_frames} frames")
        mv_forward_list = []
        mv_backward_list = []

        for i in range(n_frames):
            frame_current = gop_frames[i]
            frame_prev = gop_frames[max(0, i - 1)]
            frame_next = gop_frames[min(n_frames - 1, i + 1)]

            mv_fwd, mv_bwd = block_matching_bidirectional(
                frame_current, frame_prev, frame_next,
                self.block_size, self.search_range
            )
            mv_forward_list.append(mv_fwd)
            mv_backward_list.append(mv_bwd)

        # === Paso 2: Filtrado temporal (lifting) ===
        logging.info(f"  Temporal filtering with {self.wavelet_type} wavelet")
        low_pass, high_pass = temporal_filter_lifting(
            gop_frames,
            mv_forward_list,
            mv_backward_list,
            self.wavelet_type,
            self.block_size
        )

        # === Paso 3: Codificar frames L y H con transformada espacial ===
        logging.info(f"  Encoding {len(low_pass)} L frames and {len(high_pass)} H frames")

        encoded_low = []
        for i, l_frame in enumerate(low_pass):
            l_frame_uint8 = np.clip(l_frame, 0, 255).astype(np.uint8)
            fn_l = os.path.join(TMP_DIR, f"gop{gop_idx:02d}_L_{i:02d}")
            self._save_and_encode_frame(l_frame_uint8, fn_l)
            encoded_low.append(fn_l)

        encoded_high = []
        for i, h_frame in enumerate(high_pass):
            # Residuos pueden ser negativos: offset a positivo
            h_frame_offset = h_frame + 128
            h_frame_uint8 = np.clip(h_frame_offset, 0, 255).astype(np.uint8)
            fn_h = os.path.join(TMP_DIR, f"gop{gop_idx:02d}_H_{i:02d}")
            self._save_and_encode_frame(h_frame_uint8, fn_h)
            encoded_high.append(fn_h)

        return {
            'n_frames': n_frames,
            'mv_forward': mv_forward_list,
            'mv_backward': mv_backward_list,
            'encoded_low': encoded_low,
            'encoded_high': encoded_high,
            'wavelet_type': self.wavelet_type
        }

    def _encode_intra_only(self, frames, gop_idx):
        """Codifica frames como intra solamente."""
        encoded_frames = []
        for i, frame in enumerate(frames):
            fn = os.path.join(TMP_DIR, f"gop{gop_idx:02d}_I_{i:02d}")
            self._save_and_encode_frame(frame, fn)
            encoded_frames.append(fn)

        return {
            'n_frames': len(frames),
            'intra_only': True,
            'encoded_frames': encoded_frames
        }

    def _save_and_encode_frame(self, frame, fn_prefix):
        """Guarda y codifica un frame usando TIFF con compresión lossless."""
        # Convertir a RGB si es grayscale
        if len(frame.shape) == 2:
            frame_rgb = np.stack([frame] * 3, axis=-1)
        else:
            frame_rgb = frame

        # Guardar como TIFF con compresión lossless
        img_fn = f"{fn_prefix}.tif"
        img = Image.fromarray(frame_rgb)
        img.save(img_fn, compression='tiff_deflate')

        output_size = os.path.getsize(img_fn)
        self.total_output_size += output_size
        return output_size

    def _write_mctf_stream(self, gop_data_list):
        """Escribe el stream MCTF codificado."""
        stream_fn = f"{self.args.video_output}.mctf"

        header = {
            'n_frames': self.N_frames,
            'height': self.height,
            'width': self.width,
            'gop_size': self.gop_size,
            'temporal_levels': self.temporal_levels,
            'block_size': self.block_size,
            'wavelet_type': self.wavelet_type,
            'num_gops': len(gop_data_list)
        }

        with open(stream_fn, 'wb') as f:
            pickle.dump(header, f)
            for gop_data in gop_data_list:
                pickle.dump(gop_data, f)

        stream_size = os.path.getsize(stream_fn)
        self.total_output_size += stream_size
        logging.info(f"Written MCTF stream: {stream_fn} ({stream_size} bytes)")

    def decode(self):
        """
        Decodifica un video codificado con MCTF.

        Proceso inverso:
        1. Lee stream MCTF
        2. Para cada GOP:
           a. Decodifica frames L y H con transformada espacial inversa
           b. Aplica filtrado temporal inverso (lifting inverso)
        3. Escribe frames decodificados
        """
        logging.debug("trace")
        logging.info(f"MCTF Decoding {self.args.video_input}")

        # Leer stream MCTF
        stream_fn = f"{self.args.video_input}.mctf"
        header, gop_data_list = self._read_mctf_stream(stream_fn)

        self.N_frames = header['n_frames']
        self.height = header['height']
        self.width = header['width']

        logging.info(f"Decoding {self.N_frames} frames of size {self.width}x{self.height}")
        logging.info(f"  {header['num_gops']} GOPs")

        # Decodificar cada GOP
        all_frames = []
        for gop_idx, gop_data in enumerate(gop_data_list):
            logging.info(f"Decoding GOP {gop_idx}")

            if gop_data.get('intra_only', False):
                gop_frames = self._decode_intra_only(gop_data)
            else:
                gop_frames = self._decode_gop(gop_data)

            all_frames.extend(gop_frames)

        # Escribir frames decodificados
        for i, frame in enumerate(all_frames):
            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            out_fn = os.path.join(TMP_DIR, f"decoded_{i:04d}.png")

            if len(frame_uint8.shape) == 2:
                frame_rgb = np.stack([frame_uint8] * 3, axis=-1)
            else:
                frame_rgb = frame_uint8

            Image.fromarray(frame_rgb).save(out_fn)
            logging.info(f"Decoded frame {i} to {out_fn}")

        logging.info(f"MCTF decoding complete: {len(all_frames)} frames")

    def _read_mctf_stream(self, stream_fn):
        """Lee el stream MCTF codificado."""
        with open(stream_fn, 'rb') as f:
            header = pickle.load(f)
            gop_data_list = []
            for _ in range(header['num_gops']):
                gop_data = pickle.load(f)
                gop_data_list.append(gop_data)

        return header, gop_data_list

    def _decode_gop(self, gop_data):
        """Decodifica un GOP usando MCTF inverso."""
        n_frames = gop_data['n_frames']
        mv_forward = gop_data['mv_forward']
        mv_backward = gop_data['mv_backward']
        wavelet_type = gop_data['wavelet_type']

        # === Paso 1: Decodificar frames L y H ===
        logging.info(f"  Decoding L and H frames")

        low_pass = []
        for fn_l in gop_data['encoded_low']:
            frame = self._decode_frame(fn_l)
            low_pass.append(frame.astype(np.float32))

        high_pass = []
        for fn_h in gop_data['encoded_high']:
            frame = self._decode_frame(fn_h)
            # Remover offset añadido en codificación
            frame_float = frame.astype(np.float32) - 128
            high_pass.append(frame_float)

        # === Paso 2: Filtrado temporal inverso ===
        logging.info(f"  Inverse temporal filtering with {wavelet_type} wavelet")
        reconstructed = inverse_temporal_filter_lifting(
            low_pass,
            high_pass,
            mv_forward,
            mv_backward,
            wavelet_type,
            self.block_size
        )

        return reconstructed

    def _decode_intra_only(self, gop_data):
        """Decodifica frames intra solamente."""
        frames = []
        for fn in gop_data['encoded_frames']:
            frame = self._decode_frame(fn)
            frames.append(frame.astype(np.float32))
        return frames

    def _decode_frame(self, fn_prefix):
        """Decodifica un frame leyendo el TIFF."""
        img_fn = f"{fn_prefix}.tif"

        # Leer frame desde TIFF
        img = Image.open(img_fn)
        frame = np.array(img.convert('L'))  # Grayscale
        return frame


if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)

