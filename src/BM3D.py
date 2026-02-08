'''Image denoising using BM3D filter. *** Effective only when decoding! ***'''

import numpy as np
import logging
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)
import parser
import main
import importlib
import cv2

default_sigma = 25.0

parser.parser_decode.add_argument("-s", "--sigma", type=parser.int_or_str, help=f"Sigma for BM3D denoising (default: {default_sigma})", default=default_sigma)
import no_filter

args = parser.parser.parse_known_args()[0]

class CoDec(no_filter.CoDec):

    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        self.args = args

    def decode(self):
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")        
        y = self.filter(k)
        output_size = self.decode_write(y)
        return output_size

    def filter(self, img):
        logging.debug(f"trace y={img}")
        logging.info(f"BM3D sigma={self.args.sigma}")

        sigma = float(self.args.sigma)

        try:
            from bm3d import bm3d_rgb
        except ImportError:
            raise SystemExit(
                "No tienes instalada la librería bm3d.\n"
                "Instala con:\n"
                "  pip install bm3d\n"
            )
        # Convierte la imagen de VCF (que son números 0-255) a decimales entre 0 y 1
        img_float = img.astype(np.float32) / 255.0

        # Ejecución del algoritmo BM3D
        sigma_psd = sigma / 255.0
        denoised = bm3d_rgb(img_float, sigma_psd=sigma_psd)

        # Convierte la imagen de vuelta a uint8 (0-255)
        denoised_uint8 = np.clip(denoised * 255.0 + 0.5, 0, 255).astype(np.uint8)

        return denoised_uint8

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
