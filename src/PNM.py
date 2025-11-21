'''Fake entropy coding using Portable aNy Map (PNM). '''

import io
import netpbmfile
import main
import logging
import numpy as np
import cv2 as cv
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_image_coding as EIC

# Encoder parser
parser.parser_encode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {EIC.ENCODE_INPUT})", default=EIC.ENCODE_INPUT)
parser.parser_encode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {EIC.ENCODE_OUTPUT}.pnm)", default=f"{EIC.ENCODE_OUTPUT}")

# Decoder parser
parser.parser_decode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {EIC.DECODE_INPUT}.pnm)", default=f"{EIC.DECODE_INPUT}")
parser.parser_decode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {EIC.DECODE_OUTPUT})", default=f"{EIC.DECODE_OUTPUT}")    

class CoDec(EIC.CoDec):

    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        self.file_extension = ".pnm"

    def compress_fn(self, img, fn):
        logging.debug(f"trace img={img}")
        logging.debug(f"trace fn={fn}")
        logging.debug(f"img.dtype={img.dtype}")
        assert (img.dtype == np.uint8) or (img.dtype == np.uint16), f"current type = {img.dtype}"
        compressed_img = io.BytesIO()
        netpbmfile.imwrite(compressed_img, img)  # It is not allowed to use netpbmfile.imwrite(file=compressed_img, data=img)
        return compressed_img

    def decompress_fn(self, compressed_img, fn):
        logging.debug(f"trace compressed_img={compressed_img}")
        logging.debug(f"trace fn={fn}")
        compressed_img = io.BytesIO(compressed_img)
        img = netpbmfile.imread(compressed_img)
        logging.debug(f"img.dtype={img.dtype}")
        return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
