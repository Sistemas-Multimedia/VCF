'''Fake entropy coding using Portable aNy Map (PNM). '''

import io
import netpbmfile
import main
import logging
import numpy as np
import cv2 as cv
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import entropy_image_coding as EIC
import parser

class CoDec(EIC.CoDec):

    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        self.file_extension = ".pnm"

    def compress(self, img):
        logging.debug(f"trace img={img}")
        logging.debug(f"img.dtype={img.dtype}")
        assert (img.dtype == np.uint8) or (img.dtype == np.uint16), f"current type = {img.dtype}"
        compressed_img = io.BytesIO()
        netpbmfile.imwrite(compressed_img, img)  # It is not allowed to use netpbmfile.imwrite(file=compressed_img, data=img)
        return compressed_img

    def decompress(self, compressed_img):
        logging.debug(f"trace compressed_img={compressed_img}")
        compressed_img = io.BytesIO(compressed_img)
        img = netpbmfile.imread(compressed_img)
        logging.debug(f"img.dtype={img.dtype}")
        return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
