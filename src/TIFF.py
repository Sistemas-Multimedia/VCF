'''Entropy Encoding of images using TIFF (Tag Image File Format). '''

import tifffile
import io as pyio  # Avoid conflict with skimage.io
import main
import logging
import numpy as np
import cv2 as cv # pip install opencv-python
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_image_coding as EIC

COMPRESSION_LEVEL = 9

class CoDec(EIC.CoDec):

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        self.file_extension = ".tif"

    def compress(self, img):
        logging.debug("trace")
        logging.debug(f"img.dtype={img.dtype}")
        assert (img.dtype == np.uint8) or (img.dtype == np.uint16), f"current type = {img.dtype}"
        #assert (img.dtype == np.uint8), f"current type = {img.dtype}"
        compressed_img = pyio.BytesIO()
        tifffile.imwrite(compressed_img, data=img, compression='zlib')
        compressed_img.seek(0)
        return compressed_img

    def decompress(self, compressed_img):
        logging.debug("trace")
        compressed_img = pyio.BytesIO(compressed_img)
        img = tifffile.imread(compressed_img)
        logging.debug(f"img.dtype={img.dtype}")
        logging.debug(f"img = {img}")
        return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
