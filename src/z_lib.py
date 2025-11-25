'''Entropy Encoding of images using zlib.'''

import io
import numpy as np
import main
import logging
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_image_coding as EIC

class CoDec (EIC.CoDec):

    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        self.file_extension = ".npz"

    def compress(self, img):
        logging.debug(f"trace img={img}")
        compressed_img = io.BytesIO()
        np.savez_compressed(file=compressed_img, a=img)
        return compressed_img

    def decompress(self, compressed_img):
        logging.debug(f"trace compressed_img={compressed_img}")
        compressed_img = io.BytesIO(compressed_img)
        img = np.load(compressed_img)['a']
        return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
