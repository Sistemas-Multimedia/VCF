'''Entropy Encoding of images using PNG (Portable Network Graphics).'''

import logging
import os
import main
import zlib
import PNG as EC
import numpy as np

from skimage import io
from sklearn.cluster import KMeans

TEMP_PATH = '/tmp/temp.png'


class CoDec(EC.CoDec):

    def __init__(self, args):  # ??
        super().__init__(args)

    def encode(self):
        img = self.read()
        io.imsave(TEMP_PATH, img)

        f = open(TEMP_PATH, "rb")
        img = f.read()
        f.close()

        comp = zlib.compress(img)

        os.remove(TEMP_PATH)

        f = open(self.args.output, "wb")
        f.write(comp)
        f.close()
        self.output_bytes = os.path.getsize(self.args.output)

        return 0

    def decode(self):
        self.input_bytes = os.path.getsize(self.args.input)
        f = open(self.args.input, "rb")
        img = f.read()
        f.close()

        comp = zlib.decompress(img)

        f = open(self.args.output, "wb")
        f.write(comp)
        f.close()
        self.output_bytes = os.path.getsize(self.args.output)

        return 0

    def __del__(self):
        logging.info(f"Total {self.input_bytes} bytes read")
        logging.info(f"Total {self.output_bytes} bytes written")


if __name__ == "__main__":
    main.main(EC.parser, logging, CoDec)
