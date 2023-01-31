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
EXTENSION = '.zlib'


class CoDec(EC.CoDec):

    def __init__(self, args):  # ??
        super().__init__(args)

    def spatial_predictor(self, img):
        rows = img.shape[0]
        cols = img.shape[1]

        img = img.reshape(rows*cols, 3)
        kmeans = KMeans(n_clusters=64, n_init='auto')
        kmeans.fit(img)

        ci = kmeans.cluster_centers_[kmeans.labels_]
        ci = np.clip(ci.astype('uint8'), 0, 255)

        ci = ci.reshape(rows, cols, 3)
        io.imsave(TEMP_PATH, ci)
        return

    def encode(self):
        img = self.read()
        self.spatial_predictor(img)

        f = open(TEMP_PATH, "rb")
        img = f.read()
        f.close()

        comp = zlib.compress(img)

        f = open(self.args.output + EXTENSION, "wb")
        f.write(comp)
        f.close()
        self.output_bytes = os.path.getsize(self.args.output + EXTENSION)

        return 0

    def decode(self):
        self.input_bytes = os.path.getsize(self.args.input + EXTENSION)
        f = open(self.args.input + EXTENSION, "rb")
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
