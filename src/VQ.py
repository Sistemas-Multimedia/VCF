'''Image quantization using a LloydMax quantizer.'''

# Some work could be done with the encoded histograms!

import argparse
import os
from skimage import io # pip install scikit-image
import numpy as np
import gzip
import logging
import main

# pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"
from scalar_quantization.LloydMax_quantization import LloydMax_Quantizer as Quantizer
from scalar_quantization.LloydMax_quantization import name as quantizer_name

import PNG as EC # Entropy Coding

EC.parser_encode.add_argument("-q", "--QSS", type=EC.int_or_str, help=f"Quantization step size (default: 32)", default=32)

class CoDec(EC.CoDec):
    
    def __init__(self, args): # ??
        super().__init__(args)

    def quantize(self, img):
        with open(f"{self.args.output}_QSS.txt", 'w') as f:
            f.write(f"{self.args.QSS}")
        self.output_bytes = 1 # We suppose that the representation of the QSS requires 1 byte
        if len(img.shape) < 3:
            extended_img = np.expand_dims(img, axis=2)
        else:
            extended_img = img
        k = np.empty_like(extended_img)
        for c in range(extended_img.shape[2]):
            histogram_img, bin_edges_img = np.histogram(extended_img[..., c], bins=256, range=(0, 256))
            histogram_img += 1 # Bins cannot be zeroed
            self.Q = Quantizer(Q_step=self.args.QSS, counts=histogram_img)
            centroids = self.Q.get_representation_levels()
            with gzip.GzipFile(f"{self.args.output}_centroids_{c}.gz", "w") as f:
                np.save(file=f, arr=centroids)
            len_codebook = os.path.getsize(f"{self.args.output}_centroids_{c}.gz")
            self.output_bytes += len_codebook
            k[..., c] = self.Q.encode(extended_img[..., c])
        return k

    def dequantize(self, k):

        # We ahve to undo the operatino
        with open(f"{self.args.input}_QSS.txt", 'r') as f:
            QSS = int(f.read())
        if len(k.shape) < 3:
            extended_k = np.expand_dims(k, axis=2)
        else:
            extended_k = k
        y = np.empty_like(extended_k)
        for c in range(y.shape[2]):
            with gzip.GzipFile(f"{self.args.input}_centroids_{c}.gz", "r") as f:
                centroids = np.load(file=f)
            self.Q = Quantizer(Q_step=QSS, counts=np.ones(shape=256))
            self.Q.set_representation_levels(centroids)
            y[..., c] = self.Q.decode(extended_k[..., c])
        return y
        
    def encode(self):
        '''Read an image, quantize the image, and save it.'''
        # We have to perform vector (Going to implement learning VQ)
        #STEP 1. Inicializa los pesos
        #STEP 2. Sele
        img = self.read()
        k = self.quantize(img)
        self.write(k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        #k = io.imread(self.args.input)
        k = self.read()
        y = self.dequantize(k)
        self.write(y)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

if __name__ == "__main__":
    main.main(EC.parser, logging, CoDec)
    logging.info(f"quantizer = {quantizer_name}")
