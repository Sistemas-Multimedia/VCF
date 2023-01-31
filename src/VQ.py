'''Image quantization using a LloydMax quantizer.'''

# Some work could be done with the encoded histograms!

import argparse
import os
from skimage import io # pip install scikit-image
import numpy as np
import gzip
import logging
import main
import sklearn as cluster

# pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"
from scalar_quantization.LloydMax_quantization import LloydMax_Quantizer as Quantizer
from scalar_quantization.LloydMax_quantization import name as quantizer_name

import PNG as EC # Entropy Coding

#Default value
num_clusters = 6

EC.parser_encode.add_argument("-q", "--QSS", type=EC.int_or_str, help=f"Quantization step size (default: 32)", default=32)
EC.parser_encode.add_argument("-b", "--num-bits", type=EC.int_or_str, help=f"number of bits to use on the quantified image", default=num_clusters)

num_clusters = np.power(2, num_clusters)

class CoDec(EC.CoDec):
    
    def __init__(self, args): # ??
        super().__init__(args)

    def quantize(self, img):
        with open(f"{self.args.output}_QSS.txt", 'w') as f:
            f.write(f"{self.args.QSS}")

        x = img.reshape(img)
        kmeans = cluster.KNmeans(n_clusters = num_clusters, n_init=4, random_state=5)
        kmeans.fit(x)
        centroids = kmeans.cluster_centers.squeeze()
        labels = kmeans.labels_
        image_compressed = np.choose(labels, centroids).reshape(img.shape)

        return image_compressed

    def dequantize(self, k):

        # We have to undo the operatino


        #but how

        
    #   with open(f"{self.args.input}_QSS.txt", 'r') as f:
    #       QSS = int(f.read())
    #   if len(k.shape) < 3:
    #       extended_k = np.expand_dims(k, axis=2)
    #   else:
    #       extended_k = k
    #   y = np.empty_like(extended_k)
    #   for c in range(y.shape[2]):
    #       with gzip.GzipFile(f"{self.args.input}_centroids_{c}.gz", "r") as f:
    #           centroids = np.load(file=f)
    #       self.Q = Quantizer(Q_step=QSS, counts=np.ones(shape=256))
    #       self.Q.set_representation_levels(centroids)
    #       y[..., c] = self.Q.decode(extended_k[..., c])
        return 0
        
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
