'''Image quantization using a vectorial quantizator based on K-N.'''

# Some work could be done with the encoded histograms!

import argparse
import os
from skimage import io # pip install scikit-image
import numpy as np
import gzip
import logging
import main
from sklearn import cluster
import imageio.v2 as imageio
# pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"
import PNG as EC # Entropy Coding

#Default value
num_clusters = 6
num_clusters = np.power(2, num_clusters)

class CoDec(EC.CoDec):
    
    def __init__(self, args): # ??
        super().__init__(args)

    def quantize(self, img):
        x = img
        x = x.reshape((-1, 1))
        kmeans = cluster.KMeans(n_clusters = num_clusters, n_init=4, random_state=5)
        kmeans.fit(x)
        compressed_image = kmeans.cluster_centers_[kmeans.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)
        compressed_image = compressed_image.reshape(img.shape)

        return compressed_image

    def dequantize(self, k):

        #Not-Implemented
        return k
        
    def encode(self):
        '''Read an image, quantize the image, and save it.'''

        img = self.read()
        k = self.quantize(img)
        self.write(k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        k = self.read()
        y = self.dequantize(k)
        self.write(y)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

if __name__ == "__main__":
    quantizer_name = "Vectorial Q. KN"
    main.main(EC.parser, logging, CoDec)
    logging.info(f"quantizer = {quantizer_name}")
