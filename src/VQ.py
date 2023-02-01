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
import cv2 as cv

#Default value

block_side = 2
block_height = block_side
block_width = block_side
n_components = 3 #RGB
block_length = block_height * block_width * n_components

class CoDec(EC.CoDec):
    
    def __init__(self, args): # ??
        super().__init__(args)

    def quantize(self, img):
        blocks = []
        #print(np.round(img.shape[0] * img.shape[1] / (block_height * block_length)).astype(np.int16))
        n_clusters = 256  # Number of bins
        #kmeans = cluster.KMeans(n_clusters = n_clusters, n_init=4, random_state=5)
        kmeans = cluster.KMeans(init="k-means++", n_clusters=n_clusters, n_init=1)
        for i in range(0, img.shape[0], block_width):
            for j in range(0, img.shape[1], block_height):
                blocks.append(np.reshape(img[i:i + block_width, j:j + block_height], block_length))

        blocks = np.asarray(blocks).astype(float)
        kmeans.fit(blocks)
        centroids = kmeans.cluster_centers_.squeeze().astype(np.uint8)
            
        labels = kmeans.labels_
        labels = labels.reshape(img.shape[0]//block_height, img.shape[1]//block_width)
        
        img_dequantized = np.empty_like(img)
        for i in range(0, img.shape[0], block_width):
            for j in range(0, img.shape[1], block_height):
                img_dequantized[i:i + block_width, j:j + block_height] = centroids[labels[i//block_width,j//block_height]].reshape(block_height, block_width, n_components)
                
        return img_dequantized        

    def dequantize(self, img, labels, centroids):
        #NOT-IMPLEMENTED
        return 0
        
    def encode(self):
        '''Read an image, quantize the image, and save it.'''
        img = self.read()
        k=self.quantize(img)
        self.write(k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        #NOT-IMPLEMENTED
        return 0

if __name__ == "__main__":
    quantizer_name = "Vectorial Q. KN"
    main.main(EC.parser, logging, CoDec)
    logging.info(f"quantizer = {quantizer_name}")
