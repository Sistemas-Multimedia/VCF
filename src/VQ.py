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
from PIL import Image
#Default value

bits_per_block = 1
block_side = np.power(2, bits_per_block)
block_height = block_side
block_width = block_side
n_components = 3 #RGB
block_length = block_height * block_width * n_components

dir_label="/tmp/_label.png"
dir_centroids="/tmp/_centroids.png"
dir_config="/tmp/_config.txt"

class CoDec(EC.CoDec):
    
    def __init__(self, args): # ??
        super().__init__(args)

    def quantize(self, img):
        blocks = []

        for i in range(0, img.shape[0], block_width):
            for j in range(0, img.shape[1], block_height):
                blocks.append(np.reshape(img[i:i + block_width, j:j + block_height], block_length))

        blocks = np.asarray(blocks).astype(int)
        n_clusters = 256 # number of centroids
        if(len(blocks) < n_clusters):
            print('\033[91m' + "Warning: Must reduce number of clusters. Image not big enough | too much pixel reducction" + '\033[0m')
            exit(-1)

        kmeans = cluster.KMeans(init="k-means++", n_clusters=n_clusters, n_init=1)
        kmeans.fit(blocks)
        centroids = kmeans.cluster_centers_.squeeze().astype(np.uint8)
        labels=kmeans.labels_
        print(img.shape[0])
        #Save original parameters
        sizes = [[],[],[]]
        sizes[0] = [img.shape[0], img.shape[1], img.shape[2]]
        sizes[1] = [centroids.shape[0],centroids.shape[1], 0]
        sizes[2] = [labels.shape[0], 0, 0]
        
        #Transform to reduce space in disck
        centroids = centroids.flatten()

        ## Save on disk 
        
        #np.savetxt(dir_label, labels.astype(int), fmt='%i', delimiter=" ")
        im_label = Image.fromarray(labels)
        im_label.save(dir_label)
        #np.savetxt(dir_centroids, centroids.astype(int), fmt='%i', delimiter=" ")
        im_centroids = Image.fromarray(centroids)
        im_centroids.save(dir_centroids)
        np.savetxt(dir_config, sizes, fmt='%i', delimiter=" ")
        
        return None

    def dequantize(self, sizes, labels, centroids):
        #Rebuiling data
        labels = labels.reshape(sizes[0][0]//block_height, sizes[0][1]//block_width)
        centroids = centroids.reshape(sizes[1][0], sizes[1][1])
        #Use data
        img_dequantized = np.empty([sizes[0][0], sizes[0][1], sizes[0][2]], dtype='uint8')
        for i in range(0, sizes[0][0], block_width):
            for j in range(0, sizes[0][1], block_height):
                img_dequantized[i:i + block_width, j:j + block_height] = centroids[labels[i//block_width,j//block_height]].reshape(block_height, block_width, n_components)
                
        return img_dequantized 
        
    def encode(self):
        '''Read an image, quantize the image, and save it.'''
        img = self.read()
        self.quantize(img)
        self.output_bytes = os.stat(dir_centroids).st_size + os.stat(dir_label).st_size + os.stat(dir_config).st_size
        rate = (self.input_bytes*8)/(img.shape[0]*img.shape[1])
        print('\033[92m'+"All saved files together weights {}% compared to the original image".format(self.output_bytes/self.input_bytes*100)+'\033[0m')
        return rate

    def decode(self):
        img_la = Image.open(dir_label)
        label = np.asarray(img_la)
        img_cen = Image.open(dir_centroids)
        centroids = np.asarray(img_cen)
        sizes=np.loadtxt(dir_config, dtype=int)
        k=self.dequantize(sizes=sizes, labels=label, centroids=centroids)
        self.write(k)
        rate = (self.output_bytes*8)/(sizes[0][0]*sizes[0][1])
        return rate

if __name__ == "__main__":
    quantizer_name = "Vectorial Q. KN"
    main.main(EC.parser, logging, CoDec)
    logging.info(f"quantizer = {quantizer_name}")
