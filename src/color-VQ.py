'''Color quantization using Vector Quantization (VQ).'''

import os
import numpy as np
import logging
import main
from sklearn import cluster  # pip install scikit-learn
import parser
import importlib

with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)

default_N_clusters = 32
default_filter = "no_filter"

parser.parser_encode.add_argument("-q", "--N_color_clusters", type=parser.int_or_str, help=f"Number of clusters (default: {default_N_clusters})", default=default_N_clusters)
parser.parser_decode.add_argument("-q", "--N_color_clusters", type=parser.int_or_str, help=f"Number of clusters (default: {default_N_clusters})", default=default_N_clusters)
parser.parser_decode.add_argument("-f", "--filter", type=parser.int_or_str, help=f"Denoising filter (default: {default_filter})", default=default_filter)

args = parser.parser.parse_known_args()[0]
try:
    #print("Denoising filter =", args.filter)
    denoiser = importlib.import_module(args.filter)
except:
    # Remember that the filter is only active when decoding.
    denoiser = importlib.import_module("no_filter")

class CoDec(denoiser.CoDec):

    def __init__(self, args, min_index_val=0, max_index_val=255):
        logging.debug("trace")
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        self.N_clusters = args.N_color_clusters

    def encode(self):
        logging.debug("trace")
        img = self.encode_read()
        logging.info("quantizing ...")
        labels, centroids = self.quantize(img)         
        logging.info("compressing ...")
        compressed_labels = self.compress(labels) 
        output_size = self.encode_write(compressed_labels)
        codebook_fn = f"{self.args.encoded}_centroids.npz"         
        np.savez_compressed(file=codebook_fn, a=centroids)
        self.total_output_size += os.path.getsize(codebook_fn)
        return output_size

    def decode(self):
        logging.debug("trace")
        compressed_labels = self.decode_read()
        labels = self.decompress(compressed_labels)
        codebook_fn = f"{self.args.encoded}_centroids.npz"         
        self.total_input_size += os.path.getsize(codebook_fn)     
        centroids = np.load(file=codebook_fn)['a']
        logging.info("Dequantizing ...")
        img = self.dequantize(labels, centroids) 
        img = denoiser.CoDec.filter(self, img)
        output_size = self.decode_write(img)
        return output_size

    # Vector quantization on the image by clustering RGB triplets
    # into N_clusters using KMeans
    def quantize(self, img):
        logging.debug(f"trace img={img}")
        height, width, _ = img.shape

        # Treat RGB triplets as single units
        rgb_vectors = img.reshape((-1, 3)).astype(int)  # The image is reshaped so that each row corresponds to an RGB triplet (pixel) from the image. Now, the image becomes a list of RGB vectors

        # Perform K-means clustering on RGB vectors
        k_means = cluster.KMeans(init="k-means++", n_clusters=self.N_clusters, n_init=1) # KMeans clustering is used to group these RGB triplets into N_clusters clusters
        logging.info("Determining centroids ...")
        k_means.fit(rgb_vectors)
        centroids = k_means.cluster_centers_.astype(np.uint8)  # Centroids are the center points (average RGB values) of each cluster. They represent the color values used to compress the image
        labels = k_means.labels_.reshape((height, width))  # Labels represent which cluster (color) each pixel belongs to, and are reshaped back into the original 2D dimensions of the image
        labels = labels.astype(np.uint16)
        return labels, centroids

    def dequantize(self, labels, centroids):
        """Reconstruct the image using the cluster centroids."""
        logging.debug("trace")
        height, width = labels.shape
        # Map each label back to its corresponding RGB centroid
        # labels.flatten() -> Flattens the 2D labels array to a 1D array
        # centroids[labels.flatten()] -> Uses the labels to map each pixel back to the corresponding centroid (color)
        # reshape((height, width, 3)) -> Reshapes the array back into the original image shape with RGB channels
        img = centroids[labels.flatten()].reshape((height, width, 3))
        return img.astype(np.uint8)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
