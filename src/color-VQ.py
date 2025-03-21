'''Color quantization using Vector Quantization (VQ).'''

import os
import numpy as np
import logging
import main
from sklearn import cluster  # pip install scikit-learn

import parser
from information_theory import information  # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"
#import blur as denoiser
#import entropy_image_coding as EIC
import importlib

#default_EIC = "PNG"
default_N_clusters = 256

#parser.parser_encode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
#parser.parser_decode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
parser.parser_encode.add_argument("-m", "--N_color_clusters", type=parser.int_or_str, help=f"Number of clusters (default: {default_N_clusters})", default=default_N_clusters)
parser.parser_decode.add_argument("-m", "--N_color_clusters", type=parser.int_or_str, help=f"Number of clusters (default: {default_N_clusters})", default=default_N_clusters)

args = parser.parser.parse_known_args()[0]
denoiser = importlib.import_module("blur")
#EC = importlib.import_module(args.entropy_image_codec)

#class CoDec(EC.CoDec):
class CoDec(denoiser.CoDec):

    def __init__(self, args, min_index_val=0, max_index_val=255):
        logging.debug("trace")
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        self.N_clusters = args.N_color_clusters
        self.input = args.input
        self.output = args.output

    def encode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        img = self.encode_read_fn(in_fn)
        labels, centroids = self.quantize(img)               
        compressed_labels = self.compress_fn(labels, in_fn) 
        output_size = self.encode_write_fn(compressed_labels, out_fn)
        codebook_fn = f"{out_fn}_centroids.npz"         
        np.savez_compressed(file=codebook_fn, a=centroids)
        self.total_output_size += os.path.getsize(codebook_fn)
        return output_size

    def encode(self):
        return self.encode_fn(in_fn=self.args.input, out_fn=self.args.output)

    def decode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        compressed_labels = self.decode_read_fn(in_fn)
        labels = self.decompress_fn(compressed_labels, in_fn)
        codebook_fn = f"{in_fn}_centroids.npz"         
        self.total_input_size += os.path.getsize(codebook_fn)     
        centroids = np.load(file=codebook_fn)['a']
        img = self.dequantize(labels, centroids) 
        img = denoiser.CoDec.filter(self, img)
        output_size = self.decode_write_fn(img, out_fn)
        return output_size

    def decode(self):
        return self.decode_fn(in_fn=self.args.input, out_fn=self.args.output)

    # Vector quantization on the image by clustering RGB triplets
    # into N_clusters using KMeans
    def quantize(self, img):
        logging.debug("trace")
        height, width, _ = img.shape

        # Treat RGB triplets as single units
        rgb_vectors = img.reshape((-1, 3)).astype(int)  # The image is reshaped so that each row corresponds to an RGB triplet (pixel) from the image. Now, the image becomes a list of RGB vectors

        # Perform K-means clustering on RGB vectors
        k_means = cluster.KMeans(init="k-means++", n_clusters=self.N_clusters, n_init=1) # KMeans clustering is used to group these RGB triplets into N_clusters clusters
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
