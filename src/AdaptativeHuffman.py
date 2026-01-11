'''Entropy Encoding of images Adaptive Huffman Coding'''

import io
import numpy as np
import main
import logging
import os
# Get description temp file
import tempfile
# Fix substitute hardcode temp file with identification with library tempfile
desc_path = os.path.join(tempfile.gettempdir(),"description.txt")
with open(desc_path, 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_image_coding as EIC
import heapq
from collections import defaultdict, Counter
import gzip
import pickle
from bitarray import bitarray
import math
from fgk import FGK




class CoDec(EIC.CoDec):

    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        self.file_extension = ".ahuf"  

    def compress_fn(self, img, fn):
        logging.debug(f"trace img={img}")

        compressed_img = io.BytesIO()

        # Flatten image to 1D array and convert to bytes
        flattened_img = img.flatten()
        data_bytes = bytes(flattened_img)

        
        # Encode using Adaptive Huffman FGK
        encoder = FGK()
        encoded_bits = encoder.encode(data_bytes)

        np.save(f"{fn}_bitlen.npy",len(encoded_bits))

        ba = bitarray(encoded_bits)
        ba.tofile(compressed_img)

        np.save(f"{fn}_shape.npy",img.shape)

        return compressed_img
        
    def compress(self, img, fn="/tmp/encoded"):
        return self.compress_fn(img, fn)

    def decompress_fn(self, compressed_img, fn):
        logging.debug(f"trace compressed_img={compressed_img[:10]}")

        compressed_img = io.BytesIO(compressed_img)

        # load Shape
        shape = np.load(f"{fn}_shape.npy")
        bitlen = int(np.load(f"{fn}_bitlen.npy"))
        

        # Read bitarray from BytesIO
        ba = bitarray()
        ba.fromfile(compressed_img)

        encoded_data = ba.to01()[:bitlen]

        # Decode using Adaptive Huffman FGK
        decoder = FGK()
        decoded_bytes = decoder.decode(encoded_data)

        # reconstruct image
        img = np.frombuffer(bytes(decoded_bytes),dtype=np.uint8).reshape(shape)
        
        return img

    def decompress(self, compressed_img, fn="/tmp/encoded"):
        return self.decompress_fn(compressed_img, fn)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)



