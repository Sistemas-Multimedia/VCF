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
        flattened_img = img.flatten().tolist()
        data_bytes = bytes(flattened_img)

        # Encode using Adaptive Huffman FGK
        encoder = FGK()
        encoded_bits = encoder.encode(data_bytes)

        # Convert bit string to bitarray and write to BytesIO
        ba = bitarray(encoded_bits)
        ba.tofile(compressed_img)
        
        # Save original shape separately
        shape_fn = f"{fn}_shape.npy"
        np.save(shape_fn, img.shape)
        logging.debug(f"Saved image shape to {shape_fn}")

        return compressed_img
        
    def compress(self, img, fn="/tmp/encoded"):
        return self.compress_fn(img, fn)

    def decompress_fn(self, compressed_img, fn):
        logging.debug(f"trace compressed_img={compressed_img[:10]}")

        compressed_img = io.BytesIO(compressed_img)

        # Load original shape
        shape_fn = f"{fn}_shape.npy"
        shape = np.load(shape_fn)

        # Read bitarray from BytesIO
        ba = bitarray()
        ba.fromfile(compressed_img)

        # Decode using Adaptive Huffman FGK
        decoder = FGK()
        decoded_bytes = decoder.decode(ba.to01())

        # Reshape decoded bytes to original image
        img = np.frombuffer(decoded_bytes, dtype=np.uint8).reshape(shape)
        return img

    def decompress(self, compressed_img, fn="/tmp/encoded"):
        return self.decompress_fn(compressed_img, fn)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)



