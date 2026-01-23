'''Entropy Encoding of images Arithmetic Coding with multiple order'''
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
from adaptive_arith_code.Ppm_compressor import PPMCompressor,PPMDecompressor

default_order = 1


parser.parser_encode.add_argument("-m","--model_order",type=parser.int_or_str,
                                 help="Order of the probabilistic model",default=default_order)

parser.parser_decode.add_argument("-m","--model_order",type=parser.int_or_str,
                                help="Order of the probabilistic model",default=default_order)

args = parser.parser.parse_known_args()[0]

#TODO: Add parser to determinate the what order to use
class CoDec(EIC.CoDec):
    
    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        self.file_extension = ".aari" # Extension reference to Adaptive Arithmetic

        # Set the order of the probabilistic model
        self.order = args.model_order
    

    def compress_fn(self, img, fn):
        logging.debug(f"trace img={img}")
        arith_fn = f"{fn}_adaptive_arith.pkl.gz"
        compressed_img = io.BytesIO()

        # Flatten the array and convert to a list
        flattened_img = img.flatten().tolist()

        # Create Encoder, use default EOF and ESC symbol internally
        encoder = PPMCompressor(model_order=self.order)

        # Compress the model and return a bitarray
        bits = encoder.compress(flattened_img)

        bit_bytes = self.bits_to_bytes(bits)

        # Write encoded image and original shape to compressed_img
        compressed_img.write(bit_bytes)

        # Compress and save shape and Arithmetic
        logging.debug(f"Saving {arith_fn}")
        with gzip.open(arith_fn, 'wb') as f:
            np.save(f, img.shape)

        return compressed_img

    def compress(self, img, fn="/tmp/encoded"):
        return self.compress_fn(img, fn)
    
    def decompress_fn(self, compressed_img, fn):
        logging.debug(f"trace compressed_img={compressed_img[:10]}")
        arith_fn = f"{fn}_adaptive_arith.pkl.gz"
        compressed_img = io.BytesIO(compressed_img)

        # Initialize the Descompressor with a determinate order
        decoder = PPMDecompressor(model_order=self.order)
        
        # Load the shape from the compressed file
        with gzip.open(arith_fn, 'rb') as f:
            shape = np.load(f)


        bits = self.bytes_to_bits(compressed_img.read())
        
        #Decode the bits, how we use a model_order we can determinate the table used
        decoded_bits = decoder.decompress(bits)

        # Reshape decoded data to original shape
        img = np.array(list(decoded_bits), dtype=np.uint8).reshape(shape)
        return img

    def decompress(self, compressed_img, fn="/tmp/encoded"):
        return self.decompress_fn(compressed_img, fn)

    def bits_to_bytes(self,bits):
        while len(bits) % 8 != 0:
            bits.append(0)
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for b in bits[i:i + 8]:
                byte = (byte << 1) | b
            result.append(byte)
        return bytes(result)


    def bytes_to_bits(self,data):
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)


