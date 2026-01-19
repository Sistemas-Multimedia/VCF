'''Entropy Encoding of images Arithmetic Coding'''
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
from arith_codding.ArithmeticCode import ArithmeticCodding,FrequencyTable,EOF


class CoDec(EIC.CoDec):
    
    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        self.file_extension = ".ari"
    

    def compress_fn(self, img, fn):
        logging.debug(f"trace img={img}")
        arith_fn = f"{fn}_nadap_arith.pkl.gz"
        compressed_img = io.BytesIO()

        encoder = ArithmeticCodding()

        # Flatten the array and convert to a list
        flattened_img = img.flatten().tolist()
        
        # convert to symbols

        # Add EOF symbol, we use the determinate in the algorithm
        flattened_img.append(EOF())

        # Encode the flattened array and gets the bits and frequency table
        bits,table = encoder.encode(flattened_img,show_table=False)

        # Write encoded image and original shape to compressed_img
        bit_bytes = self.bits_to_bytes(bits)  # Save encoded data as bytes
        compressed_img.write(bit_bytes)

        # Compress and save shape and the Huffman Tree
        logging.debug(f"Saving {arith_fn}")
        with gzip.open(arith_fn, 'wb') as f:
            np.save(f, img.shape)
            pickle.dump(table.frequencies, f)  # `gzip.open` compresses the pickle data

        return compressed_img

    def compress(self, img, fn="/tmp/encoded"):
        return self.compress_fn(img, fn)
    
    def decompress_fn(self, compressed_img, fn):
        logging.debug(f"trace compressed_img={compressed_img[:10]}")
        arith_fn = f"{fn}_nadap_arith.pkl.gz"
        compressed_img = io.BytesIO(compressed_img)

        decoder = ArithmeticCodding()
        # Load the shape and the Huffman Tree from the compressed file
        with gzip.open(arith_fn, 'rb') as f:
            shape = np.load(f)
            frequencies = pickle.load(f)

        table = FrequencyTable()

        # Build frequency table
        
        for symbol, freq in frequencies.items():
            table.frequencies[symbol] = freq
            table.alphabet.append(symbol)
            table.nsymbols += freq

        bits = self.bytes_to_bits(compressed_img.read())

        decoded_symbols = decoder.decode(bits,table)

        

        # Reshape decoded data to original shape
        img = np.array(decoded_symbols, dtype=np.uint8).reshape(shape)
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


