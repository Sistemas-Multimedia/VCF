'''Entropy Encoding of images non-adaptive Huffman Coding'''

import io
import numpy as np
import main
import logging
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_image_coding as EIC
import heapq
from collections import defaultdict, Counter
import gzip
import pickle
from bitarray import bitarray
import os
import math
from huffman_coding import huffman_coding # pip install --ignore-installed "huffman_coding @ git+https://github.com/vicente-gonzalez-ruiz/huffman_coding"

# Encoder parser
parser.parser_encode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {EIC.ENCODE_INPUT})", default=EIC.ENCODE_INPUT)
parser.parser_encode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {EIC.ENCODE_OUTPUT})", default=f"{EIC.ENCODE_OUTPUT}")

# Decoder parser
parser.parser_decode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {EIC.DECODE_INPUT})", default=f"{EIC.DECODE_INPUT}")
parser.parser_decode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {EIC.DECODE_OUTPUT})", default=f"{EIC.DECODE_OUTPUT}")    

#parser.parser.parse_known_args()

'''
class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = Counter(data)
    heap = [HuffmanNode(value, freq) for value, freq in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    
    return heap[0]

def generate_huffman_codes(node, current_code="", codes={}):
    if node is None:
        return
    if node.value is not None:
        codes[node.value] = current_code
    generate_huffman_codes(node.left, current_code + "0", codes)
    generate_huffman_codes(node.right, current_code + "1", codes)
    return codes

def encode_data(data, codes):
    # Create a single concatenated string of all encoded bits
    encoded_string = ''.join(codes[value] for value in data)
    #print("-------------_", len(data))
    # Convert this string of bits to a bitarray
    encoded_data = bitarray(encoded_string)

    return encoded_data

def decode_data(encoded_data, root):
    data = []
    node = root
    for bit in encoded_data:
        if bit == 0:
            node = node.left
        else:
            node = node.right
        # If it's a leaf node, record the value and reset to root
        if node.left is None and node.right is None:
            data.append(node.value)
            node = root
    #print("-------------_", len(data))
    return data
'''
class CoDec(EIC.CoDec):

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        self.file_extension = ".huf"

    def bye(self):
        logging.debug("trace")
        if self.encoding:
            # Write metadata
            with open(f"{self.args.output}_meta.txt", 'w') as f:
                f.write(f"{self.img_shape[0]}\n")
                f.write(f"{self.img_shape[1]}\n")
        else:
            # Read metadata
            with open(f"{self.args.input}.txt", 'r') as f:
                height = f.readline().strip()
                logging.info(f"video height = {height} pixels")
                width = f.readline().strip()
                logging.info(f"video width = {width} pixels")
        super().bye()

    def compress_fn(self, img, fn):
        logging.debug("trace")
        tree_fn = f"{fn}_huffman_tree.pkl.gz"
        compressed_img = io.BytesIO()

        # Flatten the array and convert to a list
        flattened_img = img.flatten().tolist()

        # Build Huffman Tree and generate the Huffman codes
        root = huffman_coding.build_huffman_tree(flattened_img)
        codes = huffman_coding.generate_huffman_codes(root)

        # Encode the flattened array
        encoded_img = huffman_coding.encode_data(flattened_img, codes)

        # Write encoded image and original shape to compressed_img
        compressed_img.write(encoded_img.tobytes())  # Save encoded data as bytes

        # Compress and save shape and the Huffman Tree
        logging.debug(f"Saving {tree_fn}")
        with gzip.open(tree_fn, 'wb') as f:
            np.save(f, img.shape)
            pickle.dump(root, f)  # `gzip.open` compresses the pickle data

        tree_length = os.path.getsize(tree_fn)
        logging.info(f"Length of the file \"{tree_fn}\" (Huffman tree + image shape) = {tree_length} bytes")
        #self.total_output_size += tree_length

        return compressed_img
    
    def compress(self, img):
        logging.debug("trace")
        tree_fn = f"{self.args.output}_huffman_tree.pkl.gz"
        compressed_img = io.BytesIO()

        # Flatten the array and convert to a list
        flattened_img = img.flatten().tolist()

        # Build Huffman Tree and generate the Huffman codes
        root = huffman_coding.build_huffman_tree(flattened_img)
        codes = huffman_coding.generate_huffman_codes(root)

        # Encode the flattened array
        encoded_img = huffman_coding.encode_data(flattened_img, codes)

        # Write encoded image and original shape to compressed_img
        compressed_img.write(encoded_img.tobytes())  # Save encoded data as bytes

        # Compress and save shape and the Huffman Tree
        with gzip.open(tree_fn, 'wb') as f:
            np.save(f, img.shape)
            pickle.dump(root, f)  # `gzip.open` compresses the pickle data

        tree_length = os.path.getsize(tree_fn)
        logging.info(f"Length of the file \"{tree_fn}\" (Huffman tree + image shape) = {tree_length} bytes")
        #self.total_output_size += tree_length

        return compressed_img
    
    def decompress_fn(self, compressed_img, fn):
        logging.debug("trace")
        tree_fn = f"{fn}_huffman_tree.pkl.gz"
        compressed_img = io.BytesIO(compressed_img)
        
        # Load the shape and the Huffman Tree from the compressed file
        with gzip.open(tree_fn, 'rb') as f:
            shape = np.load(f)
            root = pickle.load(f)
    
        # Read encoded image data as binary
        encoded_data = bitarray()
        encoded_data.frombytes(compressed_img.read())
    
        # Decode the image
        decoded_data = huffman_coding.decode_data(encoded_data, root)
        if math.prod(shape) < len(decoded_data):
            decoded_data = decoded_data[:math.prod(shape) - len(decoded_data)] # Sometimes, when the alphabet size is small, some extra symbols are decoded :-/

        # Reshape decoded data to original shape
        img = np.array(decoded_data).reshape(shape).astype(np.uint8)
        return img

    def decompress(self, compressed_img):
        logging.debug("trace")
        tree_fn = f"{self.args.input}_huffman_tree.pkl.gz"
        compressed_img = io.BytesIO(compressed_img)
        
        # Load the shape and the Huffman Tree from the compressed file
        with gzip.open(tree_fn, 'rb') as f:
            shape = np.load(f)
            root = pickle.load(f)
    
        # Read encoded image data as binary
        encoded_data = bitarray()
        encoded_data.frombytes(compressed_img.read())
    
        # Decode the image
        decoded_data = huffman_coding.decode_data(encoded_data, root)
        if math.prod(shape) < len(decoded_data):
            decoded_data = decoded_data[:math.prod(shape) - len(decoded_data)] # Sometimes, when the alphabet size is small, some extra symbols are decoded :-/

        # Reshape decoded data to original shape
        img = np.array(decoded_data).reshape(shape).astype(np.uint8)
        return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)





