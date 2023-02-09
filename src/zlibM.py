'''Entropy Encoding of images using PNG (Portable Network Graphics).'''

import argparse
import os
from skimage import io # pip install scikit-image
from PIL import Image # pip install 
import numpy as np
import logging
import subprocess
import cv2 as cv
import main
import urllib
import zlib
import io as readIO

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

# A way of converting a call to a object's method to a plain function
def encode(codec):
    return codec.encode()

def decode(codec):
    return codec.decode()

# Default IO images
ENCODE_INPUT = "C:/Users/pma98/Desktop/SM/images/image2.png"
ENCODE_OUTPUT = "C:/Users/pma98/Desktop/SM/images/encode.png"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "C:/Users/pma98/Desktop/SM/images/decode.png"

# Main parameter of the arguments parser: "encode" or "decode"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g", "--debug", action="store_true", help=f"Output debug information")
subparsers = parser.add_subparsers(help="You must specify one of the following subcomands:", dest="subparser_name")

# Encoder parser
parser_encode = subparsers.add_parser("encode", help="Encode an image")
parser_encode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser_encode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")
parser_encode.set_defaults(func=encode)

# Decoder parser
parser_decode = subparsers.add_parser("decode", help='Decode an image')
parser_decode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser_decode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {DECODE_OUTPUT})", default=f"{DECODE_OUTPUT}")    
parser_decode.set_defaults(func=decode)


class CoDec:

    def __init__(self, args):
        self.args = args
        logging.debug(f"args = {self.args}")
        if args.subparser_name == "encode":
            self.encoding = True
        else:
            self.encoding = False
        logging.debug(f"encoding = {self.encoding}")
        self.input_bytes = 0
        self.output_bytes = 0
    
    def read_fn(self, fn):
        '''Read the image <fn>.'''
        img = io.imread(fn)

        #info
        logging.info(f"Read {fn} of shape {img.shape}")
        return img
        
    def save_fn(self, img, fn):
        '''Read the image <fn>.'''
        io.imsave(fn, img, check_contrast=False)

        #info
        self.required_bytes = os.path.getsize(fn)
        logging.info(f"Written {self.required_bytes} bytes in {fn}")

    def read(self):
        '''Read the image specified in the class attribute
        <args.input>.'''
        return self.read_fn(self.args.input)

    def save(self,  img):
        self.save_fn(img, self.args.output)

    def encode(self):
        '''Read an image and save it in the disk. The input can be
        online. This method is overriden in child classes.'''
        img = self.read()
        self.save(img)
        #rb read-binary
        original_img = open(self.args.input, 'rb').read()
        #z_best_compression = 9
        compress_img = zlib.compress(original_img, zlib.Z_BEST_COMPRESSION)
        #wb write-binary
        with open(self.args.output, 'wb') as f:
            f.write(compress_img)
        # info
        logging.debug(f"output_bytes={self.output_bytes}, img.shape={img.shape}")
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        '''Read an image and save it in the disk. 
        '''
        original_img = open(self.args.input, 'rb').read()
        decompress_img = zlib.decompress(original_img)
        img = Image.open(readIO.BytesIO(decompress_img))
        img.save(self.args.output)

        # info
        self.input_bytes = os.path.getsize(self.args.output)
        logging.info(f"Written {self.input_bytes} bytes in {self.args.output}")
        return self.input_bytes*8/(img.size[0]*img.size[1])

    def __del__(self):
        logging.info(f"Total {self.input_bytes} bytes read")
        logging.info(f"Total {self.output_bytes} bytes written")

if __name__ == "__main__":
    main.main(parser, logging, CoDec)