'''Shared code among the entropy image codecs.'''

import os
import io
from skimage import io as skimage_io # pip install scikit-image
from PIL import Image # pip install 
import numpy as np
import logging
import subprocess
import cv2 as cv # pip install opencv-python
import main
import urllib
import math
import parser

# Default IO images
#ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ORIGINAL = "/tmp/original.png"
ENCODED = "/tmp/encoded" # File extension decided in run-time
#DECODE_INPUT = ENCODE_OUTPUT
DECODED = "/tmp/decoded.png"

# Encoder parser
parser.parser_encode.add_argument("-o", "--original", "-i", "--input", type=parser.int_or_str, help=f"Input image (default: {ORIGINAL})", default=ORIGINAL)
parser.parser_encode.add_argument("-e", "--encoded", "-o_alt", "--output", type=parser.int_or_str, help=f"Output image (default: {ENCODED})", default=f"{ENCODED}")

# Decoder parser
parser.parser_decode.add_argument("-e", "--encoded", "-i", "--input", type=parser.int_or_str, help=f"Input code-stream (default: {ENCODED})", default=f"{ENCODED}")
parser.parser_decode.add_argument("-d", "--decoded", "-o", "--output", type=parser.int_or_str, help=f"Output image (default: {DECODED})", default=f"{DECODED}")    

class CoDec:
    
    def __init__(self, args):
        logging.debug(f"trace args={args}")
        self.args = args
        if args.subparser_name == "encode":
            self.encoding = True
        else:
            self.encoding = False
        logging.debug(f"self.encoding = {self.encoding}")
        self.total_input_size = 0
        self.total_output_size = 0

    def bye(self):
        logging.debug("trace")

    def encode_read_fn(self, fn):
        '''Read an image.'''
        try:
            input_size = os.path.getsize(fn)
            img = cv.imread(fn, cv.IMREAD_UNCHANGED)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        except:
            req = urllib.request.Request(fn, method='HEAD')
            f = urllib.request.urlopen(req)
            input_size = int(f.headers['Content-Length'])
            self.total_input_size += input_size
            img = skimage_io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        logging.debug(f"Read {input_size} bytes from {fn} with shape {img.shape} and type={img.dtype}")
        self.img_shape = img.shape
        return img

    def encode_read(self, fn=None):
        if fn is None:
            fn = self.args.original
        return self.encode_read_fn(fn)

    def encode_write_fn(self, codestream, fn):
        '''Write a code-stream.'''
        logging.debug(f"trace codestream={codestream}")
        codestream.seek(0)
        with open(fn + self.file_extension, "wb") as output_file:
            output_file.write(codestream.read())
        output_size = os.path.getsize(fn + self.file_extension)
        self.total_output_size += output_size
        logging.info(f"Written {output_size} bytes in {fn + self.file_extension}")
        return output_size

    def encode_write(self, codestream, fn=None):
        if fn is None:
            fn = self.args.encoded
        return self.encode_write_fn(codestream, fn)

    def encode(self):
        img = self.encode_read()
        compressed_img = self.compress(img)
        output_size = self.encode_write(compressed_img)
        self.img_shape = img.shape
        return output_size

    def decode_read_fn(self, fn):
        if os.path.exists(fn):
            full_fn = fn
        elif os.path.exists(fn + self.file_extension):
            full_fn = fn + self.file_extension
        else:
            # Fallback to extensioned path to trigger FileNotFoundError with the expected name
            full_fn = fn + self.file_extension
            
        input_size = os.path.getsize(full_fn)
        self.total_input_size += input_size
        logging.debug(f"Read {input_size} bytes from {full_fn}")
        codestream = open(full_fn, "rb").read()
        return codestream

    def decode_read(self, fn=None):
        if fn is None:
            fn = self.args.encoded
        return self.decode_read_fn(fn)

    def decode_write_fn(self, img, fn):
        logging.debug(f"trace img={img}")
        logging.debug(f"trace fn={fn}")
        try:
            skimage_io.imsave(fn, img)
        except Exception as e:
            logging.error(f"Exception \"{e}\" saving image {fn} with shape {img.shape} and type {img.dtype}")
        self.img_shape = img.shape
        output_size = os.path.getsize(fn)
        self.total_output_size += output_size
        logging.debug(f"Written {output_size} bytes in {fn} with shape {img.shape} and type {img.dtype}")
        return output_size

    def decode_write(self, img, fn=None):
        if fn is None:
            fn = self.args.decoded
        return self.decode_write_fn(img, fn)

    def decode(self):
        compressed_img = self.decode_read()
        img = self.decompress(compressed_img)
        output_size = self.decode_write(img)
        return output_size
