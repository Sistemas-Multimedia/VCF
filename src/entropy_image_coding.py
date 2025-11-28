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
parser.parser_encode.add_argument("-o", "--original", type=parser.int_or_str, help=f"Input image (default: {ORIGINAL})", default=ORIGINAL)
parser.parser_encode.add_argument("-e", "--encoded", type=parser.int_or_str, help=f"Output image (default: {ENCODED})", default=f"{ENCODED}")

# Decoder parser
parser.parser_decode.add_argument("-e", "--encoded", type=parser.int_or_str, help=f"Input code-stream (default: {ENCODED})", default=f"{ENCODED}")
parser.parser_decode.add_argument("-d", "--decoded", type=parser.int_or_str, help=f"Output image (default: {DECODED})", default=f"{DECODED}")    

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

    def encode_read(self):
        '''Read an image.'''
        try:
            input_size = os.path.getsize(self.args.original)
            img = cv.imread(self.args.original, cv.IMREAD_UNCHANGED)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        except:
            req = urllib.request.Request(self.args.original, method='HEAD')
            f = urllib.request.urlopen(req)
            input_size = int(f.headers['Content-Length'])
            self.total_input_size += input_size
            img = skimage_io.imread(self.args.original) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        logging.debug(f"Read {input_size} bytes from {self.args.original} with shape {img.shape} and type={img.dtype}")
        self.img_shape = img.shape
        return img

    def encode_write(self, codestream):
        '''Write a code-stream.'''
        logging.debug(f"trace codestream={codestream}")
        codestream.seek(0)
        with open(self.args.encoded + self.file_extension, "wb") as output_file:
            output_file.write(codestream.read())
        output_size = os.path.getsize(self.args.encoded + self.file_extension)
        self.total_output_size += output_size
        logging.info(f"Written {output_size} bytes in {self.args.encoded + self.file_extension}")
        return output_size

    def encode(self):
        img = self.encode_read()
        compressed_img = self.compress(img)
        output_size = self.encode_write(compressed_img)
        self.img_shape = img.shape
        return output_size

    def decode_read(self):
        input_size = os.path.getsize(self.args.encoded + self.file_extension)
        self.total_input_size += input_size
        logging.debug(f"Read {input_size} bytes from {self.args.encoded + self.file_extension}")
        codestream = open(self.args.encoded + self.file_extension, "rb").read()
        return codestream

    def decode_write(self, img):
        logging.debug(f"trace img={img}")
        try:
            skimage_io.imsave(self.args.decoded, img)
        except Exception as e:
            logging.error(f"Exception \"{e}\" saving image {self.args.decoded} with shape {img.shape} and type {img.dtype}")
        self.img_shape = img.shape
        output_size = os.path.getsize(self.args.decoded)
        self.total_output_size += output_size
        logging.debug(f"Written {output_size} bytes in {self.args.decoded} with shape {img.shape} and type {img.dtype}")
        return output_size

    def decode(self):
        compressed_img = self.decode_read()
        img = self.decompress(compressed_img)
        output_size = self.decode_write(img)
        return output_size
