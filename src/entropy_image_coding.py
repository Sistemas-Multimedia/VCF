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

from information_theory import distortion # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"

# Default IO images
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ENCODE_OUTPUT = "/tmp/encoded" # File extension decided in run-time
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.png"

class CoDec:

    def __init__(self, args):
        logging.debug("trace")
        self.args = args
        logging.debug(f"args = {self.args}")
        if args.subparser_name == "encode":
            self.encoding = True
        else:
            self.encoding = False
        logging.debug(f"self.encoding = {self.encoding}")
        self.total_input_size = 0
        self.total_output_size = 0

    def bye(self):
        logging.debug("trace")
        logging.debug("trace")
        if self.encoding:
            # Write metadata
            with open(f"{self.args.output}.txt", 'w') as f:
                f.write(f"{self.img_shape[0]}\n")
                f.write(f"{self.img_shape[1]}\n")
        else:
            # Read metadata
            with open(f"{self.args.input}.txt", 'r') as f:
                height = f.readline().strip()
                logging.info(f"image height = {height} pixels")
                width = f.readline().strip()
                logging.info(f"image width = {width} pixels")
        logging.info(f"Total {self.total_input_size} bytes read")
        logging.info(f"Total {self.total_output_size} bytes written")
        if self.encoding:
            number_of_output_bits = self.total_output_size*8
            number_of_pixels = self.img_shape[0]*self.img_shape[1] # self.img_shape no existe aquí, sólo existe en las clases descendientes
            BPP = number_of_output_bits/number_of_pixels
            logging.info(f"Output bit-rate = {BPP} bits/pixel")
            with open(f"{self.args.output}_BPP.txt", 'w') as f:
                f.write(f"{BPP}")
        else:
            if __debug__:
                try:
                    img = self.encode_read_fn("file:///tmp/original.png")
                    y = self.encode_read_fn(self.args.output)
                    RMSE = distortion.RMSE(img, y)
                    logging.info(f"RMSE = {RMSE}")
                    with open(f"{self.args.input}_BPP.txt", 'r') as f:
                        BPP = float(f.read())
                    J = BPP + RMSE
                    logging.info(f"J = R + D = {J}")
                except ValueError as e:
                    logging.debug(f"Unable to read {self.args.output}")

    def compress(self, img):
        logging.debug("trace")
        return self.compress_fn(img, fn = self.args.output)
        
    def decompress(self, compressed_img):
        logging.debug("trace")
        return self.decompress_fn(compressed_img, self.args.input)

    def UNUSED_get_output_bytes(self):
        logging.debug("trace")
        #logging.info(f"output_bytes={self.total_output_size}")
        return self.total_output_size

    def encode_read_fn(self, fn):
        logging.debug("trace")
        #img = skimage_io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        #img = Image.open(fn) # https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#using-the-image-class
        try:
            input_size = os.path.getsize(fn)
            self.total_input_size += input_size 
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
        if __debug__:
            fn = "/tmp/original.png"
            output_size = self.decode_write_fn(img, "/tmp/original.png") # Save a copy for comparing later
            self.total_output_size -= output_size # This file must not increase the bit-rate
            '''
            try:
                skimage_io.imsave(fn, img)
            except Exception as e:
                logging.error(f"Exception \"{e}\" saving image {fn} with shape {img.shape} and type {img.dtype}")
            '''
        return img

    def encode_read(self):
        logging.debug("trace")
        img = self.encode_read_fn(self.args.input)
        return img
    
    def encode_write_fn(self, codestream, fn_without_extention):
        logging.debug("trace")
        codestream.seek(0)
        fn = fn_without_extention + self.file_extension
        with open(fn, "wb") as output_file:
            output_file.write(codestream.read())
        output_size = os.path.getsize(fn)
        self.total_output_size += output_size
        #print("*"*80, self.total_output_size)
        logging.info(f"Written {output_size} bytes in {fn}")
        return output_size

    def encode_write(self, compressed_img):
        logging.debug("trace")
        output_size = self.encode_write_fn(compressed_img, self.args.output)
        return output_size

    def encode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        img = self.encode_read_fn(in_fn)
        compressed_img = self.compress_fn(img, out_fn) # Some codecs
                                                       # (such as
                                                       # Huffman)
                                                       # generate
                                                       # extra files
        output_size = self.encode_write_fn(compressed_img, out_fn)
        self.img_shape = img.shape
        return output_size

    def encode(self):
        logging.debug("trace")
        img = self.encode_read()
        compressed_img = self.compress(img)
        output_size = self.encode_write(compressed_img)
        return output_size

    def decode_read_fn(self, fn_without_extention):
        logging.debug("trace")
        fn = fn_without_extention + self.file_extension
        input_size = os.path.getsize(fn)
        self.total_input_size += input_size
        logging.debug(f"Read {input_size} bytes from {fn}")
        codestream = open(fn, "rb").read()
        return codestream

    def decode_read(self):
        logging.debug("trace")
        compressed_img = self.decode_read_fn(self.args.input)
        return compressed_img

    def decode_write_fn(self, img, fn):
        logging.debug("trace")
        try:
            skimage_io.imsave(fn, img)
        except Exception as e:
            logging.error(f"Exception \"{e}\" saving image {fn} with shape {img.shape} and type {img.dtype}")
        self.img_shape = img.shape
        output_size = os.path.getsize(fn)
        self.total_output_size += output_size
        logging.debug(f"Written {output_size} bytes in {fn} with shape {img.shape} and type {img.dtype}")
        return output_size

    def decode_write(self, img):
        logging.debug("trace")
        output_size = self.decode_write_fn(img, self.args.output)
        return output_size

    def decode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        compressed_img = self.decode_read_fn(in_fn)
        img = self.decompress_fn(compressed_img, in_fn)
        #compressed_img_diskimage = io.BytesIO(compressed_img)
        #img = np.load(compressed_img_diskimage)['a']
        #decompressed_data = zlib.decompress(compressed_img)
        #img = io.BytesIO(decompressed_data))
        output_size = self.decode_write_fn(img, out_fn)
        #logging.debug(f"output_bytes={self.total_output_size}, img.shape={img.shape}")
        #self.BPP = (self.total_output_size*8)/(img.shape[0]*img.shape[1])
        #return rate, 0
        #logging.info("RMSE = 0")
        return output_size

    def decode(self):
        '''Read the code-stream of an image, decompress it, and write to
        disk the decoded image.

        Args:

        1. self.args.input: The URL of the input image.

        2. self.args.output + self.file_extension: The name of the
        output image.

        Modifies:
        
        1. self.total_output_size: Length in bytes of the output image.

        '''
        logging.debug("trace")
        compressed_img = self.decode_read()
        img = self.decompress(compressed_img)
        #compressed_img_diskimage = io.BytesIO(compressed_img)
        #img = np.load(compressed_img_diskimage)['a']
        #decompressed_data = zlib.decompress(compressed_img)
        #img = io.BytesIO(decompressed_data))
        output_size = self.decode_write(img)
        #logging.debug(f"output_bytes={self.total_output_size}, img.shape={img.shape}")
        #self.BPP = (self.total_output_size*8)/(img.shape[0]*img.shape[1])
        #return rate, 0
        #logging.info("RMSE = 0")
        return output_size

    #def filter(self, img):
    #    return img

