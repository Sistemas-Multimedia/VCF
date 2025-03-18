'''Shared code among the image entropy codecs.'''

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
ENCODE_OUTPUT = "/tmp/encoded" # The file extension is decided in run-time
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.png"

class CoDec:

    def __init__(self, args):
        self.args = args
        logging.debug(f"args = {self.args}")
        if args.subparser_name == "encode":
            self.encoding = True
        else:
            self.encoding = False
        logging.debug(f"self.encoding = {self.encoding}")
        self.input_bytes = 0
        self.output_bytes = 0

    def __del__(self):
        logging.info(f"Total {self.input_bytes} bytes read")
        logging.info(f"Total {self.output_bytes} bytes written")
        if self.encoding:
            number_of_output_bits = self.output_bytes*8
            number_of_pixels = self.img_shape[0]*self.img_shape[1]
            BPP = number_of_output_bits/number_of_pixels
            logging.info(f"rate = {BPP} bits/pixel")
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

    def get_output_bytes(self):
        #logging.info(f"output_bytes={self.output_bytes}")
        return self.output_bytes

    def encode_read_fn(self, fn):
        '''Read the image with URL <fn>, which can be stored in the
        local disk or remotely.

        Args:

        1. fn: Path in the file system or URL of an image.

        Returns:

        1. The image code-stream.

        '''
        #img = skimage_io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        #img = Image.open(fn) # https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#using-the-image-class
        try:
            input_size = os.path.getsize(fn)
            self.input_bytes += input_size 
            img = cv.imread(fn, cv.IMREAD_UNCHANGED)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        except:
            req = urllib.request.Request(fn, method='HEAD')
            f = urllib.request.urlopen(req)
            input_size = int(f.headers['Content-Length'])
            self.input_bytes += input_size
            img = skimage_io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        logging.debug(f"Read {input_size} bytes from {fn} with shape {img.shape} and type={img.dtype}")
        return img

    def encode_read(self):
        '''Read from disk an image specified in the class attribute
        <self.args.input>.

        Args:

        1. self.args.input: The URL of the input image.

        Returns:

        1. The code-stream of the image.
        
        2. self.img_shape: The shape of the image.

        '''
        img = self.encode_read_fn(self.args.input)
        '''
        if __debug__:
            fn = "/tmp/original.png"
            #self.decode_write_fn(img, "/tmp/original.png") # Save a copy for comparing later
            try:
                skimage_io.imsave(fn, img)
            except Exception as e:
                logging.error(f"Exception \"{e}\" saving image {fn} with shape {img.shape} and type {img.dtype}")
        '''
        self.img_shape = img.shape
        return img
    
    def encode_write_fn(self, codestream, fn_without_extention):
        '''Write a codestream in <fn_without_extention> +
        <self.file_extension>.

        Args:

        1. codestream: The codestream, in memory.

        2. fn_without_extension: The name of the file on disk.

        3. self.file_extension: The extension of the image. This will
        be determined by the selected entropy encoder.

        Returns:

        1. The file with the image codestream.

        2. self.output_bytes: Number of bytes written.

        '''
        codestream.seek(0)
        fn = fn_without_extention + self.file_extension
        with open(fn, "wb") as output_file:
            output_file.write(codestream.read())
        output_size = os.path.getsize(fn)
        self.output_bytes += output_size
        #print("*"*80, self.output_bytes)
        logging.info(f"Written {output_size} bytes in {fn}")

    def encode_write(self, compressed_img):
        '''Save to disk the image specified in the class attribute <
        self.args.output>.

        Args:

        1. compressed_img: The codestream of the image.

        2. self.args.output: THe name of the file.

        Returns:

        Nothing.

        '''
        self.encode_write_fn(compressed_img, self.args.output)

    def encode(self):
        '''Read an image, encode it, and write to disk the
        code-stream.

        Args:

        1. self.args.input: The URL of the input image.

        2. self.args.output + self.file_extension: The name of the
        output image.

        Returns:
        
        1. self.output_bytes: Length in bytes of the output image.

        '''
        img = self.encode_read()
        compressed_img = self.compress(img)
        self.encode_write(compressed_img)

    def decode(self):
        '''Read the code-stream of an image, decode it, and write to
        disk the decoded image.'''
        compressed_img = self.decode_read()
        img = self.decompress(compressed_img)
        #compressed_img_diskimage = io.BytesIO(compressed_img)
        #img = np.load(compressed_img_diskimage)['a']
        #decompressed_data = zlib.decompress(compressed_img)
        #img = io.BytesIO(decompressed_data))
        self.decode_write(img)
        #logging.debug(f"output_bytes={self.output_bytes}, img.shape={img.shape}")
        #self.BPP = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        #return rate, 0
        #logging.info("RMSE = 0")


    def decode_read(self):
        compressed_img = self.decode_read_fn(self.args.input)
        return compressed_img

    def decode_read_fn(self, fn_without_extention):
        fn = fn_without_extention + self.file_extension
        input_size = os.path.getsize(fn)
        self.input_bytes += input_size
        logging.debug(f"Read {os.path.getsize(fn)} bytes from {fn}")
        data = open(fn, "rb").read()
        return data

    def decode_write(self, img):
        return self.decode_write_fn(img, self.args.output)

    #def filter(self, img):
    #    return img

    def decode_write_fn(self, img, fn):
        #img = self.filter(img)
        try:
            skimage_io.imsave(fn, img)
        except Exception as e:
            logging.error(f"Exception \"{e}\" saving image {fn} with shape {img.shape} and type {img.dtype}")
        self.output_bytes += os.path.getsize(fn)
        logging.debug(f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

