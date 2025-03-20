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

    def UNUSED_get_output_bytes(self):
        logging.debug("trace")
        #logging.info(f"output_bytes={self.total_output_size}")
        return self.total_output_size

    def encode_read_fn(self, fn):
        '''Read the image with URL <fn>, which can be stored in the
        local disk or remotely.

        Args:

        1. fn: Path in the file system or URL of an image.

        Returns:

        1. The image code-stream.

        '''
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
        print(self.img_shape)
        return img

    def encode_read(self):
        '''Read from disk an image specified in the class attribute
        <self.args.input>.

        Args:

        1. self.args.input: The URL of the input image.

        Modifies:
        
        1. self.img_shape: The shape of the image.

        Returns:

        1. The code-stream of the image.
        
        '''
        logging.debug("trace")
        img = self.encode_read_fn(self.args.input)
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
        print(self.img_shape)
        return img
    
    def encode_write_fn(self, codestream, fn_without_extention):
        '''Write a codestream in <fn_without_extention> +
        <self.file_extension>.

        Args:

        1. codestream: The codestream, in memory.

        2. fn_without_extension: The name of the file on disk.

        3. self.file_extension: The extension of the image. This will
        be determined by the selected entropy encoder.

        Modifies:
        
        1. self.total_output_size: Number of bytes written.

        Returns:

        1. The file with the image codestream.

        '''
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
        '''Wrapper of encode_write_fn(). Save to disk the image
        specified in <self.args.output>.

        Args:

        1. compressed_img: The codestream of the image.

        2. self.args.output: THe name of the file.

        Modifies:
        
        1. self.total_output_size: Number of bytes written.
        
        Returns:

        1. The file with the image codestream.

        '''
        logging.debug("trace")
        output_size = self.encode_write_fn(compressed_img, self.args.output)
        return output_size

    def encode_fn(self, in_fn, out_fn):
        '''
        
        '''
        logging.debug("trace")
        img = self.encode_read_fn(in_fn)
        compressed_img = self.compress_fn(img, out_fn) # Some codecs
                                                       # (such as
                                                       # Huffman)
                                                       # generate
                                                       # extra files
        output_size = self.encode_write_fn(compressed_img, out_fn)
        print("============", self.img_shape)
        self.img_shape = img.shape
        return output_size

    def encode(self):
        '''Read an image, compress it, and write to disk the
        code-stream.

        Args:

        1. self.args.input: The URL of the input image.

        2. self.args.output + self.file_extension: The name of the
        output image.

        Modifies:
        
        1. self.total_output_size: Length in bytes of the output image.
        
        Returns:

        1. output_size (in bytes).
        
        '''
        logging.debug("trace")
        img = self.encode_read()
        compressed_img = self.compress(img)
        output_size = self.encode_write(compressed_img)
        return output_size

    def decode_read_fn(self, fn_without_extention):
        '''Read from disk the code-stream of the image with name
        <fn_without_extension> + <self.file_extension>.

        Args:

        1. fn_without_extention: The name of the file with the
        code-stream.

        2. self.file_extension: The extension of the input file,
        determined by the used entropy codec.

        Modifies:

        1. self.total_input_size: Number of bytes read.        

        Returns:

        1. codestream: Code-stream of the image.

        '''
        logging.debug("trace")
        fn = fn_without_extention + self.file_extension
        input_size = os.path.getsize(fn)
        self.total_input_size += input_size
        logging.debug(f"Read {input_size} bytes from {fn}")
        codestream = open(fn, "rb").read()
        return codestream

    def decode_read(self):
        '''Wrapper for decode_read_fn(), where <fn_without_extention>
        = <self.args.input>. Read from disk the code-stream of the
        image with name <self.args.input> + <self.file_extension>, and
        return the code-stream.

        Args:

        1. self.args.input: Name of the image to read.

        2. self.file_extension: Extension of the image to read
        (determined by the used entropy codec).

        Modifies:
        
        1. The number of read bytes in <self.total_input_size>.
        
        Returns:

        1. The code-stream.

        '''
        logging.debug("trace")
        compressed_img = self.decode_read_fn(self.args.input)
        return compressed_img

    def decode_write_fn(self, img, fn):
        '''Write an image to disk.

        Args:

        1. img: The image.

        2. fn: The filename.

        Modifies:

        1. self.total_output_size.
        '''
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
        '''Wrapper for decode_write_fn(), where the writen image is
        <self.args.output>.

        Args.

        1. img: The image.

        2. self.args.output: The filename.

        Modifies:

        1. self.total_output_size.
        
        '''
        logging.debug("trace")
        output_size = self.decode_write_fn(img, self.args.output)
        return output_size

    def decode_fn(self, in_fn, out_fn):
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

