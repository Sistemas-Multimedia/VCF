'''Entropy Encoding of images using PNG.'''

import argparse
import os
from skimage import io # pip install scikit-image
import numpy as np
import logging
#FORMAT = "%(module)s: %(message)s"
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
#logging.basicConfig(format=FORMAT)
logging.basicConfig(format=FORMAT, level=logging.INFO)
#logging.basicConfig(format=FORMAT, level=logging.DEBUG)

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

def encode(codec):
    return codec.encode()

def decode(codec):
    return codec.decode()

ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ENCODE_OUTPUT = "/tmp/encoded.png"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.png"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers(help='You must specify one of the following subcomands:')
parser_encode = subparsers.add_parser('encode', help="Encode an image")
parser_decode = subparsers.add_parser('decode', help='Decode an image')
parser_encode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser_encode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")
parser_encode.set_defaults(func=encode)
parser_decode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser_decode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {DECODE_OUTPUT}", default=f"{DECODE_OUTPUT}")    
parser_decode.set_defaults(func=decode)

class Entropy_Codec:

    def __init__(self, args):
        self.args = args
        logging.debug(f"args = {self.args}")

    def encode(self):
        '''Read an image and save it in the disk. The encoding
        algorithm depends on the output file extension. The input can
        be online.'''
        img = io.imread(self.args.input)
        logging.info(f"Read {self.args.input} of shape {img.shape}")
        rate = self.save(img)
        return rate

    def save(self, img):
        '''Save to disk the image <img> (considering the extension).'''
        io.imsave(self.args.output, img)
        obytes = os.path.getsize(self.args.output)
        rate = obytes*8/(img.shape[0]*img.shape[1])
        logging.info(f"Written {obytes} bytes in {self.args.output}.png")
        return rate

    def decode(self):
        '''Read an image (that can be online) and save it in the disk.
        The encoding algorithm depends on the file extension.'''
        img = self.read()
        io.imsave(self.args.output, img)
        obytes = os.path.getsize(self.args.output)
        rate = obytes*8/(img.shape[0]*img.shape[1])
        logging.info(f"Written {obytes} bytes in {self.args.output}")
        return rate

    def read(self):
        '''Read an image (that can be online).'''
        img = io.imread(self.args.input)
        return img

if __name__ == "__main__":
    logging.info(__doc__) # ?
    parser.description = __doc__
    args = parser.parse_known_args()[0]

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()

    codec = Entropy_Codec(args)

    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")
