'''Entropy Encoding of images using Zlib.'''

import logging
import main
import zlib
import PIL.Image as Image
import io as readIO
from skimage import io  # pip install scikit-image
import os
import argparse
import urllib
import cv2 as cv


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


# Default IO images
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ENCODE_OUTPUT = "/tmp/encoded.bin"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.png"

# Main parameter of the arguments parser: "encode" or "decode"
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g", "--debug", action="store_true",
                    help=f"Output debug information")
subparsers = parser.add_subparsers(
    help="You must specify one of the following subcomands:", dest="subparser_name")

# Encoder parser
parser_encode = subparsers.add_parser("encode", help="Encode an image")
parser_encode.add_argument("-i", "--input", type=int_or_str,
                           help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser_encode.add_argument("-o", "--output", type=int_or_str,
                           help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")
parser_encode.set_defaults(func=encode)

# Decoder parser
parser_decode = subparsers.add_parser("decode", help='Decode an image')
parser_decode.add_argument("-i", "--input", type=int_or_str,
                           help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser_decode.add_argument("-o", "--output", type=int_or_str,
                           help=f"Output image (default: {DECODE_OUTPUT})", default=f"{DECODE_OUTPUT}")
parser_decode.set_defaults(func=decode)


COMPRESSION_LEVEL = 9


class CoDec():

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
            img = io.imread(fn) 
        logging.info(f"Read {input_size} bytes from {fn} with shape {img.shape} and type={img.dtype}")
        return img

    def read_bin(self, fn):
        input_size = os.path.getsize(fn)
        self.input_bytes += input_size
        data = open(fn, 'rb').read()
        return data

    def write_bin(self, img, fn):
        '''Write to disk the bin file'''
        with open(fn, 'wb') as out_file:
            out_file.write(img)
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes")

    def write_fn(self, img, fn):
        img.save(fn)
        self.output_bytes += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn} with shape [{img.size[0]}, {img.size[1]}]")

    def read(self):
        '''Read the image specified in the class attribute
        <args.input> or the binary file, depending on enconding.'''
        if self.encoding:
            return self.read_fn(self.args.input)
        else:
            return self.read_bin(self.args.input)

    def write(self, img):
        '''Save to disk the image specified in the class attribute <
        args.output>.'''
        if self.encoding:
            return self.write_bin(img, self.args.output)
        else:
            return self.write_fn(img, self.args.output)
        

    def encode(self):
        '''Encode an image into a .bin file.'''
        img = self.read()
        io.imsave("temp.png", img)
        img_converted = open("temp.png", 'rb').read()
        compressed_data = zlib.compress(img_converted, COMPRESSION_LEVEL)
        self.write(compressed_data)
        rate = self.output_bytes*8/(img.shape[0]*img.shape[1])
        #logging.info({compressed_data})
        return rate

    def decode(self):
        compressed_data = self.read()
        decompressed_data = zlib.decompress(compressed_data)
        img = Image.open(readIO.BytesIO(decompressed_data))
        self.write(img)
        return self.input_bytes*8/(img.size[0]*img.size[1])


if __name__ == "__main__":
    main.main(parser, logging, CoDec)