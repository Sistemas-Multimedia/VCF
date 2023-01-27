'''Entropy Encoding of images using Zlib.'''

import logging
import main
import zlib
import PIL.Image as Image
import io as readIO
from skimage import io  # pip install scikit-image
import os
import argparse


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

    def encode(self):
        '''Read an image and save it in the disk.'''
        img_temp = io.imread(self.args.input)
        io.imsave("temp.png", img_temp)
        img = open("temp.png", 'rb').read()

        compressed_data = zlib.compress(img, COMPRESSION_LEVEL)
        with open(self.args.output, 'wb') as out_file:
            out_file.write(compressed_data)
        self.output_bytes = img_temp.size*8
        rate = self.output_bytes/(img_temp.shape[0]*img_temp.shape[1])
        return rate

    def decode(self):
        compressed_data = open(self.args.input, 'rb').read()
        decompressed_data = zlib.decompress(compressed_data)
        img = Image.open(readIO.BytesIO(decompressed_data))
        img.save(self.args.output)
        self.input_bytes = os.path.getsize(self.args.output)
        logging.info(
            f"Written {self.input_bytes} bytes in {self.args.output}")
        return self.input_bytes*8/(img.size[0]*img.size[1])


if __name__ == "__main__":
    main.main(parser, logging, CoDec)
