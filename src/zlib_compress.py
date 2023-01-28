'''Entropy Encoding of images using PNG (Portable Network Graphics).'''

import argparse
import logging
import os
import main
import zlib


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
ENCODE_INPUT = "/workspaces/VCF/marco.png"
ENCODE_OUTPUT = "/workspaces/VCF/encoded"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/workspaces/VCF/marco_2.png"

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

    def encode(self):
        f = open(self.args.input, "rb")
        img = f.read()
        f.close()

        comp = zlib.compress(img)

        f = open(self.args.output, "wb")
        f.write(comp)
        f.close()

        self.input_bytes = os.path.getsize(self.args.input)
        self.output_bytes = os.path.getsize(self.args.output)

        return 0

    def decode(self):
        f = open(self.args.input, "rb")
        img = f.read()
        f.close()

        comp = zlib.decompress(img)

        f = open(self.args.output, "wb")
        f.write(comp)
        f.close()

        self.input_bytes = os.path.getsize(self.args.input)
        self.output_bytes = os.path.getsize(self.args.output)

        return 0

    def __del__(self):
        logging.info(f"Total {self.input_bytes} bytes read")
        logging.info(f"Total {self.output_bytes} bytes written")


if __name__ == "__main__":
    main.main(parser, logging, CoDec)
