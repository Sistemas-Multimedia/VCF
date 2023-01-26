'''Entropy Encoding of images using Zlib.'''

import logging
import main
import zlib

import PNG as EC


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


COMPRESSION_LEVEL = 9


class CoDec(EC.CoDec):

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
        '''Read an image and save it in the disk. The input can be
        online. This method is overriden in child classes.'''
        img = self.read_fn(self.args.input)
        compressed_data = zlib.compress(img)
        self.write_fn(self.args.output, compressed_data)
        self.write(img)
        logging.debug(
            f"output_bytes={self.output_bytes}, img.shape={img.shape}")
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        '''Read an image and save it in the disk. Notice that we are
        using the PNG image format for both, decode and encode an
        image. For this reason, both methods do exactly the same.
        This method is overriden in child classes.

        '''
        return self.encode()


if __name__ == "__main__":
    main.main(EC.parser, logging, CoDec)
