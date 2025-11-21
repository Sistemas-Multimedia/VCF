'''Image blurring using low-pass filtering. *** Effective only when decoding! ***'''

import numpy as np
import logging
import main
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)
#import entropy_image_coding as EIC
import importlib
import cv2

import parser
#import entropy_image_coding as EIC
#import importlib

default_filter_size = 3
default_filter = "none"
default_EIC = "TIFF"

#_parser, parser_encode, parser_decode = parser.create_parser(description=__doc__)

# Encoder parser
parser.parser_encode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)

# Decoder parser
parser.parser_decode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
parser.parser_decode.add_argument("-f", "--filter", help=f"Filter name (none, gaussian, median or blur) (default: {default_filter})", default=default_filter)
parser.parser_decode.add_argument("-s", "--filter_size", type=parser.int_or_str, help=f"Filter size (default: {default_filter_size})", default=default_filter_size)

args = parser.parser.parse_known_args()[0]
EC = importlib.import_module(args.entropy_image_codec)

class CoDec(EC.CoDec):

    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        self.args = args
        if self.encoding:
            self.filter = "none"
            self.filter_size = 0

    def decode_fn(self, in_fn, out_fn):
        logging.debug(f"trace in_fn={in_fn}")
        logging.debug(f"trace out_fn={out_fn}")
        compressed_k = self.decode_read_fn(in_fn)
        k = self.decompress_fn(compressed_k, in_fn)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")        
        y = self.filter(k)
        output_size = self.decode_write_fn(y, out_fn)
        return output_size
            
    def decode(self):
        logging.debug("trace")
        self.decode_fn(in_fn=self.args.input, out_fn=self.args.output)
        '''
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")        
        y = self.filter(k)
        output_size = self.decode_write(y)
        return output_size
        '''
            
    def filter(self, img):
        logging.debug(f"trace y={img}")
        logging.info(f"Using filter {self.args.filter} with size {self.args.filter_size}")
        if self.args.filter == "gaussian":
            return cv2.GaussianBlur(img, (self.args.filter_size, self.args.filter_size), 0)
        elif self.args.filter == "median":
            return cv2.medianBlur(img, self.args.filter_size)
        elif self.args.filter == "blur":
            return cv2.blur(img, (self.args.filter_size, self.args.filter_size))
        else:
            return y

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
