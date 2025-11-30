'''No spatial transform.'''

import io
import numpy as np
import os
import logging
import main
import parser
import importlib
import struct
from color_transforms.YCoCg import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import to_RGB

with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)

default_CT = "YCoCg"

parser.parser_encode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)

parser.parser_decode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Color transform (default: \"{default_CT}\")", default=default_CT)

args = parser.parser.parse_known_args()[0]
CT = importlib.import_module(args.color_transform)

class CoDec(CT.CoDec):

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        if args.quantizer == "deadzone":
            self.offset = 128 # Ojo con esto
        else:
            self.offset = 0

    def encode(self):
        logging.debug("trace")
        img = self.encode_read().astype(np.int32)

        #
        # Provides numperical stability to the DCT.
        #
        #img -= self.offset
        logging.debug(f"Input to color-DCT with range [{np.min(img)}, {np.max(img)}]")
        #print(f"{np.min(img)}, {np.max(img)}")

        CT_img = from_RGB(img)
        #print(f"{CT_img.dtype} {np.min(CT_img)}, {np.max(CT_img)}")
        k = self.quantize(CT_img)#.astype(np.uint8)
        k += self.offset
        k = k.astype(np.uint8)
        print(f"{k.dtype} {np.min(k)}, {np.max(k)}")
        compressed_k = self.compress(k)
        output_size = self.encode_write(compressed_k)
        return output_size

    def decode(self):
        logging.debug("trace")
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        k -= self.offset
        CT_y = self.dequantize(k)
        y = to_RGB(CT_y)
        output_size = self.decode_write(y)
        return output_size

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
