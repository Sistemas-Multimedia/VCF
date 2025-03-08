'''Dummy RGB (no color) transform.'''

import numpy as np
import logging
import main
import importlib
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser

default_quantizer = "deadzone"

parser.parser_encode.add_argument("-c", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)
parser.parser_decode.add_argument("-c", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)

args = parser.parser.parse_known_args()[0]
Q = importlib.import_module(args.quantizer)

class CoDec(Q.CoDec):

    def __init__(self, args):
        super().__init__(args)
        if args.quantizer == "deadzone":
            self.offset = 128
        else:
            self.offset = 0

    def encode(self):
        img = self.encode_read()
        img = img.astype(np.int16)
        img -= self.offset
        RGB_img = img
        k = self.quantize(RGB_img)
        logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        k += self.offset
        if np.max(k) > 255:
            logging.warning(f"k[{np.unravel_index(np.argmax(k),k.shape)}]={np.max(k)}")
        if np.min(k) < 0:
            logging.warning(f"k[{np.unravel_index(np.argmin(k),k.shape)}]={np.min(k)}")
        k = k.astype(np.uint16)
        compressed_k = self.compress(k)
        self.encode_write(compressed_k)

    def decode(self):
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        k = k.astype(np.int16)
        logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        k -= self.offset
        RGB_y = self.dequantize(k)
        y = RGB_y
        y += self.offset
        logging.debug(f"y.shape={y.shape}, y.type={y.dtype}")
        if np.max(y) > 255:
            logging.warning(f"y[{np.unravel_index(np.argmax(y),y.shape)}]={np.max(y)}")
        if np.min(y) < 0:
            logging.warning(f"y[{np.unravel_index(np.argmin(y),y.shape)}]={np.min(y)}")
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.decode_write(y)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
