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
        logging.debug("trace")
        super().__init__(args)
        if args.quantizer == "deadzone":
            self.offset = 128
        else:
            self.offset = 0

    def encode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        img = self.encode_read_fn(in_fn)
        img = img.astype(np.int16)
        img -= self.offset
        RGB_img = img
        k = self.quantize_fn(RGB_img, out_fn)
        logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        k += self.offset
        if np.max(k) > 255:
            logging.warning(f"k[{np.unravel_index(np.argmax(k),k.shape)}]={np.max(k)}")
        if np.min(k) < 0:
            logging.warning(f"k[{np.unravel_index(np.argmin(k),k.shape)}]={np.min(k)}")
        k = k.astype(np.uint16)
        compressed_k = self.compress_fn(k, in_fn)
        output_size = self.encode_write_fn(compressed_k, out_fn)
        return output_size

    def decode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        compressed_k = self.decode_read_fn(in_fn)
        k = self.decompress_fn(compressed_k, in_fn)
        k = k.astype(np.int16)
        logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        k -= self.offset
        RGB_y = self.dequantize_fn(k, in_fn)
        y = RGB_y
        y += self.offset
        logging.debug(f"y.shape={y.shape}, y.type={y.dtype}")
        if np.max(y) > 255:
            logging.warning(f"y[{np.unravel_index(np.argmax(y),y.shape)}]={np.max(y)}")
        if np.min(y) < 0:
            logging.warning(f"y[{np.unravel_index(np.argmin(y),y.shape)}]={np.min(y)}")
        y = np.clip(y, 0, 255).astype(np.uint8)
        output_size = self.decode_write_fn(y, out_fn)
        return output_size

    def encode(self):
        return self.encode_fn(in_fn=self.args.input, out_fn=self.args.output)

    def decode(self):
        return self.decode_fn(in_fn=self.args.input, out_fn=self.args.output)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
