'''Exploiting color (perceptual) redundancy with the YCrCb transform.'''

import numpy as np
import logging
import main
import importlib
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser

from color_transforms.YCrCb import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCrCb import to_RGB

default_quantizer = "deadzone"

parser.parser_encode.add_argument("-a", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)

parser.parser_decode.add_argument("-a", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)

args = parser.parser.parse_known_args()[0]
Q = importlib.import_module(args.quantizer)

class CoDec(Q.CoDec):

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        if args.quantizer == "deadzone":
            self.offset = np.array([0, 0, 0])
        else:
            self.offset = np.array([0, 0, 0])

    def encode(self):
        logging.debug("trace")
        img = self.encode_read()
        YCrCb_img = from_RGB(img).astype(np.int16)
        print(YCrCb_img.dtype)
        # Residues should be centered at zero if deadzone.
        for i in range(YCrCb_img.shape[2]):
             YCrCb_img[..., i] += self.offset[i]
        k = self.quantize(YCrCb_img)
        if self.args.debug:
            if np.max(k) > 255:
                logging.warning(f"k[{np.unravel_index(np.argmax(k),k.shape)}]={np.max(k)}")
            if np.min(k) < 0:
                logging.warning(f"k[{np.unravel_index(np.argmin(k),k.shape)}]={np.min(k)}")
        k = k.astype(np.uint16)        
        compressed_k = self.compress(k)
        output_size = self.encode_write(compressed_k)
        return output_size

    def decode(self):
        logging.debug("trace")
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        YCrCb_y = self.dequantize(k).astype(np.int16)
        for i in range(YCrCb_y.shape[2]):
             YCrCb_y[..., i] -= self.offset[i]
        YCrCb_y = YCrCb_y.astype(np.uint8)
        y = to_RGB(YCrCb_y)
        if self.args.debug:
            if np.max(y) > 255:
                logging.warning(f"y[{np.unravel_index(np.argmax(y),y.shape)}]={np.max(y)}")
            if np.min(y) < 0:
                logging.warning(f"y[{np.unravel_index(np.argmin(y),y.shape)}]={np.min(y)}")
        y = np.clip(y, 0, 255).astype(np.uint8)
        y = Q.denoiser.CoDec.filter(self, y)
        output_size = self.decode_write(y)
        return output_size

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
