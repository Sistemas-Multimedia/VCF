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

    def UNUSED_compress(self, img):
        YCrCb_img = from_RGB(img)
        compressed_k = super().compress(YCrCb_img)
        return compressed_k

    def UNUSED_decompress(self, compressed_k):
        YCrCb_y = super().decompress(compressed_k)
        y = to_RGB(YCrCb_y)
        y = np.clip(y, 0, 255)
        y = y.astype(np.uint8)
        return y

    def encode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        img = self.encode_read_fn(in_fn)#.astype(np.int16)
        #print("----------------->", img.dtype)
        #img -= 128
        #img_128 = img.astype(np.int16) - 128
        #img = img.astype(np.uint8)
        YCrCb_img = from_RGB(img)
        k = self.quantize_fn(YCrCb_img, out_fn)
        #k += 128
        if np.max(k) > 255:
            logging.warning(f"k[{np.unravel_index(np.argmax(k),k.shape)}]={np.max(k)}")
        if np.min(k) < 0:
            logging.warning(f"k[{np.unravel_index(np.argmin(k),k.shape)}]={np.min(k)}")
        #k = k.astype(np.uint8)
        k = k.astype(np.uint16)
        
        compressed_k = self.compress_fn(k, in_fn)
        output_size = self.encode_write_fn(compressed_k, out_fn)
        #self.BPP = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        #logging.info(f"BPP = {BPP}")
        return output_size

    def decode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        compressed_k = self.decode_read_fn(in_fn)
        k = self.decompress_fn(compressed_k, in_fn)
        #k = k.astype(np.int16)
        #k -= 128
        #k = self.read()
        YCrCb_y = self.dequantize_fn(k, in_fn)
        #y_128 = to_RGB(YCoCg_y.astype(np.int16))
        YCrCb_y = YCrCb_y.astype(np.uint8)
        y = to_RGB(YCrCb_y)
        #y += 128
        #y = (y_128.astype(np.int16) + 128)
        if np.max(y) > 255:
            logging.warning(f"y[{np.unravel_index(np.argmax(y),y.shape)}]={np.max(y)}")
        if np.min(y) < 0:
            logging.warning(f"y[{np.unravel_index(np.argmin(y),y.shape)}]={np.min(y)}")
        y = np.clip(y, 0, 255).astype(np.uint8)
        y = Q.denoiser.CoDec.filter(self, y)
        output_size = self.decode_write_fn(y, out_fn)
        #self.BPP = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        #RMSE = distortion.RMSE(self.encode_read(), y)
        #logging.info(f"RMSE = {RMSE}")
        return output_size

    def encode(self):
        return self.encode_fn(in_fn=self.args.input, out_fn=self.args.output)

    def decode(self):
        return self.decode_fn(in_fn=self.args.input, out_fn=self.args.output)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
