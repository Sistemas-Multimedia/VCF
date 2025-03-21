'''Exploiting color (perceptual) redundancy with the YCoCg transform.'''

import numpy as np
import logging
import main
import importlib
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser

from color_transforms.YCoCg import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import to_RGB
#from information_theory import distortion # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"

default_quantizer = "deadzone"

#_parser, parser_encode, parser_decode = parser.create_parser(description=__doc__)

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
        img = img.astype(np.int16)
        img -= 128
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
        img = self.encode_read_fn(in_fn)
        img = img.astype(np.int16)
        # Specific for solving the issue https://github.com/vicente-gonzalez-ruiz/scalar_quantization/issues/1
        #img_128 = img.astype(np.int16) - 128
        #YCoCg_img_128 = from_RGB(img_128)
        #YCoCg_img = YCoCg_img_128 + 128
        img -= self.offset
        YCoCg_img = from_RGB(img)
        #YCoCg_img[..., 1] += 128
        #YCoCg_img[..., 2] += 128
        #logging.debug(f"max(YCoCg_img)={np.max(YCoCg_img)}, min(YCoCg_img)={np.min(YCoCg_img)}")
        #assert (YCoCg_img < 256).all()
        #assert (YCoCg_img >= 0).all()
        k = self.quantize_fn(YCoCg_img, out_fn)
        logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        #k = YCoCg_img
        #k[..., 1] += 128
        #k[..., 2] += 128
        k += self.offset
        if np.max(k) > 255:
            logging.warning(f"k[{np.unravel_index(np.argmax(k),k.shape)}]={np.max(k)}")
        if np.min(k) < 0:
            logging.warning(f"k[{np.unravel_index(np.argmin(k),k.shape)}]={np.min(k)}")
        #k = np.clip(k, 0, 255).astype(np.uint8)
        k = k.astype(np.uint16)
        compressed_k = self.compress_fn(k, in_fn)
        output_size = self.encode_write_fn(compressed_k, out_fn)
        #self.BPP = (self.total_output_size*8)/(img.shape[0]*img.shape[1])
        #logging.info(f"BPP = {BPP}")
        return output_size

    def decode_fn(self, in_fn, out_fn):
        logging.debug("trace")
        compressed_k = self.decode_read_fn(in_fn)
        k = self.decompress_fn(compressed_k, in_fn)
        k = k.astype(np.int16)
        logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        k -= self.offset
        #k[..., 1] -= 128
        #k[..., 2] -= 128
        YCoCg_y = self.dequantize_fn(k, in_fn)
        #YCoCg_y = k
#        logging.debug(f"max(YCoCg_y)={np.max(YCoCg_y)}, min(YCoCg_y)={np.min(YCoCg_y)}")
#        assert (YCoCg_y < 256).all()
#        assert (YCoCg_y >= 0).all()
#        logging.debug(f"YCoCg_y.shape={YCoCg_y.shape}, YCoCg_y.type={YCoCg_y.dtype}")
        # Specific for solving the issue https://github.com/vicente-gonzalez-ruiz/scalar_quantization/issues/1
        #YCoCg_y_128 = YCoCg_y.astype(np.int16) - 128
        #y_128 = to_RGB(YCoCg_y_128)
        #y = y_128 + 128
        #YCoCg_y[..., 1] -= 128
        #YCoCg_y[..., 2] -= 128        
        y = to_RGB(YCoCg_y)
        y += self.offset
        logging.debug(f"y.shape={y.shape}, y.type={y.dtype}")
        if np.max(y) > 255:
            logging.warning(f"y[{np.unravel_index(np.argmax(y),y.shape)}]={np.max(y)}")
        if np.min(y) < 0:
            logging.warning(f"y[{np.unravel_index(np.argmin(y),y.shape)}]={np.min(y)}")
        y = np.clip(y, 0, 255).astype(np.uint8)
        #print(dir(Q.denoiser.CoDec))
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
