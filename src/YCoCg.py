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

parser.parser_encode.add_argument("-a", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)

parser.parser_decode.add_argument("-a", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)

args = parser.parser.parse_known_args()[0]
Q = importlib.import_module(args.quantizer)

class CoDec(Q.CoDec):

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        if args.quantizer == "deadzone":
            self.offset = np.array([-128, 0, 0])
        else:
            self.offset = np.array([0, 128, 128])

    def encode(self):
        logging.debug("trace")
        img = self.encode_read()
        img = img.astype(np.int16)
        #img -= self.offset
        YCoCg_img = from_RGB(img)
        #assert (YCoCg_img < 256).all()
        #assert (YCoCg_img >= 0).all()
        for i in range(YCoCg_img.shape[2]):
             YCoCg_img[..., i] += self.offset[i]
        # Now the samples should be centered at zero. In the case of a
        # deadzone quantizer this is the correct configuration. For a
        # LloydMax quantizer, this is OK because the quantizer is
        # adaptive.
        k = self.quantize(YCoCg_img)
        logging.debug(f"k = {k}")
        #k += self.offset
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
        k = k.astype(np.int16)
        logging.debug(f"k.shape={k.shape}, k.type={k.dtype}")
        #k -= self.offset
        YCoCg_y = self.dequantize(k)
        for i in range(YCoCg_y.shape[2]):
             YCoCg_y[..., i] -= self.offset[i]
        #YCoCg_y = k
        #assert (YCoCg_y < 256).all()
        #assert (YCoCg_y >= 0).all()
        y = to_RGB(YCoCg_y)
        #y += self.offset
        logging.debug(f"y.shape={y.shape}, y.type={y.dtype}")
        if self.args.debug:
            if np.max(y) > 255:
                logging.warning(f"y[{np.unravel_index(np.argmax(y),y.shape)}]={np.max(y)}")
            if np.min(y) < 0:
                logging.warning(f"y[{np.unravel_index(np.argmin(y),y.shape)}]={np.min(y)}")
        y = np.clip(y, 0, 255).astype(np.uint8)
        #print(dir(Q.denoiser.CoDec))
        y = Q.denoiser.CoDec.filter(self, y)
        output_size = self.decode_write(y)
        #self.BPP = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        #RMSE = distortion.RMSE(self.encode_read(), y)
        #logging.info(f"RMSE = {RMSE}")
        return output_size

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
