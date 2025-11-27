'''Add a constant value.'''

import logging
import main
import importlib
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser

default_quantizer = "deadzone"
default_offset = 0

parser.parser_encode.add_argument("-a", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)
parser.parser_encode.add_argument("-x", "--offset", help=f"Offset (default: {default_offset})", default=default_offset)

parser.parser_decode.add_argument("-a", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)
parser.parser_decode.add_argument("-x", "--offset", help=f"Offset (default: {default_offset})", default=default_offset)

args = parser.parser.parse_known_args()[0]
Q = importlib.import_module(args.quantizer)

class CoDec(Q.CoDec):

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        self.offset = args.offset

    def encode(self):
        logging.debug("trace")
        img = self.encode_read()
        img = img.astype(np.int16)
        img += self.offset
        k = self.quantize(img)
        logging.debug(f"k = {k}")
        k += self.offset
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
        k -= self.offset
        #k[..., 1] -= 128
        #k[..., 2] -= 128
        YCoCg_y = self.dequantize(k)
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
        output_size = self.decode_write(y)
        #self.BPP = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        #RMSE = distortion.RMSE(self.encode_read(), y)
        #logging.info(f"RMSE = {RMSE}")
        return output_size

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
