'''Image quantization using a deadzone scalar quantizer.'''

import numpy as np
import logging
import main
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)
import parser

from scalar_quantization.deadzone_quantization import Deadzone_Quantizer as Quantizer # pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"
from scalar_quantization.deadzone_quantization import name as quantizer_name # pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"

from information_theory import distortion # pip install "information_theory @ git+https://github.com/vicente-gonzalez-ruiz/information_theory"

#import entropy_image_coding as EIC
#import blur as denoiser
#import blur

import importlib
  
default_QSS = 32
#default_EIC = "PNG"
#default_filter_size = 3
#default_filter = "none"

#_parser, parser_encode, parser_decode = parser.create_parser(description=__doc__)

#parser.parser_encode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
#parser.parser_decode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
parser.parser_encode.add_argument("-q", "--QSS", type=parser.int_or_str, help=f"Quantization step size (default: {default_QSS})", default=default_QSS)
parser.parser_decode.add_argument("-q", "--QSS", type=parser.int_or_str, help=f"Quantization step size (default: {default_QSS})", default=default_QSS)
#parser.parser_decode.add_argument("-f", "--filter", help=f"Filter name (none, gaussian, median or blur) (default: {default_filter})", default=default_filter)
#parser.parser_decode.add_argument("-s", "--filter_size", type=parser.int_or_str, help=f"Filter size (default: {default_filter_size})", default=default_filter_size)

args = parser.parser.parse_known_args()[0]
denoiser = importlib.import_module("blur")
#EC = importlib.import_module(args.entropy_image_codec)
#denoiser = importlib.import_module(blur)

class CoDec(denoiser.CoDec):

    def __init__(self, args, min_index_val=0, max_index_val=255):
        logging.debug("parse")
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        #if self.encoding:
        #    self.QSS = args.QSS
        #    logging.info(f"QSS = {self.QSS}")
        #    with open(f"{args.output}_QSS.txt", 'w') as f:
        #        f.write(f"{self.args.QSS}")
        #        logging.debug(f"Written {self.args.QSS} in {self.args.output}_QSS.txt")
        #else:
        #    with open(f"{args.input}_QSS.txt", 'r') as f:
        #        self.QSS = int(f.read())
        #        logging.debug(f"Read QSS={self.QSS} from {self.args.output}_deadzone.txt")
        self.QSS = args.QSS
        self.Q = Quantizer(Q_step=self.QSS, min_val=min_index_val, max_val=max_index_val)
        self.output_bytes = 1 # We suppose that the representation of the QSS requires 1 byte in the code-stream.

    def UNUSED_compress(self, img):
        k = self.quantize(img).astype(np.uint8)
        compressed_k = super().compress(k)
        return compressed_k
        
    def encode(self):
        '''Read an image, quantize the image, and save it.'''
        logging.debug("parse")
        img = self.encode_read()
        logging.debug(f"Input image with range [{np.min(img)}, {np.max(img)}]")
        img_128 = img.astype(np.int16) - 128
        logging.debug(f"Input to quantizer with range [{np.min(img_128)}, {np.max(img_128)}]")
        k = self.quantize(img_128).astype(np.uint8)
        #k = self.quantize(img).astype(np.uint8)
        #k = img
        #print("---------------", np.max(k))
        #logging.debug(f"k.shape={k.shape} k.dtype={k.dtype} k.max={np.max(k)} k.min={np.min(k)}")
        logging.debug(f"Input to entropy compressor with range [{np.min(k)}, {np.max(k)}]")
        compressed_k = self.compress(k)
        self.encode_write(compressed_k)
        #self.save(img)
        #rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        #return rate
        return self.output_bytes

    def UNUSED_decompress(self, compressed_k):
        logging.debug("parse")
        k = super().decompress(compressed_k)
        y = self.dequantize(k)#.astype(np.uint8)
        return y
        
    '''
    def _decompress(self, compressed_k):
        k = super().decompress(compressed_k)
        #k = k.astype(np.uint8)
        y = self.dequantize(k)
        return y
    '''
    
    def decode(self):
        '''Read a quantized image, "dequantize", and save.'''
        logging.debug("parse")
        compressed_k = self.decode_read()
        k_128 = self.decompress(compressed_k)
        logging.debug(f"Output from entropy decompressor with range [{np.min(k_128)}, {np.max(k_128)}]")
        #y_128 = self.dequantize(k)
        #y = (np.rint(y_128).astype(np.int16) + 128).astype(np.uint8)
        #y = self.dequantize(k).astype(np.uint8)
        y_128 = self.dequantize(k_128)
        logging.debug(f"Output from dequantizer with range [{np.min(y_128)}, {np.max(y_128)}]")
        y = y_128 + 128 
        #y = k
        #print("---------------", np.max(y))
        logging.debug(f"y.shape={y.shape} y.dtype={y.dtype}")        
        y = denoiser.CoDec.filter(self, y)
        self.decode_write(y)
        #rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        #RMSE = distortion.RMSE(img, y)
        #return RMSE

    def quantize(self, img):
        '''Quantize the image.'''
        logging.debug("parse")
        k = self.Q.encode(img)
        #k += 128 # Only positive components can be written in a PNG file
        #k = k.astype(np.uint8)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype} max(x)={np.max(k)} min(k)={np.min(k)}")
        return k

    def dequantize(self, k):
        '''"Dequantize" an image.'''
        logging.debug("parse")
        #k = k.astype(np.int16)
        #k -= 128
        #self.Q = Quantizer(Q_step=QSS, min_val=min_index_val, max_val=max_index_val)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype} max(x)={np.max(k)} min(k)={np.min(k)}")
        y = self.Q.decode(k)
        logging.debug(f"y.shape={y.shape} y.dtype={y.dtype}")
        return y

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
    logging.info(f"quantizer = {quantizer_name}")
