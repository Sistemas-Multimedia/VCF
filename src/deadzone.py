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
default_filter = "no_filter"

#_parser, parser_encode, parser_decode = parser.create_parser(description=__doc__)

#parser.parser_encode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
#parser.parser_decode.add_argument("-e", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)
parser.parser_encode.add_argument("-q", "--QSS", type=parser.int_or_str, help=f"Quantization step size (default: {default_QSS})", default=default_QSS)

parser.parser_decode.add_argument("-q", "--QSS", type=parser.int_or_str, help=f"Quantization step size (default: {default_QSS})", default=default_QSS)
parser.parser_decode.add_argument("-f", "--filter", type=parser.int_or_str, help=f"Denoising filter (default: {default_filter})", default=default_filter)

args = parser.parser.parse_known_args()[0]
try:
    print("Denoising filter =", args.filter)
    denoiser = importlib.import_module(args.filter)
except:
    # Remember that the filter is only active when decoding.
    denoiser = importlib.import_module("no_filter")

#EC = importlib.import_module(args.entropy_image_codec)
#denoiser = importlib.import_module(blur)

class CoDec(denoiser.CoDec):

    def __init__(self, args, min_index_val=0, max_index_val=255):
        logging.debug(f"trace args={args}")
        logging.debug(f"trace min_index_val={min_index_val}")        
        logging.debug(f"trace max_index_val={max_index_val}")        
        super().__init__(args)
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
        self.total_output_size = 1 # We suppose that the representation of the QSS requires 1 byte in the code-stream.

    def encode(self):
        logging.debug("trace")
        img = self.encode_read()
        logging.debug(f"Input image with range [{np.min(img)}, {np.max(img)}]")
        # Remember that in a deadzone quantizer the input should be
        # positive and negative, but this only makes sense when the
        # signal, by nature, is positive and negative.
        img_128 = img.astype(np.int16) #- 128
        logging.debug(f"Input to quantizer with range [{np.min(img_128)}, {np.max(img_128)}]")
        k = self.quantize(img_128).astype(np.uint8)
        logging.debug(f"Input to entropy compressor with range [{np.min(k)}, {np.max(k)}]")
        compressed_k = self.compress(k)
        output_size = self.encode_write(compressed_k)
        return output_size

    def decode(self):
        logging.debug("trace")
        compressed_k = self.decode_read()
        k_128 = self.decompress(compressed_k)
        logging.debug(f"Output from entropy decompressor with range [{np.min(k_128)}, {np.max(k_128)}]")
        y_128 = self.dequantize(k_128)
        logging.debug(f"Output from dequantizer with range [{np.min(y_128)}, {np.max(y_128)}]")
        y = y_128 #+ 128 
        logging.debug(f"y.shape={y.shape} y.dtype={y.dtype}")        
        y = denoiser.CoDec.filter(self, y)
        output_size = self.decode_write(y)
        return output_size
    
    def quantize(self, img):
        logging.debug(f"trace img={img}")
        k = self.Q.encode(img)
        #k += 128 # Only positive components can be written in a PNG file
        #k = k.astype(np.uint8)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype} max(x)={np.max(k)} min(k)={np.min(k)}")
        return k

    def dequantize(self, k):
        '''"Dequantize" an image.'''
        logging.debug(f"trace k={k}")
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
