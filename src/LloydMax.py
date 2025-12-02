'''Image quantization using a LloydMax quantizer.'''

# Some work could be done with the encoded histograms!

import os
import numpy as np
import gzip
import logging
import main
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser

from scalar_quantization.LloydMax_quantization import LloydMax_Quantizer as Quantizer # pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"
from scalar_quantization.LloydMax_quantization import name as quantizer_name # pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"

#from quantization import EC

#import entropy_image_coding as EIC
import importlib

default_QSS = 32
#default_EIC = "PNG"
default_filter = "no_filter"
default_min_val = 0
default_max_val = 255

parser.parser_encode.add_argument("-q", "--QSS", type=parser.int_or_str, help=f"Quantization step size (default: {default_QSS})", default=default_QSS)
parser.parser_encode.add_argument("-m", "--min_val", type=parser.int_or_str, help=f"Default min_val (default: {default_min_val})", default=default_min_val)
parser.parser_encode.add_argument("-n", "--max_val", type=parser.int_or_str, help=f"Default max_val (default: {default_max_val})", default=default_max_val)

parser.parser_decode.add_argument("-q", "--QSS", type=parser.int_or_str, help=f"Quantization step size (default: {default_QSS})", default=default_QSS)
parser.parser_decode.add_argument("-f", "--filter", type=parser.int_or_str, help=f"Denoising filter (default: {default_filter})", default=default_filter)
parser.parser_decode.add_argument("-m", "--min_val", type=parser.int_or_str, help=f"Default min_val (default: {default_min_val})", default=default_min_val)
parser.parser_decode.add_argument("-n", "--max_val", type=parser.int_or_str, help=f"Default max_val (default: {default_max_val})", default=default_max_val)

args = parser.parser.parse_known_args()[0]
try:
    #print("Denoising filter =", args.filter)
    denoiser = importlib.import_module(args.filter)
except:
    # Remember that the filter is only active when decoding.
    denoiser = importlib.import_module("no_filter")

class CoDec(denoiser.CoDec):
    
    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        self.min_val = args.min_val
        self.max_val = args.max_val

    def encode(self):
        '''Read, quantize, and write an image.'''
        logging.debug("trace")
        img = self.encode_read()
        k = self.quantize(img)
        compressed_k = self.compress(k)
        output_size = self.encode_write(compressed_k)
        return output_size

    def decode(self):
        '''Read, dequantize, and write an image.'''
        logging.debug("trace")
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        y = self.dequantize(k)
        y = denoiser.CoDec.filter(self, y)
        output_size = self.decode_write(y)
        return output_size

    def quantize_fn(self, img, fn):
        '''Quantize img.'''
        logging.debug("trace")
        logging.debug(f"trace img={img}")
        logging.debug(f"trace fn={fn}")
        logging.info(f"QSS = {self.args.QSS}")
        with open(f"{fn}_QSS.txt", 'w') as f:
            f.write(f"{self.args.QSS}")
        self.output_bytes = 1 # We suppose that the representation of the QSS requires 1 byte
        logging.info(f"Written {fn}_QSS.txt")
        if len(img.shape) < 3:
            extended_img = np.expand_dims(img, axis=2)
        else:
            extended_img = img
        k = np.empty_like(extended_img)
        for c in range(extended_img.shape[2]):
            histogram_img, bin_edges_img = np.histogram(
                extended_img[..., c],
                bins=(self.max_val - self.min_val + 1),
                range=(self.min_val, self.max_val))
            logging.info(f"histogram = {histogram_img}")
            histogram_img += 1 # Bins cannot be zero
            self.Q = Quantizer(
                Q_step=self.args.QSS,
                counts=histogram_img,
                min_val=self.min_val,
                max_val=self.max_val)
            centroids = self.Q.get_representation_levels()
            with gzip.GzipFile(f"{fn}_centroids_{c}.gz", "w") as f:
                np.save(file=f, arr=centroids)
            len_codebook = os.path.getsize(f"{fn}_centroids_{c}.gz")
            logging.info(f"Written {len_codebook} bytes in {self.args.encoded}_centroids_{c}.gz")
            self.total_output_size += len_codebook
            k[..., c] = self.Q.encode(extended_img[..., c])
        return k

    def quantize(self, img, fn="/tmp/encoded"):
        return self.quantize_fn(img, fn)

    def dequantize_fn(self, k, fn):
        '''Dequantize k.'''
        logging.debug("trace")
        logging.debug(f"trace k = {k}")
        with open(f"{fn}_QSS.txt", 'r') as f:
            QSS = int(f.read())
        logging.info(f"Read QSS={QSS} from {fn}_QSS.txt")
        if len(k.shape) < 3:
            extended_k = np.expand_dims(k, axis=2)
        else:
            extended_k = k
        y = np.empty_like(extended_k)
        for c in range(y.shape[2]):
            with gzip.GzipFile(f"{fn}_centroids_{c}.gz", "r") as f:
                centroids = np.load(file=f)
            logging.info(f"Read {fn}_centroids_{c}.gz")
            self.Q = Quantizer(Q_step=QSS, counts=np.ones(shape=(self.max_val - self.min_val + 1)))
            self.Q.set_representation_levels(centroids)
            y[..., c] = self.Q.decode(extended_k[..., c])
        return y

    def dequantize(self, k, fn="/tmp/encoded"):
        return self.dequantize_fn(k, fn)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
    logging.info(f"quantizer = {quantizer_name}")
