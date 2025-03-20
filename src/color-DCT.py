'''Exploiting color (perceptual) redundancy with the DCT transform.'''

import numpy as np
import logging
import main
import importlib
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser

from color_transforms.DCT import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.DCT import to_RGB

default_quantizer = "deadzone"

parser.parser_encode.add_argument("-c", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)
parser.parser_decode.add_argument("-c", "--quantizer", help=f"Quantizer (default: {default_quantizer})", default=default_quantizer)

args = parser.parser.parse_known_args()[0]
Q = importlib.import_module(args.quantizer)

class CoDec(Q.CoDec):

    def encode(self):
        logging.debug("trace")
        #
        # Read the image.
        #
        img = self.encode_read()#.astype(np.int16)

        #
        # This provides numerical stability during to the transform
        # because we will reduce the number of bits needed to
        # represent the coefficients. Notice that most image codecs
        # handle only positive integers.
        #
        img = img.astype(np.int16) - 128
        #img = img.astype(np.uint8)
        logging.debug(f"Input to color-DCT with range [{np.min(img)}, {np.max(img)}]")

        #
        # Transform the image. Coefficients are floats ideally (for
        # the DCT) with mean 0.
        #
        coefs = from_RGB(img)

        #
        # The coefs can be positive and negative, but some quantizers
        # (such as LloydMax) only input positive values. For those
        # quantizers, the coefs must be shifted. After this, coefs
        # should fit in [0, 255].
        #
        if (self.args.quantizer == "LloydMax"):
            coefs += 128
            if __debug__:
                #assert (coefs < 256).all()
                if np.max(coefs) > 255:
                    logging.error(f"coefs[{np.unravel_index(np.argmax(coefs), coefs.shape)}]={np.max(coefs)} and LloydMax only accept values < 256. Quitting ...")
                    quit()
                #assert (coefs >= 0).all()
                if np.min(coefs) > 255:
                    logging.error(f"coefs[{np.unravel_index(np.argmin(coefs), coefs.shape)}]={np.min(coefs)} and LloydMax only accept values >= 0. Quitting ...")
                    quit()
        
        #
        # Quantize the coefficients. 
        #
        logging.debug(f"Input to quantizer with range [{np.min(coefs)}, {np.max(coefs)}]")
        k = self.quantize(coefs)
        
        #
        # The entropy codecs that we are using input input values in
        # Z^+. Depending on the quantizer, the quantization indexes
        # (always integers) can be positive or negative. Concretely:
        #
        # * Deadzone: output values in Z (and input values in Z).
        #
        # * LloydMax: output values in Z^+ (and input values in Z^+).
        #
        # * VQ: output values in Z^+ (and input vectors in Z^+^n).
        #
        if (self.args.quantizer == "deadzone"):
            k = k.astype(np.int16)
            k += 16384
            k = k.astype(np.uint16)

        #
        # Compress and write to disk the quantized indexes. Remember
        # that the entropy codecs require positive values at the
        # input.
        #
        logging.debug(f"Input to entropy compressor with range [{np.min(k)}, {np.max(k)}]")
        compressed_k = self.compress(k)
        self.encode_write(compressed_k)

    def decode(self):
        logging.debug("trace")
        #
        # Read and decompress the quantized indexes.
        #
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)

        #
        # If the quantized indexes were shifted, let's restore the
        # original values.
        #
        k = k.astype(np.int16)
        if (self.args.quantizer == "deadzone"):
            k -= 16384
            
        #k = k.astype(np.int16)
        #k -= 128
        #k = self.read()
        #k -= 32768
        #print("------------->", k.dtype, np.max(k), np.min(k))

        #
        # Dequantize the indexes.
        #
        logging.debug(f"Input to dequantizer with range [{np.min(k)}, {np.max(k)}]")
        coefs = self.dequantize(k)

        #
        # Inverse transform.
        #
        #y_128 = to_RGB(YCoCg_y.astype(np.int16))
        #YCrCb_y = YCrCb_y.astype(np.uint8)
        logging.debug(f"Input to inverse color-DCT with range [{np.min(coefs)}, {np.max(coefs)}]")
        y = to_RGB(coefs)

        #
        # Reverse the pixels shift.
        #
        y = y + 128
        

        #
        # Write the image.
        #
        logging.debug(f"Input to entropy decoder with range [{np.min(y)}, {np.max(y)}]")
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.decode_write(y)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
