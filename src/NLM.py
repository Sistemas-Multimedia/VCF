
'''Non-Local Means filter. *** Effective only when decoding! ***'''

import numpy as np
import logging
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)
import parser
import main
import importlib
import cv2

default_h = 10
default_templateWindowSize = 7
default_searchWindowSize = 21

parser.parser_decode.add_argument("--nlm_h", type=parser.int_or_str, help=f"NLM h parameter (default: {default_h})", default=default_h)
parser.parser_decode.add_argument("--nlm_templateWindowSize", type=parser.int_or_str, help=f"NLM templateWindowSize (default: {default_templateWindowSize})", default=default_templateWindowSize)
parser.parser_decode.add_argument("--nlm_searchWindowSize", type=parser.int_or_str, help=f"NLM searchWindowSize (default: {default_searchWindowSize})", default=default_searchWindowSize)
import no_filter

args = parser.parser.parse_known_args()[0]

class CoDec(no_filter.CoDec):
    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        self.args = args

    def decode(self):
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")
        y = self.filter(k)
        output_size = self.decode_write(y)
        return output_size

    def filter(self, img):
        logging.debug(f"trace y={img}")
        logging.info(f"NLM params: h={self.args.nlm_h}, templateWindowSize={self.args.nlm_templateWindowSize}, searchWindowSize={self.args.nlm_searchWindowSize}")

        # Asegurarnos de que los parámetros sean enteros
        h = int(self.args.nlm_h)
        templateWS = int(self.args.nlm_templateWindowSize)
        searchWS = int(self.args.nlm_searchWindowSize)

        if len(img.shape) == 2:
            # Grayscale
            return cv2.fastNlMeansDenoising(img, None, h, templateWS, searchWS)
        else:
            # Color
            return cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWS, searchWS)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
