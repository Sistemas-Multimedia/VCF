import argparse
from skimage import io # pip install scikit-image
import numpy as np
import logging
import main

import PNG as EC
import deadzone as Q

from color_transforms.YCrCb import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCrCb import to_RGB

class CoDec(Q.CoDec):

    def encode(self):
        img = self.read()
        YCoCg_img = from_RGB(img.astype(np.uint8))
        k = self.quantize(YCoCg_img)
        self.write(k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        k = self.read()
        YCoCg_y = self.dequantize(k)
        y = to_RGB(YCoCg_y.astype(np.uint8))
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.write(y)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

if __name__ == "__main__":
    main.main(EC.parser, logging, CoDec)
