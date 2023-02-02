'''Exploiting spatial (perceptual) redundancy with the 2D dyadic Discrete Wavelet Transform.'''

import argparse
from skimage import io # pip install scikit-image
import numpy as np
import pywt
import os
import logging
import main

import PNG as EC
import YCoCg as CT # Color Transform

#from DWT import color_dyadic_DWT as DWT
from DWT2D.color_dyadic_DWT import analyze as space_analyze # pip install "DWT2D @ git+https://github.com/vicente-gonzalez-ruiz/DWT2D"
from DWT2D.color_dyadic_DWT import synthesize as space_synthesize

from color_transforms.YCoCg import from_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import to_RGB

EC.parser.add_argument("-l", "--levels", type=EC.int_or_str, help=f"Number of decomposition levels (default: 5)", default=5)
EC.parser_encode.add_argument("-w", "--wavelet", type=EC.int_or_str, help=f"Wavelet name (default: \"db5\")", default="db5")

class CoDec(CT.CoDec):

    def __init__(self, args):
        super().__init__(args)
        self.levels = args.levels
        logging.info(f"levels = {self.levels}")
        if self.encoding:
            self.wavelet = pywt.Wavelet(args.wavelet)
            with open(f"{args.output}_wavelet_name.txt", "w") as f:
                f.write(f"{args.wavelet}")
                logging.info(f"Written {args.output}_wavelet_name.txt")
            logging.info(f"wavelet={args.wavelet} ({self.wavelet})")
        else:
            with open(f"{args.input}_wavelet_name.txt", "r") as f:
                wavelet_name = f.read()
                logging.info(f"Read wavelet = \"{wavelet_name}\" from {args.input}_wavelet_name.txt")
                self.wavelet = pywt.Wavelet(wavelet_name)
            logging.info(f"wavelet={wavelet_name} ({self.wavelet})")

    def encode(self):
        img = self.read()
        img_128 = img.astype(np.int16) - 128
        CT_img = from_RGB(img_128)
        decom_img = space_analyze(CT_img, self.wavelet, self.levels)
        logging.debug(f"len(decom_img)={len(decom_img)}")
        decom_k = self.quantize_decom(decom_img)
        self.write_decom(decom_k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        decom_k = self.read_decom()
        decom_y = self.dequantize_decom(decom_k)
        CT_y = space_synthesize(decom_y, self.wavelet, self.levels)
        y_128 = to_RGB(CT_y)
        y = (y_128.astype(np.int16) + 128)
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.write(y)
        rate = (self.input_bytes*8)/(y.shape[0]*y.shape[1])
        return rate

    def quantize_decom(self, decom):
        decom_k = [self.quantize(decom[0])] # LL subband
        for spatial_resolution in decom[1:]:
            spatial_resolution_k = []
            for subband in spatial_resolution:
                subband_k = self.quantize(subband)
                spatial_resolution_k.append(subband_k)
            decom_k.append(tuple(spatial_resolution_k))
        return decom_k

    def dequantize_decom(self, decom_k):
        decom_y = [self.dequantize(decom_k[0])]
        for spatial_resolution_k in decom_k[1:]:
            spatial_resolution_y = []
            for subband_k in spatial_resolution_k:
                subband_y = self.dequantize(subband_k)
                spatial_resolution_y.append(subband_y)
            decom_y.append(tuple(spatial_resolution_y))
        return decom_y

    def read_decom(self):
        fn_without_extension = self.args.input.split('.')[0]
        fn = f"{fn_without_extension}_LL_{self.levels}.png"
        LL = self.read_fn(fn)
        decom = [LL]
        resolution_index = self.levels
        for l in range(self.levels, 0, -1):
            subband_names = ["LH", "HL", "HH"]
            spatial_resolution = []
            for subband_name in subband_names:
                fn = f"{fn_without_extension}_{subband_name}_{resolution_index}.png"
                subband = self.read_fn(fn)
                spatial_resolution.append(subband)
            decom.append(tuple(spatial_resolution))
            resolution_index -= 1
        return decom

    def write_decom(self, decom):
        LL = decom[0]
        fn_without_extension = self.args.output.split('.')[0]
        fn = f"{fn_without_extension}_LL_{self.levels}.png"
        self.write_fn(LL, fn)
        resolution_index = self.levels
        #aux_decom = [decom[0][..., 0]] # Used for computing slices
        for spatial_resolution in decom[1:]:
            subband_names = ["LH", "HL", "HH"]
            subband_index = 0
            #aux_resol = [] # Used for computing slices
            for subband_name in subband_names:
                fn = f"{fn_without_extension}_{subband_name}_{resolution_index}.png"
                self.write_fn(spatial_resolution[subband_index], fn)
                #aux_resol.append(spatial_resolution[subband_index][..., 0])
                subband_index += 1
            resolution_index -= 1
            #aux_decom.append(tuple(aux_resol))
        #self.slices = pywt.coeffs_to_array(aux_decom)[1]
        #return slices

    def quantize(self, img):
        '''Quantize the image.'''
        k = self.Q.encode(img)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype} max(x)={np.max(k)} min(k)={np.min(k)}")
        k += 32768
        k = k.astype(np.uint16)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")
        return k

    def dequantize(self, k):
        '''"Dequantize" an image.'''
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")
        k = k.astype(np.int16)
        k -= 32768
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")
        #self.Q = Quantizer(Q_step=QSS, min_val=min_index_val, max_val=max_index_val)
        y = self.Q.decode(k)
        logging.debug(f"y.shape={y.shape} y.dtype={y.dtype}")
        return y

    '''
    def __save_fn(self, img, fn):
        io.imsave(fn, img, check_contrast=False)
        self.required_bytes = os.path.getsize(fn)
        logging.info(f"Written {self.required_bytes} bytes in {fn}")

    def __read_fn(self, fn):
        img = io.imread(fn)
        logging.info(f"Read {fn} of shape {img.shape}")
        return img
    '''

if __name__ == "__main__":
    main.main(EC.parser, logging, CoDec)
