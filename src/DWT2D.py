'''Exploiting spatial (perceptual) redundancy with the 2D dyadic Discrete Wavelet Transform.'''

import argparse
from skimage import io  # pip install scikit-image
import numpy as np
import pywt
import os
import logging
import main
from color_transforms.DCT import from_RGB as DCTFromRGB
from color_transforms.DCT import to_RGB as DCT_toRGB
import PNG as EC
import YCoCg as CT  # Color Transform

import cv2

from sklearn.cluster import KMeans


#from DWT import color_dyadic_DWT as DWT
# pip install "DWT2D @ git+https://github.com/vicente-gonzalez-ruiz/DWT2D"
from DWT2D.color_dyadic_DWT import analyze as space_analyze

from DWT2D.color_dyadic_DWT import synthesize as space_synthesize

# pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import from_RGB

from color_transforms.YCoCg import to_RGB

from color_transforms.YCrCb import from_RGB as YCRCB_from_RGB
from color_transforms.YCrCb import to_RGB as YCRCB_to_RGB

from DCT2D.block_DCT import analyze_image
from DCT2D.block_DCT import synthesize_image
from scipy.fftpack import dct, idct


EC.parser.add_argument("-l", "--levels", type=EC.int_or_str,
                       help=f"Number of decomposition levels (default: 5)", default=5)
EC.parser_encode.add_argument("-w", "--wavelet", type=EC.int_or_str,
                              help=f"Wavelet name (default: \"db5\")", default="db5")


class CoDec(CT.CoDec):
    def vq_quantize(coeffs, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
            coeffs.ravel().reshape(-1, 1))
        return kmeans.cluster_centers_[kmeans.predict(coeffs)].reshape(coeffs.shape)

    def __init__(self, args):
        super().__init__(args)
        self.levels = args.levels
        logging.info(f"levels={self.levels}")
        if self.encoding:
            self.wavelet = pywt.Wavelet(args.wavelet)
            with open(f"{args.output}_wavelet_name.txt", "w") as f:
                f.write(f"{args.wavelet}")
                logging.info(f"Written {args.output}_wavelet_name.txt")
            logging.info(f"wavelet={args.wavelet} ({self.wavelet})")
        else:
            with open(f"{args.input}_wavelet_name.txt", "r") as f:
                wavelet_name = f.read()
                logging.info(
                    f"Read wavelet=\"{wavelet_name}\" from {args.input}_wavelet_name.txt")
                self.wavelet = pywt.Wavelet(wavelet_name)
            logging.info(f"wavelet={wavelet_name} ({self.wavelet})")

    def encode(self):
        img = self.read()
        img_128 = img.astype(np.int16) - 128
        CT_img = YCRCB_from_RGB(img_128)
        decom_img = space_analyze(CT_img, self.wavelet, self.levels)
        quantized_coeffs = vq_quantize(decom_img, n_clusters)
        for i in range(self.levels):
            decom_img = self.DCT2D_block_decomposition(decom_img, block_size=8)
        logging.debug(f"len(decom_img)={len(decom_img)}")
        decom_k = self.quantize_decom(decom_img)
        self.write_decom(decom_k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        decom_k = self.read_decom()
        decom_y = self.dequantize_decom(decom_k)
        for i in range(self.levels):
            decom_y = self.DCT2D_block_synthesis(decom_y, block_size=8)
        CT_y = space_synthesize(decom_y, self.wavelet, self.levels)
        y_128 = YCRCB_to_RGB(CT_y)
        y = (y_128.astype(np.int16) + 128)
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.write(y)
        rate = (self.input_bytes*8)/(y.shape[0]*y.shape[1])
        return rate

    def quantize_decom(self, decom):
        decom_k = [self.quantize(decom[0])]
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

    def DCT2D_block_decomposition(self, img, block_size=8):
        """
        Perform 2D block-based DCT decomposition of an image.
        """
        decom_img = np.zeros_like(img)
        for i in range(0, img.shape[0], block_size):
            for j in range(0, img.shape[1], block_size):
                block = img[i:i+block_size, j:j+block_size]
                dct_block = dct(dct(block, axis=0, norm='ortho'),
                                axis=1, norm='ortho')
                decom_img[i:i+block_size, j:j+block_size] = dct_block
        return decom_img

    def DCT2D_block_synthesis(self, decom_img, block_size=8):
        """
        Perform 2D block-based DCT synthesis of an image.
        """
        img = np.zeros_like(decom_img)
        for i in range(0, decom_img.shape[0], block_size):
            for j in range(0, decom_img.shape[1], block_size):
                dct_block = decom_img[i:i+block_size, j:j+block_size]
                block = idct(idct(dct_block, axis=0, norm='ortho'),
                             axis=1, norm='ortho')
                img[i:i+block_size, j:j+block_size] = block
        return img

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
        # aux_decom = [decom[0][..., 0]] # Used for computing slices
        for spatial_resolution in decom[1:]:
            subband_names = ["LH", "HL", "HH"]
            subband_index = 0
            # aux_resol = [] # Used for computing slices
            for subband_name in subband_names:
                fn = f"{fn_without_extension}_{subband_name}_{resolution_index}.png"
                self.write_fn(spatial_resolution[subband_index], fn)
                #aux_resol.append(spatial_resolution[subband_index][..., 0])
                subband_index += 1
            resolution_index -= 1

    def quantize(self, img):
        '''Quantize the image.'''
        k = self.Q.encode(img)
        logging.debug(
            f"k.shape={k.shape} k.dtype={k.dtype} max(x)={np.max(k)} min(k)={np.min(k)}")
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

if __name__ == "__main__":
    main.main(EC.parser, logging, CoDec)