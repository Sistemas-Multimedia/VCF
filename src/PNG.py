'''Entropy Encoding of images using PNG (Portable Network Graphics). '''

import io
from skimage import io as skimage_io # pip install scikit-image
import imageio.v3 as iio
import logging
import numpy as np
import cv2 as cv # pip install opencv-python
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import main
import entropy_image_coding as EIC
#import png

COMPRESSION_LEVEL = 9
class CoDec(EIC.CoDec):
    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        self.file_extension = ".png"

    def compress_fn(self, img, fn):
        logging.debug(f"trace img={img}")
        logging.debug(f"trace fn={fn}")
        logging.debug(f"Input to io.BytesIO() witn range [{np.min(img)}, {np.max(img)}]")
        assert (img.dtype == np.uint8) or (img.dtype == np.uint16), f"current type = {img.dtype}"
        compressed_img = io.BytesIO()
        # https://imageio.readthedocs.io/en/stable/examples.html#writing-to-bytes-encoding
        iio.imwrite(compressed_img, img, plugin="pillow", extension=".png")
        return compressed_img

    def compress(self, img, fn=="/tmp/encoded"):
        return self.compress_fn(img, fn)

    # pip install imageio-freeimage (not necessary now)
    def decompress_fn(self, compressed_img, fn):
        logging.debug(f"trace compressed_img={compressed_img[10]}")
        logging.debug(f"trace fn={fn}")
        compressed_img = io.BytesIO(compressed_img)
        img = skimage_io.imread(fname=compressed_img)
        logging.debug(f"Output from skimage_io.imread() witn range [{np.min(img)}, {np.max(img)}]")
        logging.debug(f"img.dtype={img.dtype}")
        return img

    def decompress(self, compressed_img, fn="/tmp/encoded"):
        return self.decompress_fn(compressed_img, fn)
    
    ##########
    # UNUSED #
    ##########
    
    def UNUSED_encode_write_fn(self, img, fn):
        '''Write to disk the image <img> with filename <fn>.'''
        skimage_io.imsave(fn, img)
        self.total_output_size += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

    def UNUSED_write_fn(self, img, fn):
        '''Write to disk the image with filename <fn>.'''
        # Notice that the encoding algorithm depends on the output
        # file extension (PNG).
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(fn, img, [cv.IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL])
        #if __debug__:
        #    len_output = os.path.getsize(fn)
        #    logging.info(f"Before optipng: {len_output} bytes")
        #subprocess.run(f"optipng {fn}", shell=True, capture_output=True)
        self.total_output_size += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

    def UNUSED_write_fn(self, img, fn):
        '''Write to disk the image with filename <fn>.'''
        # Notice that the encoding algorithm depends on the output
        # file extension (PNG).
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(fn, img, [cv.IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL])

        #skimage_io.imsave(fn, img, check_contrast=False)
        #image = Image.fromarray(img.astype('uint8'), 'RGB')
        #image.save(fn)
        #subprocess.run(f"optipng -nc {fn}", shell=True, capture_output=True)
        subprocess.run(f"pngcrush {fn} /tmp/pngcrush.png", shell=True, capture_output=True)
        subprocess.run(f"mv -f /tmp/pngcrush.png {fn}", shell=True, capture_output=True)
        # Notice that pngcrush is not installed, these two previous steps do not make any effect!
        self.total_output_size += os.path.getsize(fn)
        logging.info(f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

    def UNUSED_decode(self):
        '''Read an image and save it in the disk. Notice that we are
        using the PNG image format for both, decode and encode an
        image. For this reason, both methods do exactly the same.
        This method is overriden in child classes.

        '''
        return self.encode()

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
