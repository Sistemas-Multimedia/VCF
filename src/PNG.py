'''Entropy Encoding of images using PNG (Portable Network Graphics). '''

import io
from skimage import io as skimage_io # pip install scikit-image
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

# Encoder parser
parser.parser_encode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {EIC.ENCODE_INPUT})", default=EIC.ENCODE_INPUT)
parser.parser_encode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {EIC.ENCODE_OUTPUT})", default=f"{EIC.ENCODE_OUTPUT}")

# Decoder parser
parser.parser_decode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {EIC.DECODE_INPUT})", default=f"{EIC.DECODE_INPUT}")
parser.parser_decode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {EIC.DECODE_OUTPUT})", default=f"{EIC.DECODE_OUTPUT}")    

class CoDec(EIC.CoDec):

    def __init__(self, args):
        super().__init__(args)
        self.file_extension = ".png"

    def compress_fn(self, img, fn):
        #skimage_io.use_plugin('freeimage')
        #compressed_img = img
        logging.debug(f"Input to io.BytesIO() witn range [{np.min(img)}, {np.max(img)}]")
        assert (img.dtype == np.uint8) or (img.dtype == np.uint16), f"current type = {img.dtype}"
        #assert (img.dtype == np.uint8), f"current type = {img.dtype}"
        compressed_img = io.BytesIO()
        #skimage_io.imsave(fname=compressed_img, arr=img[:,:,0], plugin="imageio", check_contrast=False)
        #skimage_io.imsave(fname=compressed_img, arr=img[:,:,0], plugin="pil", check_contrast=False)
        skimage_io.imsave(fname=compressed_img, arr=img, plugin="pil", check_contrast=False)
        #skimage_io.imsave(fname=compressed_img, arr=img, plugin="freeimage")
        #img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        #cv.imwrite(compressed_img, img, [cv.IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL])
        #with open(compressed_img, "wb") as f:
        #    writer = png.Writer(width=img.shape[1], height=img.shape[0],
        #                        bitdeph=16, greyscale=False)
            # Convert z to the Python list of lists expected by
            # the png writer.
        #    z2list = z.reshape(-1, z.shape[1]*z.shape[2]).tolist()
        #    writer.write(f, z2list)
        return compressed_img

    # pip install imageio-freeimage (not necessary now)
    def decompress_fn(self, compressed_img, fn):
        compressed_img = io.BytesIO(compressed_img)
        #img = cv.imread(compressed_img, cv.IMREAD_UNCHANGED)
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = skimage_io.imread(fname=compressed_img)
        logging.debug(f"Output from skimage_io.imread() witn range [{np.min(img)}, {np.max(img)}]")
        logging.debug(f"img.dtype={img.dtype}")
        return img
    
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
