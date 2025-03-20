'''Entropy Encoding of images using TIFF (Tag Image File Format). '''
import tifffile
import io as pyio  # Avoid conflict with skimage.io
import main
import logging
import numpy as np
import cv2 as cv # pip install opencv-python
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_image_coding as EIC

# Encoder parser
parser.parser_encode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {EIC.ENCODE_INPUT})", default=EIC.ENCODE_INPUT)
parser.parser_encode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {EIC.ENCODE_OUTPUT})", default=f"{EIC.ENCODE_OUTPUT}")

# Decoder parser
parser.parser_decode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {EIC.DECODE_INPUT})", default=f"{EIC.DECODE_INPUT}")
parser.parser_decode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {EIC.DECODE_OUTPUT})", default=f"{EIC.DECODE_OUTPUT}")    

#parser.parser.parse_known_args()

COMPRESSION_LEVEL = 9

class CoDec(EIC.CoDec):

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        self.file_extension = ".TIFF"

    # pip install imageio-freeimage (not necessary now)
    def compress_fn(self, img, fn):
        logging.debug("trace")
        #skimage_io.use_plugin('freeimage')
        #compressed_img = img
        logging.debug(f"img.dtype={img.dtype}")
        assert (img.dtype == np.uint8) or (img.dtype == np.uint16), f"current type = {img.dtype}"
        #assert (img.dtype == np.uint8), f"current type = {img.dtype}"
        compressed_img = pyio.BytesIO()
        tifffile.imwrite(compressed_img, data=img, compression='zlib')
        compressed_img.seek(0)
        #skimage_io.imsave(fname=compressed_img, arr=img)
        #skimage_io.imsave(fname=compressed_img, arr=img[:,:,0], plugin="imageio", check_contrast=False)
        #skimage_io.imsave(fname=compressed_img, arr=img[:,:,0], plugin="pil", check_contrast=False)
        #skimage_io.imsave(fname=compressed_img, arr=img, plugin="pil", check_contrast=False)
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

    def decompress_fn(self, compressed_img, fn):
        logging.debug("trace")
        compressed_img = pyio.BytesIO(compressed_img)
        #img = cv.imread(compressed_img, cv.IMREAD_UNCHANGED)
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = tifffile.imread(compressed_img)
        logging.debug(f"img.dtype={img.dtype}")
        return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
