'''Entropy Encoding of images using PNG (Portable Network Graphics). '''
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

# Default IO images
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ENCODE_OUTPUT = "/tmp/encoded" # The file extension is decided in run-time
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.png"

#_parser, parser_encode, parser_decode = parser.create_parser(description=__doc__)

# Encoder parser
parser.parser_encode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser.parser_encode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")

# Decoder parser
parser.parser_decode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser.parser_decode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output image (default: {DECODE_OUTPUT})", default=f"{DECODE_OUTPUT}")    

#parser.parser.parse_known_args()

COMPRESSION_LEVEL = 9

class CoDec(EIC.CoDec):

    def __init__(self, args):
        super().__init__(args)
        self.file_extension = ".TIFF"

    # pip install imageio-freeimage (not necessary now)
    def compress(self, img):
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

    def decompress(self, compressed_img):
        compressed_img = pyio.BytesIO(compressed_img)
        #img = cv.imread(compressed_img, cv.IMREAD_UNCHANGED)
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = tifffile.imread(compressed_img)
        logging.debug(f"img.dtype={img.dtype}")
        return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
