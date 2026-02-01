'''III... coding: runs a 2D image codec to a sequence of images.'''

import io
import os
from skimage import io as skimage_io # pip install scikit-image
import main
import logging
import numpy as np
import cv2 as cv # pip install opencv-python
import platform_utils as pu
pu.ensure_description_file(__doc__)  # Crear description.txt antes de importar parser
import parser
import entropy_video_coding as EVC
from entropy_video_coding import Video
import av  # pip install av
from PIL import Image

# Default IOs (multiplataforma)
ENCODE_INPUT = pu.ENCODE_INPUT
ENCODE_OUTPUT_PREFIX = pu.ENCODE_OUTPUT_PREFIX
DECODE_INPUT_PREFIX = ENCODE_OUTPUT_PREFIX
DECODE_OUTPUT = pu.DECODE_OUTPUT

N_FRAMES = 16

# Encoder parser
parser.parser_encode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input video (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser.parser_encode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Prefix of the output sequence of frames (default: {ENCODE_OUTPUT_PREFIX})", default=f"{ENCODE_OUTPUT_PREFIX}")
parser.parser_encode.add_argument("-n", "--number_of_frames", type=parser.int_or_str, help=f"Number of frames to encode (default: {N_FRAMES})", default=f"{N_FRAMES}")

# Decoder parser
parser.parser_decode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Prefix of the input sequence of frames (default: {DECODE_INPUT_PREFIX})", default=f"{DECODE_INPUT_PREFIX}")
parser.parser_decode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output video (default: {DECODE_OUTPUT})", default=f"{DECODE_OUTPUT}")    
parser.parser_decode.add_argument("-n", "--number_of_frames", type=parser.int_or_str, help=f"Number of frames to decode (default: {N_FRAMES})", default=f"{N_FRAMES}")

#parser.parser.parse_known_args()

#COMPRESSION_LEVEL = 9

class CoDec(EVC.CoDec):

    def __init__(self, args):
        super().__init__(args)

    def compress(self):
        '''Input a H.264 AVI-file and output a sequence of PNG frames.'''
        fn = self.args.input
        logging.info(f"Encoding {fn}")
        container = av.open(fn)
        img_counter = 0
        for packet in container.demux():
            if __debug__:
                self.input_bytes += packet.size
            for frame in packet.decode():
                img = frame.to_image()
                #img_fn = f"{ENCODE_OUTPUT_PREFIX}_%04d.png" % frame.index
                #img_fn = f"{self.args.output}_%04d.png" % frame.index
                img_fn = f"{self.args.output}_%04d.png" % img_counter
                img_counter += 1
                #print(img_fn)
                img.save(img_fn)
                if __debug__:
                    O_bytes = os.path.getsize(img_fn)
                    self.output_bytes += O_bytes
                    logging.info(f"Extracting frame {img_fn} {img.size} {img.mode} in={packet.size} out={O_bytes}")
                else:
                    logging.info(f"Extracting frame {img_fn} {img.size} {img.mode} in={packet.size}")
        self.N_frames = img_counter
        self.width, self.height = img.size
        self.N_channels = len(img.mode)

    def _compress(self, fn):
        
        logging.info(f"Encoding {fn}")
        container = av.open(fn)
        img_counter = 0
        for frame in container.decode(video=0):
            img = frame.to_image()
            #print(type(frame))
            img_fn = f"{ENCODE_OUTPUT_PREFIX}_%04d.png" % frame.index
            img_counter += 1
            #print(img_fn)
            img.save(img_fn)
            if __debug__:
                I_bytes = len(frame.to_bytes())
                O_bytes = os.path.getsize(img_fn)
                self.output_bytes += O_bytes
                self.input_bytes += I_bytes
                logging.info(f"{img_fn} {img.size} {img.mode} {I_bytes} {O_bytes}")
            else:
                logging.info(f"{img_fn} {img.size} {img.mode}")
            # cv2.imwrite(img_fn, img)
        #compressed_vid = Video(img_counter, *vid.get_shape()[1:], ENCODE_OUTPUT_PREFIX)
        self.N_frames = img_counter + 1
        self.width, self.height = img.size
        self.N_channels = len(img.mode)
        #return compressed_vid

    def decompress(self):
        '''Input a sequence of PNG images and output a H.264 AVI-file with lossless encoding.'''
        temp_dir = pu.get_vcf_temp_dir()
        imgs = sorted(os.path.join(temp_dir, file)
            for file in os.listdir(temp_dir)
                if file.lower().startswith("encoded".lower()) and file.lower().endswith(".png".lower()))
        
        #imgs = [i for i in os.listdir(self.args.input) if i.lower().endswith('.png')]

        # Open the output file container
        container = av.open(self.args.output, 'w', format='avi')
        video_stream = container.add_stream('libx264', rate=self.framerate)

        # Set lossless encoding options
        #video_stream.options = {'crf': '0', 'preset': 'veryslow'}
        video_stream.options = {'crf': '0', 'preset': 'ultrafast'}

        # Optionally set pixel format to ensure no color space conversion happens
        video_stream.pix_fmt = 'yuv444p'  # Working but lossy because the YCrCb is floating point-based
        #video_stream.pix_fmt = 'rgb24'  # Not work
    
        #img_0 = Image.open("/tmp/encoded_0000.png").convert('RGB')
        img_0 = Image.open(imgs[0]).convert('RGB')
        width, height = img_0.size
        video_stream.width = width
        video_stream.height = height
        self.width, self.height = img_0.size
        self.N_channels = len(img_0.mode)

        img_counter = 0
        print(imgs)
        for i in imgs:
            img = Image.open(i).convert('RGB')
            logging.info(f"Decoding frame {img_counter} into {self.args.output}")

            # Convert the image to a VideoFrame
            frame = av.VideoFrame.from_image(img)

            # Encode the frame and write it to the container
            packet = video_stream.encode(frame)
            container.mux(packet)
            img_counter += 1

        # Ensure all frames are written
        container.mux(video_stream.encode())
        container.close()
        self.N_frames = img_counter
        #vid = compressed_vid
        #vid.prefix = DECODE_OUTPUT
        #return vid

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
