'''III... coding: runs a 2D image codec to a sequence of images.'''

import sys
import io
import os
from skimage import io as skimage_io # pip install scikit-image
import main
import logging
import numpy as np
import cv2 as cv # pip install opencv-python
with open("/tmp/description.txt", 'w') as f:  # Used by parser.py
    f.write(__doc__)
import parser
import entropy_video_coding as EVC
from entropy_video_coding import Video
import av  # pip install av
from PIL import Image
import importlib
import re

# Default IOs
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/videos/mobile_352x288x30x420x300.mp4"
ENCODE_OUTPUT_PREFIX = "/tmp/encoded"
DECODE_INPUT_PREFIX = ENCODE_OUTPUT_PREFIX
DECODE_OUTPUT = "/tmp/decoded.mp4"

N_FRAMES = 1

default_transform = "DCT"  # Can be "DCT" or "DWT"

# Encoder parser
parser.parser_encode.add_argument("-T", "--transform", type=str, 
    help=f"Transform type ('DCT' or 'DWT', default: {default_transform})", 
    default=default_transform)
parser.parser_encode.add_argument("-N", "--number_of_frames", type=parser.int_or_str, help=f"Number of frames to encode (default: {N_FRAMES})", default=f"{N_FRAMES}")

# Decoder parser
parser.parser_decode.add_argument("-T", "--transform", type=str,
    help=f"Transform type ('DCT' or 'DWT', default: {default_transform})", 
    default=default_transform)
parser.parser_decode.add_argument("-N", "--number_of_frames", type=parser.int_or_str, help=f"Number of frames to decode (default: {N_FRAMES})", default=f"{N_FRAMES}")

args = parser.parser.parse_known_args()[0]

if __debug__:
    if args.debug:
        print(f"III: Importing {args.transform}")

try:
    transform = importlib.import_module(args.transform)
except ImportError as e:
    print(f"Error: Could not find {args.transform} module ({e})")
    print(f"Make sure '2D-{args.transform}.py' is in the same directory as III.py")
    sys.exit(1)

def is_valid_name(name):
        pattern = r'^encoded_\d{4}\.png$'
        return bool(re.match(pattern, name))

class CoDec(EVC.CoDec):

    def __init__(self, args):
        super().__init__(args)
        codec_args: any
        codec_args = args
        self.transform_codec = transform.CoDec(args)
        logging.info(f"Using {args.transform} transform")

    def compress(self):
        '''Input a file recognized by av (that can be also a single
        image) and output one or more files depending on the 2D image
        encoder.
        '''
        fn = self.args.input
        logging.info(f"Encoding {fn}")
        container = av.open(fn)
        img_counter = 0
        exit = False
        for packet in container.demux():
            if __debug__:
                self.input_bytes += packet.size
            for frame in packet.decode():
                img = frame.to_image()
                #img_fn = f"{ENCODE_OUTPUT_PREFIX}_%04d.png" % img_counter
                img_fn = f"{ENCODE_OUTPUT_PREFIX}_%04d.png" % img_counter
                img_fnNOPNG = f"{ENCODE_OUTPUT_PREFIX}_%04d" % img_counter
                img.save(img_fn)
                if __debug__:
                    O_bytes = os.path.getsize(img_fn)
                    self.output_bytes += O_bytes
                    logging.info(f"Extracted frame {img_fn} {img.size} {img.mode} in={packet.size} out={O_bytes}")
                else:
                    logging.info(f"Extracted frame {img_fn} {img.size} {img.mode} in={packet.size}")
                self.transform_codec.args.input = img_fn
                self.transform_codec.args.output = img_fnNOPNG
                #self.transform_codec.encode_javi(img_array)
                #logging.info(f"Generated {}")
                self.transform_codec.encode()
                img_counter += 1
                print("--------------->", img_counter, args.number_of_frames)
                if img_counter > args.number_of_frames:
                    exit = True
                img_fn = ""
                img_fnNOPNG = ""
            if exit:
                break
        self.N_frames = img_counter
        self.width, self.height = img.size
        self.N_channels = len(img.mode)

    def no_sirve_compress(self, fn):
        logging.info(f"Encoding {fn}")
        container = av.open(fn)
        img_counter = 0
        for frame in container.decode(video=0):
            img = frame.to_image()
            #print(type(frame))
            img_fn = f"{ENCODE_OUTPUT_PREFIX}_%04d.png" % frame.index
            transform.encode(img, f"{self.args.output}_%04d.png" % img_counter)
            img_counter += 1
            #print(img_fn)
            #img.save(img_fn)
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
        '''
        for file in os.listdir("/tmp"):
            if is_valid_name(file):
                print("FILE: " + file + " " + str(len(file)))
                imgs = sorted(os.path.join("/tmp", file))
        '''
        img_fns = []
        for fn in os.listdir("/tmp/"):
            if is_valid_name(fn):
                img_fns.append(fn)
        sorted_img_fns = sorted(img_fns)

        print("------------------>", len(sorted_img_fns), sorted_img_fns)
        img_counter = 0
        #for img in imgs:
        for i in range(len(sorted_img_fns)):
            img_fn = f"{ENCODE_OUTPUT_PREFIX}_%04d.png" % img_counter
            logging.info(img_fn)
            self.transform_codec.args.input = img_fn[:-4]
            self.transform_codec.args.output= img_fn
            logging.info(f"Decoding frame {self.transform_codec.args.input} into {self.transform_codec.args.output}")
            self.transform_codec.decode()
            img_counter += 1

            #img.save(self.args.output)
        # Open the output file container
        
        self.args.output = DECODE_OUTPUT
        container = av.open(self.args.output, 'w', format='avi')
        video_stream = container.add_stream('libx264', rate=self.framerate)

        # Set lossless encoding options
        #video_stream.options = {'crf': '0', 'preset': 'veryslow'}
        video_stream.options = {'crf': '0', 'preset': 'ultrafast'}

        # Optionally set pixel format to ensure no color space conversion happens
        video_stream.pix_fmt = 'yuv444p'  # Working but lossy because the YCrCb is floating point-based
        #video_stream.pix_fmt = 'rgb24'  # Not work
        imgs = []
        for i in range(img_counter):
            #print("FILE: " + file + " " + str(len(file)))
            imgs.append(f"{ENCODE_OUTPUT_PREFIX}_%04d.png" % i)
        print(imgs)
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
