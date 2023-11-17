'''Motion PNG. Inputs a .AVI file and outputs a sequence of frames
encoded in PNG (Portable Network Graphics), and viceversa.'''

import io
import os
from skimage import io as skimage_io # pip install scikit-image
import main
import logging
import numpy as np
import cv2 as cv
import parser
import entropy_video_coding as EVC
from entropy_video_coding import Video
import av

# Default IOs
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/videos/mobile_352x288x30x420x300.avi"
ENCODE_OUTPUT_PREFIX = "/tmp/encoded"
DECODE_INPUT_PREFIX = ENCODE_OUTPUT_PREFIX
DECODE_OUTPUT = "/tmp/decoded.avi"

# Encoder parser
parser.parser_encode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input video (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser.parser_encode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output video prefix (default: {ENCODE_OUTPUT_PREFIX})", default=f"{ENCODE_OUTPUT_PREFIX}")

# Decoder parser
parser.parser_decode.add_argument("-i", "--input", type=parser.int_or_str, help=f"Input video prefix (default: {DECODE_INPUT_PREFIX})", default=f"{DECODE_INPUT_PREFIX}")
parser.parser_decode.add_argument("-o", "--output", type=parser.int_or_str, help=f"Output video (default: {DECODE_OUTPUT})", default=f"{DECODE_OUTPUT}")    

parser.parser.parse_known_args()

COMPRESSION_LEVEL = 9

class CoDec(EVC.CoDec):

    def __init__(self, args):
        super().__init__(args)

    def compress(self, fn):
        logging.info(f"Encoding {fn}")
        container = av.open(fn)
        img_counter = 0
        for packet in container.demux():
            if __debug__:
                self.input_bytes += packet.size
            for frame in packet.decode():
                img = frame.to_image()
                img_counter += 1
                img_fn = f"{ENCODE_OUTPUT_PREFIX}_%04d.png" % frame.index
                #print(img_fn)
                img.save(img_fn)
                if __debug__:
                    O_bytes = os.path.getsize(img_fn)
                    self.output_bytes += O_bytes
                    logging.info(f"{img_fn} {img.size} {img.mode} in={packet.size} out={O_bytes}")
                else:
                    logging.info(f"{img_fn} {img.size} {img.mode} in={packet.size}")
        self.N_frames = img_counter
        self.width, self.height = img.size
        self.N_channels = len(img.mode)

    def _compress(self, fn):
        '''Input a H.264 AVI-file and output a sequence of PNG frames.'''
        logging.info(f"Encoding {fn}")
        container = av.open(fn)
        img_counter = 0
        for frame in container.decode(video=0):
            img = frame.to_image()
            print(type(frame))
            img_fn = f"{ENCODE_OUTPUT_PREFIX}_%04d.png" % frame.index
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
            img_counter += 1
        #compressed_vid = Video(img_counter, *vid.get_shape()[1:], ENCODE_OUTPUT_PREFIX)
        self.N_frames = img_counter
        self.width, self.height = img.size
        self.N_channels = len(img.mode)
        #return compressed_vid

    def decompress(self, compressed_vid):
        '''Input a sequence of PNG images and output a H.264 AVI-file.'''
        imgs = [i for i in os.listdir(compressed_vid.prefix)]
        
        container = av.open(DECODE_OUTPUT, 'w', format='avi')
        video_stream = container.add_stream('libx264', rate=framerate)
        img_0 = Image.open(os.path.join(ENCODE_INPUT, "_0000.png")).convert('RGB')
        width, height = img_0.size
        video_stream.width = width
        video_stream.height = height
        for i in imgs:
            frame = av.VideoFrame.from_image(i)
            packet = video_stream.encode(frame)
            container.mux(packet)
        container.close()
        vid = compressed_vid
        vid.prefix = DECODE_OUTPUT
        return vid

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
