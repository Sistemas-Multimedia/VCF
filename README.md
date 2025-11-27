# Visual Coding Framework
A programming environment to develop and test image and video compression algorithms.

## Install and configuration

Supposing that a Python interpreter and Git are available:

      python -m venv ~/envs/VCF
      git clone git@github.com:Sistemas-Multimedia/VCF.git
      cd VCF
      source ~/envs/VCF/bin/activate
      pip install -r requirements

## Usage

### Image coding (example)

      cd src
      python PNG.py encode
      display /tmp/encoded.png
      python PNG.py decode
      display /tmp/decoded.png

### Video coding (example)

      cd src
      python MPNG.py encode
      ffplay /tmp/encoded_%04d.png
      python MPNG.py decode
      mplayer /tmp/decoded.avi
   
## Programming

Typically, you will need to develop a new encoding scheme for image or
video.

### Codecs organization

	+---------------------+
	| temporal transforms | III, [IPP], [IBP], [MCTF], [MC-DWT].
	+---------------------+
	| spatial transforms  | 2D-DCT*, 2D-DWT, [CAE].
	+---------------------+
	|  color transforms   | YCoCg*, YCrCb, RGB2RGB, color-DCT.
	+---------------------+--+           +--+           +--+     +-----+
	|     quantizers      |-a| deadzone* |-q|, LloydMax |-q|, VQ |-b,-n|, color-VQ (mover arriba).
	+---------------------+--+           +--+           ++-++    +-----+
	|  decoding filters   |-f| no_filter*, gaussian_blur |-s| , [NLM], [BM3D]
	+---------------------+--+                           +--+
	|   entropy codecs    |-c| TIFF*, PNG, Huffman, PNM, [adaptive_Huffman], [arith], [adaptive_arith].
	+---------------------+--+

	...* = default option
	[...] = to be implemented

### Image Coding

The simplest solution is to implement the methods `compressed_img =
compress(img)` and `img = decompress(compressed_img)`, defined in the
`entropy_image_coding`
[class interface](https://realpython.com/python-interface/). Notice
that it is not necessary to read `img` when encoding, nor write
`compressed_img` when
decoding. Example:
[`src/PNM.py`](https://github.com/Sistemas-Multimedia/VCF/blob/main/src/PNM.py).

### Video Coding

Again, it is necessary to implement the methods `None = compress()`
and `None = decompress()`, defined in the `entropy_video_coding` class
interface. In this case, because a video usually does not fit in
memory, you must read and write the frames in the methods `compress()`
and
`decompress()`. Example
[`src/MPNG.py`](https://github.com/Sistemas-Multimedia/VCF/blob/main/src/MPNG.py).
