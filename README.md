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
      python III.py encode
      ffplay /tmp/encoded_%04d.tif
      python III.py decode
      ffplay /tmp/decoded_%04d.png

## Codecs organization

	+---------------------+        +----+
	| temporal transforms |    III |-T,N|, [IPP] (9), [IBP] (10), [MCTF] (10).
	+---------------------+--+     +---++-------+
	| spatial transforms  |-T| 2D-DCT* |-B,p,L,x|, 2D-DWT, [LBT] (10), no_spatial_transform.
	+---------------------+--+         +--------+
	|  color transforms   |-t| YCoCg*, YCrCb, color-DCT, no_color_transform.
	+---------------------+--+           +--+           +------+     +----+           +--+
	|     quantizers      |-a| deadzone* |-q|, LloydMax |-q,m,n|, VQ |-q,b|, color-VQ |-q|.
	+---------------------+--+           +--+           ++--+--+     +----+           +--+
	|  decoding filters   |-f| no_filter*, gaussian_blur |-s|, [NLM] (1), [BM3D] (3)
	+---------------------+--+                           +--+
	|   entropy codecs    |-c| TIFF*, PNG, Huffman, PNM, [adaptive_Huffman] (4), [arith] (4), [adaptive_arith] (5).
	+---------------------+--+

	...* = default option
	[...] = to be implemented
	(.) = points for the evaluation of the subject

## Decoding Filters

Decoding filters are applied during the decompression phase to improve image quality by removing noise or artifacts.

### Non-Local Means (NLM)
The NLM filter reduces noise by averaging pixel values based on the similarity of their surrounding neighborhoods, rather than just their local proximity.

**Usage:**
```bash
python NLM.py decode -i <input_encoded_file> -o <output_image> --h 10 --template_window_size 7 --search_window_size 21
```
* `--h`: Filter strength (higher values remove more noise but may blur details).
* `--template_window_size`: Size of the patch to compare.
* `--search_window_size`: Size of the area to search for similar patches.

### Block-Matching and 3D Filtering (BM3D)
An advanced denoising algorithm that groups similar 2D image patches into 3D stacks and applies filtering in the transform domain. It supports grayscale, color, and multichannel images, as well as deblurring.

**Usage:**
```bash
python BM3D.py decode -i <input_encoded_file> -o <output_image> --sigma_bm3d 25.0 --profile_bm3d np
```
* `--sigma_bm3d`: Noise standard deviation (fuerza del filtro).
* `--profile_bm3d`: Complexity profile (`lc`, `np`, `high`, `vn`).
* `--psd_bm3d`: (Optional) Path to a `.npy` file with the noise Power Spectral Density.
* `--psf_bm3d`: (Optional) Path to a `.npy` file with the Point Spread Function for deblurring.



