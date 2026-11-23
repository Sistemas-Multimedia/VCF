# Visual Coding Framework
A programming environment to test and develop image and video compression algorithms.

## Install and configuration

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

## Codecs organization and [parameters]

| Stage | Flag | Options (`*` = default) |
| :--- | :---: | :--- |
| **temporal transforms** | | III `[T,N]`, `[IPP]`, `[IBP]`, `[MCTF]` |
| **spatial transforms** | `-T` | 2D-DCT* `[B,p,L,x]`, 2D-DWT `[l,w]`, 2D-KLT `[B,x]`, 2D-MDCT `[B,x]`, `[LBT]`, no_spatial_transform |
| **color transforms** | `-t` | YCoCg*, YCrCb, color-DCT, no_color_transform |
| **quantizers** | `-a` | deadzone* `[q]`, LloydMax `[q,m,n]`, VQ `[q,b]`, color-VQ `[q]` |
| **decoding filters** | `-f` | no_filter*, gaussian_blur `[s]`, NLM `[H,t,s]`, `[BM3D]` `[s]` |
| **entropy codecs** | `-c` | [TIFF](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/TIFF.ipynb)*, [PNG](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/PNG.ipynb), [Huffman](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/Huffman.ipynb), [PNM](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/PNM.ipynb), CBAHC `[o]`, [CBAAC](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/PNM.ipynb) `[o]` |
