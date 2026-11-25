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
| **spatial transforms** | `-T` | [2D-DCT*](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/2D-DCT.ipynb) `[B,p,L,x]`, [2D-DWT](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/2D-DWT.ipynb) `[l,w]`, [2D-KLT](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/2D-KLT.ipynb) `[B,x]`, [2D-MDCT](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/2D-MDCT.ipynb) `[B,x]`, [LBT](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/2D-LBT.ipynb), [no_spatial_transform](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/no_spatial_transform.ipynb) |
| **color transforms** | `-t` | [YCoCg*](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/YCoCg.ipynb), [YCrCb](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/YCrCb.ipynb), [color-DCT](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/color-DCT.ipynb), [no_color_transform](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/no_color_transform.ipynb) |
| **quantizers** | `-a` | [deadzone*]((https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/deadzone.ipynb)) `[q]`, [LloydMax]((https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/LloydMax.ipynb)) `[q,m,n]`, [VQ](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/VQ.ipynb) `[q,b]`, [color-VQ](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/color-VQ.ipynb) `[q]` |
| **decoding filters** | `-f` | [no_filter*](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/no_filter.ipynb), [gaussian_blur](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/gaussan_blur.ipynb) `[s]`, [NLM](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/NLM.ipynb) `[H,t,s]`, [BM3D](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/BM3D.ipynb) `[s]` |
| **entropy codecs** | `-c` | [zlib](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/zlib.ipynb), [TIFF](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/TIFF.ipynb)*, [PNG](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/PNG.ipynb), [Huffman](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/Huffman.ipynb), [PNM](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/PNM.ipynb), [CBAHC](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/CBAHC.ipynb) `[--order]`, [CBAAC](https://github.com/Sistemas-Multimedia/VCF/blob/main/notebooks/CBAAC.ipynb) `[--order]` |
