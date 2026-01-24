#!/bin/bash
cd "$(dirname "$0")"
source codec_env/bin/activate
cd src
python encoder_ipp.py
python decoder_ipp.py
python test_ipp.py
echo "Completado, revisa carpeta results/"