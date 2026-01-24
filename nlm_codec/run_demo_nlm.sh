#!/bin/bash
cd "$(dirname "$0")"

echo "=== Ejecutando NLM Codec ==="

# ACTIVAR ENTORNO VIRTUAL
source codec_env/bin/activate

cd src

echo "[1/3] Ejecutando encoder_nlm.py"
python encoder_nlm.py

echo "[2/3] Ejecutando decoder_nlm.py"
python decoder_nlm.py

echo "[3/3] Ejecutando test_nlm.py"
python test_nlm.py

echo "Demo NLM completada. Revisa nlm_codec/results/"