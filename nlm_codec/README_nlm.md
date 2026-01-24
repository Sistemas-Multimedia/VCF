# NLM Codec (post-filter)

Este módulo aplica un filtro Non-Local Means (NLM) como post-procesado a imágenes reconstruidas.

## Uso
1. Coloca una imagen en `frames/` (por ejemplo `frame_001.png`).
2. Ejecuta:
3. Resultado en `nlm_codec/results/`.

Parámetros ajustables en `decoder_nlm.py`: `patch_size`, `patch_distance`, `h_factor`.