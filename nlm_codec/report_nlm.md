# Informe – NLM Codec

## Objetivo

Aplicar un filtro **Non-Local Means (NLM)** como post-procesado para mejorar la calidad visual de imágenes reconstruidas.

## Flujo

1. **Encoder**
   - Carga imágenes desde `frames/`
   - Las guarda en `.npz` sin compresión (simulación)

2. **Decoder**
   - Carga `.npz`
   - Aplica NLM:
     - patch_size = 5
     - patch_distance = 6
     - h_factor = 0.8
   - Guarda imágenes filtradas en `results/`

3. **Test**
   - Calcula PSNR entre original y filtrada

## Resultados

- NLM reduce ruido
- Suaviza artefactos
- Mantiene bordes razonablemente bien

## Conclusión

El NLM Codec funciona como un filtro de post-procesado independiente, útil para mejorar la calidad visual de imágenes reconstruidas por otros códecs.