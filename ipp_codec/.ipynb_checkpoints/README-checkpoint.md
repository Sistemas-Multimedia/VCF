**Autor:** Rafael de Jesus Bautista Hernandez 
**Curso:** Sistemas Multimedia 
**Profesor:**   
**Fecha:** 

# Informe del Códec IPP (1 I + 15 P)

## 1. Objetivo

Implementar un códec temporal IPP con:

- GOP = 16 frames (1 I + 15 P)
- Predicción temporal con motion vectors
- Residuales cuantizados
- Reconstrucción idéntica entre encoder y decoder
- Métricas PSNR y CR

## 2. Flujo del Encoder

1. Se divide el video en GOPs de 16 frames.
2. El primer frame del GOP se toma como I-frame.
3. El encoder simula la decodificación del I-frame
4. Para cada P-frame:
   - Se genera predicción:
     - Con MV: block matching 16×16, búsqueda ±2.
     - Sin MV: copia del frame anterior reconstruido.
   - Se calcula residual.
   - Se cuantiza residual.
   - Se simula decodificación para actualizar la referencia.
5. Se guarda un archivo `gop_XXX.npz` con:
   - I-frame
   - I-frame reconstruida simulada
   - Motion vectors
   - Residuales cuantizados
   - q_step
   - Nombres de frames

## 3. Flujo del Decoder

1. Carga cada `gop_XXX.npz`.
2. Reconstruye I-frame.
3. Para cada P-frame:
   - Aplica motion vectors (si existen).
   - Decuantiza residual.
   - Reconstruye frame.
4. Guarda frames reconstruidos en `results/`.

## 4. Métricas

`test_ipp.py` calcula:

- PSNR por frame
- PSNR medio
- CR (tamaño original / tamaño comprimido)

## 5. Resultados

- PSNR típico: 40–50 dB
- CR depende de q_step y uso de MV
- Reconstrucción idéntica entre encoder y decoder


## 6. Estructura del proyecto


- `src/`  
  - `temporal_codec_utils.py` → Funciones de procesamiento de vídeo
  - `encoder_ipp.py` → Codificador IPP
  - `decoder_ipp.py` → Decodificador IPP 
  - `tests_ipp.py` → Ejecuta el pipeline completo y calcula métricas.
- `frames/` → Frames de entrada (extraídos de `video.mp4`).
- `compressed/` → Archivos comprimidos `.npz` generados por el encoder.
- `results/` → Frames GOP reconstruidos por el decoder.
- `requirements.txt` → Dependencias del proyecto.
- `README.md` → Instrucciones de uso.
- `report.md` → Informe técnico.
- `run_demo.sh` → Script para ejecutar una demo completa.


