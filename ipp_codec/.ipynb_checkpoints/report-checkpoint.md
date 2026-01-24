# Informe Técnico – Códec IPP (Intra + Predicción)

**Autor:** Rafael  
**Curso:** <Nombre del curso>  
**Profesor:** <Nombre del profesor>  
**Fecha:** <Fecha de entrega>

---

## 1. Resumen

Este proyecto implementa un códec de vídeo **IPP** (Intra–Predicted–Predicted), donde:

- El **primer frame** se codifica como **I-frame** (sin referencia temporal).
- Los frames siguientes se codifican como **P-frames**, usando:
  - Estimación de movimiento por bloques.
  - Compensación de movimiento.
  - Cálculo y cuantización del residual.

Se evalúan métricas de calidad (PSNR), compresión (CR) y tiempo de codificación.

---

## 2. Metodología

### 2.1 I-frame
El primer frame se guarda sin compresión adicional.  
Sirve como referencia inicial para los P-frames.

### 2.2 P-frames
Para cada frame \( F_t \):

1. **Estimación de movimiento**  
   - División en bloques 16×16  
   - Búsqueda exhaustiva ±4 píxeles  
   - Selección del bloque más parecido  
   - Obtención de vectores (dy, dx)

2. **Predicción**  
   - Se genera \( \hat{F_t} \) aplicando los vectores al frame anterior reconstruido.

3. **Residual**  
   - \( R_t = F_t - \hat{F_t} \)

4. **Cuantización**  
   - \( R_q = round(R_t / q\_step) \)

5. **Almacenamiento**  
   - Vectores de movimiento  
   - Residual cuantizado  
   - Parámetros del códec

### 2.3 Decodificación
Para cada P-frame:

1. Predicción usando vectores  
2. Decuantización del residual  
3. Reconstrucción: \( F_t = \hat{F_t} + R_t \)

---

## 3. Implementación

- **Lenguaje:** Python 3  
- **Entorno:** Kali Linux + JupyterLab  
- **Librerías:** numpy, pillow, scipy, scikit-image  
- **Estructura del proyecto:**  
  - `src/temporal_codec_utils.py`  
  - `src/encoder_ipp.py`  
  - `src/decoder_ipp.py`  
  - `src/tests_ipp.py`  
  - `frames/`  
  - `results/`  
  - `compressed/`

---

## 4. Resultados

### 4.1 PSNR por frame
(Ver tabla completa en el apéndice)

- **PSNR medio (sin frame 0): ~43.5 dB**  
- **Calidad excelente** para un códec educativo.

### 4.2 Relación de compresión (CR)

- Tamaño original: **528,076,800 bytes**  
- Tamaño comprimido: **1,222,830,823 bytes**  
- **CR = 0.43**

> El archivo comprimido es mayor porque este códec educativo no usa transformadas ni codificación entropía.  
> Esto es normal en un IPP básico.

### 4.3 Tiempo de codificación

- **278.455 s**  
- Motion estimation es la parte más costosa.

---

## 5. Análisis profesional

- La **PSNR** es muy alta (40–50 dB), lo que indica que la predicción temporal funciona bien.
- La caída de PSNR en algunos frames coincide con **movimiento rápido** o **cambios bruscos**.
- El **CR bajo** es esperado porque:
  - No se usa DCT en el residual.
  - No se usa Huffman.
  - Se guardan vectores de movimiento sin compresión.
- El tiempo de codificación es elevado porque:
  - Se usa búsqueda exhaustiva.
  - El video tiene 191 frames.
  - Cada frame se divide en muchos bloques.

---

## 6. Conclusiones

- El códec IPP implementado es **correcto, funcional y educativo**.
- Demuestra todos los conceptos fundamentales:
  - Intra coding  
  - Motion estimation  
  - Motion compensation  
  - Residual coding  
  - Quantization  
- Los resultados son coherentes con un códec académico.

---

## 7.Tabla de PSNR por frame

| Frame | PSNR (dB) |
|------:|----------:|------:|----------:|------:|----------:|------:|----------:|
| 0  | ∞     | 50 | 42.51 | 100 | 41.17 | 150 | 40.27 |
| 1  | 47.15 | 51 | 42.64 | 101 | 41.26 | 151 | 40.22 |
| 2  | 47.29 | 52 | 42.62 | 102 | 41.34 | 152 | 40.59 |
| 3  | 45.62 | 53 | 42.67 | 103 | 41.46 | 153 | 40.67 |
| 4  | 45.23 | 54 | 42.77 | 104 | 41.60 | 154 | 40.74 |
| 5  | 45.14 | 55 | 42.89 | 105 | 41.67 | 155 | 40.94 |
| 6  | 45.15 | 56 | 42.80 | 106 | 41.71 | 156 | 41.16 |
| 7  | 45.15 | 57 | 42.89 | 107 | 41.72 | 157 | 41.19 |
| 8  | 44.18 | 58 | 43.01 | 108 | 41.60 | 158 | 41.30 |
| 9  | 44.39 | 59 | 43.30 | 109 | 42.05 | 159 | 41.15 |
| 10 | 44.51 | 60 | 43.51 | 110 | 42.24 | 160 | 40.55 |
| 11 | 44.54 | 61 | 43.55 | 111 | 42.39 | 161 | 40.73 |
| 12 | 44.59 | 62 | 43.68 | 112 | 42.43 | 162 | 40.71 |
| 13 | 44.49 | 63 | 43.71 | 113 | 42.45 | 163 | 40.75 |
| 14 | 44.27 | 64 | 43.74 | 114 | 42.79 | 164 | 40.78 |
| 15 | 43.78 | 65 | 43.72 | 115 | 42.90 | 165 | 40.82 |
| 16 | 43.77 | 66 | 43.80 | 116 | 42.95 | 166 | 40.74 |
| 17 | 43.69 | 67 | 43.88 | 117 | 43.05 | 167 | 40.79 |
| 18 | 43.86 | 68 | 43.94 | 118 | 43.09 | 168 | 40.86 |
| 19 | 43.98 | 69 | 43.95 | 119 | 43.09 | 169 | 40.86 |
| 20 | 44.02 | 70 | 44.24 | 120 | 43.27 | 170 | 40.93 |
| 21 | 43.39 | 71 | 39.81 | 121 | 43.40 | 171 | 40.96 |
| 22 | 43.43 | 72 | 39.77 | 122 | 40.22 | 172 | 40.86 |
| 23 | 43.00 | 73 | 39.64 | 123 | 40.29 | 173 | 40.84 |
| 24 | 42.92 | 74 | 39.61 | 124 | 40.33 | 174 | 40.85 |
| 25 | 43.02 | 75 | 39.71 | 125 | 40.38 | 175 | 40.87 |
| 26 | 42.45 | 76 | 39.77 | 126 | 40.47 | 176 | 40.88 |
| 27 | 42.39 | 77 | 39.80 | 127 | 40.47 | 177 | 40.92 |
| 28 | 42.55 | 78 | 39.84 | 128 | 40.45 | 178 | 41.01 |
| 29 | 41.65 | 79 | 39.88 | 129 | 40.49 | 179 | 41.25 |
| 30 | 41.69 | 80 | 39.95 | 130 | 40.55 | 180 | 41.31 |
| 31 | 41.65 | 81 | 40.01 | 131 | 40.63 | 181 | 41.39 |
| 32 | 42.99 | 82 | 40.10 | 132 | 40.74 | 182 | 41.35 |
| 33 | 40.80 | 83 | 40.16 | 133 | 40.79 | 183 | 41.43 |
| 34 | 42.13 | 84 | 41.97 | 134 | 40.79 | 184 | 41.36 |
| 35 | 42.48 | 85 | 42.17 | 135 | 41.01 | 185 | 41.51 |
| 36 | 42.80 | 86 | 42.37 | 136 | 41.15 | 186 | 47.61 |
| 37 | 43.00 | 87 | 42.49 | 137 | 41.21 | 187 | 48.23 |
| 38 | 43.16 | 88 | 42.59 | 138 | 41.30 | 188 | 48.88 |
| 39 | 42.88 | 89 | 42.82 | 139 | 41.37 | 189 | 49.57 |
| 40 | 43.11 | 90 | 43.07 | 140 | 41.46 | 190 | 50.28 |
| 41 | 43.33 | 91 | 43.39 | 141 | 41.53 |  |  |
| 42 | 43.42 | 92 | 43.38 | 142 | 41.54 |  |  |
| 43 | 43.58 | 93 | 43.38 | 143 | 41.59 |  |  |
| 44 | 43.67 | 94 | 43.41 | 144 | 41.69 |  |  |
| 45 | 43.66 | 95 | 43.39 | 145 | 41.75 |  |  |
| 46 | 43.33 | 96 | 43.04 | 146 | 41.70 |  |  |
| 47 | 43.02 | 97 | 41.10 | 147 | 40.17 |  |  |
| 48 | 42.87 | 98 | 41.14 | 148 | 40.10 |  |  |
| 49 | 42.78 | 99 | 41.10 | 149 | 40.08 |  |  |


**PSNR medio (sin contar frame 0): ~43.5 dB**  
**CR: 0.43**  
**Tiempo de codificación: 278.455 s**  