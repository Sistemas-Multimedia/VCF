# Informe Técnico: Códecs IPP y NLM  
**Curso:** Sistemas Multimedia  
**Autores:**  
- Rafael de Jesús Bautista Hernández  
- Saúl David Ramos Pacheco  
- Alex César Taquila Camasca  

# 1. Introducción

Este proyecto implementa dos códecs multimedia:

1. **IPP (Intra + Predicción Temporal con Motion Vectors)**  
2. **NLM (Non-Local Means / Filtro Bilateral Aproximado)**  

Ambos códecs se integran en la estructura proporcionada por el profesor y se ejecutan mediante notebooks que generan los archivos fuente `IPP.py` y `NLM.py`.


# 2. Códec IPP
## 2.1. Descripción general

El códec IPP implementa un esquema de compresión temporal basado en:

- GOPs de 16 frames  
- Un I-frame inicial por GOP  
- P-frames predichos mediante motion estimation  
- Motion vectors por bloques  
- Residuales cuantizados  
- Reconstrucción usando la referencia reconstruida (no la original)

Este enfoque garantiza que el encoder y el decoder utilicen la misma referencia, evitando drift.

## 2.2. Flujo del códec

1. **I-frame:** se almacena sin predicción.  
2. **Motion Estimation:** búsqueda por bloques con ventana configurable.  
3. **Motion Compensation:** reconstrucción del bloque predicho.  
4. **Residual:** diferencia entre frame actual y predicción.  
5. **Cuantización:** reducción de precisión del residual.  
6. **Reconstrucción:** predicción + residual dequantizado.  
7. **Referencia reconstruida:** se usa para el siguiente P-frame.

## 2.3. Métricas

El códec calcula:

- **PSNR por frame**
- **PSNR medio**
- **Compression Ratio (CR)**

# 3. Códec NLM
## 3.1. Descripción general

El códec NLM implementa un filtro espacial basado en una aproximación del algoritmo Non-Local Means mediante un filtro bilateral:

- Kernel espacial gaussiano  
- Kernel de rango basado en diferencias de intensidad  
- Procesamiento por canal RGB  

El objetivo es reducir ruido preservando bordes

## 3.2. Flujo del códec

1. **Encoder:** almacena cada imagen como `.npz` (compresión simulada).  
2. **Decoder:** aplica el filtro NLM y guarda el resultado como `_nlm.png`.  
3. **Test:** calcula PSNR entre original y filtrada.  
4. **Comparación:** genera imágenes lado a lado para evaluación visual.



# 4. Ejecución

## 4.1. IPP

python src/IPP.py encode
python src/IPP.py decode
python src/IPP.py test

## 4.2. NLM

python src/NLM.py encode
python src/NLM.py decode
python src/NLM.py test
python src/NLM.py compare