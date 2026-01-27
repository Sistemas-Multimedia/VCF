# Proyecto de Códecs Multimedia: IPP y NLM

**Curso:** Sistemas Multimedia  
**Autores:**  
- Rafael de Jesús Bautista Hernández  
- Saúl David Ramos Pacheco  
- Alex César Taquila Camasca  

Este proyecto implementa dos códecs de procesamiento multimedia:

1. **IPP (Intra + Predicción Temporal con Motion Vectors)**  
2. **NLM (Non-Local Means / Filtro Bilateral Aproximado)**  

Cada códec cuenta con:
- Un archivo fuente único (`IPP.py` y `NLM.py`) generado desde su respectivo notebook.
- Un notebook demostrativo (`IPP.ipynb` y `NLM.ipynb`) que explica su funcionamiento y ejecución.
- Uso de la carpeta `frames/` como entrada.
- Resultados almacenados en `compressed/` y `results/`.

La estructura del proyecto respeta la plantilla proporcionada por el profesor.

---

## Estructura del Proyecto

VCF
    /compressed
    /frames
    /notebooks
        IPP.ipynb
        NLM.ipynb
    /results
    /src
        IPP.py
        NLM.py
    /venv
    README.md
    report.md
    requirements.txt
    
## Instalación de dependencias

Ejecutar:

```bash
pip install -r requirements.txt
numpy
scipy
pillow
scikit-image
matplotlib


## Ejecucion del  Codec IPP

El archivo IPP.py se genera desde notebooks/IPP.ipynb con el %%writefile ../src/IPP.py del notebook
Codificar (frames → compressed/) - python src/IPP.py encode
Decodificar (compressed → results/) - python src/IPP.py decode
python src/IPP.py test - python src/IPP.py test


## Ejecucion del  Codec NLM
El archivo NLM.py se genera desde notebooks/NLM.ipynb con el %%writefile ../src/NLM.py
Codificar (frames → compressed/) - python src/NLM.py encode
Aplicar NLM (compressed → results/) - python src/NLM.py decode
Evaluar PSNR - python src/NLM.py test
Generar comparaciones visuales - python src/NLM.py compare


## Notebooks 
Cada notebook contiene:
- Explicación del códec
- Generacion del archivo fuente con %%writefile
- Ejecución paso a paso
- Visualización de resultados
- Comparaciones visuales

 
