'''
Implementación del filtro de eliminación de ruido Non-Local Means (NLM).
Este filtro se utiliza en la etapa de decodificación para reducir artefactos 
de compresión promediando parches similares en la imagen.
'''

import numpy as np
import logging
import parser
import main
import cv2
import no_filter

# Metadatos para el framework de la asignatura
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)

# Configuración de parámetros por defecto según estándares de OpenCV
VALOR_H_POR_DEFECTO = 3      # Fuerza del filtro 
VENTANA_BUSQUEDA = 21        # Tamaño del área donde se buscan parches similares
VENTANA_PARCHE = 7           # Tamaño del bloque comparativo 

# Añadir argumentos específicos al parser del decodificador
# Se utiliza el flag -s según lo indicado en la documentación de la asignatura
parser.parser_decode.add_argument(
    "-s", "--filter_size", 
    type=parser.int_or_str, 
    help=f"Fuerza del filtro NLM 'h' (por defecto: {VALOR_H_POR_DEFECTO})", 
    default=VALOR_H_POR_DEFECTO
)

class CoDec(no_filter.CoDec):
    """
    Clase CoDec que integra el algoritmo Non-Local Means como filtro 
    de post-procesamiento durante la decodificación.
    """

    def __init__(self, args):
        """
        Constructor de la clase.
        Args:
            args: Argumentos de línea de comandos que contienen la configuración del filtro.
        """
        logging.debug(f"Inicializando filtro NLM con argumentos={args}")
        super().__init__(args)
        self.args = args

    def decode(self):
        """
        Flujo de decodificación estándar:
        1. Lectura de datos comprimidos.
        2. Descompresión al dominio espacial.
        3. Aplicación del filtro NLM para mejorar la calidad visual percibida.
        4. Escritura del resultado final.
        """
        datos_comprimidos = self.decode_read()
        imagen_reconstruida = self.decompress(datos_comprimidos)
        
        logging.debug(f"Dimensiones de imagen reconstruida: {imagen_reconstruida.shape}")
        
        # Aplicación de la etapa de filtrado
        imagen_filtrada = self.filter(imagen_reconstruida)
        
        tamano_salida = self.decode_write(imagen_filtrada)
        return tamano_salida
            
    def filter(self, img):
        """
        Aplica el algoritmo Non-Local Means Denoising.
        
        A diferencia del desenfoque gaussiano, NLM preserva mejor las texturas y 
        bordes ya que no promedia píxeles vecinos espaciales, sino parches con 
        estructuras similares en toda la ventana de búsqueda.
        
        Referencia: Buades, A. et al. "A non-local algorithm for image denoising."
        """
        try:
            # El parámetro 'h' controla la agresividad del filtro.
            # Valores altos eliminan más ruido pero pueden difuminar detalles finos.
            fuerza_h = int(self.args.filter_size)
            
            logging.info(f"Aplicando NLM: h={fuerza_h}, parche={VENTANA_PARCHE}, ventana={VENTANA_BUSQUEDA}")

            # Robustez: OpenCV requiere que la imagen sea de tipo uint8 (8 bits)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Verificación de canales (Color vs Blanco y Negro)
            if len(img.shape) == 3:
                # Versión optimizada para imágenes en color
                return cv2.fastNlMeansDenoisingColored(
                    img, 
                    None, 
                    h=fuerza_h, 
                    hColor=fuerza_h, 
                    templateWindowSize=VENTANA_PARCHE, 
                    searchWindowSize=VENTANA_BUSQUEDA
                )
            else:
                # Versión para imágenes en escala de grises
                return cv2.fastNlMeansDenoising(
                    img, 
                    None, 
                    h=fuerza_h, 
                    templateWindowSize=VENTANA_PARCHE, 
                    searchWindowSize=VENTANA_BUSQUEDA
                )
                
        except Exception as e:
            # Manejo de errores para evitar que el programa se detenga si falla el filtrado
            logging.error(f"Error durante el filtrado NLM: {e}. Se devuelve imagen original.")
            return img

if __name__ == "__main__":
    # Integración con el flujo de ejecución principal de la asignatura
    main.main(parser.parser, logging, CoDec)