'''Eliminación de ruido en imágenes utilizando el filtro Non-Local Means (NLM). 
*** ¡Solo es efectivo durante la decodificación! ***'''

import numpy as np
import logging
import tempfile
import builtins
import os

# Create a valid temporary file path
temp_desc_path = os.path.join(tempfile.gettempdir(), "description.txt")

# Write the description to the valid temporary file
with open(temp_desc_path, 'w', encoding='utf-8') as f:
    f.write(__doc__)

# Monkeypatch open to redirect /tmp/description.txt to our valid temp file
# This is necessary because parser.py (which we cannot edit) hardcodes /tmp/description.txt
_original_open = builtins.open

def _redirect_open(file, *args, **kwargs):
    if file == "/tmp/description.txt":
        return _original_open(temp_desc_path, *args, **kwargs)
    return _original_open(file, *args, **kwargs)

builtins.open = _redirect_open

try:
    import parser
finally:
    # Restore original open
    builtins.open = _original_open
import main
import importlib
import cv2

default_h = 10
default_template_window_size = 7
default_search_window_size = 21

# Configuración del parser para la decodificación - Parámetros de NLM
# h: Determina la fuerza del filtro. Valores más altos eliminan más ruido pero pueden difuminar bordes.
# template_window_size: Tamaño del bloque que se usa para comparar similitudes.
# search_window_size: Tamaño del área donde se buscan bloques similares.
parser.parser_decode.add_argument("-h_nlm", "--h", type=float, help=f"Fuerza del filtro. Un h mayor elimina más ruido pero también detalles (por defecto: {default_h})", default=default_h)
parser.parser_decode.add_argument("-t", "--template_window_size", type=int, help=f"Tamaño del parche de plantilla (debe ser impar, por defecto: {default_template_window_size})", default=default_template_window_size)
parser.parser_decode.add_argument("-s", "--search_window_size", type=int, help=f"Tamaño del área de búsqueda (debe ser impar, por defecto: {default_search_window_size})", default=default_search_window_size)

import no_filter

args = parser.parser.parse_known_args()[0]

class CoDec(no_filter.CoDec):
    """
    Codec de NLM (Non-Local Means) para la eliminación de ruido.
    Hereda de no_filter.CoDec para manejar el flujo estándar de descompresión.
    """

    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        self.args = args

    def encode(self):
        """
        Método de codificación con corrección de argumentos.
        Necesario porque la clase base usa defaults incorrectos para este entorno.
        """
        img = self.encode_read(self.args.original)
        compressed_img = self.compress(img)
        output_size = self.encode_write(compressed_img, self.args.encoded)
        return output_size

    def decode(self):
        """
        Método de decodificación que aplica el filtro después de descomprimir los datos.
        """
        compressed_k = self.decode_read(self.args.encoded)
        k = self.decompress(compressed_k)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")        
        y = self.filter(k)
        output_size = self.decode_write(y, self.args.decoded)
        return output_size
            
    def filter(self, img):
        """
        Aplica el algoritmo Non-Local Means de OpenCV.
        Detecta automáticamente si la imagen es en color o escala de grises para elegir la función adecuada.
        """
        logging.debug(f"trace y={img}")
        logging.info(f"NLM filter strength (h)={self.args.h}")
        logging.info(f"NLM template window size={self.args.template_window_size}")
        logging.info(f"NLM search window size={self.args.search_window_size}")
        
        if self.args.h < 0:
            logging.warning(f"Valor de h inválido ({self.args.h}). Debe ser no negativo. Ajustando a 0.")
            self.args.h = 0
            
        if self.args.template_window_size % 2 == 0 or self.args.template_window_size <= 0:
            logging.warning(f"Tamaño de ventana de plantilla inválido ({self.args.template_window_size}). Debe ser impar y positivo. Ajustando a {default_template_window_size}.")
            self.args.template_window_size = default_template_window_size

        if self.args.search_window_size % 2 == 0 or self.args.search_window_size <= 0:
            logging.warning(f"Tamaño de ventana de búsqueda inválido ({self.args.search_window_size}). Debe ser impar y positivo. Ajustando a {default_search_window_size}.")
            self.args.search_window_size = default_search_window_size

        try:
            # Verificar si la imagen es en color o escala de grises
            if len(img.shape) == 3:
                # Imagen en color - usar fastNlMeansDenoisingColored
                logging.info("Aplicando denoising NLM a imagen en color")
                return cv2.fastNlMeansDenoisingColored(
                    img, 
                    None, 
                    self.args.h, 
                    self.args.h,  # h para los componentes de color
                    self.args.template_window_size, 
                    self.args.search_window_size
                )
            else:
                # Imagen en escala de grises - usar fastNlMeansDenoising
                logging.info("Aplicando denoising NLM a imagen en escala de grises")
                return cv2.fastNlMeansDenoising(
                    img, 
                    None, 
                    self.args.h, 
                    self.args.template_window_size, 
                    self.args.search_window_size
                )
        except Exception as e:
            logging.error(f"Error durante el procesamiento NLM: {e}")
            logging.warning("Devolviendo imagen original debido al error.")
            return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
