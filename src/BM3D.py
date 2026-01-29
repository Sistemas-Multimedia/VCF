'''Eliminación de ruido en imágenes utilizando Block-matching y filtrado 3D (BM3D).
Solo es efectivo durante la decodificación.

Este módulo implementa una interfaz de BM3D para el framework VCF, soportando tanto imágenes
en escala de grises como en color. BM3D es un potente algoritmo de denoising que utiliza
filtrado colaborativo en el dominio de la transformada 3D.
'''

import numpy as np
import logging
import os

import tempfile
import builtins

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

try:
    import bm3d
except ImportError:
    logging.error("El paquete 'bm3d' es necesario para este módulo. Instálalo con 'pip install bm3d'.")
    raise

# Parámetros por defecto de BM3D
default_sigma = 10.0
default_profile = 'np' # Perfil Normal

# Parámetros del parser para BM3D en la decodificación
# sigma_bm3d: Desviación estándar del ruido. A mayor valor, más limpieza pero más desenfoque.
# psd_bm3d: Ruta a un archivo .npy con la densidad espectral de potencia (para ruido correlacionado).
# psf_bm3d: Ruta a un archivo .npy con la función de dispersión de punto (para deblurring).
# profile_bm3d: Perfil de rendimiento: np (normal), lc (baja complejidad), high (alta calidad), vn (ruido extremo).
parser.parser_decode.add_argument("-s_bm3d", "--sigma_bm3d", type=float, 
                                 help=f"Noise standard deviation (sigma) for BM3D. Higher values remove more noise but may blur details (default: {default_sigma})", 
                                 default=default_sigma)
parser.parser_decode.add_argument("-psd_bm3d", "--psd_bm3d", type=str, 
                                 help="Path to a .npy file containing the noise Power Spectral Density (PSD) for correlated noise.", 
                                 default=None)
parser.parser_decode.add_argument("-h_bm3d", "--psf_bm3d", type=str, 
                                 help="Path to a .npy file containing the Point Spread Function (PSF) for deblurring. If provided, BM3D deblurring is performed.", 
                                 default=None)
parser.parser_decode.add_argument("-p_bm3d", "--profile_bm3d", type=str, 
                                 choices=['np', 'lc', 'high', 'vn'],
                                 help=f"BM3D profile: np (normal), lc (low complexity), high (high quality), vn (very noisy) (default: {default_profile})", 
                                 default=default_profile)

import no_filter

# Parsear argumentos para obtener las opciones elegidas
args = parser.parser.parse_known_args()[0]

class CoDec(no_filter.CoDec):
    """
    Codec basado en BM3D para la eliminación de ruido.
    Hereda de no_filter.CoDec, que gestiona el flujo de descompresión básico.
    """

    def __init__(self, args):
        logging.debug(f"Inicializando CoDec BM3D con argumentos: {args}")
        super().__init__(args)
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
        Método de decodificación sobrescrito para aplicar el filtro tras la descompresión.
        """
        # Leer los datos comprimidos
        compressed_k = self.decode_read(self.args.encoded)
        
        # Descomprimir los datos (normalmente llama a los decodificadores de transformación espacial/color)
        k = self.decompress(compressed_k)
        logging.info(f"Forma de la imagen descomprimida: {k.shape}, tipo de dato: {k.dtype}")
        
        # Aplicar el filtro BM3D
        y = self.filter(k)
        
        # Escribir la imagen de salida
        output_size = self.decode_write(y, self.args.decoded)
        return output_size
            
    def filter(self, img):
        """
        Aplica el filtrado BM3D (denoising/deblurring) a la imagen.
        Soporta imágenes monocromáticas, RGB y multicanal.
        """
        logging.info(f"Aplicando filtro BM3D (sigma_bm3d={self.args.sigma_bm3d}, perfil='{self.args.profile_bm3d}')")
        
        # Validación de parámetros
        if self.args.sigma_bm3d < 0:
            logging.warning(f"Sigma inválido ({self.args.sigma_bm3d}). Ajustando a 0.")
            self.args.sigma_bm3d = 0.0

        # Normalizar la imagen al rango [0, 1] para el procesamiento BM3D
        img_float = img.astype(np.float32) / 255.0
        sigma_normalized = self.args.sigma_bm3d / 255.0
        
        # Manejar PSD para ruido correlacionado
        if self.args.psd_bm3d and os.path.exists(self.args.psd_bm3d):
            try:
                logging.info(f"Cargando PSD de ruido desde {self.args.psd_bm3d}")
                noise_spec = np.load(self.args.psd_bm3d)
            except Exception as e:
                logging.error(f"Error al cargar PSD: {e}. Usando sigma como respaldo.")
                noise_spec = sigma_normalized
        else:
            noise_spec = sigma_normalized

        # Cargar PSF si se solicita deblurring
        psf = None
        if self.args.psf_bm3d and os.path.exists(self.args.psf_bm3d):
            try:
                logging.info(f"Cargando PSF desde {self.args.psf_bm3d} para deblurring")
                psf = np.load(self.args.psf_bm3d)
            except Exception as e:
                logging.error(f"Error al cargar PSF: {e}. Solo se realizará denoising.")

        # Seleccionar mapeo de perfil
        profile_map = {
            'np': bm3d.BM3DProfile(),
            'lc': bm3d.BM3DProfileLC(),
            'high': bm3d.BM3DProfileHigh(),
            'vn': bm3d.BM3DProfileVN()
        }
        selected_profile = profile_map.get(self.args.profile_bm3d, bm3d.BM3DProfile())
        
        try:
            # Verificar si la imagen es en color (3 canales) o escala de grises
            if len(img.shape) == 3:
                channels = img.shape[2]
                if channels == 3:
                    if psf is not None:
                        logging.info("Aplicando deblurring BM3D a imagen RGB (canal por canal).")
                        denoised = np.zeros_like(img_float)
                        for i in range(3):
                            denoised[:,:,i] = bm3d.bm3d_deblurring(img_float[:,:,i], noise_spec, psf, profile=selected_profile)
                    else:
                        logging.info("Entrada reconocida como RGB. Usando bm3d_rgb.")
                        denoised = bm3d.bm3d_rgb(img_float, noise_spec, profile=selected_profile)
                else:
                    logging.info(f"Entrada reconocida como imagen de {channels} canales. Aplicando BM3D por canal.")
                    denoised = np.zeros_like(img_float)
                    for i in range(channels):
                        if psf is not None:
                             denoised[:,:,i] = bm3d.bm3d_deblurring(img_float[:,:,i], noise_spec, psf, profile=selected_profile)
                        else:
                             denoised[:,:,i] = bm3d.bm3d(img_float[:,:,i], noise_spec, profile=selected_profile)
            else:
                if psf is not None:
                    logging.info("Aplicando deblurring BM3D a imagen en escala de grises.")
                    denoised = bm3d.bm3d_deblurring(img_float, noise_spec, psf, profile=selected_profile)
                else:
                    logging.info("Entrada reconocida como escala de grises. Usando bm3d.")
                    denoised = bm3d.bm3d(img_float, noise_spec, profile=selected_profile)
            
            # Recortar y convertir de nuevo a uint8 [0, 255]
            result = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
            return result
        except Exception as e:
            logging.error(f"Error durante el procesamiento BM3D: {e}")
            logging.warning("Volviendo a la imagen original debido a un error.")
            return img

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
