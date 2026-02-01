'''
Módulo de compatibilidad multiplataforma para VCF.

Este módulo proporciona funciones y constantes para manejar rutas de archivos
de manera compatible con Windows, Linux y macOS.

Uso:
    import platform_utils as pu
    
    # Obtener directorio temporal
    temp_dir = pu.get_temp_dir()
    
    # Obtener ruta completa a un archivo temporal
    file_path = pu.get_temp_path("encoded.mctf")
    
    # Usar constantes predefinidas
    output_prefix = pu.ENCODE_OUTPUT_PREFIX
'''

import os
import sys
import tempfile

# ============================================================================
# Funciones de utilidad
# ============================================================================

def get_temp_dir():
    """
    Obtiene el directorio temporal del sistema de forma multiplataforma.
    
    Returns:
        str: Ruta al directorio temporal del sistema.
             - Windows: Típicamente C:\\Users\\<user>\\AppData\\Local\\Temp
             - Linux/Mac: Típicamente /tmp
    """
    return tempfile.gettempdir()


def get_vcf_temp_dir():
    """
    Obtiene el directorio temporal para VCF, creándolo si no existe.
    
    Para mantener compatibilidad con scripts existentes que esperan /tmp/,
    en Windows usamos C:/tmp y en Linux/Mac usamos /tmp.
    
    Returns:
        str: Ruta al directorio temporal de VCF.
    """
    if sys.platform == 'win32':
        temp_dir = "C:/tmp"
    else:
        temp_dir = "/tmp"
    
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def get_temp_path(filename):
    """
    Construye una ruta completa a un archivo en el directorio temporal de VCF.
    
    Args:
        filename: Nombre del archivo (sin ruta).
        
    Returns:
        str: Ruta completa al archivo en el directorio temporal.
    """
    return os.path.join(get_vcf_temp_dir(), filename)


def ensure_description_file(description_text):
    """
    Crea el archivo description.txt requerido por parser.py.
    
    Esta función debe llamarse antes de importar parser.py.
    
    Args:
        description_text: Texto de descripción para el módulo.
    """
    desc_path = get_temp_path("description.txt")
    with open(desc_path, 'w') as f:
        f.write(description_text)


def get_file_uri(path):
    """
    Convierte una ruta de archivo a formato URI file://.
    
    Args:
        path: Ruta al archivo.
        
    Returns:
        str: URI del archivo (ej: file:///tmp/image.png o file:///C:/tmp/image.png)
    """
    # Normalizar la ruta
    path = os.path.abspath(path)
    
    if sys.platform == 'win32':
        # En Windows, convertir backslashes y añadir /
        path = path.replace('\\', '/')
        return f"file:///{path}"
    else:
        return f"file://{path}"


# ============================================================================
# Constantes de rutas predefinidas (compatibles con todos los sistemas)
# ============================================================================

# Video encoding/decoding
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/videos/mobile_352x288x30x420x300.mp4"
ENCODE_OUTPUT_PREFIX = get_temp_path("encoded")
DECODE_INPUT_PREFIX = ENCODE_OUTPUT_PREFIX
DECODE_OUTPUT_PREFIX = get_temp_path("decoded")
DECODE_OUTPUT = get_temp_path("decoded.mp4")

# Image encoding/decoding
ORIGINAL = get_temp_path("original.png")
ENCODED = get_temp_path("encoded")
DECODED = get_temp_path("decoded.png")

# Frame prefixes
FRAME_PREFIX = get_temp_path("img_")
ORIGINAL_FRAME_PREFIX = get_temp_path("original_")
DECODED_FRAME_PREFIX = get_temp_path("decoded_")


def get_original_frame_path(index, digits=4):
    """Genera la ruta para un frame original."""
    return f"{ORIGINAL_FRAME_PREFIX}{index:0{digits}d}.png"


def get_decoded_frame_path(index, digits=4):
    """Genera la ruta para un frame decodificado."""
    return f"{DECODED_FRAME_PREFIX}{index:0{digits}d}.png"


def get_encoded_frame_path(prefix, index, digits=4):
    """Genera la ruta para un frame codificado."""
    return f"{prefix}_{index:0{digits}d}"


# ============================================================================
# Inicialización automática
# ============================================================================

# Asegurar que el directorio temporal existe al importar el módulo
_temp_dir = get_vcf_temp_dir()

