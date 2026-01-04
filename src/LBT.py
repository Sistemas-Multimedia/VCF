'''
Transformada de Bloque Aprendida (LBT) usando un Autoencoder de 3 capas.

Este módulo implementa un método de compresión de imágenes basado en KLT (Karhunen-Loève Transform)
que aprende la transformada óptima para maximizar la compactación de energía en bloques de píxeles.
'''

import numpy as np
import logging
import main
import os

try:
    os.makedirs("/tmp", exist_ok=True)
except:
    pass

with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)

import parser
import importlib
import struct
import cv2

# Importar transformaciones de color
# pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import from_RGB  # type: ignore
from color_transforms.YCoCg import to_RGB  # type: ignore

# Importar funciones DCT (Transformada Coseno Discreta) para el procesamiento de bloques
# pip install "DCT2D @ git+https://github.com/vicente-gonzalez-ruiz/DCT2D"
from DCT2D.block_DCT import get_subbands  # type: ignore
from DCT2D.block_DCT import get_blocks  # type: ignore

# Parámetros por defecto
default_block_size = 8
default_CT = "YCoCg"
perceptual_quantization = False
disable_subbands = False

parser.parser_encode.add_argument("-B", "--block_size_DCT", type=parser.int_or_str, help=f"Tamaño de bloque (por defecto: {default_block_size})", default=default_block_size)
parser.parser_encode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Transformada de color (por defecto: \"{default_CT}\")", default=default_CT)
parser.parser_encode.add_argument("-x", "--disable_subbands", action='store_true', help=f"Desactivar reordenamiento de coeficientes en subbandas (por defecto: \"{disable_subbands}\")", default=disable_subbands)

parser.parser_decode.add_argument("-B", "--block_size_DCT", type=parser.int_or_str, help=f"Tamaño de bloque (por defecto: {default_block_size})", default=default_block_size)
parser.parser_decode.add_argument("-t", "--color_transform", type=parser.int_or_str, help=f"Transformada de color (por defecto: \"{default_CT}\")", default=default_CT)
parser.parser_decode.add_argument("-x", "--disable_subbands", action='store_true', help=f"Desactivar reordenamiento de coeficientes en subbandas (por defecto: \"{disable_subbands}\")", default=disable_subbands)


args = parser.parser.parse_known_args()[0]
CT = importlib.import_module(args.color_transform)

class LBT_Autoencoder:
    """
    Autoencoder de 3 capas que aprende la transformada lineal óptima (KLT - Transformada Karhunen-Loève)
    para maximizar la ganancia de codificación (compactación de energía).
    
    La idea es que la KLT descorrelaciona los datos y concentra la energía en pocos coeficientes,
    mejorando la tasa de compresión respecto a transformadas fijas como DCT.
    """
    def __init__(self, block_size):
        """
        Inicializa el autoencoder con un tamaño de bloque específico.
        
        Parámetros:
            block_size: Tamaño del bloque (debe ser entero positivo)
            
        Excepciones:
            ValueError: Si block_size no es entero positivo
        """
        if not isinstance(block_size, (int, np.integer)):
            raise ValueError(f"block_size debe ser un entero, recibido: {type(block_size)}")
        if block_size <= 0:
            raise ValueError(f"block_size debe ser positivo, recibido: {block_size}")
        
        self.block_size = block_size
        self.input_dim = block_size * block_size
        self.weights = None  # Matriz de transformada aprendida (pesos hacia adelante)

    def train(self, patches):
        """
        Aprende los pesos óptimos a partir de parches de la imagen.
        
        Utiliza Análisis de Componentes Principales (PCA) / Transformada Karhunen-Loève
        para encontrar los vectores propios de la matriz de covarianza de los datos.
        Estos vectores propios forman la base ortogonal que maximiza la compactación de energía.
        
        Parámetros:
            patches: arreglo de forma (N_parches, block_size, block_size) o (N_parches, input_dim)
            
        Excepciones:
            TypeError: Si patches no es un arreglo numpy
            ValueError: Si patches tiene forma incompatible
            np.linalg.LinAlgError: Si la matriz de covarianza es singular
        """
        try:
            # Validar tipo de entrada
            if not isinstance(patches, np.ndarray):
                raise TypeError(f"patches debe ser un arreglo numpy, recibido: {type(patches)}")
            
            # Validar número de dimensiones
            if patches.ndim not in [2, 3]:
                raise ValueError(f"patches debe tener 2 o 3 dimensiones, recibido: {patches.ndim}")
            
            # Aplanar los parches si es necesario
            if patches.ndim == 3:
                N, H, W = patches.shape
                if H != self.block_size or W != self.block_size:
                    raise ValueError(f"Forma de bloque esperada: ({self.block_size}, {self.block_size}), recibida: ({H}, {W})")
                X = patches.reshape(-1, self.input_dim)
            else:
                X = patches
                if X.shape[1] != self.input_dim:
                    raise ValueError(f"Dimensión esperada: {self.input_dim}, recibida: {X.shape[1]}")
            
            # Validar número de parches
            if X.shape[0] < 2:
                logging.warning(f"Número de parches muy pequeño ({X.shape[0]}). Se requieren al menos 2 parches para calcular covarianza confiable.")
            
            # Validar que hay datos válidos
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError("Los parches contienen valores NaN o infinitos")
            
            # Calcular la matriz de covarianza: C = (X^T * X) / N
            # La covarianza describe cómo se distribuye la varianza en los datos
            C = np.cov(X, rowvar=False)
            
            # Validar covarianza
            if C.ndim != 2 or C.shape[0] != C.shape[1]:
                raise ValueError(f"Covarianza debe ser cuadrada, forma: {C.shape}")
            
            # Descomposición en valores y vectores propios
            # eigh se utiliza para matrices simétricas/Hermitianas (la covarianza lo es)
            # Retorna autovalores (w) y autovectores (v) ordenados en forma ascendente
            w, v = np.linalg.eigh(C)
            
            # Validar resultados de eigh
            if np.any(np.isnan(w)) or np.any(np.isnan(v)):
                raise np.linalg.LinAlgError("Descomposición en valores/vectores propios produjo NaN")
            
            # Ordenar los autovectores por autovalores descendentes (mayor energía primero)
            # Los autovalores grandes corresponden a direcciones con alta varianza
            idx = np.argsort(w)[::-1]
            self.weights = v[:, idx].T  # Las filas son autovectores. Forma: (D, D)
            
            logging.debug(f"LBT entrenado correctamente con {X.shape[0]} parches de tamaño {self.block_size}x{self.block_size}")
            
        except TypeError as e:
            logging.error(f"Error de tipo en train: {e}")
            raise
        except ValueError as e:
            logging.error(f"Error de valor en train: {e}")
            raise
        except np.linalg.LinAlgError as e:
            logging.error(f"Error de álgebra lineal en train: {e}")
            raise

    def set_weights(self, weights):
        """
        Establece manualmente los pesos de la transformada.
        
        Parámetros:
            weights: Matriz de pesos de forma (input_dim, input_dim)
            
        Excepciones:
            TypeError: Si weights no es un arreglo numpy
            ValueError: Si weights tiene forma incorrecta
        """
        if not isinstance(weights, np.ndarray):
            raise TypeError(f"weights debe ser un arreglo numpy, recibido: {type(weights)}")
        
        if weights.shape != (self.input_dim, self.input_dim):
            raise ValueError(f"weights debe tener forma ({self.input_dim}, {self.input_dim}), recibida: {weights.shape}")
        
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            raise ValueError("weights contiene valores NaN o infinitos")
        
        self.weights = weights
        logging.debug(f"Pesos de transformada establecidos correctamente")

    def get_weights(self):
        """
        Retorna los pesos aprendidos de la transformada.
        
        Retorna:
            Matriz de pesos, o None si no ha sido entrenado
            
        Excepciones:
            RuntimeError: Si se intenta obtener pesos sin entrenar primero
        """
        if self.weights is None:
            raise RuntimeError("Los pesos no han sido inicializados. Entrene el autoencoder primero.")
        return self.weights

    def forward(self, patches):
        """
        Paso hacia adelante (Codificar / Transformar).
        Aplica la transformada KLT aprendida a los parches.
        
        Entrada: arreglo de forma (N, block_size, block_size)
        Salida: arreglo de forma (N, block_size, block_size) con coeficientes transformados
        
        Excepciones:
            RuntimeError: Si no se ha entrenado previamente
            ValueError: Si la forma del parche es incorrecta
        """
        if self.weights is None:
            raise RuntimeError("Autoencoder no entrenado. Ejecute train() primero.")
        
        try:
            if patches.ndim != 3:
                raise ValueError(f"patches debe ser 3D, recibido: {patches.ndim}D")
            
            N, H, W = patches.shape
            if H != self.block_size or W != self.block_size:
                raise ValueError(f"Forma de bloque esperada: ({self.block_size}, {self.block_size}), recibida: ({H}, {W})")
            
            X = patches.reshape(N, -1)  # (N, D)
            
            # Verificar NaN/Inf
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError("patches contiene valores NaN o infinitos")
            
            # Transformar: C = X @ W^T 
            coeffs = np.dot(X, self.weights.T)
            
            if np.any(np.isnan(coeffs)):
                raise ValueError("Coeficientes transformados contienen NaN")
            
            return coeffs.reshape(N, H, W)
            
        except ValueError as e:
            logging.error(f"Error en forward: {e}")
            raise

    def backward(self, coeffs):
        """
        Paso hacia atrás (Decodificar / Transformada Inversa).
        Reconstruye los parches originales desde los coeficientes transformados.
        
        Entrada: arreglo de forma (N, block_size, block_size) con coeficientes
        Salida: arreglo de forma (N, block_size, block_size) reconstruido
        
        Excepciones:
            RuntimeError: Si no se ha entrenado previamente
            ValueError: Si la forma de los coeficientes es incorrecta
        """
        if self.weights is None:
            raise RuntimeError("Autoencoder no entrenado. Ejecute train() primero.")
        
        try:
            if coeffs.ndim != 3:
                raise ValueError(f"coeffs debe ser 3D, recibido: {coeffs.ndim}D")
            
            N, H, W = coeffs.shape
            if H != self.block_size or W != self.block_size:
                raise ValueError(f"Forma de bloque esperada: ({self.block_size}, {self.block_size}), recibida: ({H}, {W})")
            
            Y = coeffs.reshape(N, -1)
            
            # Verificar NaN/Inf
            if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
                raise ValueError("coeffs contiene valores NaN o infinitos")
            
            # Inversa: Rec = Y @ W
            # Dado que W es ortogonal, W^(-1) = W^T.
            rec = np.dot(Y, self.weights)
            
            if np.any(np.isnan(rec)):
                raise ValueError("Reconstrucción contiene NaN")
            
            return rec.reshape(N, H, W)
            
        except ValueError as e:
            logging.error(f"Error en backward: {e}")
            raise


class CoDec(CT.CoDec):
    """
    Codificador/Decodificador que utiliza la Transformada de Bloque Aprendida (LBT).
    
    Extiende la clase CoDec del módulo de transformada de color seleccionada.
    Implementa codificación y decodificación de imágenes usando LBT seguida de
    cuantización y compresión de entropía.
    """

    def __init__(self, args):
        """
        Inicializa el codificador/decodificador con los argumentos de configuración.
        
        Parámetros:
            args: Argumentos de línea de comandos
            
        Excepciones:
            AttributeError: Si faltan atributos requeridos en args
            ValueError: Si block_size es inválido
        """
        logging.debug("trace")
        try:
            # Validar argumentos requeridos
            if not hasattr(args, 'block_size_DCT'):
                raise AttributeError("args debe tener atributo 'block_size_DCT'")
            
            block_size = args.block_size_DCT
            if isinstance(block_size, str):
                try:
                    block_size = int(block_size)
                except ValueError:
                    raise ValueError(f"block_size_DCT no puede convertirse a entero: {block_size}")
            
            if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
                raise ValueError(f"block_size_DCT debe ser entero positivo, recibido: {block_size}")
            
            super().__init__(args)
            self.block_size = block_size
            self.lbt = LBT_Autoencoder(self.block_size)
            
            # Establecer desplazamiento según el cuantizador usado
            # El cuantizador "deadzone" requiere un desplazamiento de 128 para centrar valores
            if hasattr(args, 'quantizer') and args.quantizer == "deadzone":
                self.offset = 128
            else:
                self.offset = 0
                
            logging.debug(f"CoDec inicializado: block_size={self.block_size}, offset={self.offset}")
            
        except (AttributeError, ValueError) as e:
            logging.error(f"Error en inicialización de CoDec: {e}")
            raise

    def pad_and_center_to_multiple_of_block_size(self, img):
        """
        Rellena la imagen con ceros para que sus dimensiones sean múltiplos del tamaño de bloque.
        
        Este paso es necesario porque la transformada de bloque requiere que la imagen
        sea divisible por el tamaño de bloque.
        
        Parámetros:
            img: imagen de entrada de forma (altura, ancho, canales)
            
        Retorna:
            Imagen rellenada con forma que es múltiplo del tamaño de bloque
            
        Excepciones:
            ValueError: Si la imagen tiene formato incorrecto
            TypeError: Si la imagen no es un arreglo numpy
        """
        try:
            if not isinstance(img, np.ndarray):
                raise TypeError(f"img debe ser un arreglo numpy, recibido: {type(img)}")
            
            if img.ndim != 3:
                raise ValueError(f"La imagen debe ser un arreglo 3D (altura, ancho, canales), recibido: {img.ndim}D")
            
            if img.shape[2] != 3:
                logging.warning(f"Se esperan 3 canales, recibidos: {img.shape[2]}")

            self.original_shape = img.shape
            height, width, channels = img.shape
            
            # Validar dimensiones
            if height <= 0 or width <= 0 or channels <= 0:
                raise ValueError(f"Dimensiones de imagen inválidas: {img.shape}")
            
            if self.block_size <= 0:
                raise ValueError(f"block_size inválido: {self.block_size}")

            # Calcular las dimensiones de destino (múltiplos del tamaño de bloque)
            target_height = (height + self.block_size - 1) // self.block_size * self.block_size
            target_width = (width + self.block_size - 1) // self.block_size * self.block_size

            pad_height = target_height - height
            pad_width = target_width - width

            # Distribuir el relleno equitativamente entre arriba/abajo e izquierda/derecha
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            # Aplicar el relleno
            padded_img = np.pad(
                img,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )
            
            logging.debug(f"Imagen rellenada: {img.shape} -> {padded_img.shape}")
            return padded_img
            
        except (TypeError, ValueError) as e:
            logging.error(f"Error en pad_and_center_to_multiple_of_block_size: {e}")
            raise

    def remove_padding(self, padded_img):
        """
        Elimina el relleno agregado durante la codificación.
        
        Restaura la imagen a sus dimensiones originales.
        
        Parámetros:
            padded_img: imagen rellenada
            
        Retorna:
            Imagen sin relleno con las dimensiones originales
            
        Excepciones:
            ValueError: Si las dimensiones son inconsistentes
            RuntimeError: Si original_shape no ha sido establecido
        """
        try:
            if not hasattr(self, 'original_shape') or self.original_shape is None:
                raise RuntimeError("original_shape no ha sido establecido. Ejecute pad_and_center_to_multiple_of_block_size primero.")
            
            if not isinstance(padded_img, np.ndarray):
                raise ValueError("padded_img debe ser un arreglo numpy")
            
            original_height, original_width, _  = self.original_shape
            padded_height, padded_width, _ = padded_img.shape

            pad_height = padded_height - original_height
            pad_width = padded_width - original_width
            
            if pad_height < 0 or pad_width < 0:
                raise ValueError(f"Imagen rellenada más pequeña que original: original={self.original_shape}, padded={padded_img.shape}")

            pad_top = pad_height // 2
            pad_left = pad_width // 2

            unpadded_img = padded_img[
                pad_top:pad_top + original_height,
                pad_left:pad_left + original_width,
                :
            ]
            
            if unpadded_img.shape != self.original_shape:
                raise ValueError(f"Dimensiones de salida incorrectas: esperado {self.original_shape}, obtenido {unpadded_img.shape}")
            
            logging.debug(f"Padding removido: {padded_img.shape} -> {unpadded_img.shape}")
            return unpadded_img
            
        except (ValueError, RuntimeError) as e:
            logging.error(f"Error en remove_padding: {e}")
            raise

    def encode_fn(self, in_fn, out_fn):
        """
        Codifica una imagen utilizando LBT.
        
        Proceso:
        1. Lee la imagen
        2. Rellena a múltiplos del tamaño de bloque
        3. Aplica transformada de color
        4. Entrena y aplica LBT para cada canal
        5. Reordena coeficientes en subbandas (opcional)
        6. Cuantiza los coeficientes
        7. Comprime usando compresión de entropía
        
        Parámetros:
            in_fn: ruta del archivo de imagen original
            out_fn: ruta del archivo de salida codificado
            
        Excepciones:
            FileNotFoundError: Si el archivo de entrada no existe
            IOError: Si hay problemas al leer o escribir archivos
            ValueError: Si la imagen está dañada o es inválida
        """
        logging.debug("trace")
        
        try:
            # Validar rutas
            if not isinstance(in_fn, str):
                raise ValueError(f"in_fn debe ser string, recibido: {type(in_fn)}")
            if not isinstance(out_fn, str):
                raise ValueError(f"out_fn debe ser string, recibido: {type(out_fn)}")
            
            # Validar que el archivo existe
            if not os.path.exists(in_fn):
                raise FileNotFoundError(f"Archivo de entrada no encontrado: {in_fn}")
            
            # Validar permisos de lectura
            if not os.access(in_fn, os.R_OK):
                raise PermissionError(f"Permiso denegado para leer: {in_fn}")
            
            # Validar permisos de escritura en directorio de salida
            out_dir = os.path.dirname(out_fn) or "."
            if not os.access(out_dir, os.W_OK):
                raise PermissionError(f"Permiso denegado para escribir en: {out_dir}")
            
            # Leer imagen y convertir a punto flotante
            try:
                img = self.encode_read_fn(in_fn).astype(np.float32)
            except Exception as e:
                raise IOError(f"Error al leer imagen: {in_fn}: {e}")
            
            if img is None or img.size == 0:
                raise ValueError(f"Imagen vacía o inválida: {in_fn}")
            
            logging.info(f"Imagen leída: forma={img.shape}")
            
            # Rellenar la imagen
            self.original_shape = img.shape
            padded_img = self.pad_and_center_to_multiple_of_block_size(img)
            
            # Guardar las dimensiones originales para la decodificación
            try:
                with open(f"{out_fn}_shape.bin", "wb") as file:
                    file.write(struct.pack("iii", *self.original_shape))
            except IOError as e:
                raise IOError(f"Error al escribir archivo de dimensiones: {out_fn}_shape.bin: {e}")
                
            img = padded_img
            # Aplicar desplazamiento para centrar valores
            img -= self.offset
            
            # Transformada de color (RGB -> YCoCg u otro espacio de color)
            try:
                CT_img = from_RGB(img)  # Forma: (H, W, 3)
            except Exception as e:
                raise ValueError(f"Error en transformada de color: {e}")
            
            H, W, C = CT_img.shape
            
            # Preparar para LBT
            # Extraer todos los bloques y entrenar la LBT
            # Se utiliza una LBT independiente para cada canal para mejor decorrelación
            
            transformed_channels = []
            lbt_weights = []
            
            # Procesar cada canal de color independientemente
            for c in range(C):
                try:
                    channel = CT_img[:, :, c]
                    
                    # Extraer parches (bloques no superpuestos)
                    # channel tiene forma (H, W) donde H, W son múltiplos de block_size
                    
                    n_blocks_y = H // self.block_size
                    n_blocks_x = W // self.block_size
                    
                    if n_blocks_y <= 0 or n_blocks_x <= 0:
                        raise ValueError(f"Número de bloques inválido: ({n_blocks_y}, {n_blocks_x})")
                    
                    # Remodelar a (ny, nx, b, b) y luego a (N_bloques, b, b)
                    patches = channel.reshape(n_blocks_y, self.block_size, n_blocks_x, self.block_size).swapaxes(1, 2).reshape(-1, self.block_size, self.block_size)
                    
                    # Entrenar LBT con los parches de este canal
                    lbt = LBT_Autoencoder(self.block_size)
                    lbt.train(patches)
                    weights = lbt.get_weights()
                    lbt_weights.append(weights)
                    
                    # Aplicar la transformada LBT hacia adelante
                    coeffs_patches = lbt.forward(patches)
                    
                    # Remodelar los coeficientes de vuelta a imagen
                    coeffs_img = coeffs_patches.reshape(n_blocks_y, n_blocks_x, self.block_size, self.block_size).swapaxes(1, 2).reshape(H, W)
                    transformed_channels.append(coeffs_img)
                    
                    logging.debug(f"Canal {c} procesado: {patches.shape[0]} bloques")
                    
                except Exception as e:
                    raise ValueError(f"Error procesando canal {c}: {e}")
            
            # Combinar los canales transformados
            LBT_img = np.stack(transformed_channels, axis=2)
            lbt_weights = np.array(lbt_weights, dtype=np.float32)  # Forma: (3, D, D)
            
            # Guardar los pesos aprendidos para la decodificación
            try:
                with open(f"{out_fn}_weights.bin", "wb") as f:
                    f.write(lbt_weights.tobytes())
            except IOError as e:
                raise IOError(f"Error al escribir archivo de pesos: {out_fn}_weights.bin: {e}")
                
            # Reordenar coeficientes en subbandas (agrupar por frecuencia)
            try:
                if args.disable_subbands:
                    decom_img = LBT_img
                else:
                    decom_img = get_subbands(LBT_img, self.block_size, self.block_size)
            except Exception as e:
                raise ValueError(f"Error en procesamiento de subbandas: {e}")
                
            # Cuantizar los coeficientes para reducir la precisión
            try:
                decom_k = self.quantize_decom(decom_img)
            except Exception as e:
                raise ValueError(f"Error en cuantización: {e}")
            
            decom_k += self.offset
            
            # Verificar y advertir si hay valores fuera de rango
            if self.args.debug:
                if np.max(decom_k) > 255:
                    logging.warning(f"decom_k max={np.max(decom_k)}")
                if np.min(decom_k) < 0:
                    logging.warning(f"decom_k min={np.min(decom_k)}")
                    
            # Aplicar compresión de entropía (Huffman, zlib, etc.)
            decom_k = decom_k.astype(np.uint8)
            try:
                decom_k = self.compress(decom_k)
            except Exception as e:
                raise ValueError(f"Error en compresión de entropía: {e}")
            
            # Escribir el flujo de código comprimido
            try:
                output_size = self.encode_write_fn(decom_k, out_fn)
            except Exception as e:
                raise IOError(f"Error al escribir archivo codificado: {out_fn}: {e}")
            
            logging.info(f"Codificación completada: tamaño salida = {output_size} bytes")
            return output_size
            
        except (FileNotFoundError, IOError, PermissionError, ValueError) as e:
            logging.error(f"Error en encode_fn: {e}")
            raise
        except Exception as e:
            logging.error(f"Error inesperado en encode_fn: {e}")
            raise

    def decode_fn(self, in_fn, out_fn):
        """
        Decodifica una imagen codificada con LBT.
        
        Proceso inverso a encode_fn:
        1. Lee el flujo de código comprimido
        2. Descomprime
        3. Deshace la cuantización
        4. Restaura los bloques desde subbandas
        5. Aplica transformada inversa LBT para cada canal
        6. Aplica transformada de color inversa
        7. Elimina el relleno
        8. Escribe la imagen decodificada
        
        Parámetros:
            in_fn: ruta del archivo codificado
            out_fn: ruta del archivo de salida decodificado
            
        Excepciones:
            FileNotFoundError: Si falta alguno de los archivos necesarios
            IOError: Si hay problemas al leer o escribir archivos
            ValueError: Si los datos están dañados o son inválidos
        """
        logging.debug("trace")
        
        try:
            # Validar rutas
            if not isinstance(in_fn, str):
                raise ValueError(f"in_fn debe ser string, recibido: {type(in_fn)}")
            if not isinstance(out_fn, str):
                raise ValueError(f"out_fn debe ser string, recibido: {type(out_fn)}")
            
            # Validar que los archivos existen
            files_to_check = [in_fn + ".tif", f"{in_fn}_shape.bin", f"{in_fn}_weights.bin"]
            for file_path in files_to_check:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Archivo necesario no encontrado: {file_path}")
                if not os.access(file_path, os.R_OK):
                    raise PermissionError(f"Permiso denegado para leer: {file_path}")
            
            # Validar permisos de escritura en directorio de salida
            out_dir = os.path.dirname(out_fn) or "."
            if not os.access(out_dir, os.W_OK):
                raise PermissionError(f"Permiso denegado para escribir en: {out_dir}")
            
            # Leer y descomprimir el flujo de código
            try:
                decom_k = self.decode_read_fn(in_fn)
            except Exception as e:
                raise IOError(f"Error al leer archivo codificado: {in_fn}: {e}")
            
            if decom_k is None or len(decom_k) == 0:
                raise ValueError(f"Archivo codificado vacío o inválido: {in_fn}")
            
            # Leer las dimensiones originales
            try:
                with open(f"{in_fn}_shape.bin", "rb") as file:
                    data = file.read(12)
                    if len(data) != 12:
                        raise ValueError("Archivo de dimensiones corrupto (tamaño incorrecto)")
                    self.original_shape = struct.unpack("iii", data)
            except (IOError, struct.error) as e:
                raise IOError(f"Error al leer archivo de dimensiones: {in_fn}_shape.bin: {e}")
            
            logging.debug(f"Dimensiones originales: {self.original_shape}")
            
            # Leer los pesos aprendidos de la LBT
            try:
                dim = self.block_size * self.block_size
                with open(f"{in_fn}_weights.bin", "rb") as f:
                    weights_data = f.read()
                    expected_size = 3 * dim * dim * 4  # 3 canales, dim x dim, float32 (4 bytes)
                    if len(weights_data) != expected_size:
                        raise ValueError(f"Archivo de pesos corrupto: tamaño esperado {expected_size}, recibido {len(weights_data)}")
                    lbt_weights = np.frombuffer(weights_data, dtype=np.float32).reshape(3, dim, dim)
            except (IOError, ValueError, struct.error) as e:
                raise IOError(f"Error al leer archivo de pesos: {in_fn}_weights.bin: {e}")
            
            logging.debug(f"Pesos cargados: forma={lbt_weights.shape}")
            
            # Descomprimir el flujo de código
            try:
                decom_k = self.decompress(decom_k)
            except Exception as e:
                raise ValueError(f"Error en descompresión: {e}")
            
            decom_k = decom_k.astype(np.int16)
            # Revertir el desplazamiento
            decom_k -= self.offset
            
            # Deshaceer la cuantización
            try:
                decom_y = self.dequantize_decom(decom_k)
            except Exception as e:
                raise ValueError(f"Error en desdecuantización: {e}")
            
            # Restaurar bloques desde subbandas
            try:
                if args.disable_subbands:
                    LBT_img = decom_y
                else:
                    LBT_img = get_blocks(decom_y, self.block_size, self.block_size)
            except Exception as e:
                raise ValueError(f"Error restaurando bloques desde subbandas: {e}")
            
            # Aplicar transformada inversa LBT
            try:
                H, W, C = LBT_img.shape
                decoded_channels = []
                
                n_blocks_y = H // self.block_size
                n_blocks_x = W // self.block_size
                
                if n_blocks_y <= 0 or n_blocks_x <= 0:
                    raise ValueError(f"Número de bloques inválido: ({n_blocks_y}, {n_blocks_x})")
                
                # Procesar cada canal de forma independiente
                for c in range(C):
                    channel = LBT_img[:, :, c]
                    
                    # Extraer parches / bloques
                    patches = channel.reshape(n_blocks_y, self.block_size, n_blocks_x, self.block_size).swapaxes(1, 2).reshape(-1, self.block_size, self.block_size)
                    
                    # Configurar LBT con los pesos cargados
                    lbt = LBT_Autoencoder(self.block_size)
                    lbt.set_weights(lbt_weights[c])
                    
                    # Aplicar transformada inversa LBT
                    rec_patches = lbt.backward(patches)
                    
                    # Remodelar de vuelta a imagen
                    rec_img = rec_patches.reshape(n_blocks_y, n_blocks_x, self.block_size, self.block_size).swapaxes(1, 2).reshape(H, W)
                    decoded_channels.append(rec_img)
                    
                    logging.debug(f"Canal {c} decodificado")
                    
            except Exception as e:
                raise ValueError(f"Error en transformada inversa LBT: {e}")
            
            # Combinar los canales reconstruidos
            CT_y = np.stack(decoded_channels, axis=2)
            
            # Eliminar el relleno agregado durante la codificación
            try:
                CT_y = self.remove_padding(CT_y)
            except Exception as e:
                raise ValueError(f"Error al remover padding: {e}")
            
            # Aplicar transformada de color inversa (YCoCg -> RGB)
            try:
                y = to_RGB(CT_y)
            except Exception as e:
                raise ValueError(f"Error en transformada de color inversa: {e}")
            
            y += self.offset
            
            # Asegurar que los valores están en el rango [0, 255] y convertir a uint8
            y = np.clip(y, 0, 255).astype(np.uint8)
            
            if np.any(np.isnan(y)):
                raise ValueError("Imagen decodificada contiene valores NaN")
            
            # Escribir la imagen decodificada
            try:
                output_size = self.decode_write_fn(y, out_fn)
            except Exception as e:
                raise IOError(f"Error al escribir imagen decodificada: {out_fn}: {e}")
            
            logging.info(f"Decodificación completada: tamaño salida = {output_size} bytes")
            return output_size
            
        except (FileNotFoundError, IOError, PermissionError, ValueError) as e:
            logging.error(f"Error en decode_fn: {e}")
            raise
        except Exception as e:
            logging.error(f"Error inesperado en decode_fn: {e}")
            raise

        
    def encode(self, in_fn=None, out_fn=None):
        """
        Interfaz pública para codificación.
        Utiliza valores por defecto si no se especifican rutas.
        """
        if in_fn is None:
            in_fn = self.args.original if hasattr(self.args, 'original') and self.args.original != "/docs/original.jpg" else "../docs/Coco1.jpg"
        if out_fn is None:
            out_fn = self.args.encoded if hasattr(self.args, 'encoded') and self.args.encoded != "/docs/encoded/" else "../docs/encoded/lbt_encoded"
        return self.encode_fn(in_fn, out_fn)
        
    def decode(self, in_fn=None, out_fn=None):
        """
        Interfaz pública para decodificación.
        Utiliza valores por defecto si no se especifican rutas.
        """
        if in_fn is None:
            in_fn = self.args.encoded if hasattr(self.args, 'encoded') and self.args.encoded != "/docs/encoded/" else "../docs/encoded/lbt_encoded"
        if out_fn is None:
            out_fn = self.args.decoded if hasattr(self.args, 'decoded') and self.args.decoded != "/docs/decoded/" else "../docs/decoded/lbt_decoded.jpg"
        return self.decode_fn(in_fn, out_fn)

    def quantize_decom(self, decom):
        """Cuantiza los coeficientes descompuestos."""
        return self.quantize(decom)

    def dequantize_decom(self, decom_k):
        """Deshace la cuantización de los coeficientes."""
        return self.dequantize(decom_k)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)