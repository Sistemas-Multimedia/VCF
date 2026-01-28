'''
# Huffman adaptativo con contexto para compresión de imágenes
# 
# La diferencia con Huffman normal: aquí vamos actualizando las frecuencias
# sobre la marcha, símbolo por símbolo. El codificador y decodificador empiezan
# con las mismas frecuencias iniciales y van actualizando igual → no hace falta
# guardar el árbol completo, solo los metadatos básicos.
#
# El parámetro --order controla cuántos símbolos previos usamos como "contexto":
#   order=0 → sin contexto, solo adaptativo normal
#   order=1 → mira el píxel anterior
#   order=2 → mira los 2 píxeles anteriores, etc.
#
# Usamos suavizado de Laplace (todos empiezan con frecuencia 1) para evitar
# problemas cuando aparece un contexto nuevo por primera vez.
'''

import io
import numpy as np
import main
import logging
with open("/tmp/description.txt", 'w') as f:  # Archivo temporal para el parser
    f.write(__doc__)
import parser

# Añadimos el parámetro --order al parser
default_order = 0
parser.parser_encode.add_argument("--order", type=int, help=f"Context model order (default: {default_order})", default=default_order)
parser.parser_decode.add_argument("--order", type=int, help=f"Context model order (default: {default_order})", default=default_order)

import entropy_image_coding as EIC
import gzip
import pickle
import os
import math
from bitarray import bitarray
import heapq
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

@dataclass
class _Node:
    """Nodo del árbol de Huffman. Puede ser hoja (con símbolo) o nodo interno."""
    freq: int  # Frecuencia acumulada
    sym: Optional[int] = None  # Símbolo si es hoja, None si es nodo interno
    left: Optional["_Node"] = None  # Hijo izquierdo
    right: Optional["_Node"] = None  # Hijo derecho


def _build_huffman_tree_from_freq(freqs: Dict[int, int]) -> _Node:
    """
    Construye el árbol de Huffman a partir de las frecuencias de los símbolos.
    Usa un heap (cola de prioridad) para combinar los nodos de menor frecuencia.

    """
    heap: List[Tuple[int, int, _Node]] = []
    uid = 0
    
    # Creamos una hoja por cada símbolo
    for sym, f in freqs.items():
        heapq.heappush(heap, (f, uid, _Node(freq=f, sym=sym)))
        uid += 1

    if not heap:
        raise ValueError("Empty frequency table")

    # si solo hay 1 símbolo, necesitamos al menos 1 bit,y
    # para que el código tenga al menos 1 bit
    if len(heap) == 1:
        f, _, only = heap[0]
        dummy = _Node(freq=0, sym=None)
        return _Node(freq=f, left=dummy, right=only)

    # Combinamos los dos nodos de menor frecuencia hasta tener uno solo (la raíz)
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)  # Nodo con menor frecuencia
        f2, _, n2 = heapq.heappop(heap)  # Segundo nodo con menor frecuencia
        parent = _Node(freq=f1 + f2, left=n1, right=n2)  # Combinamos
        heapq.heappush(heap, (parent.freq, uid, parent))
        uid += 1

    return heap[0][2]


def _build_codebook(root: _Node) -> Dict[int, bitarray]:
    """
    Recorre el árbol y genera los códigos: izquierda=0, derecha=1

    """
    codes: Dict[int, bitarray] = {}

    def dfs(node: _Node, path: bitarray) -> None:
        # Llegamos a una hoja y guardamos el código
        if node.sym is not None:
            if len(path) == 0:
                codes[node.sym] = bitarray("0")  # caso 1 símbolo
            else:
                codes[node.sym] = path.copy()
            return
        
        # Nodo interno → seguir bajando
        assert node.left is not None and node.right is not None
        path.append(False)  # Izquierda = 0
        dfs(node.left, path)
        path.pop()
        path.append(True)   # Derecha = 1
        dfs(node.right, path)
        path.pop()

    dfs(root, bitarray())
    return codes


def _decode_one_symbol(bits: bitarray, start_idx: int, root: _Node) -> Tuple[int, int]:
    """
    Lee bits hasta llegar a una hoja del árbol. Devuelve el símbolo
    decodificado y dónde seguir leyendo.
    """
    node = root
    i = start_idx
    
    # Navegamos por el árbol hasta encontrar una hoja
    while node.sym is None:
        if i >= len(bits):
            raise ValueError("Truncated bitstream while decoding")
        b = bits[i]  # Leemos el siguiente bit
        i += 1
        node = node.right if b else node.left  # 1=derecha, 0=izquierda
        if node is None:
            raise ValueError("Invalid bitstream (null branch)")
    
    return node.sym, i


class _ContextModel:
    """
    Modelo probabilístico que mantiene contadores de frecuencia por contexto.
    El contexto son los N símbolos anteriores (N = order).
    Usa suavizado de Laplace: todos los símbolos empiezan con frecuencia 1.
    """
    PAD = 256  # relleno inicial del contexto

    def __init__(self, order: int, alphabet_size: int = 256) -> None:
        if order < 0:
            raise ValueError("order must be >= 0")
        self.order = order  # Número de símbolos previos que forman el contexto
        self.alphabet = list(range(alphabet_size))  # Alfabeto de símbolos (0-255)
        self._base = {s: 1 for s in self.alphabet}  # Suavizado: todos empiezan con frec=1
        self._counts: Dict[Tuple[int, ...], Dict[int, int]] = {}  # Contadores por contexto

    def ctx_init(self) -> List[int]:
        return [self.PAD] * self.order

    def freqs(self, ctx: Tuple[int, ...]) -> Dict[int, int]:
        """Frecuencias para este contexto. Si es nuevo, lo inicializamos."""
        d = self._counts.get(ctx)
        if d is None:
            # Primera vez que vemos este contexto: inicializamos con suavizado
            d = dict(self._base)
            self._counts[ctx] = d
        return d

    def update(self, ctx: Tuple[int, ...], sym: int) -> None:
        """Incrementa el contador del símbolo."""
        d = self.freqs(ctx)
        d[sym] += 1


class CoDec(EIC.CoDec):

    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        self.file_extension = ".huf"

        # Obtenemos el orden del modelo desde los argumentos
        self.order = int(getattr(args, "order", 0))
        if self.order < 0:
            raise ValueError("order must be >= 0")

    def compress_fn(self, img, fn):
        logging.debug(f"trace img.shape={img.shape}, dtype={img.dtype}")
        tree_fn = f"{fn}_adaptive_huffman_tree.pkl.gz"
        compressed_img = io.BytesIO()

        # Aplanar imagen a bytes
        flattened = img.flatten().astype(np.uint8).tolist()

        # Creamos el modelo adaptativo
        model = _ContextModel(order=self.order, alphabet_size=256)
        ctx_list = model.ctx_init()  # Inicializamos el contexto

        out_bits = bitarray(endian="big")  # Aquí guardamos todos los bits codificados

        # Codificar cada píxel
        for s in flattened:
            # Contexto = últimos N símbolos
            ctx = tuple(ctx_list) if self.order > 0 else tuple()

            # Obtenemos las frecuencias actuales para este contexto
            freqs = model.freqs(ctx)
            tree = _build_huffman_tree_from_freq(freqs)
            codebook = _build_codebook(tree)

            # Añadimos el código del símbolo al bitstream
            out_bits.extend(codebook[int(s)])

            # Actualizar frecuencias y deslizar contexto
            model.update(ctx, int(s))

            if self.order > 0:
                ctx_list.pop(0)
                ctx_list.append(int(s))

        # Escribimos los bits como bytes en el archivo comprimido
        compressed_img.write(out_bits.tobytes())

        # Guardamos los metadatos: dimensiones, orden y número exacto de bits
        logging.debug(f"Saving {tree_fn}")
        with gzip.open(tree_fn, 'wb') as f:
            np.save(f, img.shape)  # Dimensiones de la imagen original
            pickle.dump(
                {
                    "order": self.order,
                    "nbits": len(out_bits)
                },
                f
            )

        tree_length = os.path.getsize(tree_fn)
        logging.info(f"Length of the file \"{tree_fn}\" (metadata) = {tree_length} bytes")

        return compressed_img

    def compress(self, img, fn="/tmp/encoded"):
        return self.compress_fn(img, fn)

    def decompress_fn(self, compressed_img, fn):
        tree_fn = f"{fn}_adaptive_huffman_tree.pkl.gz"
        compressed_img = io.BytesIO(compressed_img)

        # Cargar metadatos
        with gzip.open(tree_fn, 'rb') as f:
            shape = np.load(f)
            tree = pickle.load(f)

        order = int(tree["order"])
        nbits = int(tree["nbits"])

        # Cargar y recortar padding
        bits = bitarray(endian="big")
        bits.frombytes(compressed_img.read())

        # Quitamos los bits de padding que se añadieron al final
        if nbits < len(bits):
            bits = bits[:nbits]

        total_symbols = int(math.prod(shape))  # Cuántos píxeles tenemos que decodificar

        # Creamos el mismo modelo que en la codificación
        model = _ContextModel(order=order, alphabet_size=256)
        ctx_list = model.ctx_init()

        decoded = []
        idx = 0

        # Decodificar símbolo por símbolo (mismo proceso que encoder)
        for _ in range(total_symbols):

            ctx = tuple(ctx_list) if order > 0 else tuple()

            # Reconstruimos el árbol con las mismas frecuencias
            freqs = model.freqs(ctx)
            tree = _build_huffman_tree_from_freq(freqs)

            sym, idx = _decode_one_symbol(bits, idx, tree)
            decoded.append(sym)

            # Actualizamos el modelo igual que hizo el codificador
            model.update(ctx, sym)

            if order > 0:
                ctx_list.pop(0)
                ctx_list.append(sym)

        # Reconstruimos la imagen con las dimensiones originales
        img = np.array(decoded, dtype=np.uint8).reshape(shape)
        return img

    def decompress(self, compressed_img, fn="/tmp/encoded"):
        return self.decompress_fn(compressed_img, fn)


if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
