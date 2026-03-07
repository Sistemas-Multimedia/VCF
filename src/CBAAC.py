'''Context-based adaptive arithmetic coding'''

import io
import sys
import numpy as np
from bitarray import bitarray
import logging
import argparse
from collections import deque  
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)  
import main
import parser
import entropy_image_coding as EIC
from arithmetic_coding.arithmetic_coding import Arithmetic_Coding, Arithmetic_Encoding, Arithmetic_Decoding

class AdaptiveModel:
    def __init__(self, num_symbols=256, max_freq=16384):
        self.num_symbols = num_symbols
        self.max_freq = max_freq
        self.freqs = [1] * num_symbols
        self.cumulative = [0] * (num_symbols + 1)
        self._update_cumulative()

    def _update_cumulative(self):
        cum = 0
        for i in range(self.num_symbols):
            self.cumulative[i] = cum
            cum += self.freqs[i]
        self.cumulative[self.num_symbols] = cum
        self.total = cum

    def update(self, symbol):
        self.freqs[symbol] += 1
        if self.total >= self.max_freq:
            for i in range(self.num_symbols):
                self.freqs[i] = (self.freqs[i] >> 1) + 1
        self._update_cumulative()

    def get_range(self, symbol):
        return (self.cumulative[symbol], self.cumulative[symbol+1], self.total)

    def get_symbol_from_scaled_value(self, scaled_value):
        for i in range(self.num_symbols):
            if scaled_value < self.cumulative[i+1]:
                return i, self.cumulative[i], self.cumulative[i+1]
        return -1, 0, 0

class ContextManager:
    def __init__(self, order=0):
        self.order = order
        # Usamos un DICCIONARIO para almacenar solo los contextos que existen.
        # Key: Tupla con los símbolos anteriores (ej: (128, 255))
        # Value: Instancia de AdaptiveModel
        self.models = {}
        
        self.fallback_model = AdaptiveModel() 
        if self.order == 0:
            self.models[()] = self.fallback_model

    def get_model(self, history):
        # history debe ser una tupla inmutable para servir de clave en el dict
        history_key = tuple(history)
        
        if history_key not in self.models:
            # Si es la primera vez que vemos este contexto, creamos un modelo nuevo
            self.models[history_key] = AdaptiveModel()
            
        return self.models[history_key]


class CoDec(EIC.CoDec):
    def __init__(self, args):
        super().__init__(args)
        self.file_extension = ".adpt_arith"
        self.ORDER = getattr(args, 'order', 0)
        #self.low = 0
        #self.high = Arithmetic_Coding.MAX
        #self.pending_bits = 0

    def compress_fn(self, img, fn):
        compressed_img = io.BytesIO()
        flat_img = img.flatten().astype(np.int32)
        
        shape = np.array(img.shape, dtype=np.uint32)
        num_dims = np.array([len(img.shape)], dtype=np.uint32)
        compressed_img.write(num_dims.tobytes())
        compressed_img.write(shape.tobytes())

        logging.info(f"Encoding [Order {self.ORDER}] {len(flat_img)} symbols ...")
        bit_stream = self._encode(flat_img)
        
        compressed_img.write(bit_stream.tobytes())
        logging.info(f"Compressed size: {len(bit_stream)} bits")
        return compressed_img

    def decompress_fn(self, compressed_bytes, fn):
        buffer = io.BytesIO(compressed_bytes)
        try:
            num_dims = np.frombuffer(buffer.read(4), dtype=np.uint32)[0]
            shape = tuple(np.frombuffer(buffer.read(4 * int(num_dims)), dtype=np.uint32))
        except:
            return np.zeros((10,10), dtype=np.uint8)

        num_symbols = int(np.prod(shape))
        bit_stream = bitarray(endian='big')
        bit_stream.frombytes(buffer.read())

        logging.info(f"Decoding [Order {self.ORDER}] {num_symbols} symbols ...")
        decoded_data = self._decode(bit_stream, num_symbols)

        return np.array(decoded_data).reshape(shape).astype(np.uint8)

    def _encode(self, data):
        encoder = Arithmetic_Encoding()
        output = bitarray(endian='big')
        ctx_manager = ContextManager(order=self.ORDER)
        # Inicializamos con ceros 
        history = deque([0] * self.ORDER, maxlen=self.ORDER if self.ORDER > 0 else 1)
        for i, symbol in enumerate(data):
            if i % 50000 == 0: sys.stdout.write(f"\rEnc: {i}")
            ctx_key = tuple(history) if self.ORDER > 0 else ()
            model = ctx_manager.get_model(ctx_key)
            encoder.encode_symbol(symbol, model, output)
            # 3. Actualizar modelo y mover la historia
            model.update(symbol)
            if self.ORDER > 0:
                history.append(symbol) 
        encoder.flush(output)
        print("") 
        return output

    def _decode(self, bits, num_symbols):
        decoder = Arithmetic_Decoding()
        decoder.start(bits)
        decoded = bytearray(num_symbols)
        ctx_manager = ContextManager(order=self.ORDER)
        # Historial idéntico al encoder
        history = deque([0] * self.ORDER, maxlen=self.ORDER if self.ORDER > 0 else 1)
        for i in range(num_symbols):
            if i % 10000 == 0: sys.stdout.write(f"\rDec: {i}/{num_symbols}")
            ctx_key = tuple(history) if self.ORDER > 0 else ()
            model = ctx_manager.get_model(ctx_key)
            symbol = decoder.decode_symbol(model)
            decoded[i] = symbol
            model.update(symbol)
            if self.ORDER > 0:
                history.append(symbol)
        print("")
        return decoded

    def compress(self, img, fn="/tmp/encoded"):
        return self.compress_fn(img, fn)

    def decompress(self, compressed_img, fn="/tmp/encoded"):
        return self.decompress_fn(compressed_img, fn)

if __name__ == "__main__":
    try:
        parser.parser_encode.add_argument('--order', type=int, default=0, help='Context order')
    except: pass
    try:
        parser.parser_decode.add_argument('--order', type=int, default=0, help='Context order')
    except: pass
        
    main.main(parser.parser, logging, CoDec)
