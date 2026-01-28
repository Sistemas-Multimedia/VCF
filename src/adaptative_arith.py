'''Codificación de Entropía de imágenes usando Adaptive Arithmetic Coding'''
import io
import sys
import numpy as np
from bitarray import bitarray
import logging

with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)
    
import main
import parser
import entropy_image_coding as EIC

class AdaptiveModel:
    """
    Modelo de probabilidad que aprende sobre la marcha.
    Mantiene las Frecuencias Acumuladas (CDF).
    """
    def __init__(self, num_symbols=256, max_freq=16384):
        self.num_symbols = num_symbols
        self.max_freq = max_freq
        
        # Inicializamos todo a 1
        self.freqs = [1] * num_symbols
        # Cumulative freqs: cumulative[i] es el inicio del rango del simbolo i
        # cumulative[i+1] es el final del rango del simbolo i
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
        
        # Si el total supera el máximo, dividimos todo por 2
        if self.total >= self.max_freq:
            for i in range(self.num_symbols):
                # Nos aseguramos de no bajar nunca a 0
                self.freqs[i] = (self.freqs[i] >> 1) + 1
        
        self._update_cumulative()

    def get_range(self, symbol):
        """Devuelve (low, high, total) para un símbolo"""
        return (self.cumulative[symbol], 
                self.cumulative[symbol+1], 
                self.total)

    def get_symbol_from_scaled_value(self, scaled_value):
        for i in range(self.num_symbols):
            if scaled_value < self.cumulative[i+1]:
                return i, self.cumulative[i], self.cumulative[i+1]
        return -1, 0, 0

class CoDec(EIC.CoDec):
    def __init__(self, args):
        super().__init__(args)
        self.file_extension = ".adpt_arith"
        self.BITS = 32
        self.MAX = 1 << self.BITS
        self.HALF = self.MAX >> 1
        self.QUARTER = self.HALF >> 1
        self.THREE_QUARTER = self.QUARTER * 3

    def compress_fn(self, img, fn):
        compressed_img = io.BytesIO()
        
        flat_img = img.flatten().astype(np.int32)
        
        # 1. Num Dimensiones (uint32)
        # 2. Shape (N * uint32)
        shape = np.array(img.shape, dtype=np.uint32)
        num_dims = np.array([len(img.shape)], dtype=np.uint32)
        
        compressed_img.write(num_dims.tobytes())
        compressed_img.write(shape.tobytes())

        logging.info(f"Codificando Adaptativamente {len(flat_img)} símbolos...")
        bit_stream = self._encode(flat_img)
        
        compressed_img.write(bit_stream.tobytes())
        
        logging.info(f"Tamaño comprimido: {len(bit_stream)} bits")
        return compressed_img

    def decompress_fn(self, compressed_bytes, fn):
        buffer = io.BytesIO(compressed_bytes)
                
        # 1. Leer número de dimensiones (4 bytes)
        try:
            num_dims_data = buffer.read(4)
            if not num_dims_data: raise ValueError("Fichero vacío o corrupto")
            num_dims = np.frombuffer(num_dims_data, dtype=np.uint32)[0]
            
            # 2. Leer la forma (shape) basada en num_dims (4 bytes * num_dims)
            shape_data = buffer.read(4 * int(num_dims))
            shape = tuple(np.frombuffer(shape_data, dtype=np.uint32))
        except Exception as e:
            logging.error(f"Error leyendo cabecera: {e}")
            return np.zeros((10,10), dtype=np.uint8)

        num_symbols = int(np.prod(shape))
        
        # 3. Leer el resto como bitstream
        bit_stream = bitarray(endian='big')
        bit_stream.frombytes(buffer.read())

        logging.info(f"Decodificando Adaptativamente {num_symbols} símbolos con shape {shape}...")
        
        decoded_data = self._decode(bit_stream, num_symbols)

        return np.array(decoded_data).reshape(shape).astype(np.uint8)

    def _encode(self, data):
        low = 0
        high = self.MAX
        pending_bits = 0
        output = bitarray(endian='big')
        
        model = AdaptiveModel()

        for i, symbol in enumerate(data):
            if i % 50000 == 0: sys.stdout.write(f"\rEnc: {i}")
            
            s_low, s_high, total_freq = model.get_range(symbol)
            range_ = high - low
            
            high = low + (range_ * s_high) // total_freq
            low  = low + (range_ * s_low)  // total_freq

            while True:
                if high <= self.HALF:
                    self._emit_bit(output, 0, pending_bits)
                    pending_bits = 0
                elif low >= self.HALF:
                    self._emit_bit(output, 1, pending_bits)
                    pending_bits = 0
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.QUARTER and high <= self.THREE_QUARTER:
                    pending_bits += 1
                    low -= self.QUARTER
                    high -= self.QUARTER
                else:
                    break
                low *= 2
                high *= 2
            
            model.update(symbol)
        
        pending_bits += 1
        if low < self.QUARTER:
            self._emit_bit(output, 0, pending_bits)
        else:
            self._emit_bit(output, 1, pending_bits)
            
        print("") 
        return output

    def _decode(self, bits, num_symbols):
        low = 0
        high = self.MAX
        value = 0
        
        bit_list = bits.tolist()
        bit_iter = iter(bit_list)
        
        for _ in range(self.BITS):
            value <<= 1
            if next(bit_iter, 0): value |= 1

        decoded = bytearray(num_symbols)
        model = AdaptiveModel()
        
        for i in range(num_symbols):
            if i % 10000 == 0: sys.stdout.write(f"\rDec: {i}/{num_symbols}")
            
            range_ = high - low
            total_freq = model.total
            
            # Usamos offset+1
            offset = value - low
            scaled_value = ((offset + 1) * total_freq - 1) // range_
            
            if scaled_value >= total_freq:
                scaled_value = total_freq - 1

            symbol, s_low, s_high = model.get_symbol_from_scaled_value(scaled_value)
            
            if symbol == -1:
                logging.error(f"FATAL: Símbolo no encontrado para valor {scaled_value} (Total: {total_freq})")
                break

            decoded[i] = symbol
            
            high = low + (range_ * s_high) // total_freq
            low  = low + (range_ * s_low)  // total_freq
            
            while True:
                if high <= self.HALF:
                    pass
                elif low >= self.HALF:
                    low -= self.HALF
                    high -= self.HALF
                    value -= self.HALF
                elif low >= self.QUARTER and high <= self.THREE_QUARTER:
                    low -= self.QUARTER
                    high -= self.QUARTER
                    value -= self.QUARTER
                else:
                    break
                
                low += low
                high += high
                value += value
                
                try:
                    if next(bit_iter): value |= 1
                except StopIteration:
                    pass
            
            model.update(symbol)

        print("")
        return decoded

    def _emit_bit(self, output, bit, pending):
        output.append(bit)
        if pending:
            output.extend([not bit] * pending)

    def compress(self, img, fn="/tmp/encoded"): return self.compress_fn(img, fn)
    def decompress(self, compressed_img, fn="/tmp/encoded"): return self.decompress_fn(compressed_img, fn)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)