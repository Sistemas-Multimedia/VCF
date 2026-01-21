'''Companded (using mu-law) scalar quantization.'''

import numpy as np
from .quantization import Quantizer
from . import deadzone_quantization as deadzone

name = "companded"

class Companded_Quantizer(deadzone.Deadzone_Quantizer):

    def muLaw_compress(self, x, mu):
        return np.log(1+mu*np.abs(x))/np.log(1+mu)*np.sign(x)

    def muLaw_expand(self, y, mu):
        return (1/mu)*(((1+mu)**np.abs(y))-1)*np.sign(y)

    def encode(self, x):
        '''Companded mu-law deadzone quantizer'''
        mu = 255
        x_compressed = (32768*(self.muLaw_compress(x/32768, mu)))
        k = super().encode(x_compressed).astype(np.int16) # Ojo, sobra astype
        return k

    def decode(self, k):
        '''Companded mu-law deadzone dequantizer'''
        mu = 255
        z_compressed = super().decode(k)
        y = np.round(32768*self.muLaw_expand(z_compressed/32768, mu)).astype(np.int16) # Ojo, lo mismo sobra el astype
        return y
