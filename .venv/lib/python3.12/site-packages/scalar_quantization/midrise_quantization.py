'''Mid-rise scalar quantization.'''

import numpy as np
from .quantization import Quantizer

name = "mid-rise"

class Midrise_Quantizer(Quantizer):
        
    def encode(self, x):
        k = np.floor(x / self.Q_step).astype(np.int)
        return k

    def decode(self, k):
        y = self.Q_step * (k + 0.5)
        return y

class Midrise_Quantizer2(Quantizer):
    def __init__(self, Q_step, min_val=0, max_val=255):
        super().__init__(Q_step, min_val, max_val)
        N_clusters = (self.max_val + 1 - self.min_val) // Q_step
        #self.decision_levels = np.linspace(min_val, max_val + 1, N_clusters + 1)
        self.centroids = 0.5 * (self.decision_levels[1:] + self.decision_levels[:-1])  # mean

    def quantize(self, x):
        k = np.searchsorted(self.decision_levels, x) - 1
        return k

    def dequantize(self, k):
        y = self.centroids[k]
        return y
