'''Scalar Uniform Quantization.'''

import logging
#import logging_config
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(levelname)s probando %(funcName)s()] %(message)s")
##logger.setLevel(logging.CRITICAL)
##logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

import numpy as np

class Quantizer(object):

    def __init__(self, Q_step=1, min_val=-128, max_val=128):
        assert Q_step > 0
        self.min_val = min_val
        self.max_val = max_val
        self.Q_step = Q_step
        logger.info(f"min={min_val} max={max_val} Q_step={Q_step}")
        
    def encode_and_decode(self, x):
        logger.debug(f"x.shape={x.shape}")
        k = self.encode(x)
        logger.debug(f"k.shape={k.shape}")
        y = self.decode(k)
        logger.debug(f"y.shape={y.shape}")
        return y, k

    def get_decision_levels(self):
        range_input_values = np.linspace(start=self.min_val + 1, stop=self.max_val - 1, num=self.max_val - self.min_val - 1)#.astype(np.uint8)
        #print("range_input_values =", range_input_values)
        range_input_indexes = self.encode(range_input_values)
        #print("range_input_indexes =", range_input_indexes)
        DPCM = np.diff(range_input_indexes)
        #print("DPCM =", DPCM)
        decision_levels_indexes = np.where(DPCM != 0)
        #print("decision_levels_indexes =", decision_levels_indexes)
        decision_levels = range_input_values[decision_levels_indexes]
        #print("decision_levels =", decision_levels)
        extended_decision_levels = np.append(self.min_val, decision_levels)
        extended_decision_levels = np.append(extended_decision_levels, self.max_val)
        return extended_decision_levels

    def get_representation_levels(self):
        #quantization_indexes = np.linspace(start=self.min_val//self.Q_step, stop=(self.max_val + 1)//self.Q_step - 1, num=(self.max_val - self.min_val + 1)//self.Q_step)#.astype(np.uint8)
        quantization_indexes = self.encode(self.get_decision_levels())
        representation_levels = self.decode(quantization_indexes)
        return representation_levels
