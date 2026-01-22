'''Lloyd-Max (scalar) quantization.'''

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
from scipy.ndimage import uniform_filter1d
from .quantization import Quantizer
#import warnings

name = "Lloyd-Max"

class LloydMax_Quantizer(Quantizer):

    def __init__(self, Q_step, counts, min_val=0, max_val=255, max_iters=10):
        '''Defines the working parameters of the quantizer:

        Q_step: quantization step size.

        counts: number of ocurrences of each possible input
        sample. 
        
        [min_val, max_val]: expected dynamic range of the input
        signal.

        max_iters: maximum number of iterations.

        '''
        super().__init__(Q_step, min_val, max_val)
        #assert np.all(counts) > 0
        #self.N_bins = (max_val + 1 - min_val) // Q_step
        self.N_bins = int(np.ceil((max_val + 1 - min_val) / Q_step))
        #initial_boundaries = np.linspace(min_val, max_val + 1, self.N_bins + 1)
        total_count = np.sum(counts)
        bin_count = total_count/self.N_bins
        initial_boundaries = [float(min_val)] #initial_boundaries = [0.]
        acc = 0
        counter = 0
        for p in counts:
            acc += p
            counter += 1
            if acc > bin_count:
                initial_boundaries.append(float(counter))
                acc = 0
        initial_boundaries.append(float(max_val + 1)) #initial_boundaries.append(256.)
        initial_boundaries = np.array(initial_boundaries)
        self.boundaries = initial_boundaries
        logger.info(f"initial_boundaries={self.boundaries}")
        initial_centroids = 0.5 * (initial_boundaries[1:] + initial_boundaries[:-1])
        self.centroids = initial_centroids
        logger.info(f"initial_centroids={self.centroids}")
        #prev_b = np.zeros(self.boundaries.size)
        for j in range(max_iters):
            #prev_b[:] = self.boundaries
            self._compute_boundaries()
            #logger.debug(f"len(prev_b)={len(prev_b)} len(boundaries)={len(self.boundaries)}")
            #logger.debug(f"prev_b={prev_b} boundaries={self.boundaries}")
            #max_abs_error = np.max(np.abs(prev_b - self.boundaries))
            prev_c = self.centroids
            end = self._compute_centroids(counts)
            if end:
                self.centroids = prev_c
                break
        logger.info(f"centroids={self.centroids}")

    def _compute_boundaries(self):
        '''The new boundaries are in the middle points between the
        current centroids.

        '''
        logger.debug(f"centroids={self.centroids}")
        self.boundaries = uniform_filter1d(self.centroids, size=2, origin=-1)[:-1]
        #self.boundaries = np.concatenate(([0], self.boundaries, [256]))
        self.boundaries = np.concatenate(
            ([self.min_val], self.boundaries, [self.max_val + 1])
        )

        logger.debug(f"boundaries={self.boundaries}")
        logger.debug(f"len(centroids)={len(self.centroids)} len(bondaries)={len(self.boundaries)}")

    def _compute_centroids(self, counts):
        '''Compute the centroid of each bin.'''
        end = False
        centroids = []
        bin_size = self.Q_step
        logger.info(f"bin_size={bin_size}")
        for i in range(self.N_bins):
            #b_i = i*bin_size
            #b_i_1 = (i+1)*bin_size
            b_i   = self.min_val + i * bin_size
            b_i_1 = self.min_val + (i+1) * bin_size
            logger.debug(f"b_i={b_i} b_i_1={b_i_1}")
            if b_i == b_i_1:
                end = True
                break
            # See from scipy.ndimage import center_of_mass
            #mass = np.sum([j*counts[j] for j in range(b_i, b_i_1)])
            mass = np.sum([(j) * counts[j - self.min_val] for j in range(b_i, b_i_1)])
            #total_counts_in_bin = np.sum([counts[j] for j in range(b_i, b_i_1)])
            total_counts_in_bin = np.sum([counts[j - self.min_val] for j in range(b_i, b_i_1)])

            centroid = mass/total_counts_in_bin
            logger.debug(f"centroid={centroid}")
            centroids.append(centroid)
        self.centroids = np.array(centroids)
        logger.info(f"centroids={self.centroids}")
        return end

    def encode(self, x):
        '''Find the quantization indexes for the signal <x>.

        '''
        logger.debug(f"x.shape={x.shape}")
        k = np.searchsorted(self.boundaries, x, side="right") - 1
        logger.debug(f"k.shape={k.shape}")
        logger.debug(f"max(k)={np.max(k)}")
        return k

    def decode(self, k):
        '''Return the centroids corresponding to the quantization indexes
        <k>.

        '''
        logger.debug(f"k.shape={k.shape}")
        logger.debug(f"min(k)={np.min(k)}, max(k)={np.max(k)}")
        logger.debug(f"centroids.shape={self.centroids.shape}")
        y = self.centroids[k]
        logger.debug(f"y.shape={y.shape}")
        return y

    def get_representation_levels(self):
        '''In a Lloyd-Max quantizer the representation levels are the
        centroids.

        '''
        return self.centroids
    
    def set_representation_levels(self, centroids):
        '''Set the centroids to provide decoding.'''
        self.centroids = centroids

    def fit(self, x):
        pass
