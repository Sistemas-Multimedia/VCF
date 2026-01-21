'''KMeans scalar quantization.'''

import logging
import logging_config
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(levelname)s probando %(funcName)s()] %(message)s")
##logger.setLevel(logging.CRITICAL)
##logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

import numpy as np
#from sklearn import cluster
from sklearn.cluster import KMeans
from .quantization import Quantizer
import warnings

name = "KMeans"

class KMeans_Quantizer(Quantizer):

    def __init__(self, Q_step, counts, min_val=0, max_val=255):
        '''Creates a KMeans clusterer using the histogram
        <counts>.

        Q_step: quantization step size.

        counts: number of ocurrences of each possible input
        sample. Used only for an initial generation of the centroids.

        [min_val, max_val]: expected dynamic range of the input
        signal.

        '''
        super().__init__(Q_step, min_val, max_val)
        N_bins = (max_val + 1 - min_val) // Q_step
        total_count = np.sum(counts)
        bin_count = total_count/N_bins
        initial_boundaries = [0.]
        acc = 0
        counter = 0
        for p in counts:
            acc += p
            counter += 1
            if acc > bin_count:
                initial_boundaries.append(float(counter))
                acc = 0
        initial_boundaries.append(256.)
        initial_boundaries = np.array(initial_boundaries).reshape(-1, 1)
        initial_centroids = 0.5 * (initial_boundaries[1:] + initial_boundaries[:-1])
        self.N_bins = len(initial_centroids)
        logger.info(f"initial_centroids={initial_centroids.squeeze()}")
        self.clusterer = KMeans(n_clusters=self.N_bins, init=initial_centroids, n_init=1)
        #self.centroids = self.clusterer.cluster_centers_

    def fit(self, x):
        self.clusterer.fit(x)
        self._sort_labels()
        self.centroids = self.clusterer.cluster_centers_
        logger.debug(f"centroids.shape={self.centroids.shape}")

    def _sort_labels(self):
        centroids = self.clusterer.cluster_centers_.squeeze()
        idx = np.argsort(self.clusterer.cluster_centers_.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(len(centroids))
        argsort_lut = np.argsort(lut)
        sorted_centroids = centroids[argsort_lut]
        sorted_labels = lut[self.clusterer.labels_]
        centroids[:] = sorted_centroids
        self.clusterer.labels_ = sorted_labels
        logger.debug(f"labels.shape={self.clusterer.labels_}")

    def __sort_labels(self):
        self.centers = self.classifier.cluster_centers_.squeeze()
        idx = np.argsort(self.classifier.cluster_centers_.sum(axis=1))
        self.lut = np.zeros_like(idx)
        self.lut[idx] = np.arange(self.N_clusters)
        logger.info(f"lut={self.lut}")
        logger.info(f"centroids={self.centers}")
        argsort_lut = np.argsort(self.lut)
        sorted_centroids = self.centers[argsort_lut]
        logger.info(f"sorted_centroids={sorted_centroids}")
        logger.info(f"labels={self.classifier.labels_} len={len(self.classifier.labels_)}")
        sorted_labels = self.lut[self.classifier.labels_]
        logger.info(f"sorted_labels={sorted_labels} len={len(sorted_labels)}")
        self.centers[:] = sorted_centroids
        self.classifier.labels_ = sorted_labels
        
    def __train(self, x):
        flatten_x = shuffle(x.reshape((-1, 1)), random_state=0, n_samples=x.size)
        self.classifier.fit(flatten_x)
        self.centers = self.classifier.cluster_centers_.squeeze()
        idx = np.argsort(self.classifier.cluster_centers_.sum(axis=1))
        self.lut = np.zeros_like(idx)
        self.lut[idx] = np.arange(self.N_clusters)
        logger.info(f"lut={self.lut}")
        logger.info(f"centroids={self.centers}")
        argsort_lut = np.argsort(self.lut)
        sorted_centroids = self.centers[argsort_lut]
        logger.info(f"sorted_centroids={sorted_centroids}")
        logger.info(f"labels={self.classifier.labels_} len={len(self.classifier.labels_)}")
        sorted_labels = self.lut[self.classifier.labels_]
        logger.info(f"sorted_labels={sorted_labels} len={len(sorted_labels)}")
        self.centers[:] = sorted_centroids
        self.classifier.labels_ = sorted_labels

    def __retrain(self, x):
        '''Retrain the classifier using previous centers.'''
        if self.algorithm == "KMeans":
            self.classifier = KMeans(n_clusters=self.N_clusters, init=self.classifier.cluster_centers_, n_init=1)
        self.train(x)

    def encode(self, x):
        '''Use the created clusterer to find the quantization indexes
        (labels) for each point of <x>.

        '''
        k = self.clusterer.predict(x.reshape(-1, 1))
        #k.shape = x.shape
        k = k.reshape(x.shape)
        logger.debug(f"k.shape={k.shape}")
        return k

    def decode(self, k):
        '''Return the centroids corresponding to the quantization indexes
        <k>.

        '''
        y = self.centroids[k]
        logger.debug(f"k.shape={k.shape}")
        logger.debug(f"centroids.shape={self.centroids.shape}")
        logger.debug(f"centroids={self.centroids}")
        logger.debug(f"y.shape={y.shape}")
        return y[...,0]

    def get_representation_levels(self):
        '''In a Lloyd-Max quantizer the representation levels are the
        centroids computed by the clusterer.

        '''
        logger.debug(f"centroids.shape={self.centroids.shape}")
        return self.centroids
