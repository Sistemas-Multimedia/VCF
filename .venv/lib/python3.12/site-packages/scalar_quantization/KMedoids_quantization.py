'''Lloyd-Max scalar quantization. Use K-Means or K-Medoids (depending
on a parameter) to compute the centers (an average in the case of
K-Means and the best input (a medoid) in the case of K-Medoids).'''

import logging
import logging_config
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(levelname)s probando %(funcName)s()] %(message)s")
##logger.setLevel(logging.CRITICAL)
##logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

import numpy as np
#from sklearn import cluster
from sklearn.utils import shuffle
from sklearn_extra.cluster import KMedoids
from .quantization import Quantizer
import warnings

name = "KMedoids"

class KMedoids_Quantizer(Quantizer):

    def __init__(self, Q_step, counts, min_val=0, max_val=255, metric='euclidean'):
        '''Creates a KMeans clusterer using the histogram
        <counts>.

        Q_step: quantization step size.

        counts: number of ocurrences of each possible input
        sample.

        [min_val, max_val]: expected dynamic range of the input
        signal.

        metric: the distance metric used for comparing the
        points. Accepted metrics are listed by
        sklearn.metrics.pairwise_distances_argmin(). It can be also a
        callable function that inputs two points and outputs a scalar
        (with the value of the distance between them).

        '''
        super().__init__(Q_step, min_val, max_val)
        self.N_clusters = (max_val + 1 - min_val) // Q_step
        self.clusterer = KMedoids(init="k-medoids++", n_clusters=self.N_clusters)

        '''
            sampled_x = shuffle(x.flatten(), random_state=0, n_samples=N_samples).reshape((-1, 1))
            self.classifier.fit(sampled_x)
            #self.train(shuffle(x.flatten(), random_state=0, n_samples=N_samples))
            #for i in range(x.shape[0]):
                #self.train(x[i])
                #self.classifier.fit(x[i].reshape((-1, 1)))
            self._sort_labels()
        '''

    def fit(self, x):
        self.clusterer.fit(x)
        self._sort_labels()
        self.medoids = self.clusterer.cluster_centers_

    def _sort_labels(self):
        medoids = self.clusterer.cluster_centers_.squeeze()
        idx = np.argsort(self.clusterer.cluster_centers_.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(len(medoids))
        argsort_lut = np.argsort(lut)
        sorted_medoids = medoids[argsort_lut]
        sorted_labels = lut[self.clusterer.labels_]
        medoids[:] = sorted_medoids
        self.clusterer.labels_ = sorted_labels

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
        '''Use the created classifier to find the quantization indexes
        (labels) for each point of <x>.

        '''
        k = self.clusterer.predict(x.reshape((-1, 1)))
        k.shape = x.shape
        return k

    def decode(self, k):
        '''Return the centroids corresponding to the quantization indexes
        <k>.

        '''
        y = self.medoids[k]
        return y

    def get_representation_levels(self):
        '''In a Lloyd-Max quantizer the representation levels are the
        centroids computed by the classifier (K-Means).

        '''
        return self.medoids
