import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous

    @property
    def centroids(self):
        return self._centroids

    @property
    def _class_labels(self):
        return self._class_labels

    def predict(self, Xts):
        """

        Parameters
        ----------
        xts

        Returns
        -------
        predicted_label --> actually the predicted label for the given data

        """

        if self._centroids is None:
            raise ValueError("The classifier is not trained. Call fit!")

        dist_euclidean = euclidean_distances(Xts, self._centroids)
        idx_min = np.argmin(dist_euclidean, axis=1)
        yc = self._class_labels[idx_min]
        return yc

    def fit(self, xtr, ytr):
        n_dim = xtr.shape[1]
        self._class_labels = np.unique(ytr)  # etichette uniche
        n_classes = self._class_labels.size  # numero di classi
    
        # Inizializziamo i centroidi con zeri
        self._centroids = np.zeros(shape=(n_classes, n_dim))
    
        # Calcoliamo il centroide per ogni classe
        for idx, label in enumerate(self._class_labels):
            self._centroids[idx, :] = np.mean(xtr[ytr == label, :], axis=0)