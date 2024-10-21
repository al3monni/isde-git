import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous

    @property
    def centroids(self):
        return self._centroids

    def predict(self, Xts):

        if self._centroids is None:
            raise ValueError("The classifier is not trained. Call fit!")

        # Calcoliamo la distanza euclidea tra i dati di test e i centroidi
        dist_euclidean = euclidean_distances(Xts, self._centroids)

        # Troviamo l'indice del centroide più vicino
        idx_min = np.argmin(dist_euclidean, axis=1)

        # Selezioniamo le etichette delle classi corrispondenti al centroide più vicino
        yc = self._class_labels[idx_min]  # Usa _class_labels, non _classes
        return yc

    def fit(self, xtr, ytr):

        n_dim = xtr.shape[1]

        # Identifichiamo le classi uniche nei dati di allenamento
        self._class_labels = np.unique(ytr)  # Etichette uniche

        n_classes = self._class_labels.size  # Numero di classi

        # Inizializziamo i centroidi con zeri
        self._centroids = np.zeros(shape=(n_classes, n_dim))

        # Calcoliamo il centroide per ogni classe
        for idx, label in enumerate(self._class_labels):
            self._centroids[idx, :] = np.mean(xtr[ytr == label, :], axis=0)
