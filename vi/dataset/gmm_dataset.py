from .dataset import TestDataset
import numpy as np


class GmmDataset(TestDataset):
    def __init__(self, n_latent, n_real, proportions, gmm_centroids, gmm_variances, total_size):
        transformation_matrix = np.random.randint(-10, 10, size=(n_real, n_latent))
        n_samples_class = total_size * proportions.reshape(-1, 1)
        X = np.zeros((total_size, n_real))
        labels = np.zeros(total_size)
        i = 0
        for component in range(n_samples_class.shape[0]):
            for n_exp in range(int(n_samples_class[component][0])):
                data_point = np.random.normal(gmm_centroids[component, :], np.sqrt(gmm_variances[component, :]))
                data_point = np.matmul(transformation_matrix, data_point)
                X[i, :] = data_point
                labels[i] = component
                i += 1
        super(GmmDataset, self).__init__(X, labels)
