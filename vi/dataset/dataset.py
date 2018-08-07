# -*- coding: utf-8 -*-

"""Handling datasets.
For the moment, is initialized with a torch Tensor of size (n_cells, nb_genes)"""
import os
import urllib.request

import numpy as np
import scipy.sparse as sp_sparse
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class TestDataset(Dataset):
    """Gene Expression dataset. It deals with:
    - log_variational expression -> torch.log(1 + X)
    - local library size normalization (mean, var) per batch
    """

    def __init__(self, X, labels):
        self.X = X
        self.dense = type(self.X) is np.ndarray
        self.labels, self.n_labels = labels, len(np.unique(labels))
        self.total_size = X.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return idx

    def download_and_preprocess(self):
        self.download()
        return self.preprocess()

    def collate_fn(self, batch):
        indexes = np.array(batch)
        X = torch.FloatTensor(self.X[indexes]) if self.dense else torch.FloatTensor(self.X[indexes].toarray())
        return X, torch.LongTensor(self.labels[indexes])
