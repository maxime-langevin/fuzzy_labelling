# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from vi.models.modules import Encoder, Decoder
from vi.models.utils import one_hot
import torch.nn
torch.backends.cudnn.benchmark = True


# VAE model
class VAE(nn.Module):
    r"""Variational auto-encoder model.

    Args:
        :n_input: Number of input genes.
        :n_batch: Default: ``0``.
        :n_labels: Default: ``0``.
        :n_hidden: Number of hidden. Default: ``128``.
        :n_latent: Default: ``1``.
        :n_layers: Number of layers. Default: ``1``.
        :dropout_rate: Default: ``0.1``.
        :dispersion: Default: ``"gene"``.
        :log_variational: Default: ``True``.
        :reconstruction_loss: Default: ``"zinb"``.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels, use_cuda=True )

    """

    def __init__(self, n_input, n_labels=0, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1,
                 reconstruction_loss="gaussian"):
        super(VAE, self).__init__()
        self.n_latent = n_latent
        self.reconstruction_loss = reconstruction_loss
        # Automatically desactivate if useless
        self.n_labels = n_labels
        self.n_latent_layers = 1

        self.z_encoder = Encoder(n_input, n_latent, n_layers=n_layers, n_hidden=n_hidden,
                                 dropout_rate=dropout_rate)
        self.decoder = Decoder(n_latent, n_input, n_layers=n_layers, n_hidden=n_hidden,
                                   dropout_rate=dropout_rate)

    def get_latents(self, x, y=None):
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(self, x, y=None):
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        if not self.training:
            z = qz_m
        return z

    def _reconstruction_loss(self, x, qx_m, qx_v):

        # Reconstruction Loss
        if self.reconstruction_loss == 'gaussian':
            criterion = nn.MSELoss()
            reconst_loss = criterion(qx_m, x)
            reconst_loss = -torch.sum(Normal(qx_m, torch.sqrt(qx_v)).log_prob(x), dim=1)
        return reconst_loss

    def forward(self, x):
        # Parameters for z latent distribution
        x_ = x

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)

        qx_m, qx_v = self.decoder(z)

        reconst_loss = self._reconstruction_loss(x, qx_m, qx_v)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
        kl_divergence = kl_divergence_z

        return reconst_loss, kl_divergence
