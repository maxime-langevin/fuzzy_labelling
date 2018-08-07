import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score
from torch.nn import functional as F

from vi.dataset.data_loaders import DataLoaders
from vi.dataset.data_loaders import TrainTestDataLoaders, AlternateSemiSupervisedDataLoaders, \
    JointSemiSupervisedDataLoaders
from vi.metrics.classification import compute_accuracy, compute_accuracy_svc, compute_accuracy_rf, \
    unsupervised_classification_accuracy, compute_predictions
from vi.metrics.classification import unsupervised_clustering_accuracy
from vi.metrics.clustering import get_latent, entropy_batch_mixing

from . import Inference, ClassifierInference

plt.switch_backend('agg')


class VariationalInference(Inference):
    r"""The VariationalInference class for the unsupervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SVAEC``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.8``.
        :\**kwargs: Other keywords arguments from the general Inference class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = VariationalInference(gene_dataset, vae, train_size=0.5)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """
    default_metrics_to_monitor = ['ll']

    def __init__(self, model, gene_dataset, train_size=0.8, **kwargs):
        super(VariationalInference, self).__init__(model, gene_dataset, **kwargs)
        self.kl = None
        self.data_loaders = TrainTestDataLoaders(self.gene_dataset, train_size=train_size, use_cuda=self.use_cuda)

    def loss(self, tensors):
        sample_batch, _ = tensors
        reconst_loss, kl_divergence = self.model(sample_batch)
        loss = torch.mean(reconst_loss + self.kl_weight * kl_divergence)
        return loss

    def on_epoch_begin(self):
        self.kl_weight = self.kl if self.kl is not None else min(1, self.epoch / self.n_epochs)

    def ll(self, name, verbose=False):
        log_lkl = 0
        for i_batch, tensors in enumerate(self.data_loaders[name]):
            sample_batch,  labels = tensors
            reconst_loss, kl_divergence = self.model(sample_batch)
            log_lkl += torch.sum(reconst_loss).item()
        n_samples = (len(self.data_loaders[name].dataset)
                     if not (hasattr(self.data_loaders[name], 'sampler') and hasattr(self.data_loaders[name].sampler,
                                                                                     'indices')) else
                     len(self.data_loaders[name].sampler.indices))
        ll = log_lkl / n_samples
        if verbose:
            print("LL for %s is : %.4f" % (name, ll))
        return ll

    ll.mode = 'min'

    def clustering_scores(self, name, verbose=True, prediction_algorithm='knn'):
        if self.gene_dataset.n_labels > 1:
            latent, labels = get_latent(self.model, self.data_loaders[name])
            if prediction_algorithm == 'knn':
                labels_pred = KMeans(self.gene_dataset.n_labels, n_init=200).fit_predict(latent)  # n_jobs>1 ?
            elif prediction_algorithm == 'gmm':
                gmm = GMM(self.gene_dataset.n_labels)
                gmm.fit(latent)
                labels_pred = gmm.predict(latent)

            asw_score = silhouette_score(latent, labels)
            nmi_score = NMI(labels, labels_pred)
            ari_score = ARI(labels, labels_pred)
            uca_score = unsupervised_clustering_accuracy(labels, labels_pred)[0]
            if verbose:
                print("Clustering Scores for %s:\nSilhouette: %.4f\nNMI: %.4f\nARI: %.4f\nUCA: %.4f" %
                      (name, asw_score, nmi_score, ari_score, uca_score))
            return asw_score, nmi_score, ari_score, uca_score

    def entropy_batch_mixing(self, name, verbose=False, **kwargs):
        if self.gene_dataset.n_batches == 2:
            latent, labels = get_latent(self.model, self.data_loaders[name])
            be_score = entropy_batch_mixing(latent, batch_indices, **kwargs)
            if verbose:
                print("Entropy batch mixing :", be_score)
            return be_score

    entropy_batch_mixing.mode = 'max'

    def show_t_sne(self, name, n_samples=1000, color_by='', save_name='', latent=None, batch_indices=None,
                   labels=None, n_batch=None):
        # If no latent representation is given
        if latent is None:
            latent, labels = get_latent(self.model, self.data_loaders[name])
            latent, idx_t_sne = self.apply_t_sne(latent, n_samples)
            labels = labels[idx_t_sne].ravel()
        if not color_by:
            plt.figure(figsize=(10, 10))
            plt.scatter(latent[:, 0], latent[:, 1])
        if color_by == 'scalar':
            plt.figure(figsize=(10, 10))
            plt.scatter(latent[:, 0], latent[:, 1], c=labels.ravel())
        plt.axis("off")
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name)

    @staticmethod
    def apply_t_sne(latent, n_samples=1000):
        idx_t_sne = np.random.permutation(len(latent))[:n_samples] if n_samples else np.arange(len(latent))
        if latent.shape[1] != 2:
            latent = TSNE().fit_transform(latent[idx_t_sne])
        return latent, idx_t_sne


class SemiSupervisedVariationalInference(VariationalInference):
    r"""The abstract SemiSupervisedVariationalInference class for the semi-supervised training of an autoencoder.
    This parent class is inherited to specify the different training schemes for semi-supervised learning
    """
    default_metrics_to_monitor = VariationalInference.default_metrics_to_monitor + ['accuracy']

    def accuracy(self, name, verbose=False):
        acc = compute_accuracy(self.model, self.data_loaders[name])
        if verbose:
            print("Acc for %s is : %.4f" % (name, acc))
        return acc

    accuracy.mode = 'max'

    def hierarchical_accuracy(self, name, verbose=False):

        all_y, all_y_pred = compute_predictions(self.model, self.data_loaders[name])
        acc = np.mean(all_y == all_y_pred)

        all_y_groups = np.array([self.model.labels_groups[y] for y in all_y])
        all_y_pred_groups = np.array([self.model.labels_groups[y] for y in all_y_pred])
        h_acc = np.mean(all_y_groups == all_y_pred_groups)

        if verbose:
            print("Acc for %s is : %.4f\nHierarchical Acc for %s is : %.4f\n" % (name, acc, name, h_acc))
        return acc

    accuracy.mode = 'max'

    def unsupervised_accuracy(self, name, verbose=False):
        uca = unsupervised_classification_accuracy(self.model, self.data_loaders[name])[0]
        if verbose:
            print("UCA for %s is : %.4f" % (name, uca))
        return uca

    unsupervised_accuracy.mode = 'max'

    def svc_rf(self, **kwargs):
        if 'train' in self.data_loaders:
            raw_data = DataLoaders.raw_data(self.data_loaders['train'], self.data_loaders['test'])
        else:
            raw_data = DataLoaders.raw_data(self.data_loaders['labelled'], self.data_loaders['unlabelled'])
        (data_train, labels_train), (data_test, labels_test) = raw_data
        svc_scores, _ = compute_accuracy_svc(data_train, labels_train, data_test, labels_test, **kwargs)
        rf_scores, _ = compute_accuracy_rf(data_train, labels_train, data_test, labels_test, **kwargs)
        return svc_scores, rf_scores


class AlternateSemiSupervisedVariationalInference(SemiSupervisedVariationalInference):
    r"""The AlternateSemiSupervisedVariationalInference class for the semi-supervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAEC``, ``SVAEC``, ...
        :gene_dataset: A gene_dataset instance with pre-annotations like ``CortexDataset()``
        :n_labelled_samples_per_class: The number of labelled training samples per class. Default: ``50``.
        :**kwargs: Other keywords arguments from the general Inference class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> svaec = SVAEC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = AlternateSemiSupervisedVariationalInference(gene_dataset, svaec, n_labelled_samples_per_class=10)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """

    def __init__(self, model, gene_dataset, n_labelled_samples_per_class=50, n_label_array=None, n_epochs_classifier=1,
                 lr_classification=0.1, **kwargs):
        super(AlternateSemiSupervisedVariationalInference, self).__init__(model, gene_dataset, **kwargs)

        self.n_epochs_classifier = n_epochs_classifier
        self.lr_classification = lr_classification
        self.data_loaders = AlternateSemiSupervisedDataLoaders(gene_dataset, n_labelled_samples_per_class, n_label_array,
                                                               use_cuda=self.use_cuda)

        self.classifier_inference = ClassifierInference(
            model.classifier, gene_dataset, metrics_to_monitor=[], verbose=True, frequency=0,
            data_loaders=self.data_loaders.classifier_data_loaders(), sampling_model=self.model
        )

    def on_epoch_end(self):
        self.classifier_inference.train(self.n_epochs_classifier, lr=self.lr_classification)
        return super(AlternateSemiSupervisedVariationalInference, self).on_epoch_end()


class JointSemiSupervisedVariationalInference(SemiSupervisedVariationalInference):
    r"""The JointSemiSupervisedVariationalInference class for the semi-supervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAEC``, ``SVAEC``, ...
        :gene_dataset: A gene_dataset instance with pre-annotations like ``CortexDataset()``
        :n_labelled_samples_per_class: The number of labelled training samples per class. Default: ``50``.
        :**kwargs: Other keywords arguments from the general Inference class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> svaec = SVAEC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = JointSemiSupervisedVariationalInference(gene_dataset, svaec, n_labelled_samples_per_class=10)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """

    def __init__(self, model, gene_dataset, n_labelled_samples_per_class=50, n_label_array=None,classification_ratio=100, **kwargs):
        super(JointSemiSupervisedVariationalInference, self).__init__(model, gene_dataset, **kwargs)
        self.data_loaders = JointSemiSupervisedDataLoaders(gene_dataset, n_labelled_samples_per_class, n_label_array,
                                                           use_cuda=self.use_cuda)
        self.classification_ratio = classification_ratio

    def loss(self, tensors_all, tensors_labelled):
        loss = super(JointSemiSupervisedVariationalInference, self).loss(tensors_all)
        sample_batch, y = tensors_labelled
        classification_loss = F.cross_entropy(self.model.classify(sample_batch), y.view(-1))
        loss += classification_loss * self.classification_ratio
        return loss


class JointSemiSupervisedVariationalInferenceUncertain(SemiSupervisedVariationalInference):
    r"""The JointSemiSupervisedVariationalInference class for the semi-supervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAEC``, ``SVAEC``, ...
        :gene_dataset: A gene_dataset instance with pre-annotations like ``CortexDataset()``
        :n_labelled_samples_per_class: The number of labelled training samples per class. Default: ``50``.
        :**kwargs: Other keywords arguments from the general Inference class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> svaec = SVAEC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = JointSemiSupervisedVariationalInference(gene_dataset, svaec, n_labelled_samples_per_class=10)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """

    def __init__(self, model, gene_dataset, n_labelled_samples_per_class=50, n_label_array=None,classification_ratio=[1, 1], **kwargs):
        super(JointSemiSupervisedVariationalInferenceUncertain, self).__init__(model, gene_dataset, **kwargs)
        self.data_loaders = JointSemiSupervisedDataLoaders(gene_dataset, n_labelled_samples_per_class, n_label_array,
                                                           use_cuda=self.use_cuda)
        self.classification_ratio = classification_ratio

    def loss(self, tensors_all, tensors_labelled):
        loss = super(JointSemiSupervisedVariationalInferenceUncertain, self).loss(tensors_all)
        sample_batch, y = tensors_labelled
        classification_loss_1 = torch.sum((self.model.classify(sample_batch)[:, 1]) * y.type(torch.FloatTensor))
        classification_loss_0 = torch.sum((self.model.classify(sample_batch)[:, 0]) * (1-y).type(torch.FloatTensor))
        loss -= (classification_loss_0 * self.classification_ratio[0] + classification_loss_1 * self.classification_ratio[1])
        return loss

