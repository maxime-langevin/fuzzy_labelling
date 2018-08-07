from vi.dataset import GmmDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from vi.models import SVAEC, VAE, UVAE
from vi.inference import JointSemiSupervisedVariationalInference, VariationalInference, JointSemiSupervisedVariationalInferenceUncertain
from vi.dataset import TestDataset
import torch

def compute_predictions(vae, data_loader, classifier=None):
    all_y_pred = []
    all_y = []
    values = []

    for i_batch, tensors in enumerate(data_loader):
        sample_batch, labels = tensors
        all_y += [labels.view(-1)]

        if hasattr(vae, 'classify'):
            y_pred = vae.classify(sample_batch).argmax(dim=-1)
        elif classifier is not None:
            # Then we use the specified classifier
            if vae is not None:
                sample_batch, _, _ = vae.z_encoder(sample_batch)
            y_pred = classifier(sample_batch).argmax(dim=-1)
        all_y_pred += [y_pred]

    all_y_pred = np.array(torch.cat(all_y_pred))
    all_y = np.array(torch.cat(all_y))
    return all_y, all_y_pred


dataset = GmmDataset(5, 100, np.array([0.3, 0.7]), np.array([[0, 0, 0, 3, 3], [1, 1, 0, 0, 0]]),
                     np.array([[1, 1, 1, 1, 1], [2, 2, 2, 3, 3]]), total_size=5000)

# corrupting the labels
new_labels = np.zeros_like(dataset.labels)
for i, label in enumerate(dataset.labels):
    if label == 0:
        p = np.random.binomial(1, 0.85)
        if p < 0.5:
            new_labels[i] = 0
        else:
            new_labels[i] = 1
    else:
        new_labels[i] = 1

new_dataset = TestDataset(dataset.X, new_labels)

svaec = UVAE(100, dataset.n_labels, n_latent=5)
print("start")
infer = JointSemiSupervisedVariationalInferenceUncertain(svaec, new_dataset, n_labelled_samples_per_class=10, n_label_array=[10, 30],
                                                          classification_ratio=[-2, -20], verbose=True, frequency=1)
#infer = JointSemiSupervisedVariationalInference(svaec, dataset, n_labelled_samples_per_class=50, n_label_array=[10, 30])

infer.train(n_epochs=50)
infer.show_t_sne('labelled', color_by='scalar', save_name='corrupted.png')
print(infer.accuracy('unlabelled'))

infer = JointSemiSupervisedVariationalInference(svaec, dataset, n_labelled_samples_per_class=50)
infer.train(n_epochs=1, lr=0)
infer.show_t_sne('labelled', color_by='scalar', save_name='original.png')
print(infer.accuracy('unlabelled'))

X = new_dataset.X

# predicting labels
x_ = torch.from_numpy(X).type(torch.FloatTensor)
all_y_pred = []
y_pred = svaec.classify(x_).argmax(dim=-1)
print(y_pred.size())
all_y_pred = np.array(y_pred)

idx_t_sne = np.random.permutation(X.shape[0])[:1000]
X_embedded = TSNE(n_components=2).fit_transform(X[idx_t_sne, :])
plt.figure(figsize=(10, 10))
plt.subplot(311)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=new_dataset.labels[idx_t_sne].ravel())
plt.subplot(312)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=dataset.labels[idx_t_sne].ravel())
plt.subplot(313)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=all_y_pred[idx_t_sne].ravel())
plt.savefig('check_original_and_corrupted.png')


# svaec = VAE(100, 10, n_hidden=52, n_latent=1, n_layers=2)
# infer = VariationalInference(svaec, dataset, verbose=True, frequency=1)
# infer.data_loaders['train'] = dataloader
# infer.data_loaders['test'] = dataloader

# print(infer.verbose)
# print(infer.data_loaders.to_monitor)
# infer.train(n_epochs=200, lr=0.01)
# infer.show_t_sne('train', color_by='scalar', save_name='latent_space.png')
# print(infer.history["ll_train"])
# print(infer.accuracy('unlabelled'))