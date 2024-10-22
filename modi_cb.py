import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Caltech256Dataset, Normalize, RandomCrop, SquarifyImage, \
    ToTensor
from utils import get_high_confidence_samples, \
    update_threshold
from model import AlexNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import torch
import logging

logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def ceal_learning_algorithm(du: DataLoader,
                            dl: DataLoader,
                            dtest: DataLoader,
                            k: int = 1000,
                            delta_0: float = 0.005,
                            dr: float = 0.00033,
                            t: int = 1,
                            epochs: int = 10,
                            criteria: str = 'cl',
                            max_iter: int = 45,
                            alpha: float = 2.0,
                            lambda_weight: float = 1.0):
    """
    Algorithm1 : Learning algorithm of CEAL with class punishment.
    Parameters
    ----------
    du: DataLoader
        Unlabeled samples
    dl : DataLoader
        Labeled samples
    dtest : DataLoader
        Test data
    k: int, (default = 1000)
        Number of uncertain samples to select
    delta_0: float
        High confidence samples selection threshold
    dr: float
        Threshold decay
    t: int
        Fine-tuning interval
    epochs: int
        Number of training epochs
    criteria: str
        Uncertainty criteria
    max_iter: int
        Maximum iteration number
    alpha: float
        Hyperparameter controlling the growth rate of punishment
    lambda_weight: float
        Weighting factor balancing uncertainty and punishment

    Returns
    -------

    """
    logger.info('Initial configuration: len(du): {}, len(dl): {} '.format(
        len(du.sampler.indices),
        len(dl.sampler.indices)))

    # Number of classes
    n_classes = 256

    # Initialize n_c_labeled based on initial labeled data
    n_c_labeled = np.zeros(n_classes, dtype=np.float32)
    for idx in dl.sampler.indices:
        label = dl.dataset.labels[idx]
        n_c_labeled[label] += 1

    # Create the model
    model = AlexNet(n_classes=n_classes, device=None)

    # Initialize the model
    logger.info('Initialize training the model on `dl` and test on `dtest`')

    model.train(epochs=epochs, train_loader=dl, valid_loader=None)

    # Evaluate model on dtest
    acc = model.evaluate(test_loader=dtest)

    print('====> Initial accuracy: {} '.format(acc))

    for iteration in range(max_iter):

        logger.info('Iteration: {}: run prediction on unlabeled data '
                    '`du` '.format(iteration))

        pred_prob = model.predict(test_loader=du)

        # Compute predicted classes
        predicted_classes = np.argmax(pred_prob, axis=1)

        # Estimate n_c_total
        n_c_total = pred_prob.sum(axis=0)  # shape: (n_classes,)

        epsilon = 1e-10

        # Compute Punishment_c
        Punishment_c = (n_c_labeled / (n_c_total + epsilon)) ** alpha

        # Compute uncertainties
        uncertainties = -np.sum(pred_prob * np.log(pred_prob + epsilon), axis=1)

        # Compute adjusted scores
        adjusted_scores = uncertainties - lambda_weight * Punishment_c[predicted_classes]

        # Select top k samples with highest adjusted scores
        k = min(k, len(adjusted_scores))
        uncert_samp_idx = np.argsort(-adjusted_scores)[:k]

        # Map uncert_samp_idx to original indices
        selected_uncert_indices = [du.sampler.indices[idx] for idx in uncert_samp_idx]

        # Update n_c_labeled with true labels of uncertain samples
        for idx in selected_uncert_indices:
            label = du.dataset.labels[idx]
            n_c_labeled[label] += 1

        # Add uncertain samples to dl
        dl.sampler.indices.extend(selected_uncert_indices)

        logger.info(
            'Update size of `dl`  and `du` by adding uncertain {} samples in `dl`'
            ' len(dl): {}, len(du) {}'.
            format(len(selected_uncert_indices), len(dl.sampler.indices),
                   len(du.sampler.indices)))

        # Remove uncertain samples from du
        for idx in selected_uncert_indices:
            if idx in du.sampler.indices:
                du.sampler.indices.remove(idx)

        # Get high confidence samples `dh`
        hcs_idx, hcs_labels = get_high_confidence_samples(pred_prob=pred_prob,
                                                          delta=delta_0)

        # Map hcs_idx to original indices
        hcs_indices = [du.sampler.indices[idx] for idx in hcs_idx]

        # Remove samples that already selected as uncertain samples
        hcs_indices = [x for x in hcs_indices if x not in selected_uncert_indices]
        hcs_labels = [hcs_labels[i] for i in range(len(hcs_idx)) if du.sampler.indices[hcs_idx[i]] not in selected_uncert_indices]

        # Update n_c_labeled with hcs_labels
        for idx, label in zip(hcs_indices, hcs_labels):
            n_c_labeled[label] += 1

        # Add high confidence samples to dl
        dl.sampler.indices.extend(hcs_indices)

        # Update labels in dl.dataset.labels
        for idx, label in zip(hcs_indices, hcs_labels):
            dl.dataset.labels[idx] = label

        logger.info(
            'Update size of `dl`  and `du` by adding {} hcs samples in `dl`'
            ' len(dl): {}, len(du) {}'.
            format(len(hcs_indices), len(dl.sampler.indices),
                   len(du.sampler.indices)))

        # Remove high confidence samples from du
        for idx in hcs_indices:
            if idx in du.sampler.indices:
                du.sampler.indices.remove(idx)

        if iteration % t == 0:
            logger.info('Iteration: {} fine-tune the model on dh U dl'.
                        format(iteration))
            model.train(epochs=epochs, train_loader=dl)

            # update delta_0
            delta_0 = update_threshold(delta=delta_0, dr=dr, t=iteration)

        acc = model.evaluate(test_loader=dtest)
        print(
            "Iteration: {}, len(dl): {}, len(du): {},"
            " len(dh) {}, acc: {} ".format(
                iteration, len(dl.sampler.indices),
                len(du.sampler.indices), len(hcs_indices), acc))


if __name__ == "__main__":

    dataset_train = Caltech256Dataset(
        root_dir="../data/256_ObjectCategories",
        transform=transforms.Compose(
            [SquarifyImage(),
             RandomCrop(224),
             Normalize(),
             ToTensor()]))

    dataset_test = Caltech256Dataset(
        root_dir="../data/divided",
        transform=transforms.Compose(
            [SquarifyImage(),
             RandomCrop(224),
             Normalize(),
             ToTensor()]))

    # Creating data indices for training and validation splits:
    random_seed = 123
    validation_split = 0.1  # 10%
    shuffling_dataset = True
    batch_size = 16
    dataset_size = len(dataset_train)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffling_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    du = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                     sampler=train_sampler, num_workers=4)
    dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                     sampler=valid_sampler, num_workers=4)
    dtest = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                        num_workers=4)

    ceal_learning_algorithm(du=du, dl=dl, dtest=dtest)
