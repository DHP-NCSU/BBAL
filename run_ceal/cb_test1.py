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
import argparse

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
                            lambda_weight: float = 1.0,
                            gamma_weight: float = 1.0):
    """
    Algorithm1 : Learning algorithm of CEAL with class punishment and incorrectness rate.
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
    gamma_weight: float
        Weighting factor for incorrectness rate

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
    # acc = model.evaluate(test_loader=dtest)
    acc, sd_acc, per_class_acc = model.evaluate_per_class(test_loader=dtest)

    # print('====> Initial accuracy: {} '.format(acc))
    print(f"====> Initial accuracy: {acc * 100:.2f}%")
    print(f"====> Initial per-class accuracies: {per_class_acc}")
    print(f"====> Initial sd of accuracies: {sd_acc:.4f}")

    for iteration in range(max_iter):
        # Compute Incorrectness_c
        Incorrectness_c = np.array([1.0 - per_class_acc[i] for i in range(n_classes)])

        logger.info('Iteration: {}: run prediction on unlabeled data '
                    '`du` '.format(iteration))

        pred_prob = model.predict(test_loader=du)

        # Save a copy of du.sampler.indices to ensure consistent mapping
        current_du_indices = du.sampler.indices.copy()

        # Compute predicted classes
        predicted_classes = np.argmax(pred_prob, axis=1)

        # Estimate n_c_total
        n_c_total = pred_prob.sum(axis=0)  # shape: (n_classes,)

        epsilon = 1e-10

        # Compute Punishment_c
        Punishment_c = (n_c_labeled / (n_c_total + epsilon)) ** alpha

        # Compute uncertainties
        uncertainties = -np.sum(pred_prob * np.log(pred_prob + epsilon), axis=1)

        #calculate average uncertainty for each class
        class_uncertainties = np.zeros(n_classes,dtype=np.float32)
        class_counts = np.zeros(n_classes, dtype=np.float32)
        # normalize the criteria
        def norm(arr):
            arr_min, arr_max = np.min(arr), np.max(arr)
            if arr_max - arr_min < epsilon:
                return np.zeros_like(arr)
            return (arr - arr_min) / (arr_max - arr_min)

        # breakpoint()
        k = min(k, len(uncertainties))
        baselien_samp_idx = np.argsort(-uncertainties)[:k]
        while True:
            lambda_weight, gamma_weight = input("input lambda and gamma:\n").split(' ')
            lambda_weight, gamma_weight = float(lambda_weight.strip()), float(gamma_weight.strip())
            adjusted_scores = norm(uncertainties) - lambda_weight * norm(Punishment_c[predicted_classes]) + gamma_weight * norm(Incorrectness_c[predicted_classes])
            uncert_samp_idx = np.argsort(-adjusted_scores)[:k]
            overlap = len(set(baselien_samp_idx) & set(uncert_samp_idx))
            decision = input(f"The overlap now is {overlap}, do you want to use this set? (y/n)\n")
            if decision.strip() == 'y':
                break
        
        # Compute adjusted scores
        # adjusted_scores = norm(uncertainties) - lambda_weight * norm(Punishment_c[predicted_classes]) + gamma_weight * norm(Incorrectness_c[predicted_classes])

        # Select top k samples with highest adjusted scores
        # k = min(k, len(adjusted_scores))
        # uncert_samp_idx = np.argsort(-adjusted_scores)[:k]

        # Map uncert_samp_idx to original indices
        selected_uncert_indices = [current_du_indices[idx] for idx in uncert_samp_idx]

        # Update n_c_labeled with true labels of uncertain samples
        '''for idx in selected_uncert_indices:
            label = du.dataset.labels[idx]
            n_c_labeled[label] += 1'''
        for idx, uncertainty in zip(current_du_indices,uncertainties):
            label = du.dataset.labels[idx]
            class_uncertainties[label] += uncertainty
            class_counts[label] += 1
        
        #Avoid division by zero
        class_uncertainties = np.divide(class_uncertainties,class_counts, where=class_counts > 0)
        #select the most uncertain 5 classes
        most_uncertain_classes = np.argsort(class_uncertainties)[:5]
        # Filter samples belonging to the most uncertain classes
        uncertain_samples = [idx for idx in current_du_indices if du.dataset.labels[idx] in most_uncertain_classes]
        uncertain_samples = uncertain_samples[:k]  # Limit to k samples
        # Update n_c_labeled with true labels of uncertain samples
        for idx in uncertain_samples:
            label = du.dataset.labels[idx]
            n_c_labeled[label] += 1
        # Add uncertain samples to dl
        #dl.sampler.indices.extend(selected_uncert_indices)
        dl.sampler.indices.extend(uncertain_samples)

        logger.info(
            'Update size of `dl`  and `du` by adding uncertain {} samples in `dl`'
            ' len(dl): {}, len(du) {}'.
            #format(len(selected_uncert_indices), len(dl.sampler.indices),
                   #len(du.sampler.indices)))
            format(len(uncertain_samples), len(dl.sampler.indices),
                   len(du.sampler.indices))
        )

        # Get high confidence samples `dh`
        hcs_idx, hcs_labels = get_high_confidence_samples(pred_prob=pred_prob,
                                                          delta=delta_0)

        # Map hcs_idx to original indices
        hcs_indices = [current_du_indices[idx] for idx in hcs_idx]

        # Remove samples that are already selected as uncertain samples
        #hcs_tuples = [(idx, label) for idx, label in zip(hcs_indices, hcs_labels) if idx not in selected_uncert_indices]
        hcs_tuples = [(idx, label) for idx, label in zip(hcs_indices, hcs_labels) if idx not in uncertain_samples]

        if hcs_tuples:
            hcs_indices, hcs_labels = zip(*hcs_tuples)
            hcs_indices = list(hcs_indices)
            hcs_labels = list(hcs_labels)
        else:
            hcs_indices = []
            hcs_labels = []

        # Update n_c_labeled with hcs_labels
        for label in hcs_labels:
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

        # Remove all selected samples from du.sampler.indices after processing
        selected_indices_to_remove = selected_uncert_indices + hcs_indices
        du.sampler.indices = [idx for idx in du.sampler.indices if idx not in selected_indices_to_remove]

        if iteration % t == 0:
            logger.info('Iteration: {} fine-tune the model on dh U dl'.
                        format(iteration))
            model.train(epochs=epochs, train_loader=dl)

            # update delta_0
            delta_0 = update_threshold(delta=delta_0, dr=dr, t=iteration)

        acc, sd_acc, per_class_acc = model.evaluate_per_class(test_loader=dtest)

        print(
            "Iteration: {}, len(dl): {}, len(du): {}, len(dh) {}\n"
            "acc: {}, sd of accuracies: {}, per-class accuracies: {}".format(
                iteration, len(dl.sampler.indices),
                len(du.sampler.indices), len(hcs_indices), acc, sd_acc, per_class_acc))


if __name__ == "__main__":

    # import argparse

    parser = argparse.ArgumentParser(description='Active Learning with CEAL and Class Punishment and Incorrectness Rate')

    parser.add_argument('--lambda_weight', type=float, default=1.0, help='Weight for class punishment')
    parser.add_argument('--gamma_weight', type=float, default=1.0, help='Weight for incorrectness rate')
    parser.add_argument('--training_epoch', type=int, default=10, help='Training epochs when validation')
    # parser.add_argument('--criteria', type=str, default='cl', help='Criteria for selection')

    args = parser.parse_args()

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
                                     sampler=train_sampler, num_workers=4, pin_memory=True)
    dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                     sampler=valid_sampler, num_workers=4, pin_memory=True)
    dtest = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                        num_workers=4, pin_memory=True)

    ceal_learning_algorithm(du=du, dl=dl, dtest=dtest, epochs=args.training_epoch, 
                            lambda_weight=args.lambda_weight, gamma_weight=args.gamma_weight)
