import sys
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Caltech256Dataset, Normalize, RandomCrop, SquarifyImage, ToTensor
from utils import get_high_confidence_samples, update_threshold
from model import AlexNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import logging

logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def ceal_learning_algorithm(config, du, dl, dtest, k=1000, delta_0=0.005, dr=0.00033, t=1, epochs=10, criteria='cl', max_iter=45, alpha=2.0):
    lambda_weight = config["lambda_weight"]
    gamma_weight = config["gamma_weight"]

    logger.info('Initial configuration: len(du): {}, len(dl): {}'.format(len(du.sampler.indices), len(dl.sampler.indices)))
    
    n_classes = 256
    n_c_labeled = np.zeros(n_classes, dtype=np.float32)
    for idx in dl.sampler.indices:
        label = dl.dataset.labels[idx]
        n_c_labeled[label] += 1

    model = AlexNet(n_classes=n_classes, device=None)
    model.train(epochs=epochs, train_loader=dl, valid_loader=None)
    acc, sd_acc, per_class_acc = model.evaluate_per_class(test_loader=dtest)

    acc_acc, acc_sd_acc = 0.0, 0.0
    for iteration in range(max_iter):
        Incorrectness_c = np.array([1.0 - per_class_acc[i] for i in range(n_classes)])
        pred_prob = model.predict(test_loader=du)
        current_du_indices = du.sampler.indices.copy()
        predicted_classes = np.argmax(pred_prob, axis=1)
        n_c_total = pred_prob.sum(axis=0)
        epsilon = 1e-10
        Punishment_c = (n_c_labeled / (n_c_total + epsilon)) ** alpha
        uncertainties = -np.sum(pred_prob * np.log(pred_prob + epsilon), axis=1)

        def norm(arr):
            arr_min, arr_max = np.min(arr), np.max(arr)
            if arr_max - arr_min < epsilon:
                return np.zeros_like(arr)
            return (arr - arr_min) / (arr_max - arr_min)

        adjusted_scores = norm(uncertainties) - lambda_weight * norm(Punishment_c[predicted_classes]) + gamma_weight * norm(Incorrectness_c[predicted_classes])
        k = min(k, len(adjusted_scores))
        uncert_samp_idx = np.argsort(-adjusted_scores)[:k]
        selected_uncert_indices = [current_du_indices[idx] for idx in uncert_samp_idx]

        for idx in selected_uncert_indices:
            label = du.dataset.labels[idx]
            n_c_labeled[label] += 1
        dl.sampler.indices.extend(selected_uncert_indices)

        hcs_idx, hcs_labels = get_high_confidence_samples(pred_prob=pred_prob, delta=delta_0)
        hcs_indices = [current_du_indices[idx] for idx in hcs_idx]
        hcs_tuples = [(idx, label) for idx, label in zip(hcs_indices, hcs_labels) if idx not in selected_uncert_indices]
        if hcs_tuples:
            hcs_indices, hcs_labels = zip(*hcs_tuples)
            hcs_indices = list(hcs_indices)
            hcs_labels = list(hcs_labels)
        else:
            hcs_indices = []
            hcs_labels = []

        for label in hcs_labels:
            n_c_labeled[label] += 1
        dl.sampler.indices.extend(hcs_indices)
        for idx, label in zip(hcs_indices, hcs_labels):
            dl.dataset.labels[idx] = label

        selected_indices_to_remove = selected_uncert_indices + hcs_indices
        du.sampler.indices = [idx for idx in du.sampler.indices if idx not in selected_indices_to_remove]

        if iteration % t == 0:
            model.train(epochs=epochs, train_loader=dl)
            delta_0 = update_threshold(delta=delta_0, dr=dr, t=iteration)

        acc, sd_acc, per_class_acc = model.evaluate_per_class(test_loader=dtest)
        
        acc_acc += acc
        acc_sd_acc += sd_acc
    tune.report(comb=acc+0.5*acc_sd_acc, acc=acc_acc, sd_acc=acc_sd_acc)

def train_ceal(config):
    dataset_train = Caltech256Dataset(
        root_dir="../data/256_ObjectCategories",
        transform=transforms.Compose([SquarifyImage(), RandomCrop(224), Normalize(), ToTensor()]))

    dataset_test = Caltech256Dataset(
        root_dir="../data/divided",
        transform=transforms.Compose([SquarifyImage(), RandomCrop(224), Normalize(), ToTensor()]))

    random_seed = 123
    validation_split = 0.1
    batch_size = 16
    dataset_size = len(dataset_train)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    du = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=valid_sampler, num_workers=4)
    dtest = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=4)

    ceal_learning_algorithm(config, du=du, dl=dl, dtest=dtest, epochs=1)

if __name__ == "__main__":
    ray.init()
    config = {
        "lambda_weight": tune.uniform(0.1, 1.0),
        "gamma_weight": tune.uniform(0.1, 1.0)
    }

    scheduler = ASHAScheduler(
        metric="comb",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    tune.run(
        train_ceal,
        config=config,
        scheduler=scheduler,
        num_samples=10,
        resources_per_trial={"cpu": 4, "gpu": 1}
    )
