# Standard library imports
import argparse
import io
import logging
import os
import sys
import time
from contextlib import redirect_stdout

# Third-party imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

# Adjust the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from model import create_model
from utils import (
    Caltech256Dataset,
    CIFAR100Dataset,
    Food101Dataset,
    MITIndoor67Dataset,
    Normalize,
    RandomCrop,
    SquarifyImage,
    ToTensor,
    get_high_confidence_samples,
    update_threshold
)

class PrintLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'a')

    def write(self, message):
        # Write to both terminal and file
        # self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        # self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def setup_logging(args):
    """
    Set up logging with hierarchical folder structure organized by model and dataset.
    
    Structure:
    logs/
    └── {model}/
        └── {dataset}/
            └── {exp_type}_{lambda}_{gamma}_{counter}.log
    """
    # Create base logs directory if it doesn't exist
    base_dir = 'logs'
    os.makedirs(base_dir, exist_ok=True)
    
    # Create model-specific directory
    model_dir = os.path.join(base_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create dataset-specific directory within model directory
    dataset_dir = os.path.join(model_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Construct base log filename
    base_filename = f"{args.exp_type}_{args.lambda_weight}_{args.gamma_weight}.log"
    log_path = os.path.join(dataset_dir, base_filename)
    
    # Handle file naming conflicts
    if os.path.exists(log_path):
        counter = 1
        while os.path.exists(os.path.join(
            dataset_dir,
            f"{args.exp_type}_{args.lambda_weight}_{args.gamma_weight}_{counter}.log"
        )):
            counter += 1
        log_path = os.path.join(
            dataset_dir,
            f"{args.exp_type}_{args.lambda_weight}_{args.gamma_weight}_{counter}.log"
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Add file handler with formatted output
    file_handler = logging.FileHandler(log_path)
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    
    return logger, log_path
        
def get_dataset(dataset_name, split, transform):
    """Factory function to get dataset based on name and split"""
    
    dataset_configs = {
        'caltech256': {
            'class': Caltech256Dataset,
            'n_classes': 256,
            'path': f'data/caltech256/{split}'
        },
        'food101': {
            'class': Food101Dataset,
            'n_classes': 101,
            'path': f'data/food101/{split}'
        },
        'cifar100': {
            'class': CIFAR100Dataset,
            'n_classes': 100,
            'path': f'data/cifar100/{split}'
        },
        'mit67': {
            'class': MITIndoor67Dataset,
            'n_classes': 67,
            'path': f'data/mit67/{split}'
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    config = dataset_configs[dataset_name]
    return config['class'](root_dir=config['path'], transform=transform), config['n_classes']

def bbal_learning_algorithm(du: DataLoader,
                            dl: DataLoader,
                            dtest: DataLoader,
                            model_name: str,
                            k: int = 1000,
                            delta_0: float = 0.005,
                            dr: float = 0.00033,
                            t: int = 1,
                            epochs: int = 10,
                            criteria: str = 'cl',
                            max_iter: int = 45,
                            alpha: float = 2.0,
                            lambda_weight: float = 1.0,
                            gamma_weight: float = 1.0,
                            n_classes: int = 256):
    """
    Algorithm1 : Learning algorithm of BBAL with class punishment and incorrectness rate.
    Parameters
    ----------
    du: DataLoader
        Unlabeled samples
    dl : DataLoader
        Labeled samples
    dtest : DataLoader
        Test data
    model_name: str
        Name of the model to use ('alexnet' or 'resnet18')
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
    # n_classes = 256

    # Initialize n_c_labeled based on initial labeled data
    n_c_labeled = np.zeros(n_classes, dtype=np.float32)
    for idx in dl.sampler.indices:
        label = dl.dataset.labels[idx]
        n_c_labeled[label] += 1

    # Create the model using factory
    model = create_model(model_name=model_name, n_classes=n_classes, device=device)

    # Initialize the model
    logger.info('Initialize training the model on `dl` and test on `dtest`')

    model.train(epochs=epochs, train_loader=dl, valid_loader=None)

    # Evaluate model on dtest
    # acc = model.evaluate(test_loader=dtest)
    per_class_acc = model.evaluate_per_class(test_loader=dtest, logger=logger, log_path=log_path)

    # print('====> Initial accuracy: {} '.format(acc))
    # print(f"====> Initial accuracy: {acc * 100:.2f}%")
    # print(f"====> Initial per-class accuracies: {per_class_acc}")
    # print(f"====> Initial sd of accuracies: {sd_acc:.4f}")

    for iteration in range(max_iter):
        iter_start = time.time()

        logger.info('Iteration: {}: run prediction on unlabeled data '
                    '`du` '.format(iteration))

        pred_prob = model.predict(test_loader=du)

        epsilon = 1e-10

        # Compute uncertainties
        uncertainties = -np.sum(pred_prob * np.log(pred_prob + epsilon), axis=1)

        cb_start = time.time()

        # Save a copy of du.sampler.indices to ensure consistent mapping
        current_du_indices = du.sampler.indices.copy()

        # Compute predicted classes
        predicted_classes = np.argmax(pred_prob, axis=1)

        # Estimate n_c_total
        n_c_total = pred_prob.sum(axis=0)  # shape: (n_classes,)

        # Compute Punishment_c
        Punishment_c = (n_c_labeled / (n_c_total + epsilon)) ** alpha

        # Compute Incorrectness_c
        Incorrectness_c = np.array([1.0 - per_class_acc[i] for i in range(n_classes)])

        # normalize the criteria
        def norm(arr):
            arr_min, arr_max = np.min(arr), np.max(arr)
            if arr_max - arr_min < epsilon:
                return np.zeros_like(arr)
            return (arr - arr_min) / (arr_max - arr_min)

        # breakpoint()
        # Compute adjusted scores
        adjusted_scores = norm(uncertainties) - lambda_weight * norm(Punishment_c[predicted_classes]) + gamma_weight * norm(Incorrectness_c[predicted_classes])

        cb_end = time.time()

        # Select top k samples with highest adjusted scores
        k = min(k, len(adjusted_scores))
        uncert_samp_idx = np.argsort(-adjusted_scores)[:k]

        # Map uncert_samp_idx to original indices
        selected_uncert_indices = [current_du_indices[idx] for idx in uncert_samp_idx]

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

        # Get high confidence samples `dh`
        hcs_idx, hcs_labels = get_high_confidence_samples(pred_prob=pred_prob,
                                                          delta=delta_0)

        # Map hcs_idx to original indices
        hcs_indices = [current_du_indices[idx] for idx in hcs_idx]

        # Remove samples that are already selected as uncertain samples
        hcs_tuples = [(idx, label) for idx, label in zip(hcs_indices, hcs_labels) if idx not in selected_uncert_indices]

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

        per_class_acc = model.evaluate_per_class(test_loader=dtest, logger=logger, 
                                       log_path=log_path, iteration=iteration)

        iter_end = time.time()
        print(f"Total time: {iter_end - iter_start}")
        print(f"Additional time: {cb_end - cb_start}")


        # print(
        #     "Iteration: {}, len(dl): {}, len(du): {}, len(dh) {}\n"
        #     "acc: {}, sd of accuracies: {}, per-class accuracies: {}".format(
        #         iteration, len(dl.sampler.indices),
        #         len(du.sampler.indices), len(hcs_indices), acc, sd_acc, per_class_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Active Learning with BBAL')
    parser.add_argument('--dataset', type=str, default='caltech256',
                        choices=['caltech256', 'food101', 'cifar100', 'mit67'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['alexnet', 'resnet18'],
                        help='Model architecture to use')
    parser.add_argument('--lambda_weight', type=float, default=1.0,
                        help='Weight for class punishment')
    parser.add_argument('--gamma_weight', type=float, default=1.0,
                        help='Weight for incorrectness rate')
    parser.add_argument('--training_epoch', type=int, default=10,
                        help='Training epochs when validation')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--exp_type', type=str, default='gs',
                        help='Experiment type for log naming')
    args = parser.parse_args()
    # parser.add_argument('--criteria', type=str, default='cl', help='Criteria for selection')

    args = parser.parse_args()
    
    # Setup logging with the new structure
    logger, log_path = setup_logging(args)
    
    # Redirect stdout to our custom logger
    sys.stdout = PrintLogger(log_path)

    # Log the configuration
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Experiment type: {args.exp_type}")
    logger.info(f"Lambda weight: {args.lambda_weight}")
    logger.info(f"Gamma weight: {args.gamma_weight}")
    logger.info(f"Training epochs: {args.training_epoch}")
    logger.info(f"Batch size: {args.batch_size}")

    # Set device to GPU if available (CUDA or MPS)
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    # Setup transforms
    transform = transforms.Compose([
        SquarifyImage(),
        RandomCrop(224),
        Normalize(),
        ToTensor()
    ])
    
    # Get dataset and number of classes
    dataset_train, n_classes = get_dataset(args.dataset, 'train', transform)
    dataset_test, _ = get_dataset(args.dataset, 'test', transform)

    # Create splits
    validation_split = 0.1
    dataset_size = len(dataset_train)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    np.random.seed(123)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    du = DataLoader(dataset_train, batch_size=args.batch_size,
                   sampler=train_sampler, num_workers=4, pin_memory=True)
    dl = DataLoader(dataset_train, batch_size=args.batch_size,
                   sampler=valid_sampler, num_workers=4, pin_memory=True)
    dtest = DataLoader(dataset_test, batch_size=args.batch_size,
                      num_workers=4, pin_memory=True)

    # Run CEAL algorithm
    bbal_learning_algorithm(
        du=du, 
        dl=dl, 
        dtest=dtest,
        model_name=args.model,
        k=2000,
        epochs=args.training_epoch,
        lambda_weight=args.lambda_weight,
        gamma_weight=args.gamma_weight,
        n_classes=n_classes
    )
