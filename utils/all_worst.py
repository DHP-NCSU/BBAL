import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Tuple

def calculate_metrics_per_class(conf_matrix: np.ndarray) -> List[float]:
    """Calculate metrics per class and their standard deviations."""
    n_classes = conf_matrix.shape[0]
    
    # Calculate per-class metrics
    precisions = []
    recalls = []
    f1_scores = []
    false_alarm_rates = []
    
    row_sums = conf_matrix.sum(axis=1)
    col_sums = conf_matrix.sum(axis=0)
    
    for i in range(n_classes):
        # True positives
        tp = conf_matrix[i, i]
        # False positives
        fp = col_sums[i] - tp
        # False negatives
        fn = row_sums[i] - tp
        # True negatives
        tn = conf_matrix.sum() - (tp + fp + fn)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        false_alarm_rates.append(far)
    
    # Calculate weighted metrics
    weights = row_sums / row_sums.sum()
    weighted_precision = np.average(precisions, weights=weights)
    weighted_recall = np.average(recalls, weights=weights)
    weighted_f1 = np.average(f1_scores, weights=weights)
    
    # 162, 158, 124, 125, and 232
    return [precisions[162], precisions[158], precisions[124], precisions[125], precisions[232]]

def calculate_auc(metrics_list: List[List[float]]) -> Dict[int, float]:
    """Calculate AUC for metrics and their standard deviations."""
    n_classes = len(metrics_list[0])
    auc_dict = {i: 0.0 for i in range(n_classes)}
    for metric in metrics_list:
        for i in range(n_classes):
            auc_dict[i] += metric[i]
    
    for i in range(n_classes):
        auc_dict[i] /= len(metrics_list)
    
    return auc_dict

def process_experiment_folder(folder_path: str) -> Dict[int, float]:
    """Process all iteration files in an experiment folder."""
    metrics_list = []
    
    # Get all .npy files and sort them
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    npy_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'iter' in x else -1)
    
    for npy_file in npy_files:
        conf_matrix = np.load(os.path.join(folder_path, npy_file))
        metrics = calculate_metrics_per_class(conf_matrix)
        metrics_list.append(metrics)
    
    return calculate_auc(metrics_list)


def main():
    base_dir = 'confusion_matrices/resnet18'
    # datasets = ['caltech256', 'cifar100', 'food101', 'mit67']
    datasets = ['caltech256']
    
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.exists(dataset_path):
            continue
            
        # Process all experiment folders
        counter = {}
        for exp_folder in sorted(os.listdir(dataset_path)):
            if not os.path.isdir(os.path.join(dataset_path, exp_folder)):
                continue
            
            
            # Skip the repeated experiments (with _1 suffix)
            # if exp_folder.endswith('_1'):
            #     continue
                
            result = process_experiment_folder(os.path.join(dataset_path, exp_folder))
            if exp_folder[-2] == '_':
                exp_folder = exp_folder[:-2]
            if exp_folder in counter.keys():
                for i in range(5):
                    counter[exp_folder][i] += result[i] / 2
            else:
                counter[exp_folder] = []
                for i in range(5):
                    counter[exp_folder].append(result[i])
        baseline = {0: 0.28200913655261195, 1: 0.3249527184913556, 2: 0.3553298072268566, 3: 0.39419862317542237, 4: 0.3995199160029491}
        for k in counter.keys():
            # 0.05 & 0.1 & 14.05\% & 21.54\% & 26.19\% & 28.01\% & 24.76\% \\
            la, ga = k.split('_')[1:]
            info = f"{la} & {ga}"
            for i in range(5):
                info += f" & {counter[k][i] * 100:.2f}\\%"
            print(info + " \\\\")
        for k in counter.keys():
            delta = 0.0
            for i in range(5):
                delta += (counter[k][i] - baseline[i])
            print(k, delta)


if __name__ == "__main__":
    main()