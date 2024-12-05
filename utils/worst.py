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
    
    # return {
    #     'false_alarm_rate': (np.mean(false_alarm_rates), np.std(false_alarm_rates)),
    #     'macro_precision': (np.mean(precisions), np.std(precisions)),
    #     'macro_recall': (np.mean(recalls), np.std(recalls)),
    #     'macro_f1': (np.mean(f1_scores), np.std(f1_scores)),
    #     'weighted_precision': (weighted_precision, np.std(precisions)),
    #     'weighted_recall': (weighted_recall, np.std(recalls)),
    #     'weighted_f1': (weighted_f1, np.std(f1_scores))
    # }
    return precisions

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
        res_auc = {i: 0.0 for i in range(256)}
        for exp_folder in ["gs_0.0_0.0_1", "gs_0.0_0.0_3"]:
            if not os.path.isdir(os.path.join(dataset_path, exp_folder)):
                continue
            
            
            # Skip the repeated experiments (with _1 suffix)
            # if exp_folder.endswith('_1'):
            #     continue
                
            result = process_experiment_folder(os.path.join(dataset_path, exp_folder))
            for i in range(256):
                res_auc[i] += result[i]
        
        for i in range(256):
            res_auc[i] /= 2
        
        sorted_auc = sorted(res_auc.items(), key=lambda x: x[1])[:5]
        print(sorted_auc)


if __name__ == "__main__":
    main()