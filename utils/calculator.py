import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Tuple

def calculate_metrics_per_class(conf_matrix: np.ndarray) -> Dict[str, Tuple[float, float]]:
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
    
    return {
        'false_alarm_rate': (np.mean(false_alarm_rates), np.std(false_alarm_rates)),
        'macro_precision': (np.mean(precisions), np.std(precisions)),
        'macro_recall': (np.mean(recalls), np.std(recalls)),
        'macro_f1': (np.mean(f1_scores), np.std(f1_scores)),
        'weighted_precision': (weighted_precision, np.std(precisions)),
        'weighted_recall': (weighted_recall, np.std(recalls)),
        'weighted_f1': (weighted_f1, np.std(f1_scores))
    }

def calculate_auc(metrics_list: List[Dict[str, Tuple[float, float]]]) -> Dict[str, Tuple[float, float]]:
    """Calculate AUC for metrics and their standard deviations."""
    auc_dict = {}
    for metric in metrics_list[0].keys():
        values = [m[metric][0] for m in metrics_list]  # means
        std_devs = [m[metric][1] for m in metrics_list]  # standard deviations
        auc_dict[metric] = (
            np.trapz(values, dx=1.0) / len(values),  # AUC of means
            np.trapz(std_devs, dx=1.0) / len(std_devs)  # AUC of standard deviations
        )
    return auc_dict

def process_experiment_folder(folder_path: str) -> Dict[str, Tuple[float, float]]:
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

def format_results(config: str, results: Dict[str, Tuple[float, float]]):
    """Format results in the specified format."""
    print(f"Config: {config}")
    print("Metric\t\tAUC\t\tStd Dev")
    print("-" * 40)
    
    for metric, (auc, std) in sorted(results.items()):
        # Ensure proper spacing for different metric names
        spacing = "\t" if len(metric) >= 15 else "\t\t"
        print(f"{metric}{spacing}{auc:.8f}\t{std:.8f}")
    print()

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
        for exp_folder in sorted(os.listdir(dataset_path)):
            if not os.path.isdir(os.path.join(dataset_path, exp_folder)):
                continue
            
            # Skip the repeated experiments (with _1 suffix)
            # if exp_folder.endswith('_1'):
            #     continue
                
            results = process_experiment_folder(os.path.join(dataset_path, exp_folder))
            format_results(exp_folder, results)

if __name__ == "__main__":
    main()