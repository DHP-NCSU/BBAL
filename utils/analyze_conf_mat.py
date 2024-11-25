import numpy as np
from typing import Tuple

def analyze_confusion_matrix(file_path: str) -> Tuple[float, float, float, float, dict]:
    """
    Analyze a confusion matrix loaded from a .npy file
    
    Args:
        file_path: Path to the .npy file containing the confusion matrix
        
    Returns:
        Tuple containing accuracy, precision, recall, f1_score, and per_class_metrics
    """
    # Load the confusion matrix
    conf_matrix = np.load(file_path)
    
    # Basic validation
    if conf_matrix.ndim != 2 or conf_matrix.shape[0] != conf_matrix.shape[1]:
        raise ValueError("Invalid confusion matrix shape. Must be a square matrix.")
    
    num_classes = conf_matrix.shape[0]
    
    # Calculate basic metrics
    total_samples = np.sum(conf_matrix)
    correct_predictions = np.sum(np.diag(conf_matrix))
    accuracy = correct_predictions / total_samples
    
    # Initialize per-class metrics
    per_class_metrics = {}
    
    # Calculate metrics for each class
    for i in range(num_classes):
        # True Positives: diagonal elements
        tp = conf_matrix[i, i]
        
        # False Positives: sum of column minus true positive
        fp = np.sum(conf_matrix[:, i]) - tp
        
        # False Negatives: sum of row minus true positive
        fn = np.sum(conf_matrix[i, :]) - tp
        
        # True Negatives: sum of all elements minus (tp + fp + fn)
        tn = total_samples - (tp + fp + fn)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics[f"class_{i}"] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": tp + fn
        }
    
    # Calculate macro-averaged metrics
    macro_precision = np.mean([metrics["precision"] for metrics in per_class_metrics.values()])
    macro_recall = np.mean([metrics["recall"] for metrics in per_class_metrics.values()])
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
    
    return accuracy, macro_precision, macro_recall, macro_f1, per_class_metrics

def print_analysis(file_path: str):
    """
    Print a comprehensive analysis of the confusion matrix
    
    Args:
        file_path: Path to the .npy file containing the confusion matrix
    """
    try:
        accuracy, precision, recall, f1, class_metrics = analyze_confusion_matrix(file_path)
        
        print("Overall Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-Precision: {precision:.4f}")
        print(f"Macro-Recall: {recall:.4f}")
        print(f"Macro-F1: {f1:.4f}")
        print("\nPer-Class Metrics:")
        
        # for class_name, metrics in class_metrics.items():
        #     print(f"\n{class_name}:")
        #     print(f"  Precision: {metrics['precision']:.4f}")
        #     print(f"  Recall: {metrics['recall']:.4f}")
        #     print(f"  F1-Score: {metrics['f1_score']:.4f}")
        #     print(f"  Support: {metrics['support']}")
            
    except Exception as e:
        print(f"Error analyzing confusion matrix: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Replace with your .npy file path
    file_path = "confusion_matrices/resnet18/caltech256/gs_1.0_1.0_8/iter_1.npy"
    print_analysis(file_path)