import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    Caltech256Dataset, 
    CIFAR100Dataset, 
    MITIndoor67Dataset
)

def analyze_dataset_distribution(datasets, name):
    """
    Analyze and plot the distribution of images across classes for combined datasets
    """
    # Combine labels from all datasets
    all_labels = []
    for dataset in datasets:
        all_labels.extend(dataset.labels)
    
    # Count images per class
    class_counts = Counter(all_labels)
    
    # Sort counts for better visualization
    counts = sorted(class_counts.values(), reverse=True)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(counts)), counts)
    
    # Customize the plot
    plt.title(f'Class Distribution in {name} Dataset (Train + Test)')
    plt.xlabel('Class Index (sorted by frequency)')
    plt.ylabel('Number of Images per Class')
    
    # Add summary statistics
    mean_count = np.mean(counts)
    median_count = np.median(counts)
    plt.axhline(y=mean_count, color='r', linestyle='--', label=f'Mean: {mean_count:.1f}')
    plt.axhline(y=median_count, color='g', linestyle='--', label=f'Median: {median_count:.1f}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(f'{name.lower()}_distribution.png')
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary for {name} Dataset (Train + Test):")
    print(f"Total number of classes: {len(class_counts)}")
    print(f"Total number of images: {sum(counts)}")
    print(f"Average images per class: {mean_count:.1f}")
    print(f"SD images per class: {np.std(counts)}")
    print(f"Median images per class: {median_count:.1f}")
    print(f"Min images in a class: {min(counts)}")
    print(f"Max images in a class: {max(counts)}")

def main():
    # Define dataset paths for both train and test
    datasets_config = [
        (Caltech256Dataset, 
         ["data/caltech256/train", "data/caltech256/test"], 
         "Caltech-256"),
        (CIFAR100Dataset, 
         ["data/cifar100/train", "data/cifar100/test"], 
         "CIFAR-100"),
        (MITIndoor67Dataset, 
         ["data/mit67/train", "data/mit67/test"], 
         "MIT Indoor-67")
    ]
    
    # Analyze each dataset
    for dataset_class, paths, name in datasets_config:
        try:
            print(f"\nAnalyzing {name}...")
            # Load both train and test datasets
            datasets = []
            for path in paths:
                try:
                    dataset = dataset_class(root_dir=path)
                    datasets.append(dataset)
                    print(f"Successfully loaded {path}")
                except Exception as e:
                    print(f"Warning: Error loading {path}: {str(e)}")
            
            if datasets:  # If we successfully loaded at least one dataset
                analyze_dataset_distribution(datasets, name)
            else:
                print(f"Error: Could not load any data for {name}")
                
        except Exception as e:
            print(f"Error analyzing {name} dataset: {str(e)}")

if __name__ == "__main__":
    main()