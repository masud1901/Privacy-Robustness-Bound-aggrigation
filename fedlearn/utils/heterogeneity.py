#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for measuring client data heterogeneity in federated learning.
"""

import numpy as np
from collections import Counter
import torch
import ot  # Optimal Transport library


def compute_earth_movers_distance(client_datasets):
    """
    Compute Earth Mover's Distance (Wasserstein distance) between client datasets.
    
    Args:
        client_datasets (list): List of client datasets
    
    Returns:
        float: The average Earth Mover's Distance across all client pairs
    """
    # Extract label distributions for each client
    distributions = []
    for dataset in client_datasets:
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
            # For Subset wrapper around torchvision dataset
            indices = dataset.indices
            targets = np.array(dataset.dataset.targets)[indices]
        elif hasattr(dataset, 'targets'):
            # For direct torchvision dataset
            targets = np.array(dataset.targets)
        else:
            # For custom datasets, extract targets by iterating
            targets = np.array([target for _, target in dataset])
        
        # Compute class distribution
        counter = Counter(targets)
        total = sum(counter.values())
        # Create a normalized distribution array
        num_classes = max(counter.keys()) + 1
        dist = np.zeros(num_classes)
        for cls, count in counter.items():
            dist[cls] = count / total
        distributions.append(dist)
    
    # Compute pairwise EMD between all client distributions
    num_clients = len(distributions)
    total_emd = 0.0
    num_pairs = 0
    
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            dist_i = distributions[i]
            dist_j = distributions[j]
            # Use 1D Wasserstein distance for label distributions
            emd = ot.wasserstein_1d(
                np.arange(len(dist_i)), np.arange(len(dist_j)), 
                dist_i, dist_j
            )
            total_emd += emd
            num_pairs += 1
    
    # Return average EMD
    if num_pairs > 0:
        return total_emd / num_pairs
    return 0.0


def compute_label_skew(client_datasets):
    """
    Compute label skew metrics for client datasets.
    
    Args:
        client_datasets (list): List of client datasets
    
    Returns:
        dict: Dictionary containing label skew metrics
    """
    # Extract label distributions
    distributions = []
    for dataset in client_datasets:
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
            indices = dataset.indices
            targets = np.array(dataset.dataset.targets)[indices]
        elif hasattr(dataset, 'targets'):
            targets = np.array(dataset.targets)
        else:
            targets = np.array([target for _, target in dataset])
        
        counter = Counter(targets)
        distributions.append(counter)
    
    # Compute metrics
    num_clients = len(distributions)
    num_classes = max(max(d.keys()) for d in distributions) + 1
    
    # Compute Gini coefficient for each class
    gini_coeffs = []
    for cls in range(num_classes):
        class_counts = np.array([d.get(cls, 0) for d in distributions])
        if np.sum(class_counts) == 0:
            continue
        class_props = class_counts / np.sum(class_counts)
        class_props = np.sort(class_props)
        
        # Compute Gini coefficient
        n = len(class_props)
        gini = 2 * np.sum([(i+1) * x for i, x in enumerate(class_props)])
        gini = gini / (n * np.sum(class_props)) - (n + 1) / n
        gini_coeffs.append(gini)
    
    # Metrics to return
    metrics = {
        'avg_gini_coeff': np.mean(gini_coeffs),
        'max_gini_coeff': np.max(gini_coeffs),
        'min_gini_coeff': np.min(gini_coeffs)
    }
    
    return metrics 