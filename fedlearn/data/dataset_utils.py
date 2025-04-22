#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset utilities for federated learning experiments.
"""

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset


def load_federated_dataset(dataset_name, num_clients, alpha=0.5):
    """
    Load dataset and partition it for federated learning.
    
    Args:
        dataset_name (str): Name of dataset ('cifar10', 'femnist')
        num_clients (int): Number of clients
        alpha (float): Dirichlet parameter controlling data heterogeneity
                      (lower alpha = more heterogeneous)
    
    Returns:
        Tuple: (train_dataset, test_dataset, client_datasets)
            - train_dataset: Full training dataset
            - test_dataset: Full test dataset
            - client_datasets: List of client datasets
    """
    if dataset_name == 'cifar10':
        return load_cifar10_federated(num_clients, alpha)
    elif dataset_name == 'femnist':
        raise NotImplementedError("FEMNIST dataset not yet implemented")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def load_cifar10_federated(num_clients, alpha=0.5):
    """
    Load CIFAR-10 dataset and partition it for federated learning.
    
    Args:
        num_clients (int): Number of clients
        alpha (float): Dirichlet parameter controlling data heterogeneity
                      (lower alpha = more heterogeneous)
    
    Returns:
        Tuple: (train_dataset, test_dataset, client_datasets)
    """
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Partition the dataset using Dirichlet distribution for non-IID data
    client_datasets = create_non_iid_partition(
        train_dataset, num_clients, alpha=alpha
    )
    
    return train_dataset, test_dataset, client_datasets


def create_non_iid_partition(dataset, num_clients, alpha=0.5):
    """
    Create a non-IID partition of the dataset using Dirichlet distribution.
    
    Args:
        dataset: Dataset to partition
        num_clients (int): Number of clients
        alpha (float): Dirichlet parameter (lower = more heterogeneous)
    
    Returns:
        List[Dataset]: List of client datasets
    """
    # Get the labels of all samples
    if isinstance(dataset, torchvision.datasets.CIFAR10):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
    
    # Number of classes
    num_classes = len(np.unique(labels))
    
    # Create indices array for each class
    class_indices = [np.where(labels == class_idx)[0] for class_idx in range(num_classes)]
    
    # Allocate samples to clients using Dirichlet distribution
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class
    for class_idx in range(num_classes):
        # Sample from Dirichlet to determine class distribution across clients
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Get the indices for this class
        indices = class_indices[class_idx]
        
        # Shuffle the indices
        np.random.shuffle(indices)
        
        # Split the indices according to the proportions
        proportions = proportions / proportions.sum()  # Normalize
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)
        
        # Assign indices to clients
        start_idx = 0
        for client_idx in range(num_clients):
            end_idx = proportions[client_idx]
            client_indices[client_idx].extend(indices[start_idx:end_idx])
            start_idx = end_idx
    
    # Create client datasets using the indices
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    
    return client_datasets 