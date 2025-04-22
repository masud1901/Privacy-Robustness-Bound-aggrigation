#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics utilities for federated learning experiments.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader


def compute_test_accuracy(model, test_dataset, batch_size=128, device=None):
    """
    Compute the accuracy of a model on a test dataset.
    
    Args:
        model (nn.Module): The model to evaluate
        test_dataset: The test dataset
        batch_size (int): Batch size for evaluation
        device: Device to use for evaluation
    
    Returns:
        float: Test accuracy
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return correct / total


def compute_model_deviation(model_a, model_b):
    """
    Compute the L2 norm of the difference between two models.
    
    Args:
        model_a (nn.Module): First model
        model_b (nn.Module): Second model
    
    Returns:
        float: L2 norm of the parameter difference
    """
    squared_sum = 0.0
    
    # Iterate through parameters of both models
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        # Ensure we're comparing corresponding parameters
        assert name_a == name_b, f"Parameter names don't match: {name_a} vs {name_b}"
        
        # Compute squared difference and add to sum
        diff = param_a.data - param_b.data
        squared_sum += torch.sum(diff * diff).item()
    
    # Return L2 norm (square root of sum of squared differences)
    return torch.sqrt(torch.tensor(squared_sum)).item()


def evaluate_membership_inference(model, train_dataset, test_dataset):
    """
    Evaluate privacy leakage using membership inference.
    
    Args:
        model (nn.Module): The model to evaluate
        train_dataset: The training dataset
        test_dataset: The test dataset
    
    Returns:
        float: AUC score for membership inference
    """
    # Simple implementation of membership inference 
    # based on confidence of predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    
    # Get confidence scores for training samples (members)
    member_scores = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # Get the probability of the correct class
            confidence = probabilities[range(len(targets)), targets]
            member_scores.extend(confidence.cpu().numpy())
    
    # Get confidence scores for test samples (non-members)
    non_member_scores = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # Get the probability of the correct class
            confidence = probabilities[range(len(targets)), targets]
            non_member_scores.extend(confidence.cpu().numpy())
    
    # Compute AUC score
    from sklearn.metrics import roc_auc_score
    
    # Create labels (1 for members, 0 for non-members)
    member_labels = np.ones(len(member_scores))
    non_member_labels = np.zeros(len(non_member_scores))
    
    # Combine scores and labels
    all_scores = np.concatenate([member_scores, non_member_scores])
    all_labels = np.concatenate([member_labels, non_member_labels])
    
    # Compute AUC score
    auc_score = roc_auc_score(all_labels, all_scores)
    
    return auc_score 