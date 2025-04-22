#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Federated Averaging (FedAvg) algorithm for federated learning.

This module provides functions to aggregate client models using the FedAvg 
algorithm, which computes a weighted average of client model parameters.
"""

import torch
import copy
from collections import OrderedDict


def fedavg_aggregate(global_model, client_models, client_weights=None):
    """
    Aggregate client models using Federated Averaging (FedAvg).
    
    Args:
        global_model (nn.Module): The current global model
        client_models (list): List of client models
        client_weights (list, optional): List of weights for each client, 
                                         proportional to their dataset sizes.
                                         If None, equal weights are used.
    
    Returns:
        nn.Module: The aggregated global model
    """
    # Make a copy of the global model to store the aggregated parameters
    aggregated_model = copy.deepcopy(global_model)
    
    # If no client weights are provided, use equal weights
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    else:
        # Normalize weights to sum to 1
        weight_sum = sum(client_weights)
        client_weights = [w / weight_sum for w in client_weights]
    
    # Initialize a dictionary to store aggregated parameters
    aggregated_params = OrderedDict()
    
    # Get the state dict from the first client to initialize aggregated_params
    # with the correct parameter names and shapes
    for name, param in client_models[0].state_dict().items():
        # Use same dtype as the client parameter to avoid type mismatch
        aggregated_params[name] = torch.zeros_like(param, dtype=param.dtype)
    
    # Aggregate parameters from all clients with their respective weights
    for client_idx, client_model in enumerate(client_models):
        client_weight = client_weights[client_idx]
        client_params = client_model.state_dict()
        
        for name, param in client_params.items():
            # Ensure consistent data types for the addition operation
            weighted_param = param.data.to(dtype=aggregated_params[name].dtype) * client_weight
            aggregated_params[name] += weighted_param
    
    # Load the aggregated parameters into the model
    aggregated_model.load_state_dict(aggregated_params)
    
    return aggregated_model


# Alias for fedavg_aggregate to match function name used in experiment_runner.py
federated_averaging = fedavg_aggregate


def compute_model_difference(model_a, model_b):
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


def robust_fedavg(global_model, client_models, client_weights=None, 
                  outlier_threshold=2.0):
    """
    Robust version of FedAvg that filters out potential outliers.
    
    Args:
        global_model (nn.Module): The current global model
        client_models (list): List of client models
        client_weights (list, optional): List of weights for each client.
                                         If None, equal weights are used.
        outlier_threshold (float): Threshold for filtering outliers based on
                                   their distance from the median update.
    
    Returns:
        nn.Module: The aggregated global model after filtering outliers
    """
    # If we have too few clients, fall back to regular FedAvg
    if len(client_models) < 5:
        return fedavg_aggregate(global_model, client_models, client_weights)
    
    # Compute differences between each client model and the global model
    differences = [
        compute_model_difference(client_model, global_model)
        for client_model in client_models
    ]
    
    # Compute median difference
    median_diff = torch.median(torch.tensor(differences)).item()
    
    # Filter out clients with differences too far from the median
    filtered_models = []
    filtered_weights = []
    
    for i, diff in enumerate(differences):
        if diff <= outlier_threshold * median_diff:
            filtered_models.append(client_models[i])
            if client_weights is not None:
                filtered_weights.append(client_weights[i])
    
    # If no models passed the filter, use the original list
    if not filtered_models:
        filtered_models = client_models
        filtered_weights = client_weights
    
    # Aggregate the filtered models
    return fedavg_aggregate(global_model, filtered_models, filtered_weights)


def aggregate_client_weights(client_weights, client_sizes):
    """
    Aggregate client model weights using the Federated Averaging algorithm.

    Parameters
    ----------
    client_weights : list of dict
        List of client model weight dictionaries.
    client_sizes : list of int
        List of client dataset sizes used for weighted averaging.

    Returns
    -------
    dict
        Aggregated model weights.
    """
    # If no clients, return empty dict
    if not client_weights or not client_sizes:
        return {}

    # Get the total number of samples across all clients
    total_size = sum(client_sizes)

    # Initialize the aggregated weights with the structure of the first client's weights
    aggregated_weights = {}
    for key in client_weights[0].keys():
        aggregated_weights[key] = torch.zeros_like(client_weights[0][key])

    # Perform weighted aggregation of weights
    for client_idx, weights in enumerate(client_weights):
        weight = client_sizes[client_idx] / total_size
        for key in weights.keys():
            aggregated_weights[key] += weights[key] * weight

    return aggregated_weights 