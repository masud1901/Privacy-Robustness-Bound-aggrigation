#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for computing the Privacy-Robustness Bound (PRB).
"""

import numpy as np


def compute_theoretical_prb(epsilon, tau, delta, model_dimension, num_clients):
    """
    Compute the theoretical Privacy-Robustness Bound (PRB).
    
    Args:
        epsilon (float): The privacy budget
        tau (float): The fraction of malicious clients
        delta (float): The failure probability
        model_dimension (int): The dimension of the model
        num_clients (int): The number of clients
    
    Returns:
        float: The theoretical PRB value
    """
    # Constants for PRB calculation
    C1 = 0.5  # Constant related to privacy impact
    C2 = 1.0  # Constant related to Byzantine impact
    C3 = 0.1  # Constant related to concentration bound
    
    # The Privacy-Robustness Bound formula
    privacy_term = C1 * (epsilon + 1 / epsilon)
    robustness_term = C2 * tau * np.sqrt(model_dimension / num_clients)
    concentration_term = C3 * np.sqrt(np.log(1 / delta) / num_clients)
    
    prb = privacy_term + robustness_term + concentration_term
    
    return prb


def compute_optimal_epsilon(tau, delta, model_dimension, num_clients):
    """
    Compute the optimal privacy budget for a given robustness requirement.
    
    Args:
        tau (float): The fraction of malicious clients
        delta (float): The failure probability
        model_dimension (int): The dimension of the model
        num_clients (int): The number of clients
    
    Returns:
        float: The optimal privacy budget
    """
    # Constants for optimal epsilon calculation
    C1 = 0.5  # Same constant as in PRB calculation
    C2 = 1.0  # Same constant as in PRB calculation
    
    # Based on the theoretical derivation in the paper
    robustness_term = C2 * tau * np.sqrt(model_dimension / num_clients)
    
    # The optimal epsilon formula from Corollary 1
    eps_star = np.sqrt(C1 / robustness_term)
    
    # Ensure epsilon is within reasonable bounds
    eps_star = max(0.1, min(10.0, eps_star))
    
    return eps_star


def compute_prb_guided_weights(client_contributions, client_deviations, epsilon):
    """
    Compute client weights for PRB-guided aggregation.
    
    Args:
        client_contributions (list): Contributions of each client to model performance
        client_deviations (list): Deviations of each client model from global model
        epsilon (float): The privacy budget
    
    Returns:
        list: The computed weights for each client
    """
    # Convert inputs to numpy arrays
    contributions = np.array(client_contributions)
    deviations = np.array(client_deviations)
    
    # Normalize contributions and deviations
    contributions = contributions / (np.sum(contributions) + 1e-10)
    deviations = deviations / (np.max(deviations) + 1e-10)
    
    # Compute trust scores based on contributions and deviations
    trust_scores = contributions / (deviations + 1e-10)
    
    # Apply privacy regularization based on epsilon
    # Higher epsilon means less privacy concern, so we apply less regularization
    privacy_factor = 1.0 / (1.0 + np.exp(-epsilon + 5.0))
    weights = trust_scores * privacy_factor
    
    # Normalize weights
    weights = weights / (np.sum(weights) + 1e-10)
    
    return weights.tolist() 