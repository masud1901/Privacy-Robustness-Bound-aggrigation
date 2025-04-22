#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Differentially Private SGD for federated learning.

This module provides functions to apply differential privacy to model updates
in federated learning settings, based on the DP-SGD algorithm by Abadi et al.
"""

import torch
import torch.nn as nn
import numpy as np


def compute_noise_multiplier(epsilon, delta=1e-5, sample_rate=0.01, iterations=100):
    """
    Compute the noise multiplier for DP-SGD based on the privacy budget.
    
    This is a simplified implementation based on the moments accountant method
    from Abadi et al., 2016.
    
    Args:
        epsilon (float): The privacy budget.
        delta (float): The failure probability.
        sample_rate (float): The sampling rate in SGD.
        iterations (int): The number of iterations.
        
    Returns:
        float: The noise multiplier to use in DP-SGD.
    """
    # This is a simplified approximation
    c = np.sqrt(2 * np.log(1.25 / delta))
    return c * sample_rate * np.sqrt(iterations) / epsilon


def clip_gradients(model, max_norm):
    """
    Clip the gradients of a model to a maximum L2 norm.
    
    Args:
        model (nn.Module): The model whose gradients will be clipped.
        max_norm (float): The maximum allowed L2 norm.
        
    Returns:
        float: The scaling factor used for clipping.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    scaling_factor = max_norm / (total_norm + 1e-8)
    scaling_factor = min(1.0, scaling_factor)
    
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(scaling_factor)
    
    return scaling_factor


def add_noise(model, noise_scale):
    """
    Add Gaussian noise to the gradients of a model.
    
    Args:
        model (nn.Module): The model whose gradients will be noised.
        noise_scale (float): The scale of the noise to add.
        
    Returns:
        nn.Module: The model with noised gradients.
    """
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * noise_scale
                p.grad.data.add_(noise)
    
    return model


def apply_dp_sgd(model, epsilon, delta=1e-5, max_grad_norm=1.0, sample_rate=0.01, iterations=100):
    """
    Apply Differentially Private SGD to a model.
    
    Args:
        model (nn.Module): The model to apply DP-SGD to.
        epsilon (float): The privacy budget.
        delta (float): The failure probability.
        max_grad_norm (float): The maximum L2 norm for gradient clipping.
        sample_rate (float): The sampling rate in SGD.
        iterations (int): The number of iterations.
        
    Returns:
        nn.Module: The model with DP-SGD applied.
    """
    # Compute noise scale based on privacy budget
    noise_multiplier = compute_noise_multiplier(epsilon, delta, sample_rate, iterations)
    noise_scale = noise_multiplier * max_grad_norm
    
    # Clip gradients
    clip_gradients(model, max_grad_norm)
    
    # Add noise to gradients
    model = add_noise(model, noise_scale)
    
    return model


def compute_privacy_spent(noise_multiplier, sample_rate, iterations, delta=1e-5):
    """
    Compute the privacy spent (epsilon) based on the parameters used.
    
    This is a simplified implementation for demonstration purposes.
    
    Args:
        noise_multiplier (float): The noise multiplier used.
        sample_rate (float): The sampling rate in SGD.
        iterations (int): The number of iterations.
        delta (float): The failure probability.
        
    Returns:
        float: The approximate epsilon value spent.
    """
    # This is a simplified approximation
    c = np.sqrt(2 * np.log(1.25 / delta))
    epsilon = noise_multiplier * sample_rate * np.sqrt(iterations) / c
    
    return epsilon 