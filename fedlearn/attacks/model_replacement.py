#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of model replacement attacks for federated learning.

This module provides functions to simulate model replacement attacks, where
malicious clients replace their model updates with malicious ones to corrupt
the global model.
"""

import torch
import numpy as np
import copy


def model_replacement_attack(client_models, num_malicious, global_model, 
                             scale_factor=10.0, targeted=False, target_label=0):
    """
    Perform a model replacement attack.
    
    In a model replacement attack, malicious clients replace their model updates
    with malicious ones designed to corrupt the global model. This can be done
    either to reduce the overall accuracy (untargeted) or to cause misclassification
    of specific classes (targeted).
    
    Args:
        client_models (list): List of client models
        num_malicious (int): Number of malicious clients
        global_model (nn.Module): The current global model
        scale_factor (float): Factor by which to scale the malicious updates
        targeted (bool): Whether the attack is targeted or untargeted
        target_label (int): The target label for misclassification in targeted attacks
        
    Returns:
        list: List of client models with malicious ones replaced
    """
    # Make a copy of client models to avoid modifying the original list
    modified_client_models = copy.deepcopy(client_models)
    
    # Select malicious clients randomly
    malicious_indices = np.random.choice(
        len(client_models), num_malicious, replace=False
    )
    
    # Generate malicious models
    for idx in malicious_indices:
        if targeted:
            # Targeted attack: modify the model to misclassify to target_label
            modified_client_models[idx] = create_targeted_malicious_model(
                global_model, target_label, scale_factor
            )
        else:
            # Untargeted attack: corrupt the model to reduce overall accuracy
            modified_client_models[idx] = create_untargeted_malicious_model(
                global_model, scale_factor
            )
    
    return modified_client_models


def create_untargeted_malicious_model(global_model, scale_factor=10.0):
    """
    Create an untargeted malicious model by scaling gradients to affect 
    overall model performance.
    
    Args:
        global_model (nn.Module): The current global model
        scale_factor (float): Factor by which to scale the malicious updates
        
    Returns:
        nn.Module: A malicious model
    """
    # Copy the global model
    malicious_model = copy.deepcopy(global_model)
    
    # For each parameter, scale it to create a large update in a random direction
    with torch.no_grad():
        for param in malicious_model.parameters():
            # Generate random noise in the same shape as the parameter
            noise = torch.randn_like(param)
            
            # Scale the noise and add it to the parameter
            param.add_(noise * scale_factor)
    
    return malicious_model


def create_targeted_malicious_model(global_model, target_label, scale_factor=10.0):
    """
    Create a targeted malicious model aimed at misclassifying inputs to a target label.
    
    Args:
        global_model (nn.Module): The current global model
        target_label (int): The target label for misclassification
        scale_factor (float): Factor by which to scale the malicious updates
        
    Returns:
        nn.Module: A malicious model targeting a specific class
    """
    # Copy the global model
    malicious_model = copy.deepcopy(global_model)
    
    # Modify the last layer to favor the target class
    with torch.no_grad():
        # Find the last layer (assumes it's a linear layer for classification)
        for name, param in reversed(list(malicious_model.named_parameters())):
            if 'weight' in name and len(param.shape) == 2:
                # This is likely the weight of the final classifier layer
                # Modify it to favor the target class
                num_classes = param.shape[0]
                
                # Make the target class weights much larger
                for c in range(num_classes):
                    if c == target_label:
                        # Increase weights for target class
                        param[c, :] *= (1 + scale_factor)
                    else:
                        # Decrease weights for other classes
                        param[c, :] *= (1 - scale_factor / num_classes)
                
                break
    
    return malicious_model 