#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN model implementations for federated learning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Small(nn.Module):
    """
    Small CNN with 3 convolutional layers and 2 fully connected layers.
    
    This model is used for CIFAR-10 classification in the PRB validation experiments.
    """
    
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Shape of input images (channels, height, width)
            num_classes (int): Number of output classes
        """
        super(CNN_Small, self).__init__()
        self.init_args = (input_shape,)
        self.init_kwargs = {'num_classes': num_classes}
        
        # Extract input dimensions
        channels, height, width = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2)
        
        # Calculate size after convolutions and pooling
        conv_output_size = width // 4 * height // 4 * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Forward pass of the CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_parameters(self):
        """
        Get model parameters.
        
        Returns:
            list: List of model parameters
        """
        return [param for param in self.parameters()] 