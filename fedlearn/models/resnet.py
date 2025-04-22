#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ResNet model implementations for federated learning experiments.
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic block for ResNet18 and ResNet34."""
    
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, 
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_Pruned(nn.Module):
    """
    ResNet18 with pruning for federated learning experiments.
    
    This is a slightly pruned version of ResNet18 to reduce computational overhead
    while maintaining good performance for CIFAR-10 classification.
    """
    
    def __init__(self, input_shape=(3, 32, 32), num_classes=10, 
                 pruning_factor=0.5):
        """
        Initialize the pruned ResNet18 model.
        
        Args:
            input_shape (tuple): Shape of input images (channels, height, width)
            num_classes (int): Number of output classes
            pruning_factor (float): Factor to reduce number of filters (0.5 = half)
        """
        super(ResNet18_Pruned, self).__init__()
        self.init_args = (input_shape,)
        self.init_kwargs = {
            'num_classes': num_classes, 
            'pruning_factor': pruning_factor
        }
        
        # Extract input dimensions
        self.in_planes = int(64 * pruning_factor)
        channels, _, _ = input_shape
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(
            channels, self.in_planes, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        # ResNet layers
        self.layer1 = self._make_layer(
            BasicBlock, int(64 * pruning_factor), 2, stride=1
        )
        self.layer2 = self._make_layer(
            BasicBlock, int(128 * pruning_factor), 2, stride=2
        )
        self.layer3 = self._make_layer(
            BasicBlock, int(256 * pruning_factor), 2, stride=2
        )
        self.layer4 = self._make_layer(
            BasicBlock, int(512 * pruning_factor), 2, stride=2
        )
        
        # Final fully connected layer
        self.linear = nn.Linear(int(512 * pruning_factor), num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the ResNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_parameters(self):
        """
        Get model parameters.
        
        Returns:
            list: List of model parameters
        """
        return [param for param in self.parameters()] 