#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run Experiment 1: PRB Validation
"""

import os
import sys
import subprocess

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the base command
command = [
    "python", "-m", "fedlearn.experiment_runner",
    "--dataset", "cifar10",
    "--model", "resnet18_pruned",
    "--num_clients", "50",
    "--alpha", "0.5",
    "--heterogeneity_measure", "earth_mover_distance",
    "--aggregator", "fedavg",
    "--privacy_mechanism", "dp",
    "--epsilon", "{0.1,1.0,10.0}",
    "--attack_type", "model_replacement",
    "--poisoned_client_fraction", "{0.1,0.3}",
    "--measure_model_deviation", "true",
    "--measure_computational_overhead", "true",
    "--prb_bound_validation", "true",
    "--communication_rounds", "150",
    "--output_dir", "experiments/prb_validation_resnet"
]


def main():
    """Run the experiment."""
    print("Starting Experiment 1: PRB Validation with ResNet-18")
    print("Command:", " ".join(command))
    
    # Run the command
    subprocess.run(command, check=True)
    
    print("Experiment 1 completed.")


if __name__ == "__main__":
    main() 