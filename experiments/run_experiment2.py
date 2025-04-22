#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run Experiment 2: Optimal Privacy-Robustness Trade-off
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
    "--num_clients", "100",
    "--alpha", "0.3",
    "--heterogeneity_measure", "earth_mover_distance",
    "--aggregator", "fedavg",
    "--privacy_mechanism", "dp",
    "--epsilon_range", "0.2*optimal,0.5*optimal,optimal,2*optimal,5*optimal",
    "--attack_type", "label_flipping",
    "--poisoned_client_fraction", "0.2",
    "--measure_privacy_leakage", "membership_inference",
    "--optimal_epsilon_validation", "true",
    "--communication_rounds", "100",
    "--output_dir", "experiments/optimal_privacy_robustness_resnet"
]


def main():
    """Run the experiment."""
    print("Starting Experiment 2: Optimal Privacy-Robustness Trade-off "
          "with ResNet-18")
    print("Command:", " ".join(command))
    
    # Run the command
    subprocess.run(command, check=True)
    
    print("Experiment 2 completed.")


if __name__ == "__main__":
    main()