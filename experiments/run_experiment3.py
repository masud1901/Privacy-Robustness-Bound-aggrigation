#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run Experiment 3: PRB-Guided Federated Learning
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
    "--alpha", "0.1",
    "--heterogeneity_measure", "earth_mover_distance",
    "--aggregators", "fedavg,multi_krum,dp_fedavg,pba,prb_guided",
    "--attack_scenarios",
    "no_attack,model_replacement_20,mixed_attack,high_fraction_30",
    "--baseline_comparison", "true",
    "--measure_computational_overhead", "true",
    "--communication_rounds", "150",
    "--output_dir", "experiments/prb_guided"
]


def main():
    """Run the experiment."""
    print("Starting Experiment 3: PRB-Guided Federated Learning")
    print("Command:", " ".join(command))
    
    # Run the command
    subprocess.run(command, check=True)
    
    print("Experiment 3 completed.")


if __name__ == "__main__":
    main() 