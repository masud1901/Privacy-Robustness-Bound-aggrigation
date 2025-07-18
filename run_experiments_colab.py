#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run federated learning experiments in Google Colab.
This script manages the execution of experiments.
"""

import os
import time
import argparse
import subprocess
import sys

# Import colored print function from colab_setup.py if available
try:
    from colab_setup import print_colored
except ImportError:
    # Define print_colored if import fails
    def print_colored(text, color="white"):
        """Print colored text in Colab."""
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "purple": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "end": "\033[0m"
        }
        print(f"{colors.get(color, colors['white'])}{text}{colors['end']}")

def run_command(cmd):
    """Run a shell command and handle errors."""
    try:
        process = subprocess.run(cmd, shell=True, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        print_colored(f"Command failed: {cmd}", "red")
        print_colored(f"Error: {e.stderr}", "red")
        return False, e.stderr

def check_colab():
    """Check if running in Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def run_experiment(experiment_num, rounds=150):
    """Run a specific experiment."""
    # Validate experiment number
    if experiment_num not in [1, 2, 3]:
        print_colored(f"Invalid experiment number: {experiment_num}", "red")
        return False
    
    experiment_names = {
        1: "PRB Validation with ResNet",
        2: "Optimal Privacy-Robustness with ResNet",
        3: "PRB-Guided Federated Learning"
    }
    
    # Print experiment information
    print_colored("\n" + "=" * 50, "blue")
    print_colored(f"Experiment {experiment_num}: {experiment_names[experiment_num]}", "blue")
    print_colored("=" * 50, "blue")
    print_colored(f"Communication rounds: {rounds}", "yellow")
    
    # Create experiment directories if they don't exist
    source_dirs = {
        1: "experiments/prb_validation_resnet",
        2: "experiments/optimal_privacy_robustness_resnet",
        3: "experiments/prb_guided"
    }
    os.makedirs(source_dirs[experiment_num], exist_ok=True)
    
    # Run the experiment
    start_time = time.time()
    cmd = f"python -m experiments.run_experiment{experiment_num} --communication_rounds {rounds}"
    success, output = run_command(cmd)
    
    if not success:
        print_colored(f"Experiment {experiment_num} failed!", "red")
        return False
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print_colored(f"Experiment {experiment_num} completed successfully!", "green")
    print_colored(f"Execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s", "yellow")
    print_colored(f"Results saved in {source_dirs[experiment_num]}", "green")
    
    return True

def run_all_experiments(experiments, rounds=150):
    """Run multiple experiments in sequence."""
    if not experiments:
        print_colored("No experiments specified to run.", "yellow")
        return
    
    total_start_time = time.time()
    results = {}
    
    print_colored(f"Starting experiments at {time.ctime()}", "yellow")
    
    for exp_num in experiments:
        results[exp_num] = run_experiment(exp_num, rounds)
    
    # Calculate total execution time
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    hours, remainder = divmod(total_execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Summary of results
    print_colored("\n" + "=" * 50, "blue")
    print_colored("Experiments Summary", "blue")
    print_colored("=" * 50, "blue")
    
    for exp_num, success in results.items():
        status = "Completed" if success else "Failed"
        color = "green" if success else "red"
        print_colored(f"Experiment {exp_num}: {status}", color)
    
    print_colored(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s", "yellow")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run federated learning experiments in Colab")
    parser.add_argument('-e', '--experiments', type=str, default="1,2,3",
                        help="Comma-separated list of experiment numbers to run (1,2,3)")
    parser.add_argument('-r', '--rounds', type=int, default=150,
                        help="Number of communication rounds (default: 150)")
    parser.add_argument('--quick', action='store_true',
                        help="Run quick tests with fewer rounds (10)")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Check if running in Google Colab
    in_colab = check_colab()
    if not in_colab:
        print_colored("Warning: Not running in Google Colab. Some features may not work.", "yellow")
    
    # Parse arguments
    args = parse_arguments()
    
    # Process experiments list
    experiments = [int(exp.strip()) for exp in args.experiments.split(',') if exp.strip().isdigit()]
    
    # Set rounds (use quick setting if specified)
    rounds = 10 if args.quick else args.rounds
    
    # Run experiments
    run_all_experiments(
        experiments=experiments,
        rounds=rounds
    ) 