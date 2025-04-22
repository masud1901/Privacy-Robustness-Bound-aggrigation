#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run all three PRB validation experiments.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run all PRB experiments')
    parser.add_argument('--exp', type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help='Which experiment to run (1, 2, 3, or all)')
    return parser.parse_args()


def run_experiment(script_path):
    """Run a single experiment script."""
    print(f"Running experiment script: {script_path}")
    start_time = time.time()
    
    try:
        subprocess.run(["python", script_path], check=True)
        elapsed_time = time.time() - start_time
        print(
            f"Experiment completed successfully in {elapsed_time:.2f} seconds."
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")
        return False


def main():
    """Run the specified experiments."""
    args = parse_args()
    
    # Get the path to the experiments directory
    experiments_dir = Path(__file__).parent.absolute()
    
    # Define experiment scripts
    experiment_scripts = {
        "1": experiments_dir / "run_experiment1.py",
        "2": experiments_dir / "run_experiment2.py",
        "3": experiments_dir / "run_experiment3.py"
    }
    
    # Determine which experiments to run
    if args.exp == "all":
        experiments_to_run = ["1", "2", "3"]
    else:
        experiments_to_run = [args.exp]
    
    # Run the selected experiments
    results = {}
    for exp_id in experiments_to_run:
        script_path = experiment_scripts[exp_id]
        print(f"\n{'='*80}\nStarting Experiment {exp_id}\n{'='*80}")
        success = run_experiment(script_path)
        results[exp_id] = "Success" if success else "Failed"
    
    # Print summary
    print("\n")
    print("="*40)
    print("Experiment Execution Summary")
    print("="*40)
    for exp_id, result in results.items():
        print(f"Experiment {exp_id}: {result}")
    print("="*40)


if __name__ == "__main__":
    main() 