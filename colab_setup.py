#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Privacy-Preserving Federated Learning in Google Colab
This script helps set up the environment and install the package in Colab.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

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

def run_command(cmd, desc=None):
    """Run a shell command and print its status."""
    if desc:
        print_colored(f"{desc}...", "yellow")
    
    try:
        process = subprocess.run(cmd, shell=True, check=True, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True)
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        print_colored(f"Error: {e.stderr}", "red")
        return False, e.stderr

def setup_colab_environment(install_package=True, create_dirs=True):
    """Setup the Colab environment for federated learning experiments."""
    
    print_colored("Setting up Privacy-Preserving FL environment in Colab...", "yellow")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print_colored(f"Python {python_version} detected.", "green")
    
    # Install PyTorch with CUDA support
    run_command("pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117", 
                "Installing PyTorch with CUDA support")
    
    # Install required packages
    run_command("pip install numpy matplotlib pandas scikit-learn tqdm", 
                "Installing common ML packages")
    
    # Clone the repository if not already available
    if install_package:
        repo_path = Path.cwd()
        repo_url = "https://github.com/yourusername/privacy_preserving_FL.git"
        
        if not (repo_path / "fedlearn").exists():
            success, _ = run_command(f"git clone {repo_url} /content/privacy_preserving_FL", 
                                    "Cloning repository")
            if success:
                os.chdir("/content/privacy_preserving_FL")
                run_command("pip install -e .", "Installing package in development mode")
            else:
                print_colored("Failed to clone repository. Please provide the repo URL.", "red")
                print_colored("You can run: !git clone YOUR_REPO_URL", "yellow")
        else:
            print_colored("Repository already exists.", "green")
            run_command("pip install -e .", "Installing package in development mode")
    
    # Create experiment directories
    if create_dirs:
        dirs = [
            "experiments/prb_validation_resnet",
            "experiments/optimal_privacy_robustness_resnet",
            "experiments/prb_guided"
        ]
        
        for directory in dirs:
            # Create directories on local Colab filesystem
            run_command(f"mkdir -p {directory}", f"Creating {directory}")
    
    print_colored("Setup completed successfully!", "green")
    print_colored("You can now run the experiments using run_experiments_colab.py", "yellow")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup Colab environment for federated learning")
    parser.add_argument('--no-install', action='store_true', help="Don't install the package")
    parser.add_argument('--no-dirs', action='store_true', help="Don't create experiment directories")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    setup_colab_environment(
        install_package=not args.no_install,
        create_dirs=not args.no_dirs
    ) 