#!/bin/bash

# Setup script for Privacy-Preserving Federated Learning

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up Privacy-Preserving Federated Learning environment...${NC}"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d '.' -f 1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d '.' -f 2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 6 ]); then
    echo -e "${RED}Error: Python 3.6 or higher is required. You have $PYTHON_VERSION.${NC}"
    exit 1
fi

echo -e "${GREEN}Python $PYTHON_VERSION detected.${NC}"

# Create a virtual environment
echo -e "${YELLOW}Creating a virtual environment...${NC}"
python3 -m venv venv

# Activate the virtual environment
echo -e "${YELLOW}Activating the virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Install package in development mode
echo -e "${YELLOW}Installing fedlearn package in development mode...${NC}"
pip install -e .

# Create directories for experiments
echo -e "${YELLOW}Creating experiment directories...${NC}"
mkdir -p experiments/prb_validation_resnet
mkdir -p experiments/optimal_privacy_robustness_resnet
mkdir -p experiments/prb_guided

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${YELLOW}To activate the environment, run:${NC}"
echo -e "    source venv/bin/activate"
echo -e "${YELLOW}To run experiments, use:${NC}"
echo -e "    ./run_experiments.sh" 