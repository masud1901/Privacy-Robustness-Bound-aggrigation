#!/bin/bash

# Experiments runner script for Privacy-Preserving Federated Learning

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RUN_EXP1=true
RUN_EXP2=true
RUN_EXP3=true
ROUNDS=150
GPU_ID=0

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -e, --experiments <list>  Comma-separated list of experiment numbers to run (e.g. 1,2,3)"
    echo "  -r, --rounds <number>     Number of communication rounds (default: 150)"
    echo "  -g, --gpu <id>            GPU ID to use (default: 0)"
    echo "  --exp1                    Run only Experiment 1 (PRB Validation)"
    echo "  --exp2                    Run only Experiment 2 (Optimal Privacy-Robustness)"
    echo "  --exp3                    Run only Experiment 3 (PRB-Guided FL)"
    echo "  --quick                   Run quick tests with fewer rounds (10)"
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -e|--experiments)
            IFS=',' read -ra EXPS <<< "$2"
            RUN_EXP1=false
            RUN_EXP2=false
            RUN_EXP3=false
            for exp in "${EXPS[@]}"; do
                if [[ $exp == "1" ]]; then RUN_EXP1=true; fi
                if [[ $exp == "2" ]]; then RUN_EXP2=true; fi
                if [[ $exp == "3" ]]; then RUN_EXP3=true; fi
            done
            shift 2
            ;;
        -r|--rounds)
            ROUNDS="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --exp1)
            RUN_EXP1=true
            RUN_EXP2=false
            RUN_EXP3=false
            shift
            ;;
        --exp2)
            RUN_EXP1=false
            RUN_EXP2=true
            RUN_EXP3=false
            shift
            ;;
        --exp3)
            RUN_EXP1=false
            RUN_EXP2=false
            RUN_EXP3=true
            shift
            ;;
        --quick)
            ROUNDS=10
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $key${NC}"
            show_help
            ;;
    esac
done

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Please run setup.sh first.${NC}"
    exit 1
fi

# Activate the virtual environment
echo -e "${YELLOW}Activating the virtual environment...${NC}"
source venv/bin/activate

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run experiments
START_TIME=$(date +%s)
echo -e "${YELLOW}Starting experiments at $(date)${NC}"
echo -e "${YELLOW}Communication rounds: $ROUNDS${NC}"
echo -e "${YELLOW}Using GPU ID: $GPU_ID${NC}"

if $RUN_EXP1; then
    echo -e "\n${BLUE}==========================================${NC}"
    echo -e "${BLUE}Experiment 1: PRB Validation with ResNet${NC}"
    echo -e "${BLUE}==========================================${NC}"
    python -m experiments.run_experiment1 --communication_rounds $ROUNDS
    if [ $? -ne 0 ]; then
        echo -e "${RED}Experiment 1 failed!${NC}"
    else
        echo -e "${GREEN}Experiment 1 completed successfully!${NC}"
    fi
fi

if $RUN_EXP2; then
    echo -e "\n${BLUE}====================================================${NC}"
    echo -e "${BLUE}Experiment 2: Optimal Privacy-Robustness with ResNet${NC}"
    echo -e "${BLUE}====================================================${NC}"
    python -m experiments.run_experiment2 --communication_rounds $ROUNDS
    if [ $? -ne 0 ]; then
        echo -e "${RED}Experiment 2 failed!${NC}"
    else
        echo -e "${GREEN}Experiment 2 completed successfully!${NC}"
    fi
fi

if $RUN_EXP3; then
    echo -e "\n${BLUE}=================================================${NC}"
    echo -e "${BLUE}Experiment 3: PRB-Guided Federated Learning${NC}"
    echo -e "${BLUE}=================================================${NC}"
    python -m experiments.run_experiment3 --communication_rounds $ROUNDS
    if [ $? -ne 0 ]; then
        echo -e "${RED}Experiment 3 failed!${NC}"
    else
        echo -e "${GREEN}Experiment 3 completed successfully!${NC}"
    fi
fi

# Calculate and display total execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))
HOURS=$((EXECUTION_TIME / 3600))
MINUTES=$(( (EXECUTION_TIME % 3600) / 60 ))
SECONDS=$((EXECUTION_TIME % 60))

echo -e "\n${GREEN}All experiments completed!${NC}"
echo -e "${YELLOW}Total execution time: $HOURS hours, $MINUTES minutes, $SECONDS seconds${NC}"
echo -e "${YELLOW}Results are saved in the experiments directory.${NC}"

# Deactivate the virtual environment
deactivate 