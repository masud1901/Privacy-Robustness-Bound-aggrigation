# PRB Validation Experiments with ResNet-18

This directory contains scripts to run the Privacy-Robustness Bound (PRB) validation experiments as described in the main project README, with ResNet-18 as the benchmark model for all experiments.

## Available Scripts

- `run_experiment1.py` - Runs Experiment 1: PRB Validation with ResNet-18
- `run_experiment2.py` - Runs Experiment 2: Optimal Privacy-Robustness Trade-off with ResNet-18
- `run_experiment3.py` - Runs Experiment 3: PRB-Guided Federated Learning with ResNet-18
- `run_all.py` - Runs all three experiments sequentially or allows running a specific experiment

## How to Run

### Running a Specific Experiment

To run a specific experiment, use one of the following commands:

```bash
# Run Experiment 1: PRB Validation
python run_experiment1.py

# Run Experiment 2: Optimal Privacy-Robustness Trade-off
python run_experiment2.py

# Run Experiment 3: PRB-Guided Federated Learning
python run_experiment3.py
```

### Running All Experiments

To run all experiments sequentially:

```bash
python run_all.py
```

To run a specific experiment using the run_all.py script:

```bash
python run_all.py --exp 1  # Run only Experiment 1
python run_all.py --exp 2  # Run only Experiment 2
python run_all.py --exp 3  # Run only Experiment 3
```

## Output

Results will be saved in the following directories:

- Experiment 1: `experiments/prb_validation_resnet/`
- Experiment 2: `experiments/optimal_privacy_robustness_resnet/`
- Experiment 3: `experiments/prb_guided/` (already using ResNet-18)

Each experiment will create a directory structure with JSON files containing the results of each run, as well as a combined results file.

## Experiment Details

### Experiment 1: PRB Validation with ResNet-18

Objective: Verify that model deviation under attack respects our theoretical PRB bound.

- Dataset: CIFAR-10
- Model: **ResNet-18 (pruned)**
- Privacy levels: ε ∈ {0.1, 1.0, 10.0}
- Attack fractions: τ ∈ {0.1, 0.3}
- Total runs: 15

### Experiment 2: Optimal Privacy-Robustness Trade-off with ResNet-18

Objective: Demonstrate that operating at ε* yields optimal balance between privacy and robustness.

- Dataset: CIFAR-10 (with α=0.3 for non-IID partitioning)
- Model: **ResNet-18 (pruned)**
- Privacy levels: 5 ε values centered around theoretical ε*
- Attack: Label flipping with τ = 0.2
- Total runs: 15

### Experiment 3: PRB-Guided Federated Learning with ResNet-18

Objective: Demonstrate that PRB-guided aggregation improves over baselines.

- Dataset: CIFAR-10
- Model: **ResNet-18 (pruned)**
- Aggregation methods: FedAvg, Multi-Krum, DP-FedAvg, PBA, PRB-guided
- Attack scenarios: No attack, Model replacement (20%), Mixed attack, High-fraction attack (30%)
- Total runs: 16 