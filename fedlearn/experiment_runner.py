#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main experiment runner for PRB validation experiments.
"""

import os
import json
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from fedlearn.models.cnn import CNN_Small
from fedlearn.models.resnet import ResNet18_Pruned
from fedlearn.data.dataset_utils import load_federated_dataset
from fedlearn.privacy.dp_sgd import apply_dp_sgd
from fedlearn.attacks.model_replacement import model_replacement_attack
from fedlearn.aggregation.fedavg import federated_averaging, robust_fedavg
from fedlearn.utils.metrics import compute_model_deviation, compute_test_accuracy, evaluate_membership_inference
from fedlearn.utils.heterogeneity import compute_earth_movers_distance
from fedlearn.utils.prb import compute_theoretical_prb, compute_optimal_epsilon, compute_prb_guided_weights

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Privacy-Robustness Trade-off Experiment Runner')
    
    # Dataset and model arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'femnist'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, required=True, choices=['cnn_small', 'resnet18_pruned'],
                        help='Model architecture to use')
    parser.add_argument('--num_clients', type=int, default=50,
                        help='Number of clients')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet parameter for non-IID data partitioning')
    
    # Privacy and attack arguments
    parser.add_argument('--heterogeneity_measure', type=str, default='earth_mover_distance',
                        help='Method to measure client heterogeneity')
    parser.add_argument('--aggregator', type=str, default='fedavg', 
                        choices=['fedavg', 'multi_krum', 'dp_fedavg', 'pba', 'prb_guided'],
                        help='Aggregation method to use')
    parser.add_argument('--aggregators', type=str, required=False,
                        help='Comma-separated list of aggregators to use in Experiment 3')
    parser.add_argument('--privacy_mechanism', type=str, default='dp',
                        choices=['dp', 'none'],
                        help='Privacy mechanism to use')
    parser.add_argument('--epsilon', type=str, required=False,
                        help='Privacy budget(s) to use, comma-separated for multiple values')
    parser.add_argument('--epsilon_range', type=str, required=False,
                        help='Range of privacy budgets relative to optimal epsilon')
    parser.add_argument('--attack_type', type=str, default='none',
                        choices=['none', 'model_replacement', 'label_flipping', 'mixed_attack'],
                        help='Type of attack to simulate')
    parser.add_argument('--attack_scenarios', type=str, required=False,
                        help='Attack scenarios to evaluate')
    parser.add_argument('--poisoned_client_fraction', type=str, required=False,
                        help='Fraction of clients that are malicious')
    
    # Experiment configuration
    parser.add_argument('--measure_model_deviation', type=bool, default=False,
                        help='Whether to measure model deviation from optimal')
    parser.add_argument('--measure_computational_overhead', type=bool, default=False,
                        help='Whether to measure computational overhead')
    parser.add_argument('--measure_privacy_leakage', type=str, required=False,
                        choices=['membership_inference'],
                        help='Method to measure privacy leakage')
    parser.add_argument('--prb_bound_validation', type=bool, default=False,
                        help='Whether to validate PRB bound')
    parser.add_argument('--optimal_epsilon_validation', type=bool, default=False,
                        help='Whether to validate optimal epsilon')
    parser.add_argument('--baseline_comparison', type=bool, default=False,
                        help='Whether to compare against baselines')
    parser.add_argument('--communication_rounds', type=int, default=100,
                        help='Number of communication rounds')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def setup_experiment_config(args):
    """Process and expand the experiment configuration."""
    config = vars(args)
    
    # Process epsilon values
    if args.epsilon:
        config['epsilon_values'] = [float(eps) for eps in args.epsilon.replace('{', '').replace('}', '').split(',')]
    else:
        config['epsilon_values'] = [1.0]  # Default
    
    # Process epsilon range if specified
    if args.epsilon_range:
        ranges = args.epsilon_range.split(',')
        config['epsilon_ranges'] = []
        for r in ranges:
            if '*optimal' in r:
                factor = float(r.split('*')[0])
                config['epsilon_ranges'].append((factor, True))
            else:
                config['epsilon_ranges'].append((float(r), False))
    
    # Process poisoned client fraction
    if args.poisoned_client_fraction:
        config['poisoned_fractions'] = [float(frac) for frac in args.poisoned_client_fraction.replace('{', '').replace('}', '').split(',')]
    else:
        config['poisoned_fractions'] = [0.0]  # Default (no attack)
    
    # Process aggregators for Experiment 3
    if args.aggregators:
        config['aggregator_list'] = args.aggregators.split(',')
    
    # Process attack scenarios for Experiment 3
    if args.attack_scenarios:
        config['attack_scenario_list'] = args.attack_scenarios.split(',')
    
    # Create experiment directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    return config

def load_model(model_name, input_shape, num_classes):
    """Load a model based on the model name."""
    if model_name == 'cnn_small':
        return CNN_Small(input_shape=input_shape, num_classes=num_classes)
    elif model_name == 'resnet18_pruned':
        return ResNet18_Pruned(input_shape=input_shape, num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

def run_experiment_1(config):
    """Run PRB Validation experiment (Experiment 1)."""
    logger.info("Starting Experiment 1: PRB Validation")
    
    # Configuration parameters
    dataset_name = config['dataset']
    model_name = config['model']
    num_clients = config['num_clients']
    alpha = config['alpha']
    epsilon_values = config['epsilon_values']
    poisoned_fractions = config['poisoned_fractions']
    communication_rounds = config['communication_rounds']
    output_dir = config['output_dir']
    
    # Data and model setup
    train_data, test_data, client_data = load_federated_dataset(
        dataset_name, num_clients, alpha
    )
    
    # Get input shape and num_classes from the dataset
    if dataset_name == 'cifar10':
        input_shape = (3, 32, 32)
        num_classes = 10
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Create results directory
    results_dir = os.path.join(output_dir, "eps_tau_experiments")
    os.makedirs(results_dir, exist_ok=True)
    
    # Track results
    results = []
    
    # Run experiments for different epsilon and tau values
    for epsilon in epsilon_values:
        for tau in poisoned_fractions:
            # Repeat experiments
            num_repeats = 3 if tau == 0.1 else 2  # More repeats for lower tau
            
            for repeat in range(num_repeats):
                logger.info(f"Running with ε={epsilon}, τ={tau}, repeat={repeat+1}/{num_repeats}")
                
                # Initialize global model
                global_model = load_model(model_name, input_shape, num_classes)
                
                # Track metrics
                test_accuracies = []
                model_deviations = []
                theoretical_prb_values = []
                start_time = time.time()
                
                # Initialize optimal model (trained without privacy or attacks)
                optimal_model = load_model(model_name, input_shape, num_classes)
                # In a real implementation, we would train this model without privacy or attacks
                # Here we're simulating this with a placeholder
                
                # Training loop
                for round_idx in range(communication_rounds):
                    logger.info(f"Communication round {round_idx+1}/{communication_rounds}")
                    
                    # Client updates
                    client_models = []
                    for client_idx in range(num_clients):
                        # Get client data
                        client_dataset = client_data[client_idx]
                        
                        # Create client model by copying global model
                        client_model = type(global_model)(*global_model.init_args, **global_model.init_kwargs)
                        client_model.load_state_dict(global_model.state_dict())
                        
                        # Local training (simplified)
                        # In a real implementation, this would be actual training
                        # For this skeleton, we're just simulating the process
                        
                        # Apply differential privacy if needed
                        if config['privacy_mechanism'] == 'dp':
                            client_model = apply_dp_sgd(client_model, epsilon)
                        
                        # Add client model to list
                        client_models.append(client_model)
                    
                    # Apply attack if needed
                    if tau > 0:
                        num_malicious = int(tau * num_clients)
                        client_models = model_replacement_attack(
                            client_models, num_malicious, global_model
                        )
                    
                    # Aggregation
                    global_model = federated_averaging(client_models)
                    
                    # Evaluate model
                    test_accuracy = compute_test_accuracy(global_model, test_data)
                    test_accuracies.append(test_accuracy)
                    
                    # Compute model deviation if requested
                    if config['measure_model_deviation']:
                        deviation = compute_model_deviation(global_model, optimal_model)
                        model_deviations.append(deviation)
                        
                        # Compute theoretical PRB bound
                        if config['prb_bound_validation']:
                            theoretical_prb = compute_theoretical_prb(
                                epsilon, tau, 0.05, input_shape[0] * input_shape[1] * input_shape[2] * num_classes, num_clients
                            )
                            theoretical_prb_values.append(theoretical_prb)
                    
                    logger.info(f"Round {round_idx+1}: Test accuracy = {test_accuracy:.4f}")
                
                # Compute execution time
                execution_time = time.time() - start_time
                
                # Save results
                experiment_result = {
                    'epsilon': epsilon,
                    'tau': tau,
                    'repeat': repeat,
                    'test_accuracies': test_accuracies,
                    'model_deviations': model_deviations,
                    'theoretical_prb_values': theoretical_prb_values,
                    'execution_time': execution_time
                }
                results.append(experiment_result)
                
                # Save result to file
                result_file = os.path.join(results_dir, f"eps_{epsilon}_tau_{tau}_rep_{repeat}.json")
                with open(result_file, 'w') as f:
                    json.dump(experiment_result, f, indent=2)
    
    # Save all results to a single file
    all_results_file = os.path.join(output_dir, "experiment1_results.json")
    with open(all_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Experiment 1 completed. Results saved to {output_dir}")
    return results

def run_experiment_2(config):
    """Run Optimal Privacy-Robustness Trade-off experiment (Experiment 2)."""
    logger.info("Starting Experiment 2: Optimal Privacy-Robustness Trade-off")
    
    # Configuration parameters
    dataset_name = config['dataset']
    model_name = config['model']
    num_clients = config['num_clients']
    alpha = config['alpha']
    epsilon_ranges = config.get('epsilon_ranges', [(1.0, False)])  # Default if not specified
    tau = 0.2  # Fixed attack fraction for this experiment
    communication_rounds = config['communication_rounds']
    output_dir = config['output_dir']
    
    # Data and model setup
    train_data, test_data, client_data = load_federated_dataset(
        dataset_name, num_clients, alpha
    )
    
    # Get input shape and num_classes from the dataset
    if dataset_name == 'cifar10':
        input_shape = (3, 32, 32)
        num_classes = 10
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Create results directory
    results_dir = os.path.join(output_dir, "optimal_epsilon_experiments")
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate the optimal epsilon based on PRB theory
    model_dimension = input_shape[0] * input_shape[1] * input_shape[2] * num_classes
    optimal_epsilon = compute_optimal_epsilon(tau, 0.05, model_dimension, num_clients)
    logger.info(f"Theoretical optimal ε* = {optimal_epsilon:.4f}")
    
    # Determine the epsilon values to test
    epsilon_values = []
    for factor, is_relative in epsilon_ranges:
        if is_relative:
            epsilon_values.append(factor * optimal_epsilon)
        else:
            epsilon_values.append(factor)
    
    # Track results
    results = []
    
    # Run experiments for different epsilon values
    for epsilon_idx, epsilon in enumerate(epsilon_values):
        # Repeat experiments
        num_repeats = 3  # Always do 3 repeats for this experiment
        
        for repeat in range(num_repeats):
            logger.info(
                f"Running with ε={epsilon:.4f} (relative to ε*: {epsilon/optimal_epsilon:.2f}), "
                f"repeat={repeat+1}/{num_repeats}"
            )
            
            # Initialize global model
            global_model = load_model(model_name, input_shape, num_classes)
            
            # Track metrics
            test_accuracies = []
            attack_success_rates = []
            privacy_leakage = []
            start_time = time.time()
            
            # Training loop
            for round_idx in range(communication_rounds):
                logger.info(f"Communication round {round_idx+1}/{communication_rounds}")
                
                # Client updates
                client_models = []
                for client_idx in range(num_clients):
                    # Get client data
                    client_dataset = client_data[client_idx]
                    
                    # Create client model by copying global model
                    client_model = type(global_model)(*global_model.init_args, **global_model.init_kwargs)
                    client_model.load_state_dict(global_model.state_dict())
                    
                    # Local training (simplified)
                    # In a real implementation, this would be actual training
                    
                    # Apply differential privacy if needed
                    if config['privacy_mechanism'] == 'dp':
                        client_model = apply_dp_sgd(client_model, epsilon)
                    
                    # Add client model to list
                    client_models.append(client_model)
                
                # Apply label flipping attack (fixed for this experiment)
                num_malicious = int(tau * num_clients)
                # In a real implementation, we would use label flipping attack
                # Here we're using model replacement as a placeholder
                client_models = model_replacement_attack(
                    client_models, num_malicious, global_model, targeted=True, target_label=0
                )
                
                # Aggregation
                global_model = federated_averaging(client_models)
                
                # Evaluate model
                test_accuracy = compute_test_accuracy(global_model, test_data)
                test_accuracies.append(test_accuracy)
                
                # Measure attack success rate (simplified)
                # In a real implementation, this would evaluate how successful the attack was
                attack_success = 0.0  # Placeholder
                attack_success_rates.append(attack_success)
                
                # Measure privacy leakage if requested
                if config.get('measure_privacy_leakage') == 'membership_inference':
                    # Evaluate membership inference every 10 rounds to reduce computation
                    if round_idx % 10 == 0 or round_idx == communication_rounds - 1:
                        leakage = evaluate_membership_inference(
                            global_model, train_data, test_data
                        )
                        privacy_leakage.append((round_idx, leakage))
                        logger.info(
                            f"Round {round_idx+1}: Privacy leakage (AUC) = {leakage:.4f}"
                        )
                
                logger.info(f"Round {round_idx+1}: Test accuracy = {test_accuracy:.4f}")
            
            # Compute execution time
            execution_time = time.time() - start_time
            
            # Save results
            epsilon_relative = epsilon / optimal_epsilon
            experiment_result = {
                'epsilon': epsilon,
                'epsilon_relative': epsilon_relative,
                'optimal_epsilon': optimal_epsilon,
                'tau': tau,
                'repeat': repeat,
                'test_accuracies': test_accuracies,
                'attack_success_rates': attack_success_rates,
                'privacy_leakage': privacy_leakage,
                'execution_time': execution_time
            }
            results.append(experiment_result)
            
            # Save result to file
            result_file = os.path.join(
                results_dir, f"eps_{epsilon:.4f}_rel_{epsilon_relative:.2f}_rep_{repeat}.json"
            )
            with open(result_file, 'w') as f:
                json.dump(experiment_result, f, indent=2)
    
    # Save all results to a single file
    all_results_file = os.path.join(output_dir, "experiment2_results.json")
    with open(all_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Experiment 2 completed. Results saved to {output_dir}")
    return results

def run_experiment_3(config):
    """Run PRB-Guided Federated Learning experiment (Experiment 3)."""
    logger.info("Starting Experiment 3: PRB-Guided Federated Learning")
    
    # Configuration parameters
    dataset_name = config['dataset']
    model_name = config['model']
    num_clients = config['num_clients']
    alpha = config['alpha']
    aggregator_list = config.get('aggregator_list', ['fedavg'])
    attack_scenarios = config.get('attack_scenario_list', ['no_attack'])
    communication_rounds = config['communication_rounds']
    output_dir = config['output_dir']
    
    # Data and model setup
    train_data, test_data, client_data = load_federated_dataset(
        dataset_name, num_clients, alpha
    )
    
    # Get input shape and num_classes from the dataset
    if dataset_name == 'cifar10':
        input_shape = (3, 32, 32)
        num_classes = 10
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Create results directory
    results_dir = os.path.join(output_dir, "comparative_analysis")
    os.makedirs(results_dir, exist_ok=True)
    
    # Track results
    results = []
    
    # Run experiments for different aggregators and attack scenarios
    for aggregator_name in aggregator_list:
        for attack_scenario in attack_scenarios:
            logger.info(f"Testing aggregator: {aggregator_name}, attack: {attack_scenario}")
            
            # Parse attack scenario
            attack_type = 'none'
            tau = 0.0
            if attack_scenario == 'no_attack':
                attack_type = 'none'
                tau = 0.0
            elif attack_scenario == 'model_replacement_20':
                attack_type = 'model_replacement'
                tau = 0.2
            elif attack_scenario == 'mixed_attack':
                attack_type = 'mixed_attack'
                tau = 0.2
            elif attack_scenario == 'high_fraction_30':
                attack_type = 'model_replacement'
                tau = 0.3
            
            # Initialize global model
            global_model = load_model(model_name, input_shape, num_classes)
            
            # Track metrics
            test_accuracies = []
            model_deviations = []
            privacy_leakage = []
            start_time = time.time()
            
            # Set epsilon based on the aggregator
            epsilon = 1.0
            if aggregator_name == 'dp_fedavg':
                epsilon = 1.0
            elif aggregator_name == 'prb_guided':
                # Compute optimal epsilon for PRB-guided aggregation
                model_dimension = input_shape[0] * input_shape[1] * input_shape[2] * num_classes
                epsilon = compute_optimal_epsilon(tau, 0.05, model_dimension, num_clients)
            
            # Training loop
            for round_idx in range(communication_rounds):
                logger.info(f"Communication round {round_idx+1}/{communication_rounds}")
                
                # Client updates
                client_models = []
                client_contributions = []  # For PRB-guided aggregation
                
                for client_idx in range(num_clients):
                    # Get client data
                    client_dataset = client_data[client_idx]
                    
                    # Create client model by copying global model
                    client_model = type(global_model)(*global_model.init_args, **global_model.init_kwargs)
                    client_model.load_state_dict(global_model.state_dict())
                    
                    # Local training (simplified)
                    # In a real implementation, this would be actual training
                    
                    # Apply differential privacy if needed
                    if aggregator_name in ['dp_fedavg', 'prb_guided']:
                        client_model = apply_dp_sgd(client_model, epsilon)
                    
                    # Compute client contribution (simplified)
                    # In a real implementation, this would be based on validation performance
                    client_contribution = 1.0  # Placeholder
                    client_contributions.append(client_contribution)
                    
                    # Add client model to list
                    client_models.append(client_model)
                
                # Apply attack if needed
                if tau > 0:
                    num_malicious = int(tau * num_clients)
                    if attack_type == 'model_replacement':
                        client_models = model_replacement_attack(
                            client_models, num_malicious, global_model
                        )
                    elif attack_type == 'mixed_attack':
                        # In a real implementation, this would be a mix of attacks
                        # Here we're using model replacement as a placeholder
                        client_models = model_replacement_attack(
                            client_models, num_malicious, global_model, targeted=True
                        )
                
                # Aggregation based on the specified method
                if aggregator_name == 'fedavg':
                    global_model = federated_averaging(client_models)
                elif aggregator_name == 'multi_krum':
                    # Simplified Multi-Krum (using robust_fedavg as a placeholder)
                    global_model = robust_fedavg(
                        global_model, client_models, outlier_threshold=1.5
                    )
                elif aggregator_name == 'dp_fedavg':
                    # DP-FedAvg (already applied DP during client updates)
                    global_model = federated_averaging(client_models)
                elif aggregator_name == 'pba':
                    # Simplified PBA (using robust_fedavg as a placeholder)
                    global_model = robust_fedavg(
                        global_model, client_models, outlier_threshold=2.0
                    )
                elif aggregator_name == 'prb_guided':
                    # PRB-guided aggregation
                    client_deviations = [
                        compute_model_deviation(client_model, global_model)
                        for client_model in client_models
                    ]
                    client_weights = compute_prb_guided_weights(
                        client_contributions, client_deviations, epsilon
                    )
                    global_model = federated_averaging(client_models, client_weights)
                
                # Evaluate model
                test_accuracy = compute_test_accuracy(global_model, test_data)
                test_accuracies.append(test_accuracy)
                
                # Measure privacy leakage for selected rounds
                if round_idx % 20 == 0 or round_idx == communication_rounds - 1:
                    leakage = evaluate_membership_inference(
                        global_model, train_data, test_data
                    )
                    privacy_leakage.append((round_idx, leakage))
                
                logger.info(f"Round {round_idx+1}: Test accuracy = {test_accuracy:.4f}")
            
            # Compute execution time
            execution_time = time.time() - start_time
            
            # Save results
            experiment_result = {
                'aggregator': aggregator_name,
                'attack_scenario': attack_scenario,
                'tau': tau,
                'test_accuracies': test_accuracies,
                'privacy_leakage': privacy_leakage,
                'execution_time': execution_time
            }
            results.append(experiment_result)
            
            # Save result to file
            result_file = os.path.join(
                results_dir, f"agg_{aggregator_name}_attack_{attack_scenario}.json"
            )
            with open(result_file, 'w') as f:
                json.dump(experiment_result, f, indent=2)
    
    # Save all results to a single file
    all_results_file = os.path.join(output_dir, "experiment3_results.json")
    with open(all_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Experiment 3 completed. Results saved to {output_dir}")
    return results

def main():
    """Main function to run experiments."""
    args = parse_args()
    config = setup_experiment_config(args)
    
    # Save configuration
    config_file = os.path.join(args.output_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run appropriate experiment based on configuration
    if config['prb_bound_validation']:
        # Experiment 1: PRB Validation
        run_experiment_1(config)
    elif config['optimal_epsilon_validation']:
        # Experiment 2: Optimal Privacy-Robustness Trade-off
        run_experiment_2(config)
    elif config['baseline_comparison']:
        # Experiment 3: PRB-Guided Federated Learning
        run_experiment_3(config)
    else:
        logger.error("No experiment type specified. Please set appropriate flags.")

if __name__ == "__main__":
    main() 