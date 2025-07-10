"""
Pre-defined hypothesis collections for common neural architecture questions.
"""

from typing import Dict, Any, List, Callable
import torch
import torch.nn as nn
import numpy as np

from .core import Hypothesis, HypothesisCategory
from structure_net.core.network_factory import create_standard_network
from structure_net.evolution.components import create_standard_evolution_system
from structure_net.evolution.adaptive_learning_rates.unified_manager import AdaptiveLearningRateManager
from seed_search.architecture_generator import ArchitectureGenerator
from data_factory import get_dataset_config, create_dataset

def seed_search_experiment(config: Dict[str, Any]) -> tuple[Any, Dict[str, float]]:
    """
    Test function for a single seed search experiment.
    This function is designed to be executed by the NAL runner.
    """
    # Get dataset configuration
    dataset_name = config.get('dataset', 'mnist')
    dataset_config = get_dataset_config(dataset_name)
    
    # This is a simplified version of the logic in GPUSeedHunter._test_seed_impl
    model = create_standard_network(
        architecture=config['architecture'],
        sparsity=config['sparsity'],
        seed=config['seed'],
        device=config.get('device', 'cpu')
    )
    
    # Dummy training and evaluation
    accuracy = np.random.uniform(0.1, 0.9)
    extrema_score = np.random.uniform(0, 1)
    parameters = sum(p.numel() for p in model.parameters())
    
    metrics = {
        'accuracy': accuracy,
        'extrema_score': extrema_score,
        'parameters': parameters,
        'patchability': extrema_score * (1 - accuracy),
        'efficiency': accuracy / (parameters / 1e6) if parameters > 0 else 0
    }
    
    # The primary metric for seed search is often patchability
    metrics['primary_metric'] = metrics['patchability']
    
    return model, metrics

class SeedSearchHypotheses:
    """Hypotheses for finding optimal seed networks."""

    @staticmethod
    def find_optimal_seeds(dataset_name: str = 'mnist') -> Hypothesis:
        """A hypothesis for a comprehensive seed search.
        
        Args:
            dataset_name: Name of dataset to use for seed search
        """
        
        # Get dataset configuration
        dataset_config = get_dataset_config(dataset_name)
        
        # Generate a portfolio of architectures to test
        arch_gen = ArchitectureGenerator(
            input_size=dataset_config.input_size, 
            num_classes=dataset_config.num_classes
        )
        architectures = arch_gen.generate_systematic_batch(num_architectures=20)

        return Hypothesis(
            id=f"seed_search_optimal_{dataset_name}",
            name=f"Find Optimal Seed Networks for {dataset_name.upper()}",
            description=f"Run a comprehensive search for seed networks with high potential on {dataset_name}.",
            category=HypothesisCategory.ARCHITECTURE,
            question="Which combinations of architecture, sparsity, and random seed produce the most promising starting points for growth?",
            prediction="A diverse set of optimal seeds will be found, balancing accuracy, efficiency, and patchability.",
            test_function=seed_search_experiment,
            parameter_space={
                'architecture': architectures,
                'sparsity': {'min': 0.01, 'max': 0.1, 'n_samples': 3},
                'seed': list(range(10)), # Test 10 random seeds for each config
                'dataset': [dataset_name]  # Pass dataset as parameter
            },
            control_parameters={
                'dataset': dataset_name,
                'epochs': 5, # Short training for seed evaluation
            },
            success_metrics={'patchability': 0.5}
        )



class ArchitectureHypotheses:
    """Hypotheses about neural network architectures."""
    
    @staticmethod
    def depth_vs_width() -> Hypothesis:
        """Is it better to have deeper or wider networks?"""
        
        def test_function(config: Dict[str, Any]):
            # Test implementation would go here
            # This is a placeholder
            from src.structure_net.core.network_factory import create_standard_network
            
            if config['architecture_type'] == 'deep':
                # Deep and narrow
                architecture = [784, 64, 64, 64, 64, 10]
            else:
                # Shallow and wide
                architecture = [784, 256, 128, 10]
            
            model = create_standard_network(
                architecture=architecture,
                sparsity=config.get('sparsity', 0.02)
            )
            
            # Training would happen here
            # Return model and metrics
            metrics = {
                'accuracy': 0.0,  # Placeholder
                'parameters': sum(p.numel() for p in model.parameters()),
                'depth': len(architecture) - 1
            }
            
            return model, metrics
        
        return Hypothesis(
            id="arch_depth_vs_width",
            name="Deep vs Wide Networks",
            description="Compare deep narrow networks against shallow wide networks",
            category=HypothesisCategory.ARCHITECTURE,
            question="Do deeper networks outperform wider networks with similar parameter counts?",
            prediction="Deeper networks will achieve better accuracy due to hierarchical feature learning",
            test_function=test_function,
            parameter_space={
                'architecture_type': ['deep', 'wide'],
                'sparsity': {'min': 0.01, 'max': 0.1, 'n_samples': 3}
            },
            control_parameters={
                'dataset': 'cifar10',
                'epochs': 50,
                'batch_size': 128
            },
            success_metrics={
                'accuracy': 0.5,
                'efficiency': 0.001
            },
            tags=['architecture', 'depth', 'width', 'fundamental']
        )
    
    @staticmethod
    def pyramid_architecture() -> Hypothesis:
        """Do pyramid-shaped architectures perform better?"""
        
        def test_function(config: Dict[str, Any]):
            from src.structure_net.core.network_factory import create_standard_network
            
            input_size = 784
            output_size = 10
            n_layers = config['n_layers']
            
            if config['shape'] == 'pyramid':
                # Pyramid shape - gradually decreasing
                sizes = []
                current_size = input_size
                for i in range(n_layers):
                    next_size = int(current_size * 0.5)
                    if next_size < output_size * 2:
                        next_size = output_size * 2
                    sizes.append(next_size)
                    current_size = next_size
                architecture = [input_size] + sizes[:-1] + [output_size]
            
            elif config['shape'] == 'hourglass':
                # Hourglass shape - decrease then increase
                mid_point = n_layers // 2
                sizes = []
                
                # Contracting path
                current_size = input_size
                for i in range(mid_point):
                    next_size = int(current_size * 0.6)
                    sizes.append(next_size)
                    current_size = next_size
                
                # Expanding path
                for i in range(n_layers - mid_point):
                    next_size = int(current_size * 1.5)
                    if next_size > input_size // 2:
                        next_size = input_size // 2
                    sizes.append(next_size)
                    current_size = next_size
                
                architecture = [input_size] + sizes + [output_size]
            
            else:  # uniform
                size = int((input_size + output_size) / 2)
                architecture = [input_size] + [size] * (n_layers - 1) + [output_size]
            
            model = create_standard_network(
                architecture=architecture,
                sparsity=config.get('sparsity', 0.02)
            )
            
            metrics = {
                'accuracy': 0.0,  # Placeholder
                'parameters': sum(p.numel() for p in model.parameters()),
                'architecture': architecture
            }
            
            return model, metrics
        
        return Hypothesis(
            id="arch_pyramid_shape",
            name="Pyramid Architecture Effectiveness",
            description="Test if pyramid-shaped architectures outperform uniform architectures",
            category=HypothesisCategory.ARCHITECTURE,
            question="Do pyramid-shaped networks learn more efficient representations?",
            prediction="Pyramid architectures will achieve similar accuracy with fewer parameters",
            test_function=test_function,
            parameter_space={
                'shape': ['pyramid', 'hourglass', 'uniform'],
                'n_layers': [3, 4, 5],
                'sparsity': {'min': 0.01, 'max': 0.05, 'n_samples': 2}
            },
            control_parameters={
                'dataset': 'cifar10',
                'epochs': 50,
                'batch_size': 128,
                'learning_rate': 0.001
            },
            success_metrics={
                'accuracy': 0.45,
                'parameter_efficiency': 0.01  # accuracy per million parameters
            },
            tags=['architecture', 'pyramid', 'efficiency']
        )
    
    @staticmethod
    def skip_connections() -> Hypothesis:
        """Do skip connections improve training?"""
        
        def test_function(config: Dict[str, Any]):
            # This would use the residual block functionality
            from src.structure_net.evolution.residual_blocks import create_residual_network
            
            if config['use_skip']:
                model = create_residual_network(
                    architecture=config['architecture'],
                    skip_frequency=config['skip_frequency']
                )
            else:
                from src.structure_net.core.network_factory import create_standard_network
                model = create_standard_network(
                    architecture=config['architecture'],
                    sparsity=0.02
                )
            
            metrics = {
                'accuracy': 0.0,
                'convergence_speed': 0.0,
                'gradient_flow': 0.0
            }
            
            return model, metrics
        
        return Hypothesis(
            id="arch_skip_connections",
            name="Skip Connection Benefits",
            description="Evaluate the impact of skip connections on training",
            category=HypothesisCategory.ARCHITECTURE,
            question="Do skip connections significantly improve gradient flow and convergence?",
            prediction="Skip connections will improve convergence speed by 30% and final accuracy by 5%",
            test_function=test_function,
            parameter_space={
                'use_skip': [True, False],
                'skip_frequency': [2, 3, 4],
                'architecture': [
                    [784, 256, 256, 256, 256, 10],
                    [784, 128, 128, 128, 128, 128, 128, 10]
                ]
            },
            control_parameters={
                'dataset': 'cifar10',
                'epochs': 50,
                'batch_size': 128
            },
            success_metrics={
                'accuracy': 0.5,
                'convergence_speed': 1.3  # 30% improvement
            },
            tags=['architecture', 'residual', 'skip_connections']
        )


class GrowthHypotheses:
    """Hypotheses about network growth strategies."""
    
    @staticmethod
    def growth_timing(dataset_name: str = 'cifar10') -> Hypothesis:
        """When is the best time to grow a network?
        
        Args:
            dataset_name: Name of dataset to use for testing
        """
        
        # Get dataset configuration
        dataset_config = get_dataset_config(dataset_name)
        
        def test_function(config: Dict[str, Any]):
            from src.structure_net.evolution.components import create_standard_evolution_system
            
            # Use new composable evolution system
            evolution_system = create_standard_evolution_system()
            
            # Get dataset info from config
            ds_name = config.get('dataset', dataset_name)
            ds_config = get_dataset_config(ds_name)
            
            # Initial small network
            model = create_standard_network(
                architecture=[ds_config.input_size, 64, ds_config.num_classes],
                sparsity=0.02
            )
            
            # Simulate training with growth
            metrics = {
                'final_accuracy': 0.0,
                'growth_events': 0,
                'final_parameters': 0,
                'efficiency': 0.0
            }
            
            return model, metrics
        
        return Hypothesis(
            id="growth_timing",
            name="Optimal Growth Timing",
            description="Determine the best timing strategy for network growth",
            category=HypothesisCategory.GROWTH,
            question="Should networks grow early, late, or continuously during training?",
            prediction="Early growth will be most effective as it allows new capacity to be fully trained",
            test_function=test_function,
            parameter_space={
                'growth_trigger': ['epoch_based', 'loss_plateau', 'gradient_based'],
                'growth_interval': [5, 10, 20],
                'growth_amount': {'min': 0.1, 'max': 0.5, 'n_samples': 3}
            },
            control_parameters={
                'initial_architecture': [dataset_config.input_size, 64, dataset_config.num_classes],
                'max_growth_events': 5,
                'dataset': dataset_name,
                'epochs': 100
            },
            success_metrics={
                'final_accuracy': 0.6,
                'efficiency': 0.01
            },
            tags=['growth', 'timing', 'adaptive']
        )
    
    @staticmethod
    def growth_location(dataset_name: str = 'cifar10') -> Hypothesis:
        """Where should new neurons be added?
        
        Args:
            dataset_name: Name of dataset to use for testing
        """
        
        # Get dataset configuration
        dataset_config = get_dataset_config(dataset_name)
        
        def test_function(config: Dict[str, Any]):
            # Test different growth locations
            growth_location = config['growth_location']
            
            # Get dataset info from config
            ds_name = config.get('dataset', dataset_name)
            ds_config = get_dataset_config(ds_name)
            
            model = create_standard_network(
                architecture=[ds_config.input_size, 128, 64, ds_config.num_classes],
                sparsity=0.02
            )
            
            # Simulate targeted growth
            metrics = {
                'accuracy_improvement': 0.0,
                'layer_utilization': {},
                'gradient_flow': 0.0
            }
            
            return model, metrics
        
        return Hypothesis(
            id="growth_location",
            name="Optimal Growth Location",
            description="Determine where in the network to add new capacity",
            category=HypothesisCategory.GROWTH,
            question="Should growth target early layers, late layers, or be distributed?",
            prediction="Growth in middle layers will be most effective for feature learning",
            test_function=test_function,
            parameter_space={
                'growth_location': ['early', 'middle', 'late', 'distributed', 'adaptive'],
                'growth_metric': ['gradient_magnitude', 'activation_variance', 'weight_magnitude']
            },
            control_parameters={
                'architecture': [dataset_config.input_size, 128, 64, 32, dataset_config.num_classes],
                'growth_events': 3,
                'neurons_per_growth': 32,
                'dataset': dataset_name
            },
            success_metrics={
                'accuracy_improvement': 0.05,
                'layer_utilization': 0.8
            },
            tags=['growth', 'location', 'architecture']
        )


class SparsityHypotheses:
    """Hypotheses about network sparsity."""
    
    @staticmethod
    def optimal_sparsity(dataset_name: str = 'cifar10') -> Hypothesis:
        """What is the optimal sparsity level?
        
        Args:
            dataset_name: Name of dataset to use for testing
        """
        
        # Get dataset configuration
        dataset_config = get_dataset_config(dataset_name)
        
        def test_function(config: Dict[str, Any]):
            model = create_standard_network(
                architecture=config['architecture'],
                sparsity=config['sparsity']
            )
            
            # Measure various aspects
            metrics = {
                'accuracy': 0.0,
                'inference_speed': 0.0,
                'memory_usage': 0.0,
                'gradient_flow': 0.0
            }
            
            return model, metrics
        
        return Hypothesis(
            id="optimal_sparsity",
            name="Optimal Sparsity Level",
            description="Find the best trade-off between sparsity and performance",
            category=HypothesisCategory.SPARSITY,
            question="What sparsity level maximizes accuracy while maintaining efficiency?",
            prediction="5-10% sparsity will provide optimal accuracy-efficiency trade-off",
            test_function=test_function,
            parameter_space={
                'sparsity': {'min': 0.0, 'max': 0.5, 'n_samples': 10, 'log_scale': True},
                'architecture': [
                    [dataset_config.input_size, 256, 128, dataset_config.num_classes],
                    [dataset_config.input_size, 512, 256, 128, dataset_config.num_classes]
                ]
            },
            control_parameters={
                'dataset': dataset_name,
                'epochs': 50,
                'initialization': 'sparse'
            },
            success_metrics={
                'accuracy': 0.5,
                'efficiency_ratio': 1.5  # accuracy / memory_usage
            },
            tags=['sparsity', 'efficiency', 'optimization']
        )
    
    @staticmethod
    def dynamic_sparsity(dataset_name: str = 'cifar10') -> Hypothesis:
        """Should sparsity change during training?
        
        Args:
            dataset_name: Name of dataset to use for testing
        """
        
        # Get dataset configuration
        dataset_config = get_dataset_config(dataset_name)
        
        def test_function(config: Dict[str, Any]):
            # Test dynamic vs static sparsity
            if config['sparsity_schedule'] == 'static':
                sparsity = config['initial_sparsity']
            else:
                # Implement dynamic sparsity
                sparsity = config['initial_sparsity']
            
            # Get dataset info from config
            ds_name = config.get('dataset', dataset_name)
            ds_config = get_dataset_config(ds_name)
            
            model = create_standard_network(
                architecture=[ds_config.input_size, 256, 128, ds_config.num_classes],
                sparsity=sparsity
            )
            
            metrics = {
                'final_accuracy': 0.0,
                'convergence_speed': 0.0,
                'final_sparsity': 0.0
            }
            
            return model, metrics
        
        return Hypothesis(
            id="dynamic_sparsity",
            name="Dynamic Sparsity Benefits",
            description="Test if changing sparsity during training improves results",
            category=HypothesisCategory.SPARSITY,
            question="Does dynamic sparsity lead to better final networks?",
            prediction="Gradually increasing sparsity will improve generalization",
            test_function=test_function,
            parameter_space={
                'sparsity_schedule': ['static', 'linear_increase', 'exponential_increase', 'cyclic'],
                'initial_sparsity': [0.0, 0.01, 0.05],
                'final_sparsity': [0.1, 0.2, 0.3]
            },
            control_parameters={
                'architecture': [dataset_config.input_size, 256, 128, dataset_config.num_classes],
                'epochs': 100,
                'pruning_frequency': 10,
                'dataset': dataset_name
            },
            success_metrics={
                'final_accuracy': 0.55,
                'convergence_speed': 1.0
            },
            tags=['sparsity', 'dynamic', 'pruning']
        )


class TrainingHypotheses:
    """Hypotheses about training strategies."""
    
    @staticmethod
    def learning_rate_adaptation(dataset_name: str = 'cifar10') -> Hypothesis:
        """Which adaptive learning rate strategy is most effective?
        
        Args:
            dataset_name: Name of dataset to use for testing
        """
        
        # Get dataset configuration
        dataset_config = get_dataset_config(dataset_name)
        
        def test_function(config: Dict[str, Any]):
            from src.structure_net.evolution.adaptive_learning_rates.unified_manager import AdaptiveLearningRateManager
            
            lr_manager = AdaptiveLearningRateManager(
                strategy=config['lr_strategy']
            )
            
            # Get dataset info from config
            ds_name = config.get('dataset', dataset_name)
            ds_config = get_dataset_config(ds_name)
            
            model = create_standard_network(
                architecture=[ds_config.input_size, 256, 128, ds_config.num_classes],
                sparsity=0.02
            )
            
            # Configure optimizer with adaptive LR
            optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'])
            
            metrics = {
                'final_accuracy': 0.0,
                'convergence_epochs': 0,
                'lr_variance': 0.0,
                'stability': 0.0
            }
            
            return model, metrics
        
        return Hypothesis(
            id="lr_adaptation",
            name="Learning Rate Adaptation Strategy",
            description="Compare different adaptive learning rate strategies",
            category=HypothesisCategory.TRAINING,
            question="Which learning rate adaptation strategy leads to fastest, most stable convergence?",
            prediction="Layer-wise adaptation will outperform global strategies",
            test_function=test_function,
            parameter_space={
                'lr_strategy': ['basic', 'advanced', 'comprehensive', 'ultimate'],
                'base_lr': {'min': 0.0001, 'max': 0.01, 'n_samples': 5, 'log_scale': True},
                'lr_warmup': [0, 5, 10]
            },
            control_parameters={
                'architecture': [dataset_config.input_size, 256, 128, dataset_config.num_classes],
                'dataset': dataset_name,
                'epochs': 50,
                'batch_size': 128
            },
            success_metrics={
                'final_accuracy': 0.5,
                'convergence_epochs': 30,
                'stability': 0.9
            },
            tags=['training', 'learning_rate', 'optimization']
        )
    
    @staticmethod
    def batch_size_scaling(dataset_name: str = 'cifar10') -> Hypothesis:
        """How does batch size affect training?
        
        Args:
            dataset_name: Name of dataset to use for testing
        """
        
        # Get dataset configuration
        dataset_config = get_dataset_config(dataset_name)
        
        def test_function(config: Dict[str, Any]):
            batch_size = config['batch_size']
            
            # Scale learning rate with batch size
            if config['lr_scaling'] == 'linear':
                lr = config['base_lr'] * (batch_size / 128)
            elif config['lr_scaling'] == 'sqrt':
                lr = config['base_lr'] * ((batch_size / 128) ** 0.5)
            else:
                lr = config['base_lr']
            
            # Get dataset info from config
            ds_name = config.get('dataset', dataset_name)
            ds_config = get_dataset_config(ds_name)
            
            model = create_standard_network(
                architecture=[ds_config.input_size, 256, 128, ds_config.num_classes],
                sparsity=0.02
            )
            
            metrics = {
                'accuracy': 0.0,
                'convergence_speed': 0.0,
                'gradient_noise': 0.0,
                'generalization': 0.0
            }
            
            return model, metrics
        
        return Hypothesis(
            id="batch_size_scaling",
            name="Batch Size Effects",
            description="Investigate the relationship between batch size and training dynamics",
            category=HypothesisCategory.TRAINING,
            question="How should learning rate scale with batch size for optimal training?",
            prediction="Square root scaling will provide best accuracy across batch sizes",
            test_function=test_function,
            parameter_space={
                'batch_size': [32, 64, 128, 256, 512, 1024],
                'lr_scaling': ['none', 'linear', 'sqrt'],
                'base_lr': [0.001, 0.01]
            },
            control_parameters={
                'architecture': [dataset_config.input_size, 256, 128, dataset_config.num_classes],
                'dataset': dataset_name,
                'epochs': 50
            },
            success_metrics={
                'accuracy': 0.5,
                'convergence_speed': 1.0,
                'generalization': 0.95  # val_acc / train_acc
            },
            tags=['training', 'batch_size', 'learning_rate', 'scaling']
        )
    
    @staticmethod
    def data_augmentation(dataset_name: str = 'cifar10') -> Hypothesis:
        """Which data augmentation strategies are most effective?
        
        Args:
            dataset_name: Name of dataset to use for testing
        """
        
        # Get dataset configuration
        dataset_config = get_dataset_config(dataset_name)
        
        def test_function(config: Dict[str, Any]):
            # This would integrate with data loading
            augmentation = config['augmentation_type']
            intensity = config['augmentation_intensity']
            
            # Get dataset info from config
            ds_name = config.get('dataset', dataset_name)
            ds_config = get_dataset_config(ds_name)
            
            model = create_standard_network(
                architecture=[ds_config.input_size, 256, 128, ds_config.num_classes],
                sparsity=0.02
            )
            
            metrics = {
                'accuracy': 0.0,
                'robustness': 0.0,
                'overfitting_ratio': 0.0
            }
            
            return model, metrics
        
        return Hypothesis(
            id="data_augmentation",
            name="Data Augmentation Effectiveness",
            description="Compare different data augmentation strategies",
            category=HypothesisCategory.TRAINING,
            question="Which augmentation strategies improve generalization most?",
            prediction="Moderate augmentation will improve accuracy by 5-10%",
            test_function=test_function,
            parameter_space={
                'augmentation_type': ['none', 'basic', 'cutout', 'mixup', 'all'],
                'augmentation_intensity': {'min': 0.0, 'max': 1.0, 'n_samples': 5}
            },
            control_parameters={
                'architecture': [dataset_config.input_size, 256, 128, dataset_config.num_classes],
                'dataset': dataset_name,
                'epochs': 50,
                'base_training': True
            },
            success_metrics={
                'accuracy': 0.55,
                'robustness': 0.8,
                'overfitting_ratio': 0.9
            },
            tags=['training', 'augmentation', 'generalization']
        )


def get_all_hypotheses(dataset_name: str = 'cifar10') -> List[Hypothesis]:
    """Get all pre-defined hypotheses.
    
    Args:
        dataset_name: Name of dataset to use for all hypotheses
    """
    hypotheses = []
    
    # Architecture hypotheses
    hypotheses.extend([
        ArchitectureHypotheses.depth_vs_width(dataset_name),
        ArchitectureHypotheses.pyramid_architecture(dataset_name),
        ArchitectureHypotheses.skip_connections(dataset_name)
    ])
    
    # Growth hypotheses
    hypotheses.extend([
        GrowthHypotheses.growth_timing(dataset_name),
        GrowthHypotheses.growth_location(dataset_name)
    ])
    
    # Sparsity hypotheses
    hypotheses.extend([
        SparsityHypotheses.optimal_sparsity(dataset_name),
        SparsityHypotheses.dynamic_sparsity(dataset_name)
    ])
    
    # Training hypotheses
    hypotheses.extend([
        TrainingHypotheses.learning_rate_adaptation(dataset_name),
        TrainingHypotheses.batch_size_scaling(dataset_name),
        TrainingHypotheses.data_augmentation(dataset_name)
    ])
    
    # Seed search hypothesis
    hypotheses.append(
        SeedSearchHypotheses.find_optimal_seeds(dataset_name)
    )
    
    return hypotheses


def get_hypothesis_by_category(category: HypothesisCategory) -> List[Hypothesis]:
    """Get all hypotheses for a specific category."""
    all_hypotheses = get_all_hypotheses()
    return [h for h in all_hypotheses if h.category == category]


def get_fundamental_hypotheses() -> List[Hypothesis]:
    """Get fundamental hypotheses that should be tested first."""
    all_hypotheses = get_all_hypotheses()
    return [h for h in all_hypotheses if 'fundamental' in h.tags]