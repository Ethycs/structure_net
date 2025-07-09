#!/usr/bin/env python3
"""
Standardized Logging Example - Latest Architecture

This example demonstrates the integration of the latest standardized logging system
with the optimized profiling architecture, featuring:

- Pydantic validation with WandB artifacts
- Integration with advanced profiling system
- Production-ready logging with minimal overhead
- Real-time monitoring and queue management
- Component-level profiling integration
- Adaptive sampling and overhead management
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import standardized logging system
from structure_net.logging import (
    create_growth_logger,
    create_training_logger,
    create_tournament_logger,
    get_queue_status,
    StandardizedLogger
)

# Import latest profiling system
from structure_net.profiling import (
    create_production_profiler, create_research_profiler,
    profile_component, profile_if_enabled, profile_memory_intensive,
    profile_operation, profile_batch_operation,
    ProfilerLevel
)

# Import structure_net components
from structure_net import create_standard_network


# Example 1: Production Logging with Profiling Integration
@profile_component(component_name="production_experiment", 
                  level=ProfilerLevel.BASIC)
class ProductionExperiment:
    """Production experiment with integrated logging and profiling."""
    
    def __init__(self, config):
        self.config = config
        self.logger = create_growth_logger(
            project_name="production_logging_demo",
            experiment_name=f"production_{datetime.now().strftime('%H%M%S')}",
            config=config,
            tags=['production', 'integrated', 'profiling']
        )
        
        # Create production profiler with minimal overhead
        self.profiler = create_production_profiler(max_overhead_percent=1.0)
        self.profiler.start_session("production_experiment")
    
    def setup_network(self):
        """Setup network with profiling."""
        network = create_standard_network(
            architecture=self.config['architecture'],
            sparsity=self.config['sparsity'],
            device=self.config['device']
        )
        
        # Log experiment start with profiling
        self.logger.log_experiment_start(
            network=network,
            target_accuracy=self.config['target_accuracy'],
            seed_architecture=self.config['architecture']
        )
        
        return network
    
    def run_growth_iteration(self, iteration, network):
        """Run growth iteration with integrated logging and profiling."""
        # Simulate training with batch profiling for high-frequency operations
        accuracy = 0.70 + (iteration * 0.05)
        loss = 0.8 - (iteration * 0.1)
        
        # Simulate extrema analysis with profiling
        with profile_operation("extrema_analysis", "evolution", 
                             level=ProfilerLevel.BASIC) as ctx:
            extrema_analysis = {
                'total_extrema': 10 + (iteration * 3),
                'extrema_ratio': 0.08 + (iteration * 0.02),
                'layer_health': {
                    '0': 0.90 - (iteration * 0.02),
                    '1': 0.85 - (iteration * 0.03)
                }
            }
            ctx.add_metric("extrema_count", extrema_analysis['total_extrema'])
            ctx.add_metric("extrema_ratio", extrema_analysis['extrema_ratio'])
        
        # Determine growth actions
        growth_occurred = iteration > 0 and extrema_analysis['extrema_ratio'] > 0.10
        growth_actions = []
        
        if growth_occurred:
            growth_actions.append({
                'action': 'add_patch',
                'position': 1,
                'size': 3,
                'reason': f"High extrema ratio: {extrema_analysis['extrema_ratio']:.3f}",
                'success': True
            })
        
        # Log iteration with validation and profiling
        self.logger.log_growth_iteration(
            iteration=iteration,
            network=network,
            accuracy=accuracy,
            loss=loss,
            extrema_analysis=extrema_analysis,
            growth_actions=growth_actions,
            growth_occurred=growth_occurred
        )
        
        return accuracy
    
    def finish_experiment(self, final_accuracy):
        """Finish experiment with integrated cleanup."""
        # Finish logging
        artifact_hash = self.logger.finish_experiment(final_accuracy=final_accuracy)
        
        # End profiling session
        profiling_results = self.profiler.end_session()
        
        # Log profiling results to standardized logger
        self.logger.log_profiling_session(profiling_results)
        
        return artifact_hash, profiling_results


def example_1_production_integration():
    """Example 1: Production logging with profiling integration."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Production Logging + Profiling Integration")
    print("="*70)
    
    config = {
        'architecture': [784, 128, 64, 10],
        'sparsity': 0.02,
        'device': 'cpu',
        'target_accuracy': 0.90,
        'batch_size': 64,
        'learning_rate': 0.001
    }
    
    # Create production experiment
    experiment = ProductionExperiment(config)
    
    print("📊 Starting production experiment with integrated profiling...")
    
    # Setup network
    network = experiment.setup_network()
    
    # Run growth iterations
    final_accuracy = 0.70
    for iteration in range(3):
        print(f"🔄 Production iteration {iteration}")
        final_accuracy = experiment.run_growth_iteration(iteration, network)
    
    # Finish experiment
    print("✅ Finishing production experiment...")
    artifact_hash, profiling_results = experiment.finish_experiment(final_accuracy)
    
    print(f"🎯 Production experiment completed!")
    print(f"📦 Artifact hash: {artifact_hash}")
    print(f"📊 Profiling overhead: {profiling_results.get('total_overhead', 0):.6f}s")
    print(f"🔗 WandB URL: {experiment.logger.wandb_logger.run.url}")
    
    return artifact_hash


# Example 2: Research Logging with Comprehensive Profiling
@profile_component(component_name="research_experiment", 
                  level=ProfilerLevel.COMPREHENSIVE)
class ResearchExperiment:
    """Research experiment with comprehensive logging and profiling."""
    
    def __init__(self, experiment_name):
        self.logger = create_tournament_logger(
            project_name="research_logging_demo",
            experiment_name=experiment_name,
            config={
                'dataset': 'mnist',
                'tournament_strategies': ['extrema_growth', 'random_growth'],
                'research_mode': True,
                'comprehensive_tracking': True
            },
            tags=['research', 'comprehensive', 'tournament']
        )
        
        # Create research profiler with comprehensive tracking
        self.profiler = create_research_profiler(
            experiment_name=experiment_name,
            level=ProfilerLevel.COMPREHENSIVE,
            enable_all_integrations=True
        )
        self.profiler.start_session("research_experiment")
    
    @profile_memory_intensive
    def analyze_network_architecture(self, network):
        """Analyze network architecture with memory profiling."""
        # Simulate comprehensive analysis
        time.sleep(0.1)
        
        analysis = {
            'parameter_count': sum(p.numel() for p in network.parameters()),
            'layer_count': len(list(network.modules())) - 1,
            'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'complexity_score': 0.75
        }
        
        return analysis
    
    def run_tournament_iteration(self, iteration, network):
        """Run tournament iteration with detailed profiling."""
        strategies = ['extrema_growth', 'random_growth', 'layer_addition']
        
        # Profile each strategy
        strategy_results = []
        
        for strategy in strategies:
            with profile_operation(f"strategy_{strategy}", "tournament", 
                                 level=ProfilerLevel.DETAILED) as ctx:
                # Simulate strategy execution
                execution_time = 0.05 + (hash(strategy) % 100) / 1000
                time.sleep(execution_time)
                
                improvement = 0.05 + (hash(strategy + str(iteration)) % 50) / 1000
                final_accuracy = 0.80 + improvement
                
                result = {
                    'strategy': strategy,
                    'improvement': improvement,
                    'final_accuracy': final_accuracy,
                    'execution_time': execution_time,
                    'success': True
                }
                
                strategy_results.append(result)
                
                # Add strategy-specific metrics
                ctx.add_metric("strategy_name", strategy)
                ctx.add_metric("improvement", improvement)
                ctx.add_metric("execution_time", execution_time)
        
        # Determine winner
        winner = max(strategy_results, key=lambda x: x['improvement'])
        
        tournament_results = {
            'winner': winner,
            'all_results': strategy_results
        }
        
        # Log tournament results
        self.logger.log_tournament_results(tournament_results, iteration)
        
        # Also log as growth iteration
        self.logger.log_growth_iteration(
            iteration=iteration,
            network=network,
            accuracy=winner['final_accuracy'],
            growth_occurred=True,
            growth_actions=[{
                'action': 'tournament_winner',
                'reason': f"Tournament selected: {winner['strategy']}",
                'success': True
            }]
        )
        
        return winner['final_accuracy']
    
    def finish_experiment(self, final_accuracy):
        """Finish research experiment."""
        # Finish logging
        artifact_hash = self.logger.finish_experiment(final_accuracy=final_accuracy)
        
        # End profiling session
        profiling_results = self.profiler.end_session()
        
        # Log comprehensive profiling results
        self.logger.log_profiling_session(profiling_results)
        
        return artifact_hash, profiling_results


def example_2_research_integration():
    """Example 2: Research logging with comprehensive profiling."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Research Logging + Comprehensive Profiling")
    print("="*70)
    
    experiment_name = f"research_{datetime.now().strftime('%H%M%S')}"
    experiment = ResearchExperiment(experiment_name)
    
    print("🔬 Starting research experiment with comprehensive profiling...")
    
    # Create network
    network = create_standard_network([784, 256, 128, 10], sparsity=0.05, device='cpu')
    experiment.logger.log_experiment_start(network=network)
    
    # Analyze network architecture
    print("🔍 Analyzing network architecture...")
    architecture_analysis = experiment.analyze_network_architecture(network)
    print(f"   Parameters: {architecture_analysis['parameter_count']:,}")
    print(f"   Complexity: {architecture_analysis['complexity_score']:.3f}")
    
    # Run tournament iterations
    final_accuracy = 0.80
    for iteration in range(2):
        print(f"🏆 Tournament iteration {iteration}")
        final_accuracy = experiment.run_tournament_iteration(iteration, network)
    
    # Finish experiment
    print("✅ Finishing research experiment...")
    artifact_hash, profiling_results = experiment.finish_experiment(final_accuracy)
    
    print(f"🎯 Research experiment completed!")
    print(f"📦 Artifact hash: {artifact_hash}")
    print(f"📊 Operations profiled: {profiling_results.get('total_operations', 0)}")
    print(f"🔗 WandB URL: {experiment.logger.wandb_logger.run.url}")
    
    return artifact_hash


def example_3_conditional_logging():
    """Example 3: Conditional logging with environment-based profiling."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Conditional Logging + Environment-Based Profiling")
    print("="*70)
    
    # Create logger
    logger = create_training_logger(
        project_name="conditional_logging_demo",
        experiment_name=f"conditional_{datetime.now().strftime('%H%M%S')}",
        config={
            'dataset': 'cifar10',
            'conditional_profiling': True,
            'environment': 'development'
        },
        tags=['conditional', 'environment_based']
    )
    
    # Create conditional profiler
    profiler = create_production_profiler(max_overhead_percent=2.0)
    profiler.start_session("conditional_demo")
    
    # Define conditional functions
    @profile_if_enabled(condition=lambda: os.getenv('PROFILE_TRAINING', '0') == '1')
    def conditional_training_step(batch_data):
        """Training step that only profiles when enabled."""
        time.sleep(0.01)  # Simulate training
        return 0.5 + (hash(str(batch_data)) % 100) / 1000
    
    print("📊 Testing conditional profiling...")
    
    # Test without profiling enabled
    print("   Running without profiling enabled...")
    losses = []
    for i in range(10):
        loss = conditional_training_step(f"batch_{i}")
        losses.append(loss)
    
    # Enable profiling
    os.environ['PROFILE_TRAINING'] = '1'
    print("   Running with profiling enabled...")
    
    # Test with profiling enabled
    for i in range(5):
        loss = conditional_training_step(f"enabled_batch_{i}")
        losses.append(loss)
    
    # Log training results
    avg_loss = sum(losses) / len(losses)
    logger.log_training_epoch(
        epoch=0,
        train_loss=avg_loss,
        train_acc=0.85,
        val_loss=avg_loss * 1.1,
        val_acc=0.82,
        learning_rate=0.001,
        duration=len(losses) * 0.01
    )
    
    # Finish experiment
    profiling_results = profiler.end_session()
    artifact_hash = logger.finish_experiment(final_accuracy=0.82)
    
    # Clean up
    os.environ.pop('PROFILE_TRAINING', None)
    
    print(f"🎯 Conditional logging completed!")
    print(f"📦 Artifact hash: {artifact_hash}")
    print(f"📊 Conditional profiling overhead: Minimal when disabled")
    
    return artifact_hash


def example_4_batch_logging_integration():
    """Example 4: Batch logging with high-frequency profiling."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Logging + High-Frequency Profiling")
    print("="*70)
    
    # Create logger
    logger = StandardizedLogger("batch_logging_demo")
    
    # Create standard profiler for batch operations
    profiler = create_production_profiler(max_overhead_percent=1.5)
    profiler.start_session("batch_demo")
    
    print("⚡ Running batch logging with high-frequency profiling...")
    
    # Log experiment start
    experiment_data = {
        "experiment_id": f"batch_demo_{datetime.now().strftime('%H%M%S')}",
        "timestamp": time.time(),
        "configuration": {
            "batch_size": 32,
            "high_frequency_logging": True,
            "profiling_mode": "batch"
        }
    }
    logger.log_experiment_start(experiment_data)
    
    # Simulate high-frequency training with batch profiling
    total_batches = 100
    batch_losses = []
    
    for batch_idx in range(total_batches):
        # Use batch profiling for high-frequency operations
        with profile_batch_operation("training_batch", "training", 
                                    {"batch_idx": batch_idx}) as ctx:
            # Simulate batch processing
            batch_loss = 1.0 - (batch_idx * 0.008)
            batch_losses.append(batch_loss)
            
            # Log every 10th batch to avoid overwhelming the logging system
            if batch_idx % 10 == 0:
                logger.log_iteration({
                    "batch": batch_idx,
                    "loss": batch_loss,
                    "progress": batch_idx / total_batches
                })
    
    # Force flush of batch profiler
    from structure_net.profiling.core.context_manager import get_global_batch_profiler
    batch_profiler = get_global_batch_profiler()
    batch_profiler.flush()
    
    # Log final results
    avg_loss = sum(batch_losses) / len(batch_losses)
    logger.log_experiment_end({
        "status": "completed",
        "total_batches": total_batches,
        "average_loss": avg_loss,
        "final_loss": batch_losses[-1]
    })
    
    # Finish profiling
    profiling_results = profiler.end_session()
    
    print(f"🎯 Batch logging completed!")
    print(f"⚡ Processed {total_batches} batches with minimal overhead")
    print(f"📊 Average loss: {avg_loss:.4f}")
    print(f"📈 Batch profiling overhead: Minimal (batched processing)")
    
    return profiling_results


def example_5_validation_with_profiling():
    """Example 5: Validation error handling with profiling integration."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Validation + Profiling Error Handling")
    print("="*70)
    
    logger = create_growth_logger(
        project_name="validation_demo",
        experiment_name=f"validation_{datetime.now().strftime('%H%M%S')}",
        tags=['validation', 'error_handling', 'profiling']
    )
    
    profiler = create_production_profiler()
    profiler.start_session("validation_demo")
    
    network = create_standard_network([784, 32, 10], sparsity=0.02, device='cpu')
    logger.log_experiment_start(network=network)
    
    print("🧪 Testing validation with profiling integration...")
    
    # Test 1: Valid operation with profiling
    print("\n1. Testing valid operation with profiling...")
    try:
        with profile_operation("valid_iteration", "validation") as ctx:
            logger.log_growth_iteration(
                iteration=0,
                network=network,
                accuracy=0.85,
                growth_occurred=False
            )
            ctx.add_metric("validation_success", True)
        print("✅ Valid operation completed successfully!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 2: Invalid operation with profiling
    print("\n2. Testing invalid operation with profiling...")
    try:
        with profile_operation("invalid_iteration", "validation") as ctx:
            logger.log_growth_iteration(
                iteration=0,
                network=network,
                accuracy=1.5,  # ❌ Invalid: > 1.0
                growth_occurred=False
            )
            ctx.add_metric("validation_success", False)
        print("❌ Should have failed!")
    except Exception as e:
        print(f"✅ Caught validation error: {type(e).__name__}")
        print(f"   Profiling still tracked the failed operation")
    
    # Test 3: Profiling overhead during validation errors
    print("\n3. Testing profiling overhead during validation errors...")
    error_count = 0
    start_time = time.time()
    
    for i in range(10):
        try:
            with profile_operation(f"error_test_{i}", "validation"):
                logger.log_growth_iteration(
                    iteration=i,
                    network=network,
                    accuracy=1.2,  # ❌ Always invalid
                    growth_occurred=False
                )
        except:
            error_count += 1
    
    error_time = time.time() - start_time
    print(f"   Processed {error_count} validation errors in {error_time:.3f}s")
    print(f"   Profiling overhead during errors: {error_time/10:.4f}s per error")
    
    # Finish experiment
    profiling_results = profiler.end_session()
    logger.finish_experiment(final_accuracy=0.85, save_artifact=False)
    
    print("\n✅ Validation + profiling testing completed!")
    print(f"📊 Total operations profiled: {profiling_results.get('total_operations', 0)}")
    
    return profiling_results


def main():
    """Run all integrated logging and profiling examples."""
    print("🚀 INTEGRATED LOGGING + PROFILING SYSTEM EXAMPLES")
    print("="*80)
    print("This demo shows the integration of:")
    print("✅ Standardized logging with Pydantic validation")
    print("✅ Advanced profiling with adaptive overhead management")
    print("✅ Production-ready configurations")
    print("✅ Conditional and batch profiling")
    print("✅ Real-time monitoring and queue management")
    print("✅ Error handling and validation")
    
    try:
        # Run all examples
        artifact_1 = example_1_production_integration()
        artifact_2 = example_2_research_integration()
        artifact_3 = example_3_conditional_logging()
        profiling_4 = example_4_batch_logging_integration()
        profiling_5 = example_5_validation_with_profiling()
        
        print("\n" + "="*80)
        print("🎉 ALL INTEGRATED EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\n🎯 Integration Features Demonstrated:")
        print(f"   ✅ Production logging with minimal profiling overhead")
        print(f"   ✅ Research logging with comprehensive profiling")
        print(f"   ✅ Conditional profiling based on environment variables")
        print(f"   ✅ Batch profiling for high-frequency operations")
        print(f"   ✅ Error handling with profiling integration")
        print(f"   ✅ Component-level automatic profiling")
        print(f"   ✅ Memory-intensive operation profiling")
        print(f"   ✅ Real-time WandB integration")
        
        # Final queue status
        print("\n📊 Final queue status:")
        status = get_queue_status()
        for key, value in status.items():
            if key != 'directories':
                print(f"   {key}: {value}")
        
        print(f"\n💡 Best Practices Demonstrated:")
        print(f"   • Use production profilers for minimal overhead logging")
        print(f"   • Use research profilers for comprehensive analysis")
        print(f"   • Use conditional profiling for environment-specific behavior")
        print(f"   • Use batch profiling for high-frequency operations")
        print(f"   • Integrate profiling results with standardized logging")
        print(f"   • Handle validation errors gracefully with profiling")
        
        print(f"\n🔧 Next steps:")
        print(f"   1. Check WandB projects for real-time metrics and profiling data")
        print(f"   2. Process artifact queue: python -m structure_net.logging.cli process")
        print(f"   3. View integrated profiling + logging artifacts in WandB")
        print(f"   4. Use CLI tools: python -m structure_net.logging.cli status")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("💡 This might be due to missing dependencies or WandB setup")
        print("   Try: wandb login")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
