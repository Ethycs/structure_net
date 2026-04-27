#!/usr/bin/env python3
"""
Comprehensive Profiling System Example - Latest Architecture

This example demonstrates the latest optimized profiling system with:
- Advanced overhead management and adaptive sampling
- Component-level automatic profiling
- Conditional and threshold-based profiling
- Batch profiling for high-frequency operations
- Production-ready configurations
- Integration with standardized logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
import os

# Import the latest optimized profiling system
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.structure_net.profiling import (
    # Core profiling system
    ProfilerManager, ProfilerConfig, ProfilerLevel,
    ProfilerContext, BatchProfilerContext,
    
    # Advanced decorators
    profile_function, profile_method, profile_component,
    profile_if_enabled, profile_if_slow, profile_memory_intensive,
    
    # Context managers and utilities
    profile_operation, profile_batch_operation, profile_function_call,
    
    # Factory functions for optimized configurations
    create_standard_profiler, create_lightweight_profiler,
    create_comprehensive_profiler, create_evolution_focused_profiler,
    create_production_profiler, create_research_profiler,
    quick_evolution_profiler, quick_lightweight_profiler
)

# Import structure_net components
from src.structure_net.core.network_factory import create_standard_network
from src.structure_net.evolution.components import create_standard_evolution_system
from src.structure_net.evolution.interfaces import NetworkContext

# Import standardized logging for integration
from src.structure_net.logging import StandardizedLogger


# Example 1: Production-Ready Function Profiling
@profile_if_enabled(condition=lambda: os.getenv('PROFILE_FUNCTIONS', '0') == '1')
def production_computation(size: int = 1000):
    """Production function that only profiles when explicitly enabled."""
    data = torch.randn(size, size)
    result = torch.matmul(data, data.T)
    return result.sum().item()


@profile_if_slow(threshold_seconds=0.1)
def potentially_expensive_operation(complexity: float = 0.05):
    """Only profiles if operation takes longer than threshold."""
    time.sleep(complexity)
    return f"completed in {complexity}s"


@profile_memory_intensive
def memory_heavy_computation(size: int = 1000):
    """Automatically profiles memory usage for memory-intensive operations."""
    data = torch.randn(size, size, size)  # Large tensor
    result = torch.sum(data, dim=0)
    return result.mean().item()


# Example 2: Advanced Training Class with Optimized Profiling
@profile_component(component_name="advanced_trainer", 
                  exclude_methods=["__init__", "_private_method"],
                  level=ProfilerLevel.DETAILED)
class AdvancedTrainer:
    """Advanced trainer with automatic component-level profiling."""
    
    def __init__(self, model, device, profiler=None):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.profiler = profiler
        self.batch_count = 0
    
    def train_epoch(self, dataloader):
        """Train for one epoch with optimized profiling."""
        self.model.train()
        total_loss = 0.0
        
        # Use batch profiling for high-frequency training steps
        for batch_idx, (data, target) in enumerate(dataloader):
            with profile_batch_operation("training_step", "training", 
                                        {"batch_idx": batch_idx, "epoch": 1}):
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                self.batch_count += 1
            
            if batch_idx >= 10:  # Limit for demo
                break
        
        return total_loss / (batch_idx + 1)
    
    def evaluate(self, dataloader):
        """Evaluate model with automatic profiling."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)
                
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                if batch_idx >= 5:  # Limit for demo
                    break
        
        return correct / total
    
    def _private_method(self):
        """Private method excluded from profiling."""
        pass


# Example 3: Evolution System with Advanced Profiling
@profile_component(component_name="optimized_evolution", 
                  exclude_methods=["__init__"])
class OptimizedEvolutionSystem:
    """Evolution system with optimized automatic profiling."""
    
    def __init__(self, profiler=None):
        self.networks = []
        self.performance_history = []
        self.profiler = profiler
    
    def analyze_network(self, network):
        """Analyze network with automatic profiling and custom metrics."""
        # Simulate varying analysis complexity
        complexity = np.random.uniform(0.01, 0.1)
        time.sleep(complexity)
        
        # Add custom metrics through context if available
        if hasattr(self, '_current_context'):
            self._current_context.add_metric("analysis_complexity", complexity)
            self._current_context.add_metric("network_size", sum(p.numel() for p in network.parameters()))
        
        return {"analysis": "complete", "score": np.random.random(), "complexity": complexity}
    
    def grow_network(self, network, strategy="add_layer"):
        """Grow network with strategy-specific profiling."""
        # Simulate different growth strategies with different costs
        strategy_costs = {"add_layer": 0.2, "add_connections": 0.1, "prune": 0.05}
        cost = strategy_costs.get(strategy, 0.1)
        time.sleep(cost)
        
        if hasattr(self, '_current_context'):
            self._current_context.add_metric("growth_strategy", strategy)
            self._current_context.add_metric("growth_cost", cost)
        
        return network  # In real case, would return modified network
    
    def evaluate_performance(self, network, dataloader):
        """Evaluate performance with detailed profiling."""
        # Simulate evaluation with memory profiling
        with profile_operation("performance_evaluation", "evolution", 
                             level=ProfilerLevel.DETAILED) as ctx:
            time.sleep(0.15)
            performance = np.random.uniform(0.7, 0.95)
            
            ctx.add_metric("performance_score", performance)
            ctx.add_metric("evaluation_time", 0.15)
            
            return performance


def demonstrate_production_profiling():
    """Demonstrate production-ready profiling with minimal overhead."""
    print("\n" + "="*70)
    print("🏭 PRODUCTION PROFILING DEMONSTRATION")
    print("="*70)
    
    # Create production profiler with strict overhead limits
    profiler = create_production_profiler(max_overhead_percent=2.0)
    
    # Integrate with standardized logging
    logger = StandardizedLogger("production_profiling_demo")
    
    profiler.start_session("production_demo")
    
    print("\n📊 Running production profiling (overhead < 2%)...")
    
    # Test conditional profiling (disabled by default)
    print("   Testing conditional profiling...")
    for i in range(50):
        result = production_computation(100)
    
    # Enable profiling for critical operations
    os.environ['PROFILE_FUNCTIONS'] = '1'
    print("   Testing enabled conditional profiling...")
    for i in range(10):
        result = production_computation(200)
    
    # Test threshold-based profiling
    print("   Testing threshold-based profiling...")
    for i in range(20):
        # Fast operations (won't be profiled)
        potentially_expensive_operation(0.01)
        
        # Slow operations (will be profiled)
        if i % 5 == 0:
            potentially_expensive_operation(0.15)
    
    # Test memory-intensive profiling
    print("   Testing memory-intensive profiling...")
    for size in [100, 200, 300]:
        result = memory_heavy_computation(size)
    
    session_results = profiler.end_session()
    
    # Log results to standardized logging system
    logger.log_profiling_session(session_results)
    
    # Print production metrics
    print("\n📈 Production Profiling Results:")
    aggregated = profiler.get_aggregated_metrics()
    print(f"   Operations tracked: {aggregated['total_operations']}")
    print(f"   Total profiling time: {aggregated['total_time']:.4f}s")
    
    # Calculate and display overhead
    evolution_profiler = profiler.get_profiler("evolution_profiler")
    if evolution_profiler:
        overhead = evolution_profiler.average_overhead
        print(f"   Average overhead: {overhead:.6f}s per operation")
        print(f"   Overhead percentage: {(overhead / 0.01) * 100:.3f}%")
    
    # Clean up
    os.environ.pop('PROFILE_FUNCTIONS', None)
    
    return profiler, logger


def demonstrate_research_profiling():
    """Demonstrate comprehensive research profiling."""
    print("\n" + "="*70)
    print("🔬 RESEARCH PROFILING DEMONSTRATION")
    print("="*70)
    
    # Create research profiler with comprehensive tracking
    profiler = create_research_profiler(
        experiment_name="advanced_evolution_research",
        level=ProfilerLevel.COMPREHENSIVE,
        enable_all_integrations=True
    )
    
    profiler.start_session("research_demo")
    
    # Create sample data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.randn(200, 784)
    y = torch.randint(0, 10, (200,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32)
    
    print("\n🔬 Running comprehensive research profiling...")
    
    # Profile network creation with detailed metrics
    with profile_operation("network_creation", "research", 
                         level=ProfilerLevel.COMPREHENSIVE) as ctx:
        network = create_standard_network([784, 256, 128, 10], 0.02, device=str(device))
        param_count = sum(p.numel() for p in network.parameters())
        
        ctx.add_metric("parameter_count", param_count)
        ctx.add_metric("architecture", [784, 256, 128, 10])
        ctx.add_metric("device", str(device))
        ctx.add_metric("sparsity", 0.02)
    
    # Profile evolution system with advanced tracking
    evolution_system = OptimizedEvolutionSystem(profiler)
    
    for iteration in range(3):
        print(f"   Research iteration {iteration + 1}")
        
        # Profile complete evolution cycle
        with profile_operation(f"evolution_cycle_{iteration}", "research") as ctx:
            # Analysis phase
            analysis = evolution_system.analyze_network(network)
            
            # Growth phase with strategy comparison
            strategies = ["add_layer", "add_connections", "prune"]
            strategy = strategies[iteration % len(strategies)]
            grown_network = evolution_system.grow_network(network, strategy)
            
            # Evaluation phase
            performance = evolution_system.evaluate_performance(network, dataloader)
            
            # Add cycle metrics
            ctx.add_metric("iteration", iteration)
            ctx.add_metric("strategy_used", strategy)
            ctx.add_metric("analysis_score", analysis["score"])
            ctx.add_metric("performance_score", performance)
    
    session_results = profiler.end_session()
    
    # Print comprehensive research metrics
    print("\n📊 Research Profiling Results:")
    print(profiler.get_performance_report())
    
    # Show detailed component breakdown
    aggregated = profiler.get_aggregated_metrics()
    if aggregated['component_breakdown']:
        print("\n🔍 Component Performance Analysis:")
        for component, stats in aggregated['component_breakdown'].items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            print(f"   {component}: {stats['count']} ops, {avg_time:.4f}s avg, {stats['total_time']:.3f}s total")
    
    return profiler


def demonstrate_batch_profiling():
    """Demonstrate high-performance batch profiling."""
    print("\n" + "="*70)
    print("⚡ BATCH PROFILING DEMONSTRATION")
    print("="*70)
    
    # Create standard profiler for batch operations
    profiler = create_standard_profiler(level=ProfilerLevel.BASIC)
    profiler.start_session("batch_demo")
    
    # Create trainer with batch profiling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = create_standard_network([784, 128, 64, 10], 0.02, device=str(device))
    trainer = AdvancedTrainer(network, device, profiler)
    
    # Create data
    X = torch.randn(320, 784)  # More data for batch demo
    y = torch.randint(0, 10, (320,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print("\n⚡ Running batch profiling for training...")
    
    # Train with batch profiling (high-frequency operations)
    for epoch in range(2):
        print(f"   Epoch {epoch + 1} with batch profiling")
        
        # Training uses batch profiling internally
        train_loss = trainer.train_epoch(train_loader)
        
        # Evaluation uses automatic component profiling
        val_accuracy = trainer.evaluate(val_loader)
        
        print(f"      Train loss: {train_loss:.3f}, Val accuracy: {val_accuracy:.2%}")
    
    # Force flush of batch profiler
    from src.structure_net.profiling.core.context_manager import get_global_batch_profiler
    batch_profiler = get_global_batch_profiler()
    batch_profiler.flush()
    
    session_results = profiler.end_session()
    
    print(f"\n📈 Batch Profiling Results:")
    print(f"   Total batches processed: {trainer.batch_count}")
    aggregated = profiler.get_aggregated_metrics()
    print(f"   Total operations: {aggregated['total_operations']}")
    print(f"   Batch profiling overhead: Minimal (batched processing)")
    
    return profiler


def demonstrate_adaptive_profiling():
    """Demonstrate adaptive profiling with overhead management."""
    print("\n" + "="*70)
    print("🎯 ADAPTIVE PROFILING DEMONSTRATION")
    print("="*70)
    
    # Create profiler with strict overhead limits
    profiler = create_standard_profiler(
        level=ProfilerLevel.DETAILED,
        enable_memory=True,
        enable_compute=torch.cuda.is_available()
    )
    
    # Configure for adaptive behavior
    profiler.global_config.max_overhead_percent = 5.0
    profiler.global_config.adaptive_sampling = True
    
    profiler.start_session("adaptive_demo")
    
    print("\n🎯 Running adaptive profiling (will auto-adjust overhead)...")
    
    # Simulate operations that might cause high overhead
    for i in range(50):
        with profile_operation(f"adaptive_operation_{i}", "adaptive") as ctx:
            # Simulate varying operation costs
            cost = np.random.uniform(0.001, 0.01)
            time.sleep(cost)
            
            ctx.add_metric("operation_cost", cost)
            ctx.add_metric("iteration", i)
            
            # Add some expensive profiling operations
            if i % 10 == 0:
                ctx.add_metric("expensive_metric", sum(range(1000)))
    
    session_results = profiler.end_session()
    
    print("\n📊 Adaptive Profiling Results:")
    evolution_profiler = profiler.get_profiler("evolution_profiler")
    if evolution_profiler:
        print(f"   Final profiling level: {evolution_profiler.current_level.name}")
        print(f"   Average overhead: {evolution_profiler.average_overhead:.6f}s")
        print(f"   Overhead adjustments made: {evolution_profiler.level_adjustments}")
    
    return profiler


def demonstrate_integration_with_logging():
    """Demonstrate integration with standardized logging system."""
    print("\n" + "="*70)
    print("📝 PROFILING + LOGGING INTEGRATION DEMONSTRATION")
    print("="*70)
    
    # Create profiler with logging integration
    profiler = create_comprehensive_profiler(
        output_dir="integrated_profiling",
        enable_wandb=False  # Set to True if wandb is configured
    )
    
    # Create standardized logger
    logger = StandardizedLogger("profiling_integration_demo")
    
    profiler.start_session("integration_demo")
    
    print("\n📝 Running integrated profiling and logging...")
    
    # Simulate experiment with both profiling and logging
    experiment_data = {
        "experiment_id": "integration_test_001",
        "timestamp": time.time(),
        "configuration": {
            "profiling_level": "comprehensive",
            "logging_enabled": True,
            "adaptive_sampling": True
        }
    }
    
    # Log experiment start
    logger.log_experiment_start(experiment_data)
    
    # Run profiled operations
    for phase in ["initialization", "training", "evaluation"]:
        with profile_operation(f"experiment_phase_{phase}", "experiment") as ctx:
            # Simulate phase-specific work
            phase_duration = {"initialization": 0.1, "training": 0.3, "evaluation": 0.2}
            time.sleep(phase_duration[phase])
            
            # Log phase completion
            phase_data = {
                "phase": phase,
                "duration": phase_duration[phase],
                "status": "completed"
            }
            logger.log_iteration(phase_data)
            
            ctx.add_metric("phase_name", phase)
            ctx.add_metric("expected_duration", phase_duration[phase])
    
    session_results = profiler.end_session()
    
    # Log profiling results
    logger.log_profiling_session(session_results)
    
    # Log experiment completion
    logger.log_experiment_end({
        "status": "completed",
        "total_duration": sum(phase_duration.values()),
        "profiling_overhead": session_results.get('total_overhead', 0)
    })
    
    print("\n📊 Integration Results:")
    print(f"   Profiling session: {session_results['session_id']}")
    print(f"   Logging artifacts: {len(logger.get_artifacts())}")
    print(f"   Combined data available for analysis")
    
    return profiler, logger


def main():
    """Run all advanced profiling demonstrations."""
    print("🚀 ADVANCED PROFILING SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        production_profiler, prod_logger = demonstrate_production_profiling()
        research_profiler = demonstrate_research_profiling()
        batch_profiler = demonstrate_batch_profiling()
        adaptive_profiler = demonstrate_adaptive_profiling()
        integrated_profiler, int_logger = demonstrate_integration_with_logging()
        
        print("\n" + "="*80)
        print("✅ ALL ADVANCED PROFILING DEMONSTRATIONS COMPLETED!")
        print("="*80)
        
        print(f"\n🎯 Advanced Features Demonstrated:")
        print(f"   ✅ Production-ready profiling (< 2% overhead)")
        print(f"   ✅ Conditional profiling (@profile_if_enabled)")
        print(f"   ✅ Threshold-based profiling (@profile_if_slow)")
        print(f"   ✅ Memory-intensive profiling (@profile_memory_intensive)")
        print(f"   ✅ Component-level automatic profiling (@profile_component)")
        print(f"   ✅ Batch profiling for high-frequency operations")
        print(f"   ✅ Adaptive overhead management")
        print(f"   ✅ Research-grade comprehensive profiling")
        print(f"   ✅ Integration with standardized logging")
        print(f"   ✅ Context managers with custom metrics")
        
        print(f"\n📊 Performance Summary:")
        print(f"   🏭 Production: Minimal overhead with conditional profiling")
        print(f"   🔬 Research: Comprehensive insights with adaptive sampling")
        print(f"   ⚡ Batch: High-frequency operations with minimal impact")
        print(f"   🎯 Adaptive: Automatic overhead management")
        print(f"   📝 Integrated: Seamless logging and profiling")
        
        print(f"\n💡 Best Practices Demonstrated:")
        print(f"   • Use create_production_profiler() for production environments")
        print(f"   • Use @profile_component for automatic class profiling")
        print(f"   • Use profile_batch_operation() for high-frequency operations")
        print(f"   • Use conditional decorators for optional profiling")
        print(f"   • Integrate with standardized logging for complete tracking")
        print(f"   • Let adaptive sampling manage overhead automatically")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
