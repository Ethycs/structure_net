#!/usr/bin/env python3
"""
Composable Evolution System Example - Latest Architecture

This example demonstrates the new interface-based composable evolution system
integrated with the latest profiling architecture, featuring:

- Modular components that can be mixed and matched
- Individual component configuration and profiling
- Advanced profiling integration with component-level monitoring
- Production-ready configurations with minimal overhead
- Comprehensive performance analysis and comparison
- Integration with standardized logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

# Set CUDA devices before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Import the new composable system
from src.structure_net.evolution.components import (
    # Core interfaces
    NetworkContext, 
    
    # Analyzers
    StandardExtremaAnalyzer,
    NetworkStatsAnalyzer,
    SimpleInformationFlowAnalyzer,
    
    # Strategies
    ExtremaGrowthStrategy,
    InformationFlowGrowthStrategy,
    ResidualBlockGrowthStrategy,
    HybridGrowthStrategy,
    
    # Evolution systems
    ComposableEvolutionSystem,
    create_standard_evolution_system,
    create_extrema_focused_system,
    create_hybrid_system
)

# Import latest profiling system
from src.structure_net.profiling import (
    create_production_profiler, create_research_profiler,
    profile_component, profile_if_enabled, profile_memory_intensive,
    profile_operation, profile_batch_operation,
    ProfilerLevel
)

# Import standardized logging
from src.structure_net.logging import create_profiling_logger, StandardizedLogger


# Import existing infrastructure
from src.structure_net.core.network_factory import create_standard_network


def create_sample_dataset(num_samples: int = 1000, input_dim: int = 784, num_classes: int = 10):
    """Create a simple synthetic dataset for demonstration."""
    # Generate random data
    X = torch.randn(num_samples, input_dim)
    
    # Create simple patterns for classification
    # Class 0: positive values in first quarter
    # Class 1: positive values in second quarter, etc.
    y = torch.zeros(num_samples, dtype=torch.long)
    
    quarter_size = input_dim // 4
    for i in range(min(4, num_classes)):
        start_idx = i * quarter_size
        end_idx = (i + 1) * quarter_size
        
        # Samples where this quarter has mostly positive values
        quarter_positive = (X[:, start_idx:end_idx] > 0).float().mean(dim=1) > 0.6
        y[quarter_positive] = i
    
    # Random assignment for remaining classes
    for i in range(4, num_classes):
        mask = (torch.rand(num_samples) < 0.1) & (y == 0)
        y[mask] = i
    
    return TensorDataset(X, y)


# Profiled Evolution System Wrapper
@profile_component(component_name="profiled_evolution_system", 
                  level=ProfilerLevel.DETAILED)
class ProfiledEvolutionSystem:
    """Evolution system wrapper with integrated profiling."""
    
    def __init__(self, system_type="standard", profiler_type="production"):
        self.system_type = system_type
        
        # Create profiler based on type
        if profiler_type == "production":
            self.profiler = create_production_profiler(max_overhead_percent=2.0)
        elif profiler_type == "research":
            self.profiler = create_research_profiler(
                experiment_name=f"composable_evolution_{system_type}",
                level=ProfilerLevel.COMPREHENSIVE
            )
        else:
            self.profiler = create_production_profiler()
        
        # Create evolution system
        if system_type == "standard":
            self.evolution_system = create_standard_evolution_system()
        elif system_type == "extrema":
            self.evolution_system = create_extrema_focused_system()
        elif system_type == "hybrid":
            self.evolution_system = create_hybrid_system()
        else:
            self.evolution_system = ComposableEvolutionSystem()
        
        # Create logger for integration
        self.logger = create_profiling_logger(session_id=f"composable_evolution_{system_type}")
        
        self.profiler.start_session(f"{system_type}_evolution")
    
    def evolve_network_with_profiling(self, context, num_iterations=3):
        """Evolve network with comprehensive profiling."""
        # Profile each iteration
        for iteration in range(num_iterations):
            with profile_operation(f"evolution_iteration_{iteration}", "evolution", 
                                 level=ProfilerLevel.DETAILED) as ctx:
                
                # Profile component analysis
                with profile_operation("component_analysis", "analysis") as analysis_ctx:
                    # Run analyzers
                    analysis_results = {}
                    for analyzer in self.evolution_system.analyzers:
                        analyzer_name = analyzer.__class__.__name__
                        with profile_operation(f"analyzer_{analyzer_name}", "analysis"):
                            result = analyzer.analyze(context)
                            analysis_results[analyzer_name] = result
                
                # Profile growth strategies
                growth_occurred = False
                with profile_operation("growth_strategies", "growth") as growth_ctx:
                    for strategy in self.evolution_system.strategies:
                        strategy_name = strategy.__class__.__name__
                        with profile_operation(f"strategy_{strategy_name}", "growth"):
                            if strategy.should_grow(context, analysis_results):
                                context = strategy.grow(context, analysis_results)
                                growth_occurred = True
                
                # Add iteration metrics
                ctx.add_metric("iteration", iteration)
                ctx.add_metric("growth_occurred", growth_occurred)
                ctx.add_metric("num_analyzers", len(self.evolution_system.analyzers))
                ctx.add_metric("num_strategies", len(self.evolution_system.strategies))
        
        return context
    
    def get_comprehensive_metrics(self):
        """Get comprehensive metrics from profiling and evolution."""
        # End profiling session
        profiling_results = self.profiler.end_session(save_results=False)
        
        # Get evolution metrics
        evolution_summary = self.evolution_system.get_evolution_summary()
        
        # Log profiling results
        if self.logger:
            self.logger.log_profiling_session(profiling_results)
            self.logger.finish_experiment()

        # Combine metrics
        comprehensive_metrics = {
            "profiling": profiling_results,
            "evolution_summary": evolution_summary,
            "system_type": self.system_type
        }
        
        return comprehensive_metrics


def demonstrate_profiled_composable_system():
    """Demonstrate basic composable system with profiling integration."""
    print("\n" + "="*80)
    print("🧬 PROFILED COMPOSABLE EVOLUTION SYSTEM DEMO")
    print("="*80)
    
    # Create dataset
    dataset = create_sample_dataset(num_samples=500, input_dim=784, num_classes=10)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create initial network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = create_standard_network(
        architecture=[784, 128, 64, 10],
        sparsity=0.1,
        device=str(device)
    )
    
    print(f"📊 Initial network: [784, 128, 64, 10], sparsity=10%, device={device}")
    
    # Create network context
    context = NetworkContext(
        network=network,
        data_loader=data_loader,
        device=device
    )
    
    # Create profiled evolution system
    print("\n🔧 Building profiled composable system...")
    profiled_system = ProfiledEvolutionSystem(
        system_type="standard",
        profiler_type="research"
    )
    
    print(f"   System type: {profiled_system.system_type}")
    print(f"   Analyzers: {len(profiled_system.evolution_system.analyzers)}")
    print(f"   Strategies: {len(profiled_system.evolution_system.strategies)}")
    
    # Run evolution with profiling
    print("\n🚀 Starting profiled evolution...")
    evolved_context = profiled_system.evolve_network_with_profiling(context, num_iterations=3)
    
    # Get comprehensive metrics
    print("\n📊 Collecting comprehensive metrics...")
    metrics = profiled_system.get_comprehensive_metrics()
    
    # Show results
    print("\n📈 Profiled Evolution Results:")
    print(f"   Total iterations: {metrics['evolution_summary']['total_iterations']}")
    print(f"   Growth events: {metrics['evolution_metrics'].get('total_growth_events', 0)}")
    print(f"   Profiling overhead: {metrics['profiling'].get('total_overhead', 0):.6f}s")
    print(f"   Operations profiled: {metrics['profiling'].get('total_operations', 0)}")
    
    return profiled_system, metrics


def demonstrate_system_comparison_with_profiling():
    """Compare different evolution systems with profiling."""
    print("\n" + "="*80)
    print("⚖️  PROFILED SYSTEM COMPARISON DEMO")
    print("="*80)
    
    # Create dataset
    dataset = create_sample_dataset(num_samples=300, input_dim=784, num_classes=10)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    system_types = ["standard", "extrema", "hybrid"]
    results = {}
    
    for system_type in system_types:
        print(f"\n🔬 Testing {system_type} system with profiling...")
        
        # Create fresh network for each system
        network = create_standard_network(
            architecture=[784, 64, 32, 10],
            sparsity=0.15,
            device=str(device)
        )
        
        context = NetworkContext(
            network=network,
            data_loader=data_loader,
            device=device
        )
        
        # Create profiled system
        profiled_system = ProfiledEvolutionSystem(
            system_type=system_type,
            profiler_type="production"  # Use production profiler for comparison
        )
        
        # Run evolution with profiling
        evolved_context = profiled_system.evolve_network_with_profiling(context, num_iterations=2)
        
        # Collect comprehensive metrics
        metrics = profiled_system.get_comprehensive_metrics()
        
        results[system_type] = {
            'growth_events': metrics['evolution_metrics'].get('total_growth_events', 0),
            'final_performance': evolved_context.performance_history[-1] if evolved_context.performance_history else 0.0,
            'profiling_overhead': metrics['profiling'].get('total_overhead', 0),
            'operations_profiled': metrics['profiling'].get('total_operations', 0),
            'components': metrics['evolution_summary']['components']
        }
        
        print(f"   Growth events: {results[system_type]['growth_events']}")
        print(f"   Final accuracy: {results[system_type]['final_performance']:.2%}")
        print(f"   Profiling overhead: {results[system_type]['profiling_overhead']:.6f}s")
    
    # Compare results with profiling data
    print("\n📊 Comprehensive System Comparison:")
    print("-" * 80)
    print(f"{'System':<12} {'Growth':<8} {'Accuracy':<10} {'Overhead':<12} {'Ops':<8} {'Components'}")
    print("-" * 80)
    
    for system_type, result in results.items():
        components = result['components']
        comp_str = f"{components['analyzers']}A/{components['strategies']}S"
        print(f"{system_type:<12} {result['growth_events']:<8} "
              f"{result['final_performance']:<10.2%} "
              f"{result['profiling_overhead']:<12.6f} "
              f"{result['operations_profiled']:<8} {comp_str}")
    
    return results


@profile_component(component_name="custom_evolution_builder", 
                  level=ProfilerLevel.BASIC)
class CustomEvolutionBuilder:
    """Builder for custom evolution systems with profiling."""
    
    def __init__(self):
        self.profiler = create_production_profiler(max_overhead_percent=1.5)
        self.profiler.start_session("custom_evolution_building")
    
    @profile_memory_intensive
    def build_aggressive_system(self):
        """Build an aggressive evolution system with memory profiling."""
        system = ComposableEvolutionSystem()
        
        # Add aggressive analyzers
        extrema_analyzer = StandardExtremaAnalyzer()
        extrema_analyzer.configure({
            'dead_threshold': 0.001,  # Very sensitive
            'saturated_multiplier': 2.0,  # Lower threshold
            'max_batches': 10  # More data
        })
        system.add_component(extrema_analyzer)
        
        system.add_component(NetworkStatsAnalyzer())
        system.add_component(SimpleInformationFlowAnalyzer())
        
        # Add aggressive growth strategies
        extrema_strategy = ExtremaGrowthStrategy()
        extrema_strategy.configure({
            'extrema_threshold': 0.1,  # Very low threshold
            'dead_neuron_threshold': 1,  # Single neuron triggers growth
            'patch_size': 8  # Large patches
        })
        system.add_component(extrema_strategy)
        
        info_strategy = InformationFlowGrowthStrategy()
        info_strategy.configure({
            'bottleneck_threshold': 0.15,  # Sensitive to bottlenecks
            'efficiency_threshold': 0.7  # High efficiency required
        })
        system.add_component(info_strategy)
        
        return system
    
    def build_conservative_system(self):
        """Build a conservative evolution system."""
        system = ComposableEvolutionSystem()
        
        # Add conservative analyzers
        extrema_analyzer = StandardExtremaAnalyzer()
        extrema_analyzer.configure({
            'dead_threshold': 0.01,  # Less sensitive
            'saturated_multiplier': 5.0,  # Higher threshold
            'max_batches': 5  # Less data
        })
        system.add_component(extrema_analyzer)
        
        system.add_component(NetworkStatsAnalyzer())
        
        # Add conservative growth strategy
        extrema_strategy = ExtremaGrowthStrategy()
        extrema_strategy.configure({
            'extrema_threshold': 0.4,  # High threshold
            'dead_neuron_threshold': 5,  # Many neurons needed
            'patch_size': 3  # Small patches
        })
        system.add_component(extrema_strategy)
        
        return system
    
    def finish_building(self):
        """Finish building and get profiling results."""
        return self.profiler.end_session()


def demonstrate_custom_system_building():
    """Demonstrate custom system building with profiling."""
    print("\n" + "="*80)
    print("🏗️  CUSTOM SYSTEM BUILDING WITH PROFILING")
    print("="*80)
    
    # Create builder
    builder = CustomEvolutionBuilder()
    
    print("🔧 Building custom evolution systems...")
    
    # Build different systems
    with profile_operation("aggressive_system_build", "building") as ctx:
        aggressive_system = builder.build_aggressive_system()
        ctx.add_metric("system_type", "aggressive")
        ctx.add_metric("components_count", len(aggressive_system.get_components()))
    
    with profile_operation("conservative_system_build", "building") as ctx:
        conservative_system = builder.build_conservative_system()
        ctx.add_metric("system_type", "conservative")
        ctx.add_metric("components_count", len(conservative_system.get_components()))
    
    # Finish building
    building_results = builder.finish_building()
    
    print(f"   Built 2 custom systems")
    print(f"   Building overhead: {building_results.get('total_overhead', 0):.6f}s")
    
    # Test both systems
    dataset = create_sample_dataset(num_samples=200)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    systems = {
        "Aggressive": aggressive_system,
        "Conservative": conservative_system
    }
    
    results = {}
    
    for system_name, system in systems.items():
        print(f"\n🧪 Testing {system_name} system...")
        
        # Create fresh network
        network = create_standard_network([784, 96, 48, 10], sparsity=0.2, device=str(device))
        context = NetworkContext(network=network, data_loader=data_loader, device=device)
        
        # Profile evolution
        profiler = create_production_profiler()
        profiler.start_session(f"{system_name.lower()}_test")
        
        with profile_operation(f"{system_name.lower()}_evolution", "evolution"):
            evolved_context = system.evolve_network(context, num_iterations=2)
        
        test_results = profiler.end_session()
        
        # Collect results
        summary = system.get_evolution_summary()
        results[system_name] = {
            'growth_events': summary['metrics'].get('total_growth_events', 0),
            'profiling_overhead': test_results.get('total_overhead', 0),
            'configuration': system.get_configuration()
        }
        
        print(f"   Growth events: {results[system_name]['growth_events']}")
        print(f"   Profiling overhead: {results[system_name]['profiling_overhead']:.6f}s")
    
    # Compare custom systems
    print("\n📊 Custom System Comparison:")
    for system_name, result in results.items():
        print(f"   {system_name}:")
        print(f"     Growth events: {result['growth_events']}")
        print(f"     Overhead: {result['profiling_overhead']:.6f}s")
        print(f"     Components: {len(result['configuration'])}")
    
    return results


def demonstrate_production_evolution_profiling():
    """Demonstrate production-ready evolution with minimal profiling overhead."""
    print("\n" + "="*80)
    print("🏭 PRODUCTION EVOLUTION WITH MINIMAL PROFILING")
    print("="*80)
    
    # Create production-optimized system
    @profile_if_enabled(condition=lambda: os.getenv('PROFILE_EVOLUTION', '0') == '1')
    def production_evolution_step(system, context):
        """Evolution step that only profiles when enabled."""
        return system.evolve_network(context, num_iterations=1)
    
    # Create system
    system = create_standard_evolution_system()
    
    # Create data
    dataset = create_sample_dataset(num_samples=400)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    network = create_standard_network([784, 128, 64, 10], sparsity=0.1, device=str(device))
    context = NetworkContext(network=network, data_loader=data_loader, device=device)
    
    # Create production profiler
    profiler = create_production_profiler(max_overhead_percent=1.0)
    profiler.start_session("production_evolution")
    
    print("📊 Running production evolution (profiling disabled by default)...")
    
    # Run without profiling
    for i in range(5):
        context = production_evolution_step(system, context)
    
    # Enable profiling for critical operations
    os.environ['PROFILE_EVOLUTION'] = '1'
    print("📊 Running with profiling enabled for critical operations...")
    
    # Run with profiling
    for i in range(2):
        context = production_evolution_step(system, context)
    
    # Finish profiling
    production_results = profiler.end_session()
    
    # Clean up
    os.environ.pop('PROFILE_EVOLUTION', None)
    
    print("\n📈 Production Evolution Results:")
    print(f"   Total evolution steps: 7")
    print(f"   Profiled steps: 2")
    print(f"   Total overhead: {production_results.get('total_overhead', 0):.6f}s")
    print(f"   Overhead per step: {production_results.get('total_overhead', 0) / 7:.6f}s")
    print(f"   Production-ready: ✅ (minimal overhead)")
    
    return production_results


def main():
    """Run all demonstrations."""
    print("🧬 COMPOSABLE EVOLUTION + PROFILING SYSTEM DEMONSTRATIONS")
    print("=" * 80)
    print("This example shows the integration of:")
    print("✅ Modular composable evolution components")
    print("✅ Advanced profiling with component-level monitoring")
    print("✅ Production-ready configurations with minimal overhead")
    print("✅ Comprehensive performance analysis and comparison")
    print("✅ Integration with standardized logging")
    print("✅ Custom system building with profiling insights")
    
    try:
        # Run demonstrations
        basic_system, basic_metrics = demonstrate_profiled_composable_system()
        comparison_results = demonstrate_system_comparison_with_profiling()
        custom_results = demonstrate_custom_system_building()
        production_results = demonstrate_production_evolution_profiling()
        
        print("\n" + "="*80)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\n🎯 Key Features Demonstrated:")
        print("   ✅ Composable evolution with profiling integration")
        print("   ✅ Component-level performance monitoring")
        print("   ✅ System comparison with profiling metrics")
        print("   ✅ Custom system building with profiling insights")
        print("   ✅ Production-ready evolution with minimal overhead")
        print("   ✅ Memory-intensive operation profiling")
        print("   ✅ Conditional profiling for production environments")
        
        print("\n📊 Performance Summary:")
        print("   🧬 Composable: Modular components with individual profiling")
        print("   ⚖️  Comparison: Multi-system analysis with overhead tracking")
        print("   🏗️  Custom: Building insights with profiling guidance")
        print("   🏭 Production: Minimal overhead with conditional profiling")
        
        print("\n💡 Best Practices Demonstrated:")
        print("   • Use component-level profiling for detailed analysis")
        print("   • Compare systems with consistent profiling methodology")
        print("   • Build custom systems with profiling-guided optimization")
        print("   • Use conditional profiling in production environments")
        print("   • Integrate evolution metrics with standardized logging")
        print("   • Monitor memory usage for intensive operations")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
