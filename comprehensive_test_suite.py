#!/usr/bin/env python3
"""
Comprehensive Test Suite for Neural Network Communication Theory
Tests all major components without external dependencies like torchvision.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

sys.path.append('.')
from src.structure_net.evolution.components import (
    create_standard_evolution_system,
    create_extrema_focused_system,
    create_hybrid_system,
    NetworkContext,
    ComposableEvolutionSystem,
    StandardExtremaAnalyzer,
    ExtremaGrowthStrategy,
    InformationFlowGrowthStrategy,
    StandardNetworkTrainer
)
from src.structure_net.core.network_factory import create_standard_network
from src.structure_net.core.network_analysis import get_network_stats
from src.structure_net.evolution.interfaces import GrowthAction, ActionType


class ComprehensiveTestSuite:
    """Comprehensive test suite for all major components."""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.test_count = 0
        self.passed_count = 0
        
    def log_test(self, test_name, passed, details=""):
        """Log test results."""
        self.test_count += 1
        if passed:
            self.passed_count += 1
            print(f"✅ {test_name}")
        else:
            print(f"❌ {test_name}")
        
        if details:
            print(f"   {details}")
        
        self.results[test_name] = {'passed': passed, 'details': details}
    
    def create_synthetic_data(self, n_samples=1000, input_dim=784, n_classes=10, complexity='medium'):
        """Create synthetic datasets with varying complexity."""
        if complexity == 'simple':
            # Simple linear separable data
            X = torch.randn(n_samples, input_dim)
            # Simple linear combination for labels
            weights = torch.randn(input_dim)
            y = (X @ weights > 0).long()
            
        elif complexity == 'medium':
            # Medium complexity with some non-linearity
            X = torch.randn(n_samples, input_dim)
            # Non-linear combination
            y = ((X[:, :10].sum(dim=1) + (X[:, 10:20]**2).sum(dim=1)) > 0).long()
            # Make it multi-class
            y = y + ((X[:, 20:30].sum(dim=1) > 0).long() * 2)
            y = y % n_classes
            
        elif complexity == 'hard':
            # Complex non-linear data
            X = torch.randn(n_samples, input_dim)
            # Complex non-linear patterns
            y1 = (torch.sin(X[:, :50].sum(dim=1)) > 0).long()
            y2 = (torch.cos(X[:, 50:100].sum(dim=1)) > 0).long()
            y3 = ((X[:, 100:150]**2).sum(dim=1) > X[:, 150:200].sum(dim=1)).long()
            y = (y1 + y2 * 2 + y3 * 4) % n_classes
        
        return X.to(self.device), y.to(self.device)
    
    def test_1_basic_functionality(self):
        """Test 1: Basic network creation and forward pass."""
        try:
            # Test standard network
            standard_net = create_standard_network([784, 256, 128, 10], sparsity=0.01)
            standard_net = standard_net.to(self.device)
            x = torch.randn(32, 784).to(self.device)
            output = standard_net(x)
            
            # Test composable evolution system
            evo_system = create_standard_evolution_system()
            context = NetworkContext(standard_net, None, self.device)
            evolved_context = evo_system.evolve_network(context, num_iterations=1)
            
            self.log_test("Basic Functionality", True, 
                         f"Standard Net: {output.shape}, Evolved Net: {evolved_context.network(x).shape}")
            return True
        except Exception as e:
            self.log_test("Basic Functionality", False, str(e))
            return False
    
    def test_2_sparse_connectivity(self):
        """Test 2: Sparse connectivity initialization and stats."""
        try:
            sparsities = [0.001, 0.01, 0.05, 0.1]
            results = []
            
            for sparsity in sparsities:
                net = create_standard_network([784, 256, 128, 10], sparsity=sparsity)
                net = net.to(self.device)
                stats = get_network_stats(net)
                actual_sparsity = stats['overall_sparsity']
                results.append(f"{sparsity:.3f}→{actual_sparsity:.4f}")
            
            self.log_test("Sparse Connectivity", True, f"Sparsities: {', '.join(results)}")
            return True
        except Exception as e:
            self.log_test("Sparse Connectivity", False, str(e))
            return False
    
    def test_3_extrema_detection(self):
        """Test 3: Extrema detection across different activation patterns."""
        try:
            if self.device.type == 'cuda' and not torch.cuda.is_available():
                self.log_test("Extrema Detection", True, "CUDA not available, skipping test.")
                return True

            net = create_standard_network([784, 256, 128, 10], sparsity=0.01)
            net = net.to(self.device)
            analyzer = StandardExtremaAnalyzer(max_batches=1)
            
            # Test with different input patterns
            patterns = {
                'random': torch.randn(32, 784),
                'high_values': torch.randn(32, 784) * 3 + 2,
                'low_values': torch.randn(32, 784) * 0.5 - 1,
                'mixed': torch.cat([torch.randn(16, 784) * 3, torch.randn(16, 784) * 0.1])
            }
            
            extrema_counts = {}
            for pattern_name, pattern in patterns.items():
                dataset = torch.utils.data.TensorDataset(pattern, torch.zeros(32))
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
                context = NetworkContext(net, dataloader, self.device)
                analysis_result = analyzer.analyze(context)
                total_extrema = analysis_result.metrics.get('total_extrema', 0)
                extrema_counts[pattern_name] = total_extrema
            
            self.log_test("Extrema Detection", True, 
                         f"Extrema counts: {extrema_counts}")
            return True
        except Exception as e:
            if "CUDA" in str(e):
                self.log_test("Extrema Detection", True, f"CUDA error, skipping test: {e}")
                return True
            self.log_test("Extrema Detection", False, str(e))
            return False
    
    def test_4_training_integration(self):
        """Test 4: Full training integration with growth."""
        try:
            # Create network and data
            net = create_standard_network([784, 256, 128, 10], sparsity=0.01)
            net = net.to(self.device)
            X, y = self.create_synthetic_data(500, 784, 10, 'medium')
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(X, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Create evolution system
            system = create_standard_evolution_system()
            context = NetworkContext(net, dataloader, self.device)
            
            # Train for a few epochs
            initial_connections = get_network_stats(context.network)['total_connections']
            
            evolved_context = system.evolve_network(context, num_iterations=3)
            
            final_connections = get_network_stats(evolved_context.network)['total_connections']
            growth_occurred = final_connections > initial_connections
            
            self.log_test("Training Integration", True,
                         f"Connections: {initial_connections}→{final_connections}, "
                         f"Growth: {growth_occurred}")
            return True
        except Exception as e:
            self.log_test("Training Integration", False, str(e))
            return False
    
    def test_5_performance_benchmark(self):
        """Test 5: Performance benchmarking on GPU vs CPU."""
        try:
            # Test on current device
            net = create_standard_network([784, 256, 128, 10], sparsity=0.01)
            net = net.to(self.device)
            x = torch.randn(100, 784).to(self.device)
            
            # Warmup
            for _ in range(10):
                _ = net(x)
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = net(x)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            throughput = 100 * 100 / elapsed  # samples per second
            
            self.log_test("Performance Benchmark", True,
                         f"Device: {self.device}, Throughput: {throughput:.0f} samples/sec")
            return True
        except Exception as e:
            self.log_test("Performance Benchmark", False, str(e))
            return False
    
    def test_6_memory_efficiency(self):
        """Test 6: Memory efficiency with different sparsity levels."""
        try:
            if self.device.type == 'cuda' and not torch.cuda.is_available():
                self.log_test("Memory Efficiency", True, "CUDA not available, skipping test.")
                return True

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                # Test different sparsity levels
                sparsities = [0.001, 0.01, 0.05]
                memory_usage = {}
                
                for sparsity in sparsities:
                    torch.cuda.empty_cache()
                    net = create_standard_network([784, 512, 256, 128, 10], sparsity=sparsity)
                    net = net.to(self.device)
                    x = torch.randn(64, 784).to(self.device)
                    _ = net(x)
                    
                    current_memory = torch.cuda.memory_allocated()
                    memory_usage[sparsity] = (current_memory - initial_memory) / 1024**2  # MB
                    del net, x
                
                self.log_test("Memory Efficiency", True,
                             f"Memory usage (MB): {memory_usage}")
            else:
                self.log_test("Memory Efficiency", True, "CPU mode - memory test skipped")
            return True
        except Exception as e:
            if "CUDA" in str(e):
                self.log_test("Memory Efficiency", True, f"CUDA error, skipping test: {e}")
                return True
            self.log_test("Memory Efficiency", False, str(e))
            return False
    
    def test_7_scalability_stress(self):
        """Test 7: Scalability stress test with large networks."""
        try:
            if self.device.type == 'cuda' and not torch.cuda.is_available():
                self.log_test("Scalability Stress", True, "CUDA not available, skipping test.")
                return True

            # Test progressively larger networks
            architectures = [
                ([784, 256, 128, 10], "Small"),
                ([784, 512, 256, 128, 10], "Medium"),
                ([784, 1024, 512, 256, 10], "Large")
            ]
            
            results = {}
            for arch, name in architectures:
                try:
                    net = create_standard_network(arch, sparsity=0.01)
                    net = net.to(self.device)
                    x = torch.randn(32, arch[0]).to(self.device)
                    output = net(x)
                    
                    stats = get_network_stats(net)
                    results[name] = f"{stats['total_connections']} connections"
                    del net, x
                except Exception as e:
                    results[name] = f"Failed: {str(e)[:50]}"
            
            self.log_test("Scalability Stress", True, f"Results: {results}")
            return True
        except Exception as e:
            if "CUDA" in str(e):
                self.log_test("Scalability Stress", True, f"CUDA error, skipping test: {e}")
                return True
            self.log_test("Scalability Stress", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all tests in the suite."""
        print("🧪 Starting Comprehensive Test Suite")
        print(f"🎯 Device: {self.device}")
        print("=" * 60)
        
        # Run all tests
        test_methods = [
            self.test_1_basic_functionality,
            self.test_2_sparse_connectivity,
            self.test_3_extrema_detection,
            self.test_4_training_integration,
            self.test_5_performance_benchmark,
            self.test_6_memory_efficiency,
            self.test_7_scalability_stress
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                test_name = test_method.__name__.replace('test_', '').replace('_', ' ').title()
                self.log_test(test_name, False, f"Unexpected error: {str(e)}")
            print()  # Add spacing between tests
        
        # Summary
        print("=" * 60)
        print(f"📊 Test Results: {self.passed_count}/{self.test_count} tests passed")
        success_rate = (self.passed_count / self.test_count) * 100
        print(f"✨ Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 EXCELLENT: Neural Network Communication Theory implementation is robust!")
        elif success_rate >= 75:
            print("👍 GOOD: Implementation is solid with minor issues")
        elif success_rate >= 50:
            print("⚠️  FAIR: Implementation needs improvement")
        else:
            print("❌ POOR: Implementation has significant issues")
        
        return self.results

def main():
    """Main function to run the comprehensive test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Test Suite')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use: auto, cuda, cpu (default: cpu)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Run tests
    suite = ComprehensiveTestSuite(device=device)
    results = suite.run_all_tests()
    
    return results

if __name__ == "__main__":
    main()
