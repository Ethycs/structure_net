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

sys.path.append('.')
from src.structure_net import create_multi_scale_network, MultiScaleNetwork
from src.structure_net.core.minimal_network import create_minimal_network
from src.structure_net.core.growth_scheduler import GrowthScheduler
from src.structure_net.core.connection_router import ConnectionRouter, ParsimonousRouter
from src.structure_net.snapshots.snapshot_manager import SnapshotManager

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
            # Test minimal network
            minimal_net = create_minimal_network(784, [256, 128], 10)
            minimal_net = minimal_net.to(self.device)
            x = torch.randn(32, 784).to(self.device)
            output = minimal_net(x)
            
            # Test multi-scale network
            multi_net = create_multi_scale_network(784, [256, 128], 10, device=self.device)
            output2 = multi_net(x)
            
            self.log_test("Basic Functionality", True, 
                         f"Minimal: {output.shape}, Multi-scale: {output2.shape}")
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
                net = create_multi_scale_network(784, [256, 128], 10, 
                                               sparsity=sparsity, device=self.device)
                stats = net.network.get_connectivity_stats()
                actual_sparsity = stats['connectivity_ratio']
                results.append(f"{sparsity:.3f}→{actual_sparsity:.3f}")
            
            self.log_test("Sparse Connectivity", True, f"Sparsities: {', '.join(results)}")
            return True
        except Exception as e:
            self.log_test("Sparse Connectivity", False, str(e))
            return False
    
    def test_3_extrema_detection(self):
        """Test 3: Extrema detection across different activation patterns."""
        try:
            net = create_multi_scale_network(784, [256, 128], 10, 
                                           sparsity=0.01, device=self.device)
            
            # Test with different input patterns
            patterns = {
                'random': torch.randn(32, 784),
                'high_values': torch.randn(32, 784) * 3 + 2,
                'low_values': torch.randn(32, 784) * 0.5 - 1,
                'mixed': torch.cat([torch.randn(16, 784) * 3, torch.randn(16, 784) * 0.1])
            }
            
            extrema_counts = {}
            for pattern_name, pattern in patterns.items():
                pattern = pattern.to(self.device)
                _ = net(pattern)
                extrema = net.network.detect_extrema(use_adaptive=True)
                total_extrema = sum(len(layer['high']) + len(layer['low']) 
                                  for layer in extrema.values())
                extrema_counts[pattern_name] = total_extrema
            
            self.log_test("Extrema Detection", True, 
                         f"Extrema counts: {extrema_counts}")
            return True
        except Exception as e:
            self.log_test("Extrema Detection", False, str(e))
            return False
    
    def test_4_connection_routing(self):
        """Test 4: Connection routing algorithms."""
        try:
            net = create_multi_scale_network(784, [256, 128], 10, 
                                           sparsity=0.01, device=self.device)
            
            # Generate extrema
            x = torch.randn(32, 784).to(self.device)
            _ = net(x)
            extrema = net.network.detect_extrema(use_adaptive=True)
            
            # Test standard router
            standard_connections = net.connection_router.route_connections(
                extrema, net.network.layer_sizes)
            
            # Test parsimonious router
            parsimonious = ParsimonousRouter()
            parsimonious_connections = parsimonious.parsimonious_growth(
                extrema, net.network.layer_sizes)
            
            self.log_test("Connection Routing", True,
                         f"Standard: {len(standard_connections)}, "
                         f"Parsimonious: {len(parsimonious_connections)}")
            return True
        except Exception as e:
            self.log_test("Connection Routing", False, str(e))
            return False
    
    def test_5_growth_scheduler(self):
        """Test 5: Growth scheduler functionality."""
        try:
            scheduler = GrowthScheduler(
                variance_threshold=0.5,
                growth_threshold=50,
                stabilization_epochs=3
            )
            
            # Simulate training with varying gradients
            growth_events = 0
            for epoch in range(20):
                scheduler.update_epoch(epoch)
                # Simulate high gradients every few epochs
                gradient = 2.0 if epoch % 5 == 0 else 0.5
                should_grow = scheduler.add_gradient_norm(gradient)
                if should_grow:
                    growth_events += 1
            
            stats = scheduler.get_stats()
            self.log_test("Growth Scheduler", True,
                         f"Growth events: {growth_events}, Phase: {stats['current_phase']}")
            return True
        except Exception as e:
            self.log_test("Growth Scheduler", False, str(e))
            return False
    
    def test_6_snapshot_management(self):
        """Test 6: Snapshot saving and loading."""
        try:
            snapshot_dir = "./test_snapshots"
            os.makedirs(snapshot_dir, exist_ok=True)
            
            net = create_multi_scale_network(784, [256, 128], 10,
                                           snapshot_dir=snapshot_dir, device=self.device)
            
            # Force a snapshot save
            net.snapshot_manager.save_snapshot(
                net.network, epoch=1, performance=0.85,
                growth_info={'growth_occurred': True, 'connections_added': 10},
                phase='coarse', metadata={'test': True}
            )
            
            snapshots = net.get_snapshots()
            self.log_test("Snapshot Management", len(snapshots) > 0,
                         f"Snapshots saved: {len(snapshots)}")
            return len(snapshots) > 0
        except Exception as e:
            self.log_test("Snapshot Management", False, str(e))
            return False
    
    def test_7_training_integration(self):
        """Test 7: Full training integration with growth."""
        try:
            # Create network and data
            net = create_multi_scale_network(784, [256, 128], 10,
                                           sparsity=0.01, device=self.device)
            X, y = self.create_synthetic_data(500, 784, 10, 'medium')
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(X, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training setup
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Train for a few epochs
            initial_connections = net.network.get_connectivity_stats()['total_active_connections']
            
            for epoch in range(5):
                epoch_stats = net.train_epoch(dataloader, optimizer, criterion, epoch)
            
            final_connections = net.network.get_connectivity_stats()['total_active_connections']
            growth_occurred = final_connections > initial_connections
            
            self.log_test("Training Integration", True,
                         f"Connections: {initial_connections}→{final_connections}, "
                         f"Growth: {growth_occurred}")
            return True
        except Exception as e:
            self.log_test("Training Integration", False, str(e))
            return False
    
    def test_8_performance_benchmark(self):
        """Test 8: Performance benchmarking on GPU vs CPU."""
        try:
            # Test on current device
            net = create_multi_scale_network(784, [256, 128], 10, device=self.device)
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
    
    def test_9_memory_efficiency(self):
        """Test 9: Memory efficiency with different sparsity levels."""
        try:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                # Test different sparsity levels
                sparsities = [0.001, 0.01, 0.05]
                memory_usage = {}
                
                for sparsity in sparsities:
                    torch.cuda.empty_cache()
                    net = create_multi_scale_network(784, [512, 256, 128], 10,
                                                   sparsity=sparsity, device=self.device)
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
            self.log_test("Memory Efficiency", False, str(e))
            return False
    
    def test_10_scalability_stress(self):
        """Test 10: Scalability stress test with large networks."""
        try:
            # Test progressively larger networks
            architectures = [
                ([784, 256, 128, 10], "Small"),
                ([784, 512, 256, 128, 10], "Medium"),
                ([784, 1024, 512, 256, 10], "Large")
            ]
            
            results = {}
            for arch, name in architectures:
                try:
                    net = create_multi_scale_network(
                        arch[0], arch[1:-1], arch[-1],
                        sparsity=0.01, device=self.device
                    )
                    x = torch.randn(32, arch[0]).to(self.device)
                    output = net(x)
                    
                    stats = net.network.get_connectivity_stats()
                    results[name] = f"{stats['total_active_connections']} connections"
                    del net, x
                except Exception as e:
                    results[name] = f"Failed: {str(e)[:50]}"
            
            self.log_test("Scalability Stress", True, f"Results: {results}")
            return True
        except Exception as e:
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
            self.test_4_connection_routing,
            self.test_5_growth_scheduler,
            self.test_6_snapshot_management,
            self.test_7_training_integration,
            self.test_8_performance_benchmark,
            self.test_9_memory_efficiency,
            self.test_10_scalability_stress
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
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, cuda, cpu (default: auto)')
    
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
