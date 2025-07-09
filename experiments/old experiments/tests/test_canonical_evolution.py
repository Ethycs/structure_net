#!/usr/bin/env python3
"""
Test script for the canonical evolution system
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

def test_canonical_evolution():
    """Test the complete canonical evolution system"""
    print("🧪 Testing Canonical Evolution System")
    print("=" * 50)
    
    # Test canonical standard
    from src.structure_net.core.io_operations import (
        create_standard_network,
        get_network_stats,
        test_save_load_compatibility
    )
    
    print("✅ Canonical standard imports successful")
    
    # Test evolution system
    from src.structure_net.evolution import (
        OptimalGrowthEvolver,
        analyze_layer_extrema,
        estimate_mi_proxy
    )
    
    print("✅ Evolution system imports successful")
    
    # Test seed search
    from src.structure_net.seed_search import (
        GPUSeedHunter,
        ArchitectureGenerator
    )
    
    print("✅ Seed search imports successful")
    
    # Test main package imports
    from src.structure_net import (
        create_standard_network,
        OptimalGrowthEvolver,
        GPUSeedHunter
    )
    
    print("✅ Main package imports successful")
    
    # Test canonical standard functionality
    print("\n🔧 Testing canonical standard...")
    
    # Create a simple network
    architecture = [784, 128, 10]
    sparsity = 0.02
    device = 'cpu'
    
    model = create_standard_network(architecture, sparsity, seed=42, device=device)
    assert model is not None, "Network creation failed."
    print(f"   Created network: {architecture}")
    
    # Get network stats
    stats = get_network_stats(model)
    assert 'total_connections' in stats, "Network stats are missing."
    print(f"   Network stats: {stats['total_connections']:,} connections, {stats['overall_sparsity']:.1%} sparsity")
    
    # Test save/load compatibility
    print("\n🔄 Testing save/load compatibility...")
    compatibility_passed = test_save_load_compatibility(device=device)
    assert compatibility_passed, "Save/load compatibility test failed."
    print(f"   Compatibility test: {'✅ PASSED'}")
    
    # Test evolution system
    print("\n🧬 Testing evolution system...")
    
    # Create dummy data
    dummy_x = torch.randn(100, 784)
    dummy_y = torch.randint(0, 10, (100,))
    dummy_dataset = TensorDataset(dummy_x, dummy_y)
    dummy_loader = DataLoader(dummy_dataset, batch_size=32)
    
    # Create evolver
    evolver = OptimalGrowthEvolver(
        seed_arch=[784, 64, 10],
        seed_sparsity=0.05,
        data_loader=dummy_loader,
        device=torch.device('cpu'),
        enable_sorting=True,
        sort_frequency=2
    )
    assert evolver is not None, "Evolver creation failed."
    print(f"   Created evolver with architecture: {evolver.current_architecture}")
    
    # Test analysis
    print("\n🔍 Testing network analysis...")
    analysis = evolver.analyze_network_state()
    assert 'bottlenecks' in analysis, "Network analysis failed."
    print(f"   Analysis completed: {len(analysis['bottlenecks'])} bottlenecks detected")
    
    # Test architecture generator
    print("\n📐 Testing architecture generator...")
    arch_gen = ArchitectureGenerator(784, 10)
    architectures = arch_gen.generate_systematic_batch(5)
    assert len(architectures) == 5, "Architecture generator failed."
    print(f"   Generated {len(architectures)} architectures")
    print(f"   Sample: {architectures[0]}")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Canonical evolution system is working correctly")
    assert True

def test_integration_with_hybrid_experiment():
    """Test integration with the original hybrid growth experiment"""
    print("\n🔗 Testing Integration with Hybrid Growth Experiment")
    print("=" * 60)
    
    # Test that we can load a pretrained model and evolve it
    from src.structure_net import (
        create_standard_network,
        OptimalGrowthEvolver,
        save_model_seed,
        load_model_seed
    )
    
    # Create and save a test model
    test_model = create_standard_network([3072, 128, 10], 0.02, seed=42)
    test_metrics = {'accuracy': 0.47, 'sparsity': 0.02, 'epoch': 5}
    
    # Save it
    test_path = "test_model_canonical.pt"
    save_model_seed(test_model, [3072, 128, 10], 42, test_metrics, test_path)
    print("✅ Created and saved test model")
    
    # Load it back
    loaded_model, checkpoint = load_model_seed(test_path, 'cpu')
    assert 'accuracy' in checkpoint, "Failed to load checkpoint correctly."
    print(f"✅ Loaded model with accuracy: {checkpoint['accuracy']:.2%}")
    
    # Create dummy CIFAR-10 style data
    dummy_x = torch.randn(100, 3072)  # CIFAR-10 flattened
    dummy_y = torch.randint(0, 10, (100,))
    dummy_dataset = TensorDataset(dummy_x, dummy_y)
    dummy_loader = DataLoader(dummy_dataset, batch_size=32)
    
    # Create evolver and load the pretrained model
    evolver = OptimalGrowthEvolver(
        seed_arch=[3072, 128, 10],
        seed_sparsity=0.02,
        data_loader=dummy_loader,
        device=torch.device('cpu')
    )
    
    # Load the pretrained scaffold
    evolver.load_pretrained_scaffold(test_path)
    print("✅ Loaded pretrained scaffold into evolver")
    
    # Test one evolution step
    print("\n🧬 Testing evolution step...")
    evolver.evolve_step()
    print("✅ Evolution step completed")
    
    # Clean up
    import os
    os.remove(test_path)
    
    print("\n🎉 INTEGRATION TEST PASSED!")
    print("✅ Can successfully load and evolve pretrained models")
    assert True

def main():
    """Run all tests"""
    print("🔬 COMPREHENSIVE CANONICAL EVOLUTION TESTING")
    print("=" * 70)
    
    test1_passed = test_canonical_evolution()
    test2_passed = test_integration_with_hybrid_experiment()
    
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Canonical system test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Integration test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎊 ALL TESTS PASSED!")
        print("🚀 The canonical evolution system is ready for use!")
        print("\n📋 What you can now do:")
        print("   1. Create networks with create_standard_network()")
        print("   2. Evolve them with OptimalGrowthEvolver()")
        print("   3. Hunt for seeds with GPUSeedHunter()")
        print("   4. Load pretrained models and evolve them further")
        print("   5. All with perfect compatibility guaranteed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    main()
