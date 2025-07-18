#!/usr/bin/env python3
"""
Test script for the refactored GPU seed hunter
"""

from src.structure_net.seed_search import GPUSeedHunter
from src.structure_net.core.io_operations import test_save_load_compatibility

def test_save_load_compatibility():
    """Test the canonical model I/O standard"""
    print("🧪 Testing canonical model I/O standard...")
    from src.structure_net.core.io_operations import test_save_load_compatibility
    success = test_save_load_compatibility()
    assert success, "Save/load compatibility test failed."
    print(f"Canonical standard test: {'✅ PASSED'}")

def test_refactored_seed_hunter():
    """Test the refactored GPU seed hunter"""
    print("\n🧪 Testing refactored GPU seed hunter...")
    
    hunter = GPUSeedHunter(dataset='mnist', save_promising=False)
    assert hunter is not None, "Failed to initialize GPU seed hunter."
    print("✅ GPU seed hunter initialized successfully with canonical standard!")
    print(f"   Input size: {hunter.input_size}")
    print(f"   Architecture generator: {type(hunter.arch_generator).__name__}")
    print(f"   Device: {hunter.device}")
    print(f"   Batch size: {hunter.batch_size}")

def test_architecture_generation():
    """Test architecture generation"""
    print("\n🧪 Testing architecture generation...")
    
    from src.structure_net.seed_search import ArchitectureGenerator
    
    # Test MNIST architectures
    mnist_gen = ArchitectureGenerator(784, 10)
    mnist_archs = mnist_gen.generate_mnist_architectures(10)
    assert len(mnist_archs) == 10, "Failed to generate MNIST architectures."
    print(f"✅ Generated {len(mnist_archs)} MNIST architectures")
    print(f"   Sample: {mnist_archs[0]}")
    
    # Test CIFAR-10 architectures  
    cifar_gen = ArchitectureGenerator(3072, 10)
    cifar_archs = cifar_gen.generate_cifar10_architectures(10)
    assert len(cifar_archs) == 10, "Failed to generate CIFAR-10 architectures."
    print(f"✅ Generated {len(cifar_archs)} CIFAR-10 architectures")
    print(f"   Sample: {cifar_archs[0]}")



def main():
    """Run all tests"""
    print("🔬 TESTING REFACTORED SEED HUNTER WITH CANONICAL STANDARD")
    print("=" * 60)
    
    tests = [
        test_canonical_standard,
        test_refactored_seed_hunter, 
        test_architecture_generation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Refactored seed hunter is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return all(results)

if __name__ == "__main__":
    main()
