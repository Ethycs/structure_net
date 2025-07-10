#!/usr/bin/env python3
"""
Architecture Generator

Systematic generation of diverse network architectures for seed hunting.
"""

from typing import List


class ArchitectureGenerator:
    """
    Generates diverse network architectures for systematic exploration.
    
    This class systematically builds a "portfolio" of network shapes,
    each designed to test different architectural philosophies.
    """
    
    def __init__(self, input_size: int, num_classes: int):
        self.input_size = input_size
        self.num_classes = num_classes
    
    def generate_systematic_batch(self, num_architectures: int = 100) -> List[List[int]]:
        """
        Generate systematic batch of diverse architectures.
        
        Args:
            num_architectures: Maximum number of architectures to generate
            
        Returns:
            List of architecture specifications [input, hidden1, hidden2, ..., output]
        """
        architectures = []
        
        # Type 1: Direct Connections [input_size, C]
        # Purpose: Test if meaningful mapping can be learned without hidden representations
        for c in [10, 20, 30, 40, 50]:
            architectures.append([self.input_size, c])
            
        # Type 2: Classic MLP - Single Hidden Layer [input_size, H, C]
        # Purpose: Explore effect of single bottleneck with various widths
        for h in [16, 32, 64, 128, 256, 512]:
            architectures.append([self.input_size, h, self.num_classes])
            
        # Type 3: The Funnel - Decreasing Width, Two Hidden Layers
        # Purpose: Test progressive compression through deeper layers
        for h1 in [128, 256, 512]:
            for h2 in [32, 64, 128]:
                if h2 < h1:  # Decreasing size
                    architectures.append([self.input_size, h1, h2, self.num_classes])
                    
        # Type 4: Deep Funnel - Decreasing Width, Three Hidden Layers
        # Purpose: Push "funnel" idea further with more gradual compression
        for h1 in [256, 512]:
            for h2 in [128, 256]:
                for h3 in [32, 64]:
                    if h3 < h2 < h1:
                        architectures.append([self.input_size, h1, h2, h3, self.num_classes])
                        
        # Type 5: Wide and Shallow - Single, Massive Hidden Layer
        # Purpose: Test opposite extreme of deep funnels
        for w in [1024, 2048]:
            architectures.append([self.input_size, w, self.num_classes])
            
        # Type 6: The Column - Constant Width, Deep
        # Purpose: Test maintaining representational capacity throughout network
        narrow_arch = [self.input_size]
        for depth in range(5):
            narrow_arch.append(64)
        narrow_arch.append(self.num_classes)
        architectures.append(narrow_arch)
        
        print(f"📐 Generated {len(architectures)} systematic architectures:")
        print(f"   - Direct connections: 5 variants")
        print(f"   - Classic MLPs: 6 variants") 
        print(f"   - Funnels (2-layer): {sum(1 for h1 in [128,256,512] for h2 in [32,64,128] if h2 < h1)} variants")
        print(f"   - Deep funnels (3-layer): {sum(1 for h1 in [256,512] for h2 in [128,256] for h3 in [32,64] if h3 < h2 < h1)} variants")
        print(f"   - Wide & shallow: 2 variants")
        print(f"   - Deep columns: 1 variant")
        
        return architectures[:num_architectures]
    
    def generate_cifar10_architectures(self, num_architectures: int = 50) -> List[List[int]]:
        """Generate architectures optimized for CIFAR-10."""
        # Note: This method is kept for backward compatibility
        # The generic generate_systematic_batch now handles all datasets
        return self.generate_systematic_batch(num_architectures)
    
    def generate_mnist_architectures(self, num_architectures: int = 50) -> List[List[int]]:
        """Generate architectures optimized for MNIST."""
        # Note: This method is kept for backward compatibility
        # The generic generate_systematic_batch now handles all datasets
        return self.generate_systematic_batch(num_architectures)
    
    @classmethod
    def from_dataset(cls, dataset_name: str, num_architectures: int = 50) -> 'ArchitectureGenerator':
        """Create an ArchitectureGenerator for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            num_architectures: Number of architectures to generate
            
        Returns:
            ArchitectureGenerator instance configured for the dataset
        """
        from data_factory import get_dataset_config
        
        config = get_dataset_config(dataset_name)
        generator = cls(
            input_size=config.input_size,
            num_classes=config.num_classes
        )
        return generator
