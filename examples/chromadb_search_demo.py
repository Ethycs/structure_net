#!/usr/bin/env python3
"""
ChromaDB Search Demo for Structure Net

This script demonstrates how to use the ChromaDB search functionality
integrated with the data factory system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_factory import (
    ExperimentSearcher,
    search_similar_experiments,
    search_by_architecture,
    search_by_performance,
    get_chroma_client
)
import numpy as np


def demo_basic_indexing():
    """Demonstrate basic experiment indexing."""
    print("\n=== Basic Experiment Indexing ===")
    
    searcher = ExperimentSearcher()
    
    # Index a sample experiment
    experiment_data = {
        'experiment_id': 'demo_001',
        'experiment_type': 'training',
        'architecture': [784, 512, 256, 128, 10],
        'final_performance': {
            'accuracy': 0.945,
            'loss': 0.178,
            'precision': 0.943,
            'recall': 0.942,
            'f1_score': 0.942
        },
        'config': {
            'dataset': 'mnist',
            'epochs': 20,
            'batch_size': 128,
            'learning_rate': 0.001
        },
        'training_history': [
            {'epoch': 1, 'val_accuracy': 0.85},
            {'epoch': 5, 'val_accuracy': 0.91},
            {'epoch': 10, 'val_accuracy': 0.93},
            {'epoch': 20, 'val_accuracy': 0.945}
        ]
    }
    
    searcher.index_experiment(
        experiment_id='demo_001',
        experiment_data=experiment_data
    )
    
    print(f"✓ Indexed experiment: {experiment_data['experiment_id']}")
    print(f"  Architecture: {experiment_data['architecture']}")
    print(f"  Accuracy: {experiment_data['final_performance']['accuracy']}")
    
    # Index more experiments with variations
    architectures = [
        [784, 400, 200, 10],
        [784, 600, 300, 150, 10],
        [784, 1024, 512, 256, 128, 10],
        [784, 256, 128, 64, 10]
    ]
    
    for i, arch in enumerate(architectures):
        exp_id = f'demo_{i+2:03d}'
        accuracy = np.random.uniform(0.88, 0.96)
        
        searcher.index_experiment(
            experiment_id=exp_id,
            experiment_data={
                'experiment_id': exp_id,
                'architecture': arch,
                'final_performance': {'accuracy': accuracy},
                'config': {'dataset': 'mnist', 'epochs': 15}
            }
        )
    
    print(f"\n✓ Total experiments indexed: {searcher.get_experiment_count()}")


def demo_similarity_search():
    """Demonstrate similarity search."""
    print("\n=== Similarity Search ===")
    
    searcher = ExperimentSearcher()
    
    # Search for experiments similar to a query
    query_experiment = {
        'architecture': [784, 500, 250, 125, 10],
        'final_performance': {'accuracy': 0.92},
        'config': {'dataset': 'mnist'}
    }
    
    results = searcher.search_similar_experiments(
        query_experiment=query_experiment,
        n_results=3
    )
    
    print("\nSearching for experiments similar to:")
    print(f"  Architecture: {query_experiment['architecture']}")
    print(f"  Target accuracy: {query_experiment['final_performance']['accuracy']}")
    
    print("\nTop 3 similar experiments:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. ID: {result['id']}")
        print(f"   Similarity: {result['similarity_score']:.3f}")
        print(f"   Architecture: {result['metadata'].get('architecture', 'unknown')}")
        print(f"   Accuracy: {result['metadata'].get('accuracy', 0):.3f}")


def demo_architecture_search():
    """Demonstrate architecture-based search."""
    print("\n=== Architecture Search ===")
    
    # Search for experiments with similar architectures
    target_arch = [784, 512, 256, 128, 10]
    
    results = search_by_architecture(
        architecture=target_arch,
        n_results=3
    )
    
    print(f"\nSearching for architectures similar to: {target_arch}")
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. ID: {result['id']}")
        print(f"   Architecture: {result['architecture']}")
        print(f"   Similarity: {result['similarity_score']:.3f}")


def demo_performance_search():
    """Demonstrate performance-based search."""
    print("\n=== Performance Search ===")
    
    # Add some high-performing experiments
    searcher = ExperimentSearcher()
    
    high_performers = [
        {'id': 'high_001', 'accuracy': 0.98, 'params': 500_000},
        {'id': 'high_002', 'accuracy': 0.97, 'params': 1_200_000},
        {'id': 'high_003', 'accuracy': 0.96, 'params': 300_000},
        {'id': 'high_004', 'accuracy': 0.95, 'params': 800_000}
    ]
    
    for exp in high_performers:
        searcher.index_experiment(
            experiment_id=exp['id'],
            experiment_data={
                'experiment_id': exp['id'],
                'final_performance': {'accuracy': exp['accuracy']},
                'architecture': {'total_parameters': exp['params']},
                'config': {'dataset': 'mnist'}
            }
        )
    
    # Search for high performers
    results = search_by_performance(
        min_accuracy=0.95,
        max_parameters=1_000_000,
        dataset='mnist',
        n_results=5
    )
    
    print("\nSearching for high-performing experiments:")
    print("  Min accuracy: 0.95")
    print("  Max parameters: 1,000,000")
    print("  Dataset: mnist")
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. ID: {result['id']}")
        print(f"   Accuracy: {result['accuracy']:.3f}")
        print(f"   Parameters: {result['parameters']:,}")


def demo_batch_indexing():
    """Demonstrate batch indexing for efficiency."""
    print("\n=== Batch Indexing ===")
    
    searcher = ExperimentSearcher()
    
    # Prepare batch of experiments
    experiments = []
    for i in range(10):
        exp_id = f'batch_{i:03d}'
        exp_data = {
            'experiment_id': exp_id,
            'architecture': [784, 512 + i*50, 256, 10],
            'final_performance': {
                'accuracy': 0.90 + np.random.uniform(0, 0.08)
            },
            'config': {
                'dataset': 'mnist',
                'epochs': 10 + i
            }
        }
        experiments.append((exp_id, exp_data))
    
    # Index in batch
    searcher.index_experiments_batch(experiments)
    
    print(f"✓ Indexed {len(experiments)} experiments in batch")
    print(f"  Total experiments in database: {searcher.get_experiment_count()}")


def demo_hypothesis_search():
    """Demonstrate searching by hypothesis."""
    print("\n=== Hypothesis Search ===")
    
    searcher = ExperimentSearcher()
    
    # Index experiments for specific hypotheses
    hypotheses = ['optimal_seeds', 'canonical_operators', 'sparse_networks']
    
    for hyp_id in hypotheses:
        for i in range(3):
            exp_id = f'{hyp_id}_exp_{i:03d}'
            searcher.index_experiment(
                experiment_id=exp_id,
                experiment_data={
                    'experiment_id': exp_id,
                    'hypothesis_id': hyp_id,
                    'architecture': [784, 512, 256, 10],
                    'final_performance': {'accuracy': 0.85 + np.random.uniform(0, 0.1)},
                    'config': {'dataset': 'mnist'}
                },
                additional_metadata={'hypothesis_id': hyp_id}
            )
    
    # Search for experiments from a specific hypothesis
    results = searcher.search_by_hypothesis(
        hypothesis_id='optimal_seeds',
        n_results=5
    )
    
    print("\nSearching for experiments from hypothesis: optimal_seeds")
    print(f"\nFound {len(results)} experiments:")
    for result in results:
        print(f"  - {result['id']}")


def main():
    """Run all ChromaDB demos."""
    print("=== ChromaDB Search Demo for Structure Net ===")
    print("This demo shows how to use the integrated search functionality.")
    
    # Get client info
    client = get_chroma_client()
    print(f"\nChromaDB initialized")
    print(f"Existing experiments in database: {client.count()}")
    
    # Run demos
    demo_basic_indexing()
    demo_similarity_search()
    demo_architecture_search()
    demo_performance_search()
    demo_batch_indexing()
    demo_hypothesis_search()
    
    print("\n=== Demo Complete ===")
    print(f"Final experiment count: {client.count()}")
    print("\nYou can now use these search functions in your experiments!")
    print("Check the integration guide at: docs/data_system_integration_guide.md")


if __name__ == "__main__":
    main()