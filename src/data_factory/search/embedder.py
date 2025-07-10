"""
Embedding functions for converting experiments, architectures, and hypotheses
into vector representations for ChromaDB.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import hashlib
import json

from structure_net.logging.schemas import (
    ExperimentSchema,
    NetworkArchitecture,
    PerformanceMetrics,
    GrowthExperiment,
    TrainingExperiment
)
from neural_architecture_lab.core import Hypothesis, ExperimentResult


class ExperimentEmbedder:
    """Converts experiment data into vector embeddings."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    def embed(self, experiment: Union[Dict[str, Any], ExperimentSchema]) -> np.ndarray:
        """
        Create an embedding for an experiment.
        
        The embedding captures:
        - Architecture characteristics (depth, width, parameters)
        - Performance metrics (accuracy, loss, efficiency)
        - Training dynamics (convergence speed, stability)
        - Growth patterns (if applicable)
        """
        features = []
        
        # Extract experiment data
        if isinstance(experiment, dict):
            exp_data = experiment
        else:
            exp_data = experiment.model_dump()
        
        # Architecture features
        if 'architecture' in exp_data:
            arch = exp_data['architecture']
            if isinstance(arch, dict):
                features.extend([
                    arch.get('depth', 0) / 20.0,  # Normalized depth
                    arch.get('total_parameters', 0) / 1e7,  # Normalized params
                    arch.get('sparsity', 0.0),
                    len(arch.get('layers', [])) / 20.0,
                ])
                
                # Layer size statistics
                layers = arch.get('layers', [])
                if layers:
                    features.extend([
                        np.mean(layers) / 1000.0,
                        np.std(layers) / 1000.0,
                        np.max(layers) / 1000.0,
                        np.min(layers) / 1000.0,
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                # Legacy format - list of layer sizes
                features.extend([
                    len(arch) / 20.0,  # Depth
                    sum(arch) / 1e7,  # Approx params
                    0.02,  # Default sparsity
                    len(arch) / 20.0,
                ])
                if arch:
                    features.extend([
                        np.mean(arch) / 1000.0,
                        np.std(arch) / 1000.0,
                        np.max(arch) / 1000.0,
                        np.min(arch) / 1000.0,
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0] * 8)
        
        # Performance features
        if 'final_performance' in exp_data and exp_data['final_performance']:
            perf = exp_data['final_performance']
            if isinstance(perf, dict):
                features.extend([
                    perf.get('accuracy', 0.0),
                    perf.get('loss', 0.0) / 10.0,  # Normalized loss
                    perf.get('precision', 0.0),
                    perf.get('recall', 0.0),
                    perf.get('f1_score', 0.0),
                ])
            else:
                features.extend([0.0] * 5)
        elif 'metrics' in exp_data:
            metrics = exp_data['metrics']
            features.extend([
                metrics.get('accuracy', 0.0),
                metrics.get('loss', 0.0) / 10.0,
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('f1_score', 0.0),
            ])
        else:
            features.extend([0.0] * 5)
        
        # Training dynamics
        if 'training_history' in exp_data:
            history = exp_data['training_history']
            if history:
                # Convergence speed
                accuracies = [h.get('val_accuracy', h.get('test_accuracy', 0)) for h in history]
                if accuracies:
                    final_acc = accuracies[-1]
                    # Epochs to reach 90% of final accuracy
                    target = final_acc * 0.9
                    convergence_epoch = next((i for i, acc in enumerate(accuracies) if acc >= target), len(accuracies))
                    features.append(convergence_epoch / len(history))
                    
                    # Stability (variance in later epochs)
                    if len(accuracies) > 5:
                        stability = 1.0 - np.std(accuracies[-5:])
                    else:
                        stability = 0.5
                    features.append(stability)
                else:
                    features.extend([0.5, 0.5])
            else:
                features.extend([0.5, 0.5])
        else:
            features.extend([0.5, 0.5])
        
        # Growth features (if applicable)
        if 'growth_history' in exp_data:
            growth = exp_data['growth_history']
            features.extend([
                len(growth) / 10.0,  # Number of growth events
                sum(1 for g in growth if g.get('growth_occurred', False)) / max(len(growth), 1),
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Dataset features
        dataset = exp_data.get('config', {}).get('dataset', 'unknown')
        dataset_encoding = {
            'mnist': [1, 0, 0, 0],
            'cifar10': [0, 1, 0, 0],
            'cifar100': [0, 0, 1, 0],
            'imagenet': [0, 0, 0, 1],
        }
        features.extend(dataset_encoding.get(dataset, [0, 0, 0, 0]))
        
        # Convert to numpy array
        embedding = np.array(features, dtype=np.float32)
        
        # Pad or truncate to embedding_dim
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


class ArchitectureEmbedder:
    """Converts network architectures into vector embeddings."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
    
    def embed(self, architecture: Union[List[int], NetworkArchitecture, Dict[str, Any]]) -> np.ndarray:
        """
        Create an embedding for a network architecture.
        
        The embedding captures:
        - Layer sizes and patterns
        - Depth and width characteristics
        - Bottlenecks and expansions
        - Overall shape
        """
        features = []
        
        # Extract layer sizes
        if isinstance(architecture, list):
            layers = architecture
        elif isinstance(architecture, dict):
            layers = architecture.get('layers', [])
        else:
            layers = architecture.layers
        
        if not layers:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Basic statistics
        features.extend([
            len(layers) / 20.0,  # Normalized depth
            np.mean(layers) / 1000.0,
            np.std(layers) / 1000.0,
            np.max(layers) / 1000.0,
            np.min(layers) / 1000.0,
            layers[0] / 10000.0,  # Input size (normalized)
            layers[-1] / 1000.0,  # Output size (normalized)
        ])
        
        # Layer size ratios (capturing shape)
        if len(layers) > 1:
            ratios = [layers[i+1] / layers[i] for i in range(len(layers)-1)]
            features.extend([
                np.mean(ratios),
                np.std(ratios),
                np.min(ratios),
                np.max(ratios),
            ])
            
            # Bottleneck detection
            min_layer = min(layers[1:-1]) if len(layers) > 2 else layers[0]
            bottleneck_ratio = min_layer / layers[0]
            features.append(bottleneck_ratio)
            
            # Expansion detection
            max_layer = max(layers[1:-1]) if len(layers) > 2 else layers[0]
            expansion_ratio = max_layer / layers[0]
            features.append(expansion_ratio)
        else:
            features.extend([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        
        # Shape classification
        if len(layers) > 2:
            # Pyramid shape (decreasing)
            is_pyramid = all(layers[i] >= layers[i+1] for i in range(len(layers)-1))
            features.append(float(is_pyramid))
            
            # Hourglass shape (decrease then increase)
            mid = len(layers) // 2
            is_hourglass = (
                all(layers[i] >= layers[i+1] for i in range(mid)) and
                all(layers[i] <= layers[i+1] for i in range(mid, len(layers)-1))
            )
            features.append(float(is_hourglass))
            
            # Expanding shape (increasing)
            is_expanding = all(layers[i] <= layers[i+1] for i in range(len(layers)-1))
            features.append(float(is_expanding))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Convert to numpy array
        embedding = np.array(features, dtype=np.float32)
        
        # Pad or truncate
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


def embed_experiment(experiment: Union[Dict[str, Any], ExperimentSchema]) -> np.ndarray:
    """Convenience function to embed an experiment."""
    embedder = ExperimentEmbedder()
    return embedder.embed(experiment)


def embed_architecture(architecture: Union[List[int], NetworkArchitecture, Dict[str, Any]]) -> np.ndarray:
    """Convenience function to embed an architecture."""
    embedder = ArchitectureEmbedder()
    return embedder.embed(architecture)


def embed_hypothesis(hypothesis: Hypothesis) -> np.ndarray:
    """
    Create an embedding for a hypothesis based on its description and parameters.
    
    This is a simple embedding based on hashing the text content.
    In production, you might want to use a proper text embedding model.
    """
    # Combine relevant text fields
    text = f"{hypothesis.name} {hypothesis.description} {hypothesis.question} {hypothesis.prediction}"
    
    # Create a hash-based embedding
    hash_object = hashlib.sha256(text.encode())
    hash_bytes = hash_object.digest()
    
    # Convert to float array
    embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32) / 255.0
    
    # Pad or truncate to standard size
    embedding_dim = 128
    if len(embedding) < embedding_dim:
        embedding = np.pad(embedding, (0, embedding_dim - len(embedding)))
    else:
        embedding = embedding[:embedding_dim]
    
    return embedding