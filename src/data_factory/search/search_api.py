"""
High-level search API for Structure Net experiments.

Provides convenient functions for searching experiments by various criteria.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import json

from .embedder import ExperimentEmbedder, ArchitectureEmbedder, embed_experiment, embed_architecture
from .chroma_client import ChromaSearchClient, get_chroma_client, ChromaConfig
from structure_net.logging.schemas import ExperimentSchema


logger = logging.getLogger(__name__)


class ExperimentSearcher:
    """
    High-level search interface for Structure Net experiments.
    
    This class provides methods to:
    - Index experiments automatically
    - Search by similarity
    - Search by architecture patterns
    - Search by performance criteria
    - Search by hypothesis
    """
    
    def __init__(self, chroma_config: ChromaConfig = None):
        self.client = get_chroma_client(chroma_config)
        self.experiment_embedder = ExperimentEmbedder()
        self.architecture_embedder = ArchitectureEmbedder()
    
    def index_experiment(
        self,
        experiment_id: str,
        experiment_data: Union[Dict[str, Any], ExperimentSchema],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Index an experiment in ChromaDB.
        
        Args:
            experiment_id: Unique identifier for the experiment
            experiment_data: Experiment data (dict or schema object)
            additional_metadata: Optional additional metadata to store
        """
        # Convert to dict if needed
        if hasattr(experiment_data, 'model_dump'):
            exp_dict = experiment_data.model_dump()
        else:
            exp_dict = experiment_data
        
        # Create embedding
        embedding = self.experiment_embedder.embed(exp_dict)
        
        # Extract metadata
        metadata = self._extract_metadata(exp_dict)
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Create document (searchable text)
        document = self._create_document(exp_dict)
        
        # Add to ChromaDB
        self.client.add_experiment(
            experiment_id=experiment_id,
            embedding=embedding,
            metadata=metadata,
            document=document
        )
    
    def index_experiments_batch(
        self,
        experiments: List[Tuple[str, Union[Dict[str, Any], ExperimentSchema]]]
    ) -> None:
        """
        Index multiple experiments in batch.
        
        Args:
            experiments: List of (experiment_id, experiment_data) tuples
        """
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for exp_id, exp_data in experiments:
            # Convert to dict
            if hasattr(exp_data, 'model_dump'):
                exp_dict = exp_data.model_dump()
            else:
                exp_dict = exp_data
            
            ids.append(exp_id)
            embeddings.append(self.experiment_embedder.embed(exp_dict))
            metadatas.append(self._extract_metadata(exp_dict))
            documents.append(self._create_document(exp_dict))
        
        # Batch add
        self.client.add_experiments_batch(ids, embeddings, metadatas, documents)
    
    def search_similar_experiments(
        self,
        query_experiment: Union[Dict[str, Any], ExperimentSchema],
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find experiments similar to a query experiment.
        
        Args:
            query_experiment: The experiment to find similar ones to
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of similar experiments with metadata and scores
        """
        # Create embedding for query
        if hasattr(query_experiment, 'model_dump'):
            query_dict = query_experiment.model_dump()
        else:
            query_dict = query_experiment
        
        query_embedding = self.experiment_embedder.embed(query_dict)
        
        # Search
        results = self.client.search_similar(
            query_embedding=query_embedding,
            n_results=n_results,
            where=filters
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'])):
            formatted_results.append({
                'id': results['ids'][i],
                'similarity_score': 1.0 - results['distances'][i],  # Convert distance to similarity
                'metadata': results['metadatas'][i],
                'description': results['documents'][i] if results['documents'] else None
            })
        
        return formatted_results
    
    def search_by_architecture(
        self,
        architecture: Union[List[int], Dict[str, Any]],
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find experiments with similar architectures.
        
        Args:
            architecture: Target architecture (layer sizes or architecture dict)
            n_results: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of experiments with similar architectures
        """
        # Create architecture embedding
        arch_embedding = self.architecture_embedder.embed(architecture)
        
        # For architecture search, we need to combine with experiment features
        # Create a synthetic experiment with just the architecture
        synthetic_exp = {
            'architecture': architecture if isinstance(architecture, dict) else {'layers': architecture},
            'final_performance': {'accuracy': 0.5},  # Neutral performance
            'config': {'dataset': 'unknown'}
        }
        
        query_embedding = self.experiment_embedder.embed(synthetic_exp)
        
        # Search
        results = self.client.search_similar(
            query_embedding=query_embedding,
            n_results=n_results,
            where=filters
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'])):
            formatted_results.append({
                'id': results['ids'][i],
                'similarity_score': 1.0 - results['distances'][i],
                'metadata': results['metadatas'][i],
                'architecture': results['metadatas'][i].get('architecture', 'unknown')
            })
        
        return formatted_results
    
    def search_by_performance(
        self,
        min_accuracy: Optional[float] = None,
        max_parameters: Optional[int] = None,
        dataset: Optional[str] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search experiments by performance criteria.
        
        Args:
            min_accuracy: Minimum accuracy threshold
            max_parameters: Maximum number of parameters
            dataset: Dataset name filter
            n_results: Number of results
            
        Returns:
            List of experiments matching criteria
        """
        # Build metadata filter
        where_clause = {}
        
        if min_accuracy is not None:
            where_clause['accuracy'] = {'$gte': min_accuracy}
        
        if max_parameters is not None:
            where_clause['parameters'] = {'$lte': max_parameters}
        
        if dataset is not None:
            where_clause['dataset'] = dataset
        
        # Create a high-performance query embedding
        high_perf_exp = {
            'final_performance': {'accuracy': 0.95},
            'architecture': {'total_parameters': 1e6 if max_parameters is None else max_parameters},
            'config': {'dataset': dataset or 'cifar10'}
        }
        
        query_embedding = self.experiment_embedder.embed(high_perf_exp)
        
        # Search
        results = self.client.search_similar(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        # Format and sort by accuracy
        formatted_results = []
        for i in range(len(results['ids'])):
            result = {
                'id': results['ids'][i],
                'accuracy': float(results['metadatas'][i].get('accuracy', 0)),
                'parameters': int(results['metadatas'][i].get('parameters', 0)),
                'dataset': results['metadatas'][i].get('dataset', 'unknown'),
                'metadata': results['metadatas'][i]
            }
            formatted_results.append(result)
        
        # Sort by accuracy (descending)
        formatted_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return formatted_results
    
    def search_by_hypothesis(
        self,
        hypothesis_id: str,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find all experiments related to a specific hypothesis.
        
        Args:
            hypothesis_id: The hypothesis ID to search for
            n_results: Number of results
            
        Returns:
            List of experiments for this hypothesis
        """
        # Search by hypothesis ID in metadata
        results = self.client.search_similar(
            query_embedding=np.zeros(self.experiment_embedder.embedding_dim),  # Dummy embedding
            n_results=n_results,
            where={'hypothesis_id': hypothesis_id}
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'])):
            formatted_results.append({
                'id': results['ids'][i],
                'hypothesis_id': hypothesis_id,
                'metadata': results['metadatas'][i]
            })
        
        return formatted_results
    
    def get_experiment_count(self) -> int:
        """Get the total number of indexed experiments."""
        return self.client.count()
    
    def _extract_metadata(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Extract searchable metadata from experiment."""
        metadata = {}
        
        # Basic info
        metadata['experiment_id'] = experiment.get('experiment_id', 'unknown')
        metadata['experiment_type'] = experiment.get('experiment_type', 'unknown')
        
        # Performance metrics
        if 'final_performance' in experiment and experiment['final_performance']:
            perf = experiment['final_performance']
            metadata['accuracy'] = float(perf.get('accuracy', 0))
            metadata['loss'] = float(perf.get('loss', 0))
        elif 'metrics' in experiment:
            metadata['accuracy'] = float(experiment['metrics'].get('accuracy', 0))
            metadata['loss'] = float(experiment['metrics'].get('loss', 0))
        
        # Architecture info
        if 'architecture' in experiment:
            arch = experiment['architecture']
            if isinstance(arch, dict):
                metadata['parameters'] = int(arch.get('total_parameters', 0))
                metadata['depth'] = int(arch.get('depth', len(arch.get('layers', []))))
                metadata['architecture'] = str(arch.get('layers', []))
            else:
                metadata['parameters'] = sum(arch)  # Rough estimate
                metadata['depth'] = len(arch)
                metadata['architecture'] = str(arch)
        
        # Dataset and config
        if 'config' in experiment:
            config = experiment['config']
            metadata['dataset'] = config.get('dataset', 'unknown')
            metadata['epochs'] = int(config.get('epochs', 0))
            metadata['batch_size'] = int(config.get('batch_size', 0))
        
        # Growth info
        if 'growth_history' in experiment:
            metadata['growth_events'] = len(experiment['growth_history'])
            metadata['has_growth'] = True
        else:
            metadata['has_growth'] = False
        
        # Hypothesis info
        metadata['hypothesis_id'] = experiment.get('hypothesis_id', 'none')
        
        return metadata
    
    def _create_document(self, experiment: Dict[str, Any]) -> str:
        """Create searchable document text from experiment."""
        parts = []
        
        # Experiment type and ID
        parts.append(f"Experiment {experiment.get('experiment_id', 'unknown')}")
        parts.append(f"Type: {experiment.get('experiment_type', 'unknown')}")
        
        # Architecture description
        if 'architecture' in experiment:
            arch = experiment['architecture']
            if isinstance(arch, dict):
                layers = arch.get('layers', [])
            else:
                layers = arch
            parts.append(f"Architecture: {layers}")
        
        # Performance
        accuracy = 0.0
        if 'final_performance' in experiment and experiment['final_performance']:
            accuracy = experiment['final_performance'].get('accuracy', 0)
        elif 'metrics' in experiment:
            accuracy = experiment['metrics'].get('accuracy', 0)
        parts.append(f"Accuracy: {accuracy:.3f}")
        
        # Dataset
        if 'config' in experiment:
            dataset = experiment['config'].get('dataset', 'unknown')
            parts.append(f"Dataset: {dataset}")
        
        return " | ".join(parts)


# Convenience functions
def search_similar_experiments(
    query_experiment: Union[Dict[str, Any], ExperimentSchema],
    n_results: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Find experiments similar to a query experiment."""
    searcher = ExperimentSearcher()
    return searcher.search_similar_experiments(query_experiment, n_results, filters)


def search_by_architecture(
    architecture: Union[List[int], Dict[str, Any]],
    n_results: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Find experiments with similar architectures."""
    searcher = ExperimentSearcher()
    return searcher.search_by_architecture(architecture, n_results, filters)


def search_by_performance(
    min_accuracy: Optional[float] = None,
    max_parameters: Optional[int] = None,
    dataset: Optional[str] = None,
    n_results: int = 10
) -> List[Dict[str, Any]]:
    """Search experiments by performance criteria."""
    searcher = ExperimentSearcher()
    return searcher.search_by_performance(min_accuracy, max_parameters, dataset, n_results)


def search_by_hypothesis(
    hypothesis_id: str,
    n_results: int = 10
) -> List[Dict[str, Any]]:
    """Find all experiments for a hypothesis."""
    searcher = ExperimentSearcher()
    return searcher.search_by_hypothesis(hypothesis_id, n_results)