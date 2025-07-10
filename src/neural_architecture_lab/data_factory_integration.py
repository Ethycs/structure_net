#!/usr/bin/env python3
"""
NAL-ChromaDB Integration

Provides integration between Neural Architecture Lab and ChromaDB for
offloading experiment tracking and enabling semantic search.
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
from datetime import datetime

from data_factory.search import (
    ExperimentSearcher,
    get_chroma_client,
    ChromaConfig
)
from data_factory.time_series_storage import (
    TimeSeriesStorage,
    TimeSeriesConfig,
    HybridExperimentStorage
)
from neural_architecture_lab.core import (
    ExperimentResult,
    Hypothesis,
    HypothesisResult
)


class NALChromaIntegration:
    """
    Integrates NAL with ChromaDB and time series storage for persistent experiment storage.
    
    This allows NAL to offload experiment data immediately:
    - Metadata and searchable info -> ChromaDB
    - Large time series data -> Efficient time series storage
    """
    
    def __init__(
        self,
        chroma_config: Optional[ChromaConfig] = None,
        timeseries_config: Optional[TimeSeriesConfig] = None,
        auto_index: bool = True,
        batch_size: int = 100,
        timeseries_threshold: int = 50  # Store histories > 50 epochs externally
    ):
        """
        Initialize NAL-ChromaDB integration with time series support.
        
        Args:
            chroma_config: ChromaDB configuration
            timeseries_config: Time series storage configuration
            auto_index: Automatically index experiments as they complete
            batch_size: Batch size for bulk indexing
            timeseries_threshold: Minimum epochs to use time series storage
        """
        self.hybrid_storage = HybridExperimentStorage(chroma_config, timeseries_config)
        self.searcher = self.hybrid_storage.searcher
        self.timeseries = self.hybrid_storage.timeseries
        self.auto_index = auto_index
        self.batch_size = batch_size
        self.timeseries_threshold = timeseries_threshold
        self.pending_experiments = []
        
    def index_experiment_result(
        self,
        result: ExperimentResult,
        hypothesis_id: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Index a single NAL experiment result using hybrid storage.
        
        Args:
            result: NAL ExperimentResult object
            hypothesis_id: ID of the hypothesis this experiment belongs to
            additional_metadata: Extra metadata to store
            
        Returns:
            Experiment ID used in storage
        """
        # Generate unique experiment ID
        exp_id = f"{hypothesis_id}_{result.experiment.id}_{int(time.time())}"
        
        # Convert NAL result to storage format
        experiment_data, training_history = self._convert_nal_result(result, hypothesis_id)
        
        # Add additional metadata if provided
        if additional_metadata:
            experiment_data.update(additional_metadata)
        
        # Use hybrid storage
        self.hybrid_storage.store_experiment(
            experiment_id=exp_id,
            experiment_data=experiment_data,
            training_history=training_history
        )
        
        return exp_id
    
    def index_hypothesis_results(
        self,
        hypothesis: Hypothesis,
        results: List[ExperimentResult],
        clear_from_nal: bool = True
    ) -> List[str]:
        """
        Index all results from a hypothesis to ChromaDB.
        
        Args:
            hypothesis: The hypothesis object
            results: List of experiment results
            clear_from_nal: Whether to clear results from NAL after indexing
            
        Returns:
            List of experiment IDs in ChromaDB
        """
        exp_ids = []
        
        # Batch index for efficiency
        batch = []
        for result in results:
            exp_id = f"{hypothesis.id}_{result.experiment.id}_{int(time.time())}"
            exp_data = self._convert_nal_result(result, hypothesis.id)
            batch.append((exp_id, exp_data))
            
            if len(batch) >= self.batch_size:
                self.searcher.index_experiments_batch(batch)
                exp_ids.extend([b[0] for b in batch])
                batch = []
        
        # Index remaining
        if batch:
            self.searcher.index_experiments_batch(batch)
            exp_ids.extend([b[0] for b in batch])
        
        # Clear from NAL if requested
        if clear_from_nal and hasattr(hypothesis, 'results'):
            hypothesis.results.clear()
        
        return exp_ids
    
    def _convert_nal_result(
        self,
        result: ExperimentResult,
        hypothesis_id: str
    ) -> Dict[str, Any]:
        """Convert NAL ExperimentResult to storage format."""
        # Extract key information from NAL result
        experiment_data = {
            'experiment_id': result.experiment.id,
            'hypothesis_id': hypothesis_id,
            'experiment_type': result.experiment.type,
            'status': result.status.value if hasattr(result.status, 'value') else str(result.status),
            'start_time': result.start_time.isoformat() if result.start_time else None,
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'duration': result.duration,
            'error': result.error,
            
            # Metrics
            'metrics': result.metrics or {},
            'model_parameters': result.model_parameters,
            
            # Architecture info
            'architecture': result.experiment.parameters.get('architecture', []),
            'config': result.experiment.parameters,
            
            # Performance
            'final_performance': {
                'accuracy': result.metrics.get('accuracy', 0.0) if result.metrics else 0.0,
                'loss': result.metrics.get('loss', 0.0) if result.metrics else 0.0,
            }
        }
        
        # Handle training history based on size
        training_history = None
        if hasattr(result, 'training_history') and result.training_history:
            if len(result.training_history) > self.timeseries_threshold:
                # Large history - will be stored separately
                training_history = result.training_history
                # Add summary for ChromaDB
                experiment_data['training_summary'] = {
                    'num_epochs': len(training_history),
                    'final_accuracy': training_history[-1].get('accuracy', 0) if training_history else 0,
                    'best_accuracy': max(h.get('accuracy', 0) for h in training_history) if training_history else 0
                }
            else:
                # Small history - include directly
                experiment_data['training_history'] = result.training_history
        
        return experiment_data, training_history
    
    def create_nal_hooks(self):
        """
        Create hooks for NAL to automatically index experiments.
        
        Returns:
            Dict of hook functions to integrate with NAL
        """
        def on_experiment_complete(result: ExperimentResult, hypothesis_id: str):
            """Hook called when an experiment completes."""
            if self.auto_index:
                self.index_experiment_result(result, hypothesis_id)
        
        def on_hypothesis_complete(hypothesis: Hypothesis, results: List[ExperimentResult]):
            """Hook called when a hypothesis completes."""
            if self.auto_index:
                self.index_hypothesis_results(hypothesis, results, clear_from_nal=True)
        
        return {
            'on_experiment_complete': on_experiment_complete,
            'on_hypothesis_complete': on_hypothesis_complete
        }
    
    def search_nal_experiments(
        self,
        query: Dict[str, Any],
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for NAL experiments using ChromaDB.
        
        Args:
            query: Query parameters
            n_results: Number of results to return
            
        Returns:
            List of matching experiments
        """
        # Convert query to experiment format
        query_exp = {
            'architecture': query.get('architecture', []),
            'final_performance': {
                'accuracy': query.get('target_accuracy', 0.9)
            },
            'config': query.get('config', {})
        }
        
        return self.searcher.search_similar_experiments(
            query_experiment=query_exp,
            n_results=n_results
        )
    
    def get_hypothesis_summary(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Get summary of all experiments for a hypothesis.
        
        Args:
            hypothesis_id: The hypothesis ID
            
        Returns:
            Summary statistics and best experiments
        """
        # Search all experiments for this hypothesis
        results = self.searcher.search_by_hypothesis(
            hypothesis_id=hypothesis_id,
            n_results=1000  # Get all
        )
        
        if not results:
            return {'error': 'No experiments found for hypothesis'}
        
        # Calculate summary statistics
        accuracies = [r['metadata'].get('accuracy', 0) for r in results]
        parameters = [r['metadata'].get('parameters', 0) for r in results]
        
        summary = {
            'hypothesis_id': hypothesis_id,
            'total_experiments': len(results),
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'max_accuracy': np.max(accuracies),
            'min_accuracy': np.min(accuracies),
            'avg_parameters': np.mean(parameters),
            'best_experiment': max(results, key=lambda r: r['metadata'].get('accuracy', 0)),
            'most_efficient': max(
                results,
                key=lambda r: r['metadata'].get('accuracy', 0) / max(r['metadata'].get('parameters', 1), 1)
            )
        }
        
        return summary


class MemoryEfficientNAL:
    """
    Memory-efficient wrapper for NAL that uses ChromaDB for storage.
    
    This wrapper intercepts NAL's result storage and immediately offloads
    to ChromaDB, keeping only minimal data in memory.
    """
    
    def __init__(self, nal_instance, chroma_integration: NALChromaIntegration):
        """
        Wrap a NAL instance with ChromaDB integration.
        
        Args:
            nal_instance: The NeuralArchitectureLab instance
            chroma_integration: ChromaDB integration instance
        """
        self.nal = nal_instance
        self.chroma = chroma_integration
        self._original_test_hypothesis = nal_instance.test_hypothesis
        
        # Monkey-patch the test_hypothesis method
        nal_instance.test_hypothesis = self._wrapped_test_hypothesis
    
    async def _wrapped_test_hypothesis(self, hypothesis_id: str) -> HypothesisResult:
        """
        Wrapped test_hypothesis that offloads results to ChromaDB.
        """
        # Call original method
        result = await self._original_test_hypothesis(hypothesis_id)
        
        # Immediately offload to ChromaDB
        if result and result.experiment_results:
            hypothesis = self.nal.hypotheses.get(hypothesis_id)
            if hypothesis:
                # Index all results
                self.chroma.index_hypothesis_results(
                    hypothesis,
                    result.experiment_results,
                    clear_from_nal=True
                )
                
                # Clear NAL's internal storage
                if hasattr(self.nal, 'results') and hypothesis_id in self.nal.results:
                    # Keep only summary, not full results
                    summary = {
                        'total_experiments': len(result.experiment_results),
                        'successful': sum(1 for r in result.experiment_results if not r.error),
                        'avg_accuracy': np.mean([
                            r.metrics.get('accuracy', 0) 
                            for r in result.experiment_results 
                            if r.metrics and not r.error
                        ])
                    }
                    self.nal.results[hypothesis_id] = summary
                
                # Clear experiments
                if hasattr(self.nal, 'experiments'):
                    exp_ids = [r.experiment.id for r in result.experiment_results]
                    for exp_id in exp_ids:
                        if exp_id in self.nal.experiments:
                            del self.nal.experiments[exp_id]
        
        return result


def create_memory_efficient_nal(nal_config, chroma_config: Optional[ChromaConfig] = None):
    """
    Create a memory-efficient NAL instance with ChromaDB integration.
    
    Args:
        nal_config: NAL configuration
        chroma_config: Optional ChromaDB configuration
        
    Returns:
        Wrapped NAL instance that automatically offloads to ChromaDB
    """
    from neural_architecture_lab.lab import NeuralArchitectureLab
    
    # Create NAL instance
    nal = NeuralArchitectureLab(nal_config)
    
    # Create ChromaDB integration
    chroma_integration = NALChromaIntegration(chroma_config)
    
    # Wrap with memory-efficient layer
    wrapped_nal = MemoryEfficientNAL(nal, chroma_integration)
    
    return wrapped_nal.nal, chroma_integration


# Example usage for stress test
if __name__ == "__main__":
    # Example of how to use in stress test
    from neural_architecture_lab.core import LabConfig
    
    # Configure NAL with minimal memory usage
    nal_config = LabConfig(
        max_parallel_experiments=8,
        save_best_models=False,  # Don't save models
        results_dir="/tmp/nal_results"
    )
    
    # Configure ChromaDB
    chroma_config = ChromaConfig(
        persist_directory="/data/chroma_nal",
        collection_name="nal_experiments"
    )
    
    # Create memory-efficient NAL
    nal, chroma = create_memory_efficient_nal(nal_config, chroma_config)
    
    print("Created memory-efficient NAL with ChromaDB integration")
    print(f"Experiments will be stored in: {chroma_config.persist_directory}")