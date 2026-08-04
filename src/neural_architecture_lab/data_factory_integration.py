#!/usr/bin/env python3
"""
NAL-ChromaDB Integration

Provides integration between Neural Architecture Lab and ChromaDB for
offloading experiment tracking and enabling semantic search.
"""

from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import numpy as np

from structure_net.data_factory.search import (
    ExperimentSearcher,
    ChromaConfig
)
from structure_net.data_factory.time_series_storage import (
    TimeSeriesConfig,
    HybridExperimentStorage
)
from neural_architecture_lab.core import (
    ExperimentResult,
    Hypothesis,
    HypothesisResult
)

if TYPE_CHECKING:
    from neural_architecture_lab.core import LabConfig


class NALChromaIntegration:
    """
    Integrates NAL with ChromaDB and time series storage for persistent experiment storage.
    
    This allows NAL to offload experiment data immediately:
    - Metadata and searchable info -> ChromaDB
    - Large time series data -> Efficient time series storage
    """
    
    def __init__(
        self,
        nal_config: Optional['LabConfig'] = None,
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
        if isinstance(nal_config, ChromaConfig):
            if isinstance(chroma_config, TimeSeriesConfig) and timeseries_config is None:
                timeseries_config = chroma_config
            chroma_config = nal_config
            nal_config = None

        self.nal_config = nal_config
        self.hybrid_storage = HybridExperimentStorage(
            chroma_config,
            timeseries_config,
            timeseries_threshold=timeseries_threshold,
        )
        self.searcher = self.hybrid_storage.searcher
        self.timeseries = self.hybrid_storage.timeseries
        self.auto_index = auto_index
        self.batch_size = batch_size
        self.timeseries_threshold = timeseries_threshold
        self.pending_experiments = []
        
    def index_experiment_result(
        self,
        result: ExperimentResult,
        hypothesis_id: Optional[str] = None,
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
        hypothesis_id = hypothesis_id or result.hypothesis_id
        exp_id = getattr(result, 'experiment_id', None)
        if exp_id is None:
            exp_id = result.experiment.id
        
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
        
        for result in results:
            exp_ids.append(self.index_experiment_result(result, hypothesis.id))
        
        # Clear from NAL if requested
        if clear_from_nal and hasattr(hypothesis, 'results'):
            hypothesis.results.clear()
        
        return exp_ids
    
    def _convert_nal_result(
        self,
        result: ExperimentResult,
        hypothesis_id: str
    ) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
        """Convert NAL ExperimentResult to storage format."""
        metrics = result.metrics or {}
        legacy_experiment = getattr(result, 'experiment', None)
        experiment_id = getattr(result, 'experiment_id', None)
        if experiment_id is None and legacy_experiment is not None:
            experiment_id = legacy_experiment.id

        result_hypothesis_id = getattr(result, 'hypothesis_id', hypothesis_id)
        architecture = getattr(result, 'model_architecture', None)
        if architecture is None and legacy_experiment is not None:
            architecture = legacy_experiment.parameters.get('architecture', [])

        parameters = getattr(legacy_experiment, 'parameters', {}) if legacy_experiment else {}
        status = getattr(result, 'status', None)
        if status is None:
            status = 'failed' if result.error else 'completed'
        elif hasattr(status, 'value'):
            status = status.value

        experiment_data = {
            'schema_version': 1,
            'experiment_id': experiment_id,
            'hypothesis_id': result_hypothesis_id,
            'experiment_type': getattr(legacy_experiment, 'type', 'nal'),
            'status': str(status),
            'timestamp': getattr(result, 'timestamp', None),
            'training_time': getattr(result, 'training_time', getattr(result, 'duration', 0.0)),
            'error': result.error,
            'metrics': metrics,
            'primary_metric': getattr(result, 'primary_metric', 0.0),
            'model_parameters': result.model_parameters,
            'architecture': architecture or [],
            'config': parameters,
            'model_checkpoint': getattr(result, 'model_checkpoint', None),
            'observations': getattr(result, 'observations', []),
            'anomalies': getattr(result, 'anomalies', []),
            'final_performance': {
                'accuracy': metrics.get('accuracy', 0.0),
                'loss': metrics.get('loss', 0.0),
            }
        }
        
        # Handle training history based on size
        training_history = None
        if result.training_history:
            if len(result.training_history) >= self.timeseries_threshold:
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
    
    def __init__(
        self,
        nal_or_config,
        chroma_integration: Optional[NALChromaIntegration] = None,
        chroma_config: Optional[ChromaConfig] = None,
        timeseries_config: Optional[TimeSeriesConfig] = None,
    ):
        """
        Wrap a NAL instance with ChromaDB integration.
        
        Args:
            nal_instance: The NeuralArchitectureLab instance
            chroma_integration: ChromaDB integration instance
        """
        if isinstance(chroma_integration, ChromaConfig):
            if isinstance(chroma_config, TimeSeriesConfig) and timeseries_config is None:
                timeseries_config = chroma_config
            chroma_config = chroma_integration
            chroma_integration = None

        if hasattr(nal_or_config, 'test_hypothesis'):
            self.nal = nal_or_config
        else:
            from .lab import NeuralArchitectureLab
            self.nal = NeuralArchitectureLab(nal_or_config)

        self.integration = chroma_integration or NALChromaIntegration(
            getattr(self.nal, 'config', None),
            chroma_config=chroma_config,
            timeseries_config=timeseries_config,
        )
        self.chroma = self.integration
        self._original_test_hypothesis = self.nal.test_hypothesis
        
        # Monkey-patch the test_hypothesis method
        self.nal.test_hypothesis = self._wrapped_test_hypothesis
    
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
                    exp_ids = [r.experiment_id for r in result.experiment_results]
                    for exp_id in exp_ids:
                        if exp_id in self.nal.experiments:
                            del self.nal.experiments[exp_id]
        
        return result


def create_memory_efficient_nal(
    nal_config: 'LabConfig',
    chroma_config: Optional[ChromaConfig] = None,
    timeseries_config: Optional[TimeSeriesConfig] = None,
):
    """
    Create a memory-efficient NAL instance with ChromaDB integration.
    
    Args:
        nal_config: The master NAL configuration object.
        
    Returns:
        Wrapped NAL instance that automatically offloads to ChromaDB.
    """
    from .lab import NeuralArchitectureLab
    
    run_dir = Path(nal_config.results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    nal = NeuralArchitectureLab(nal_config)
    if chroma_config is None:
        chroma_config = ChromaConfig(persist_directory=str(run_dir / "chroma_db"))
    if timeseries_config is None:
        timeseries_config = TimeSeriesConfig(storage_dir=str(run_dir / "timeseries_db"))

    # Create ChromaDB integration
    chroma_integration = NALChromaIntegration(
        nal_config,
        chroma_config=chroma_config,
        timeseries_config=timeseries_config,
    )
    
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
