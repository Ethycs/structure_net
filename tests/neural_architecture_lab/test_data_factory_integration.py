"""
Tests for Data Factory ChromaDB and HDF5 integration.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json
import h5py

from src.data_factory.search import (
    ChromaConfig, 
    ChromaDBClient,
    ExperimentEmbedder,
    ExperimentSearcher
)
from src.data_factory.time_series_storage import (
    TimeSeriesConfig,
    TimeSeriesStorage,
    HybridExperimentStorage
)
from src.neural_architecture_lab.data_factory_integration import NALChromaIntegration


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def chroma_config(temp_dir):
    """Create ChromaDB configuration for testing."""
    return ChromaConfig(
        persist_directory=str(Path(temp_dir) / "chroma_test"),
        collection_name="test_experiments"
    )


@pytest.fixture
def timeseries_config(temp_dir):
    """Create time series configuration for testing."""
    return TimeSeriesConfig(
        storage_dir=str(Path(temp_dir) / "timeseries_test"),
        use_hdf5=True,
        compression="gzip"
    )


@pytest.fixture
def sample_experiment_data():
    """Create sample experiment data for testing."""
    return {
        'experiment_id': 'test_exp_001',
        'architecture': [784, 256, 128, 10],
        'config': {
            'learning_rate': 0.001,
            'batch_size': 128,
            'epochs': 50
        },
        'final_performance': {
            'accuracy': 0.95,
            'loss': 0.15
        },
        'metrics': {
            'parameters': 234567,
            'training_time': 123.45
        }
    }


@pytest.fixture
def sample_training_history():
    """Create sample training history for testing."""
    return [
        {'epoch': i, 'loss': 2.0 - i * 0.1, 'accuracy': i * 0.02}
        for i in range(50)
    ]


class TestChromaDBClient:
    """Test ChromaDB client functionality."""
    
    def test_client_initialization(self, chroma_config):
        """Test ChromaDB client can be initialized."""
        client = ChromaDBClient(chroma_config)
        assert client.collection is not None
        assert client.collection.name == "test_experiments"
    
    def test_add_and_search_documents(self, chroma_config):
        """Test adding and searching documents in ChromaDB."""
        client = ChromaDBClient(chroma_config)
        
        # Add a document
        client.add_documents(
            ids=["doc1"],
            embeddings=[[0.1] * 384],  # 384-dimensional embedding
            metadatas=[{"type": "test", "value": 42}],
            documents=["Test document"]
        )
        
        # Search for similar documents
        results = client.search(
            query_embeddings=[[0.1] * 384],
            n_results=1
        )
        
        assert len(results['ids'][0]) == 1
        assert results['ids'][0][0] == "doc1"
        assert results['metadatas'][0][0]['value'] == 42
    
    def test_delete_collection(self, chroma_config):
        """Test collection deletion."""
        client = ChromaDBClient(chroma_config)
        client.add_documents(
            ids=["doc1"],
            embeddings=[[0.1] * 384],
            metadatas=[{"test": True}]
        )
        
        # Delete and verify
        client.delete_collection()
        
        # Recreate client should have empty collection
        new_client = ChromaDBClient(chroma_config)
        count = new_client.collection.count()
        assert count == 0


class TestExperimentEmbedder:
    """Test experiment embedding functionality."""
    
    def test_experiment_embedding(self, sample_experiment_data):
        """Test experiment data can be embedded."""
        embedder = ExperimentEmbedder()
        embedding = embedder.embed_experiment(sample_experiment_data)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # Sentence transformer dimension
        assert -1 <= embedding.min() <= embedding.max() <= 1
    
    def test_architecture_embedding(self):
        """Test architecture embedding."""
        embedder = ExperimentEmbedder()
        arch = [784, 256, 128, 10]
        embedding = embedder.embed_architecture(arch)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)


class TestTimeSeriesStorage:
    """Test HDF5-based time series storage."""
    
    def test_storage_initialization(self, timeseries_config):
        """Test time series storage initialization."""
        storage = TimeSeriesStorage(timeseries_config)
        assert storage.storage_dir.exists()
        assert storage.use_hdf5 is True
    
    def test_store_and_load_training_history(self, timeseries_config, sample_training_history):
        """Test storing and loading training history."""
        storage = TimeSeriesStorage(timeseries_config)
        exp_id = "test_exp_001"
        
        # Store history
        file_path = storage.store_training_history(
            exp_id, 
            sample_training_history,
            metadata={'test': True}
        )
        
        assert file_path.exists()
        assert file_path.suffix == '.h5'
        
        # Load history
        loaded_history, loaded_metadata = storage.load_training_history(exp_id)
        
        assert len(loaded_history) == len(sample_training_history)
        assert loaded_history[0]['epoch'] == 0
        assert loaded_history[-1]['epoch'] == 49
        assert loaded_metadata['test'] is True
    
    def test_hdf5_compression(self, timeseries_config):
        """Test HDF5 compression is working."""
        storage = TimeSeriesStorage(timeseries_config)
        exp_id = "compression_test"
        
        # Create large history
        large_history = [
            {'epoch': i, 'values': np.random.randn(1000).tolist()}
            for i in range(100)
        ]
        
        file_path = storage.store_training_history(exp_id, large_history)
        
        # Check compression
        with h5py.File(file_path, 'r') as f:
            # Verify compression is applied
            assert f['history'].compression == 'gzip'
    
    def test_json_fallback_for_small_data(self, timeseries_config):
        """Test JSON storage for small datasets."""
        storage = TimeSeriesStorage(timeseries_config)
        storage.epochs_threshold = 100  # Set high threshold
        
        small_history = [{'epoch': i} for i in range(5)]
        file_path = storage.store_training_history("small_test", small_history)
        
        assert file_path.suffix == '.json.gz'


class TestExperimentSearcher:
    """Test experiment search functionality."""
    
    def test_searcher_initialization(self, chroma_config):
        """Test experiment searcher initialization."""
        searcher = ExperimentSearcher(chroma_config)
        assert searcher.client is not None
        assert searcher.embedder is not None
    
    def test_index_and_search_experiment(self, chroma_config, sample_experiment_data):
        """Test indexing and searching experiments."""
        searcher = ExperimentSearcher(chroma_config)
        
        # Index experiment
        searcher.index_experiment(
            "exp_001",
            sample_experiment_data,
            additional_metadata={'tag': 'test'}
        )
        
        # Search similar
        results = searcher.search_similar_experiments(
            sample_experiment_data,
            n_results=1
        )
        
        assert len(results) == 1
        assert results[0]['experiment_id'] == "exp_001"
        assert results[0]['tag'] == 'test'
    
    def test_search_by_performance(self, chroma_config):
        """Test searching by performance criteria."""
        searcher = ExperimentSearcher(chroma_config)
        
        # Index multiple experiments
        for i in range(5):
            searcher.index_experiment(
                f"exp_{i}",
                {
                    'experiment_id': f'exp_{i}',
                    'architecture': [784, 256, 10],
                    'final_performance': {'accuracy': 0.8 + i * 0.02},
                    'metrics': {'parameters': 100000 + i * 10000}
                }
            )
        
        # Search high accuracy
        results = searcher.search_by_performance(
            min_accuracy=0.85,
            n_results=10
        )
        
        assert len(results) >= 2
        assert all(r['final_performance']['accuracy'] >= 0.85 for r in results)


class TestHybridExperimentStorage:
    """Test hybrid ChromaDB + HDF5 storage."""
    
    def test_hybrid_storage_initialization(self, chroma_config, timeseries_config):
        """Test hybrid storage initialization."""
        storage = HybridExperimentStorage(chroma_config, timeseries_config)
        assert storage.searcher is not None
        assert storage.timeseries_storage is not None
    
    def test_store_experiment_with_large_history(self, chroma_config, timeseries_config):
        """Test storing experiment with large training history."""
        storage = HybridExperimentStorage(chroma_config, timeseries_config)
        storage.timeseries_threshold = 10  # Low threshold for testing
        
        # Create experiment with large history
        exp_data = {
            'experiment_id': 'hybrid_test',
            'architecture': [784, 256, 10],
            'training_history': [{'epoch': i} for i in range(50)],
            'final_performance': {'accuracy': 0.95}
        }
        
        storage.store_experiment(exp_data)
        
        # Verify storage
        # 1. Metadata in ChromaDB
        search_results = storage.search_similar({'architecture': [784, 256, 10]})
        assert len(search_results) > 0
        
        # 2. Training history in HDF5
        history_file = storage.timeseries_storage.storage_dir / "hybrid_test_history.h5"
        assert history_file.exists()
    
    def test_load_complete_experiment(self, chroma_config, timeseries_config):
        """Test loading complete experiment data."""
        storage = HybridExperimentStorage(chroma_config, timeseries_config)
        
        # Store experiment
        exp_data = {
            'experiment_id': 'load_test',
            'architecture': [784, 128, 10],
            'training_history': [{'epoch': i, 'loss': 2.0 - i * 0.1} for i in range(20)],
            'final_performance': {'accuracy': 0.9}
        }
        storage.store_experiment(exp_data)
        
        # Load complete data
        loaded = storage.load_experiment('load_test')
        
        assert loaded is not None
        assert loaded['experiment_id'] == 'load_test'
        assert len(loaded['training_history']) == 20
        assert loaded['final_performance']['accuracy'] == 0.9


class TestNALChromaIntegration:
    """Test NAL integration with ChromaDB/HDF5."""
    
    def test_nal_integration_initialization(self, chroma_config, timeseries_config):
        """Test NAL integration initialization."""
        integration = NALChromaIntegration(chroma_config, timeseries_config)
        assert integration.hybrid_storage is not None
        assert integration.searcher is not None
    
    def test_convert_nal_result(self, chroma_config, timeseries_config):
        """Test converting NAL result format."""
        integration = NALChromaIntegration(chroma_config, timeseries_config)
        
        # Mock NAL result
        class MockExperiment:
            id = "test_exp"
            type = "test"
            parameters = {
                'architecture': [784, 256, 10],
                'learning_rate': 0.001
            }
        
        class MockResult:
            experiment = MockExperiment()
            status = "completed"
            start_time = None
            end_time = None
            duration = 123.45
            error = None
            metrics = {'accuracy': 0.95, 'loss': 0.15}
            model_parameters = 234567
            training_history = [{'epoch': i} for i in range(5)]
        
        # Convert
        exp_data, history = integration._convert_nal_result(
            MockResult(), 
            "test_hypothesis"
        )
        
        assert exp_data['experiment_id'] == "test_exp"
        assert exp_data['architecture'] == [784, 256, 10]
        assert exp_data['final_performance']['accuracy'] == 0.95
        assert history is None  # Small history, included directly
    
    def test_memory_efficient_nal(self, chroma_config, timeseries_config):
        """Test memory-efficient NAL wrapper."""
        from src.neural_architecture_lab.data_factory_integration import MemoryEfficientNAL
        
        # Create wrapper
        nal_config = type('LabConfig', (), {
            'max_parallel_experiments': 4,
            'results_dir': './test_results'
        })()
        
        wrapper = MemoryEfficientNAL(
            nal_config,
            chroma_config,
            timeseries_config
        )
        
        assert wrapper.nal is not None
        assert wrapper.integration is not None


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_full_experiment_lifecycle(self, chroma_config, timeseries_config):
        """Test complete experiment storage and retrieval lifecycle."""
        # Create storage
        storage = HybridExperimentStorage(chroma_config, timeseries_config)
        
        # Store multiple experiments
        exp_ids = []
        for i in range(3):
            exp_data = {
                'experiment_id': f'lifecycle_exp_{i}',
                'architecture': [784, 256 - i * 50, 10],
                'training_history': [
                    {'epoch': j, 'accuracy': j * 0.02 + i * 0.1}
                    for j in range(30)
                ],
                'final_performance': {
                    'accuracy': 0.8 + i * 0.05,
                    'loss': 0.3 - i * 0.05
                },
                'config': {
                    'learning_rate': 0.001 * (i + 1),
                    'batch_size': 128
                }
            }
            storage.store_experiment(exp_data)
            exp_ids.append(exp_data['experiment_id'])
        
        # Search by architecture similarity
        similar = storage.search_similar({
            'architecture': [784, 200, 10]
        }, n_results=2)
        
        assert len(similar) == 2
        assert all('experiment_id' in exp for exp in similar)
        
        # Search by performance
        high_perf = storage.search_by_performance(min_accuracy=0.82)
        assert len(high_perf) >= 1
        assert all(exp['final_performance']['accuracy'] >= 0.82 for exp in high_perf)
        
        # Load complete experiment
        loaded = storage.load_experiment(exp_ids[1])
        assert loaded is not None
        assert len(loaded['training_history']) == 30
        assert loaded['architecture'] == [784, 206, 10]
    
    def test_stress_test_integration(self, chroma_config, timeseries_config, temp_dir):
        """Test integration with stress test scenario."""
        # Simulate stress test storage pattern
        storage = HybridExperimentStorage(chroma_config, timeseries_config)
        
        # Store generation of experiments
        generation_size = 10
        for i in range(generation_size):
            exp_data = {
                'experiment_id': f'competitor_{i}',
                'architecture': [784] + [np.random.randint(32, 512) for _ in range(3)] + [10],
                'training_history': [
                    {'epoch': j, 'loss': np.random.random()}
                    for j in range(20)
                ],
                'final_performance': {
                    'accuracy': np.random.random(),
                    'fitness': np.random.random()
                },
                'generation': 0,
                'parent_ids': ['random']
            }
            storage.store_experiment(exp_data)
        
        # Verify all stored
        all_exps = storage.search_by_metadata({'generation': 0})
        assert len(all_exps) == generation_size
        
        # Test cleanup
        storage.clear_generation(0)
        remaining = storage.search_by_metadata({'generation': 0})
        assert len(remaining) == 0


@pytest.mark.parametrize("use_hdf5,compression", [
    (True, "gzip"),
    (True, "lzf"),
    (True, None),
    (False, None)
])
def test_storage_formats(temp_dir, use_hdf5, compression):
    """Test different storage format configurations."""
    config = TimeSeriesConfig(
        storage_dir=str(Path(temp_dir) / "format_test"),
        use_hdf5=use_hdf5,
        compression=compression
    )
    
    storage = TimeSeriesStorage(config)
    history = [{'epoch': i, 'data': np.random.randn(100).tolist()} for i in range(20)]
    
    file_path = storage.store_training_history("format_test", history)
    
    if use_hdf5:
        assert file_path.suffix == '.h5'
        if compression:
            with h5py.File(file_path, 'r') as f:
                assert f['history'].compression == compression
    else:
        assert file_path.suffix in ['.json', '.json.gz']


def test_error_handling(chroma_config, timeseries_config):
    """Test error handling in storage systems."""
    storage = HybridExperimentStorage(chroma_config, timeseries_config)
    
    # Test storing invalid data
    with pytest.raises(KeyError):
        storage.store_experiment({})  # Missing required fields
    
    # Test loading non-existent experiment
    result = storage.load_experiment("non_existent")
    assert result is None
    
    # Test search with invalid query
    results = storage.search_similar({})  # Empty query
    assert isinstance(results, list)
    assert len(results) == 0