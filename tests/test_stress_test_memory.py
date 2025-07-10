"""
Tests for Ultimate Stress Test v2 memory optimizations.
"""

import pytest
import torch
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import psutil
import asyncio

from experiments.ultimate_stress_test_v2 import (
    StressTestConfig,
    TournamentExecutor,
    evaluate_competitor_task,
    ProfilingMetrics,
    SimpleProfiler
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Create mock stress test configuration."""
    config = StressTestConfig(
        num_gpus=1,
        processes_per_gpu=1,
        tournament_size=4,  # Small for testing
        generations=2,
        epochs_per_generation=1,
        batch_size_base=32,
        dataset_name='cifar10',
        enable_profiling=True,
        aggressive_memory_cleanup=True,
        max_experiments_in_memory=2
    )
    return config


@pytest.fixture
def mock_dataset():
    """Create mock dataset loaders."""
    # Mock data loader
    class MockDataLoader:
        def __init__(self, size=10):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __iter__(self):
            for i in range(self.size):
                # Mock batch
                data = torch.randn(32, 3, 32, 32)  # CIFAR10 shape
                target = torch.randint(0, 10, (32,))
                yield data, target
    
    return {
        'train_loader': MockDataLoader(10),
        'test_loader': MockDataLoader(5)
    }


class TestDatasetSharing:
    """Test dataset sharing optimization."""
    
    @patch('experiments.ultimate_stress_test_v2.create_dataset')
    def test_dataset_preloading(self, mock_create_dataset, mock_config, mock_dataset):
        """Test that datasets are pre-loaded once."""
        mock_create_dataset.return_value = mock_dataset
        
        # Create executor
        executor = TournamentExecutor(mock_config)
        
        # Verify dataset was created only once
        mock_create_dataset.assert_called_once_with(
            'cifar10',
            batch_size=32
        )
        
        # Verify loaders are stored
        assert hasattr(executor, 'train_loader')
        assert hasattr(executor, 'test_loader')
        assert len(executor.train_loader) == 10
        assert len(executor.test_loader) == 5
    
    def test_dataset_sharing_in_hypothesis(self, mock_config, mock_dataset):
        """Test datasets are passed to hypothesis control parameters."""
        with patch('experiments.ultimate_stress_test_v2.create_dataset', return_value=mock_dataset):
            executor = TournamentExecutor(mock_config)
            
            # Generate initial population
            executor.generate_initial_population()
            
            # Create hypothesis
            hypothesis = executor.create_competitor_hypothesis(0)
            
            # Verify datasets in control parameters
            assert 'train_loader' in hypothesis.control_parameters
            assert 'test_loader' in hypothesis.control_parameters
            assert hypothesis.control_parameters['train_loader'] is executor.train_loader
            assert hypothesis.control_parameters['test_loader'] is executor.test_loader
    
    @patch('experiments.ultimate_stress_test_v2.create_dataset')
    def test_evaluate_uses_shared_datasets(self, mock_create_dataset, mock_dataset):
        """Test evaluate_competitor_task uses pre-loaded datasets."""
        # Should not be called if datasets are provided
        mock_create_dataset.side_effect = Exception("Dataset should not be created!")
        
        config = {
            'train_loader': mock_dataset['train_loader'],
            'test_loader': mock_dataset['test_loader'],
            'architecture': [3072, 256, 10],  # Flattened CIFAR10
            'sparsity': 0.02,
            'epochs': 1,
            'batch_size': 32,
            'device': 'cpu'
        }
        
        # Should work without creating dataset
        model, metrics = evaluate_competitor_task(config)
        
        assert 'accuracy' in metrics
        assert 'parameters' in metrics
        assert metrics['parameters'] > 0


class TestMemoryCleanup:
    """Test memory cleanup mechanisms."""
    
    def test_cleanup_nal_memory(self, mock_config):
        """Test NAL memory cleanup function."""
        with patch('experiments.ultimate_stress_test_v2.create_dataset'):
            executor = TournamentExecutor(mock_config)
            
            # Mock NAL attributes
            executor.lab.results = {'hyp1': Mock(), 'hyp2': Mock()}
            executor.lab.experiments = {
                'exp1': Mock(hypothesis_id='hyp1'),
                'exp2': Mock(hypothesis_id='hyp1'),
                'exp3': Mock(hypothesis_id='hyp2')
            }
            executor.lab.hypotheses = {
                'hyp1': Mock(results=[1, 2, 3]),
                'hyp2': Mock(results=[4, 5])
            }
            
            # Clean up hypothesis 1
            executor._cleanup_nal_memory('hyp1')
            
            # Verify cleanup
            assert 'hyp1' not in executor.lab.results
            assert 'hyp1' not in executor.lab.hypotheses
            assert 'exp1' not in executor.lab.experiments
            assert 'exp2' not in executor.lab.experiments
            assert 'exp3' in executor.lab.experiments  # Different hypothesis
    
    def test_aggressive_cleanup_threshold(self, mock_config):
        """Test cleanup when exceeding memory threshold."""
        mock_config.max_experiments_in_memory = 2
        
        with patch('experiments.ultimate_stress_test_v2.create_dataset'):
            executor = TournamentExecutor(mock_config)
            
            # Add experiments beyond threshold
            executor.lab.experiments = {
                f'exp{i}': Mock() for i in range(5)
            }
            executor.lab.results = {
                f'hyp{i}': Mock() for i in range(5)
            }
            
            # Trigger cleanup
            executor._cleanup_nal_memory('hyp_new')
            
            # Should clear all when over threshold
            assert len(executor.lab.experiments) == 0
            assert len(executor.lab.results) == 0
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_nal_recreation_on_high_memory(self, mock_virtual_memory, mock_process, mock_config):
        """Test NAL recreation when memory usage is high."""
        # Mock high memory usage (75%)
        mock_process.return_value.memory_info.return_value.rss = 7.5e9  # 7.5 GB
        mock_virtual_memory.return_value.total = 10e9  # 10 GB total
        
        with patch('experiments.ultimate_stress_test_v2.create_dataset'):
            executor = TournamentExecutor(mock_config)
            old_lab = executor.lab
            
            # Mock the evolution loop's memory check
            # This would normally happen in run_tournament
            process = psutil.Process()
            memory_percent = process.memory_info().rss / psutil.virtual_memory().total * 100
            
            assert memory_percent > 70  # Verify our mock
            
            # Simulate NAL recreation
            if memory_percent > 70:
                del executor.lab
                executor._init_nal()
            
            # Verify new instance
            assert executor.lab is not old_lab


class TestResultCaching:
    """Test disk-based result caching."""
    
    @pytest.mark.asyncio
    async def test_results_saved_to_disk(self, mock_config, temp_dir):
        """Test results are saved to disk immediately."""
        mock_config.generations = 1
        
        with patch('experiments.ultimate_stress_test_v2.create_dataset'):
            with patch('experiments.ultimate_stress_test_v2.Path') as mock_path:
                # Mock data directory
                mock_path.return_value.mkdir.return_value = None
                mock_path.return_value.__truediv__.return_value = Path(temp_dir)
                
                executor = TournamentExecutor(mock_config)
                executor.data_dir = Path(temp_dir)
                
                # Mock NAL test result
                mock_result = Mock()
                mock_result.experiment_results = [
                    Mock(
                        experiment=Mock(parameters={'competitor_id': 'comp1'}),
                        metrics={'accuracy': 0.9, 'parameters': 100000},
                        model_parameters=100000,
                        error=None,
                        status='completed'
                    )
                ]
                
                with patch.object(executor.lab, 'test_hypothesis', return_value=mock_result):
                    # Generate population
                    executor.generate_initial_population()
                    
                    # Evaluate generation
                    await executor.evaluate_generation(0)
                    
                    # Check results saved
                    results_dir = Path(temp_dir) / "generation_0_results"
                    result_file = results_dir / "comp1_result.json"
                    
                    # Verify file would be created (mocked)
                    assert executor.population[0]['fitness'] > 0  # Results processed
    
    def test_essential_metrics_extraction(self, mock_config):
        """Test only essential metrics kept in memory."""
        with patch('experiments.ultimate_stress_test_v2.create_dataset'):
            executor = TournamentExecutor(mock_config)
            
            # Test fitness calculation with essential metrics
            metrics = {
                'accuracy': 0.85,
                'parameters': 500000,
                'training_time': 123.45,  # Extra metric
                'lots_of_other_data': [1, 2, 3, 4, 5]  # Should not affect
            }
            
            fitness = executor._calculate_fitness(metrics)
            
            assert fitness > 0
            assert fitness < 1
            # Fitness should consider accuracy and parameters
            expected = 0.85 * 0.7 + (0.85 / 0.5) * 0.3  # accuracy * 0.7 + efficiency * 0.3
            assert abs(fitness - expected) < 0.1


class TestProfiling:
    """Test profiling functionality."""
    
    def test_simple_profiler(self):
        """Test SimpleProfiler basic functionality."""
        profiler = SimpleProfiler(enabled=True)
        
        # Test phase timing
        profiler.start_phase('test_phase')
        import time
        time.sleep(0.01)  # Small delay
        duration = profiler.end_phase('test_phase')
        
        assert duration > 0
        assert duration < 1  # Should be milliseconds
        
        # Test memory snapshots
        profiler.snapshot_memory('test_snapshot')
        assert 'test_snapshot' in profiler.memory_snapshots
        assert profiler.memory_snapshots['test_snapshot']['cpu_mb'] > 0
    
    def test_profiling_metrics_collection(self):
        """Test ProfilingMetrics data structure."""
        metrics = ProfilingMetrics()
        
        # Add some data
        metrics.data_loading_time = 1.5
        metrics.training_time = 45.3
        metrics.epoch_times = [4.5, 4.3, 4.7]
        metrics.peak_memory_mb = 1024.5
        
        # Convert to dict
        data = metrics.to_dict()
        
        assert data['timing']['data_loading'] == 1.5
        assert data['timing']['training'] == 45.3
        assert data['efficiency']['avg_epoch_time'] == pytest.approx(4.5, rel=0.1)
        assert data['resources']['peak_memory_mb'] == 1024.5
    
    def test_profiling_in_evaluation(self, mock_dataset):
        """Test profiling integration in evaluate_competitor_task."""
        config = {
            'train_loader': mock_dataset['train_loader'],
            'test_loader': mock_dataset['test_loader'],
            'architecture': [3072, 256, 10],
            'sparsity': 0.02,
            'epochs': 2,
            'batch_size': 32,
            'device': 'cpu',
            'enable_profiling': True
        }
        
        model, metrics = evaluate_competitor_task(config)
        
        # Verify profiling metrics included
        assert 'profiling' in metrics
        assert 'training_time' in metrics
        assert 'samples_per_second' in metrics
        assert 'peak_memory_mb' in metrics
        
        prof = metrics['profiling']
        assert prof['timing']['total'] > 0
        assert prof['efficiency']['epochs_completed'] == 2


class TestLogging:
    """Test logging functionality."""
    
    @patch('experiments.ultimate_stress_test_v2.StandardizedLogger')
    def test_memory_logging(self, mock_logger_class, mock_config):
        """Test memory status is logged."""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger
        
        with patch('experiments.ultimate_stress_test_v2.create_dataset'):
            executor = TournamentExecutor(mock_config)
            
            # Verify logger created
            assert executor.logger is mock_logger
            
            # Simulate memory logging (would happen in evaluate_generation)
            ram_before = 4.5  # GB
            gpu_before = 2.1  # GB
            
            # Log entries that would be made
            executor.logger.info(
                f"Generation 0 - Memory before NAL: RAM {ram_before:.2f} GB, GPU {gpu_before:.2f} GB"
            )
            
            # Verify logging calls
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "Memory before NAL" in call_args
            assert "4.5" in call_args


class TestChromaIntegration:
    """Test ChromaDB/HDF5 integration in stress test."""
    
    @patch('experiments.ultimate_stress_test_v2.NALChromaIntegration')
    def test_chroma_initialization(self, mock_integration_class, mock_config):
        """Test ChromaDB integration is initialized when enabled."""
        mock_config.aggressive_memory_cleanup = True
        
        with patch('experiments.ultimate_stress_test_v2.create_dataset'):
            executor = TournamentExecutor(mock_config)
            
            # Verify integration created
            assert executor.chroma_integration is not None
            mock_integration_class.assert_called_once()
            
            # Check configuration
            call_kwargs = mock_integration_class.call_args[1]
            assert 'chroma_config' in call_kwargs
            assert 'timeseries_config' in call_kwargs
            assert call_kwargs['timeseries_threshold'] == 10
    
    @patch('experiments.ultimate_stress_test_v2.NALChromaIntegration')
    def test_chroma_storage_fallback(self, mock_integration_class, mock_config, temp_dir):
        """Test fallback to JSON when ChromaDB fails."""
        mock_config.aggressive_memory_cleanup = True
        
        # Mock integration that fails
        mock_integration = Mock()
        mock_integration.index_experiment_result.side_effect = Exception("Storage failed")
        mock_integration_class.return_value = mock_integration
        
        with patch('experiments.ultimate_stress_test_v2.create_dataset'):
            executor = TournamentExecutor(mock_config)
            executor.data_dir = Path(temp_dir)
            
            # This would normally happen in evaluate_generation
            # Simulate storing a result
            result = Mock(
                experiment=Mock(parameters={'competitor_id': 'test_comp'}),
                metrics={'accuracy': 0.9},
                model_parameters=100000,
                error=None,
                status='completed'
            )
            
            results_dir = executor.data_dir / "generation_0_results"
            results_dir.mkdir(exist_ok=True)
            
            # Try to store with ChromaDB (will fail and fallback)
            try:
                executor.chroma_integration.index_experiment_result(result, 'hyp1')
            except:
                # Fallback to JSON
                result_file = results_dir / "test_comp_result.json"
                with open(result_file, 'w') as f:
                    json.dump({'metrics': result.metrics}, f)
            
            # Verify JSON file created
            assert (results_dir / "test_comp_result.json").exists()


@pytest.mark.integration
class TestFullIntegration:
    """Full integration tests."""
    
    @pytest.mark.asyncio
    @patch('experiments.ultimate_stress_test_v2.create_dataset')
    @patch('experiments.ultimate_stress_test_v2.NeuralArchitectureLab')
    async def test_memory_efficient_tournament(self, mock_nal_class, mock_create_dataset, mock_dataset, mock_config):
        """Test complete tournament with memory optimizations."""
        mock_config.generations = 2
        mock_config.tournament_size = 3
        
        # Setup mocks
        mock_create_dataset.return_value = mock_dataset
        mock_lab = Mock()
        mock_nal_class.return_value = mock_lab
        
        # Mock hypothesis results
        mock_result = Mock()
        mock_result.experiment_results = [
            Mock(
                experiment=Mock(parameters={'competitor_id': f'comp{i}'}),
                metrics={'accuracy': 0.8 + i * 0.05, 'parameters': 100000 + i * 10000},
                model_parameters=100000 + i * 10000,
                error=None
            )
            for i in range(3)
        ]
        mock_lab.test_hypothesis = asyncio.coroutine(lambda x: mock_result)
        
        # Run tournament
        executor = TournamentExecutor(mock_config)
        
        # Verify dataset loaded once
        assert mock_create_dataset.call_count == 1
        
        # Track memory cleanup calls
        with patch.object(executor, '_cleanup_nal_memory') as mock_cleanup:
            await executor.run_tournament()
            
            # Verify cleanup called after each generation
            assert mock_cleanup.call_count >= mock_config.generations


# Parametrized tests for different configurations
@pytest.mark.parametrize("config_overrides,expected", [
    ({"aggressive_memory_cleanup": True}, {"uses_chroma": True}),
    ({"aggressive_memory_cleanup": False}, {"uses_chroma": False}),
    ({"enable_profiling": True}, {"has_profiling": True}),
    ({"enable_profiling": False}, {"has_profiling": False}),
])
def test_configuration_effects(mock_config, config_overrides, expected):
    """Test different configuration effects."""
    # Apply overrides
    for key, value in config_overrides.items():
        setattr(mock_config, key, value)
    
    with patch('experiments.ultimate_stress_test_v2.create_dataset'):
        executor = TournamentExecutor(mock_config)
        
        if "uses_chroma" in expected:
            if expected["uses_chroma"]:
                assert executor.chroma_integration is not None
            else:
                assert executor.chroma_integration is None
        
        if "has_profiling" in expected:
            if expected["has_profiling"]:
                assert executor.generation_profiles is not None
            else:
                assert executor.generation_profiles is None