#!/usr/bin/env python3
"""
Test suite for unified configuration migration.

Tests that the new unified configuration system works correctly and maintains
backward compatibility with all existing components.
"""

import pytest
from pathlib import Path


class TestUnifiedConfig:
    """Test the unified configuration system."""
    
    def test_unified_config_import(self):
        """Test that unified config can be imported."""
        from config import UnifiedConfig, get_config, LabConfigShim, LoggingConfigShim
        assert UnifiedConfig is not None
        assert get_config is not None
        assert LabConfigShim is not None
        assert LoggingConfigShim is not None
    
    def test_unified_config_creation(self):
        """Test creating a unified config instance."""
        from config import UnifiedConfig
        
        config = UnifiedConfig()
        assert config is not None
        assert config.storage.data_root.exists()
        assert config.experiment.project_name == "structure_net"
        assert isinstance(config.wandb.enabled, bool)
    
    def test_config_subsections(self):
        """Test that all config subsections are properly initialized."""
        from config import UnifiedConfig
        
        config = UnifiedConfig()
        
        # Test storage config
        assert config.storage.results_dir.exists()
        assert config.storage.cache_dir.exists()
        assert config.storage.experiments_dir.exists()
        
        # Test wandb config
        assert config.wandb.project == "structure_net"
        assert config.wandb.mode in ["online", "offline", "disabled"]
        
        # Test logging config
        assert config.logging.level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert isinstance(config.logging.enable_console, bool)
        
        # Test compute config
        assert isinstance(config.compute.max_parallel_experiments, int)
        assert config.compute.max_parallel_experiments > 0
        
        # Test experiment config
        assert config.experiment.batch_size > 0
        assert config.experiment.learning_rate > 0


class TestLabConfigCompatibility:
    """Test backward compatibility with LabConfig."""
    
    def test_lab_config_from_unified(self):
        """Test creating LabConfig from UnifiedConfig."""
        from config import UnifiedConfig
        
        config = UnifiedConfig()
        lab_config = config.get_lab_config()
        
        assert lab_config.project_name == config.experiment.project_name
        assert lab_config.results_dir == str(config.storage.results_dir)
        assert lab_config.max_parallel_experiments == config.compute.max_parallel_experiments
        assert lab_config.enable_wandb == config.wandb.enabled
    
    def test_lab_config_shim(self):
        """Test LabConfig shim for backward compatibility."""
        from config import LabConfigShim
        
        shim = LabConfigShim(
            project_name="test_project",
            max_parallel_experiments=4,
            device_ids=[0, 1]
        )
        
        assert shim.project_name == "test_project"
        assert shim.max_parallel_experiments == 4
        assert shim.device_ids == [0, 1]
        
        # Test conversion to unified
        unified = shim.to_unified_config()
        assert unified.experiment.project_name == "test_project"
        assert unified.compute.max_parallel_experiments == 4
        assert unified.compute.device_ids == [0, 1]
    
    def test_nal_with_unified_config(self):
        """Test that NAL accepts UnifiedConfig."""
        from src.neural_architecture_lab import NeuralArchitectureLab, LabConfig
        from config import UnifiedConfig
        
        # Test with old LabConfig
        old_config = LabConfig(project_name="test_old_style")
        nal1 = NeuralArchitectureLab(old_config)
        assert nal1.config.project_name == "test_old_style"
        
        # Test with UnifiedConfig
        unified = UnifiedConfig()
        unified.experiment.project_name = "test_unified"
        nal2 = NeuralArchitectureLab(unified)
        assert nal2.config.project_name == "test_unified"
        
        # Test with dict
        nal3 = NeuralArchitectureLab({"project_name": "test_dict"})
        assert nal3.config.project_name == "test_dict"
        
        # Test with no config (uses global)
        nal4 = NeuralArchitectureLab()
        assert nal4.config is not None


class TestLoggingConfigCompatibility:
    """Test backward compatibility with LoggingConfig."""
    
    def test_logging_config_from_unified(self):
        """Test creating LoggingConfig from UnifiedConfig."""
        from config import UnifiedConfig
        
        config = UnifiedConfig()
        logging_config = config.get_logging_config()
        
        assert logging_config.project_name == config.experiment.project_name
        assert logging_config.queue_dir == str(config.storage.queue_dir)
        assert logging_config.enable_wandb == config.wandb.enabled
        assert logging_config.chromadb_path == str(config.storage.chromadb_path)
    
    def test_logging_config_shim(self):
        """Test LoggingConfig shim for backward compatibility."""
        from config import LoggingConfigShim
        
        shim = LoggingConfigShim(
            project_name="test_logging",
            enable_wandb=False,
            queue_dir="test_queue"
        )
        
        assert shim.project_name == "test_logging"
        assert shim.enable_wandb == False
        assert shim.queue_dir == "test_queue"
        
        # Test conversion to unified
        unified = shim.to_unified_config()
        assert unified.experiment.project_name == "test_logging"
        assert unified.wandb.enabled == False
        assert str(unified.storage.queue_dir) == "test_queue"
    
    def test_standardized_logger_with_unified_config(self):
        """Test that StandardizedLogger accepts UnifiedConfig."""
        from src.structure_net.logging import StandardizedLogger, LoggingConfig
        from config import UnifiedConfig
        
        # Test with old LoggingConfig
        old_config = LoggingConfig(project_name="test_old_logging")
        logger1 = StandardizedLogger(old_config)
        assert logger1.config.project_name == "test_old_logging"
        
        # Test with UnifiedConfig
        unified = UnifiedConfig()
        unified.experiment.project_name = "test_unified_logging"
        logger2 = StandardizedLogger(unified)
        assert logger2.config.project_name == "test_unified_logging"
        
        # Test with no config (uses global)
        logger3 = StandardizedLogger()
        assert logger3.config is not None


class TestDataFactoryCompatibility:
    """Test backward compatibility with Data Factory configs."""
    
    def test_chroma_config_from_unified(self):
        """Test creating ChromaConfig from UnifiedConfig."""
        from config import UnifiedConfig
        
        config = UnifiedConfig()
        chroma_config = config.get_chroma_config()
        
        assert chroma_config.persist_directory == str(config.storage.chromadb_path)
        assert chroma_config.collection_name == config.storage.chromadb_collection
    
    def test_timeseries_config_from_unified(self):
        """Test creating TimeSeriesConfig from UnifiedConfig."""
        from config import UnifiedConfig
        
        config = UnifiedConfig()
        ts_config = config.get_timeseries_config()
        
        assert ts_config.storage_dir == str(config.storage.timeseries_path)
        assert ts_config.compression == config.storage.timeseries_compression
        assert ts_config.chunk_size == config.storage.timeseries_chunk_size
    
    def test_chroma_client_with_unified_config(self):
        """Test that ChromaSearchClient accepts UnifiedConfig."""
        from src.data_factory.search.chroma_client import ChromaSearchClient, ChromaConfig
        from config import UnifiedConfig
        
        # Test with old ChromaConfig
        old_config = ChromaConfig(persist_directory="test_chroma")
        client1 = ChromaSearchClient(old_config)
        assert client1.config.persist_directory == "test_chroma"
        
        # Test with UnifiedConfig
        unified = UnifiedConfig()
        client2 = ChromaSearchClient(unified)
        assert client2.config.persist_directory == str(unified.storage.chromadb_path)
        
        # Test with no config (uses global)
        client3 = ChromaSearchClient()
        assert client3.config is not None
    
    def test_timeseries_storage_with_unified_config(self):
        """Test that TimeSeriesStorage accepts UnifiedConfig."""
        from src.data_factory.time_series_storage import TimeSeriesStorage, TimeSeriesConfig
        from config import UnifiedConfig
        
        # Test with old TimeSeriesConfig
        old_config = TimeSeriesConfig(storage_dir="test_ts")
        storage1 = TimeSeriesStorage(old_config)
        assert storage1.config.storage_dir == "test_ts"
        
        # Test with UnifiedConfig
        unified = UnifiedConfig()
        storage2 = TimeSeriesStorage(unified)
        assert storage2.config.storage_dir == str(unified.storage.timeseries_path)
        
        # Test with no config (uses global)
        storage3 = TimeSeriesStorage()
        assert storage3.config is not None


class TestConfigMigration:
    """Test config migration helpers."""
    
    def test_auto_migrate_config(self):
        """Test automatic config migration."""
        from config import auto_migrate_config, UnifiedConfig
        from src.neural_architecture_lab.core import LabConfig
        
        # Test with LabConfig
        lab_config = LabConfig(project_name="test_migrate")
        unified = auto_migrate_config(lab_config)
        assert isinstance(unified, UnifiedConfig)
        assert unified.experiment.project_name == "test_migrate"
        
        # Test with already unified
        unified2 = auto_migrate_config(unified)
        assert unified2 is unified
    
    def test_config_file_operations(self):
        """Test saving and loading config files."""
        from config import UnifiedConfig
        import tempfile
        import yaml
        
        config = UnifiedConfig()
        config.experiment.project_name = "test_save"
        config.compute.max_parallel_experiments = 16
        
        # Test YAML save/load
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
            config.save(f.name)
            
            # Load and verify
            loaded = UnifiedConfig.from_file(f.name)
            assert loaded.experiment.project_name == "test_save"
            assert loaded.compute.max_parallel_experiments == 16
            
            # Clean up
            Path(f.name).unlink()
    
    def test_environment_variable_loading(self):
        """Test that config loads from environment variables."""
        import os
        from config import UnifiedConfig, reset_config
        
        # Set some env vars
        os.environ['WANDB_PROJECT'] = 'test_env_project'
        os.environ['LOG_LEVEL'] = 'DEBUG'
        os.environ['MAX_PARALLEL_EXPERIMENTS'] = '32'
        
        # Reset global config to force reload
        reset_config()
        
        # Create new config (should pick up env vars)
        config = UnifiedConfig()
        assert config.wandb.project == 'test_env_project'
        assert config.logging.level == 'DEBUG'
        assert config.compute.max_parallel_experiments == 32
        
        # Clean up
        del os.environ['WANDB_PROJECT']
        del os.environ['LOG_LEVEL']
        del os.environ['MAX_PARALLEL_EXPERIMENTS']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])