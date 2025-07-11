#!/usr/bin/env python3
"""
Unified Configuration System for Structure Net

This module provides a single, composable configuration system that replaces
the scattered config classes throughout the codebase. Following the principle
of "one source of truth" for all configuration.

Key features:
- Hierarchical configuration with sensible defaults
- Environment variable support
- JSON/YAML serialization
- Validation with helpful error messages
- Backward compatibility helpers
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import yaml
import multiprocessing as mp

# Lazy import torch to avoid import errors if not installed
def _get_cuda_device_count():
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


@dataclass
class StorageConfig:
    """Unified storage configuration for all data persistence."""
    
    # Root directories
    data_root: Path = field(default_factory=lambda: DATA_ROOT)
    results_dir: Path = field(default_factory=lambda: DATA_ROOT / "results")
    cache_dir: Path = field(default_factory=lambda: DATA_ROOT / "cache")
    
    # Dataset storage
    datasets_dir: Path = field(default_factory=lambda: DATA_ROOT / "datasets")
    dataset_metadata_dir: Path = field(default_factory=lambda: DATA_ROOT / "dataset_metadata")
    
    # Experiment artifacts
    experiments_dir: Path = field(default_factory=lambda: DATA_ROOT / "experiments")
    models_dir: Path = field(default_factory=lambda: DATA_ROOT / "models")
    logs_dir: Path = field(default_factory=lambda: DATA_ROOT / "logs")
    
    # Queue system (for offline resilience)
    queue_dir: Path = field(default_factory=lambda: DATA_ROOT / "experiment_queue")
    sent_dir: Path = field(default_factory=lambda: DATA_ROOT / "experiment_sent")
    rejected_dir: Path = field(default_factory=lambda: DATA_ROOT / "experiment_rejected")
    
    # Database paths
    chromadb_path: Path = field(default_factory=lambda: DATA_ROOT / "chroma_db")
    timeseries_path: Path = field(default_factory=lambda: DATA_ROOT / "timeseries_db")
    
    # ChromaDB settings
    chromadb_collection: str = "structure_net_experiments"
    
    # Time series settings
    timeseries_compression: str = "gzip"
    timeseries_chunk_size: int = 1000
    
    def __post_init__(self):
        """Ensure all directories exist."""
        for field_name, field_value in asdict(self).items():
            if isinstance(field_value, (str, Path)) and field_name.endswith(('_dir', '_path', '_root')):
                Path(field_value).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'StorageConfig':
        """Create config from environment variables."""
        data_root = Path(os.getenv('STRUCTURE_NET_DATA_ROOT', str(DATA_ROOT)))
        return cls(data_root=data_root)


@dataclass
class WandBConfig:
    """Unified Weights & Biases configuration."""
    
    enabled: bool = field(default_factory=lambda: os.getenv('WANDB_ENABLED', 'true').lower() == 'true')
    project: str = field(default_factory=lambda: os.getenv('WANDB_PROJECT', 'structure_net'))
    entity: Optional[str] = field(default_factory=lambda: os.getenv('WANDB_ENTITY'))
    
    # API settings
    api_key: Optional[str] = field(default_factory=lambda: os.getenv('WANDB_API_KEY'))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv('WANDB_BASE_URL'))
    
    # Run settings
    job_type: str = "experiment"
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Offline settings
    mode: str = field(default_factory=lambda: os.getenv('WANDB_MODE', 'online'))
    resume: str = "allow"  # allow, must, never, auto
    
    # Artifact settings
    artifact_type: str = "experiment_result"
    log_artifacts: bool = True
    
    def to_wandb_init_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for wandb.init()."""
        return {
            'project': self.project,
            'entity': self.entity,
            'job_type': self.job_type,
            'tags': self.tags,
            'notes': self.notes,
            'mode': self.mode,
            'resume': self.resume,
            'dir': str(StorageConfig().data_root)  # Always use data directory
        }


@dataclass  
class LoggingConfig:
    """Unified logging configuration."""
    
    # Levels
    level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    file_level: str = "DEBUG"
    console_level: str = "INFO"
    
    # Output settings
    enable_console: bool = True
    enable_file: bool = True
    enable_wandb: bool = True
    enable_chromadb: bool = True
    
    # File settings
    log_file: Optional[str] = None  # Auto-generated if None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Format
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'
    
    # Module-specific levels
    module_levels: Dict[str, str] = field(default_factory=lambda: {
        'chromadb': 'WARNING',
        'urllib3': 'WARNING',
        'matplotlib': 'WARNING'
    })
    
    def get_log_file_path(self, storage: StorageConfig) -> Path:
        """Get the log file path."""
        if self.log_file:
            return Path(self.log_file)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return storage.logs_dir / f"structure_net_{timestamp}.log"


@dataclass
class ComputeConfig:
    """Unified compute resource configuration."""
    
    # Device settings
    device: str = field(default_factory=lambda: os.getenv('DEVICE', 'auto'))
    device_ids: List[int] = field(default_factory=lambda: 
        [int(x) for x in os.getenv('CUDA_VISIBLE_DEVICES', '').split(',') if x]
        or list(range(_get_cuda_device_count()))
    )
    
    # Parallelism
    max_parallel_experiments: int = field(default_factory=lambda: 
        int(os.getenv('MAX_PARALLEL_EXPERIMENTS', '0')) or 8
    )
    num_workers: int = field(default_factory=lambda: int(os.getenv('NUM_WORKERS', '4')))
    
    # Memory management
    pin_memory: bool = True
    non_blocking: bool = True
    empty_cache_interval: int = 100  # iterations
    
    # Resource limits
    max_memory_percent: float = 90.0
    max_gpu_memory_percent: float = 95.0
    max_cpu_percent: float = 90.0  # Added for NAL compatibility
    
    # Auto-balancing
    enable_auto_balance: bool = True
    target_gpu_utilization: float = 85.0
    target_cpu_utilization: float = 75.0
    
    def get_device(self, experiment_id: int = 0) -> str:
        """Get device string for an experiment."""
        import torch
        
        if self.device != 'auto':
            return self.device
            
        if torch.cuda.is_available() and self.device_ids:
            device_id = self.device_ids[experiment_id % len(self.device_ids)]
            return f'cuda:{device_id}'
        
        return 'cpu'


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""
    
    # Basic settings
    project_name: str = "structure_net"
    experiment_name: Optional[str] = None  # Auto-generated if None
    random_seed: Optional[int] = None
    
    # Training defaults
    batch_size: int = 128
    learning_rate: float = 0.001
    epochs: int = 100
    
    # Validation
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    
    # Checkpointing
    save_checkpoints: bool = True
    save_all_checkpoints: bool = False
    checkpoint_interval: int = 10
    keep_n_checkpoints: int = 3
    
    # Profiling
    enable_profiling: bool = False
    profile_wait: int = 1
    profile_warmup: int = 1
    profile_active: int = 3
    
    # Scientific rigor (from NAL)
    min_experiments: int = 5
    require_significance: bool = True
    significance_level: float = 0.05
    
    # Adaptive exploration
    enable_adaptive: bool = True
    max_hypothesis_depth: int = 3
    
    # Timeouts
    timeout_minutes: int = 60  # Per experiment timeout
    
    def get_experiment_name(self) -> str:
        """Get experiment name, generating if needed."""
        if self.experiment_name:
            return self.experiment_name
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{self.project_name}_{timestamp}"


@dataclass
class UnifiedConfig:
    """
    The main unified configuration that combines all subsystems.
    
    This is the single configuration object that should be used throughout
    the codebase, replacing all the scattered config classes.
    """
    
    # Sub-configurations
    storage: StorageConfig = field(default_factory=StorageConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Global settings
    debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')
    verbose: bool = field(default_factory=lambda: os.getenv('VERBOSE', 'false').lower() == 'true')
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'UnifiedConfig':
        """Load configuration from JSON or YAML file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedConfig':
        """Create config from dictionary."""
        config = cls()
        
        # Update sub-configs
        if 'storage' in data:
            config.storage = StorageConfig(**data['storage'])
        if 'wandb' in data:
            config.wandb = WandBConfig(**data['wandb'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        if 'compute' in data:
            config.compute = ComputeConfig(**data['compute'])
        if 'experiment' in data:
            config.experiment = ExperimentConfig(**data['experiment'])
        
        # Update global settings
        config.debug = data.get('debug', config.debug)
        config.verbose = data.get('verbose', config.verbose)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'storage': asdict(self.storage),
            'wandb': asdict(self.wandb),
            'logging': asdict(self.logging),
            'compute': asdict(self.compute),
            'experiment': asdict(self.experiment),
            'debug': self.debug,
            'verbose': self.verbose
        }
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            return obj
        
        data = convert_paths(data)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def get_lab_config(self) -> 'LabConfig':
        """Get backward-compatible LabConfig."""
        from src.neural_architecture_lab.core import LabConfig
        
        return LabConfig(
            # Execution
            max_parallel_experiments=self.compute.max_parallel_experiments,
            experiment_timeout=self.experiment.timeout_minutes * 60,  # Convert to seconds
            device_ids=self.compute.device_ids,
            
            # Scientific rigor
            min_experiments_per_hypothesis=self.experiment.min_experiments,
            require_statistical_significance=self.experiment.require_significance,
            significance_level=self.experiment.significance_level,
            
            # Resource management
            max_memory_per_experiment=self.compute.max_gpu_memory_percent / 100,  # Convert to fraction
            checkpoint_frequency=self.experiment.checkpoint_interval,
            
            # Logging and output
            project_name=self.experiment.project_name,
            results_dir=str(self.storage.results_dir),
            save_all_models=self.experiment.save_all_checkpoints,
            save_best_models=self.experiment.save_checkpoints,
            verbose=self.verbose,
            log_level=self.logging.level,
            module_log_levels=self.logging.module_levels,
            log_file=str(self.logging.get_log_file_path(self.storage)) if self.logging.enable_file else None,
            enable_wandb=self.wandb.enabled,
            
            # Adaptive exploration
            enable_adaptive_hypotheses=self.experiment.enable_adaptive,
            max_hypothesis_depth=self.experiment.max_hypothesis_depth,
            
            # Auto-balancing settings
            auto_balance=self.compute.enable_auto_balance,
            target_cpu_percent=self.compute.target_cpu_utilization,
            max_cpu_percent=self.compute.max_cpu_percent,
            target_gpu_percent=self.compute.target_gpu_utilization,
            max_gpu_percent=self.compute.max_gpu_memory_percent,
            target_memory_percent=self.compute.max_memory_percent,
            max_memory_percent=self.compute.max_memory_percent
        )
    
    def get_logging_config(self) -> 'LoggingConfig':
        """Get backward-compatible LoggingConfig for StandardizedLogger."""
        from src.structure_net.logging.standardized_logging import LoggingConfig as OldLoggingConfig
        
        return OldLoggingConfig(
            project_name=self.experiment.project_name,
            queue_dir=str(self.storage.queue_dir),
            sent_dir=str(self.storage.sent_dir),
            rejected_dir=str(self.storage.rejected_dir),
            enable_wandb=self.wandb.enabled,
            enable_chromadb=self.logging.enable_chromadb,
            chromadb_path=str(self.storage.chromadb_path)
        )
    
    def get_chroma_config(self) -> 'ChromaConfig':
        """Get backward-compatible ChromaConfig for ChromaDB client."""
        from src.data_factory.search.chroma_client import ChromaConfig
        
        return ChromaConfig(
            persist_directory=str(self.storage.chromadb_path),
            collection_name=self.storage.chromadb_collection
        )
    
    def get_timeseries_config(self) -> 'TimeSeriesConfig':
        """Get backward-compatible TimeSeriesConfig."""
        from src.data_factory.time_series_storage import TimeSeriesConfig
        
        return TimeSeriesConfig(
            storage_dir=str(self.storage.timeseries_path),
            compression=self.storage.timeseries_compression,
            chunk_size=self.storage.timeseries_chunk_size
        )


# Global configuration instance
_global_config: Optional[UnifiedConfig] = None


def get_config() -> UnifiedConfig:
    """Get the global configuration instance."""
    global _global_config
    
    if _global_config is None:
        # Try to load from default locations
        config_paths = [
            Path.home() / '.structure_net' / 'config.yaml',
            PROJECT_ROOT / 'config.yaml',
            Path('config.yaml')
        ]
        
        for path in config_paths:
            if path.exists():
                _global_config = UnifiedConfig.from_file(path)
                break
        else:
            # Create default config
            _global_config = UnifiedConfig()
    
    return _global_config


def set_config(config: UnifiedConfig):
    """Set the global configuration."""
    global _global_config
    _global_config = config


def reset_config():
    """Reset to default configuration."""
    global _global_config
    _global_config = UnifiedConfig()


# Import guards for missing dependencies
try:
    import torch
    import torch.multiprocessing as mp
except ImportError:
    torch = None
    mp = None
    
    # Patch ComputeConfig defaults
    ComputeConfig.device_ids = field(default_factory=list)
    ComputeConfig.max_parallel_experiments = field(default_factory=lambda: 2)


# Backward compatibility helpers
def create_lab_config(**kwargs) -> 'LabConfig':
    """Create LabConfig with unified config defaults."""
    config = get_config()
    lab_config = config.get_lab_config()
    
    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(lab_config, key):
            setattr(lab_config, key, value)
    
    return lab_config


def create_logging_config(**kwargs) -> 'LoggingConfig':
    """Create LoggingConfig with unified config defaults."""
    config = get_config()
    logging_config = config.get_logging_config()
    
    # Apply any overrides  
    for key, value in kwargs.items():
        if hasattr(logging_config, key):
            setattr(logging_config, key, value)
    
    return logging_config