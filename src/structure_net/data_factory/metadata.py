"""
Dataset Metadata Tracking

Extends the existing logging system to track dataset usage and metadata.
Integrates with StandardizedLogger for comprehensive experiment tracking.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import hashlib
from pydantic import BaseModel, Field

from structure_net.logging.standardized_logging import get_logger, StandardizedLogger

import logging
logger = logging.getLogger(__name__)


class DatasetMetadata(BaseModel):
    """Schema for dataset metadata."""
    
    dataset_name: str = Field(..., description="Name of the dataset")
    dataset_version: str = Field(..., description="Version of the dataset")
    download_timestamp: str = Field(..., description="When dataset was downloaded")
    size_bytes: int = Field(..., ge=0, description="Size of dataset in bytes")
    num_samples: int = Field(..., gt=0, description="Total number of samples")
    shape: List[int] = Field(..., description="Shape of individual samples")
    checksum: str = Field(..., description="Dataset checksum for verification")
    
    # Additional metadata
    config: Dict[str, Any] = Field(default_factory=dict, description="Dataset configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        extra = "allow"


class DatasetUsage(BaseModel):
    """Schema for tracking dataset usage in experiments."""
    
    experiment_id: str = Field(..., description="ID of experiment using the dataset")
    dataset_name: str = Field(..., description="Name of dataset used")
    dataset_version: str = Field(..., description="Version of dataset used")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    subset_used: Optional[str] = Field(None, description="Which subset (train/test/val)")
    samples_used: int = Field(..., gt=0, description="Number of samples used")
    transformations_applied: List[str] = Field(default_factory=list, description="List of transformations")
    
    # Performance tracking
    loading_time: Optional[float] = Field(None, description="Time to load dataset")
    
    class Config:
        extra = "allow"


class DatasetRegistry(BaseModel):
    """Registry of all datasets and their metadata."""
    
    datasets: Dict[str, DatasetMetadata] = Field(default_factory=dict)
    usage_history: List[DatasetUsage] = Field(default_factory=list)
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


# Global dataset registry path
DATASET_METADATA_PATH = Path("data/dataset_metadata")
DATASET_REGISTRY_FILE = DATASET_METADATA_PATH / "dataset_registry.json"
USAGE_LOG_DIR = DATASET_METADATA_PATH / "usage_logs"

# Ensure directories exist
DATASET_METADATA_PATH.mkdir(parents=True, exist_ok=True)
USAGE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset_registry() -> DatasetRegistry:
    """Load the dataset registry from disk."""
    if DATASET_REGISTRY_FILE.exists():
        try:
            with open(DATASET_REGISTRY_FILE, 'r') as f:
                data = json.load(f)
            return DatasetRegistry(**data)
        except Exception as e:
            logger.error(f"Failed to load dataset registry: {e}")
    
    return DatasetRegistry()


def save_dataset_registry(registry: DatasetRegistry) -> None:
    """Save the dataset registry to disk."""
    try:
        with open(DATASET_REGISTRY_FILE, 'w') as f:
            json.dump(registry.model_dump(), f, indent=2)
        logger.info("Saved dataset registry")
    except Exception as e:
        logger.error(f"Failed to save dataset registry: {e}")


def save_dataset_metadata(metadata: DatasetMetadata) -> None:
    """Save metadata for a dataset."""
    registry = load_dataset_registry()
    
    # Update or add dataset metadata
    registry.datasets[metadata.dataset_name] = metadata.model_dump()
    registry.last_updated = datetime.now().isoformat()
    
    save_dataset_registry(registry)
    logger.info(f"Saved metadata for dataset: {metadata.dataset_name}")


def get_dataset_metadata(dataset_name: str) -> Optional[DatasetMetadata]:
    """Get metadata for a dataset."""
    registry = load_dataset_registry()
    
    if dataset_name in registry.datasets:
        return DatasetMetadata(**registry.datasets[dataset_name])
    
    return None


def track_dataset_usage(
    experiment_id: str,
    dataset_name: str,
    dataset_version: str,
    subset_used: Optional[str] = None,
    samples_used: int = 0,
    transformations_applied: List[str] = None,
    loading_time: Optional[float] = None
) -> str:
    """
    Track dataset usage in an experiment.
    
    Returns:
        Usage tracking ID
    """
    usage = DatasetUsage(
        experiment_id=experiment_id,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        subset_used=subset_used,
        samples_used=samples_used,
        transformations_applied=transformations_applied or [],
        loading_time=loading_time
    )
    
    # Save to registry
    registry = load_dataset_registry()
    registry.usage_history.append(usage.model_dump())
    registry.last_updated = datetime.now().isoformat()
    save_dataset_registry(registry)
    
    # Also log to experiment logging system
    try:
        logging_system = get_logger()
        if logging_system:
            # Add to experiment metadata
            custom_metrics = {
                "dataset_usage": usage.model_dump()
            }
            # This will be picked up by the experiment logging
            logger.info(f"Tracked dataset usage for {experiment_id}: {dataset_name} v{dataset_version}")
    except Exception as e:
        logger.warning(f"Could not log to experiment system: {e}")
    
    # Generate usage ID
    usage_id = hashlib.md5(
        f"{experiment_id}_{dataset_name}_{dataset_version}_{usage.timestamp}".encode()
    ).hexdigest()[:8]
    
    # Save detailed usage log
    usage_file = USAGE_LOG_DIR / f"{experiment_id}_{usage_id}.json"
    with open(usage_file, 'w') as f:
        json.dump(usage.model_dump(), f, indent=2)
    
    return usage_id


def get_dataset_usage_history(
    dataset_name: Optional[str] = None,
    experiment_id: Optional[str] = None
) -> List[DatasetUsage]:
    """
    Get dataset usage history.
    
    Args:
        dataset_name: Filter by dataset name
        experiment_id: Filter by experiment ID
        
    Returns:
        List of DatasetUsage records
    """
    registry = load_dataset_registry()
    usage_history = [DatasetUsage(**u) for u in registry.usage_history]
    
    # Apply filters
    if dataset_name:
        usage_history = [u for u in usage_history if u.dataset_name == dataset_name]
    
    if experiment_id:
        usage_history = [u for u in usage_history if u.experiment_id == experiment_id]
    
    return usage_history


def get_dataset_statistics(dataset_name: str) -> Dict[str, Any]:
    """
    Get usage statistics for a dataset.
    
    Returns:
        Dictionary with usage statistics
    """
    usage_history = get_dataset_usage_history(dataset_name=dataset_name)
    
    if not usage_history:
        return {
            "dataset_name": dataset_name,
            "total_uses": 0,
            "unique_experiments": 0,
            "total_samples_used": 0
        }
    
    unique_experiments = set(u.experiment_id for u in usage_history)
    total_samples = sum(u.samples_used for u in usage_history)
    
    # Group by subset
    subset_counts = {}
    for usage in usage_history:
        subset = usage.subset_used or "unknown"
        subset_counts[subset] = subset_counts.get(subset, 0) + 1
    
    return {
        "dataset_name": dataset_name,
        "total_uses": len(usage_history),
        "unique_experiments": len(unique_experiments),
        "total_samples_used": total_samples,
        "subset_distribution": subset_counts,
        "first_used": min(u.timestamp for u in usage_history),
        "last_used": max(u.timestamp for u in usage_history),
    }


def cleanup_old_usage_logs(days_to_keep: int = 30) -> int:
    """
    Clean up old usage log files.
    
    Args:
        days_to_keep: Keep logs from this many days
        
    Returns:
        Number of files deleted
    """
    import time
    
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    deleted = 0
    
    for log_file in USAGE_LOG_DIR.glob("*.json"):
        if log_file.stat().st_mtime < cutoff_time:
            log_file.unlink()
            deleted += 1
    
    if deleted > 0:
        logger.info(f"Cleaned up {deleted} old usage log files")
    
    return deleted


# Integration with experiment logging
def log_dataset_info_to_experiment(
    experiment_result: Dict[str, Any],
    dataset_name: str,
    dataset_version: str,
    subset_info: Dict[str, int]
) -> None:
    """
    Add dataset information to experiment result.
    
    This enriches the experiment result with dataset metadata
    before it's logged to the standardized logging system.
    """
    if "custom_metrics" not in experiment_result:
        experiment_result["custom_metrics"] = {}
    
    experiment_result["custom_metrics"]["dataset_info"] = {
        "name": dataset_name,
        "version": dataset_version,
        "subsets": subset_info,
        "metadata_tracked": True
    }