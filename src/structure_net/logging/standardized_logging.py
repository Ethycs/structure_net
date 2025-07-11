#!/usr/bin/env python3
"""
Standardized Logging System for Structure Net

Implements a WandB artifact-based logging standard with Pydantic validation.
This system ensures all experimental data follows a consistent JSON schema
and is properly uploaded to WandB as versioned artifacts.

Key Features:
- Pydantic schema validation for all logged data
- WandB artifact standard compliance
- Local queue system for offline resilience
- Automatic deduplication via content hashing
- Schema migration support
- Comprehensive experiment tracking
"""

import json
import hashlib
import time
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import torch
import wandb
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC SCHEMAS FOR VALIDATION
# ============================================================================

class MetricsData(BaseModel):
    """Schema for metrics data."""
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy")
    loss: float = Field(..., ge=0.0, description="Training loss")
    epoch: int = Field(..., ge=0, description="Training epoch")
    iteration: Optional[int] = Field(None, ge=0, description="Training iteration")
    learning_rate: Optional[float] = Field(None, gt=0.0, description="Learning rate")
    
    # Network structure metrics
    total_parameters: Optional[int] = Field(None, ge=0, description="Total parameters")
    active_connections: Optional[int] = Field(None, ge=0, description="Active connections")
    sparsity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Network sparsity")
    
    # Growth metrics
    growth_occurred: Optional[bool] = Field(None, description="Whether growth occurred")
    architecture: Optional[List[int]] = Field(None, description="Network architecture")
    extrema_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Extrema ratio")


class ExperimentConfig(BaseModel):
    """Schema for experiment configuration."""
    experiment_id: str = Field(..., description="Unique experiment identifier")
    experiment_type: str = Field(..., description="Type of experiment")
    dataset: str = Field(..., description="Dataset used")
    model_type: str = Field(..., description="Model architecture type")
    
    # Training parameters
    batch_size: int = Field(..., gt=0, description="Batch size")
    learning_rate: float = Field(..., gt=0.0, description="Initial learning rate")
    epochs: int = Field(..., gt=0, description="Number of epochs")
    
    # Network parameters
    seed_architecture: Optional[List[int]] = Field(None, description="Initial architecture")
    sparsity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Target sparsity")
    growth_enabled: Optional[bool] = Field(None, description="Whether growth is enabled")
    
    # System parameters
    device: str = Field(..., description="Computing device")
    random_seed: Optional[int] = Field(None, description="Random seed")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @field_validator('experiment_id')
    def validate_experiment_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError('experiment_id must be at least 3 characters')
        return v


class GrowthEvent(BaseModel):
    """Schema for growth events."""
    epoch: int = Field(..., ge=0, description="Epoch when growth occurred")
    iteration: Optional[int] = Field(None, ge=0, description="Iteration when growth occurred")
    
    # Growth details
    growth_type: str = Field(..., description="Type of growth (add_layer, add_connections, etc.)")
    growth_location: Optional[str] = Field(None, description="Where growth occurred")
    connections_added: Optional[int] = Field(None, ge=0, description="Number of connections added")
    
    # Performance impact
    accuracy_before: float = Field(..., ge=0.0, le=1.0, description="Accuracy before growth")
    accuracy_after: float = Field(..., ge=0.0, le=1.0, description="Accuracy after growth")
    performance_delta: Optional[float] = Field(None, description="Performance change")
    
    # Architecture state
    architecture_before: Optional[List[int]] = Field(None, description="Architecture before growth")
    architecture_after: Optional[List[int]] = Field(None, description="Architecture after growth")
    
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HomologicalMetrics(BaseModel):
    """Schema for homological analysis metrics."""
    rank: int = Field(..., ge=0, description="Matrix rank")
    betti_numbers: List[int] = Field(..., description="Betti numbers")
    information_efficiency: float = Field(..., ge=0.0, le=1.0, description="Information efficiency")
    kernel_dimension: int = Field(..., ge=0, description="Kernel dimension")
    image_dimension: int = Field(..., ge=0, description="Image dimension")
    bottleneck_severity: float = Field(..., ge=0.0, le=1.0, description="Bottleneck severity")


class TopologicalMetrics(BaseModel):
    """Schema for topological analysis metrics."""
    extrema_count: int = Field(..., ge=0, description="Number of extrema detected")
    extrema_density: float = Field(..., ge=0.0, description="Extrema density")
    persistence_entropy: float = Field(..., ge=0.0, description="Persistence entropy")
    connectivity_density: float = Field(..., ge=0.0, le=1.0, description="Connectivity density")
    topological_complexity: float = Field(..., ge=0.0, description="Topological complexity")


class CompactificationMetrics(BaseModel):
    """Schema for compactification metrics."""
    compression_ratio: float = Field(..., ge=0.0, le=1.0, description="Compression ratio")
    patch_count: int = Field(..., ge=0, description="Number of patches")
    memory_efficiency: float = Field(..., ge=0.0, le=1.0, description="Memory efficiency")
    reconstruction_error: float = Field(..., ge=0.0, description="Reconstruction error")
    information_preservation: float = Field(..., ge=0.0, le=1.0, description="Information preservation")


class ExperimentResult(BaseModel):
    """Main schema for experiment results."""
    # Metadata
    schema_version: str = Field(default="1.0", description="Schema version")
    experiment_id: str = Field(..., description="Unique experiment identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Configuration
    config: ExperimentConfig = Field(..., description="Experiment configuration")
    
    # Core metrics
    metrics: MetricsData = Field(..., description="Core performance metrics")
    
    # Optional specialized metrics
    homological_metrics: Optional[HomologicalMetrics] = Field(None, description="Homological analysis")
    topological_metrics: Optional[TopologicalMetrics] = Field(None, description="Topological analysis")
    compactification_metrics: Optional[CompactificationMetrics] = Field(None, description="Compactification analysis")
    
    # Growth tracking
    growth_events: Optional[List[GrowthEvent]] = Field(None, description="Growth events")
    
    # Additional data
    custom_metrics: Optional[Dict[str, Any]] = Field(None, description="Custom metrics")
    
    class Config:
        # Allow arbitrary types for torch tensors, etc.
        arbitrary_types_allowed = True
        validate_by_name = True


# ============================================================================
# LOGGING SYSTEM IMPLEMENTATION
# ============================================================================

@dataclass
class LoggingConfig:
    """Configuration for the logging system."""
    project_name: str = "structure_net"
    queue_dir: str = "data/experiment_queue"
    sent_dir: str = "data/experiment_sent"
    rejected_dir: str = "data/experiment_rejected"
    enable_wandb: bool = True
    enable_local_backup: bool = True
    auto_upload: bool = True
    upload_interval: int = 30  # seconds
    max_retries: int = 3
    enable_chromadb: bool = True
    chromadb_path: str = "./chroma_db"


class StandardizedLogger:
    """
    Standardized logging system with WandB artifact integration.
    
    Ensures all experimental data follows the defined Pydantic schemas
    and is properly versioned as WandB artifacts.
    """
    
    def __init__(self, config: LoggingConfig = None):
        self.config = config or LoggingConfig()
        
        # Setup directories
        self.queue_dir = Path(self.config.queue_dir)
        self.sent_dir = Path(self.config.sent_dir)
        self.rejected_dir = Path(self.config.rejected_dir)
        
        for dir_path in [self.queue_dir, self.sent_dir, self.rejected_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize WandB if enabled
        self.wandb_initialized = False
        if self.config.enable_wandb:
            self._initialize_wandb()
        
        # Initialize ChromaDB if enabled
        self.chromadb_client = None
        self.experiments_collection = None
        self.hypotheses_collection = None
        if self.config.enable_chromadb:
            self._initialize_chromadb()
        
        # Upload daemon state
        self.upload_daemon_running = False
        
        logger.info(f"StandardizedLogger initialized with project: {self.config.project_name}")
    
    def _initialize_wandb(self):
        """Initialize WandB connection."""
        try:
            if not wandb.run:
                wandb.init(
                    project=self.config.project_name,
                    job_type="experiment_logger",
                    reinit=True
                )
            self.wandb_initialized = True
            logger.info("WandB initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_initialized = False
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB connection."""
        try:
            self.chromadb_client = chromadb.PersistentClient(
                path=self.config.chromadb_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collections
            try:
                self.experiments_collection = self.chromadb_client.get_collection("experiments")
            except:
                self.experiments_collection = self.chromadb_client.create_collection(
                    name="experiments",
                    metadata={"description": "Neural architecture experiments"}
                )
            
            try:
                self.hypotheses_collection = self.chromadb_client.get_collection("hypotheses")
            except:
                self.hypotheses_collection = self.chromadb_client.create_collection(
                    name="hypotheses",
                    metadata={"description": "Experiment hypotheses"}
                )
            
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ChromaDB: {e}")
            self.chromadb_client = None
    
    def log_experiment_result(self, result: Union[ExperimentResult, Dict[str, Any]]) -> str:
        """
        Log an experiment result with full validation.
        
        Args:
            result: ExperimentResult object or dictionary
            
        Returns:
            Hash of the logged result
        """
        # Convert to ExperimentResult if needed
        if isinstance(result, dict):
            try:
                result = ExperimentResult(**result)
            except Exception as e:
                logger.error(f"Failed to validate experiment result: {e}")
                self._quarantine_invalid_data(result, str(e))
                raise ValueError(f"Invalid experiment result: {e}")
        
        # Convert to JSON
        result_dict = asdict(result)
        json_payload = json.dumps(result_dict, separators=(",", ":"), default=self._json_serializer)
        
        # Generate hash for deduplication
        content_hash = hashlib.sha256(json_payload.encode()).hexdigest()[:16]
        
        # Save to queue
        queue_file = self.queue_dir / f"{content_hash}.json"
        queue_file.write_text(json_payload)
        
        logger.info(f"Queued experiment result {content_hash} ({len(json_payload)} bytes)")
        
        # Log to ChromaDB if enabled
        if self.chromadb_client and self.experiments_collection:
            try:
                # Prepare metadata for ChromaDB
                metadata = {
                    'experiment_id': result.experiment_id,
                    'hypothesis_id': getattr(result, 'hypothesis_id', 'unknown'),
                    'status': 'failed' if hasattr(result, 'error') and result.error else 'completed',
                    'accuracy': float(result.metrics.get('accuracy', 0)) if hasattr(result, 'metrics') else 0.0,
                    'model_parameters': int(result.model_parameters) if hasattr(result, 'model_parameters') else 0,
                    'training_time': float(result.training_time) if hasattr(result, 'training_time') else 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'architecture': str(result.model_architecture) if hasattr(result, 'model_architecture') else '[]',
                    'primary_metric': float(result.primary_metric) if hasattr(result, 'primary_metric') else 0.0
                }
                
                # Add error if present
                if hasattr(result, 'error') and result.error:
                    metadata['error'] = str(result.error)[:500]  # Limit error message length
                
                # Create document text for semantic search
                doc_text = f"Experiment {result.experiment_id} "
                if hasattr(result, 'hypothesis_id'):
                    doc_text += f"testing hypothesis {result.hypothesis_id} "
                doc_text += f"with architecture {metadata['architecture']} "
                doc_text += f"achieved accuracy {metadata['accuracy']:.4f}"
                
                # Add to ChromaDB
                self.experiments_collection.add(
                    ids=[content_hash],
                    documents=[doc_text],
                    metadatas=[metadata]
                )
                logger.debug(f"Logged experiment {content_hash} to ChromaDB")
            except Exception as e:
                logger.warning(f"Failed to log to ChromaDB: {e}")
        
        # Auto-upload if enabled
        if self.config.auto_upload and self.wandb_initialized:
            self._upload_file(queue_file)
        
        return content_hash
    
    def log_hypothesis(self, hypothesis: Dict[str, Any]) -> str:
        """
        Log a hypothesis to ChromaDB.
        
        Args:
            hypothesis: Hypothesis data dictionary
            
        Returns:
            Hypothesis ID
        """
        if not self.chromadb_client or not self.hypotheses_collection:
            return hypothesis.get('id', 'unknown')
        
        try:
            hyp_id = hypothesis.get('id', 'unknown')
            
            # Prepare metadata
            metadata = {
                'id': hyp_id,
                'name': hypothesis.get('name', 'Unknown'),
                'category': hypothesis.get('category', 'unknown'),
                'created_at': hypothesis.get('created_at', datetime.now().isoformat()),
                'tested': hypothesis.get('tested', False),
                'description': hypothesis.get('description', '')[:500],
                'question': hypothesis.get('question', '')[:500],
                'prediction': hypothesis.get('prediction', '')[:500]
            }
            
            # Create document for semantic search
            doc_text = f"Hypothesis {metadata['name']}: {metadata['description']} "
            doc_text += f"Question: {metadata['question']} "
            doc_text += f"Category: {metadata['category']}"
            
            # Add to ChromaDB
            self.hypotheses_collection.add(
                ids=[hyp_id],
                documents=[doc_text],
                metadatas=[metadata]
            )
            
            logger.debug(f"Logged hypothesis {hyp_id} to ChromaDB")
            return hyp_id
            
        except Exception as e:
            logger.warning(f"Failed to log hypothesis to ChromaDB: {e}")
            return hypothesis.get('id', 'unknown')
    
    def log_metrics(self, experiment_id: str, metrics: Union[MetricsData, Dict[str, Any]]) -> str:
        """
        Log metrics data with validation.
        
        Args:
            experiment_id: Experiment identifier
            metrics: MetricsData object or dictionary
            
        Returns:
            Hash of the logged metrics
        """
        # Validate metrics
        if isinstance(metrics, dict):
            try:
                metrics = MetricsData(**metrics)
            except Exception as e:
                logger.error(f"Failed to validate metrics: {e}")
                raise ValueError(f"Invalid metrics data: {e}")
        
        # Create minimal experiment result
        config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_type="metrics_only",
            dataset="unknown",
            model_type="unknown",
            batch_size=32,
            learning_rate=0.001,
            epochs=1,
            device="unknown"
        )
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            metrics=metrics
        )
        
        return self.log_experiment_result(result)
    
    def log_growth_event(self, experiment_id: str, growth_event: Union[GrowthEvent, Dict[str, Any]]) -> str:
        """
        Log a growth event with validation.
        
        Args:
            experiment_id: Experiment identifier
            growth_event: GrowthEvent object or dictionary
            
        Returns:
            Hash of the logged event
        """
        # Validate growth event
        if isinstance(growth_event, dict):
            try:
                growth_event = GrowthEvent(**growth_event)
            except Exception as e:
                logger.error(f"Failed to validate growth event: {e}")
                raise ValueError(f"Invalid growth event: {e}")
        
        # Create experiment result with growth event
        config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_type="growth_event",
            dataset="unknown",
            model_type="unknown",
            batch_size=32,
            learning_rate=0.001,
            epochs=1,
            device="unknown"
        )
        
        metrics = MetricsData(
            accuracy=growth_event.accuracy_after,
            loss=0.0,
            epoch=growth_event.epoch
        )
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            metrics=metrics,
            growth_events=[growth_event]
        )
        
        return self.log_experiment_result(result)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for special types."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return str(obj)
    
    def _quarantine_invalid_data(self, data: Any, error_msg: str):
        """Move invalid data to quarantine directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_file = self.rejected_dir / f"invalid_{timestamp}.json"
        
        quarantine_data = {
            "error": error_msg,
            "timestamp": timestamp,
            "data": data
        }
        
        try:
            with open(quarantine_file, 'w') as f:
                json.dump(quarantine_data, f, indent=2, default=self._json_serializer)
            logger.warning(f"Invalid data quarantined to {quarantine_file}")
        except Exception as e:
            logger.error(f"Failed to quarantine invalid data: {e}")
    
    def _upload_file(self, file_path: Path) -> bool:
        """Upload a single file to WandB as an artifact."""
        if not self.wandb_initialized:
            return False
        
        try:
            content_hash = file_path.stem
            
            # Check if already uploaded
            if self._already_uploaded(content_hash):
                logger.info(f"Skipping {content_hash}, already in WandB")
                self._move_to_sent(file_path)
                return True
            
            # Load and validate data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate against schema
            try:
                ExperimentResult(**data)
            except Exception as e:
                logger.error(f"Schema validation failed for {file_path}: {e}")
                self._move_to_rejected(file_path, str(e))
                return False
            
            # Create WandB artifact
            artifact = wandb.Artifact(
                name=content_hash,
                type="experiment-result",
                description=f"Validated experiment result v{data.get('schema_version', '1.0')}",
                metadata={
                    "experiment_id": data.get("experiment_id"),
                    "experiment_type": data.get("config", {}).get("experiment_type"),
                    "schema_version": data.get("schema_version", "1.0"),
                    "upload_timestamp": datetime.now().isoformat()
                }
            )
            
            # Add file to artifact
            artifact.add_file(str(file_path), name=f"{content_hash}.json")
            
            # Log artifact
            wandb.log_artifact(artifact)
            
            # Move to sent directory
            self._move_to_sent(file_path)
            
            logger.info(f"Successfully uploaded {content_hash} to WandB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            return False
    
    def _already_uploaded(self, content_hash: str) -> bool:
        """Check if artifact already exists in WandB."""
        try:
            api = wandb.Api()
            api.artifact(f"{self.config.project_name}/{content_hash}:latest")
            return True
        except wandb.errors.CommError:
            return False
        except Exception as e:
            logger.warning(f"Error checking if {content_hash} exists: {e}")
            return False
    
    def _move_to_sent(self, file_path: Path):
        """Move file to sent directory."""
        sent_path = self.sent_dir / file_path.name
        shutil.move(str(file_path), str(sent_path))
    
    def _move_to_rejected(self, file_path: Path, error_msg: str):
        """Move file to rejected directory with error info."""
        rejected_path = self.rejected_dir / file_path.name
        
        # Add error information
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            data['_rejection_info'] = {
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(rejected_path, 'w') as f:
                json.dump(data, f, indent=2, default=self._json_serializer)
            
            file_path.unlink()  # Remove original
            
        except Exception as e:
            logger.error(f"Failed to move {file_path} to rejected: {e}")
            # Fallback: just move the file
            shutil.move(str(file_path), str(rejected_path))
    
    def start_upload_daemon(self):
        """Start background upload daemon."""
        if self.upload_daemon_running:
            logger.warning("Upload daemon already running")
            return
        
        import threading
        
        def upload_loop():
            self.upload_daemon_running = True
            logger.info("Upload daemon started")
            
            while self.upload_daemon_running:
                try:
                    # Process all files in queue
                    for file_path in self.queue_dir.glob("*.json"):
                        if not self.upload_daemon_running:
                            break
                        
                        self._upload_file(file_path)
                    
                    # Wait before next iteration
                    time.sleep(self.config.upload_interval)
                    
                except Exception as e:
                    logger.error(f"Upload daemon error: {e}")
                    time.sleep(self.config.upload_interval)
            
            logger.info("Upload daemon stopped")
        
        daemon_thread = threading.Thread(target=upload_loop, daemon=True)
        daemon_thread.start()
    
    def stop_upload_daemon(self):
        """Stop background upload daemon."""
        self.upload_daemon_running = False
    
    def upload_all_queued(self) -> Dict[str, int]:
        """Upload all queued files immediately."""
        stats = {"uploaded": 0, "failed": 0, "skipped": 0}
        
        for file_path in self.queue_dir.glob("*.json"):
            if self._upload_file(file_path):
                stats["uploaded"] += 1
            else:
                stats["failed"] += 1
        
        logger.info(f"Upload complete: {stats}")
        return stats
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get status of the upload queue."""
        return {
            "queued": len(list(self.queue_dir.glob("*.json"))),
            "sent": len(list(self.sent_dir.glob("*.json"))),
            "rejected": len(list(self.rejected_dir.glob("*.json")))
        }
    
    def validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validate data against the experiment result schema."""
        try:
            ExperimentResult(**data)
            return True
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    def migrate_schema(self, old_data: Dict[str, Any], target_version: str = "1.0") -> Dict[str, Any]:
        """Migrate old data to current schema version."""
        # This is where you would implement schema migration logic
        # For now, just add schema version if missing
        if "schema_version" not in old_data:
            old_data["schema_version"] = target_version
        
        return old_data


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global logger instance
_global_logger: Optional[StandardizedLogger] = None


def initialize_logging(config: LoggingConfig = None) -> StandardizedLogger:
    """Initialize the global logging system."""
    global _global_logger
    _global_logger = StandardizedLogger(config)
    return _global_logger


def get_logger() -> StandardizedLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StandardizedLogger()
    return _global_logger


def log_experiment(result: Union[ExperimentResult, Dict[str, Any]]) -> str:
    """Convenience function to log experiment result."""
    return get_logger().log_experiment_result(result)


def log_metrics(experiment_id: str, metrics: Union[MetricsData, Dict[str, Any]]) -> str:
    """Convenience function to log metrics."""
    return get_logger().log_metrics(experiment_id, metrics)


def log_growth_event(experiment_id: str, growth_event: Union[GrowthEvent, Dict[str, Any]]) -> str:
    """Convenience function to log growth event."""
    return get_logger().log_growth_event(experiment_id, growth_event)


# Export all components
__all__ = [
    # Schemas
    'ExperimentResult',
    'ExperimentConfig',
    'MetricsData',
    'GrowthEvent',
    'HomologicalMetrics',
    'TopologicalMetrics',
    'CompactificationMetrics',
    
    # Main classes
    'StandardizedLogger',
    'LoggingConfig',
    
    # Convenience functions
    'initialize_logging',
    'get_logger',
    'log_experiment',
    'log_metrics',
    'log_growth_event'
]
