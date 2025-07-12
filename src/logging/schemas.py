#!/usr/bin/env python3
"""
Pydantic Schemas for Structure Net Logging

This module defines strict schemas for all experiment data types to ensure
consistent data structure and automatic validation across all experiments.

Key features:
- Comprehensive validation for all experiment types
- Automatic timestamp generation
- Type safety and data consistency
- Schema evolution support
- WandB artifact compatibility
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import json
import hashlib


class BaseExperimentSchema(BaseModel):
    """Base schema for all experiments with common fields."""
    
    experiment_id: str = Field(..., description="Unique experiment identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Experiment timestamp")
    experiment_type: str = Field(..., description="Type of experiment")
    schema_version: str = Field(default="1.0", description="Schema version for migration")
    
    class Config:
        # Allow extra fields for backward compatibility during migration
        extra = "allow"
        # Use enum values for serialization
        use_enum_values = True
        # Allow population by field name or alias (renamed in Pydantic v2)
        populate_by_name = True  # This is the v2 name
    
    def generate_artifact_id(self) -> str:
        """Generate a unique artifact ID based on content hash."""
        content = self.json(sort_keys=True, exclude={'timestamp'})
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @field_validator('experiment_id')
    @classmethod
    def validate_experiment_id(cls, v):
        """Ensure experiment ID is non-empty and valid."""
        if not v or not v.strip():
            raise ValueError("experiment_id cannot be empty")
        return v.strip()


class NetworkArchitecture(BaseModel):
    """Schema for network architecture information."""
    
    layers: List[int] = Field(..., description="Number of neurons in each layer")
    total_parameters: int = Field(..., ge=0, description="Total number of parameters")
    total_connections: int = Field(..., ge=0, description="Total number of connections")
    sparsity: float = Field(..., ge=0.0, le=1.0, description="Overall network sparsity")
    depth: int = Field(..., ge=1, description="Number of layers")
    
    @field_validator('layers')
    @classmethod
    def validate_layers(cls, v):
        """Ensure layers list is valid."""
        if not v:
            raise ValueError("layers cannot be empty")
        if any(layer <= 0 for layer in v):
            raise ValueError("all layer sizes must be positive")
        return v
    
    @model_validator(mode='after')
    def validate_depth_consistency(self):
        """Ensure depth matches layers length."""
        if len(self.layers) != self.depth:
            raise ValueError("depth must match number of layers")
        return self


class PerformanceMetrics(BaseModel):
    """Schema for performance metrics."""
    
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy")
    loss: Optional[float] = Field(None, ge=0.0, description="Training/validation loss")
    precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Precision score")
    recall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Recall score")
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="F1 score")
    
    @field_validator('accuracy')
    @classmethod
    def validate_accuracy(cls, v):
        """Ensure accuracy is reasonable."""
        if v < 0.0 or v > 1.0:
            raise ValueError("accuracy must be between 0.0 and 1.0")
        return v


class ExtremaAnalysis(BaseModel):
    """Schema for extrema detection analysis."""
    
    total_extrema: int = Field(..., ge=0, description="Total number of extrema neurons")
    extrema_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of extrema to total neurons")
    dead_neurons: Dict[str, List[int]] = Field(default_factory=dict, description="Dead neurons by layer")
    saturated_neurons: Dict[str, List[int]] = Field(default_factory=dict, description="Saturated neurons by layer")
    layer_health: Dict[str, float] = Field(default_factory=dict, description="Health score by layer")
    
    @field_validator('extrema_ratio')
    @classmethod
    def validate_extrema_ratio(cls, v):
        """Ensure extrema ratio is valid."""
        if v < 0.0 or v > 1.0:
            raise ValueError("extrema_ratio must be between 0.0 and 1.0")
        return v


class GrowthAction(BaseModel):
    """Schema for individual growth actions."""
    
    action_type: Literal["add_layer", "add_patch", "increase_density", "prune"] = Field(..., description="Type of growth action")
    layer_position: Optional[int] = Field(None, ge=0, description="Layer position for action")
    size: Optional[int] = Field(None, ge=1, description="Size of addition (neurons/connections)")
    reason: str = Field(..., description="Reason for taking this action")
    success: bool = Field(default=True, description="Whether action was successful")
    
    @field_validator('reason')
    @classmethod
    def validate_reason(cls, v):
        """Ensure reason is provided."""
        if not v or not v.strip():
            raise ValueError("reason cannot be empty")
        return v.strip()


class TrainingEpoch(BaseModel):
    """Schema for individual training epoch data."""
    
    epoch: int = Field(..., ge=0, description="Epoch number")
    train_loss: float = Field(..., ge=0.0, description="Training loss")
    train_accuracy: float = Field(..., ge=0.0, le=1.0, description="Training accuracy")
    val_loss: Optional[float] = Field(None, ge=0.0, description="Validation loss")
    val_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Validation accuracy")
    learning_rate: Optional[float] = Field(None, gt=0.0, description="Learning rate used")
    duration: Optional[float] = Field(None, ge=0.0, description="Epoch duration in seconds")
    
    @field_validator('train_accuracy', 'val_accuracy')
    @classmethod
    def validate_accuracy_range(cls, v):
        """Ensure accuracy is in valid range."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("accuracy must be between 0.0 and 1.0")
        return v


class GrowthIteration(BaseModel):
    """Schema for growth iteration data."""
    
    iteration: int = Field(..., ge=0, description="Growth iteration number")
    architecture: NetworkArchitecture = Field(..., description="Network architecture")
    performance: PerformanceMetrics = Field(..., description="Performance metrics")
    extrema_analysis: Optional[ExtremaAnalysis] = Field(None, description="Extrema analysis results")
    growth_actions: List[GrowthAction] = Field(default_factory=list, description="Growth actions taken")
    growth_occurred: bool = Field(default=False, description="Whether growth occurred")
    credits: Optional[float] = Field(None, ge=0.0, description="Credits in growth economy")
    
    @field_validator('iteration')
    @classmethod
    def validate_iteration(cls, v):
        """Ensure iteration is non-negative."""
        if v < 0:
            raise ValueError("iteration must be non-negative")
        return v


class TournamentResult(BaseModel):
    """Schema for tournament-based growth results."""
    
    strategy_name: str = Field(..., description="Name of the growth strategy")
    improvement: float = Field(..., description="Performance improvement achieved")
    final_accuracy: float = Field(..., ge=0.0, le=1.0, description="Final accuracy achieved")
    execution_time: Optional[float] = Field(None, ge=0.0, description="Time taken to execute strategy")
    success: bool = Field(default=True, description="Whether strategy executed successfully")
    
    @field_validator('strategy_name')
    @classmethod
    def validate_strategy_name(cls, v):
        """Ensure strategy name is valid."""
        if not v or not v.strip():
            raise ValueError("strategy_name cannot be empty")
        return v.strip()


class ExperimentConfig(BaseModel):
    """Schema for experiment configuration."""
    
    dataset: str = Field(..., description="Dataset used")
    batch_size: int = Field(..., gt=0, description="Batch size")
    learning_rate: float = Field(..., gt=0.0, description="Learning rate")
    max_epochs: int = Field(..., gt=0, description="Maximum epochs")
    device: str = Field(..., description="Device used (cpu/cuda)")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    target_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Target accuracy")
    
    @field_validator('dataset')
    @classmethod
    def validate_dataset(cls, v):
        """Ensure dataset name is valid."""
        if not v or not v.strip():
            raise ValueError("dataset cannot be empty")
        return v.strip().lower()


class GrowthExperiment(BaseExperimentSchema):
    """Schema for growth-based experiments."""
    
    experiment_type: Literal["growth_experiment"] = "growth_experiment"
    config: ExperimentConfig = Field(..., description="Experiment configuration")
    seed_architecture: List[int] = Field(..., description="Initial network architecture")
    scaffold_sparsity: float = Field(..., ge=0.0, le=1.0, description="Initial scaffold sparsity")
    growth_history: List[GrowthIteration] = Field(default_factory=list, description="Growth iteration history")
    final_performance: Optional[PerformanceMetrics] = Field(None, description="Final performance metrics")
    total_iterations: int = Field(default=0, ge=0, description="Total growth iterations")
    
    @field_validator('scaffold_sparsity')
    @classmethod
    def validate_sparsity(cls, v):
        """Ensure sparsity is in valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError("scaffold_sparsity must be between 0.0 and 1.0")
        return v
    
    @model_validator(mode='before')
    @classmethod
    def validate_consistency(cls, values):
        """Ensure data consistency across fields."""
        growth_history = values.get('growth_history', [])
        total_iterations = values.get('total_iterations', 0)
        
        if len(growth_history) != total_iterations:
            values['total_iterations'] = len(growth_history)
        
        return values


class TrainingExperiment(BaseExperimentSchema):
    """Schema for standard training experiments."""
    
    experiment_type: Literal["training_experiment"] = "training_experiment"
    config: ExperimentConfig = Field(..., description="Experiment configuration")
    architecture: NetworkArchitecture = Field(..., description="Network architecture")
    training_history: List[TrainingEpoch] = Field(default_factory=list, description="Training epoch history")
    final_performance: Optional[PerformanceMetrics] = Field(None, description="Final performance metrics")
    total_epochs: int = Field(default=0, ge=0, description="Total training epochs")
    
    @model_validator(mode='before')
    @classmethod
    def validate_training_consistency(cls, values):
        """Ensure training data consistency."""
        training_history = values.get('training_history', [])
        total_epochs = values.get('total_epochs', 0)
        
        if len(training_history) != total_epochs:
            values['total_epochs'] = len(training_history)
        
        return values


class TournamentExperiment(BaseExperimentSchema):
    """Schema for tournament-based growth experiments."""
    
    experiment_type: Literal["tournament_experiment"] = "tournament_experiment"
    config: ExperimentConfig = Field(..., description="Experiment configuration")
    seed_architecture: List[int] = Field(..., description="Initial network architecture")
    tournament_results: List[TournamentResult] = Field(default_factory=list, description="Tournament results")
    winner: Optional[TournamentResult] = Field(None, description="Winning strategy")
    growth_history: List[GrowthIteration] = Field(default_factory=list, description="Growth iteration history")
    final_performance: Optional[PerformanceMetrics] = Field(None, description="Final performance metrics")


class ComparativeExperiment(BaseExperimentSchema):
    """Schema for comparative experiments."""
    
    experiment_type: Literal["comparative_experiment"] = "comparative_experiment"
    experiments: Dict[str, Union[GrowthExperiment, TrainingExperiment, TournamentExperiment]] = Field(
        ..., description="Map of experiment names to results"
    )
    comparison_metrics: Dict[str, Any] = Field(default_factory=dict, description="Comparative analysis results")
    best_performer: Optional[str] = Field(None, description="Name of best performing experiment")


class ExperimentSummary(BaseExperimentSchema):
    """Schema for experiment summary data."""
    
    experiment_type: Literal["experiment_summary"] = "experiment_summary"
    total_experiments: int = Field(..., ge=0, description="Total number of experiments")
    best_accuracy: float = Field(..., ge=0.0, le=1.0, description="Best accuracy achieved")
    total_time: Optional[float] = Field(None, ge=0.0, description="Total experiment time")
    summary_statistics: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")


class ProfilingResultSchema(BaseModel):
    """Schema for the results of a single profiler."""
    summary_stats: Dict[str, Any] = Field(..., description="Summary statistics from the profiler")
    specialized_metrics: Dict[str, Any] = Field(default_factory=dict, description="Specialized metrics from the profiler")
    operations_count: int = Field(..., ge=0, description="Number of operations recorded by the profiler")


class ProfilingExperiment(BaseExperimentSchema):
    """Schema for profiling session experiments."""
    
    experiment_type: Literal["profiling_experiment"] = "profiling_experiment"
    session_duration: float = Field(..., ge=0, description="Total duration of the profiling session in seconds")
    profilers: Dict[str, ProfilingResultSchema] = Field(..., description="Results from each profiler in the session")

# Union type for all experiment schemas
ExperimentSchema = Union[
    GrowthExperiment,
    TrainingExperiment, 
    TournamentExperiment,
    ComparativeExperiment,
    ExperimentSummary,
    ProfilingExperiment
]


def validate_experiment_data(data: Dict[str, Any]) -> ExperimentSchema:
    """
    Validate experiment data against appropriate schema.
    
    Args:
        data: Raw experiment data dictionary
        
    Returns:
        Validated experiment schema instance
        
    Raises:
        ValueError: If data doesn't match any known schema
        ValidationError: If data fails validation
    """
    experiment_type = data.get('experiment_type', 'unknown')
    
    schema_map = {
        'growth_experiment': GrowthExperiment,
        'training_experiment': TrainingExperiment,
        'tournament_experiment': TournamentExperiment,
        'comparative_experiment': ComparativeExperiment,
        'experiment_summary': ExperimentSummary,
        'profiling_experiment': ProfilingExperiment
    }
    
    if experiment_type not in schema_map:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    schema_class = schema_map[experiment_type]
    return schema_class(**data)


def migrate_legacy_data(data: Dict[str, Any], target_version: str = "1.0") -> Dict[str, Any]:
    """
    Migrate legacy experiment data to current schema format.
    
    Args:
        data: Legacy experiment data
        target_version: Target schema version
        
    Returns:
        Migrated data dictionary
    """
    # Add schema version if missing
    if 'schema_version' not in data:
        data['schema_version'] = "0.1"
    
    current_version = data.get('schema_version', "0.1")
    
    # Migration chain
    if current_version == "0.1":
        data = _migrate_v01_to_v10(data)
        data['schema_version'] = "1.0"
    
    return data


def _migrate_v01_to_v10(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate from version 0.1 to 1.0."""
    
    # Add experiment_id if missing
    if 'experiment_id' not in data:
        timestamp = data.get('timestamp', datetime.now().isoformat())
        exp_type = data.get('experiment_type', 'unknown')
        data['experiment_id'] = f"{exp_type}_{timestamp}"
    
    # Ensure timestamp is present
    if 'timestamp' not in data:
        data['timestamp'] = datetime.now().isoformat()
    
    # Convert old architecture format
    if 'architecture' in data and isinstance(data['architecture'], list):
        if not isinstance(data['architecture'], dict):
            layers = data['architecture']
            data['architecture'] = {
                'layers': layers,
                'total_parameters': sum(layers) * 2,  # Rough estimate
                'total_connections': sum(layers) * 2,  # Rough estimate
                'sparsity': 0.02,  # Default sparsity
                'depth': len(layers)
            }
    
    # Convert old performance format
    if 'accuracy' in data and 'performance' not in data:
        data['performance'] = {
            'accuracy': data['accuracy'],
            'loss': data.get('loss')
        }
    
    return data


# Export all schemas and utilities
__all__ = [
    'BaseExperimentSchema',
    'NetworkArchitecture',
    'PerformanceMetrics',
    'ExtremaAnalysis',
    'GrowthAction',
    'TrainingEpoch',
    'GrowthIteration',
    'TournamentResult',
    'ExperimentConfig',
    'GrowthExperiment',
    'TrainingExperiment',
    'TournamentExperiment',
    'ComparativeExperiment',
    'ExperimentSummary',
    'ExperimentSchema',
    'validate_experiment_data',
    'migrate_legacy_data'
]
