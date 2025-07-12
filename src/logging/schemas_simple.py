#!/usr/bin/env python3
"""
Simplified Pydantic Schemas for Structure Net Logging

This is a simplified version without complex validators to ensure compatibility.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
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
    
    def generate_artifact_id(self) -> str:
        """Generate a unique artifact ID based on content hash."""
        content = self.model_dump_json(exclude={'timestamp'})
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class NetworkArchitecture(BaseModel):
    """Schema for network architecture information."""
    
    layers: List[int] = Field(..., description="Number of neurons in each layer")
    total_parameters: int = Field(..., ge=0, description="Total number of parameters")
    total_connections: int = Field(..., ge=0, description="Total number of connections")
    sparsity: float = Field(..., ge=0.0, le=1.0, description="Overall network sparsity")
    depth: int = Field(..., ge=1, description="Number of layers")


class PerformanceMetrics(BaseModel):
    """Schema for performance metrics."""
    
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy")
    loss: Optional[float] = Field(None, ge=0.0, description="Training/validation loss")
    precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Precision score")
    recall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Recall score")
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="F1 score")


class ExtremaAnalysis(BaseModel):
    """Schema for extrema detection analysis."""
    
    total_extrema: int = Field(..., ge=0, description="Total number of extrema neurons")
    extrema_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of extrema to total neurons")
    dead_neurons: Dict[str, List[int]] = Field(default_factory=dict, description="Dead neurons by layer")
    saturated_neurons: Dict[str, List[int]] = Field(default_factory=dict, description="Saturated neurons by layer")
    layer_health: Dict[str, float] = Field(default_factory=dict, description="Health score by layer")


class GrowthAction(BaseModel):
    """Schema for individual growth actions."""
    
    action_type: Literal["add_layer", "add_patch", "increase_density", "prune"] = Field(..., description="Type of growth action")
    layer_position: Optional[int] = Field(None, ge=0, description="Layer position for action")
    size: Optional[int] = Field(None, ge=1, description="Size of addition (neurons/connections)")
    reason: str = Field(..., description="Reason for taking this action")
    success: bool = Field(default=True, description="Whether action was successful")


class TrainingEpoch(BaseModel):
    """Schema for individual training epoch data."""
    
    epoch: int = Field(..., ge=0, description="Epoch number")
    train_loss: float = Field(..., ge=0.0, description="Training loss")
    train_accuracy: float = Field(..., ge=0.0, le=1.0, description="Training accuracy")
    val_loss: Optional[float] = Field(None, ge=0.0, description="Validation loss")
    val_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Validation accuracy")
    learning_rate: Optional[float] = Field(None, gt=0.0, description="Learning rate used")
    duration: Optional[float] = Field(None, ge=0.0, description="Epoch duration in seconds")


class GrowthIteration(BaseModel):
    """Schema for growth iteration data."""
    
    iteration: int = Field(..., ge=0, description="Growth iteration number")
    architecture: NetworkArchitecture = Field(..., description="Network architecture")
    performance: PerformanceMetrics = Field(..., description="Performance metrics")
    extrema_analysis: Optional[ExtremaAnalysis] = Field(None, description="Extrema analysis results")
    growth_actions: List[GrowthAction] = Field(default_factory=list, description="Growth actions taken")
    growth_occurred: bool = Field(default=False, description="Whether growth occurred")
    credits: Optional[float] = Field(None, ge=0.0, description="Credits in growth economy")


class TournamentResult(BaseModel):
    """Schema for tournament-based growth results."""
    
    strategy_name: str = Field(..., description="Name of the growth strategy")
    improvement: float = Field(..., description="Performance improvement achieved")
    final_accuracy: float = Field(..., ge=0.0, le=1.0, description="Final accuracy achieved")
    execution_time: Optional[float] = Field(None, ge=0.0, description="Time taken to execute strategy")
    success: bool = Field(default=True, description="Whether strategy executed successfully")


class ExperimentConfig(BaseModel):
    """Schema for experiment configuration."""
    
    dataset: str = Field(..., description="Dataset used")
    batch_size: int = Field(..., gt=0, description="Batch size")
    learning_rate: float = Field(..., gt=0.0, description="Learning rate")
    max_epochs: int = Field(..., gt=0, description="Maximum epochs")
    device: str = Field(..., description="Device used (cpu/cuda)")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    target_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Target accuracy")


class GrowthExperiment(BaseExperimentSchema):
    """Schema for growth-based experiments."""
    
    experiment_type: Literal["growth_experiment"] = "growth_experiment"
    config: ExperimentConfig = Field(..., description="Experiment configuration")
    seed_architecture: List[int] = Field(..., description="Initial network architecture")
    scaffold_sparsity: float = Field(..., ge=0.0, le=1.0, description="Initial scaffold sparsity")
    growth_history: List[GrowthIteration] = Field(default_factory=list, description="Growth iteration history")
    final_performance: Optional[PerformanceMetrics] = Field(None, description="Final performance metrics")
    total_iterations: int = Field(default=0, ge=0, description="Total growth iterations")


class TrainingExperiment(BaseExperimentSchema):
    """Schema for standard training experiments."""
    
    experiment_type: Literal["training_experiment"] = "training_experiment"
    config: ExperimentConfig = Field(..., description="Experiment configuration")
    architecture: NetworkArchitecture = Field(..., description="Network architecture")
    training_history: List[TrainingEpoch] = Field(default_factory=list, description="Training epoch history")
    final_performance: Optional[PerformanceMetrics] = Field(None, description="Final performance metrics")
    total_epochs: int = Field(default=0, ge=0, description="Total training epochs")


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
    'validate_experiment_data'
]
