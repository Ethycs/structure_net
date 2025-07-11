#!/usr/bin/env python3
"""
Component-Based Schemas for Structure Net Logging

Following the Photoshop-like layer system with 5 core components:
1. Metric (Analysis layer)
2. Evolver (Modification layer)  
3. Model (Architecture layer)
4. Trainer (Learning layer)
5. NAL (Orchestration layer)

Each component has its own schema, and experiments are compositions of these components.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import json
import hashlib


# ============================================================================
# BASE COMPONENT SCHEMAS (Layer Types)
# ============================================================================

class ComponentSchema(BaseModel):
    """Base schema for all components."""
    
    component_id: str = Field(..., description="Unique component identifier")
    component_version: str = Field(default="1.0", description="Component version")
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        extra = "forbid"  # Strict validation - no extra fields
        populate_by_name = True


class MetricSchema(ComponentSchema):
    """Schema for any metric component (analysis layer)."""
    
    component_type: Literal["metric"] = "metric"
    metric_name: str = Field(..., description="Type of metric (extrema_analysis, curvature, etc.)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Metric configuration")
    outputs: List[str] = Field(..., description="What metrics this produces")
    requires_gradients: bool = Field(default=False, description="Whether this needs gradients")
    
    @field_validator('outputs')
    @classmethod
    def validate_outputs(cls, v):
        if not v:
            raise ValueError("Metric must produce at least one output")
        return v


class EvolverSchema(ComponentSchema):
    """Schema for any evolver component (modification layer)."""
    
    component_type: Literal["evolver"] = "evolver"
    evolver_name: str = Field(..., description="Type of evolver (genetic, smart_growth, etc.)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Evolver configuration")
    inputs: List[str] = Field(..., description="What metrics it needs")
    outputs: List[str] = Field(..., description="What changes it makes")
    preserves_function: bool = Field(default=True, description="Whether it preserves network function")
    
    @field_validator('inputs')
    @classmethod
    def validate_inputs(cls, v):
        if not v:
            raise ValueError("Evolver must consume at least one input")
        return v


class ModelSchema(ComponentSchema):
    """Schema for any model component (architecture layer)."""
    
    component_type: Literal["model"] = "model"
    model_name: str = Field(..., description="Type of model (sparse_mlp, fiber_bundle, etc.)")
    architecture: List[int] = Field(..., description="Layer sizes")
    total_parameters: int = Field(..., ge=0, description="Total parameters")
    sparsity: float = Field(..., ge=0.0, le=1.0, description="Network sparsity")
    config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")
    supports_growth: bool = Field(default=True, description="Whether model can grow")
    
    @field_validator('architecture')
    @classmethod
    def validate_architecture(cls, v):
        if len(v) < 2:
            raise ValueError("Architecture must have at least input and output layers")
        if any(size <= 0 for size in v):
            raise ValueError("All layer sizes must be positive")
        return v


class TrainerSchema(ComponentSchema):
    """Schema for any trainer component (learning layer)."""
    
    component_type: Literal["trainer"] = "trainer"
    trainer_name: str = Field(..., description="Type of trainer (standard, geometric_constrained, etc.)")
    optimizer: str = Field(..., description="Optimizer type")
    learning_rate: float = Field(..., gt=0.0, description="Base learning rate")
    batch_size: int = Field(..., gt=0, description="Batch size")
    config: Dict[str, Any] = Field(default_factory=dict, description="Trainer configuration")
    
    @field_validator('optimizer')
    @classmethod
    def validate_optimizer(cls, v):
        valid_optimizers = ['adam', 'sgd', 'adamw', 'rmsprop', 'custom']
        if v.lower() not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of {valid_optimizers}")
        return v.lower()


class NALSchema(ComponentSchema):
    """Schema for NAL orchestration (experiment layer)."""
    
    component_type: Literal["nal"] = "nal"
    hypothesis: str = Field(..., description="Hypothesis being tested")
    statistical_tests: List[str] = Field(default_factory=list, description="Statistical tests to run")
    success_criteria: Dict[str, float] = Field(..., description="Success metrics and thresholds")
    config: Dict[str, Any] = Field(default_factory=dict, description="NAL configuration")
    
    @field_validator('hypothesis')
    @classmethod
    def validate_hypothesis(cls, v):
        if len(v) < 10:
            raise ValueError("Hypothesis must be substantive (>10 chars)")
        return v


# ============================================================================
# COMPOSITION SCHEMA (Layer Stack)
# ============================================================================

class ExperimentComposition(BaseModel):
    """Defines how components are stacked together (like Photoshop layers)."""
    
    composition_id: str = Field(..., description="Unique composition identifier")
    name: str = Field(..., description="Human-readable name")
    
    # The 5-component stack
    metric: MetricSchema
    evolver: EvolverSchema
    model: ModelSchema
    trainer: TrainerSchema
    nal: NALSchema
    
    # Component interactions (like blend modes)
    interactions: Dict[str, Any] = Field(default_factory=dict, description="How components interact")
    
    # Template info (like Smart Objects)
    template_name: Optional[str] = None
    template_version: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_component_compatibility(self):
        """Ensure components can work together."""
        # Check that evolver inputs are satisfied by metric outputs
        missing_inputs = set(self.evolver.inputs) - set(self.metric.outputs)
        if missing_inputs:
            raise ValueError(f"Evolver needs inputs that metric doesn't provide: {missing_inputs}")
        
        # Check model supports required evolver operations
        if 'add_layers' in self.evolver.outputs and not self.model.supports_growth:
            raise ValueError("Evolver wants to add layers but model doesn't support growth")
        
        return self
    
    def generate_hash(self) -> str:
        """Generate unique hash for this composition."""
        content = json.dumps({
            'metric': self.metric.metric_name,
            'evolver': self.evolver.evolver_name,
            'model': self.model.model_name,
            'trainer': self.trainer.trainer_name,
            'nal': self.nal.hypothesis
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ============================================================================
# EXECUTION SCHEMAS (What Happened)
# ============================================================================

class IterationData(BaseModel):
    """Data from one iteration (like one frame in animation)."""
    
    iteration: int = Field(..., ge=0, description="Iteration number")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # What each component did this iteration
    metric_outputs: Dict[str, Any] = Field(..., description="Metrics computed")
    evolver_actions: List[str] = Field(default_factory=list, description="Actions taken")
    model_changes: Dict[str, Any] = Field(default_factory=dict, description="Architecture changes")
    trainer_metrics: Dict[str, Any] = Field(..., description="Training metrics")
    nal_decisions: Dict[str, Any] = Field(default_factory=dict, description="NAL decisions")
    
    # Performance snapshot
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Current accuracy")
    loss: float = Field(..., ge=0.0, description="Current loss")


class ExperimentExecution(BaseModel):
    """Records what happened when the composition was executed."""
    
    execution_id: str = Field(..., description="Unique execution identifier")
    composition: ExperimentComposition = Field(..., description="What was executed")
    
    # Execution metadata
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    
    # Layer-by-layer execution log
    iteration_log: List[IterationData] = Field(default_factory=list)
    
    # Final results
    final_metrics: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None
    error: Optional[str] = None
    
    # Resource usage
    peak_memory_gb: Optional[float] = None
    total_gpu_hours: Optional[float] = None
    
    def add_iteration(self, iteration_data: IterationData):
        """Add iteration data maintaining order."""
        self.iteration_log.append(iteration_data)
        self.iteration_log.sort(key=lambda x: x.iteration)
    
    def finalize(self, status: str = "completed", error: str = None):
        """Mark execution as complete."""
        self.completed_at = datetime.now()
        self.status = status
        self.error = error
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()


# ============================================================================
# TEMPLATE SYSTEM (Smart Objects)
# ============================================================================

class ParameterSpec(BaseModel):
    """Specification for a customizable parameter."""
    
    name: str
    type: str  # "float", "int", "str", "bool", "list"
    default: Any
    description: str
    constraints: Optional[Dict[str, Any]] = None  # min, max, choices, etc.


class ExperimentTemplate(BaseModel):
    """Pre-built component compositions (like Photoshop templates)."""
    
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="What this template does")
    category: str = Field(..., description="Template category")
    
    # The pre-configured composition
    composition: ExperimentComposition
    
    # Customizable parameters (like template variables)
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    
    # Usage examples
    examples: List[str] = Field(default_factory=list)
    
    # Template metadata
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    
    def instantiate(self, **kwargs) -> ExperimentComposition:
        """Create a composition instance with custom parameters."""
        composition = self.composition.model_copy(deep=True)
        
        # Apply parameter customizations
        for param_name, param_value in kwargs.items():
            if param_name not in self.parameters:
                raise ValueError(f"Unknown parameter: {param_name}")
            
            # Apply parameter based on its path (e.g., "metric.config.threshold")
            self._apply_parameter(composition, param_name, param_value)
        
        return composition
    
    def _apply_parameter(self, composition: ExperimentComposition, 
                        param_path: str, value: Any):
        """Apply a parameter value to the composition."""
        # Simple implementation - could be enhanced
        parts = param_path.split('.')
        obj = composition
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_experiment_data(data: Dict[str, Any]) -> Union[ExperimentExecution, ExperimentTemplate]:
    """Validate experiment data and return appropriate schema object."""
    
    if 'execution_id' in data:
        return ExperimentExecution(**data)
    elif 'template_id' in data:
        return ExperimentTemplate(**data)
    else:
        raise ValueError("Data must be either an execution or template")


def validate_component_compatibility(composition: ExperimentComposition) -> List[str]:
    """Check component compatibility and return any warnings."""
    warnings = []
    
    # Check metric -> evolver flow
    unused_metrics = set(composition.metric.outputs) - set(composition.evolver.inputs)
    if unused_metrics:
        warnings.append(f"Metric produces unused outputs: {unused_metrics}")
    
    # Check evolver -> model compatibility
    if 'prune_connections' in composition.evolver.outputs and composition.model.sparsity >= 0.99:
        warnings.append("Model is already highly sparse, pruning may fail")
    
    return warnings


# ============================================================================
# PRE-BUILT TEMPLATES
# ============================================================================

STANDARD_TEMPLATES = {
    "architecture_evolution": ExperimentTemplate(
        template_id="tpl_arch_evo_001",
        name="Architecture Evolution",
        description="Standard genetic evolution of sparse networks",
        category="evolution",
        composition=ExperimentComposition(
            composition_id="comp_arch_evo_001",
            name="Standard Architecture Evolution",
            metric=MetricSchema(
                component_id="metric_fitness_001",
                metric_name="fitness",
                outputs=["accuracy", "efficiency", "fitness_score"],
                config={"efficiency_weight": 0.3}
            ),
            evolver=EvolverSchema(
                component_id="evolver_genetic_001",
                evolver_name="genetic",
                inputs=["fitness_score"],
                outputs=["mutate_architecture", "crossover"],
                config={"mutation_rate": 0.1, "population_size": 20}
            ),
            model=ModelSchema(
                component_id="model_sparse_001",
                model_name="sparse_mlp",
                architecture=[784, 512, 256, 10],
                total_parameters=550000,
                sparsity=0.95,
                config={"activation": "relu"}
            ),
            trainer=TrainerSchema(
                component_id="trainer_standard_001",
                trainer_name="standard",
                optimizer="adam",
                learning_rate=0.001,
                batch_size=128,
                config={"epochs": 10}
            ),
            nal=NALSchema(
                component_id="nal_evo_001",
                hypothesis="Genetic evolution improves network efficiency",
                statistical_tests=["t_test", "effect_size"],
                success_criteria={"efficiency_gain": 0.2, "accuracy_maintained": 0.95}
            )
        ),
        parameters={
            "metric.config.efficiency_weight": ParameterSpec(
                name="Efficiency Weight",
                type="float",
                default=0.3,
                description="Weight given to efficiency vs accuracy",
                constraints={"min": 0.0, "max": 1.0}
            ),
            "evolver.config.mutation_rate": ParameterSpec(
                name="Mutation Rate",
                type="float", 
                default=0.1,
                description="Probability of mutation",
                constraints={"min": 0.0, "max": 0.5}
            )
        }
    )
}