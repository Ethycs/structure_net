#!/usr/bin/env python3
"""
Component-Based Schemas for Structure Net Logging — v2

Redesigned composition system grounded in the real component hierarchy
(core/interfaces.py: IAnalyzer, IStrategy, IEvolver, ITrainer, IMetric,
IScheduler, IOrchestrator).

Key changes from v1:
- Flexible: only ModelSpec is required; everything else is zero-or-more.
- Categories match the actual interface types.
- ComponentSpec resolves to real classes via the CompositionRegistry.
- No ``interactions`` field (unimplemented concept removed).

Backward-compatible deprecated aliases are provided at the bottom.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import json
import hashlib
import warnings


# ============================================================================
# COMPONENT SPEC (universal slot for any registry-backed component)
# ============================================================================

class ComponentSpec(BaseModel):
    """Specification for a single component in a composition.

    ``component_type`` must be one of the categories known to the
    CompositionRegistry: analyzer, strategy, evolver, trainer, metric,
    scheduler, orchestrator.

    ``component_name`` is the registry key (e.g. "extrema", "hybrid_growth").
    """

    component_type: str = Field(
        ..., description="Registry category (analyzer, strategy, evolver, ...)"
    )
    component_name: str = Field(
        ..., description="Registry key within the category"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs passed to the component constructor"
    )

    @field_validator("component_type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        valid = {"analyzer", "strategy", "evolver", "trainer", "metric", "scheduler", "orchestrator"}
        if v not in valid:
            raise ValueError(f"component_type must be one of {sorted(valid)}, got '{v}'")
        return v

    class Config:
        extra = "forbid"


# ============================================================================
# MODEL SPEC
# ============================================================================

class ModelSpec(BaseModel):
    """Specification for the neural network model (always required)."""

    factory_name: str = Field(
        ..., description="Model factory name in the registry (standard, extrema_aware, evolvable)"
    )
    architecture: List[int] = Field(..., description="Layer sizes [input, hidden..., output]")
    sparsity: float = Field(default=0.0, ge=0.0, le=1.0, description="Network sparsity")
    config: Dict[str, Any] = Field(default_factory=dict, description="Extra factory kwargs")

    @field_validator("architecture")
    @classmethod
    def _validate_arch(cls, v: List[int]) -> List[int]:
        if len(v) < 2:
            raise ValueError("Architecture must have at least input and output layers")
        if any(s <= 0 for s in v):
            raise ValueError("All layer sizes must be positive")
        return v

    class Config:
        extra = "forbid"


# ============================================================================
# TRAINING SPEC
# ============================================================================

class TrainingSpec(BaseModel):
    """Training loop parameters."""

    epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=128, gt=0)
    learning_rate: float = Field(default=0.001, gt=0.0)
    dataset: str = Field(default="cifar10")
    optimizer: str = Field(default="adam")
    config: Dict[str, Any] = Field(default_factory=dict, description="Extra training kwargs")

    @field_validator("optimizer")
    @classmethod
    def _validate_optimizer(cls, v: str) -> str:
        valid = {"adam", "sgd", "adamw", "rmsprop", "custom"}
        if v.lower() not in valid:
            raise ValueError(f"Optimizer must be one of {sorted(valid)}, got '{v}'")
        return v.lower()

    class Config:
        extra = "forbid"


# ============================================================================
# HYPOTHESIS SPEC (optional NAL-level metadata)
# ============================================================================

class HypothesisSpec(BaseModel):
    """NAL-level metadata for hypothesis-driven experiments."""

    hypothesis: str = Field(..., description="Hypothesis being tested")
    success_criteria: Dict[str, float] = Field(..., description="Metric thresholds for success")
    statistical_tests: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("hypothesis")
    @classmethod
    def _validate_hypothesis(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("Hypothesis must be substantive (>10 chars)")
        return v

    class Config:
        extra = "forbid"


# ============================================================================
# EXPERIMENT COMPOSITION (the main schema)
# ============================================================================

class ExperimentComposition(BaseModel):
    """
    Flexible experiment composition.

    Only ``model`` is required.  All component lists default to empty,
    allowing analysis-only, training-only, or full evolution setups.
    """

    composition_id: str = Field(..., description="Unique composition identifier")
    name: str = Field(..., description="Human-readable name")

    # Required
    model: ModelSpec

    # Optional component lists (zero or more of each)
    analyzers: List[ComponentSpec] = Field(default_factory=list)
    strategies: List[ComponentSpec] = Field(default_factory=list)
    evolvers: List[ComponentSpec] = Field(default_factory=list)
    trainers: List[ComponentSpec] = Field(default_factory=list)
    metrics: List[ComponentSpec] = Field(default_factory=list)
    schedulers: List[ComponentSpec] = Field(default_factory=list)

    # Training loop parameters (optional)
    training: Optional[TrainingSpec] = None

    # NAL-level metadata (optional)
    hypothesis: Optional[HypothesisSpec] = None

    # Template provenance
    template_name: Optional[str] = None
    template_version: Optional[str] = None

    @model_validator(mode="after")
    def _validate_component_types(self):
        """Ensure each ComponentSpec has the right type for its list."""
        _expected = {
            "analyzers": "analyzer",
            "strategies": "strategy",
            "evolvers": "evolver",
            "trainers": "trainer",
            "metrics": "metric",
            "schedulers": "scheduler",
        }
        for field_name, expected_type in _expected.items():
            for spec in getattr(self, field_name):
                if spec.component_type != expected_type:
                    raise ValueError(
                        f"ComponentSpec in '{field_name}' has type '{spec.component_type}', "
                        f"expected '{expected_type}'"
                    )
        return self

    def generate_hash(self) -> str:
        """Generate unique hash for this composition."""
        content = json.dumps(
            {
                "model": self.model.factory_name,
                "architecture": self.model.architecture,
                "analyzers": [s.component_name for s in self.analyzers],
                "strategies": [s.component_name for s in self.strategies],
                "evolvers": [s.component_name for s in self.evolvers],
                "trainers": [s.component_name for s in self.trainers],
                "metrics": [s.component_name for s in self.metrics],
                "schedulers": [s.component_name for s in self.schedulers],
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_all_component_specs(self) -> List[ComponentSpec]:
        """Return a flat list of all ComponentSpecs in this composition."""
        return (
            list(self.analyzers)
            + list(self.strategies)
            + list(self.evolvers)
            + list(self.trainers)
            + list(self.metrics)
            + list(self.schedulers)
        )


# ============================================================================
# EXECUTION SCHEMAS (What Happened)
# ============================================================================

class IterationData(BaseModel):
    """Data from one training iteration."""

    iteration: int = Field(..., ge=0, description="Iteration number")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Per-component outputs (all optional for flexibility)
    metric_outputs: Dict[str, Any] = Field(default_factory=dict, description="Metrics computed")
    evolver_actions: List[str] = Field(default_factory=list, description="Actions taken")
    model_changes: Dict[str, Any] = Field(default_factory=dict, description="Architecture changes")
    trainer_metrics: Dict[str, Any] = Field(default_factory=dict, description="Training metrics")

    # Performance snapshot
    accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    loss: Optional[float] = Field(default=None, ge=0.0)


class ExperimentExecution(BaseModel):
    """Records what happened when a composition was executed."""

    execution_id: str = Field(..., description="Unique execution identifier")
    composition: ExperimentComposition = Field(..., description="What was executed")

    # Execution metadata
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: Literal["running", "completed", "failed", "cancelled"] = "running"

    # Iteration log
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
# TEMPLATE SYSTEM
# ============================================================================

class ParameterSpec(BaseModel):
    """Specification for a customizable parameter."""

    name: str
    type: str  # "float", "int", "str", "bool", "list"
    default: Any
    description: str
    constraints: Optional[Dict[str, Any]] = None


class ExperimentTemplate(BaseModel):
    """Pre-built composition with customizable parameters."""

    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="What this template does")
    category: str = Field(..., description="Template category")

    # The pre-configured composition
    composition: ExperimentComposition

    # Customizable parameters
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)

    # Metadata
    examples: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)

    def instantiate(self, **kwargs) -> ExperimentComposition:
        """Create a composition instance with custom parameters."""
        composition = self.composition.model_copy(deep=True)
        for param_name, param_value in kwargs.items():
            if param_name not in self.parameters:
                raise ValueError(f"Unknown parameter: {param_name}")
            self._apply_parameter(composition, param_name, param_value)
        return composition

    def _apply_parameter(self, composition: ExperimentComposition, param_path: str, value: Any):
        parts = param_path.split(".")
        obj = composition
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_experiment_data(data: Dict[str, Any]) -> Union[ExperimentExecution, ExperimentTemplate]:
    """Validate experiment data and return appropriate schema object."""
    if "execution_id" in data:
        return ExperimentExecution(**data)
    elif "template_id" in data:
        return ExperimentTemplate(**data)
    else:
        raise ValueError("Data must be either an execution or template")


def validate_component_compatibility(composition: ExperimentComposition) -> List[str]:
    """Check composition for potential issues. Returns warnings."""
    warn_list: List[str] = []

    if not composition.trainers and composition.training:
        warn_list.append("TrainingSpec provided but no trainer components specified")

    if composition.strategies and not composition.analyzers:
        warn_list.append("Strategies present without analyzers — strategies may lack input data")

    if composition.evolvers and not composition.strategies:
        warn_list.append("Evolvers present without strategies — evolvers need plans to execute")

    return warn_list


# ============================================================================
# PRE-BUILT TEMPLATES (v2 format)
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
            model=ModelSpec(
                factory_name="evolvable",
                architecture=[784, 512, 256, 10],
                sparsity=0.95,
                config={"activation": "relu"},
            ),
            analyzers=[
                ComponentSpec(component_type="analyzer", component_name="extrema"),
            ],
            strategies=[
                ComponentSpec(component_type="strategy", component_name="extrema_growth"),
            ],
            training=TrainingSpec(
                epochs=10,
                batch_size=128,
                learning_rate=0.001,
                optimizer="adam",
            ),
            hypothesis=HypothesisSpec(
                hypothesis="Genetic evolution improves network efficiency",
                statistical_tests=["t_test", "effect_size"],
                success_criteria={"efficiency_gain": 0.2, "accuracy_maintained": 0.95},
            ),
        ),
        parameters={
            "training.learning_rate": ParameterSpec(
                name="Learning Rate",
                type="float",
                default=0.001,
                description="Base learning rate",
                constraints={"min": 1e-5, "max": 0.1},
            ),
        },
    ),
    "smart_growth": ExperimentTemplate(
        template_id="tpl_smart_growth_001",
        name="Smart Growth",
        description="Extrema-driven network growth with topological analysis",
        category="growth",
        composition=ExperimentComposition(
            composition_id="comp_smart_growth_001",
            name="Smart Growth Experiment",
            model=ModelSpec(
                factory_name="standard",
                architecture=[784, 128, 10],
                sparsity=0.9,
                config={"activation": "relu", "sparse_init": True},
            ),
            analyzers=[
                ComponentSpec(
                    component_type="analyzer",
                    component_name="extrema",
                    config={"dead_threshold": 0.01},
                ),
                ComponentSpec(component_type="analyzer", component_name="information_flow"),
            ],
            strategies=[
                ComponentSpec(
                    component_type="strategy",
                    component_name="extrema_growth",
                    config={"extrema_threshold": 0.3},
                ),
            ],
            training=TrainingSpec(
                epochs=50, batch_size=128, learning_rate=0.001, optimizer="adam"
            ),
            hypothesis=HypothesisSpec(
                hypothesis="Smart growth achieves better accuracy than fixed architecture",
                statistical_tests=["t_test", "effect_size", "wilcoxon"],
                success_criteria={"accuracy_improvement": 0.05, "efficiency": 0.8},
            ),
        ),
        parameters={
            "training.epochs": ParameterSpec(
                name="Epochs",
                type="int",
                default=50,
                description="Number of training epochs",
                constraints={"min": 1, "max": 500},
            ),
        },
        tags=["growth", "sparse", "extrema"],
    ),
    "geometric_analysis": ExperimentTemplate(
        template_id="tpl_geometric_001",
        name="Geometric Analysis",
        description="Curvature-based network analysis and optimization",
        category="geometric",
        composition=ExperimentComposition(
            composition_id="comp_geometric_001",
            name="Geometric Deep Learning Experiment",
            model=ModelSpec(
                factory_name="standard",
                architecture=[784, 512, 256, 10],
                sparsity=0.0,
            ),
            analyzers=[
                ComponentSpec(component_type="analyzer", component_name="topological"),
                ComponentSpec(component_type="analyzer", component_name="homological"),
            ],
            training=TrainingSpec(
                epochs=30, batch_size=64, learning_rate=0.01, optimizer="custom"
            ),
            hypothesis=HypothesisSpec(
                hypothesis="Geometric constraints improve generalization over unconstrained training",
                statistical_tests=["t_test"],
                success_criteria={"generalization_gap": 0.05},
            ),
        ),
        tags=["geometric", "curvature"],
    ),
    "pruning_optimization": ExperimentTemplate(
        template_id="tpl_pruning_001",
        name="Pruning Optimization",
        description="Iterative magnitude pruning with accuracy recovery",
        category="pruning",
        composition=ExperimentComposition(
            composition_id="comp_pruning_001",
            name="Iterative Pruning Experiment",
            model=ModelSpec(
                factory_name="standard",
                architecture=[784, 1024, 512, 256, 10],
                sparsity=0.0,
            ),
            analyzers=[
                ComponentSpec(component_type="analyzer", component_name="sensitivity"),
            ],
            training=TrainingSpec(
                epochs=20, batch_size=64, learning_rate=0.0005, optimizer="adamw"
            ),
            hypothesis=HypothesisSpec(
                hypothesis="Iterative pruning achieves 90%+ sparsity with less than 2% accuracy loss",
                statistical_tests=["t_test", "bootstrap_ci"],
                success_criteria={"final_sparsity": 0.9, "accuracy_retained": 0.98},
            ),
        ),
        tags=["pruning", "sparsity", "compression"],
    ),
}


# ============================================================================
# DEPRECATED v1 ALIASES — will be removed in a future release
# ============================================================================

class ComponentSchema(BaseModel):
    """DEPRECATED: use ComponentSpec instead."""

    component_id: str = Field(..., description="Unique component identifier")
    component_version: str = Field(default="1.0")
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        extra = "forbid"
        populate_by_name = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} is deprecated. Use ComponentSpec / ModelSpec instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class MetricSchema(ComponentSchema):
    """DEPRECATED: use ComponentSpec(component_type='metric', ...) instead."""
    component_type: Literal["metric"] = "metric"
    metric_name: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)
    outputs: List[str] = Field(default_factory=list)
    requires_gradients: bool = False


class EvolverSchema(ComponentSchema):
    """DEPRECATED: use ComponentSpec(component_type='evolver', ...) instead."""
    component_type: Literal["evolver"] = "evolver"
    evolver_name: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    preserves_function: bool = True


class ModelSchema(ComponentSchema):
    """DEPRECATED: use ModelSpec instead."""
    component_type: Literal["model"] = "model"
    model_name: str = ""
    architecture: List[int] = Field(default_factory=lambda: [1, 1])
    total_parameters: int = 0
    sparsity: float = 0.0
    config: Dict[str, Any] = Field(default_factory=dict)
    supports_growth: bool = True


class TrainerSchema(ComponentSchema):
    """DEPRECATED: use TrainingSpec instead."""
    component_type: Literal["trainer"] = "trainer"
    trainer_name: str = ""
    optimizer: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 128
    config: Dict[str, Any] = Field(default_factory=dict)


class NALSchema(ComponentSchema):
    """DEPRECATED: use HypothesisSpec instead."""
    component_type: Literal["nal"] = "nal"
    hypothesis: str = "placeholder hypothesis"
    statistical_tests: List[str] = Field(default_factory=list)
    success_criteria: Dict[str, float] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
