"""
Experiment Presets

Named preset functions that return fully-formed ExperimentComposition objects,
following the pattern established by ``profiling/factory.py``::

    from structure_net.experiment_presets import create_evolution_preset
    composition = create_evolution_preset()

Each preset can be resolved directly::

    from structure_net.core.component_registry import get_global_registry
    from structure_net.core.composition_resolver import CompositionResolver
    resolved = CompositionResolver(get_global_registry()).resolve(composition)
"""

from .logging.component_schemas import (
    ExperimentComposition,
    ComponentSpec,
    ModelSpec,
    TrainingSpec,
    HypothesisSpec,
)


def create_evolution_preset(
    architecture: list = None,
    sparsity: float = 0.95,
    epochs: int = 50,
    dataset: str = "mnist",
) -> ExperimentComposition:
    """Standard genetic evolution of sparse networks."""
    return ExperimentComposition(
        composition_id="preset_evolution",
        name="Standard Architecture Evolution",
        model=ModelSpec(
            factory_name="evolvable",
            architecture=architecture or [784, 512, 256, 10],
            sparsity=sparsity,
        ),
        analyzers=[
            ComponentSpec(component_type="analyzer", component_name="extrema"),
            ComponentSpec(component_type="analyzer", component_name="activity"),
        ],
        strategies=[
            ComponentSpec(component_type="strategy", component_name="extrema_growth"),
        ],
        training=TrainingSpec(
            epochs=epochs,
            batch_size=128,
            learning_rate=0.001,
            dataset=dataset,
        ),
        hypothesis=HypothesisSpec(
            hypothesis="Evolutionary architecture search improves network efficiency",
            success_criteria={"accuracy": 0.90},
        ),
    )


def create_smart_growth_preset(
    architecture: list = None,
    sparsity: float = 0.9,
    epochs: int = 50,
    dataset: str = "mnist",
) -> ExperimentComposition:
    """Extrema-driven network growth with information flow analysis."""
    return ExperimentComposition(
        composition_id="preset_smart_growth",
        name="Smart Growth Experiment",
        model=ModelSpec(
            factory_name="standard",
            architecture=architecture or [784, 128, 10],
            sparsity=sparsity,
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
            epochs=epochs,
            batch_size=128,
            learning_rate=0.001,
            dataset=dataset,
        ),
        hypothesis=HypothesisSpec(
            hypothesis="Smart growth achieves better accuracy than fixed architecture",
            success_criteria={"accuracy": 0.85},
        ),
    )


def create_geometric_analysis_preset(
    architecture: list = None,
    epochs: int = 30,
    dataset: str = "mnist",
) -> ExperimentComposition:
    """Topological and homological analysis of network structure."""
    return ExperimentComposition(
        composition_id="preset_geometric",
        name="Geometric Analysis Experiment",
        model=ModelSpec(
            factory_name="standard",
            architecture=architecture or [784, 512, 256, 10],
            sparsity=0.0,
        ),
        analyzers=[
            ComponentSpec(component_type="analyzer", component_name="topological"),
            ComponentSpec(component_type="analyzer", component_name="homological"),
            ComponentSpec(component_type="analyzer", component_name="graph"),
        ],
        training=TrainingSpec(
            epochs=epochs,
            batch_size=64,
            learning_rate=0.01,
            dataset=dataset,
            optimizer="adam",
        ),
        hypothesis=HypothesisSpec(
            hypothesis="Geometric constraints improve generalization over unconstrained training",
            success_criteria={"accuracy": 0.80},
        ),
    )


def create_pruning_preset(
    architecture: list = None,
    epochs: int = 20,
    dataset: str = "mnist",
) -> ExperimentComposition:
    """Iterative magnitude pruning with accuracy recovery."""
    return ExperimentComposition(
        composition_id="preset_pruning",
        name="Iterative Pruning Experiment",
        model=ModelSpec(
            factory_name="standard",
            architecture=architecture or [784, 1024, 512, 256, 10],
            sparsity=0.0,
        ),
        analyzers=[
            ComponentSpec(component_type="analyzer", component_name="sensitivity"),
        ],
        training=TrainingSpec(
            epochs=epochs,
            batch_size=64,
            learning_rate=0.0005,
            optimizer="adamw",
            dataset=dataset,
        ),
        hypothesis=HypothesisSpec(
            hypothesis="Iterative pruning achieves 90%+ sparsity with less than 2% accuracy loss",
            success_criteria={"accuracy": 0.90},
        ),
    )


def create_analysis_only_preset(
    architecture: list = None,
    sparsity: float = 0.5,
) -> ExperimentComposition:
    """
    Analysis-only composition — no training, no evolvers.

    Useful for post-hoc analysis of a pre-trained model.
    """
    return ExperimentComposition(
        composition_id="preset_analysis_only",
        name="Analysis-Only Experiment",
        model=ModelSpec(
            factory_name="standard",
            architecture=architecture or [784, 256, 128, 10],
            sparsity=sparsity,
        ),
        analyzers=[
            ComponentSpec(component_type="analyzer", component_name="extrema"),
            ComponentSpec(component_type="analyzer", component_name="graph"),
            ComponentSpec(component_type="analyzer", component_name="sensitivity"),
            ComponentSpec(component_type="analyzer", component_name="activity"),
        ],
    )


def create_hybrid_evolution_preset(
    architecture: list = None,
    sparsity: float = 0.8,
    epochs: int = 100,
    dataset: str = "cifar10",
) -> ExperimentComposition:
    """Full hybrid evolution with multiple analyzers and strategies."""
    return ExperimentComposition(
        composition_id="preset_hybrid_evolution",
        name="Hybrid Evolution Experiment",
        model=ModelSpec(
            factory_name="evolvable",
            architecture=architecture or [3072, 512, 256, 128, 10],
            sparsity=sparsity,
        ),
        analyzers=[
            ComponentSpec(component_type="analyzer", component_name="extrema"),
            ComponentSpec(component_type="analyzer", component_name="information_flow"),
            ComponentSpec(component_type="analyzer", component_name="performance_correlation"),
        ],
        strategies=[
            ComponentSpec(component_type="strategy", component_name="hybrid_growth"),
        ],
        training=TrainingSpec(
            epochs=epochs,
            batch_size=128,
            learning_rate=0.001,
            dataset=dataset,
        ),
        hypothesis=HypothesisSpec(
            hypothesis="Hybrid evolution with multiple analyzers outperforms single-strategy approaches",
            success_criteria={"accuracy": 0.70},
        ),
    )
