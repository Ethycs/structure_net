# Neural Architecture Lab (NAL)

The Neural Architecture Lab (NAL) is a Python framework for running systematic, hypothesis-driven experiments on neural network architectures and training procedures. It is the primary tool for research and development in the `structure_net` project.

## Core Concepts

- **Hypothesis-Driven**: Instead of writing ad-hoc scripts, you define a formal `Hypothesis` with a clear research question, prediction, and parameter space.
- **Automated Orchestration**: The NAL automatically generates and runs the experiments needed to test a hypothesis, parallelizing across multiple GPUs.
- **Reproducibility**: By structuring experiments this way, the NAL ensures that your research is reproducible and easy to share.
- **Insight Extraction**: The NAL includes tools for statistical analysis and insight extraction, helping you draw meaningful conclusions from your experimental results.

## Quick Start

1.  **Define a `test_function`**: This is a Python function that takes a configuration dictionary and returns a trained model and a dictionary of metrics.

    ```python
    def my_test_function(config: dict) -> (object, dict):
        # ... create and train a model based on config ...
        return model, {'accuracy': 0.95}
    ```

2.  **Define a `Hypothesis`**:

    ```python
    from neural_architecture_lab import Hypothesis, HypothesisCategory

    my_hypothesis = Hypothesis(
        id="my_first_test",
        name="My First Test",
        description="A simple example hypothesis.",
        category=HypothesisCategory.TRAINING,
        question="Does a higher learning rate improve accuracy?",
        prediction="Yes, up to a point.",
        test_function=my_test_function,
        parameter_space={'learning_rate': [0.001, 0.01, 0.1]},
        control_parameters={'epochs': 10},
        success_metrics={'accuracy': 0.9}
    )
    ```

3.  **Run the Lab**:

    ```python
    from neural_architecture_lab import NeuralArchitectureLab, LabConfig

    lab_config = LabConfig()
    lab = NeuralArchitectureLab(lab_config)
    lab.register_hypothesis(my_hypothesis)

    # Run the experiments
    import asyncio
    results = asyncio.run(lab.run_all_hypotheses())
    ```

For more detailed examples, see the scripts in the `examples/` directory of the main project.
