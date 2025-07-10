# Neural Architecture Lab (NAL) Architecture Guide

## Overview

The Neural Architecture Lab (NAL) is a framework designed for the systematic, scientific, and reproducible exploration of neural network architectures and training strategies. It provides a high-level, hypothesis-driven interface that allows researchers to focus on the science of their experiments, while the lab handles the complex orchestration of execution, parallelization, and data management.

## Core Philosophy

The NAL is built on the principle of **inversion of control**. Instead of writing procedural scripts that manually create, train, and evaluate models, the researcher defines a `Hypothesis` and a `test_function`. The NAL then takes over, generating and running the necessary experiments to validate or reject the hypothesis.

## Key Components

The NAL is composed of several key components that work together to manage the experimental lifecycle:

1.  **`Hypothesis`**: The central abstraction of the NAL. A `Hypothesis` is a formal statement of a research question, including the parameter space to be explored and the criteria for success.

2.  **`Experiment`**: A single, concrete instance of a `Hypothesis` with a specific set of parameters. The NAL generates a series of `Experiment` objects from a single `Hypothesis`.

3.  **`ExperimentRunner` (`AdvancedExperimentRunner`)**: This is the workhorse of the NAL. It is responsible for executing a single `Experiment`. The `AdvancedExperimentRunner` is the default and provides a wealth of features, including:
    *   **Parallel Execution**: Manages a pool of worker processes to run experiments in parallel across multiple GPUs.
    *   **Resource Management**: Includes a `GPUMemoryManager` to optimize batch sizes and prevent CUDA OOM errors.
    *   **Data Handling**: Automatically loads the correct dataset for an experiment using the `data_factory`.
    *   **Fault Tolerance**: Catches errors within a single experiment, allowing the rest of the batch to continue.

4.  **`NeuralArchitectureLab`**: The main orchestrator. This class manages the entire lifecycle of a hypothesis test, from experiment generation to result analysis.

5.  **Analyzers (`StatisticalAnalyzer`, `InsightExtractor`)**: These components are responsible for processing the results of the experiments. They determine if a hypothesis is statistically significant, extract key findings, and can even suggest follow-up hypotheses.

## Relationship with `structure_net`

The NAL is the **user-facing research framework**, while `structure_net` is the **backend library** that provides the tools and components to build and evolve the networks.

*   **NAL orchestrates `structure_net`**: The NAL's `test_function`s are where you write the code that uses the `structure_net` library (e.g., `create_standard_network`, `ComposableEvolutionSystem`).
*   **`structure_net` is the engine**: It provides the layers, models, and evolution strategies that the NAL experiments with.

This separation of concerns is key to the project's design. It allows the `structure_net` library to focus on providing powerful, low-level tools, while the NAL focuses on providing a high-level, scientific framework for using those tools.
  The entire system is designed like a relay race, where data is passed in a standardized format from one component to the next. The key "baton" that carries the configuration
  is the `Hypothesis` object.


  Here is the step-by-step flow:

  Step 1: The Experiment Script (ultimate_stress_test_v2.py)


  This is the starting point. The script holds two separate configuration objects:
   * stress_config: Contains parameters specific to the tournament (dataset_name, generations, etc.).
   * lab_config: Contains parameters for the NAL framework itself (max_parallel_experiments, results_dir, etc.).


  The script creates the TournamentExecutor and passes the stress_config to it.



   1 # In main() of ultimate_stress_test_v2.py
   2 stress_config = get_default_stress_test_config()
   3 lab_config = get_default_lab_config()
   4 # ... overrides from args ...
   5
   6 # The executor gets the specific experiment protocol
   7 executor = TournamentExecutor(stress_config, lab)



  Step 2: The TournamentExecutor (The Translator)


  This is the most critical translation step. The TournamentExecutor's job is to translate the specific StressTestConfig into the generic Hypothesis format that the NAL can
  understand.


  It does this in its create_hypothesis method by packaging the parameters into the control_parameters dictionary.



    1 # In TournamentExecutor.create_hypothesis()
    2 def create_hypothesis(self, generation: int) -> Hypothesis:
    3     # ...
    4     return Hypothesis(
    5         # ...
    6         test_function=evaluate_competitor_task,
    7         parameter_space={'params': param_list},
    8
    9         # HERE is the translation!
   10         # The specific 'dataset_name' from StressTestConfig is put
   11         # into a generic dictionary for the NAL.
   12         control_parameters={
   13             'dataset': self.config.dataset_name,  # <--- The parameter is passed here
   14             'epochs': self.config.epochs_per_generation,
   15             'batch_size': self.config.batch_size_base,
   16             'enable_growth': self.config.enable_growth,
   17         },
   18         # ...
   19     )



  Step 3: Submitting the Work to the NeuralArchitectureLab


  The TournamentExecutor submits this fully-packaged Hypothesis object to the lab. At this point, the NAL has received its work order. It doesn't know or care that the
  parameters originally came from a StressTestConfig; it only sees the generic Hypothesis.



   1 # In TournamentExecutor.run_tournament()
   2 hypothesis = self.create_hypothesis(generation)
   3 self.lab.register_hypothesis(hypothesis)
   4 hypothesis_result = await self.lab.test_hypothesis(hypothesis.id) # <--- Hand-off to NAL


  Step 4: The NAL Generates Individual Experiments


  Inside lab.test_hypothesis(), the NAL looks at the Hypothesis and generates a list of individual Experiment objects to run.


  It combines the control_parameters (which contains our dataset_name) with each set of variables from the parameter_space.



    1 # In NeuralArchitectureLab.generate_experiments()
    2 def generate_experiments(self, hypothesis: Hypothesis) -> List[Experiment]:
    3     experiments = []
    4     param_combinations = self._generate_parameter_grid(hypothesis.parameter_space)
    5
    6     for i, params in enumerate(param_combinations):
    7         # HERE, the control parameters are merged with the specific
    8         # parameters for this single run.
    9         full_params = {**hypothesis.control_parameters, **params}
   10
   11         experiment = Experiment(
   12             id=f"{hypothesis.id}_exp_{i:03d}",
   13             # ...
   14             parameters=full_params, # <--- The combined dictionary is stored here
   15         )
   16         experiments.append(experiment)
   17
   18     return experiments



  Step 5: The ExperimentRunner Executes the Task


  The NAL passes the list of Experiment objects to its internal runner (e.g., AsyncExperimentRunner). The runner's job is to execute the test_function that was defined in the
  original Hypothesis.

  When it calls the function, it passes the experiment.parameters dictionary as the config argument.



   1 # Simplified logic inside an ExperimentRunner
   2 async def run_experiment_in_worker(experiment: Experiment, test_function: Callable):
   3     # ... (code to run in a separate process) ...
   4
   5     # The runner calls the user-provided function with the
   6     # parameters from the Experiment object.
   7     model, metrics = test_function(config=experiment.parameters) # <--- The final hand-off
   8
   9     # ... (code to package up the results) ...



  Step 6: The Final Destination (evaluate_competitor_task)


  The evaluate_competitor_task function finally receives the config dictionary. It can now access the dataset_name that started its journey way back in the StressTestConfig.



   1 # In ultimate_stress_test_v2.py
   2 def evaluate_competitor_task(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
   3     # The parameter has arrived at its destination!
   4     device = config.get('device', 'cpu')
   5     dataset_name = config.get('dataset', 'cifar10') # <--- And here it is used!
   6
   7     dataset = create_dataset(dataset_name, batch_size=config['batch_size'])
   8     # ... rest of the function ...


  Visual Summary of the Flow

  Here is a simple diagram of the journey:


  StressTestConfig -> TournamentExecutor -> Hypothesis -> NeuralArchitectureLab -> Experiment -> ExperimentRunner -> evaluate_competitor_task


  This clean separation of concerns is what makes the NAL a powerful and reusable framework.