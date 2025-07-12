import torch
import torch.nn as nn
import time
import traceback
from typing import Dict, Any, Tuple

from src.neural_architecture_lab.core import Experiment, ExperimentResult
from src.structure_net.core.network_factory import create_standard_network
from src.data_factory import create_dataset
from src.structure_net.core.io_operations import load_model_seed

def evaluate_seed_task(experiment: Experiment, device_id: int) -> ExperimentResult:
    """
    NAL worker function for evaluating a single seed candidate.
    """
    config = experiment.parameters
    device = f'cuda:{device_id}' if torch.cuda.is_available() and device_id >= 0 else 'cpu'
    start_time = time.time()

    try:
        # 1. Create the network
        model = create_standard_network(
            architecture=config['architecture'],
            sparsity=config.get('sparsity', 0.02),
            device=device
        )
        model.to(device)

        # 2. Load the dataset
        dataset_name = config.get('dataset', 'cifar10')
        dataset = create_dataset(dataset_name, batch_size=config.get('batch_size', 128))
        train_loader = dataset['train_loader']
        test_loader = dataset['test_loader']

        # 3. Train for a few epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(config.get('epochs', 10)):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                if data.dim() > 2 and hasattr(model, 'layers') and model.layers and isinstance(model.layers[0], nn.Linear):
                    data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 4. Evaluate the model
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if data.dim() > 2 and hasattr(model, 'layers') and model.layers and isinstance(model.layers[0], nn.Linear):
                    data = data.view(data.size(0), -1)
                
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        parameters = sum(p.numel() for p in model.parameters())

        # 5. Calculate patchability (a placeholder metric for this example)
        # A real implementation would involve a more complex analysis.
        patchability = accuracy / (parameters / 1e6) if parameters > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'parameters': parameters,
            'patchability': patchability,
        }
        
        return ExperimentResult(
            experiment_id=experiment.id,
            hypothesis_id=experiment.hypothesis_id,
            metrics=metrics,
            primary_metric=patchability,
            model_architecture=config['architecture'],
            model_parameters=parameters,
            training_time=time.time() - start_time
        )

    except Exception as e:
        return ExperimentResult(
            experiment_id=experiment.id,
            hypothesis_id=experiment.hypothesis_id,
            metrics={},
            primary_metric=0.0,
            model_architecture=config.get('architecture', []),
            model_parameters=0,
            training_time=time.time() - start_time,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )