import torch
import torch.nn as nn
import time
import traceback
from typing import Dict, Any, Tuple

from src.neural_architecture_lab.core import Experiment, ExperimentResult
from src.structure_net.core.network_factory import create_standard_network
from src.data_factory import create_dataset
from src.structure_net.core.io_operations import load_model_seed

def evaluate_competitor_task(experiment: Experiment, device_id: int) -> ExperimentResult:
    """
    NAL worker function for evaluating a single tournament competitor.
    """
    config = experiment.parameters
    device = f'cuda:{device_id}' if torch.cuda.is_available() and device_id >= 0 else 'cpu'
    start_time = time.time()

    try:
        # Extract the actual parameters from the 'params' wrapper
        if 'params' in config and isinstance(config['params'], dict):
            # Merge the params dict with the control parameters
            actual_config = {**config}
            actual_config.update(config['params'])
            config = actual_config
            
        dataset_name = config.get('dataset', 'cifar10')
        dataset = create_dataset(
            dataset_name, 
            batch_size=config.get('batch_size', 128),
            num_workers=config.get('num_workers', 2),
            pin_memory=True
        )
        train_loader = dataset['train_loader']
        test_loader = dataset['test_loader']

        if 'seed_path' in config and config['seed_path']:
            model, _ = load_model_seed(config['seed_path'], device=device)
        else:
            model = create_standard_network(
                architecture=config['architecture'],
                sparsity=config.get('sparsity', 0.02),
                device=device
            )
        
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(config['epochs']):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)
                
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        parameters = sum(p.numel() for p in model.parameters())
        
        metrics = {
            'accuracy': accuracy,
            'parameters': parameters,
            'fitness': (accuracy / (parameters / 1e6)) if parameters > 0 else 0,
            'competitor_id': config.get('competitor_id')
        }
        
        return ExperimentResult(
            experiment_id=experiment.id,
            hypothesis_id=experiment.hypothesis_id,
            metrics=metrics,
            primary_metric=metrics['fitness'],
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