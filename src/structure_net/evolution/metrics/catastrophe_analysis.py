"""
Catastrophe Analysis Module

This module provides metrics for analyzing the dynamical properties of neural networks,
with a focus on detecting potential catastrophic events and instabilities.
"""

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, Any, List

from .base import BaseMetricAnalyzer

class CatastropheAnalyzer(BaseMetricAnalyzer):
    """
    Analyzes the dynamical stability of a network to predict catastrophic forgetting
    or sudden performance drops.
    
    DEPRECATED: This class has been migrated to the component architecture.
    Please use the following components instead:
    - src.structure_net.components.metrics.ActivationStabilityMetric
    - src.structure_net.components.metrics.LyapunovMetric
    - src.structure_net.components.metrics.TransitionEntropyMetric
    - src.structure_net.components.analyzers.CatastropheAnalyzer
    """

    def __init__(self, model, threshold_config=None):
        raise DeprecationWarning(
            "CatastropheAnalyzer has been migrated to component architecture.\n"
            "Please use the following components instead:\n"
            "- For activation stability: src.structure_net.components.metrics.ActivationStabilityMetric\n"
            "- For Lyapunov analysis: src.structure_net.components.metrics.LyapunovMetric\n"
            "- For transition entropy: src.structure_net.components.metrics.TransitionEntropyMetric\n"
            "- For comprehensive analysis: src.structure_net.components.analyzers.CatastropheAnalyzer\n"
            "\nExample migration:\n"
            "# Old:\n"
            "# analyzer = CatastropheAnalyzer(model)\n"
            "# metrics = analyzer.compute_metrics(test_data)\n"
            "\n"
            "# New:\n"
            "from src.structure_net.components.analyzers import CatastropheAnalyzer\n"
            "analyzer = CatastropheAnalyzer()\n"
            "context = EvolutionContext({'test_data': test_data})\n"
            "report = AnalysisReport()\n"
            "analysis = analyzer.analyze(model, context, report)"
        )
        super().__init__(threshold_config)
        self.model = model

    def get_activations(self, x: np.ndarray) -> np.ndarray:
        """
        Gets all activations from the model for a given input.
        
        NOTE: This is a simplified placeholder. A real implementation would
        use forward hooks to capture activations from all layers.
        """
        self.model.eval()
        activations = []
        hooks = []

        def hook_fn(module, input, output):
            activations.append(output.detach().cpu().numpy().flatten())

        for layer in self.model.modules():
            if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
                hooks.append(layer.register_forward_hook(hook_fn))
        
        with torch.no_grad():
            # Model expects a batch, so we unsqueeze and squeeze.
            input_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            self.model(input_tensor)
        
        for hook in hooks:
            hook.remove()
            
        return np.concatenate(activations) if activations else np.array([])

    def apcr_score(self, trajectory: List[np.ndarray]) -> Dict[str, float]:
        """How fast do activation patterns change?"""
        pattern_changes = []
    
        for t in range(len(trajectory)-1):
            pattern_t = (self.get_activations(trajectory[t]) > 0)
            pattern_t1 = (self.get_activations(trajectory[t+1]) > 0)
            if pattern_t.shape != pattern_t1.shape or pattern_t.size == 0:
                continue
            change_rate = np.mean(pattern_t != pattern_t1)
            pattern_changes.append(change_rate)
        
        return {
            'mean_change_rate': np.mean(pattern_changes) if pattern_changes else 0,
            'variance': np.var(pattern_changes) if pattern_changes else 0,
            'max_change': np.max(pattern_changes) if pattern_changes else 0
        }

    def local_lyapunov_spectrum(self, x: np.ndarray, n_directions: int = 10, epsilon: float = 1e-6) -> Dict[str, float]:
        """Estimate local expansion/contraction rates"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            perturbations = torch.randn(n_directions, x_tensor.shape[0])
            perturbations /= torch.linalg.norm(perturbations, axis=1, keepdims=True)
            
            growth_rates = []
            for delta in perturbations:
                x_pert = x_tensor + epsilon * delta
                
                y = self.model(x_tensor.unsqueeze(0)).squeeze(0)
                y_pert = self.model(x_pert.unsqueeze(0)).squeeze(0)
                
                growth = torch.log(torch.linalg.norm(y_pert - y) / epsilon)
                growth_rates.append(growth.item())
        
        return {
            'max_lyapunov': np.max(growth_rates) if growth_rates else 0,
            'mean_lyapunov': np.mean(growth_rates) if growth_rates else 0,
            'lyapunov_variance': np.var(growth_rates) if growth_rates else 0
        }

    def transition_entropy(self, trajectories: List[List[np.ndarray]], n_symbols: int = 256) -> float:
        """Entropy of activation pattern transitions"""
        patterns = []
        for traj in trajectories:
            for x in traj:
                patterns.append(self.get_activations(x).flatten())
        
        if not patterns:
            return 0.0

        patterns = np.array(patterns)
        if patterns.shape[0] < n_symbols:
            return 0.0

        kmeans = MiniBatchKMeans(n_clusters=n_symbols, batch_size=256, n_init=10, random_state=0)
        try:
            kmeans.fit(patterns)
        except ValueError:
            return 0.0

        transitions = np.zeros((n_symbols, n_symbols))
        for traj in trajectories:
            if not traj: continue
            traj_patterns = np.array([self.get_activations(x).flatten() for x in traj])
            if traj_patterns.size == 0: continue
            symbols = kmeans.predict(traj_patterns)
            for i in range(len(symbols)-1):
                transitions[symbols[i], symbols[i+1]] += 1
        
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        
        probs = transitions / row_sums
        log_probs = np.log2(probs + 1e-12)
        entropy = -np.sum(probs * log_probs)
        
        return entropy

    def neural_dmd_analysis(self, test_data: np.ndarray) -> Dict[str, List[float]]:
        """Placeholder for Dynamic Mode Decomposition."""
        # In a real implementation, this would perform DMD on the network's hidden states.
        print("Warning: neural_dmd_analysis is a placeholder and not fully implemented.")
        return {'growth_rates': np.random.rand(5).tolist()}

    def generate_trajectories(self, test_data: np.ndarray, num_traj: int = 10, traj_len: int = 5) -> List[List[np.ndarray]]:
        """Placeholder for generating state trajectories."""
        # This could involve running data through the model for several steps,
        # especially for recurrent or time-series models.
        # For a feed-forward network, we can simulate trajectories by taking sequential data points.
        print("Warning: generate_trajectories is a placeholder.")
        if len(test_data) < traj_len:
            return []
        
        trajectories = []
        for i in range(num_traj):
            start_idx = np.random.randint(0, len(test_data) - traj_len)
            trajectories.append(test_data[start_idx:start_idx+traj_len])
        return trajectories

    def compute_risk_score(self, results: Dict[str, float]) -> float:
        """Combines various metrics into a single catastrophe risk score."""
        risk = 0.0
        risk += results.get('unstable_modes', 0) * 0.2
        risk += results.get('max_lyapunov', 0) * 0.3
        risk += results.get('mean_apcr', 0) * 0.3
        # Higher entropy suggests more stable/varied transitions, so it reduces risk.
        risk -= results.get('transition_entropy', 0) * 0.05 
        return np.clip(risk, 0, 1)

    def compute_metrics(self, test_data: np.ndarray) -> Dict[str, Any]:
        """
        Computes all catastrophe-related metrics.
        This is the main entry point for this analyzer.
        """
        results = {}
        
        # 1. DMD for dominant modes
        dmd_results = self.neural_dmd_analysis(test_data)
        results['unstable_modes'] = sum(1 for gr in dmd_results['growth_rates'] if gr > 1)
        
        # 2. Local Lyapunov Spectrum
        if len(test_data) > 0:
            lyapunov_samples = [self.local_lyapunov_spectrum(x) for x in test_data[:min(100, len(test_data))]]
            if lyapunov_samples:
                results['max_lyapunov'] = np.max([l['max_lyapunov'] for l in lyapunov_samples])
            else:
                results['max_lyapunov'] = 0
        else:
            results['max_lyapunov'] = 0

        # Generate trajectories for APCR and Transition Entropy
        trajectories = self.generate_trajectories(test_data)

        # 3. Activation Pattern Change Rate (APCR)
        if trajectories:
            apcr_scores = [self.apcr_score(traj) for traj in trajectories]
            if apcr_scores:
                results['mean_apcr'] = np.mean([a['mean_change_rate'] for a in apcr_scores])
            else:
                results['mean_apcr'] = 0
        else:
            results['mean_apcr'] = 0

        # 4. Transition Entropy
        if trajectories:
            results['transition_entropy'] = self.transition_entropy(trajectories)
        else:
            results['transition_entropy'] = 0
        
        # 5. Combine into a single risk score
        results['catastrophe_risk'] = self.compute_risk_score(results)
        
        return results
