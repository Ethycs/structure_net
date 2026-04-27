"""
Test pipelines for specific metric types.

Provides specialized test pipelines for different categories
of metrics with appropriate test data and validation.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import numpy as np

from src.structure_net.core import ILayer, IModel, EvolutionContext
from src.structure_net.components.metrics import (
    # Information flow metrics
    LayerMIMetric, EntropyMetric, InformationFlowMetric,
    RedundancyMetric, AdvancedMIMetric,
    # Homological metrics
    ChainComplexMetric, RankMetric, BettiNumberMetric,
    HomologyMetric, InformationEfficiencyMetric,
    # Sensitivity/Topological metrics
    GradientSensitivityMetric, BottleneckMetric,
    ExtremaMetric, PersistenceMetric,
    ConnectivityMetric, TopologicalSignatureMetric,
    # Activity metrics
    NeuronActivityMetric, ActivationDistributionMetric,
    ActivityPatternMetric, LayerHealthMetric,
    # Graph metrics
    GraphStructureMetric, CentralityMetric,
    SpectralGraphMetric, PathAnalysisMetric,
    # Catastrophe metrics
    ActivationStabilityMetric, LyapunovMetric, TransitionEntropyMetric,
    # Compactification metrics
    CompressionRatioMetric, PatchEffectivenessMetric,
    MemoryEfficiencyMetric, ReconstructionQualityMetric
)
from tests.fixtures import (
    create_test_layer, create_test_model,
    create_test_activations, create_test_gradients,
    create_compact_data, create_trajectory_data
)
from .base_test_pipeline import MetricTestPipeline


# ===== Information Flow Metrics =====

class LayerMIMetricPipeline(MetricTestPipeline):
    """Test pipeline for LayerMIMetric."""
    
    def get_component_class(self):
        return LayerMIMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create activation data for MI computation."""
        X = create_test_activations(batch_size=100, features=10)
        Y = create_test_activations(batch_size=100, features=5)
        
        return {
            'layer_activations': {
                'input': X,
                'output': Y
            }
        }
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge cases for MI computation."""
        return {
            'identical_data': {
                'layer_activations': {
                    'input': torch.randn(50, 8),
                    'output': torch.randn(50, 8)  # Same dimensions
                }
            },
            'correlated_data': {
                'layer_activations': {
                    'input': torch.randn(200, 10),
                    'output': torch.randn(200, 10) * 0.8 + torch.randn(200, 10) * 0.2
                }
            },
            'knn_method': {
                'layer_activations': {
                    'input': torch.randn(100, 8),
                    'output': torch.randn(100, 4)
                },
                'mi_method': 'knn',
                'k_neighbors': 5
            }
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Validate MI metric outputs."""
        expected_keys = ['mutual_information', 'normalized_mi', 
                        'information_ratio', 'computation_method']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Validate ranges
        assert outputs['mutual_information'] >= 0, "MI must be non-negative"
        assert 0 <= outputs['normalized_mi'] <= 1, "Normalized MI must be in [0,1]"
        assert 0 <= outputs['information_ratio'] <= 1, "Info ratio must be in [0,1]"


class EntropyMetricPipeline(MetricTestPipeline):
    """Test pipeline for EntropyMetric."""
    
    def get_component_class(self):
        return EntropyMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create activation data for entropy computation."""
        return {
            'activations': create_test_activations(100, 10, sparsity=0.2)
        }
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge cases for entropy computation."""
        # Low entropy: mostly zeros
        low_entropy_data = torch.zeros(100, 10)
        low_entropy_data[0, 0] = 1.0
        
        # High entropy: random uniform
        high_entropy_data = torch.rand(1000, 20) * 10 - 5
        
        return {
            'low_entropy': {
                'activations': low_entropy_data
            },
            'high_entropy': {
                'activations': high_entropy_data
            },
            'weight_entropy': {
                'target': create_test_layer(sparsity=0.8)
            }
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Validate entropy metric outputs."""
        expected_keys = ['entropy', 'normalized_entropy', 
                        'effective_bits', 'entropy_ratio']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Validate ranges
        assert outputs['entropy'] >= 0, "Entropy must be non-negative"
        assert 0 <= outputs['normalized_entropy'] <= 1, "Normalized entropy must be in [0,1]"


class InformationFlowMetricPipeline(MetricTestPipeline):
    """Test pipeline for InformationFlowMetric."""
    
    def get_component_class(self):
        return InformationFlowMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create layer sequence data for flow computation."""
        layer_data = {}
        prev_size = 20
        
        for i in range(4):
            out_size = max(5, prev_size - 5)
            layer_data[f'layer_{i}'] = {
                'input': torch.randn(50, prev_size),
                'output': torch.randn(50, out_size)
            }
            prev_size = out_size
        
        return {
            'layer_sequence': ['layer_0', 'layer_1', 'layer_2', 'layer_3'],
            'layer_activations': layer_data
        }
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge cases for information flow."""
        # Bottleneck architecture: 20 -> 5 -> 20
        bottleneck_data = {
            'layer_0': {
                'input': torch.randn(100, 20),
                'output': torch.randn(100, 20)
            },
            'layer_1': {
                'input': torch.randn(100, 20),
                'output': torch.randn(100, 5)  # Bottleneck
            },
            'layer_2': {
                'input': torch.randn(100, 5),
                'output': torch.randn(100, 20)
            }
        }
        
        return {
            'bottleneck_detection': {
                'layer_sequence': ['layer_0', 'layer_1', 'layer_2'],
                'layer_activations': bottleneck_data
            }
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Validate information flow outputs."""
        expected_keys = ['total_flow', 'mean_flow', 'flow_variance',
                        'bottleneck_severity', 'flow_efficiency']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Validate flow values
        layer_count = len(inputs['layer_sequence'])
        assert len(outputs['layer_flows']) == layer_count - 1, "Wrong number of layer flows"
        assert outputs['total_flow'] >= 0, "Total flow must be non-negative"
        assert 0 <= outputs['flow_efficiency'] <= 1, "Flow efficiency must be in [0,1]"


class RedundancyMetricPipeline(MetricTestPipeline):
    """Test pipeline for RedundancyMetric."""
    
    def get_component_class(self):
        return RedundancyMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create data with some redundancy."""
        # Create redundant features (duplicated columns)
        base_features = torch.randn(100, 5)
        redundant_features = torch.cat([
            base_features,
            base_features + torch.randn(100, 5) * 0.01,  # Near duplicates
            torch.randn(100, 5)  # Independent features
        ], dim=1)
        
        return {
            'activations': redundant_features
        }
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge cases for redundancy detection."""
        # Independent features
        independent = torch.randn(200, 10)
        
        return {
            'independent_features': {
                'activations': independent,
                'threshold': 0.9
            }
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Validate redundancy metric outputs."""
        expected_keys = ['total_redundancy', 'mean_redundancy',
                        'max_redundancy', 'redundancy_ratio']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Validate redundancy values
        assert outputs['total_redundancy'] >= 0, "Total redundancy must be non-negative"
        assert 0 <= outputs['redundancy_ratio'] <= 1, "Redundancy ratio must be in [0,1]"


class AdvancedMIMetricPipeline(MetricTestPipeline):
    """Test pipeline for AdvancedMIMetric."""
    
    def get_component_class(self):
        return AdvancedMIMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create data for advanced MI analysis."""
        X = torch.randn(150, 15)
        Y = X[:, :8] * 2 + torch.randn(150, 8) * 0.5  # Correlated
        
        return {
            'X': X,
            'Y': Y
        }
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge cases for advanced MI."""
        # Conditional MI data
        Z = torch.randn(100, 5)  # Conditioning variable
        X = Z + torch.randn(100, 5) * 0.3
        Y = Z + X * 0.5 + torch.randn(100, 5) * 0.2
        
        # MI gradient data
        samples = 100
        X_grad = torch.randn(samples, 10)
        Y_grad = torch.zeros(samples, 10)
        for i in range(10):
            correlation = 1.0 - (i / 10.0)
            Y_grad[:, i] = X_grad[:, i] * correlation + torch.randn(samples) * (1 - correlation)
        
        return {
            'conditional_mi': {
                'X': X,
                'Y': Y,
                'Z': Z,
                'n_neighbors': 5
            },
            'mi_gradient': {
                'X': X_grad,
                'Y': Y_grad,
                'compute_gradients': True
            }
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Validate advanced MI outputs."""
        expected_keys = ['mi', 'entropy_X', 'entropy_Y', 
                        'entropy_XY', 'method_used']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Validate MI values
        assert outputs['mi'] >= 0, "MI must be non-negative"
        assert outputs['entropy_X'] >= 0, "Entropy X must be non-negative"
        assert outputs['entropy_Y'] >= 0, "Entropy Y must be non-negative"


# ===== Homological Metrics =====

class ChainComplexMetricPipeline(MetricTestPipeline):
    """Test pipeline for ChainComplexMetric."""
    
    def get_component_class(self):
        return ChainComplexMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create model for chain complex construction."""
        model = create_test_model([10, 8, 6, 4])
        return {'model': model}
    
    def create_target(self) -> Optional[Union[ILayer, IModel]]:
        return create_test_model([10, 8, 6, 4])
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge cases for chain complex."""
        return {
            'weight_threshold': {
                'model': create_test_model([5, 5, 5]),
                'tolerance': 0.5
            },
            'single_layer': {
                'model': create_test_model([10, 5])
            }
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Validate chain complex outputs."""
        expected_keys = ['chain_length', 'boundary_ranks', 
                        'chain_connectivity', 'complex_dimension']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Validate structure
        assert outputs['chain_length'] >= 1, "Chain length must be positive"
        assert outputs['complex_dimension'] >= 1, "Complex dimension must be positive"
        assert isinstance(outputs['boundary_ranks'], list), "Boundary ranks must be a list"


class RankMetricPipeline(MetricTestPipeline):
    """Test pipeline for RankMetric."""
    
    def get_component_class(self):
        return RankMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create weight matrix for rank computation."""
        # Create matrix with known rank
        rank_3_matrix = torch.randn(8, 10)
        rank_3_matrix[:, 3:] = 0  # Force rank 3
        
        return {'weight_matrix': rank_3_matrix}
    
    def create_target(self) -> Optional[Union[ILayer, IModel]]:
        return create_test_layer()
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge cases for rank computation."""
        # Low rank matrix
        low_rank = torch.randn(5, 10)
        low_rank[:, 2:] = 0  # Rank 2
        
        # Full rank square matrix
        full_rank = torch.randn(6, 6)
        
        return {
            'low_rank': {
                'weight_matrix': low_rank,
                'tolerance': 1e-6
            },
            'full_rank_square': {
                'weight_matrix': full_rank
            }
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Validate rank metric outputs."""
        expected_keys = ['rank', 'normalized_rank', 'rank_deficiency', 
                        'effective_rank', 'condition_number']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Validate rank values
        assert outputs['rank'] >= 0, "Rank must be non-negative"
        assert 0 <= outputs['normalized_rank'] <= 1, "Normalized rank must be in [0,1]"
        assert outputs['rank_deficiency'] >= 0, "Rank deficiency must be non-negative"


class BettiNumberMetricPipeline(MetricTestPipeline):
    """Test pipeline for BettiNumberMetric."""
    
    def get_component_class(self):
        return BettiNumberMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create adjacency matrix for Betti number computation."""
        # Create connected graph adjacency matrix
        adj_matrix = torch.zeros(6, 6)
        # Create a cycle: 0-1-2-3-4-5-0
        for i in range(6):
            adj_matrix[i, (i+1) % 6] = 1
            adj_matrix[(i+1) % 6, i] = 1
        
        return {'adjacency_matrix': adj_matrix}
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge cases for Betti numbers."""
        # Disconnected components
        disconnected = torch.zeros(8, 8)
        # Two separate triangles
        for i in range(3):
            for j in range(i+1, 3):
                disconnected[i, j] = 1
                disconnected[j, i] = 1
                disconnected[i+4, j+4] = 1
                disconnected[j+4, i+4] = 1
        
        # Tree (no cycles)
        tree = torch.zeros(5, 5)
        edges = [(0,1), (1,2), (1,3), (3,4)]
        for i, j in edges:
            tree[i, j] = 1
            tree[j, i] = 1
        
        return {
            'disconnected_components': {
                'adjacency_matrix': disconnected
            },
            'tree_structure': {
                'adjacency_matrix': tree
            }
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Validate Betti number outputs."""
        expected_keys = ['betti_0', 'betti_1', 'betti_2', 'euler_characteristic']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Validate Betti numbers
        assert outputs['betti_0'] >= 1, "Must have at least one connected component"
        assert outputs['betti_1'] >= 0, "Betti_1 must be non-negative"
        assert outputs['betti_2'] >= 0, "Betti_2 must be non-negative"


class HomologyMetricPipeline(MetricTestPipeline):
    """Test pipeline for HomologyMetric."""
    
    def get_component_class(self):
        return HomologyMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create simplicial complex for homology computation."""
        # Create simple complex with known homology
        vertices = list(range(6))
        edges = [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3)]  # Two triangles
        triangles = [(0,1,2), (3,4,5)]
        
        return {
            'vertices': vertices,
            'edges': edges,
            'triangles': triangles
        }
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge cases for homology computation."""
        # Line graph
        line_vertices = [0, 1, 2, 3]
        line_edges = [(0,1), (1,2), (2,3)]
        
        return {
            'line_graph': {
                'vertices': line_vertices,
                'edges': line_edges,
                'triangles': []
            }
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Validate homology outputs."""
        expected_keys = ['homology_groups', 'persistent_homology', 
                        'homology_generators', 'torsion_coefficients']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Validate homology structure
        assert isinstance(outputs['homology_groups'], dict), "Homology groups must be dict"
        assert 'H_0' in outputs['homology_groups'], "Must have H_0"


class InformationEfficiencyMetricPipeline(MetricTestPipeline):
    """Test pipeline for InformationEfficiencyMetric."""
    
    def get_component_class(self):
        return InformationEfficiencyMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create data for information efficiency computation."""
        # Create layer with activations
        layer_activations = create_test_activations(100, 20, sparsity=0.3)
        
        return {
            'layer_activations': layer_activations,
            'expected_output_size': 10
        }
    
    def create_target(self) -> Optional[Union[ILayer, IModel]]:
        return create_test_layer()
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge cases for information efficiency."""
        # High efficiency case
        sparse_activations = create_test_activations(100, 30, sparsity=0.8)
        
        # Low efficiency case  
        dense_activations = create_test_activations(100, 30, sparsity=0.1)
        
        return {
            'high_efficiency': {
                'layer_activations': sparse_activations,
                'expected_output_size': 5
            },
            'low_efficiency': {
                'layer_activations': dense_activations,
                'expected_output_size': 25
            }
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Validate information efficiency outputs."""
        expected_keys = ['efficiency_score', 'information_density', 
                        'redundancy_factor', 'optimal_size']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Validate efficiency values
        assert 0 <= outputs['efficiency_score'] <= 1, "Efficiency score must be in [0,1]"
        assert outputs['information_density'] >= 0, "Information density must be non-negative"
        assert outputs['redundancy_factor'] >= 1, "Redundancy factor must be >= 1"


# ===== Activity Metrics =====

class NeuronActivityMetricPipeline(MetricTestPipeline):
    """Test pipeline for NeuronActivityMetric."""
    
    def get_component_class(self):
        return NeuronActivityMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        return {'activations': create_test_activations(100, 20, sparsity=0.3)}
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        expected_keys = ['active_neurons', 'inactive_neurons', 'activity_rate', 
                        'firing_pattern', 'sparsity_level']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        assert 0 <= outputs['activity_rate'] <= 1, "Activity rate must be in [0,1]"


class ActivationDistributionMetricPipeline(MetricTestPipeline):
    """Test pipeline for ActivationDistributionMetric."""
    
    def get_component_class(self):
        return ActivationDistributionMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        return {'activations': create_test_activations(100, 25)}
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        expected_keys = ['distribution_stats', 'skewness', 'kurtosis', 'outliers']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"


class ActivityPatternMetricPipeline(MetricTestPipeline):
    """Test pipeline for ActivityPatternMetric."""
    
    def get_component_class(self):
        return ActivityPatternMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        return {'activation_patterns': [create_test_activations(50, 15) for _ in range(5)]}
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        expected_keys = ['pattern_diversity', 'temporal_consistency', 'pattern_stability']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"


class LayerHealthMetricPipeline(MetricTestPipeline):
    """Test pipeline for LayerHealthMetric."""
    
    def get_component_class(self):
        return LayerHealthMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        return {
            'layer_activations': create_test_activations(100, 30, sparsity=0.4),
            'layer_gradients': create_test_gradients((100, 30))
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        expected_keys = ['health_score', 'dead_neurons', 'overactive_neurons', 'gradient_health']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        assert 0 <= outputs['health_score'] <= 1, "Health score must be in [0,1]"


# ===== Graph Metrics =====

class GraphStructureMetricPipeline(MetricTestPipeline):
    """Test pipeline for GraphStructureMetric."""
    
    def get_component_class(self):
        return GraphStructureMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        graph_data = create_graph_data(num_nodes=20, edge_probability=0.3)
        return {'graph_data': graph_data}
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        expected_keys = ['num_nodes', 'num_edges', 'density', 'clustering_coefficient']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        assert 0 <= outputs['density'] <= 1, "Density must be in [0,1]"


class CentralityMetricPipeline(MetricTestPipeline):
    """Test pipeline for CentralityMetric."""
    
    def get_component_class(self):
        return CentralityMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        # Create a graph with clear central nodes
        G = nx.karate_club_graph()
        return {'graph': G}
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        expected_keys = ['degree_centrality', 'betweenness_centrality', 
                        'closeness_centrality', 'hub_nodes']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"


class SpectralGraphMetricPipeline(MetricTestPipeline):
    """Test pipeline for SpectralGraphMetric."""
    
    def get_component_class(self):
        return SpectralGraphMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        return {'adjacency_matrix': torch.rand(15, 15)}
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        expected_keys = ['eigenvalues', 'eigenvectors', 'spectral_gap', 'connectivity']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"


class PathAnalysisMetricPipeline(MetricTestPipeline):
    """Test pipeline for PathAnalysisMetric."""
    
    def get_component_class(self):
        return PathAnalysisMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        return {'graph': nx.erdos_renyi_graph(20, 0.3)}
    
    def validate_outputs(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        expected_keys = ['average_path_length', 'diameter', 'shortest_paths', 'path_efficiency']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"


class ActivationBasedMetricPipeline(MetricTestPipeline):
    """Pipeline for metrics that analyze activations."""
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create activation data."""
        return {
            'activations': create_test_activations(100, 50, sparsity=0.2)
        }
    
    def create_invalid_inputs(self) -> List[Dict[str, Any]]:
        """Create invalid activation inputs."""
        return [
            {},  # Missing activations
            {'activations': None},  # None activations
            {'activations': torch.tensor([])},  # Empty tensor
            {'activations': "not a tensor"},  # Wrong type
        ]
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge case scenarios."""
        return {
            'all_zero': {'activations': torch.zeros(50, 20)},
            'all_ones': {'activations': torch.ones(50, 20)},
            'single_sample': {'activations': torch.randn(1, 20)},
            'high_dimensional': {'activations': torch.randn(10, 1000)},
            'very_sparse': {'activations': create_test_activations(50, 30, sparsity=0.95)}
        }
    
    def create_target(self) -> Optional[ILayer]:
        """Activation metrics usually don't need a target."""
        return None


class GradientBasedMetricPipeline(MetricTestPipeline):
    """Pipeline for metrics that analyze gradients."""
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create gradient data."""
        activations = create_test_activations(50, 30)
        gradients = create_test_gradients((50, 30))
        
        return {
            'activations': activations,
            'gradients': gradients
        }
    
    def create_invalid_inputs(self) -> List[Dict[str, Any]]:
        """Create invalid gradient inputs."""
        return [
            {'activations': torch.randn(50, 30)},  # Missing gradients
            {'gradients': torch.randn(50, 30)},  # Missing activations
            {  # Shape mismatch
                'activations': torch.randn(50, 30),
                'gradients': torch.randn(50, 20)
            },
        ]
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create gradient edge cases."""
        return {
            'zero_gradients': {
                'activations': torch.randn(30, 20),
                'gradients': torch.zeros(30, 20)
            },
            'exploding_gradients': {
                'activations': torch.randn(30, 20),
                'gradients': torch.randn(30, 20) * 1000
            },
            'vanishing_gradients': {
                'activations': torch.randn(30, 20),
                'gradients': torch.randn(30, 20) * 1e-8
            }
        }


class LayerAnalysisMetricPipeline(MetricTestPipeline):
    """Pipeline for metrics that analyze individual layers."""
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create basic context."""
        return {}  # Layer metrics get layer from target
    
    def create_invalid_inputs(self) -> List[Dict[str, Any]]:
        """Layer metrics should handle various contexts."""
        return [
            {'invalid_key': 'invalid_value'},  # Unexpected data
        ]
    
    def create_target(self) -> ILayer:
        """Create test layer."""
        return create_test_layer(20, 15, sparsity=0.3)
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge case layers."""
        # Return different layer configurations
        return {
            'tiny_layer': {'target': create_test_layer(2, 2)},
            'large_layer': {'target': create_test_layer(100, 100)},
            'sparse_layer': {'target': create_test_layer(50, 50, sparsity=0.9)},
            'dense_layer': {'target': create_test_layer(30, 30, sparsity=0.0)}
        }


class ModelAnalysisMetricPipeline(MetricTestPipeline):
    """Pipeline for metrics that analyze entire models."""
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create model context."""
        model = create_test_model([20, 15, 10, 5])
        return {'model': model}
    
    def create_invalid_inputs(self) -> List[Dict[str, Any]]:
        """Create invalid model inputs."""
        return [
            {},  # Missing model
            {'model': None},  # None model
            {'model': "not a model"},  # Wrong type
        ]
    
    def create_target(self) -> IModel:
        """Create test model."""
        return create_test_model([25, 20, 15, 10])
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        """Create edge case models."""
        return {
            'tiny_model': {'model': create_test_model([2, 2])},
            'deep_model': {'model': create_test_model([50, 40, 30, 20, 10, 5])},
            'bottleneck_model': {'model': create_test_model([100, 10, 100])},
            'expanding_model': {'model': create_test_model([10, 50, 100])}
        }


# Specific metric pipelines

class EntropyMetricPipeline(ActivationBasedMetricPipeline):
    """Test pipeline for EntropyMetric."""
    
    def get_component_class(self):
        return EntropyMetric
    
    def get_expected_metrics(self) -> List[str]:
        return ['entropy', 'normalized_entropy', 'effective_bits', 'entropy_ratio']
    
    def get_metric_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'entropy': (0.0, 10.0),  # Depends on bins
            'normalized_entropy': (0.0, 1.0),
            'effective_bits': (0.0, 20.0),
            'entropy_ratio': (0.0, 1.0)
        }


class LayerMIMetricPipeline(MetricTestPipeline):
    """Test pipeline for LayerMIMetric."""
    
    def get_component_class(self):
        return LayerMIMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        """Create layer activation pairs."""
        return {
            'layer_activations': {
                'input': create_test_activations(100, 20),
                'output': create_test_activations(100, 15)
            }
        }
    
    def create_invalid_inputs(self) -> List[Dict[str, Any]]:
        return [
            {},  # Missing layer_activations
            {'layer_activations': {}},  # Missing input/output
            {'layer_activations': {'input': torch.randn(50, 20)}},  # Missing output
            {  # Batch size mismatch
                'layer_activations': {
                    'input': torch.randn(50, 20),
                    'output': torch.randn(30, 15)
                }
            }
        ]
    
    def create_target(self) -> Optional[ILayer]:
        return None
    
    def get_expected_metrics(self) -> List[str]:
        return ['mutual_information', 'normalized_mi', 
                'information_ratio', 'computation_method']
    
    def get_metric_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'mutual_information': (0.0, np.inf),
            'normalized_mi': (0.0, 1.0),
            'information_ratio': (0.0, 1.0)
        }


class GradientSensitivityPipeline(MetricTestPipeline):
    """Test pipeline for GradientSensitivityMetric."""
    
    def get_component_class(self):
        return GradientSensitivityMetric
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        return {
            'layer_pair': ('layer_0', 'layer_1'),
            'activations': {
                'layer_0': create_test_activations(50, 20),
                'layer_1': create_test_activations(50, 15)
            },
            'gradients': {
                'layer_0': create_test_gradients((50, 20)),
                'layer_1': create_test_gradients((50, 15))
            }
        }
    
    def create_invalid_inputs(self) -> List[Dict[str, Any]]:
        return [
            {},  # Missing everything
            {'layer_pair': ('a', 'b')},  # Missing data
            {  # Missing gradients
                'layer_pair': ('a', 'b'),
                'activations': {'a': torch.randn(10, 5), 'b': torch.randn(10, 3)}
            }
        ]
    
    def create_target(self) -> Optional[ILayer]:
        return None
    
    def get_expected_metrics(self) -> List[str]:
        return ['sensitivity_score', 'gradient_correlation',
                'activation_gradient_alignment', 'gradient_norm_ratio',
                'effective_sensitivity']
    
    def get_metric_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'sensitivity_score': (0.0, 1.0),
            'gradient_correlation': (-1.0, 1.0),
            'activation_gradient_alignment': (-1.0, 1.0),
            'gradient_norm_ratio': (0.0, np.inf),
            'effective_sensitivity': (0.0, 1.0)
        }


class CompactificationMetricPipeline(MetricTestPipeline):
    """Test pipeline for compactification metrics."""
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        return {
            'compact_data': create_compact_data(
                original_size=10000,
                compression_ratio=0.2,
                num_patches=10
            )
        }
    
    def create_invalid_inputs(self) -> List[Dict[str, Any]]:
        return [
            {},  # Missing compact_data
            {'compact_data': None},
            {'compact_data': {}},  # Empty compact data
        ]
    
    def create_target(self) -> Optional[ILayer]:
        return None
    
    def create_edge_cases(self) -> Dict[str, Dict[str, Any]]:
        return {
            'no_compression': {
                'compact_data': create_compact_data(
                    original_size=1000,
                    compression_ratio=1.0,
                    num_patches=1
                )
            },
            'extreme_compression': {
                'compact_data': create_compact_data(
                    original_size=10000,
                    compression_ratio=0.01,
                    num_patches=5
                )
            },
            'many_patches': {
                'compact_data': create_compact_data(
                    original_size=5000,
                    compression_ratio=0.5,
                    num_patches=100
                )
            }
        }


class DynamicsMetricPipeline(MetricTestPipeline):
    """Test pipeline for dynamics/catastrophe metrics."""
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        # Different metrics need different inputs
        component_class = self.get_component_class()
        
        if component_class == ActivationStabilityMetric:
            trajectory = create_trajectory_data(1, 10, 20)[0]
            return {'activation_trajectory': trajectory}
        
        elif component_class == LyapunovMetric:
            model = create_test_model([20, 15, 10])
            samples = create_test_activations(50, 20)
            return {'model': model, 'input_samples': samples}
        
        elif component_class == TransitionEntropyMetric:
            trajectories = create_trajectory_data(5, 10, 20)
            return {'activation_trajectories': trajectories}
        
        return {}
    
    def create_invalid_inputs(self) -> List[Dict[str, Any]]:
        component_class = self.get_component_class()
        
        if component_class == ActivationStabilityMetric:
            return [
                {},  # Missing trajectory
                {'activation_trajectory': []},  # Empty trajectory
                {'activation_trajectory': [torch.randn(20)]},  # Too short
            ]
        
        return [{}]
    
    def create_target(self) -> Optional[Union[ILayer, IModel]]:
        # Some dynamics metrics need a model
        if self.get_component_class() == LyapunovMetric:
            return create_test_model([15, 10, 5])
        return None


# Pipeline implementations for each metric

class CompressionRatioMetricPipeline(CompactificationMetricPipeline):
    def get_component_class(self):
        return CompressionRatioMetric
    
    def get_expected_metrics(self) -> List[str]:
        return ['original_size', 'compressed_size', 'compression_ratio',
                'space_saved', 'efficiency_score']
    
    def get_metric_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'compression_ratio': (0.0, 1.0),
            'efficiency_score': (0.0, 1.0)
        }


# Register all pipelines
from .base_test_pipeline import TestPipelineRegistry

# Information flow metrics
TestPipelineRegistry.register(LayerMIMetric, LayerMIMetricPipeline)
TestPipelineRegistry.register(EntropyMetric, EntropyMetricPipeline)
TestPipelineRegistry.register(InformationFlowMetric, InformationFlowMetricPipeline)
TestPipelineRegistry.register(RedundancyMetric, RedundancyMetricPipeline)
TestPipelineRegistry.register(AdvancedMIMetric, AdvancedMIMetricPipeline)

# Homological metrics
TestPipelineRegistry.register(ChainComplexMetric, ChainComplexMetricPipeline)
TestPipelineRegistry.register(RankMetric, RankMetricPipeline)
TestPipelineRegistry.register(BettiNumberMetric, BettiNumberMetricPipeline)
TestPipelineRegistry.register(HomologyMetric, HomologyMetricPipeline)
TestPipelineRegistry.register(InformationEfficiencyMetric, InformationEfficiencyMetricPipeline)

# Activity metrics
TestPipelineRegistry.register(NeuronActivityMetric, NeuronActivityMetricPipeline)
TestPipelineRegistry.register(ActivationDistributionMetric, ActivationDistributionMetricPipeline)
TestPipelineRegistry.register(ActivityPatternMetric, ActivityPatternMetricPipeline)
TestPipelineRegistry.register(LayerHealthMetric, LayerHealthMetricPipeline)

# Graph metrics
TestPipelineRegistry.register(GraphStructureMetric, GraphStructureMetricPipeline)
TestPipelineRegistry.register(CentralityMetric, CentralityMetricPipeline)
TestPipelineRegistry.register(SpectralGraphMetric, SpectralGraphMetricPipeline)
TestPipelineRegistry.register(PathAnalysisMetric, PathAnalysisMetricPipeline)

# Sensitivity metrics  
TestPipelineRegistry.register(GradientSensitivityMetric, GradientSensitivityPipeline)

# Compactification metrics
TestPipelineRegistry.register(CompressionRatioMetric, CompressionRatioMetricPipeline)

# Add more registrations as needed...