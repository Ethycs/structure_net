"""
Compactification Evolver Component.

This evolver implements network compactification strategies:
- Homologically-guided sparsification
- Dense patch placement at extrema
- Layer-wise compactification management
- Input highway preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import scipy.sparse as sp

from ...core import (
    BaseEvolver, ComponentContract, ComponentVersion,
    Maturity, ResourceRequirements, ResourceLevel,
    EvolutionPlan, IModel, ITrainer
)


@dataclass
class PatchInfo:
    """Information about a dense patch."""
    position: Tuple[int, int]  # (row_start, col_start)
    size: Tuple[int, int]      # (height, width)
    density: float
    data: torch.Tensor
    importance_score: float = 0.0


@dataclass
class CompactificationData:
    """Data structure for layer compactification state."""
    layer_name: str
    original_weight: torch.Tensor
    sparse_skeleton: torch.Tensor
    patches: List[PatchInfo]
    compression_ratio: float
    sparsity_level: float


class CompactificationEvolver(BaseEvolver):
    """
    Evolves networks through compactification.
    
    Implements the strategy of reducing network size while preserving
    critical information through dense patches at extrema locations.
    """
    
    def __init__(self,
                 target_sparsity: float = 0.05,
                 patch_density: float = 0.2,
                 patch_size: int = 8,
                 preserve_input_highway: bool = True,
                 name: str = None):
        """
        Initialize compactification evolver.
        
        Args:
            target_sparsity: Target overall sparsity (default 5%)
            patch_density: Density within patches (default 20%)
            patch_size: Size of dense patches (default 8x8)
            preserve_input_highway: Whether to preserve input highways
            name: Optional custom name
        """
        super().__init__(name or "CompactificationEvolver")
        
        self.target_sparsity = target_sparsity
        self.patch_density = patch_density
        self.patch_size = patch_size
        self.preserve_input_highway = preserve_input_highway
        
        # Track supported plan types
        self._supported_plan_types = {
            'compactification',
            'sparse_evolution',
            'patch_optimization'
        }
        
        # Store compactification state for each layer
        self.compactification_data: Dict[str, CompactificationData] = {}
        
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={'model'},
            optional_inputs={'extrema_analyzer', 'homology_data'},
            provided_outputs={
                'compactified_model',
                'compression_stats',
                'patch_info'
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=False
            )
        )
    
    @property
    def contract(self) -> ComponentContract:
        """Return the component contract."""
        return self._contract
    
    def can_execute_plan(self, plan: EvolutionPlan) -> bool:
        """Check if this evolver can execute the given plan."""
        plan_type = plan.get('type', '')
        return plan_type in self._supported_plan_types
    
    def apply_plan(self, plan: EvolutionPlan, model: IModel, 
                   trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """
        Execute the compactification plan.
        
        Args:
            plan: Evolution plan with compactification parameters
            model: Model to compactify
            trainer: Training interface (not used for compactification)
            optimizer: Optimizer (may need adjustment after compactification)
            
        Returns:
            Dictionary with execution results
        """
        plan_type = plan.get('type', '')
        
        if plan_type == 'compactification':
            return self._apply_compactification(plan, model, optimizer)
        elif plan_type == 'patch_optimization':
            return self._optimize_patches(plan, model)
        elif plan_type == 'sparse_evolution':
            return self._evolve_sparsity(plan, model, optimizer)
        else:
            raise ValueError(f"Unsupported plan type: {plan_type}")
    
    def _apply_compactification(self, plan: EvolutionPlan, 
                               model: IModel, optimizer: Any) -> Dict[str, Any]:
        """Apply compactification to model layers."""
        results = {
            'compactified_layers': [],
            'compression_stats': {},
            'total_compression_ratio': 1.0,
            'patch_count': 0
        }
        
        # Get compactification parameters from plan
        target_sparsity = plan.get('target_sparsity', self.target_sparsity)
        patch_density = plan.get('patch_density', self.patch_density)
        patch_size = plan.get('patch_size', self.patch_size)
        layer_names = plan.get('layer_names', None)
        
        # Get extrema data if provided
        extrema_data = plan.get('extrema_data', {})
        
        # Iterate through model layers
        total_original_params = 0
        total_compressed_params = 0
        
        for name, module in model.named_modules():
            # Skip if specific layers requested and this isn't one
            if layer_names and name not in layer_names:
                continue
            
            # Check if layer can be compactified
            if self._can_compactify_layer(module):
                # Get extrema locations for this layer if available
                layer_extrema = extrema_data.get(name, None)
                
                # Compactify the layer
                compact_data = self._compactify_layer(
                    module, name, target_sparsity, patch_density, 
                    patch_size, layer_extrema
                )
                
                # Store compactification data
                self.compactification_data[name] = compact_data
                
                # Collect statistics
                results['compactified_layers'].append(name)
                results['compression_stats'][name] = {
                    'compression_ratio': compact_data.compression_ratio,
                    'patch_count': len(compact_data.patches),
                    'sparsity': compact_data.sparsity_level
                }
                
                # Update totals
                original_params = compact_data.original_weight.numel()
                compressed_params = int(original_params * compact_data.compression_ratio)
                total_original_params += original_params
                total_compressed_params += compressed_params
                results['patch_count'] += len(compact_data.patches)
        
        # Compute overall compression
        if total_original_params > 0:
            results['total_compression_ratio'] = (
                total_compressed_params / total_original_params
            )
        
        # Update optimizer if needed
        if optimizer and results['compactified_layers']:
            self._update_optimizer(optimizer, model)
        
        return results
    
    def _optimize_patches(self, plan: EvolutionPlan, model: IModel) -> Dict[str, Any]:
        """Optimize existing patches based on importance scores."""
        results = {
            'optimized_layers': [],
            'patches_removed': 0,
            'patches_added': 0,
            'quality_improvement': 0.0
        }
        
        importance_threshold = plan.get('importance_threshold', 0.1)
        
        for name, compact_data in self.compactification_data.items():
            module = dict(model.named_modules())[name]
            # Filter patches by importance
            original_patches = len(compact_data.patches)
            compact_data.patches = [
                p for p in compact_data.patches 
                if p.importance_score >= importance_threshold
            ]
            
            patches_removed = original_patches - len(compact_data.patches)
            if patches_removed > 0:
                results['optimized_layers'].append(name)
                results['patches_removed'] += patches_removed
                
                # Recompute sparse skeleton and update weights
                weight = self._reconstruct_weight(compact_data)
                compact_data.sparse_skeleton = self._create_sparse_skeleton(
                    weight, compact_data.patches, 
                    self.target_sparsity, self.patch_size
                )
                
                # Apply updated compactification
                self._apply_compactification_to_layer(module, compact_data)
        
        return results
    
    def _evolve_sparsity(self, plan: EvolutionPlan, model: IModel, 
                        optimizer: Any) -> Dict[str, Any]:
        """Evolve sparsity patterns based on gradient information."""
        results = {
            'evolved_layers': [],
            'sparsity_changes': {},
            'gradient_guided_updates': 0
        }
        
        gradient_threshold = plan.get('gradient_threshold', 1e-4)
        evolution_rate = plan.get('evolution_rate', 0.1)
        
        for name, compact_data in self.compactification_data.items():
            module = dict(model.named_modules())[name]
            if hasattr(module, 'weight') and module.weight.grad is not None:
                grad = module.weight.grad
                    
                # Find high-gradient regions outside current patches
                grad_magnitude = grad.abs()
                high_grad_mask = grad_magnitude > gradient_threshold
                    
                # Update sparse skeleton based on gradients
                if compact_data.sparse_skeleton is not None:
                    # Gradually include high-gradient weights
                    update_mask = high_grad_mask & (compact_data.sparse_skeleton == 0)
                    update_count = update_mask.sum().item()
                    
                    if update_count > 0:
                        # Add fraction of high-gradient weights
                        num_to_add = int(update_count * evolution_rate)
                        if num_to_add > 0:
                            # Get top gradient magnitudes
                            update_values = grad_magnitude[update_mask]
                            threshold = torch.quantile(update_values, 1 - evolution_rate)
                            
                            # Update skeleton
                            final_mask = update_mask & (grad_magnitude >= threshold)
                            compact_data.sparse_skeleton[final_mask] = module.weight.data[final_mask]
                            
                            results['evolved_layers'].append(name)
                            results['gradient_guided_updates'] += final_mask.sum().item()
                            
                            # Apply updated compactification
                            self._apply_compactification_to_layer(module, compact_data)
                
                # Record sparsity change
                new_sparsity = self._compute_sparsity(compact_data)
                results['sparsity_changes'][name] = new_sparsity
        
        return results
    
    def _can_compactify_layer(self, module: nn.Module) -> bool:
        """Check if a layer can be compactified."""
        # Can compactify linear and convolutional layers
        compactifiable_types = (nn.Linear, nn.Conv2d, nn.Conv1d)
        
        # Check base types
        if isinstance(module, compactifiable_types):
            # Must have weight parameter
            if hasattr(module, 'weight') and module.weight is not None:
                # Minimum size requirements
                weight_size = module.weight.numel()
                return weight_size >= self.patch_size ** 2
        
        return False
    
    def _replace_module(self, model: nn.Module, module_name: str, 
                       new_module: nn.Module):
        """Replace a module in the model."""
        parts = module_name.split('.')
        parent = model
        
        # Navigate to parent module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the module
        setattr(parent, parts[-1], new_module)
    
    def _update_optimizer(self, optimizer: Any, model: nn.Module):
        """Update optimizer after compactification."""
        # Re-initialize optimizer with new parameters
        # This is a simplified version - in practice you'd want to
        # preserve optimizer state where possible
        
        # Get current learning rate
        lr = optimizer.param_groups[0]['lr']
        
        # Get optimizer class and create new instance
        optimizer_class = type(optimizer)
        new_optimizer = optimizer_class(model.parameters(), lr=lr)
        
        # Copy state where possible
        for group in new_optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    new_optimizer.state[p] = optimizer.state[p]
        
        # Update in place
        optimizer.param_groups = new_optimizer.param_groups
        optimizer.state = new_optimizer.state
    
    def create_compactification_plan(self, 
                                   analysis_report: Dict[str, Any],
                                   layer_names: Optional[List[str]] = None) -> EvolutionPlan:
        """
        Create a compactification plan based on analysis.
        
        Args:
            analysis_report: Analysis report with network statistics
            layer_names: Specific layers to compactify
            
        Returns:
            Evolution plan for compactification
        """
        plan = EvolutionPlan({
            'type': 'compactification',
            'target_sparsity': self.target_sparsity,
            'patch_density': self.patch_density,
            'patch_size': self.patch_size,
            'layer_names': layer_names
        })
        
        # Add extrema data if available
        if 'extrema_analysis' in analysis_report:
            plan['extrema_data'] = analysis_report['extrema_analysis']
        
        # Set priority based on network size
        if 'model_stats' in analysis_report:
            total_params = analysis_report['model_stats'].get('total_parameters', 0)
            if total_params > 10_000_000:  # 10M parameters
                plan.priority = 0.9
            elif total_params > 1_000_000:  # 1M parameters
                plan.priority = 0.7
            else:
                plan.priority = 0.5
        
        plan.created_by = self.name
        plan.estimated_impact = 1.0 - self.target_sparsity
        
        return plan
    
    def _compactify_layer(self, module: nn.Module, name: str,
                         target_sparsity: float, patch_density: float,
                         patch_size: int, extrema_locations: Optional[List[Tuple[int, int]]]) -> CompactificationData:
        """Compactify a single layer."""
        # Store original weight
        original_weight = module.weight.data.clone()
        
        # Convert to 2D for processing
        weight_shape = original_weight.shape
        if len(weight_shape) == 4:  # Conv2d
            weight_2d = original_weight.view(weight_shape[0], -1)
        elif len(weight_shape) == 2:  # Linear
            weight_2d = original_weight
        else:
            raise ValueError(f"Unsupported weight shape: {weight_shape}")
        
        # Find extrema locations if not provided
        if extrema_locations is None:
            extrema_locations = self._find_extrema_locations(weight_2d, target_sparsity, patch_size)
        
        # Create patches at extrema
        patches = self._create_patches(weight_2d, extrema_locations, patch_size)
        
        # Create sparse skeleton
        sparse_skeleton = self._create_sparse_skeleton(weight_2d, patches, target_sparsity, patch_size)
        
        # Apply compactification to the layer
        sparse_weight = self._apply_sparsity_mask(weight_2d, patches, sparse_skeleton)
        if len(weight_shape) == 4:
            sparse_weight = sparse_weight.view(weight_shape)
        
        module.weight.data = sparse_weight
        
        # Compute statistics
        sparsity_level = (sparse_weight.abs() < 1e-6).float().mean().item()
        compression_ratio = self._compute_compression_ratio(original_weight, patches, sparse_skeleton)
        
        return CompactificationData(
            layer_name=name,
            original_weight=original_weight,
            sparse_skeleton=sparse_skeleton,
            patches=patches,
            compression_ratio=compression_ratio,
            sparsity_level=sparsity_level
        )
    
    def _find_extrema_locations(self, weight: torch.Tensor, 
                               target_sparsity: float,
                               patch_size: int) -> List[Tuple[int, int]]:
        """Find locations for dense patches based on gradient extrema."""
        h, w = weight.shape
        
        # Compute gradient magnitude
        grad_x = torch.zeros_like(weight)
        grad_y = torch.zeros_like(weight)
        
        if w > 1:
            grad_x[:, 1:] = weight[:, 1:] - weight[:, :-1]
        if h > 1:
            grad_y[1:, :] = weight[1:, :] - weight[:-1, :]
        
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate number of patches needed
        total_elements = h * w
        sparse_elements = int(total_elements * target_sparsity)
        patch_elements = patch_size * patch_size
        num_patches = max(1, sparse_elements // patch_elements)
        
        # Find top gradient locations with spatial separation
        locations = []
        min_separation = patch_size  # Minimum distance between patches
        
        # Flatten and sort by gradient magnitude
        flat_grad = grad_magnitude.flatten()
        sorted_indices = torch.argsort(flat_grad, descending=True)
        
        for idx in sorted_indices:
            row = idx // w
            col = idx % w
            
            # Check if patch fits
            if row + patch_size > h or col + patch_size > w:
                continue
            
            # Check separation from existing patches
            too_close = False
            for existing_row, existing_col in locations:
                if (abs(row - existing_row) < min_separation and 
                    abs(col - existing_col) < min_separation):
                    too_close = True
                    break
            
            if not too_close:
                locations.append((row.item(), col.item()))
                if len(locations) >= num_patches:
                    break
        
        # If not enough extrema found, add on a grid
        if len(locations) < num_patches:
            for i in range(0, h - patch_size, patch_size * 2):
                for j in range(0, w - patch_size, patch_size * 2):
                    if len(locations) < num_patches:
                        if (i, j) not in locations:
                            locations.append((i, j))
        
        return locations
    
    def _create_patches(self, weight: torch.Tensor,
                       locations: List[Tuple[int, int]],
                       patch_size: int) -> List[PatchInfo]:
        """Create dense patches at specified locations."""
        patches = []
        
        for row, col in locations:
            # Extract patch data
            patch_data = weight[row:row+patch_size, col:col+patch_size].clone()
            
            # Compute patch statistics
            density = (patch_data.abs() > 1e-6).float().mean().item()
            importance = patch_data.abs().mean().item() + patch_data.var().item()
            
            patch = PatchInfo(
                position=(row, col),
                size=(patch_size, patch_size),
                density=density,
                data=patch_data,
                importance_score=importance
            )
            patches.append(patch)
        
        return patches
    
    def _create_sparse_skeleton(self, weight: torch.Tensor,
                               patches: List[PatchInfo],
                               target_sparsity: float,
                               patch_size: int) -> torch.Tensor:
        """Create sparse skeleton excluding patch regions."""
        skeleton = weight.clone()
        
        # Zero out patch regions
        for patch in patches:
            row, col = patch.position
            skeleton[row:row+patch_size, col:col+patch_size] = 0
        
        # Apply additional sparsity to skeleton
        non_zero_skeleton = skeleton[skeleton.abs() > 1e-6]
        if len(non_zero_skeleton) > 0:
            # Keep only the most important weights in skeleton
            keep_ratio = target_sparsity * 0.5  # Use half the sparsity budget for skeleton
            threshold = torch.quantile(non_zero_skeleton.abs(), 1 - keep_ratio)
            skeleton[skeleton.abs() < threshold] = 0
        
        return skeleton
    
    def _apply_sparsity_mask(self, weight: torch.Tensor,
                            patches: List[PatchInfo],
                            sparse_skeleton: torch.Tensor) -> torch.Tensor:
        """Apply sparsity mask to create final sparse weight."""
        sparse_weight = sparse_skeleton.clone()
        
        # Add patches
        for patch in patches:
            row, col = patch.position
            h, w = patch.size
            sparse_weight[row:row+h, col:col+w] = patch.data
        
        return sparse_weight
    
    def _reconstruct_weight(self, compact_data: CompactificationData) -> torch.Tensor:
        """Reconstruct full weight from compactification data."""
        # Get shape from original weight
        weight_shape = compact_data.original_weight.shape
        
        # Start with sparse skeleton
        if len(weight_shape) == 4:
            weight_2d = compact_data.sparse_skeleton.view(weight_shape[0], -1)
        else:
            weight_2d = compact_data.sparse_skeleton.clone()
        
        # Add patches
        for patch in compact_data.patches:
            row, col = patch.position
            h, w = patch.size
            weight_2d[row:row+h, col:col+w] = patch.data
        
        # Reshape if needed
        if len(weight_shape) == 4:
            return weight_2d.view(weight_shape)
        return weight_2d
    
    def _compute_compression_ratio(self, original_weight: torch.Tensor,
                                  patches: List[PatchInfo],
                                  sparse_skeleton: torch.Tensor) -> float:
        """Compute compression ratio."""
        original_size = original_weight.numel()
        
        # Compute compressed size
        compressed_size = 0
        
        # Skeleton non-zeros
        compressed_size += (sparse_skeleton.abs() > 1e-6).sum().item()
        
        # Patch data
        for patch in patches:
            compressed_size += patch.data.numel()
        
        # Add metadata overhead (positions, sizes)
        compressed_size += len(patches) * 4  # position and size info
        
        return compressed_size / original_size if original_size > 0 else 0.0
    
    def _compute_sparsity(self, compact_data: CompactificationData) -> float:
        """Compute current sparsity level."""
        weight = self._reconstruct_weight(compact_data)
        return (weight.abs() < 1e-6).float().mean().item()
    
    def _apply_compactification_to_layer(self, module: nn.Module, 
                                        compact_data: CompactificationData):
        """Apply compactification data to a layer."""
        weight = self._reconstruct_weight(compact_data)
        
        # Apply sparsity mask
        sparse_weight = self._apply_sparsity_mask(
            weight.view(compact_data.sparse_skeleton.shape) if len(weight.shape) == 4 else weight,
            compact_data.patches,
            compact_data.sparse_skeleton
        )
        
        # Reshape and apply
        if len(compact_data.original_weight.shape) == 4:
            sparse_weight = sparse_weight.view(compact_data.original_weight.shape)
        
        module.weight.data = sparse_weight