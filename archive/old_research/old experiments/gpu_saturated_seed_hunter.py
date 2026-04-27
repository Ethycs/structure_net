#!/usr/bin/env python3
"""
GPU Saturated Seed Hunter

Maximally utilize GPU for parallel seed exploration to find optimal sparse 
network initializations for patching experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue
import time
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Note: CUDA_VISIBLE_DEVICES will be set by command line arguments in main()

class ModelCheckpointer:
    """Save promising models for future experiments with datetime organization"""
    
    def __init__(self, save_dir="data/promising_models", dataset="mnist", run_args=None):
        self.dataset = dataset.lower()
        
        # Create datetime-based subdirectory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create args string for folder name
        if run_args:
            args_str = f"_{run_args.mode}"
            if hasattr(run_args, 'sparsity_list') and run_args.sparsity_list:
                args_str += f"_custom"
            elif run_args.mode == 'range':
                args_str += f"_{run_args.sparsity_min:.3f}-{run_args.sparsity_max:.3f}"
            elif run_args.mode == 'single':
                args_str += f"_{run_args.sparsity:.3f}"
            if hasattr(run_args, 'disable_sorting') and run_args.disable_sorting:
                args_str += "_nosort"
            if hasattr(run_args, 'gpu_ids') and run_args.gpu_ids:
                args_str += f"_gpu{run_args.gpu_ids.replace(',', '')}"
        else:
            args_str = ""
        
        self.run_dir = os.path.join(save_dir, f"{timestamp}{args_str}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Track global best across all sparsity levels
        self.global_best = {
            'accuracy': {'value': 0.0, 'model': None},
            'patchability': {'value': 0.0, 'model': None},
            'efficiency': {'value': 0.0, 'model': None}
        }
        
        # Track best for each sparsity level
        self.sparsity_best = {}  # sparsity -> {category -> {value, model}}
        
        print(f"📁 ModelCheckpointer initialized:")
        print(f"   Run directory: {self.run_dir}")
        print(f"   Dataset: {dataset}")
        print(f"   Global best tracking enabled")
    
    def save_promising_model(self, model, architecture, seed, metrics, optimizer=None):
        """Save model if it's best globally OR best for its sparsity level in any category"""
        
        # Calculate metrics for comparison
        accuracy = metrics['accuracy']
        patchability = metrics.get('patchability', 0)
        parameters = sum(p.numel() for p in model.parameters())
        efficiency = accuracy / parameters
        sparsity = metrics.get('sparsity', 0.02)
        
        # Initialize sparsity tracking if needed
        if sparsity not in self.sparsity_best:
            self.sparsity_best[sparsity] = {
                'accuracy': {'value': 0.0, 'model': None},
                'patchability': {'value': 0.0, 'model': None},
                'efficiency': {'value': 0.0, 'model': None}
            }
        
        # Check if this model is best globally OR best for this sparsity level
        categories_to_save = []
        save_reasons = []
        
        # Global best checks
        if accuracy > self.global_best['accuracy']['value']:
            categories_to_save.append(('accuracy', 'GLOBAL'))
            self.global_best['accuracy']['value'] = accuracy
            save_reasons.append(f"New global best accuracy: {accuracy:.2%}")
            
        if patchability > self.global_best['patchability']['value']:
            categories_to_save.append(('patchability', 'GLOBAL'))
            self.global_best['patchability']['value'] = patchability
            save_reasons.append(f"New global best patchability: {patchability:.3f}")
            
        if efficiency > self.global_best['efficiency']['value']:
            categories_to_save.append(('efficiency', 'GLOBAL'))
            self.global_best['efficiency']['value'] = efficiency
            save_reasons.append(f"New global best efficiency: {efficiency:.6f}")
        
        # Sparsity-specific best checks
        if accuracy > self.sparsity_best[sparsity]['accuracy']['value']:
            categories_to_save.append(('accuracy', f'SPARSITY_{sparsity:.3f}'))
            self.sparsity_best[sparsity]['accuracy']['value'] = accuracy
            save_reasons.append(f"New sparsity {sparsity:.1%} best accuracy: {accuracy:.2%}")
            
        if patchability > self.sparsity_best[sparsity]['patchability']['value']:
            categories_to_save.append(('patchability', f'SPARSITY_{sparsity:.3f}'))
            self.sparsity_best[sparsity]['patchability']['value'] = patchability
            save_reasons.append(f"New sparsity {sparsity:.1%} best patchability: {patchability:.3f}")
            
        if efficiency > self.sparsity_best[sparsity]['efficiency']['value']:
            categories_to_save.append(('efficiency', f'SPARSITY_{sparsity:.3f}'))
            self.sparsity_best[sparsity]['efficiency']['value'] = efficiency
            save_reasons.append(f"New sparsity {sparsity:.1%} best efficiency: {efficiency:.6f}")
        
        # Only save if model is best in at least one category
        if not categories_to_save:
            return None
        
        saved_files = []
        
        for category, scope in categories_to_save:
            checkpoint = {
                # Model state
                'model_state_dict': model.state_dict(),
                'architecture': architecture,
                'seed': seed,
                'sparsity': sparsity,
                
                # Training state
                'epoch': metrics.get('epoch', 0),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                
                # Performance metrics
                'accuracy': accuracy,
                'patchability_score': patchability,
                'extrema_counts': metrics.get('extrema_score', 0), 
                'efficiency': efficiency,
                
                # Training settings (IMPORTANT!)
                'neuron_sorting_enabled': metrics.get('sorted', True),
                'sort_frequency': metrics.get('sort_frequency', 5),
                'training_epochs': metrics.get('epochs', 15),
                
                # Neuron analysis
                'dead_neurons': metrics.get('dead_neurons', 0),
                'saturated_neurons': metrics.get('saturated_neurons', 0),
                'activation_patterns': metrics.get('activation_patterns'),
                
                # Reproducibility
                'torch_version': torch.__version__,
                'random_state': torch.get_rng_state(),
                'cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            }
            
            # Build filename with sparsity and category scope
            base_filename = f"model_{self.dataset}_{len(architecture)}layers_seed{seed}_acc{accuracy:.2f}_patch{patchability:.3f}_sparse{sparsity:.3f}"
            category_suffix = f"_BEST_{category.upper()}_{scope}"
            filename = f"{base_filename}{category_suffix}.pt"
            
            filepath = os.path.join(self.run_dir, filename)
            torch.save(checkpoint, filepath)
            saved_files.append(filepath)
            
            if scope == 'GLOBAL':
                print(f"      💾 New global best {category}: {filepath}")
            else:
                print(f"      💾 New sparsity {sparsity:.1%} best {category}: {filepath}")
        
        # Print summary of why this model was saved
        if save_reasons:
            print(f"      🎯 Save reasons: {'; '.join(save_reasons)}")
        
        return saved_files
    
    def _manage_top_k_models(self):
        """Keep only top-K models per category and clean up the rest"""
        try:
            # Get all model files in the directory
            model_files = [f for f in os.listdir(self.save_dir) if f.endswith('.pt')]
            
            if len(model_files) <= self.keep_top_k:
                return  # Not enough models to need cleanup
            
            # Parse model files and group by category
            categories = {
                'accuracy': [],
                'patchability': [], 
                'efficiency': [],
                'general': []  # Models without specific category
            }
            
            for filename in model_files:
                filepath = os.path.join(self.save_dir, filename)
                
                try:
                    # Load checkpoint to get metrics
                    checkpoint = torch.load(filepath, map_location='cpu')
                    
                    model_info = {
                        'filename': filename,
                        'filepath': filepath,
                        'accuracy': checkpoint.get('accuracy', 0),
                        'patchability': checkpoint.get('patchability_score', 0),
                        'parameters': sum(p.numel() for p in checkpoint['model_state_dict'].values()),
                        'efficiency': checkpoint.get('accuracy', 0) / max(sum(p.numel() for p in checkpoint['model_state_dict'].values()), 1)
                    }
                    
                    # Determine category from filename
                    if 'BEST_ACCURACY' in filename:
                        categories['accuracy'].append(model_info)
                    elif 'BEST_PATCHABILITY' in filename:
                        categories['patchability'].append(model_info)
                    elif 'BEST_EFFICIENCY' in filename:
                        categories['efficiency'].append(model_info)
                    else:
                        categories['general'].append(model_info)
                        
                except Exception as e:
                    print(f"⚠️  Warning: Could not parse model file {filename}: {e}")
                    continue
            
            # Keep top-K in each category and mark others for deletion
            files_to_delete = []
            
            for category, models in categories.items():
                if len(models) <= self.keep_top_k:
                    continue
                
                # Sort by appropriate metric
                if category == 'accuracy':
                    models.sort(key=lambda x: x['accuracy'], reverse=True)
                elif category == 'patchability':
                    models.sort(key=lambda x: x['patchability'], reverse=True)
                elif category == 'efficiency':
                    models.sort(key=lambda x: x['efficiency'], reverse=True)
                else:  # general - sort by combined score
                    models.sort(key=lambda x: x['accuracy'] + x['patchability'], reverse=True)
                
                # Mark excess models for deletion
                models_to_delete = models[self.keep_top_k:]
                files_to_delete.extend([m['filepath'] for m in models_to_delete])
                
                if models_to_delete:
                    print(f"🧹 Cleaning up {category} category: keeping top-{self.keep_top_k}, removing {len(models_to_delete)} models")
            
            # Delete excess files
            deleted_count = 0
            for filepath in files_to_delete:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️  Warning: Could not delete {filepath}: {e}")
            
            if deleted_count > 0:
                print(f"✅ Cleaned up {deleted_count} models, keeping top-{self.keep_top_k} per category")
                
        except Exception as e:
            print(f"⚠️  Warning: Top-K management failed: {e}")

class SparsitySweepConfig:
    """Configuration for sparsity sweep strategies"""
    
    # Phase 1: Coarse sweep - map the entire landscape
    COARSE_SPARSITIES = [0.001, 0.01, 0.1]
    
    # Phase 2: Fine sweep ranges based on Phase 1 results
    FINE_RANGES = {
        'ultra_sparse': [0.0005, 0.001, 0.002, 0.005],
        'moderate_sparse': [0.005, 0.01, 0.02, 0.05], 
        'dense_sparse': [0.05, 0.1, 0.2]
    }
    
    # Training epochs per phase
    COARSE_EPOCHS = 3  # Fast exploration
    FINE_EPOCHS = 7    # Detailed evaluation
    
    # Thresholds for determining promising ranges
    MIN_ACCURACY_THRESHOLD = 0.15  # Minimum accuracy to consider
    MIN_PATCHABILITY_THRESHOLD = 0.2  # Minimum patchability to consider

class PersistentSparseLayer(nn.Module):
    """A sparse layer that enforces sparsity during every forward pass."""
    def __init__(self, in_features, out_features, sparsity):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # Create and register the mask
        mask = torch.rand_like(self.linear.weight) < sparsity
        self.register_buffer('mask', mask.float())
        
        # Apply mask at initialization
        with torch.no_grad():
            self.linear.weight.data *= self.mask

    def forward(self, x):
        # CRITICAL: Enforce sparsity on every forward pass
        return torch.nn.functional.linear(x, self.linear.weight * self.mask, self.linear.bias)

class GPUSaturatedSeedHunter:
    """
    Maximally utilize GPU for parallel seed exploration with sparsity sweeping
    """
    
    def __init__(self, num_gpus=1, device='cuda', save_promising=True, dataset='mnist', save_threshold=0.25, keep_top_k=3):
        self.num_gpus = num_gpus
        self.device = device
        self.save_promising = save_promising
        self.dataset = dataset.lower()
        self.save_threshold = save_threshold  # Minimum accuracy threshold for saving
        self.keep_top_k = keep_top_k  # Keep only top K models per category
        
        # Dataset-specific parameters
        if self.dataset == 'cifar10':
            self.input_size = 3072  # 32*32*3
            self.num_classes = 10
            self.test_input_size = 3072
        else:  # mnist
            self.input_size = 784   # 28*28
            self.num_classes = 10
            self.test_input_size = 784
        
        # GPU saturation parameters
        self.batch_size = self._find_max_batch_size()
        self.num_streams = 4  # CUDA streams per GPU
        self.parallel_models = 8  # Models training simultaneously
        
        # Create CUDA streams
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        
        # Results queue
        self.results_queue = queue.Queue()
        
        # Model checkpointer for saving promising models
        if self.save_promising:
            self.checkpointer = ModelCheckpointer(dataset=self.dataset, run_args=getattr(self, '_run_args', None))
        
        # Sparsity sweep configuration
        self.sparsity_config = SparsitySweepConfig()
        
        print(f"🚀 GPU Saturated Seed Hunter initialized")
        print(f"   Dataset: {self.dataset.upper()}")
        print(f"   Input size: {self.input_size}")
        print(f"   Device: {self.device}")
        print(f"   Max batch size: {self.batch_size}")
        print(f"   CUDA streams: {self.num_streams}")
        print(f"   Parallel models: {self.parallel_models}")
        print(f"   Model saving: {'Enabled' if self.save_promising else 'Disabled'}")
        if self.save_promising:
            print(f"   Save threshold: {self.save_threshold:.1%} accuracy")
        
    def _find_max_batch_size(self):
        """Find maximum batch size that fits in GPU memory"""
        if not torch.cuda.is_available():
            return 64
            
        test_batch_sizes = [512, 1024, 2048, 4096, 8192]
        
        for bs in test_batch_sizes:
            try:
                # Test allocation with dataset-specific input size
                dummy = torch.randn(bs, self.input_size, device=self.device)
                model = nn.Linear(self.input_size, self.num_classes).to(self.device)
                output = model(dummy)
                loss = output.sum()
                loss.backward()
                
                # Clear
                del dummy, model, output, loss
                torch.cuda.empty_cache()
                
            except RuntimeError:
                if test_batch_sizes.index(bs) > 0:
                    return test_batch_sizes[test_batch_sizes.index(bs) - 1]
                else:
                    return 64
                
        return test_batch_sizes[-1]

    def create_seed_batch(self, num_seeds=100):
        """
        Create batch of different seed architectures using systematic exploration.
        
        This function systematically builds a diverse "portfolio" of network shapes,
        each designed to test a different architectural philosophy or heuristic.
        It explores the fundamental trade-off between depth and width.
        """
        
        architectures = []
        
        # Type 1: The Simplest Baseline - Direct Connections [input_size, C]
        # Purpose: Test if any meaningful mapping can be learned without hidden representations
        # Example: [3072, 10] - rock-bottom baseline for performance
        for c in [10, 20, 30, 40, 50]:
            architectures.append([self.input_size, c])
            
        # Type 2: The Classic MLP - Single Hidden Layer [input_size, H, C]
        # Purpose: Explore the effect of a single bottleneck with various widths
        # Example: [3072, 128, 10] - tests optimal "compression" size
        for h in [16, 32, 64, 128, 256, 512]:
            architectures.append([self.input_size, h, self.num_classes])
            
        # Type 3: The Funnel - Decreasing Width, Two Hidden Layers
        # Purpose: Test progressive compression through deeper layers
        # Example: [3072, 512, 64, 10] - gradual funnel vs sharp bottleneck
        for h1 in [128, 256, 512]:
            for h2 in [32, 64, 128]:
                if h2 < h1:  # Decreasing size
                    architectures.append([self.input_size, h1, h2, self.num_classes])
                    
        # Type 4: The Deep Funnel - Decreasing Width, Three Hidden Layers
        # Purpose: Push the "funnel" idea further with more gradual compression
        # Example: [3072, 512, 256, 64, 10] - test if more depth helps
        for h1 in [256, 512]:
            for h2 in [128, 256]:
                for h3 in [32, 64]:
                    if h3 < h2 < h1:
                        architectures.append([self.input_size, h1, h2, h3, self.num_classes])
                        
        # Type 5: Wide and Shallow - A Single, Massive Hidden Layer
        # Purpose: Test opposite extreme of deep funnels
        # Example: [3072, 2048, 10] - large features in single layer vs sequence of smaller transformations
        for w in [1024, 2048]:
            architectures.append([self.input_size, w, self.num_classes])
            
        # Type 6: The Column - Constant Width, Deep
        # Purpose: Test maintaining "representational capacity" throughout network
        # Example: [3072, 64, 64, 64, 64, 64, 10] - transform features without compression
        narrow_arch = [self.input_size]
        for depth in range(5):
            narrow_arch.append(64)
        narrow_arch.append(self.num_classes)
        architectures.append(narrow_arch)
        
        print(f"📐 Generated {len(architectures)} systematic architectures:")
        print(f"   - Direct connections: 5 variants")
        print(f"   - Classic MLPs: 6 variants") 
        print(f"   - Funnels (2-layer): {sum(1 for h1 in [128,256,512] for h2 in [32,64,128] if h2 < h1)} variants")
        print(f"   - Deep funnels (3-layer): {sum(1 for h1 in [256,512] for h2 in [128,256] for h3 in [32,64] if h3 < h2 < h1)} variants")
        print(f"   - Wide & shallow: 2 variants")
        print(f"   - Deep columns: 1 variant")
        
        return architectures[:num_seeds]

    def create_sparse_network(self, architecture, sparsity=0.02, seed=None):
        """Create sparse network with persistent sparsity enforcement"""
        if seed is not None:
            torch.manual_seed(seed)
        
        layers = []
        for i in range(len(architecture) - 1):
            # Use the new persistent sparse layer
            layer = PersistentSparseLayer(architecture[i], architecture[i+1], sparsity)
            layers.append(layer)
            if i < len(architecture) - 2:
                layers.append(nn.ReLU())
                
        return nn.Sequential(*layers)
    
    def cache_dataset_gpu(self):
        """Pre-load dataset to GPU for faster access"""
        if self.dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # Load CIFAR-10
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
            
            # Convert to tensors and move to GPU
            train_x = torch.from_numpy(train_dataset.data).float().reshape(-1, 3072) / 255.0
            train_y = torch.tensor(train_dataset.targets)
            test_x = torch.from_numpy(test_dataset.data).float().reshape(-1, 3072) / 255.0
            test_y = torch.tensor(test_dataset.targets)
            
            # Normalize CIFAR-10
            train_mean = train_x.mean(dim=0)
            train_std = train_x.std(dim=0)
            train_x = (train_x - train_mean) / (train_std + 1e-8)
            test_x = (test_x - train_mean) / (train_std + 1e-8)
            
        else:  # MNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # Load MNIST
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            
            # Convert to tensors and move to GPU
            train_x = train_dataset.data.float().view(-1, 784) / 255.0
            train_y = train_dataset.targets
            test_x = test_dataset.data.float().view(-1, 784) / 255.0
            test_y = test_dataset.targets
            
            # Normalize MNIST
            train_mean = train_x.mean()
            train_std = train_x.std()
            train_x = (train_x - train_mean) / train_std
            test_x = (test_x - train_mean) / train_std
        
        if torch.cuda.is_available():
            self.cached_dataset = {
                'train_x': train_x.to(self.device),
                'train_y': train_y.to(self.device),
                'test_x': test_x.to(self.device),
                'test_y': test_y.to(self.device)
            }
        else:
            self.cached_dataset = {
                'train_x': train_x,
                'train_y': train_y,
                'test_x': test_x,
                'test_y': test_y
            }
    
    def get_cached_dataset(self):
        """Return cached dataset"""
        return self.cached_dataset
    
    @torch.no_grad()
    def apply_permutation(self, model: nn.Sequential, layer_idx_to_sort: int, perm_indices: torch.Tensor):
        """
        Applies a permutation to a layer within an nn.Sequential model,
        perfectly preserving the network's function.
        """
        # The actual nn.Linear module is at index `layer_idx_to_sort * 2`
        # because of the ReLUs in between.
        linear_layer_idx = layer_idx_to_sort * 2

        # --- Part A: Permute the output neurons of the target layer ---
        # This means re-ordering the rows of its weight and bias.
        layer_i = model[linear_layer_idx]
        if hasattr(layer_i, 'linear'):  # PersistentSparseLayer
            layer_i.linear.weight.data = layer_i.linear.weight.data[perm_indices, :]
            layer_i.linear.bias.data = layer_i.linear.bias.data[perm_indices]
            # Also permute the mask
            layer_i.mask = layer_i.mask[perm_indices, :]
        else:  # Regular nn.Linear
            layer_i.weight.data = layer_i.weight.data[perm_indices, :]
            layer_i.bias.data = layer_i.bias.data[perm_indices]
        
        # --- Part B: Permute the input connections of the NEXT layer ---
        # This means re-ordering the columns of the next layer's weight matrix.
        next_linear_layer_idx = (layer_idx_to_sort + 1) * 2
        if next_linear_layer_idx < len(model):
            layer_i_plus_1 = model[next_linear_layer_idx]
            if hasattr(layer_i_plus_1, 'linear'):  # PersistentSparseLayer
                layer_i_plus_1.linear.weight.data = layer_i_plus_1.linear.weight.data[:, perm_indices]
                # Also permute the mask
                layer_i_plus_1.mask = layer_i_plus_1.mask[:, perm_indices]
            else:  # Regular nn.Linear
                layer_i_plus_1.weight.data = layer_i_plus_1.weight.data[:, perm_indices]

    def sort_network_layers(self, model: nn.Sequential):
        """
        Performs a maintenance step on the network by sorting the neurons
        in each hidden layer by importance.
        """
        num_linear_layers = (len(model) + 1) // 2
        
        # We sort every hidden layer, but not the final output layer.
        for i in range(num_linear_layers - 1):
            linear_layer_idx = i * 2
            current_layer = model[linear_layer_idx]
            
            # Calculate importance of each output neuron based on its weight norm (L2 norm of rows)
            if hasattr(current_layer, 'linear'):  # PersistentSparseLayer
                importance = torch.linalg.norm(current_layer.linear.weight, ord=2, dim=1)
            else:  # Regular nn.Linear
                importance = torch.linalg.norm(current_layer.weight, ord=2, dim=1)
            
            # Get the indices that would sort the neurons by importance
            perm_indices = torch.argsort(importance, descending=True)
            
            # Apply this permutation to the network
            self.apply_permutation(model, i, perm_indices)

    def calculate_extrema_score(self, model):
        """Calculate extrema score for patchability prediction"""
        extrema_count = 0
        total_neurons = 0
        
        with torch.no_grad():
            # Run a few batches to get activations
            for i in range(3):
                start_idx = i * 1000
                end_idx = min((i + 1) * 1000, len(self.cached_dataset['train_x']))
                x = self.cached_dataset['train_x'][start_idx:end_idx]
                
                activations = []
                
                # Hook to capture activations
                def hook(module, input, output):
                    if isinstance(module, (nn.Linear, PersistentSparseLayer)):
                        activations.append(output)
                
                hooks = []
                for layer in model:
                    if isinstance(layer, (nn.Linear, PersistentSparseLayer)):
                        hooks.append(layer.register_forward_hook(hook))
                
                _ = model(x)
                
                # Count extrema (skip output layer)
                for j, act in enumerate(activations[:-1]):
                    # Apply ReLU to get post-activation values
                    if j < len(activations) - 1:
                        act = torch.relu(act)
                    
                    mean_act = act.mean(dim=0)
                    high_threshold = mean_act.mean() + 2 * mean_act.std()
                    low_threshold = 0.1
                    
                    high = (mean_act > high_threshold).sum().item()
                    low = (mean_act < low_threshold).sum().item()
                    extrema_count += high + low
                    total_neurons += act.size(1)
                
                # Remove hooks
                for h in hooks:
                    h.remove()
                    
        return extrema_count / max(total_neurons, 1)  # Normalize by total neurons
    
    def parallel_seed_test(self, architecture, seed, stream_idx, sparsity=0.02, epochs=5):
        """Test single seed on GPU with given stream"""
        
        if torch.cuda.is_available():
            with torch.cuda.stream(self.streams[stream_idx]):
                return self._test_seed_impl(architecture, seed, sparsity, epochs)
        else:
            return self._test_seed_impl(architecture, seed, sparsity, epochs)
    
    def _test_seed_impl(self, architecture, seed, sparsity, epochs=15):
        """Implementation of seed testing with configurable neuron sorting."""
        torch.manual_seed(seed)
        
        # Get sorting configuration from hunter instance
        disable_sorting = getattr(self, '_disable_sorting', False)
        sort_frequency = getattr(self, '_sort_frequency', 5)
        
        # Create sparse network
        model = self.create_sparse_network(architecture, sparsity=sparsity, seed=seed)
        model = model.to(self.device)
        
        # Use mixed precision for speed if available
        scaler = GradScaler() if torch.cuda.is_available() else None
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        dataset = self.get_cached_dataset()
        
        # Only print for first few models to avoid spam
        should_print = seed < 3 and len(architecture) <= 3
        if should_print:
            if disable_sorting:
                print(f"  🌱 Training arch {architecture} seed {seed} (no sorting)...")
            else:
                print(f"  🌱 Training arch {architecture} seed {seed} with sorting every {sort_frequency} epochs...")
        
        best_acc = 0
        for epoch in range(epochs):
            # Training
            model.train()
            for batch_idx in range(0, len(dataset['train_x']), self.batch_size):
                optimizer.zero_grad()
                
                # Get batch
                end_idx = min(batch_idx + self.batch_size, len(dataset['train_x']))
                x = dataset['train_x'][batch_idx:end_idx]
                y = dataset['train_y'][batch_idx:end_idx]
                
                # Forward with mixed precision if available
                if scaler:
                    with autocast():
                        output = model(x)
                        loss = nn.functional.cross_entropy(output, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(x)
                    loss = nn.functional.cross_entropy(output, y)
                    loss.backward()
                    optimizer.step()

            # --- NEURON SORTING MAINTENANCE ---
            # Perform maintenance (sorting) every few epochs if enabled
            if not disable_sorting and (epoch + 1) % sort_frequency == 0 and epoch < epochs - 1:
                if should_print:
                    print(f"    🔄 Epoch {epoch+1}: Performing maintenance sort...")
                self.sort_network_layers(model)
            
            # Evaluation on final epoch
            if epoch == epochs - 1:
                model.eval()
                correct = 0
                with torch.no_grad():
                    for batch_idx in range(0, len(dataset['test_x']), self.batch_size):
                        end_idx = min(batch_idx + self.batch_size, len(dataset['test_x']))
                        x = dataset['test_x'][batch_idx:end_idx]
                        y = dataset['test_y'][batch_idx:end_idx]
                        
                        output = model(x)
                        pred = output.argmax(dim=1)
                        correct += (pred == y).sum().item()
                
                accuracy = correct / len(dataset['test_y'])
                best_acc = max(best_acc, accuracy)
        
        # Calculate extrema for patchability score
        extrema_score = self.calculate_extrema_score(model)
        
        return {
            'architecture': architecture,
            'seed': seed,
            'accuracy': best_acc,
            'extrema_score': extrema_score,
            'parameters': sum(p.numel() for p in model.parameters()),
            'patchability': extrema_score * (1 - best_acc),  # High extrema + low acc = patchable
            'sparsity': sparsity,
            'epochs': epochs,
            'sorted': not disable_sorting,  # Flag to indicate if this model used neuron sorting
            'sort_frequency': sort_frequency if not disable_sorting else None
        }
    
    def gpu_saturated_search(self, num_architectures=50, seeds_per_arch=20, sparsity=0.02):
        """Saturate GPU with parallel seed searches (single sparsity)"""
        
        print(f"\n🚀 GPU Saturated Seed Hunt")
        print(f"   Architectures: {num_architectures}")
        print(f"   Seeds per arch: {seeds_per_arch}")
        print(f"   Total experiments: {num_architectures * seeds_per_arch}")
        print(f"   Sparsity: {sparsity:.1%}")
        
        architectures = self.create_seed_batch(num_architectures)
        all_results = []
        
        # Pre-cache dataset in GPU memory
        print("📦 Caching dataset...")
        self.cache_dataset_gpu()
        
        start_time = time.time()
        
        # Process architectures in batches
        for arch_idx in range(0, len(architectures), self.parallel_models):
            arch_batch = architectures[arch_idx:arch_idx + self.parallel_models]
            
            print(f"🔄 Processing architectures {arch_idx+1}-{min(arch_idx+len(arch_batch), len(architectures))}")
            
            # Launch parallel seeds for each architecture
            with ThreadPoolExecutor(max_workers=self.parallel_models) as executor:
                futures = []
                
                for i, arch in enumerate(arch_batch):
                    for seed in range(seeds_per_arch):
                        stream_idx = (i * seeds_per_arch + seed) % self.num_streams
                        future = executor.submit(
                            self.parallel_seed_test, 
                            arch, 
                            seed, 
                            stream_idx,
                            sparsity
                        )
                        futures.append(future)
                
                # Collect results
                for future in futures:
                    result = future.result()
                    all_results.append(result)
                    
                    # Print promising seeds but don't save yet
                    if result['accuracy'] > self.save_threshold:
                        print(f"🌟 Promising: {result['architecture']} "
                              f"seed={result['seed']} "
                              f"acc={result['accuracy']:.2%} "
                              f"patch_score={result['patchability']:.3f}")
        
        # Synchronize all streams if using CUDA
        if torch.cuda.is_available():
            for stream in self.streams:
                stream.synchronize()
            
        elapsed = time.time() - start_time
        total_experiments = len(architectures) * seeds_per_arch
        
        print(f"\n✅ Completed {total_experiments} experiments in {elapsed:.1f}s")
        print(f"   Throughput: {total_experiments/elapsed:.1f} seeds/second")
        
        # Analyze results
        return self.analyze_seed_results(all_results)
    
    def analyze_seed_results(self, results):
        """Find best seeds by different criteria"""
        
        print(f"\n📊 ANALYZING {len(results)} SEED RESULTS...")
        
        # Sort by different metrics
        by_accuracy = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        by_patchability = sorted(results, key=lambda x: x['patchability'], reverse=True)
        by_efficiency = sorted(results, key=lambda x: x['accuracy']/x['parameters'], reverse=True)
        
        print("\n🏆 Top Seeds by Accuracy:")
        for r in by_accuracy[:5]:
            print(f"   {r['architecture']} seed={r['seed']}: {r['accuracy']:.2%}")
            
        print("\n🎯 Top Seeds by Patchability:")
        for r in by_patchability[:5]:
            print(f"   {r['architecture']} seed={r['seed']}: "
                  f"acc={r['accuracy']:.2%}, patch_score={r['patchability']:.3f}")
            
        print("\n⚡ Top Seeds by Efficiency:")
        for r in by_efficiency[:5]:
            eff = r['accuracy'] / r['parameters'] * 1000
            print(f"   {r['architecture']} seed={r['seed']}: "
                  f"{eff:.3f} acc/kparam")
        
        # Save top models if checkpointer is available
        if hasattr(self, 'checkpointer') and self.save_promising:
            print("\n💾 Saving top models with category markers...")
            
            # Save best models from each category
            top_models = {
                'accuracy': by_accuracy[0],
                'patchability': by_patchability[0], 
                'efficiency': by_efficiency[0]
            }
            
            for category, result in top_models.items():
                print(f"\n   🔧 Saving best {category}:")
                print(f"      Architecture: {result['architecture']}")
                print(f"      Seed: {result['seed']}")
                print(f"      Accuracy: {result['accuracy']:.3f}")
                print(f"      Patchability: {result['patchability']:.3f}")
                
                try:
                    # Recreate the model
                    torch.manual_seed(result['seed'])
                    model = self.create_sparse_network(
                        result['architecture'], 
                        sparsity=result['sparsity'], 
                        seed=result['seed']
                    )
                    
                    # Add category and epoch info to metrics
                    metrics_with_info = result.copy()
                    metrics_with_info['epoch'] = 5
                    metrics_with_info['category'] = f'best_{category}'
                    
                    filepath = self.checkpointer.save_promising_model(
                        model, 
                        result['architecture'], 
                        result['seed'], 
                        metrics_with_info
                    )
                    print(f"      ✅ Saved: {filepath}")
                    
                except Exception as e:
                    print(f"      ❌ Failed to save best {category}: {e}")
        else:
            print("   ⚠️  Best model saving skipped (checkpointer not available or saving disabled)")
        
        return {
            'best_accuracy': by_accuracy[0],
            'best_patchable': by_patchability[0],
            'best_efficient': by_efficiency[0],
            'all_results': results
        }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Saturated Seed Hunter with Hybrid Sparsity Sweep')
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'sweep', 'range'],
                       help='Search mode: single sparsity, predefined sweep, or custom range (default: single)')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--num-architectures', type=int, default=30,
                       help='Number of architectures to test (default: 30)')
    parser.add_argument('--seeds-per-arch', type=int, default=10,
                       help='Number of seeds per architecture (default: 10)')
    parser.add_argument('--sparsity', type=float, default=0.02,
                       help='Sparsity level for single mode (default: 0.02)')
    parser.add_argument('--sparsity-min', type=float, default=0.001,
                       help='Minimum sparsity for range mode (default: 0.001)')
    parser.add_argument('--sparsity-max', type=float, default=0.1,
                       help='Maximum sparsity for range mode (default: 0.1)')
    parser.add_argument('--sparsity-step', type=float, default=0.01,
                       help='Sparsity increment for range mode (default: 0.01)')
    parser.add_argument('--sparsity-list', type=str, default=None,
                       help='Comma-separated list of sparsities (e.g., "0.01,0.02,0.05,0.1")')
    parser.add_argument('--thresh', type=float, default=0.25,
                       help='Patchability threshold for saving models (default: 0.25 = 25%%)')
    parser.add_argument('--disable-sorting', action='store_true',
                       help='Disable neuron sorting during training')
    parser.add_argument('--sort-frequency', type=int, default=5,
                       help='Sort neurons every N epochs (default: 5)')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs to use for parallel processing (default: 1)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='Comma-separated GPU IDs to use (e.g., "0,1,2"). If not specified, uses first N GPUs.')
    
    args = parser.parse_args()
    
    # Configure GPU usage
    if args.gpu_ids:
        gpu_list = [int(gpu.strip()) for gpu in args.gpu_ids.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_list))
        effective_num_gpus = len(gpu_list)
        print(f"🎮 Using specified GPUs: {gpu_list}")
    else:
        effective_num_gpus = min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        if torch.cuda.is_available():
            gpu_list = list(range(effective_num_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_list))
            print(f"🎮 Using first {effective_num_gpus} GPUs: {gpu_list}")
        else:
            print("🎮 CUDA not available, using CPU")
    
    # Determine sparsity values to test
    if args.mode == 'single':
        sparsity_values = [args.sparsity]
    elif args.mode == 'sweep':
        # Predefined comprehensive sweep
        sparsity_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    elif args.mode == 'range':
        if args.sparsity_list:
            # Parse comma-separated list
            sparsity_values = [float(s.strip()) for s in args.sparsity_list.split(',')]
        else:
            # Generate range
            sparsity_values = []
            current = args.sparsity_min
            while current <= args.sparsity_max:
                sparsity_values.append(current)
                current += args.sparsity_step
    
    print("🔍 GPU Saturated Seed Hunter with Systematic Architecture Generation & Neuron Sorting")
    print("="*80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Architectures: {args.num_architectures}")
    print(f"Seeds per arch: {args.seeds_per_arch}")
    print(f"Neuron sorting: {'Disabled' if args.disable_sorting else f'Every {args.sort_frequency} epochs'}")
    
    if args.mode == 'single':
        print(f"Sparsity: {args.sparsity:.1%}")
    else:
        print(f"Sparsity values: {[f'{s:.1%}' for s in sparsity_values]}")
        print(f"Total sparsity levels: {len(sparsity_values)}")
        print(f"Total experiments: {args.num_architectures * args.seeds_per_arch * len(sparsity_values)}")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hunter = GPUSaturatedSeedHunter(num_gpus=effective_num_gpus, device=device, dataset=args.dataset, save_threshold=args.thresh)
    
    # Store all results across sparsity levels
    all_sparsity_results = {}
    
    for sparsity in sparsity_values:
        print(f"\n🎯 Testing Sparsity Level: {sparsity:.1%}")
        print("-" * 50)
        
        # Update hunter's test implementation to use sorting settings
        hunter._disable_sorting = args.disable_sorting
        hunter._sort_frequency = args.sort_frequency
        
        results = hunter.gpu_saturated_search(
            num_architectures=args.num_architectures,
            seeds_per_arch=args.seeds_per_arch,
            sparsity=sparsity
        )
        
        all_sparsity_results[sparsity] = results
        
        # Print summary for this sparsity level
        best_acc = results['best_accuracy']
        print(f"\n📊 Sparsity {sparsity:.1%} Summary:")
        print(f"   Best accuracy: {best_acc['accuracy']:.2%} (arch: {best_acc['architecture']}, seed: {best_acc['seed']})")
    
    # Cross-sparsity analysis
    if len(sparsity_values) > 1:
        print(f"\n🔬 CROSS-SPARSITY ANALYSIS")
        print("="*50)
        
        # Find overall best across all sparsity levels
        all_results = []
        for sparsity, results in all_sparsity_results.items():
            for result in results['all_results']:
                result['tested_sparsity'] = sparsity
                all_results.append(result)
        
        # Overall best performers
        overall_best_acc = max(all_results, key=lambda x: x['accuracy'])
        overall_best_patch = max(all_results, key=lambda x: x['patchability'])
        overall_best_eff = max(all_results, key=lambda x: x['accuracy']/x['parameters'])
        
        print(f"\n🏆 Overall Best Accuracy: {overall_best_acc['accuracy']:.2%}")
        print(f"   Architecture: {overall_best_acc['architecture']}")
        print(f"   Seed: {overall_best_acc['seed']}")
        print(f"   Sparsity: {overall_best_acc['tested_sparsity']:.1%}")
        
        print(f"\n🎯 Overall Best Patchability: {overall_best_patch['patchability']:.3f}")
        print(f"   Architecture: {overall_best_patch['architecture']}")
        print(f"   Accuracy: {overall_best_patch['accuracy']:.2%}")
        print(f"   Sparsity: {overall_best_patch['tested_sparsity']:.1%}")
        
        print(f"\n⚡ Overall Best Efficiency: {overall_best_eff['accuracy']/overall_best_eff['parameters']*1000:.3f} acc/kparam")
        print(f"   Architecture: {overall_best_eff['architecture']}")
        print(f"   Accuracy: {overall_best_eff['accuracy']:.2%}")
        print(f"   Sparsity: {overall_best_eff['tested_sparsity']:.1%}")
        
        # Sparsity trend analysis
        print(f"\n📈 Sparsity Trend Analysis:")
        sparsity_summary = {}
        for sparsity in sorted(sparsity_values):
            results = all_sparsity_results[sparsity]
            avg_acc = np.mean([r['accuracy'] for r in results['all_results']])
            avg_patch = np.mean([r['patchability'] for r in results['all_results']])
            best_acc = results['best_accuracy']['accuracy']
            sparsity_summary[sparsity] = {
                'avg_accuracy': avg_acc,
                'best_accuracy': best_acc,
                'avg_patchability': avg_patch
            }
            print(f"   {sparsity:.1%}: avg_acc={avg_acc:.2%}, best_acc={best_acc:.2%}, avg_patch={avg_patch:.3f}")
        
        # Save comprehensive results
        results_dir = f'data/seed_hunt_results_{args.dataset}'
        os.makedirs(results_dir, exist_ok=True)
        
        comprehensive_results = {
            'mode': args.mode,
            'dataset': args.dataset,
            'sparsity_values': sparsity_values,
            'neuron_sorting': not args.disable_sorting,
            'sort_frequency': args.sort_frequency,
            'overall_best_accuracy': overall_best_acc,
            'overall_best_patchability': overall_best_patch,
            'overall_best_efficiency': overall_best_eff,
            'sparsity_summary': sparsity_summary,
            'detailed_results': all_sparsity_results
        }
        
        with open(f'{results_dir}/comprehensive_sparsity_sweep.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive results saved to {results_dir}/comprehensive_sparsity_sweep.json")
    
    else:
        # Single sparsity mode - use original result handling
        results = all_sparsity_results[sparsity_values[0]]
    
    print("\n🎊 BEST SEEDS SUMMARY")
    print("="*50)
    
    best = results['best_patchable']
    print(f"🎯 Best for patching:")
    print(f"   Architecture: {best['architecture']}")
    print(f"   Seed: {best['seed']}")
    print(f"   Initial accuracy: {best['accuracy']:.2%}")
    print(f"   Patchability score: {best['patchability']:.3f}")
    print(f"   Parameters: {best['parameters']:,}")
    
    best_acc = results['best_accuracy']
    print(f"\n🏆 Best accuracy:")
    print(f"   Architecture: {best_acc['architecture']}")
    print(f"   Seed: {best_acc['seed']}")
    print(f"   Accuracy: {best_acc['accuracy']:.2%}")
    
    best_eff = results['best_efficient']
    print(f"\n⚡ Most efficient:")
    print(f"   Architecture: {best_eff['architecture']}")
    print(f"   Seed: {best_eff['seed']}")
    print(f"   Efficiency: {best_eff['accuracy']/best_eff['parameters']*1000:.3f} acc/kparam")
    
    # Save results
    results_dir = f'data/seed_hunt_results_{args.dataset}'
    os.makedirs(results_dir, exist_ok=True)
    with open(f'{results_dir}/gpu_saturated_results.json', 'w') as f:
        json.dump({
            'mode': args.mode,
            'dataset': args.dataset,
            'sparsity': args.sparsity,
            'best_accuracy': results['best_accuracy'],
            'best_patchable': results['best_patchable'],
            'best_efficient': results['best_efficient'],
            'summary_stats': {
                'total_experiments': len(results['all_results']),
                'avg_accuracy': np.mean([r['accuracy'] for r in results['all_results']]),
                'avg_patchability': np.mean([r['patchability'] for r in results['all_results']]),
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_dir}/gpu_saturated_results.json")
    print("\n🎉 Seed hunt completed!")
    print("Use the best seeds for your patching experiments.")

if __name__ == "__main__":
    main()
