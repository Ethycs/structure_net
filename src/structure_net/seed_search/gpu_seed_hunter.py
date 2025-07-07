#!/usr/bin/env python3
"""
GPU Seed Hunter - Refactored with Canonical Standard

Maximally utilize GPU for parallel seed exploration using the canonical
model standard for perfect compatibility across the project.
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
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple

# Import the canonical standard
from ..core.model_io import (
    create_standard_network, 
    save_model_seed, 
    load_model_seed,
    sort_all_network_layers,
    get_network_stats
)
from .architecture_generator import ArchitectureGenerator


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


class ModelCheckpointer:
    """Save promising models using the canonical standard"""
    
    def __init__(self, save_dir="data/promising_models", dataset="mnist", run_args=None):
        self.dataset = dataset.lower()
        
        # Create datetime-based subdirectory
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
        print(f"   Using canonical model standard")
    
    def save_promising_model(self, model: nn.Sequential, architecture: List[int], 
                           seed: int, metrics: Dict[str, Any], 
                           optimizer: Optional[torch.optim.Optimizer] = None) -> Optional[List[str]]:
        """Save model using canonical standard if it's best in any category"""
        
        # Calculate metrics for comparison
        accuracy = metrics['accuracy']
        patchability = metrics.get('patchability', 0)
        network_stats = get_network_stats(model)
        efficiency = accuracy / network_stats['total_parameters']
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
            # Build filename with category scope
            base_filename = f"model_{self.dataset}_{len(architecture)}layers_seed{seed}_acc{accuracy:.2f}_patch{patchability:.3f}_sparse{sparsity:.3f}"
            category_suffix = f"_BEST_{category.upper()}_{scope}"
            filename = f"{base_filename}{category_suffix}.pt"
            
            filepath = os.path.join(self.run_dir, filename)
            
            # Use canonical save function
            save_model_seed(
                model=model,
                architecture=architecture,
                seed=seed,
                metrics=metrics,
                filepath=filepath,
                optimizer=optimizer
            )
            
            saved_files.append(filepath)
            
            if scope == 'GLOBAL':
                print(f"      💾 New global best {category}: {filename}")
            else:
                print(f"      💾 New sparsity {sparsity:.1%} best {category}: {filename}")
        
        # Print summary of why this model was saved
        if save_reasons:
            print(f"      🎯 Save reasons: {'; '.join(save_reasons)}")
        
        return saved_files


class GPUSeedHunter:
    """
    GPU-accelerated seed hunter using canonical model standard.
    
    This class provides the same functionality as the original GPU seed hunter
    but uses the canonical model standard for perfect compatibility.
    """
    
    def __init__(self, num_gpus: int = 1, device: str = 'cuda', 
                 save_promising: bool = True, dataset: str = 'mnist', 
                 save_threshold: float = 0.25, keep_top_k: int = 3):
        self.num_gpus = num_gpus
        self.device = device
        self.save_promising = save_promising
        self.dataset = dataset.lower()
        self.save_threshold = save_threshold
        self.keep_top_k = keep_top_k
        
        # Dataset-specific parameters
        if self.dataset == 'cifar10':
            self.input_size = 3072  # 32*32*3
            self.num_classes = 10
        else:  # mnist
            self.input_size = 784   # 28*28
            self.num_classes = 10
        
        # GPU saturation parameters
        self.batch_size = self._find_max_batch_size()
        self.num_streams = 4  # CUDA streams per GPU
        self.parallel_models = 8  # Models training simultaneously
        
        # Create CUDA streams
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        
        # Results queue
        self.results_queue = queue.Queue()
        
        # Architecture generator
        self.arch_generator = ArchitectureGenerator(self.input_size, self.num_classes)
        
        # Model checkpointer for saving promising models
        if self.save_promising:
            self.checkpointer = ModelCheckpointer(
                dataset=self.dataset, 
                run_args=getattr(self, '_run_args', None)
            )
        
        # Sparsity sweep configuration
        self.sparsity_config = SparsitySweepConfig()
        
        print(f"🚀 GPU Seed Hunter initialized (Canonical Standard)")
        print(f"   Dataset: {self.dataset.upper()}")
        print(f"   Input size: {self.input_size}")
        print(f"   Device: {self.device}")
        print(f"   Max batch size: {self.batch_size}")
        print(f"   CUDA streams: {self.num_streams}")
        print(f"   Parallel models: {self.parallel_models}")
        print(f"   Model saving: {'Enabled' if self.save_promising else 'Disabled'}")
        if self.save_promising:
            print(f"   Save threshold: {self.save_threshold:.1%} accuracy")
        
    def _find_max_batch_size(self) -> int:
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
            
            # Convert to tensors and move to GPU (EXACT GPU seed hunter preprocessing)
            train_x = torch.from_numpy(train_dataset.data).float().reshape(-1, 3072) / 255.0
            train_y = torch.tensor(train_dataset.targets)
            test_x = torch.from_numpy(test_dataset.data).float().reshape(-1, 3072) / 255.0
            test_y = torch.tensor(test_dataset.targets)
            
            # Normalize EXACTLY like GPU seed hunter
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
            
            # Convert to tensors
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
    
    def get_cached_dataset(self) -> Dict[str, torch.Tensor]:
        """Return cached dataset"""
        return self.cached_dataset

    def calculate_extrema_score(self, model: nn.Sequential) -> float:
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
                    from ..core.model_io import StandardSparseLayer
                    if isinstance(module, StandardSparseLayer):
                        activations.append(output)
                
                hooks = []
                for layer in model:
                    from ..core.model_io import StandardSparseLayer
                    if isinstance(layer, StandardSparseLayer):
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
    
    def parallel_seed_test(self, architecture: List[int], seed: int, 
                          stream_idx: int, sparsity: float = 0.02, 
                          epochs: int = 5) -> Dict[str, Any]:
        """Test single seed on GPU with given stream"""
        
        if torch.cuda.is_available():
            with torch.cuda.stream(self.streams[stream_idx]):
                return self._test_seed_impl(architecture, seed, sparsity, epochs)
        else:
            return self._test_seed_impl(architecture, seed, sparsity, epochs)
    
    def _test_seed_impl(self, architecture: List[int], seed: int, 
                       sparsity: float, epochs: int = 15) -> Dict[str, Any]:
        """Implementation of seed testing using canonical standard"""
        
        # Get sorting configuration from hunter instance
        disable_sorting = getattr(self, '_disable_sorting', False)
        sort_frequency = getattr(self, '_sort_frequency', 5)
        
        # Create network using canonical standard
        model = create_standard_network(
            architecture=architecture,
            sparsity=sparsity,
            seed=seed,
            device=self.device
        )
        
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

            # Neuron sorting maintenance using canonical function
            if not disable_sorting and (epoch + 1) % sort_frequency == 0 and epoch < epochs - 1:
                if should_print:
                    print(f"    🔄 Epoch {epoch+1}: Performing maintenance sort...")
                sort_all_network_layers(model)
            
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
        
        # Get network statistics using canonical function
        network_stats = get_network_stats(model)
        
        return {
            'architecture': architecture,
            'seed': seed,
            'accuracy': best_acc,
            'extrema_score': extrema_score,
            'parameters': network_stats['total_parameters'],
            'patchability': extrema_score * (1 - best_acc),  # High extrema + low acc = patchable
            'sparsity': sparsity,
            'epochs': epochs,
            'sorted': not disable_sorting,  # Flag to indicate if this model used neuron sorting
            'sort_frequency': sort_frequency if not disable_sorting else None
        }
    
    def gpu_saturated_search(self, num_architectures: int = 50, 
                           seeds_per_arch: int = 20, 
                           sparsity: float = 0.02) -> Dict[str, Any]:
        """Saturate GPU with parallel seed searches (single sparsity)"""
        
        print(f"\n🚀 GPU Saturated Seed Hunt (Canonical Standard)")
        print(f"   Architectures: {num_architectures}")
        print(f"   Seeds per arch: {seeds_per_arch}")
        print(f"   Total experiments: {num_architectures * seeds_per_arch}")
        print(f"   Sparsity: {sparsity:.1%}")
        
        # Generate architectures using canonical generator
        architectures = self.arch_generator.generate_systematic_batch(num_architectures)
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
    
    def analyze_seed_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find best seeds by different criteria and save using canonical standard"""
        
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
        
        # Save top models using canonical standard
        if hasattr(self, 'checkpointer') and self.save_promising:
            print("\n💾 Saving top models using canonical standard...")
            
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
                    # Recreate the model using canonical standard
                    model = create_standard_network(
                        architecture=result['architecture'],
                        sparsity=result['sparsity'],
                        seed=result['seed'],
                        device=self.device
                    )
                    
                    # Add category and epoch info to metrics
                    metrics_with_info = result.copy()
                    metrics_with_info['epoch'] = 5
                    metrics_with_info['category'] = f'best_{category}'
                    
                    filepaths = self.checkpointer.save_promising_model(
                        model, 
                        result['architecture'], 
                        result['seed'], 
                        metrics_with_info
                    )
                    if filepaths:
                        print(f"      ✅ Saved: {len(filepaths)} files")
                    
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
