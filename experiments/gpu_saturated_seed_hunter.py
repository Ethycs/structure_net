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

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

class ModelCheckpointer:
    """Save promising models for future experiments"""
    
    def __init__(self, save_dir="data/promising_models", dataset="mnist"):
        self.save_dir = save_dir
        self.dataset = dataset.lower()
        os.makedirs(save_dir, exist_ok=True)
    
    def save_promising_model(self, model, architecture, seed, metrics, optimizer=None):
        """Save the complete model state for future experiments"""
        
        checkpoint = {
            # Model state
            'model_state_dict': model.state_dict(),
            'architecture': architecture,
            'seed': seed,
            
            # Training state
            'epoch': metrics.get('epoch', 0),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            
            # Performance metrics
            'accuracy': metrics['accuracy'],
            'patchability_score': metrics.get('patchability', 0),
            'extrema_counts': metrics.get('extrema_score', 0),
            
            # Neuron analysis
            'dead_neurons': metrics.get('dead_neurons', 0),
            'saturated_neurons': metrics.get('saturated_neurons', 0),
            'activation_patterns': metrics.get('activation_patterns'),
            
            # Reproducibility
            'torch_version': torch.__version__,
            'random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        
        # Build filename with category if present
        base_filename = f"model_{self.dataset}_{len(architecture)}layers_seed{seed}_acc{metrics['accuracy']:.2f}_patch{metrics.get('patchability', 0):.3f}"
        
        # Add category suffix if present
        if 'category' in metrics:
            category_suffix = f"_{metrics['category'].upper()}"
            filename = f"{base_filename}{category_suffix}.pt"
        else:
            filename = f"{base_filename}.pt"
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        
        return filepath
    
    def load_model(self, filepath, model_class=None):
        """Load a saved model checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if model_class is None:
            # Reconstruct model from architecture
            architecture = checkpoint['architecture']
            model = self._reconstruct_model(architecture)
        else:
            model = model_class()
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint
    
    def _reconstruct_model(self, architecture):
        """Reconstruct model from architecture specification"""
        layers = []
        
        for i in range(len(architecture) - 1):
            layer = nn.Linear(architecture[i], architecture[i+1])
            layers.append(layer)
            if i < len(architecture) - 2:  # No ReLU after last layer
                layers.append(nn.ReLU())
                
        return nn.Sequential(*layers)

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

class GPUSaturatedSeedHunter:
    """
    Maximally utilize GPU for parallel seed exploration with sparsity sweeping
    """
    
    def __init__(self, num_gpus=1, device='cuda', save_promising=True, dataset='mnist', save_threshold=0.25):
        self.num_gpus = num_gpus
        self.device = device
        self.save_promising = save_promising
        self.dataset = dataset.lower()
        self.save_threshold = save_threshold  # Minimum accuracy threshold for saving
        
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
            self.checkpointer = ModelCheckpointer(dataset=self.dataset)
        
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
        """Create batch of different seed architectures"""
        
        architectures = []
        
        # Type 1: Direct connections [input_size, C]
        for c in [10, 20, 30, 40, 50]:
            architectures.append([self.input_size, c])
            
        # Type 2: Single hidden [input_size, H, C]
        for h in [16, 32, 64, 128, 256, 512]:
            architectures.append([self.input_size, h, self.num_classes])
            
        # Type 3: Double hidden [input_size, H1, H2, C]
        for h1 in [128, 256, 512]:
            for h2 in [32, 64, 128]:
                if h2 < h1:  # Decreasing size
                    architectures.append([self.input_size, h1, h2, self.num_classes])
                    
        # Type 4: Triple hidden [input_size, H1, H2, H3, C]
        for h1 in [256, 512]:
            for h2 in [128, 256]:
                for h3 in [32, 64]:
                    if h3 < h2 < h1:
                        architectures.append([self.input_size, h1, h2, h3, self.num_classes])
                        
        # Type 5: Wide shallow [input_size, W, C]
        for w in [1024, 2048]:
            architectures.append([self.input_size, w, self.num_classes])
            
        # Type 6: Narrow deep
        narrow_arch = [self.input_size]
        for depth in range(5):
            narrow_arch.append(64)
        narrow_arch.append(self.num_classes)
        architectures.append(narrow_arch)
        
        return architectures[:num_seeds]
    
    def create_sparse_network(self, architecture, sparsity=0.02, seed=None):
        """Create sparse network with given architecture and seed"""
        if seed is not None:
            torch.manual_seed(seed)
            
        layers = []
        
        for i in range(len(architecture) - 1):
            # Create sparse linear layer
            layer = nn.Linear(architecture[i], architecture[i+1])
            
            # Apply sparsity mask
            mask = torch.rand_like(layer.weight) < sparsity
            with torch.no_grad():
                layer.weight *= mask.float()
                
            # Register mask as buffer
            layer.register_buffer('mask', mask)
            
            layers.append(layer)
            if i < len(architecture) - 2:  # No ReLU after last layer
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
                    if isinstance(module, nn.Linear):
                        activations.append(output)
                
                hooks = []
                for layer in model:
                    if isinstance(layer, nn.Linear):
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
    
    def _test_seed_impl(self, architecture, seed, sparsity, epochs=5):
        """Implementation of seed testing with configurable epochs"""
        torch.manual_seed(seed)
        
        # Create sparse network
        model = self.create_sparse_network(architecture, sparsity=sparsity, seed=seed)
        model = model.to(self.device)
        
        # Use mixed precision for speed if available
        if torch.cuda.is_available():
            scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training with configurable epochs
        dataset = self.get_cached_dataset()
        
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
                if torch.cuda.is_available():
                    with autocast():
                        output = model(x)
                        loss = nn.functional.cross_entropy(output, y)
                    
                    # Backward
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(x)
                    loss = nn.functional.cross_entropy(output, y)
                    loss.backward()
                    optimizer.step()
            
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
            'sparsity': sparsity
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
                    
                    # Print and save promising seeds immediately
                    if result['accuracy'] > self.save_threshold:
                        print(f"🌟 Promising: {result['architecture']} "
                              f"seed={result['seed']} "
                              f"acc={result['accuracy']:.2%} "
                              f"patch_score={result['patchability']:.3f}")
                        
                        # Save promising model if enabled
                        if self.save_promising:
                            # Recreate the model to save it
                            torch.manual_seed(result['seed'])
                            model = self.create_sparse_network(
                                result['architecture'], 
                                sparsity=result['sparsity'], 
                                seed=result['seed']
                            )
                            
                            # Add epoch info to metrics
                            metrics_with_epoch = result.copy()
                            metrics_with_epoch['epoch'] = 5  # We trained for 5 epochs
                            
                            filepath = self.checkpointer.save_promising_model(
                                model, 
                                result['architecture'], 
                                result['seed'], 
                                metrics_with_epoch
                            )
                            print(f"💾 Saved to: {filepath}")
        
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
    
    def hybrid_sparsity_sweep(self, num_architectures=30, seeds_per_arch=10, sweep_mode='hybrid'):
        """Run hybrid sparsity sweep: coarse then fine exploration"""
        
        print(f"\n🌊 HYBRID SPARSITY SWEEP")
        print(f"   Mode: {sweep_mode}")
        print(f"   Architectures: {num_architectures}")
        print(f"   Seeds per arch: {seeds_per_arch}")
        
        # Pre-cache dataset
        print("📦 Caching dataset...")
        self.cache_dataset_gpu()
        
        all_results = {
            'phase_1_coarse': {},
            'phase_2_fine': {},
            'best_combinations': {}
        }
        
        # Phase 1: Coarse sweep
        print(f"\n🔍 PHASE 1: COARSE SPARSITY SWEEP")
        print(f"   Sparsities: {self.sparsity_config.COARSE_SPARSITIES}")
        print(f"   Epochs per experiment: {self.sparsity_config.COARSE_EPOCHS}")
        
        phase1_results = []
        architectures = self.create_seed_batch(num_architectures)
        
        for sparsity in self.sparsity_config.COARSE_SPARSITIES:
            print(f"\n🎯 Testing sparsity {sparsity:.3f}")
            sparsity_results = self._run_sparsity_batch(
                architectures, seeds_per_arch, sparsity, 
                epochs=self.sparsity_config.COARSE_EPOCHS
            )
            
            all_results['phase_1_coarse'][f'sparsity_{sparsity}'] = sparsity_results
            phase1_results.extend(sparsity_results)
            
            # Quick analysis
            avg_acc = np.mean([r['accuracy'] for r in sparsity_results])
            avg_patch = np.mean([r['patchability'] for r in sparsity_results])
            print(f"   📊 Sparsity {sparsity:.3f}: avg_acc={avg_acc:.3f}, avg_patch={avg_patch:.3f}")
        
        if sweep_mode == 'coarse':
            print(f"\n✅ Coarse sweep completed!")
            return self._analyze_sparsity_results(all_results, phase='coarse')
        
        # Analyze Phase 1 to determine promising ranges
        print(f"\n📊 ANALYZING PHASE 1 RESULTS...")
        promising_ranges = self._identify_promising_sparsity_ranges(phase1_results)
        
        if not promising_ranges:
            print("⚠️  No promising sparsity ranges found. Returning coarse results.")
            return self._analyze_sparsity_results(all_results, phase='coarse')
        
        # Phase 2: Fine sweep on promising ranges
        print(f"\n🔬 PHASE 2: FINE SPARSITY SWEEP")
        print(f"   Promising ranges: {list(promising_ranges.keys())}")
        print(f"   Epochs per experiment: {self.sparsity_config.FINE_EPOCHS}")
        
        # Select top architectures from Phase 1
        top_architectures = self._select_top_architectures(phase1_results, top_k=10)
        print(f"   Selected top {len(top_architectures)} architectures for fine sweep")
        
        for range_name, sparsities in promising_ranges.items():
            print(f"\n🎯 Fine sweep: {range_name} range {sparsities}")
            
            range_results = []
            for sparsity in sparsities:
                sparsity_results = self._run_sparsity_batch(
                    top_architectures, seeds_per_arch, sparsity,
                    epochs=self.sparsity_config.FINE_EPOCHS
                )
                range_results.extend(sparsity_results)
            
            all_results['phase_2_fine'][range_name] = range_results
            
            # Quick analysis
            avg_acc = np.mean([r['accuracy'] for r in range_results])
            avg_patch = np.mean([r['patchability'] for r in range_results])
            print(f"   📊 {range_name}: avg_acc={avg_acc:.3f}, avg_patch={avg_patch:.3f}")
        
        print(f"\n✅ Hybrid sparsity sweep completed!")
        return self._analyze_sparsity_results(all_results, phase='hybrid')
    
    def _run_sparsity_batch(self, architectures, seeds_per_arch, sparsity, epochs=5):
        """Run a batch of experiments for a specific sparsity level"""
        results = []
        
        # Process architectures in batches
        for arch_idx in range(0, len(architectures), self.parallel_models):
            arch_batch = architectures[arch_idx:arch_idx + self.parallel_models]
            
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
                            sparsity,
                            epochs
                        )
                        futures.append(future)
                
                # Collect results
                for future in futures:
                    result = future.result()
                    results.append(result)
                    
                    # Save very promising models
                    if result['accuracy'] > self.save_threshold and self.save_promising:
                        torch.manual_seed(result['seed'])
                        model = self.create_sparse_network(
                            result['architecture'], 
                            sparsity=result['sparsity'], 
                            seed=result['seed']
                        )
                        
                        metrics_with_epoch = result.copy()
                        metrics_with_epoch['epoch'] = epochs
                        
                        filepath = self.checkpointer.save_promising_model(
                            model, result['architecture'], result['seed'], metrics_with_epoch
                        )
        
        return results
    
    def _identify_promising_sparsity_ranges(self, phase1_results):
        """Identify promising sparsity ranges from Phase 1 results"""
        promising_ranges = {}
        
        # Group results by sparsity
        sparsity_groups = {}
        for result in phase1_results:
            sparsity = result['sparsity']
            if sparsity not in sparsity_groups:
                sparsity_groups[sparsity] = []
            sparsity_groups[sparsity].append(result)
        
        # Analyze each sparsity level
        for sparsity, results in sparsity_groups.items():
            avg_accuracy = np.mean([r['accuracy'] for r in results])
            avg_patchability = np.mean([r['patchability'] for r in results])
            max_patchability = max([r['patchability'] for r in results])
            
            print(f"   Sparsity {sparsity:.3f}: avg_acc={avg_accuracy:.3f}, "
                  f"avg_patch={avg_patchability:.3f}, max_patch={max_patchability:.3f}")
            
            # Determine which fine range this sparsity belongs to
            if sparsity <= 0.005 and (avg_accuracy >= self.sparsity_config.MIN_ACCURACY_THRESHOLD or 
                                     max_patchability >= self.sparsity_config.MIN_PATCHABILITY_THRESHOLD):
                promising_ranges['ultra_sparse'] = self.sparsity_config.FINE_RANGES['ultra_sparse']
            elif 0.005 < sparsity <= 0.05 and (avg_accuracy >= self.sparsity_config.MIN_ACCURACY_THRESHOLD or 
                                              max_patchability >= self.sparsity_config.MIN_PATCHABILITY_THRESHOLD):
                promising_ranges['moderate_sparse'] = self.sparsity_config.FINE_RANGES['moderate_sparse']
            elif sparsity > 0.05 and (avg_accuracy >= self.sparsity_config.MIN_ACCURACY_THRESHOLD or 
                                     max_patchability >= self.sparsity_config.MIN_PATCHABILITY_THRESHOLD):
                promising_ranges['dense_sparse'] = self.sparsity_config.FINE_RANGES['dense_sparse']
        
        return promising_ranges
    
    def _select_top_architectures(self, results, top_k=10):
        """Select top architectures based on Phase 1 performance"""
        # Group by architecture and calculate average performance
        arch_performance = {}
        for result in results:
            arch_key = str(result['architecture'])
            if arch_key not in arch_performance:
                arch_performance[arch_key] = []
            arch_performance[arch_key].append(result['patchability'] + result['accuracy'])
        
        # Calculate average performance per architecture
        arch_avg_performance = {
            arch: np.mean(scores) for arch, scores in arch_performance.items()
        }
        
        # Select top architectures
        top_arch_keys = sorted(arch_avg_performance.keys(), 
                              key=lambda x: arch_avg_performance[x], reverse=True)[:top_k]
        
        # Convert back to architecture lists
        top_architectures = [eval(arch_key) for arch_key in top_arch_keys]
        
        return top_architectures
    
    def _analyze_sparsity_results(self, all_results, phase='hybrid'):
        """Analyze sparsity sweep results"""
        print(f"\n📊 SPARSITY SWEEP ANALYSIS ({phase.upper()})")
        print("=" * 60)
        
        # Collect all results
        all_experiments = []
        
        if 'phase_1_coarse' in all_results:
            for sparsity_key, results in all_results['phase_1_coarse'].items():
                all_experiments.extend(results)
        
        if 'phase_2_fine' in all_results:
            for range_key, results in all_results['phase_2_fine'].items():
                all_experiments.extend(results)
        
        if not all_experiments:
            print("⚠️  No results to analyze!")
            return all_results
        
        # Find best combinations
        by_accuracy = sorted(all_experiments, key=lambda x: x['accuracy'], reverse=True)
        by_patchability = sorted(all_experiments, key=lambda x: x['patchability'], reverse=True)
        by_efficiency = sorted(all_experiments, key=lambda x: x['accuracy']/x['parameters'], reverse=True)
        
        # Best per sparsity level
        sparsity_best = {}
        for result in all_experiments:
            sparsity = result['sparsity']
            if sparsity not in sparsity_best or result['patchability'] > sparsity_best[sparsity]['patchability']:
                sparsity_best[sparsity] = result
        
        print(f"🏆 BEST OVERALL RESULTS:")
        print(f"   Best accuracy: {by_accuracy[0]['accuracy']:.3f} at sparsity {by_accuracy[0]['sparsity']:.3f}")
        print(f"   Best patchability: {by_patchability[0]['patchability']:.3f} at sparsity {by_patchability[0]['sparsity']:.3f}")
        print(f"   Best efficiency: {by_efficiency[0]['accuracy']/by_efficiency[0]['parameters']*1000:.3f} at sparsity {by_efficiency[0]['sparsity']:.3f}")
        
        print(f"\n🎯 BEST PER SPARSITY LEVEL:")
        for sparsity in sorted(sparsity_best.keys()):
            result = sparsity_best[sparsity]
            print(f"   Sparsity {sparsity:.3f}: acc={result['accuracy']:.3f}, "
                  f"patch={result['patchability']:.3f}, arch={result['architecture']}")
        
        # Store best combinations
        all_results['best_combinations'] = {
            'best_accuracy': by_accuracy[0],
            'best_patchability': by_patchability[0],
            'best_efficiency': by_efficiency[0],
            'best_per_sparsity': sparsity_best
        }
        
        return all_results
    
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
        print(f"\n🔍 CHECKING BEST MODEL SAVING...")
        print(f"   Has checkpointer: {hasattr(self, 'checkpointer')}")
        print(f"   Save promising enabled: {self.save_promising}")
        
        if hasattr(self, 'checkpointer') and self.save_promising:
            print("\n💾 Saving top models with category markers...")
            
            # Save best models from each category
            top_models = {
                'accuracy': by_accuracy[0],
                'patchability': by_patchability[0], 
                'efficiency': by_efficiency[0]
            }
            
            print(f"   Top models to save: {len(top_models)}")
            
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

def run_saturated_seed_hunt(dataset='mnist'):
    """Main function to run the seed hunt"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hunter = GPUSaturatedSeedHunter(num_gpus=1, device=device, dataset=dataset)
    
    results = hunter.gpu_saturated_search(
        num_architectures=30,
        seeds_per_arch=10,
        sparsity=0.02
    )
    
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
    results_dir = f'data/seed_hunt_results_{dataset}'
    os.makedirs(results_dir, exist_ok=True)
    with open(f'{results_dir}/gpu_saturated_results.json', 'w') as f:
        json.dump({
            'dataset': dataset,
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
    
    return results

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Saturated Seed Hunter with Hybrid Sparsity Sweep')
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'coarse', 'hybrid'],
                       help='Search mode: single sparsity, coarse sweep, or hybrid sweep (default: single)')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--num-architectures', type=int, default=30,
                       help='Number of architectures to test (default: 30)')
    parser.add_argument('--seeds-per-arch', type=int, default=10,
                       help='Number of seeds per architecture (default: 10)')
    parser.add_argument('--sparsity', type=float, default=0.02,
                       help='Sparsity level for single mode (default: 0.02)')
    parser.add_argument('--thresh', type=float, default=0.25,
                       help='Patchability threshold for saving models (default: 0.25 = 25%%)')
    
    args = parser.parse_args()
    
    print("🔍 GPU Saturated Seed Hunter with Hybrid Sparsity Sweep")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Architectures: {args.num_architectures}")
    print(f"Seeds per arch: {args.seeds_per_arch}")
    if args.mode == 'single':
        print(f"Sparsity: {args.sparsity:.1%}")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hunter = GPUSaturatedSeedHunter(num_gpus=1, device=device, dataset=args.dataset, save_threshold=args.thresh)
    
    if args.mode == 'single':
        # Single sparsity search
        results = hunter.gpu_saturated_search(
            num_architectures=args.num_architectures,
            seeds_per_arch=args.seeds_per_arch,
            sparsity=args.sparsity
        )
        
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
        
    else:
        # Sparsity sweep (coarse or hybrid)
        results = hunter.hybrid_sparsity_sweep(
            num_architectures=args.num_architectures,
            seeds_per_arch=args.seeds_per_arch,
            sweep_mode=args.mode
        )
        
        print("\n🎊 SPARSITY SWEEP SUMMARY")
        print("="*50)
        
        if 'best_combinations' in results:
            best_combos = results['best_combinations']
            
            print(f"🏆 BEST OVERALL COMBINATIONS:")
            
            best_acc = best_combos['best_accuracy']
            print(f"\n🎯 Best Accuracy: {best_acc['accuracy']:.3f}")
            print(f"   Architecture: {best_acc['architecture']}")
            print(f"   Sparsity: {best_acc['sparsity']:.3f}")
            print(f"   Seed: {best_acc['seed']}")
            
            best_patch = best_combos['best_patchability']
            print(f"\n🔧 Best Patchability: {best_patch['patchability']:.3f}")
            print(f"   Architecture: {best_patch['architecture']}")
            print(f"   Sparsity: {best_patch['sparsity']:.3f}")
            print(f"   Seed: {best_patch['seed']}")
            print(f"   Accuracy: {best_patch['accuracy']:.3f}")
            
            best_eff = best_combos['best_efficiency']
            eff_score = best_eff['accuracy'] / best_eff['parameters'] * 1000
            print(f"\n⚡ Best Efficiency: {eff_score:.3f} acc/kparam")
            print(f"   Architecture: {best_eff['architecture']}")
            print(f"   Sparsity: {best_eff['sparsity']:.3f}")
            print(f"   Seed: {best_eff['seed']}")
            
            print(f"\n🎯 BEST PER SPARSITY LEVEL:")
            for sparsity, result in sorted(best_combos['best_per_sparsity'].items()):
                print(f"   Sparsity {sparsity:.3f}: acc={result['accuracy']:.3f}, "
                      f"patch={result['patchability']:.3f}")
        
        # Save sparsity sweep results
        results_dir = f'data/sparsity_sweep_results_{args.dataset}'
        os.makedirs(results_dir, exist_ok=True)
        with open(f'{results_dir}/sparsity_sweep_{args.mode}.json', 'w') as f:
            json.dump({
                'mode': args.mode,
                'dataset': args.dataset,
                'results': results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Sparsity sweep results saved to {results_dir}/sparsity_sweep_{args.mode}.json")
    
    print("\n🎉 Seed hunt completed!")
    if args.mode == 'single':
        print("Use the best seeds for your patching experiments.")
    else:
        print("Use the best sparsity-architecture combinations for your experiments.")

if __name__ == "__main__":
    main()
