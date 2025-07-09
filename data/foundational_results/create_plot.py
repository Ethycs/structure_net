
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_file(file_path, content):
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    results = []
    
    category = 'Unknown'
    if 'comparative' in file_path or 'modern' in file_path or 'benchmark' in file_path:
        category = 'Comparative Growth'
    elif 'sparsity_sweep' in file_path or 'single_layer' in file_path or 'full_mnist' in file_path or 'seed_hunt' in file_path:
        category = 'Sparsity Sweep'
    elif 'experiment_1' in file_path or 'max_utilization' in file_path:
        category = 'Complex Experiment'
    elif 'validation' in file_path:
        category = 'Validation'
    elif 'growth_rescue' in file_path:
        category = 'Growth Rescue'
    elif 'fixed_growth' in file_path:
        category = 'Fixed Growth'
    elif 'debug_growth' in file_path:
        category = 'Debug'
        
    dataset = 'mnist'
    if 'cifar10' in file_path:
        dataset = 'cifar10'

    if 'final_accuracy' in data and 'final_sparsity' in data:
        results.append({
            'Category': category,
            'Dataset': dataset,
            'Experiment': os.path.basename(file_path).replace('.json', ''),
            'Architecture': str(data.get('final_architecture', data.get('seed_architecture'))),
            'Accuracy': data.get('final_accuracy'),
            'Sparsity': data.get('final_sparsity'),
        })

    elif 'goldilocks_candidates' in data:
        for item in data['goldilocks_candidates']:
            results.append({
                'Category': category,
                'Dataset': dataset,
                'Experiment': os.path.basename(file_path).replace('.json', ''),
                'Architecture': str(data.get('architecture')),
                'Accuracy': item.get('performance'),
                'Sparsity': item.get('sparsity'),
            })
            
    elif '[256]' in data or '[512, 256, 128]' in data: # full_mnist_results/goldilocks_zone_results.json
        for arch_key, arch_results in data.items():
            for item in arch_results:
                acc = item.get('best_accuracy', 0)
                if acc > 1: acc = acc / 100.0
                results.append({
                    'Category': category,
                    'Dataset': 'mnist',
                    'Experiment': 'goldilocks_zone',
                    'Architecture': str(item.get('architecture')),
                    'Accuracy': acc,
                    'Sparsity': item.get('sparsity'),
                })

    elif 'sparsity_summary' in data: # comprehensive_sparsity_sweep.json
        for sparsity, summary in data['sparsity_summary'].items():
            results.append({
                'Category': category,
                'Dataset': data.get('dataset', dataset),
                'Experiment': 'comprehensive_sparsity_sweep',
                'Architecture': 'Multiple',
                'Accuracy': summary.get('best_accuracy'),
                'Sparsity': float(sparsity),
            })

    elif 'best_accuracy' in data and 'sparsity' in data: # For files like seed_hunt
         acc = data['best_accuracy'].get('accuracy', 0)
         if acc > 1: acc = acc / 100.0
         results.append({
            'Category': category,
            'Dataset': dataset,
            'Experiment': os.path.basename(file_path).replace('.json', ''),
            'Architecture': str(data['best_accuracy'].get('architecture')),
            'Accuracy': acc,
            'Sparsity': data['best_accuracy'].get('sparsity'),
        })

    return results

# List of files to read (from previous steps)
files_to_read = [
    '/home/rabbit/structure_net/data/foundational_results/benchmark_direct_growth.json',
    '/home/rabbit/structure_net/data/foundational_results/comparative_direct_growth.json',
    '/home/rabbit/structure_net/data/foundational_results/comparative_tournament_growth.json',
    '/home/rabbit/structure_net/data/foundational_results/comparative_sparse_baseline.json',
    '/home/rabbit/structure_net/data/foundational_results/cifar10_combined_results/optimal_seed_results_cifar10.json',
    '/home/rabbit/structure_net/data/foundational_results/cifar10_improved_results/results_model_cifar10_4layers_seed4_acc0.47_patch0.065_BEST_ACCURACY.json',
    '/home/rabbit/structure_net/data/foundational_results/debug_growth_results/debug_results.json',
    '/home/rabbit/structure_net/data/foundational_results/debug_growth_results/debug_summary.json',
    '/home/rabbit/structure_net/data/foundational_results/experiment_1_parallel_results/aggregated_results.json',
    '/home/rabbit/structure_net/data/foundational_results/experiment_1_results/experiment_summary.json',
    '/home/rabbit/structure_net/data/foundational_results/fixed_growth_results/fixed_growth_results.json',
    '/home/rabbit/structure_net/data/foundational_results/fixed_growth_results/fixed_growth_summary.json',
    '/home/rabbit/structure_net/data/foundational_results/full_mnist_results/goldilocks_zone_results.json',
    '/home/rabbit/structure_net/data/foundational_results/growth_rescue_results/cliff_rescue_results.json',
    '/home/rabbit/structure_net/data/foundational_results/growth_rescue_results/cliff_rescue_summary.json',
    '/home/rabbit/structure_net/data/foundational_results/growth_results/indefinite_growth_results.json',
    '/home/rabbit/structure_net/data/foundational_results/max_utilization_results/aggregated_results.json',
    '/home/rabbit/structure_net/data/foundational_results/modern_growth_results_layers_and_patches.json',
    '/home/rabbit/structure_net/data/foundational_results/modern_growth_results_sparse_only.json',
    '/home/rabbit/structure_net/data/foundational_results/modern_indefinite_growth_results.json',
    '/home/rabbit/structure_net/data/foundational_results/optimal_bootstrap_results/optimal_seed_results.json',
    '/home/rabbit/structure_net/data/foundational_results/seed_hunt_results_cifar10/comprehensive_sparsity_sweep.json',
    '/home/rabbit/structure_net/data/foundational_results/seed_hunt_results_cifar10/gpu_saturated_results.json',
    '/home/rabbit/structure_net/data/foundational_results/seed_hunt_results/gpu_saturated_results.json',
    '/home/rabbit/structure_net/data/foundational_results/single_layer_results/sparsity_ladder_results.json',
    '/home/rabbit/structure_net/data/foundational_results/single_layer_results/sparsity_ladder_summary.json',
    '/home/rabbit/structure_net/data/foundational_results/sparsity_sweep_results_cifar10/sparsity_sweep_hybrid.json',
    '/home/rabbit/structure_net/data/foundational_results/validation_results/validation_results.json',
    '/home/rabbit/structure_net/data/foundational_results/validation_results/validation_summary.json'
]

all_data = []
for file_path in files_to_read:
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            all_data.extend(parse_file(file_path, content))
    except FileNotFoundError:
        print(f'File not found: {file_path}')
    except Exception as e:
        print(f'Error processing {file_path}: {e}')

df = pd.DataFrame(all_data)
df.dropna(subset=['Accuracy', 'Sparsity'], inplace=True)

# Create plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(data=df, x='Sparsity', y='Accuracy', hue='Dataset', style='Category', s=150, ax=ax, alpha=0.7)

ax.set_title('Accuracy vs. Sparsity Across All Experiments', fontsize=16)
ax.set_xlabel('Sparsity (Lower is Sparser)', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.legend(title='Experiment Type')
plt.tight_layout()
plt.savefig('accuracy_vs_sparsity.png')
print('Plot saved to accuracy_vs_sparsity.png')
