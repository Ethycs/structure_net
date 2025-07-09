#!/usr/bin/env python3
"""
Analyze and rank promising models to help choose the best seed for stress testing.

This script analyzes all models in the promising_models directory and ranks them
based on various criteria like accuracy, efficiency, patchability, and architecture.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd


def parse_model_filename(filename):
    """Extract model metadata from filename."""
    # Example: model_cifar10_3layers_seed1_acc0.47_patch0.188_sparse0.100_BEST_ACCURACY_SPARSITY_0.100.pt
    
    pattern = r'model_cifar10_(\d+)layers_seed(\d+)_acc([\d.]+)_patch([\d.]+)_sparse([\d.]+)_(.+)\.pt'
    match = re.match(pattern, filename)
    
    if match:
        layers = int(match.group(1))
        seed = int(match.group(2))
        accuracy = float(match.group(3))
        patchability = float(match.group(4))
        sparsity = float(match.group(5))
        category = match.group(6)
        
        # Calculate efficiency (accuracy per connection)
        # Approximate connections based on layers and sparsity
        if layers == 2:
            base_connections = 3072 * 128 + 128 * 10  # CIFAR-10 2-layer
        elif layers == 3:
            base_connections = 3072 * 128 + 128 * 64 + 64 * 10  # CIFAR-10 3-layer
        elif layers == 4:
            base_connections = 3072 * 128 + 128 * 64 + 64 * 32 + 32 * 10  # CIFAR-10 4-layer
        else:
            base_connections = 1000000  # Default estimate
            
        actual_connections = base_connections * sparsity
        efficiency = accuracy / (actual_connections / 1000000)  # Accuracy per million connections
        
        return {
            'filename': filename,
            'layers': layers,
            'seed': seed,
            'accuracy': accuracy,
            'patchability': patchability,
            'sparsity': sparsity,
            'category': category,
            'efficiency': efficiency,
            'connections': actual_connections
        }
    return None


def analyze_models():
    """Analyze all models in the promising_models directory."""
    models_dir = Path("data/promising_models")
    
    all_models = []
    
    # Recursively find all .pt files
    for model_path in models_dir.rglob("*.pt"):
        model_info = parse_model_filename(model_path.name)
        if model_info:
            model_info['path'] = str(model_path)
            model_info['directory'] = model_path.parent.name
            all_models.append(model_info)
    
    if not all_models:
        print("No models found in data/promising_models/")
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_models)
    
    return df


def rank_models(df, criteria='balanced'):
    """
    Rank models based on different criteria.
    
    Criteria options:
    - 'accuracy': Pure accuracy
    - 'efficiency': Accuracy per connection
    - 'patchability': How well the model responds to patches
    - 'balanced': Weighted combination of all factors
    - 'growth_potential': Best for growth experiments (high patchability, moderate accuracy)
    """
    
    df = df.copy()
    
    if criteria == 'accuracy':
        df['score'] = df['accuracy']
    
    elif criteria == 'efficiency':
        df['score'] = df['efficiency']
    
    elif criteria == 'patchability':
        df['score'] = df['patchability']
    
    elif criteria == 'growth_potential':
        # High patchability + moderate accuracy + low sparsity (room to grow)
        df['score'] = (
            df['patchability'] * 0.4 +  # High weight on patchability
            df['accuracy'] * 0.3 +       # Moderate weight on accuracy
            (1 - df['sparsity']) * 0.3  # Prefer less sparse (more room to grow)
        )
    
    elif criteria == 'balanced':
        # Normalize each metric to 0-1 range
        acc_norm = (df['accuracy'] - df['accuracy'].min()) / (df['accuracy'].max() - df['accuracy'].min())
        eff_norm = (df['efficiency'] - df['efficiency'].min()) / (df['efficiency'].max() - df['efficiency'].min())
        patch_norm = (df['patchability'] - df['patchability'].min()) / (df['patchability'].max() - df['patchability'].min())
        
        # Weighted combination
        df['score'] = (
            acc_norm * 0.4 +      # 40% weight on accuracy
            eff_norm * 0.3 +      # 30% weight on efficiency
            patch_norm * 0.3      # 30% weight on patchability
        )
    
    else:
        raise ValueError(f"Unknown criteria: {criteria}")
    
    # Sort by score
    df_sorted = df.sort_values('score', ascending=False)
    
    return df_sorted


def print_recommendations(df_ranked, top_n=10):
    """Print model recommendations."""
    print(f"\n🏆 Top {top_n} Recommended Models:")
    print("=" * 100)
    print(f"{'Rank':<5} {'Layers':<7} {'Seed':<6} {'Accuracy':<9} {'Patch':<7} {'Sparse':<8} {'Eff':<8} {'Score':<7} {'Category':<20}")
    print("-" * 100)
    
    for i, row in df_ranked.head(top_n).iterrows():
        print(f"{df_ranked.index.get_loc(i)+1:<5} "
              f"{row['layers']:<7} "
              f"{row['seed']:<6} "
              f"{row['accuracy']:<9.2%} "
              f"{row['patchability']:<7.3f} "
              f"{row['sparsity']:<8.3f} "
              f"{row['efficiency']:<8.2f} "
              f"{row['score']:<7.3f} "
              f"{row['category'][:20]:<20}")


def main():
    print("🔍 Analyzing Promising Models")
    print("=" * 60)
    
    # Analyze all models
    df = analyze_models()
    
    if df is None:
        return
    
    print(f"\n📊 Found {len(df)} models")
    print(f"   Architectures: {sorted(df['layers'].unique())} layers")
    print(f"   Accuracy range: {df['accuracy'].min():.1%} - {df['accuracy'].max():.1%}")
    print(f"   Sparsity range: {df['sparsity'].min():.1%} - {df['sparsity'].max():.1%}")
    
    # Show statistics by architecture
    print("\n📈 Statistics by Architecture:")
    arch_stats = df.groupby('layers').agg({
        'accuracy': ['count', 'mean', 'max'],
        'efficiency': 'mean',
        'patchability': 'mean'
    }).round(3)
    print(arch_stats)
    
    # Rank by different criteria
    criteria_list = ['accuracy', 'efficiency', 'growth_potential', 'balanced']
    
    for criteria in criteria_list:
        print(f"\n\n🎯 Ranking by: {criteria.upper()}")
        df_ranked = rank_models(df, criteria)
        print_recommendations(df_ranked, top_n=5)
    
    # Special recommendations
    print("\n\n💡 RECOMMENDATIONS:")
    print("=" * 60)
    
    # Best for accuracy
    best_accuracy = df.loc[df['accuracy'].idxmax()]
    print(f"\n🎯 Best for Pure Accuracy:")
    print(f"   {best_accuracy['path']}")
    print(f"   {best_accuracy['layers']} layers, {best_accuracy['accuracy']:.2%} accuracy")
    
    # Best for growth experiments
    df_growth = rank_models(df, 'growth_potential')
    best_growth = df_growth.iloc[0]
    print(f"\n🌱 Best for Growth Experiments:")
    print(f"   {best_growth['path']}")
    print(f"   {best_growth['layers']} layers, {best_growth['accuracy']:.2%} accuracy, {best_growth['patchability']:.3f} patchability")
    
    # Best efficiency
    best_efficiency = df.loc[df['efficiency'].idxmax()]
    print(f"\n⚡ Best for Efficiency:")
    print(f"   {best_efficiency['path']}")
    print(f"   {best_efficiency['layers']} layers, {best_efficiency['efficiency']:.2f} acc/M connections")
    
    # Save full analysis
    output_file = "model_analysis.csv"
    df_ranked = rank_models(df, 'balanced')
    df_ranked.to_csv(output_file, index=False)
    print(f"\n💾 Full analysis saved to: {output_file}")


if __name__ == "__main__":
    main()