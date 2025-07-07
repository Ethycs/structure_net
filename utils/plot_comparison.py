#!/usr/bin/env python3
"""
Plot Comparison of Growth Strategies

This script loads the results from the dual-GPU experiment and generates
a comparative graph of their performance.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison(file1, file2, output_path):
    """
    Loads two experiment result files and plots a comparison of their accuracy.
    """
    # Load data
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)

    # Create DataFrames
    df1 = pd.DataFrame(data1['growth_history'])
    df1['strategy'] = 'Sparse Only'
    
    df2 = pd.DataFrame(data2['growth_history'])
    df2['strategy'] = 'Layers and Patches'

    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.plot(df1['iteration'], df1['accuracy'], marker='o', linestyle='-', label=df1['strategy'].iloc[0])
    plt.plot(df2['iteration'], df2['accuracy'], marker='o', linestyle='-', label=df2['strategy'].iloc[0])
    
    plt.title('Comparison of Growth Strategies: Accuracy vs. Iteration')
    plt.xlabel('Growth Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()
    
    print(f"📊 Comparison chart saved to {output_path}")

def main():
    file_sparse_only = 'data/modern_growth_results_sparse_only.json'
    file_layers_patches = 'data/modern_growth_results_layers_and_patches.json'
    output_file = 'data/comparison_charts/accuracy_comparison_advanced.png'
    
    plot_comparison(file_sparse_only, file_layers_patches, output_file)

if __name__ == "__main__":
    main()
