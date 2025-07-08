#!/usr/bin/env python3
"""
Comparative Analysis Plotter

This script loads the results from the comparative experiments and generates
comprehensive side-by-side graphs comparing Direct Growth, Tournament Growth,
and Sparse Baseline strategies.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def load_experiment_data(filepath):
    """Load experiment data from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"⚠️  Invalid JSON in: {filepath}")
        return None

def extract_performance_data(data, experiment_name):
    """Extract performance data from experiment results."""
    if data is None:
        return None
    
    # Handle different data structures
    if 'epoch_performance' in data and data['epoch_performance']:
        # Sparse Baseline format (has epoch_performance with data)
        df = pd.DataFrame(data['epoch_performance'])
        df['experiment'] = experiment_name
        
        # Add iteration column if epoch exists
        if 'epoch' in df.columns:
            df['iteration'] = df['epoch'] // 10  # Group epochs into iterations
        else:
            df['iteration'] = range(len(df))  # Fallback to sequential numbering
        
        # Add connection stats if available
        if 'connection_stats' in data and data['connection_stats']:
            conn_df = pd.DataFrame(data['connection_stats'])
            if len(conn_df) == len(df):
                df['total_connections'] = conn_df['total_connections']
        
        return df
    elif 'growth_history' in data and data['growth_history']:
        # Direct Growth and Tournament format (has growth_history with data)
        df = pd.DataFrame(data['growth_history'])
        df['experiment'] = experiment_name
        
        # Rename 'accuracy' to 'val_performance' for consistency
        if 'accuracy' in df.columns:
            df['val_performance'] = df['accuracy']
        
        return df
    else:
        print(f"⚠️  No usable data found for {experiment_name}")
        print(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        if isinstance(data, dict):
            if 'epoch_performance' in data:
                print(f"   epoch_performance length: {len(data['epoch_performance'])}")
            if 'growth_history' in data:
                print(f"   growth_history length: {len(data['growth_history'])}")
        return None

def plot_comparative_analysis(direct_file, tournament_file, sparse_file, output_dir='data/comparison_charts'):
    """Generate comprehensive comparative analysis plots."""
    
    print("📊 COMPARATIVE ANALYSIS PLOTTER")
    print("=" * 50)
    
    # Load data
    direct_data = load_experiment_data(direct_file)
    tournament_data = load_experiment_data(tournament_file)
    sparse_data = load_experiment_data(sparse_file)
    
    # Debug: Print data structure
    print(f"🔍 Data structure analysis:")
    if direct_data:
        print(f"   Direct data keys: {list(direct_data.keys())}")
    if tournament_data:
        print(f"   Tournament data keys: {list(tournament_data.keys())}")
        if 'epoch_performance' in tournament_data and tournament_data['epoch_performance']:
            sample_entry = tournament_data['epoch_performance'][0]
            print(f"   Tournament epoch_performance sample keys: {list(sample_entry.keys())}")
    if sparse_data:
        print(f"   Sparse data keys: {list(sparse_data.keys())}")
        if 'epoch_performance' in sparse_data and sparse_data['epoch_performance']:
            sample_entry = sparse_data['epoch_performance'][0]
            print(f"   Sparse epoch_performance sample keys: {list(sample_entry.keys())}")
    
    # Extract performance data
    direct_df = extract_performance_data(direct_data, 'Direct Growth')
    tournament_df = extract_performance_data(tournament_data, 'Tournament Growth')
    sparse_df = extract_performance_data(sparse_data, 'Sparse Baseline')
    
    # Filter out None dataframes
    dataframes = [df for df in [direct_df, tournament_df, sparse_df] if df is not None]
    experiment_names = [df['experiment'].iloc[0] for df in dataframes if df is not None]
    
    if len(dataframes) == 0:
        print("❌ No valid data found in any file!")
        return
    
    print(f"✅ Loaded data for {len(dataframes)} experiments: {', '.join(experiment_names)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Performance Over Time (Main Plot)
    ax1 = plt.subplot(2, 3, (1, 2))
    
    for i, df in enumerate(dataframes):
        if 'val_performance' in df.columns:
            # Use appropriate x-axis
            if 'iteration' in df.columns and df['iteration'].max() > 0:
                x_data = df['iteration']
                x_label = 'Growth Iteration'
            elif 'epoch' in df.columns:
                x_data = df['epoch']
                x_label = 'Epoch'
            else:
                x_data = range(len(df))
                x_label = 'Step'
            
            plt.plot(x_data, df['val_performance'], 
                    marker='o', linestyle='-', linewidth=2, markersize=4,
                    color=colors[i % len(colors)], 
                    label=df['experiment'].iloc[0])
    
    plt.title('Performance Comparison Over Time', fontsize=16, fontweight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 2. Final Performance Comparison (Bar Chart)
    ax2 = plt.subplot(2, 3, 3)
    
    final_performances = []
    labels = []
    
    for df in dataframes:
        if 'val_performance' in df.columns:
            final_perf = df['val_performance'].iloc[-1]
            final_performances.append(final_perf)
            labels.append(df['experiment'].iloc[0])
    
    bars = plt.bar(labels, final_performances, color=colors[:len(final_performances)], alpha=0.7)
    plt.title('Final Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Final Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, perf in zip(bars, final_performances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{perf:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Training Loss Comparison (if available)
    ax3 = plt.subplot(2, 3, 4)
    
    loss_plotted = False
    for i, df in enumerate(dataframes):
        if 'train_loss' in df.columns:
            if 'epoch' in df.columns:
                x_data = df['epoch']
            else:
                x_data = range(len(df))
            
            plt.plot(x_data, df['train_loss'], 
                    linestyle='-', linewidth=2,
                    color=colors[i % len(colors)], 
                    label=df['experiment'].iloc[0])
            loss_plotted = True
    
    if loss_plotted:
        plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
    else:
        plt.text(0.5, 0.5, 'Training Loss\nData Not Available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    
    # 4. Network Growth Analysis (if available)
    ax4 = plt.subplot(2, 3, 5)
    
    growth_plotted = False
    for i, df in enumerate(dataframes):
        if 'total_connections' in df.columns:
            if 'iteration' in df.columns and df['iteration'].max() > 0:
                x_data = df['iteration']
            elif 'epoch' in df.columns:
                x_data = df['epoch']
            else:
                x_data = range(len(df))
            
            plt.plot(x_data, df['total_connections'], 
                    linestyle='-', linewidth=2,
                    color=colors[i % len(colors)], 
                    label=df['experiment'].iloc[0])
            growth_plotted = True
    
    if growth_plotted:
        plt.title('Network Growth (Connections)', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Total Connections', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Network Growth\nData Not Available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        plt.title('Network Growth Analysis', fontsize=14, fontweight='bold')
    
    # 5. Summary Statistics Table
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')
    
    # Create summary table
    summary_data = []
    for df in dataframes:
        exp_name = df['experiment'].iloc[0]
        
        # Calculate statistics
        if 'val_performance' in df.columns:
            final_acc = df['val_performance'].iloc[-1]
            max_acc = df['val_performance'].max()
            epochs = len(df) if 'epoch' in df.columns else 'N/A'
        else:
            final_acc = max_acc = epochs = 'N/A'
        
        if 'total_connections' in df.columns:
            final_connections = df['total_connections'].iloc[-1]
        else:
            final_connections = 'N/A'
        
        summary_data.append([exp_name, 
                           f'{final_acc:.2%}' if isinstance(final_acc, float) else final_acc,
                           f'{max_acc:.2%}' if isinstance(max_acc, float) else max_acc,
                           str(epochs),
                           str(final_connections)])
    
    # Create table
    table = ax5.table(cellText=summary_data,
                     colLabels=['Experiment', 'Final Acc', 'Max Acc', 'Epochs', 'Connections'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # Add overall title and metadata
    fig.suptitle('Comparative Analysis: Growth Strategies Performance', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.02, 0.02, f'Generated: {timestamp}', fontsize=8, alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.08)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'comparative_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparative analysis saved to: {output_path}")
    
    # Generate individual performance plot
    plt.figure(figsize=(12, 8))
    
    for i, df in enumerate(dataframes):
        if 'val_performance' in df.columns:
            if 'iteration' in df.columns and df['iteration'].max() > 0:
                x_data = df['iteration']
                x_label = 'Growth Iteration'
            elif 'epoch' in df.columns:
                x_data = df['epoch']
                x_label = 'Epoch'
            else:
                x_data = range(len(df))
                x_label = 'Step'
            
            plt.plot(x_data, df['val_performance'], 
                    marker='o', linestyle='-', linewidth=3, markersize=6,
                    color=colors[i % len(colors)], 
                    label=df['experiment'].iloc[0])
    
    plt.title('Growth Strategies: Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance annotations
    for i, df in enumerate(dataframes):
        if 'val_performance' in df.columns:
            final_perf = df['val_performance'].iloc[-1]
            if 'iteration' in df.columns and df['iteration'].max() > 0:
                final_x = df['iteration'].iloc[-1]
            elif 'epoch' in df.columns:
                final_x = df['epoch'].iloc[-1]
            else:
                final_x = len(df) - 1
            
            plt.annotate(f'{final_perf:.2%}', 
                        xy=(final_x, final_perf),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i % len(colors)], alpha=0.7),
                        fontsize=10, fontweight='bold', color='white')
    
    individual_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(individual_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Performance comparison saved to: {individual_path}")
    
    return {
        'experiments_loaded': len(dataframes),
        'experiment_names': experiment_names,
        'output_files': [output_path, individual_path]
    }

def main():
    """Main function to run comparative analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comparative analysis plots')
    parser.add_argument('--direct', default='data/comparative_direct_growth.json',
                       help='Path to direct growth results')
    parser.add_argument('--tournament', default='data/comparative_tournament_growth.json',
                       help='Path to tournament growth results')
    parser.add_argument('--sparse', default='data/comparative_sparse_baseline.json',
                       help='Path to sparse baseline results')
    parser.add_argument('--output-dir', default='data/comparison_charts',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Run the analysis
    results = plot_comparative_analysis(
        direct_file=args.direct,
        tournament_file=args.tournament,
        sparse_file=args.sparse,
        output_dir=args.output_dir
    )
    
    if results:
        print(f"\n✅ Analysis complete!")
        print(f"📊 Experiments analyzed: {results['experiments_loaded']}")
        print(f"📈 Plots generated: {len(results['output_files'])}")
        for file in results['output_files']:
            print(f"   - {file}")
    else:
        print(f"\n❌ Analysis failed - check input files")

if __name__ == "__main__":
    main()
