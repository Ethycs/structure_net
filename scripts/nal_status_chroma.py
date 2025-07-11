#!/usr/bin/env python3
"""
NAL Status Monitor - ChromaDB Edition

Queries ChromaDB for experiment status, statistics, and trends.
"""

import argparse
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict, Counter
import chromadb
from chromadb.config import Settings
from tabulate import tabulate

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_factory.search import ExperimentSearcher


class NALStatusMonitor:
    """Monitor NAL experiments via ChromaDB."""
    
    def __init__(self, db_path: str = "./chroma_db"):
        """Initialize status monitor with ChromaDB connection."""
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.searcher = ExperimentSearcher(db_path)
        
        # Get collections
        try:
            self.experiments_collection = self.client.get_collection("experiments")
            self.hypotheses_collection = self.client.get_collection("hypotheses")
        except:
            print(f"❌ ChromaDB collections not found at {db_path}")
            print("Make sure experiments have been run and logged to ChromaDB.")
            sys.exit(1)
    
    def get_summary_stats(self) -> Dict:
        """Get overall summary statistics."""
        # Get all experiments
        all_experiments = self.experiments_collection.get()
        all_hypotheses = self.hypotheses_collection.get()
        
        if not all_experiments['ids']:
            return {
                'total_experiments': 0,
                'total_hypotheses': 0,
                'success_rate': 0.0,
                'avg_accuracy': 0.0,
                'avg_training_time': 0.0
            }
        
        # Extract metadata
        metadatas = all_experiments['metadatas']
        
        # Calculate statistics
        total_experiments = len(metadatas)
        successful_experiments = sum(1 for m in metadatas if m.get('status') == 'completed')
        failed_experiments = sum(1 for m in metadatas if m.get('status') == 'failed')
        
        accuracies = [m.get('accuracy', 0) for m in metadatas if m.get('accuracy') is not None]
        training_times = [m.get('training_time', 0) for m in metadatas if m.get('training_time') is not None]
        
        return {
            'total_experiments': total_experiments,
            'total_hypotheses': len(all_hypotheses['ids']) if all_hypotheses['ids'] else 0,
            'successful_experiments': successful_experiments,
            'failed_experiments': failed_experiments,
            'success_rate': (successful_experiments / total_experiments * 100) if total_experiments > 0 else 0,
            'avg_accuracy': np.mean(accuracies) if accuracies else 0.0,
            'avg_training_time': np.mean(training_times) if training_times else 0.0,
            'total_parameters': sum(m.get('model_parameters', 0) for m in metadatas),
            'unique_architectures': len(set(str(m.get('architecture', [])) for m in metadatas))
        }
    
    def get_recent_experiments(self, hours: int = 24, limit: int = 10) -> List[Dict]:
        """Get experiments from the last N hours."""
        since_timestamp = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        # Query recent experiments
        results = self.experiments_collection.query(
            query_texts=[""],  # Empty query to get all
            n_results=1000,
            where={"timestamp": {"$gte": since_timestamp}}
        )
        
        if not results['ids'][0]:
            return []
        
        experiments = []
        for i, exp_id in enumerate(results['ids'][0][:limit]):
            metadata = results['metadatas'][0][i]
            experiments.append({
                'id': exp_id,
                'hypothesis': metadata.get('hypothesis_id', 'Unknown'),
                'status': metadata.get('status', 'Unknown'),
                'accuracy': metadata.get('accuracy', 0.0),
                'parameters': metadata.get('model_parameters', 0),
                'training_time': metadata.get('training_time', 0.0),
                'timestamp': metadata.get('timestamp', ''),
                'error': metadata.get('error', '')
            })
        
        return sorted(experiments, key=lambda x: x['timestamp'], reverse=True)
    
    def get_hypothesis_performance(self) -> List[Dict]:
        """Get performance statistics by hypothesis."""
        all_experiments = self.experiments_collection.get()
        
        if not all_experiments['ids']:
            return []
        
        # Group by hypothesis
        hypothesis_stats = defaultdict(lambda: {
            'experiments': 0,
            'successful': 0,
            'failed': 0,
            'accuracies': [],
            'training_times': [],
            'parameters': []
        })
        
        for metadata in all_experiments['metadatas']:
            hyp_id = metadata.get('hypothesis_id', 'Unknown')
            stats = hypothesis_stats[hyp_id]
            
            stats['experiments'] += 1
            if metadata.get('status') == 'completed':
                stats['successful'] += 1
                if metadata.get('accuracy') is not None:
                    stats['accuracies'].append(metadata['accuracy'])
                if metadata.get('training_time') is not None:
                    stats['training_times'].append(metadata['training_time'])
                if metadata.get('model_parameters') is not None:
                    stats['parameters'].append(metadata['model_parameters'])
            elif metadata.get('status') == 'failed':
                stats['failed'] += 1
        
        # Calculate aggregates
        results = []
        for hyp_id, stats in hypothesis_stats.items():
            # Get hypothesis details
            try:
                hyp_data = self.hypotheses_collection.get(ids=[hyp_id])
                hyp_name = hyp_data['metadatas'][0].get('name', hyp_id) if hyp_data['ids'] else hyp_id
                hyp_category = hyp_data['metadatas'][0].get('category', 'Unknown') if hyp_data['ids'] else 'Unknown'
            except:
                hyp_name = hyp_id
                hyp_category = 'Unknown'
            
            results.append({
                'hypothesis': hyp_name,
                'category': hyp_category,
                'experiments': stats['experiments'],
                'success_rate': (stats['successful'] / stats['experiments'] * 100) if stats['experiments'] > 0 else 0,
                'avg_accuracy': np.mean(stats['accuracies']) if stats['accuracies'] else 0.0,
                'std_accuracy': np.std(stats['accuracies']) if len(stats['accuracies']) > 1 else 0.0,
                'avg_time': np.mean(stats['training_times']) if stats['training_times'] else 0.0,
                'avg_params': np.mean(stats['parameters']) if stats['parameters'] else 0
            })
        
        return sorted(results, key=lambda x: x['avg_accuracy'], reverse=True)
    
    def get_architecture_trends(self) -> Dict:
        """Analyze architecture trends."""
        all_experiments = self.experiments_collection.get()
        
        if not all_experiments['ids']:
            return {}
        
        # Analyze architectures
        architecture_stats = defaultdict(lambda: {
            'count': 0,
            'accuracies': [],
            'parameters': []
        })
        
        for metadata in all_experiments['metadatas']:
            if metadata.get('status') == 'completed':
                arch = metadata.get('architecture', [])
                if arch:
                    arch_str = f"{len(arch)-1} layers, {arch[0]}→{arch[-1]}"
                    stats = architecture_stats[arch_str]
                    stats['count'] += 1
                    if metadata.get('accuracy') is not None:
                        stats['accuracies'].append(metadata['accuracy'])
                    if metadata.get('model_parameters') is not None:
                        stats['parameters'].append(metadata['model_parameters'])
        
        # Find best architectures
        best_architectures = []
        for arch, stats in architecture_stats.items():
            if stats['accuracies']:
                best_architectures.append({
                    'architecture': arch,
                    'count': stats['count'],
                    'avg_accuracy': np.mean(stats['accuracies']),
                    'max_accuracy': max(stats['accuracies']),
                    'avg_parameters': np.mean(stats['parameters']) if stats['parameters'] else 0
                })
        
        return {
            'total_unique': len(architecture_stats),
            'best_architectures': sorted(best_architectures, key=lambda x: x['avg_accuracy'], reverse=True)[:10]
        }
    
    def get_error_analysis(self) -> Dict:
        """Analyze experiment failures."""
        all_experiments = self.experiments_collection.get()
        
        if not all_experiments['ids']:
            return {'total_errors': 0, 'error_types': {}}
        
        errors = []
        for metadata in all_experiments['metadatas']:
            if metadata.get('status') == 'failed' and metadata.get('error'):
                error_msg = metadata['error']
                # Extract error type (first line)
                error_type = error_msg.split('\n')[0].split(':')[0]
                errors.append(error_type)
        
        error_counts = Counter(errors)
        
        return {
            'total_errors': len(errors),
            'error_types': dict(error_counts.most_common(10))
        }
    
    def display_summary(self):
        """Display summary statistics."""
        stats = self.get_summary_stats()
        
        print("📊 NAL Experiment Summary")
        print("=" * 60)
        print(f"Total Experiments:     {stats['total_experiments']}")
        print(f"Total Hypotheses:      {stats['total_hypotheses']}")
        print(f"Successful:            {stats['successful_experiments']}")
        print(f"Failed:                {stats['failed_experiments']}")
        print(f"Success Rate:          {stats['success_rate']:.1f}%")
        print(f"Average Accuracy:      {stats['avg_accuracy']:.4f}")
        print(f"Avg Training Time:     {stats['avg_training_time']:.1f}s")
        print(f"Total Parameters:      {stats['total_parameters']:,}")
        print(f"Unique Architectures:  {stats['unique_architectures']}")
        print()
    
    def display_recent_experiments(self, hours: int = 24):
        """Display recent experiments."""
        experiments = self.get_recent_experiments(hours)
        
        print(f"🕒 Recent Experiments (last {hours} hours)")
        print("=" * 60)
        
        if not experiments:
            print("No experiments found in the specified time period.")
            return
        
        # Format for table
        table_data = []
        for exp in experiments:
            status_icon = "✅" if exp['status'] == 'completed' else "❌"
            time_str = datetime.fromisoformat(exp['timestamp']).strftime('%m-%d %H:%M')
            table_data.append([
                exp['id'][:8],
                exp['hypothesis'][:20],
                f"{status_icon} {exp['status']}",
                f"{exp['accuracy']:.4f}" if exp['accuracy'] > 0 else "-",
                f"{exp['parameters']:,}" if exp['parameters'] > 0 else "-",
                f"{exp['training_time']:.1f}s" if exp['training_time'] > 0 else "-",
                time_str
            ])
        
        headers = ["ID", "Hypothesis", "Status", "Accuracy", "Params", "Time", "When"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
    
    def display_hypothesis_performance(self):
        """Display hypothesis performance ranking."""
        hypotheses = self.get_hypothesis_performance()
        
        print("🎯 Hypothesis Performance Ranking")
        print("=" * 60)
        
        if not hypotheses:
            print("No hypothesis data available.")
            return
        
        # Format for table
        table_data = []
        for hyp in hypotheses[:10]:  # Top 10
            table_data.append([
                hyp['hypothesis'][:30],
                hyp['category'],
                hyp['experiments'],
                f"{hyp['success_rate']:.1f}%",
                f"{hyp['avg_accuracy']:.4f} ± {hyp['std_accuracy']:.4f}",
                f"{hyp['avg_params']:,.0f}",
                f"{hyp['avg_time']:.1f}s"
            ])
        
        headers = ["Hypothesis", "Category", "Runs", "Success", "Accuracy", "Params", "Time"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
    
    def display_architecture_trends(self):
        """Display architecture trends."""
        trends = self.get_architecture_trends()
        
        print("🏗️  Architecture Analysis")
        print("=" * 60)
        print(f"Total unique architectures tested: {trends.get('total_unique', 0)}")
        print("\nTop Performing Architectures:")
        
        if trends.get('best_architectures'):
            table_data = []
            for arch in trends['best_architectures'][:5]:
                table_data.append([
                    arch['architecture'],
                    arch['count'],
                    f"{arch['avg_accuracy']:.4f}",
                    f"{arch['max_accuracy']:.4f}",
                    f"{arch['avg_parameters']:,.0f}"
                ])
            
            headers = ["Architecture", "Tested", "Avg Acc", "Max Acc", "Params"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
    
    def display_error_analysis(self):
        """Display error analysis."""
        errors = self.get_error_analysis()
        
        print("❌ Error Analysis")
        print("=" * 60)
        print(f"Total failed experiments: {errors['total_errors']}")
        
        if errors['error_types']:
            print("\nMost common error types:")
            for error_type, count in errors['error_types'].items():
                print(f"  {error_type}: {count}")
        print()
    
    def search_experiments(self, query: str, limit: int = 10):
        """Search experiments using natural language."""
        print(f"🔍 Searching for: '{query}'")
        print("=" * 60)
        
        results = self.searcher.search_by_description(query, limit=limit)
        
        if not results:
            print("No matching experiments found.")
            return
        
        # Format results
        table_data = []
        for result in results:
            exp_id = result.get('experiment_id', 'Unknown')[:8]
            hyp = result.get('hypothesis_id', 'Unknown')[:20]
            acc = result.get('accuracy', 0.0)
            params = result.get('model_parameters', 0)
            status = "✅" if result.get('status') == 'completed' else "❌"
            
            table_data.append([
                exp_id,
                hyp,
                status,
                f"{acc:.4f}" if acc > 0 else "-",
                f"{params:,}" if params > 0 else "-",
                f"{result.get('similarity', 0):.3f}"
            ])
        
        headers = ["ID", "Hypothesis", "Status", "Accuracy", "Params", "Similarity"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="NAL Status Monitor - ChromaDB Edition")
    parser.add_argument('--db-path', '-d', default='./chroma_db',
                       help='Path to ChromaDB database (default: ./chroma_db)')
    parser.add_argument('--recent', '-r', type=int, metavar='HOURS',
                       help='Show experiments from last N hours')
    parser.add_argument('--hypotheses', '-y', action='store_true',
                       help='Show hypothesis performance ranking')
    parser.add_argument('--architectures', '-a', action='store_true',
                       help='Show architecture trends')
    parser.add_argument('--errors', '-e', action='store_true',
                       help='Show error analysis')
    parser.add_argument('--search', '-s', type=str,
                       help='Search experiments by description')
    parser.add_argument('--all', action='store_true',
                       help='Show all analyses')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = NALStatusMonitor(args.db_path)
    
    # Display requested information
    if args.search:
        monitor.search_experiments(args.search)
    elif args.recent:
        monitor.display_recent_experiments(args.recent)
    elif args.hypotheses:
        monitor.display_hypothesis_performance()
    elif args.architectures:
        monitor.display_architecture_trends()
    elif args.errors:
        monitor.display_error_analysis()
    elif args.all:
        monitor.display_summary()
        monitor.display_recent_experiments(24)
        monitor.display_hypothesis_performance()
        monitor.display_architecture_trends()
        monitor.display_error_analysis()
    else:
        # Default: show summary and recent
        monitor.display_summary()
        monitor.display_recent_experiments(24)


if __name__ == "__main__":
    main()