#!/usr/bin/env python3
"""
NAL Status Monitor - CLI tool to check the status of NAL experiments.

This tool monitors experiment queues, results, and system resources.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import psutil
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp_str

def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def get_system_status() -> Dict:
    """Get current system resource status."""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'disk_percent': psutil.disk_usage('/').percent,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_info': []
    }

def get_gpu_status() -> List[Dict]:
    """Get GPU status if available."""
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                used = torch.cuda.memory_allocated(i)
                total = props.total_memory
                gpu_info.append({
                    'id': i,
                    'name': props.name,
                    'memory_used_gb': used / (1024**3),
                    'memory_total_gb': total / (1024**3),
                    'memory_percent': (used / total) * 100,
                    'temperature': 'N/A'  # Would need nvidia-ml-py for this
                })
            except Exception as e:
                gpu_info.append({
                    'id': i,
                    'name': 'Unknown',
                    'error': str(e)
                })
    return gpu_info

def scan_experiment_directory(results_dir: str) -> Dict:
    """Scan experiment directory for status information."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return {'error': f"Directory {results_dir} does not exist"}
    
    # Find experiment directories
    experiment_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('nal_')]
    
    status = {
        'experiment_dirs': len(experiment_dirs),
        'experiments': [],
        'queue_status': {},
        'recent_results': []
    }
    
    # Check each experiment directory
    for exp_dir in experiment_dirs:
        exp_info = {
            'name': exp_dir.name,
            'path': str(exp_dir),
            'created': datetime.fromtimestamp(exp_dir.stat().st_ctime).isoformat(),
            'size_mb': sum(f.stat().st_size for f in exp_dir.rglob('*') if f.is_file()) / (1024**2)
        }
        
        # Check for queue directories
        queue_dir = exp_dir / 'experiment_queue'
        sent_dir = exp_dir / 'experiment_sent'
        rejected_dir = exp_dir / 'experiment_rejected'
        
        if queue_dir.exists():
            queued_files = list(queue_dir.glob('*.json'))
            exp_info['queued'] = len(queued_files)
            
            # Get recent queue files
            if queued_files:
                recent_queued = sorted(queued_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]
                exp_info['recent_queued'] = [f.name for f in recent_queued]
        
        if sent_dir.exists():
            exp_info['sent'] = len(list(sent_dir.glob('*.json')))
        
        if rejected_dir.exists():
            exp_info['rejected'] = len(list(rejected_dir.glob('*.json')))
        
        # Check for hypothesis results
        result_files = list(exp_dir.glob('*_results.json'))
        exp_info['hypothesis_results'] = len(result_files)
        
        # Get recent results
        if result_files:
            recent_results = sorted(result_files, key=lambda f: f.stat().st_mtime, reverse=True)[:3]
            exp_info['recent_results'] = []
            for rf in recent_results:
                try:
                    with open(rf, 'r') as f:
                        data = json.load(f)
                        exp_info['recent_results'].append({
                            'file': rf.name,
                            'hypothesis': data.get('hypothesis', {}).get('name', 'Unknown'),
                            'confirmed': data.get('result', {}).get('confirmed', False),
                            'experiments': data.get('result', {}).get('num_experiments', 0),
                            'successful': data.get('result', {}).get('successful_experiments', 0)
                        })
                except Exception as e:
                    exp_info['recent_results'].append({
                        'file': rf.name,
                        'error': str(e)
                    })
        
        status['experiments'].append(exp_info)
    
    return status

def display_system_status():
    """Display system resource status."""
    print("🖥️  System Status")
    print("=" * 50)
    
    sys_status = get_system_status()
    gpu_status = get_gpu_status()
    
    print(f"CPU Usage:      {sys_status['cpu_percent']:5.1f}%")
    print(f"Memory Usage:   {sys_status['memory_percent']:5.1f}% ({sys_status['memory_total_gb']:.1f} GB)")
    print(f"Disk Usage:     {sys_status['disk_percent']:5.1f}%")
    print(f"GPUs Available: {sys_status['gpu_count']}")
    
    if gpu_status:
        print("\n📊 GPU Status:")
        for gpu in gpu_status:
            if 'error' in gpu:
                print(f"  GPU {gpu['id']}: Error - {gpu['error']}")
            else:
                print(f"  GPU {gpu['id']}: {gpu['name']}")
                print(f"    Memory: {gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.1f} GB ({gpu['memory_percent']:.1f}%)")
    
    print()

def display_experiment_status(results_dir: str):
    """Display experiment status."""
    print("🧪 Experiment Status")
    print("=" * 50)
    
    status = scan_experiment_directory(results_dir)
    
    if 'error' in status:
        print(f"❌ Error: {status['error']}")
        return
    
    if status['experiment_dirs'] == 0:
        print("No experiment directories found.")
        return
    
    print(f"Found {status['experiment_dirs']} experiment directories:\n")
    
    for exp in status['experiments']:
        print(f"📁 {exp['name']}")
        print(f"   Created: {format_timestamp(exp['created'])}")
        print(f"   Size: {exp['size_mb']:.1f} MB")
        
        # Queue status
        queue_info = []
        if 'queued' in exp:
            queue_info.append(f"Queued: {exp['queued']}")
        if 'sent' in exp:
            queue_info.append(f"Sent: {exp['sent']}")
        if 'rejected' in exp:
            queue_info.append(f"Rejected: {exp['rejected']}")
        
        if queue_info:
            print(f"   Queue: {', '.join(queue_info)}")
        
        # Results
        if exp.get('hypothesis_results', 0) > 0:
            print(f"   Hypothesis Results: {exp['hypothesis_results']}")
            
            if exp.get('recent_results'):
                print("   Recent Results:")
                for result in exp['recent_results']:
                    if 'error' in result:
                        print(f"     ❌ {result['file']}: {result['error']}")
                    else:
                        status_icon = "✅" if result['confirmed'] else "❌"
                        print(f"     {status_icon} {result['hypothesis']}: {result['successful']}/{result['experiments']} experiments")
        
        print()

def monitor_experiments(results_dir: str, interval: int = 10):
    """Monitor experiments continuously."""
    print(f"🔄 Monitoring experiments in {results_dir}")
    print(f"Refresh interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"NAL Status Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            
            display_system_status()
            display_experiment_status(results_dir)
            
            print(f"Refreshing in {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")

def tail_experiment_logs(results_dir: str, lines: int = 50):
    """Show recent experiment logs."""
    print(f"📄 Recent Experiment Logs ({lines} lines)")
    print("=" * 50)
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Directory {results_dir} does not exist")
        return
    
    # Find recent log files
    log_files = []
    for exp_dir in results_path.iterdir():
        if exp_dir.is_dir():
            for log_file in exp_dir.rglob('*.log'):
                log_files.append(log_file)
    
    if not log_files:
        print("No log files found.")
        return
    
    # Sort by modification time
    log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    print(f"Most recent log: {log_files[0]}")
    print("-" * 50)
    
    try:
        with open(log_files[0], 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            for line in recent_lines:
                print(line.rstrip())
    except Exception as e:
        print(f"❌ Error reading log file: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="NAL Status Monitor")
    parser.add_argument('--results-dir', '-d', default='data', 
                       help='Directory containing NAL results (default: data)')
    parser.add_argument('--monitor', '-m', action='store_true',
                       help='Continuously monitor experiments')
    parser.add_argument('--interval', '-i', type=int, default=10,
                       help='Monitor refresh interval in seconds (default: 10)')
    parser.add_argument('--system', '-s', action='store_true',
                       help='Show only system status')
    parser.add_argument('--logs', '-l', type=int, const=50, nargs='?',
                       help='Show recent log lines (default: 50)')
    
    args = parser.parse_args()
    
    if args.system:
        display_system_status()
    elif args.logs is not None:
        tail_experiment_logs(args.results_dir, args.logs)
    elif args.monitor:
        monitor_experiments(args.results_dir, args.interval)
    else:
        display_system_status()
        display_experiment_status(args.results_dir)

if __name__ == "__main__":
    main()