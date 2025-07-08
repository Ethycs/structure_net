#!/usr/bin/env python3
"""
Simple JSON to WandB Converter

Following industry-standard pattern:
1. Flatten nested JSON to key/value pairs
2. Convert arrays to wandb.Table
3. Create plots with wandb.plot.*

This is much simpler and more maintainable than custom handlers.
"""

import json
import wandb
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Any, Union


def flatten(d: Dict, prefix: str = "") -> Dict[str, Union[int, float]]:
    """
    Yield ('a/b/c', value) for every scalar in a nested dict.
    Uses slash-separated keys for WandB auto-grouping.
    """
    result = {}
    
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}/{k}"
        
        if isinstance(v, dict):
            result.update(flatten(v, key + "/"))
        elif isinstance(v, (int, float)):
            result[key] = v
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            # Handle lists of numbers (like architecture)
            for i, item in enumerate(v):
                if isinstance(item, (int, float)):
                    result[f"{key}/{i}"] = item
    
    return result


def convert_json_to_wandb(json_path: str, project_name: str = "structure_net"):
    """
    Convert any JSON file to WandB using industry-standard pattern.
    
    Args:
        json_path: Path to JSON file
        project_name: WandB project name
    """
    
    # Load JSON
    with open(json_path, 'r') as f:
        raw = json.load(f)
    
    # Create experiment name from file
    json_file = Path(json_path)
    parent_dir = json_file.parent.name
    experiment_name = f"{parent_dir}_{json_file.stem}"
    
    # Initialize WandB
    run = wandb.init(
        project=project_name,
        name=experiment_name,
        tags=["json_import", parent_dir],
        reinit=True
    )
    
    print(f"🔗 WandB run: {run.url}")
    
    try:
        if isinstance(raw, list):
            # Case: Array of records (time-series data)
            convert_array_to_wandb(raw, json_file.stem)
            
        elif isinstance(raw, dict):
            # Case: Single object (summary/config data)
            convert_dict_to_wandb(raw, json_file.stem)
            
        else:
            print(f"⚠️  Unknown JSON structure: {type(raw)}")
        
        # Log metadata
        run.log({
            "import/timestamp": datetime.now().isoformat(),
            "import/source_file": json_file.name,
            "import/file_size_kb": json_file.stat().st_size / 1024
        })
        
        print(f"✅ Successfully converted: {json_path}")
        return run.url
        
    finally:
        wandb.finish()


def convert_array_to_wandb(records: List[Dict], filename: str):
    """Convert array of records to WandB table + plots."""
    
    print(f"📊 Converting {len(records)} records from array")
    
    # Flatten all records and convert to DataFrame
    df = pd.json_normalize(records)
    
    # Log as table for queryability
    table = wandb.Table(dataframe=df)
    wandb.log({f"{filename}_table": table})
    
    # Auto-detect step column
    step_candidates = ["epoch", "iteration", "step", "time", "frame"]
    step_col = None
    
    for candidate in step_candidates:
        if candidate in df.columns:
            step_col = candidate
            break
    
    if step_col is None:
        # Use index as step
        df['step'] = range(len(df))
        step_col = 'step'
    
    # Create automatic plots for interesting metrics
    numeric_cols = df.select_dtypes(include=['number']).columns
    interesting_cols = [col for col in numeric_cols 
                       if any(keyword in col.lower() 
                             for keyword in ['accuracy', 'loss', 'performance', 'count', 'ratio', 'mean', 'std'])]
    
    # Limit to 5 most interesting plots
    for col in interesting_cols[:5]:
        if col != step_col:
            try:
                plot_title = col.replace('_', ' ').replace('.', ' ').title()
                wandb.log({
                    f"{col}_over_time": wandb.plot.line(
                        table, step_col, col,
                        title=f"{plot_title} Over Time"
                    )
                })
                print(f"📈 Created plot: {plot_title}")
            except Exception as e:
                print(f"⚠️  Could not create plot for {col}: {e}")
    
    # Log flattened metrics for line plots
    for i, record in enumerate(records):
        step_value = record.get(step_col, i)
        flat_metrics = flatten(record)
        
        if flat_metrics:
            wandb.log(flat_metrics, step=step_value)
    
    print(f"📊 Logged {len(records)} time-series records")


def convert_dict_to_wandb(data: Dict, filename: str):
    """Convert single dict to WandB config + metrics."""
    
    print(f"📋 Converting summary data from dict")
    
    # Flatten and log all metrics
    flat_metrics = flatten(data)
    if flat_metrics:
        wandb.log(flat_metrics)
        print(f"📊 Logged {len(flat_metrics)} metrics")
    
    # Extract config-worthy data (strings, small numbers, small lists)
    config_data = {}
    
    def extract_config(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}/{key}" if prefix else key
                
                if isinstance(value, (str, int, float, bool)):
                    config_data[new_key] = value
                elif isinstance(value, list) and len(value) < 20:
                    if all(isinstance(x, (int, float)) for x in value):
                        config_data[new_key] = value
                elif isinstance(value, dict) and len(value) < 10:
                    extract_config(value, new_key)
    
    extract_config(data)
    
    if config_data:
        wandb.config.update(config_data)
        print(f"⚙️  Updated config with {len(config_data)} items")
    
    # Handle nested time-series arrays
    time_series_keys = ["performance_history", "growth_events", "growth_history", "training_log", "extrema_evolution"]
    
    for key in time_series_keys:
        if key in data and isinstance(data[key], list):
            print(f"🔍 Found nested time-series: {key}")
            convert_array_to_wandb(data[key], f"{filename}_{key}")


def main():
    """CLI interface for JSON to WandB conversion."""
    parser = argparse.ArgumentParser(description='Convert JSON files to WandB')
    parser.add_argument('json_path', help='Path to JSON file or directory')
    parser.add_argument('--project', default='structure_net_simple', help='WandB project name')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    
    if json_path.is_file():
        # Single file
        convert_json_to_wandb(str(json_path), args.project)
        
    elif json_path.is_dir():
        # Directory of JSON files
        json_files = list(json_path.glob('**/*.json'))
        print(f"🔍 Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                convert_json_to_wandb(str(json_file), args.project)
            except Exception as e:
                print(f"❌ Failed to convert {json_file}: {e}")
        
        print(f"✅ Processed {len(json_files)} files")
        
    else:
        print(f"❌ Path not found: {json_path}")


if __name__ == "__main__":
    main()
