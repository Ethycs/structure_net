#!/usr/bin/env python3
"""
Convert JSON Experiment Data to WandB

This script converts existing JSON experiment files to WandB with
automatic visualization and dashboard creation.

Usage:
    python utils/convert_json_to_wandb.py data/comparative_direct_growth.json
    python utils/convert_json_to_wandb.py data/  # Convert all JSON files in directory
"""

import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path to import structure_net
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.structure_net.logging.wandb_integration import convert_json_to_wandb


def main():
    parser = argparse.ArgumentParser(description='Convert JSON experiment data to WandB')
    parser.add_argument('path', help='Path to JSON file or directory containing JSON files')
    parser.add_argument('--project', default='structure_net_experiments', 
                       help='WandB project name')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be converted without actually doing it')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file() and path.suffix == '.json':
        # Single file conversion
        json_files = [path]
    elif path.is_dir():
        # Directory conversion - find all JSON files recursively
        json_files = list(path.rglob('*.json'))
    else:
        print(f"❌ Invalid path: {path}")
        return
    
    if not json_files:
        print(f"❌ No JSON files found in {path}")
        return
    
    print(f"🔍 Found {len(json_files)} JSON files to convert:")
    for json_file in json_files:
        print(f"  - {json_file}")
    
    if args.dry_run:
        print("🔍 Dry run mode - no actual conversion performed")
        return
    
    print(f"\n🚀 Converting to WandB project: {args.project}")
    
    converted_urls = []
    
    for json_file in json_files:
        try:
            print(f"\n📊 Converting {json_file.name}...")
            
            # Convert to WandB
            url = convert_json_to_wandb(
                str(json_file),
                project_name=args.project,
                experiment_name=f"imported_{json_file.stem}"
            )
            
            converted_urls.append(url)
            print(f"✅ Converted: {url}")
            
        except Exception as e:
            print(f"❌ Failed to convert {json_file}: {e}")
    
    print(f"\n🎉 Conversion complete!")
    print(f"📊 Converted {len(converted_urls)} experiments")
    print(f"🔗 View your experiments at: https://wandb.ai/")
    
    if converted_urls:
        print("\n📋 Direct links:")
        for url in converted_urls:
            print(f"  {url}")


if __name__ == "__main__":
    main()
