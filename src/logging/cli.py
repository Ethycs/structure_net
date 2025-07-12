#!/usr/bin/env python3
"""
CLI Tools for Structure Net Logging System

This module provides command-line tools for managing the standardized logging
system, including queue management, validation, and artifact operations.

Key features:
- Queue status and management
- Validation testing
- Artifact upload control
- Schema migration utilities
- Debugging tools
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .artifact_manager import ArtifactManager, ArtifactConfig, ArtifactUploader
from .schemas import validate_experiment_data, migrate_legacy_data
from .standardized_logger import StandardizedLogger


def cmd_queue_status(args):
    """Show current queue status."""
    config = ArtifactConfig(
        queue_dir=args.queue_dir,
        sent_dir=args.sent_dir,
        rejected_dir=args.rejected_dir,
        project_name=args.project
    )
    
    manager = ArtifactManager(config)
    status = manager.get_queue_status()
    
    print("📊 QUEUE STATUS")
    print("=" * 50)
    print(f"Queue size: {status['queue_size']} files")
    print(f"Sent count: {status['sent_count']} files")
    print(f"Rejected count: {status['rejected_count']} files")
    print()
    
    print("📈 PROCESSING STATS")
    print("-" * 30)
    stats = status['processing_stats']
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print()
    
    print("📁 DIRECTORIES")
    print("-" * 30)
    for name, path in status['directories'].items():
        print(f"{name.title()}: {path}")


def cmd_process_queue(args):
    """Process the artifact queue."""
    config = ArtifactConfig(
        queue_dir=args.queue_dir,
        sent_dir=args.sent_dir,
        rejected_dir=args.rejected_dir,
        project_name=args.project,
        max_retries=args.max_retries
    )
    
    manager = ArtifactManager(config)
    
    print(f"🔄 Processing queue (max files: {args.max_files or 'all'})")
    stats = manager.process_queue(max_files=args.max_files)
    
    print("\n📊 PROCESSING RESULTS")
    print("=" * 40)
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


def cmd_validate_file(args):
    """Validate a specific JSON file against schemas."""
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"🔍 Validating: {file_path}")
        
        # Auto-migrate if requested
        if args.migrate:
            print("🔄 Applying migration...")
            data = migrate_legacy_data(data)
        
        # Validate
        validated_data = validate_experiment_data(data)
        
        print("✅ Validation successful!")
        print(f"Experiment type: {validated_data.experiment_type}")
        print(f"Experiment ID: {validated_data.experiment_id}")
        print(f"Schema version: {validated_data.schema_version}")
        
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(validated_data.dict(), f, indent=2, default=str)
            print(f"💾 Validated data saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)


def cmd_rejected_files(args):
    """Show information about rejected files."""
    config = ArtifactConfig(
        rejected_dir=args.rejected_dir
    )
    
    manager = ArtifactManager(config)
    rejected_info = manager.get_rejected_files_info()
    
    if not rejected_info:
        print("✅ No rejected files found")
        return
    
    print(f"❌ REJECTED FILES ({len(rejected_info)})")
    print("=" * 60)
    
    for info in rejected_info:
        print(f"File: {info['filename']}")
        print(f"Size: {info['size']} bytes")
        print(f"Modified: {info['modified']}")
        if info['error']:
            print(f"Error: {info['error']}")
        print("-" * 40)


def cmd_requeue_file(args):
    """Requeue a rejected file."""
    config = ArtifactConfig(
        queue_dir=args.queue_dir,
        rejected_dir=args.rejected_dir
    )
    
    manager = ArtifactManager(config)
    
    success = manager.requeue_rejected_file(args.filename)
    
    if success:
        print(f"✅ Successfully requeued: {args.filename}")
    else:
        print(f"❌ Failed to requeue: {args.filename}")
        sys.exit(1)


def cmd_start_uploader(args):
    """Start the background uploader daemon."""
    config = ArtifactConfig(
        queue_dir=args.queue_dir,
        sent_dir=args.sent_dir,
        rejected_dir=args.rejected_dir,
        project_name=args.project,
        max_retries=args.max_retries
    )
    
    uploader = ArtifactUploader(config, interval=args.interval)
    
    print(f"🚀 Starting uploader daemon (interval: {args.interval}s)")
    print("Press Ctrl+C to stop")
    
    try:
        uploader.start()
    except KeyboardInterrupt:
        print("\n🛑 Uploader daemon stopped")


def cmd_clean_sent(args):
    """Clean old files from sent directory."""
    config = ArtifactConfig(
        sent_dir=args.sent_dir
    )
    
    manager = ArtifactManager(config)
    removed_count = manager.clean_sent_files(older_than_days=args.days)
    
    print(f"🧹 Cleaned {removed_count} files older than {args.days} days")


def cmd_migrate_file(args):
    """Migrate a legacy file to current schema format."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        print(f"🔄 Migrating: {input_path}")
        
        # Migrate data
        migrated_data = migrate_legacy_data(data, target_version=args.target_version)
        
        # Validate migrated data
        validated_data = validate_experiment_data(migrated_data)
        
        # Save migrated data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(validated_data.dict(), f, indent=2, default=str)
        
        print(f"✅ Migration successful!")
        print(f"💾 Migrated data saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)


def cmd_test_logger(args):
    """Test the standardized logger with sample data."""
    print("🧪 Testing StandardizedLogger...")
    
    # Create test logger
    logger = StandardizedLogger(
        project_name=args.project,
        experiment_name=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        experiment_type=args.experiment_type,
        config={
            'dataset': 'test_dataset',
            'batch_size': 32,
            'learning_rate': 0.001,
            'max_epochs': 10,
            'device': 'cpu'
        },
        tags=['test', 'cli']
    )
    
    # Test validation
    is_valid = logger.validate_current_data()
    print(f"Initial validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    # Test artifact saving
    try:
        artifact_hash = logger.save_experiment_artifact()
        print(f"✅ Artifact saved: {artifact_hash}")
    except Exception as e:
        print(f"❌ Artifact save failed: {e}")
    
    # Get summary
    summary = logger.get_experiment_summary()
    print("\n📊 EXPERIMENT SUMMARY")
    print("-" * 30)
    for key, value in summary.items():
        if key != 'queue_status':
            print(f"{key}: {value}")
    
    # Finish experiment
    logger.finish_experiment(save_artifact=False)
    print("✅ Test completed")


def cmd_queue_experiment(args):
    """Queue an experiment from JSON file."""
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        config = ArtifactConfig(
            queue_dir=args.queue_dir,
            project_name=args.project
        )
        
        manager = ArtifactManager(config)
        artifact_hash = manager.queue_experiment(data, experiment_id=args.experiment_id)
        
        print(f"✅ Experiment queued: {artifact_hash}")
        
        if args.process_immediately:
            print("🔄 Processing immediately...")
            stats = manager.process_queue(max_files=1)
            print(f"📊 Upload stats: {stats}")
        
    except Exception as e:
        print(f"❌ Failed to queue experiment: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Structure Net Logging System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check queue status
  python -m structure_net.logging.cli status
  
  # Process all queued files
  python -m structure_net.logging.cli process
  
  # Validate a specific file
  python -m structure_net.logging.cli validate experiment.json
  
  # Start background uploader
  python -m structure_net.logging.cli uploader --interval 30
  
  # Migrate legacy file
  python -m structure_net.logging.cli migrate old.json new.json
        """
    )
    
    # Global arguments
    parser.add_argument('--queue-dir', default='experiments/queue',
                       help='Queue directory path')
    parser.add_argument('--sent-dir', default='experiments/sent',
                       help='Sent directory path')
    parser.add_argument('--rejected-dir', default='experiments/rejected',
                       help='Rejected directory path')
    parser.add_argument('--project', default='structure_net',
                       help='WandB project name')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show queue status')
    status_parser.set_defaults(func=cmd_queue_status)
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process queue')
    process_parser.add_argument('--max-files', type=int,
                               help='Maximum files to process')
    process_parser.add_argument('--max-retries', type=int, default=3,
                               help='Maximum retry attempts')
    process_parser.set_defaults(func=cmd_process_queue)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate JSON file')
    validate_parser.add_argument('file', help='JSON file to validate')
    validate_parser.add_argument('--migrate', action='store_true',
                                help='Apply migration before validation')
    validate_parser.add_argument('--output', help='Save validated data to file')
    validate_parser.set_defaults(func=cmd_validate_file)
    
    # Rejected command
    rejected_parser = subparsers.add_parser('rejected', help='Show rejected files')
    rejected_parser.set_defaults(func=cmd_rejected_files)
    
    # Requeue command
    requeue_parser = subparsers.add_parser('requeue', help='Requeue rejected file')
    requeue_parser.add_argument('filename', help='Rejected filename to requeue')
    requeue_parser.set_defaults(func=cmd_requeue_file)
    
    # Uploader command
    uploader_parser = subparsers.add_parser('uploader', help='Start uploader daemon')
    uploader_parser.add_argument('--interval', type=float, default=60.0,
                                help='Processing interval in seconds')
    uploader_parser.add_argument('--max-retries', type=int, default=3,
                                help='Maximum retry attempts')
    uploader_parser.set_defaults(func=cmd_start_uploader)
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean old sent files')
    clean_parser.add_argument('--days', type=int, default=30,
                             help='Remove files older than N days')
    clean_parser.set_defaults(func=cmd_clean_sent)
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate legacy file')
    migrate_parser.add_argument('input', help='Input JSON file')
    migrate_parser.add_argument('output', help='Output JSON file')
    migrate_parser.add_argument('--target-version', default='1.0',
                               help='Target schema version')
    migrate_parser.set_defaults(func=cmd_migrate_file)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test logger functionality')
    test_parser.add_argument('--experiment-type', default='growth_experiment',
                            choices=['growth_experiment', 'training_experiment', 'tournament_experiment'],
                            help='Experiment type to test')
    test_parser.set_defaults(func=cmd_test_logger)
    
    # Queue command
    queue_parser = subparsers.add_parser('queue', help='Queue experiment from file')
    queue_parser.add_argument('file', help='JSON file to queue')
    queue_parser.add_argument('--experiment-id', help='Custom experiment ID')
    queue_parser.add_argument('--process-immediately', action='store_true',
                             help='Process immediately after queuing')
    queue_parser.set_defaults(func=cmd_queue_experiment)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        args.func(args)
    except Exception as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
