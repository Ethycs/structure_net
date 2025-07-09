#!/usr/bin/env python3
"""
WandB Artifact Manager for Structure Net

This module implements the queue → validation → WandB artifact pipeline
with hash-based deduplication, retry logic, and schema validation.

Key features:
- Local-first logging with queue system
- Automatic schema validation
- Hash-based deduplication
- Retry logic for network failures
- Schema migration support
- Offline-safe operation
"""

import os
import json
import hashlib
import shutil
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import wandb
from wandb import Api
from pydantic import ValidationError

from .schemas import (
    validate_experiment_data, 
    migrate_legacy_data,
    ExperimentSchema
)


@dataclass
class ArtifactConfig:
    """Configuration for artifact management."""
    
    queue_dir: str = "data/experiments/queue"
    sent_dir: str = "data/experiments/sent"
    rejected_dir: str = "data/experiments/rejected"
    logs_dir: str = "data/experiments/logs"
    
    # Upload settings
    max_retries: int = 3
    retry_delay: float = 30.0  # seconds
    batch_size: int = 10
    
    # WandB settings
    project_name: str = "structure_net"
    entity: Optional[str] = None
    
    # Validation settings
    strict_validation: bool = True
    auto_migrate: bool = True


class ArtifactManager:
    """
    Manages the complete artifact lifecycle from logging to WandB upload.
    
    Implements the pattern:
    1. Log experiment data to local queue
    2. Validate against Pydantic schemas
    3. Upload to WandB as artifacts with hash-based naming
    4. Move to sent directory for tracking
    """
    
    def __init__(self, config: ArtifactConfig = None):
        """Initialize artifact manager with configuration."""
        self.config = config or ArtifactConfig()
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize WandB API
        self.api = Api()
        
        # Track processing statistics
        self.stats = {
            'queued': 0,
            'validated': 0,
            'uploaded': 0,
            'rejected': 0,
            'duplicates_skipped': 0,
            'errors': 0
        }
        
        self.logger.info(f"ArtifactManager initialized with config: {self.config}")
    
    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [
            self.config.queue_dir,
            self.config.sent_dir,
            self.config.rejected_dir,
            self.config.logs_dir
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging for artifact manager."""
        log_file = Path(self.config.logs_dir) / "artifact_manager.log"
        
        # Create logger
        self.logger = logging.getLogger('artifact_manager')
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def queue_experiment(self, data: Dict[str, Any], experiment_id: str = None) -> str:
        """
        Queue experiment data for validation and upload.
        
        Args:
            data: Experiment data dictionary
            experiment_id: Optional experiment ID (auto-generated if None)
            
        Returns:
            Artifact hash ID for tracking
        """
        # Generate experiment ID if not provided
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_type = data.get('experiment_type', 'unknown')
            experiment_id = f"{exp_type}_{timestamp}"
        
        # Ensure experiment_id is in data
        data['experiment_id'] = experiment_id
        
        # Generate stable hash for deduplication
        content = json.dumps(data, sort_keys=True, default=str)
        artifact_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Save to queue
        queue_file = Path(self.config.queue_dir) / f"{artifact_hash}.json"
        
        with open(queue_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.stats['queued'] += 1
        self.logger.info(f"Queued experiment {experiment_id} as {artifact_hash}")
        
        return artifact_hash
    
    def validate_queued_file(self, file_path: Path) -> Optional[ExperimentSchema]:
        """
        Validate a queued file against schemas.
        
        Args:
            file_path: Path to queued JSON file
            
        Returns:
            Validated schema instance or None if validation fails
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Auto-migrate if enabled
            if self.config.auto_migrate:
                data = migrate_legacy_data(data)
            
            # Validate against schema
            validated_data = validate_experiment_data(data)
            
            self.stats['validated'] += 1
            self.logger.info(f"Validated {file_path.name}")
            
            return validated_data
            
        except ValidationError as e:
            self.stats['rejected'] += 1
            self.logger.error(f"Validation failed for {file_path.name}: {e}")
            
            # Move to rejected directory
            rejected_file = Path(self.config.rejected_dir) / file_path.name
            shutil.move(str(file_path), str(rejected_file))
            
            # Save error details
            error_file = rejected_file.with_suffix('.error.json')
            with open(error_file, 'w') as f:
                json.dump({
                    'file': file_path.name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            return None
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Unexpected error validating {file_path.name}: {e}")
            return None
    
    def check_artifact_exists(self, artifact_hash: str) -> bool:
        """
        Check if artifact already exists in WandB.
        
        Args:
            artifact_hash: Hash ID of artifact
            
        Returns:
            True if artifact exists, False otherwise
        """
        try:
            artifact_name = f"{self.config.project_name}/{artifact_hash}:latest"
            if self.config.entity:
                artifact_name = f"{self.config.entity}/{artifact_name}"
            
            self.api.artifact(artifact_name)
            return True
            
        except wandb.errors.CommError:
            return False
        except Exception as e:
            self.logger.warning(f"Error checking artifact {artifact_hash}: {e}")
            return False
    
    def upload_artifact(self, file_path: Path, validated_data: ExperimentSchema) -> bool:
        """
        Upload validated data as WandB artifact.
        
        Args:
            file_path: Path to validated file
            validated_data: Validated experiment data
            
        Returns:
            True if upload successful, False otherwise
        """
        artifact_hash = file_path.stem
        
        try:
            # Check for duplicates
            if self.check_artifact_exists(artifact_hash):
                self.stats['duplicates_skipped'] += 1
                self.logger.info(f"Skipping duplicate artifact {artifact_hash}")
                
                # Move to sent directory
                sent_file = Path(self.config.sent_dir) / file_path.name
                shutil.move(str(file_path), str(sent_file))
                
                return True
            
            # Initialize WandB run
            run = wandb.init(
                project=self.config.project_name,
                entity=self.config.entity,
                job_type="artifact_upload",
                name=f"upload_{artifact_hash}",
                reinit=True
            )
            
            try:
                # Create artifact
                artifact = wandb.Artifact(
                    name=artifact_hash,
                    type="experiment_result",
                    description=f"Experiment: {validated_data.experiment_type}",
                    metadata={
                        'experiment_id': validated_data.experiment_id,
                        'experiment_type': validated_data.experiment_type,
                        'schema_version': validated_data.schema_version,
                        'upload_timestamp': datetime.now().isoformat()
                    }
                )
                
                # Add file to artifact
                artifact.add_file(str(file_path), name=f"{artifact_hash}.json")
                
                # Log artifact
                run.log_artifact(artifact)
                
                self.stats['uploaded'] += 1
                self.logger.info(f"Uploaded artifact {artifact_hash}")
                
                # Move to sent directory
                sent_file = Path(self.config.sent_dir) / file_path.name
                shutil.move(str(file_path), str(sent_file))
                
                return True
                
            finally:
                run.finish()
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Failed to upload {artifact_hash}: {e}")
            return False
    
    def process_queue(self, max_files: int = None) -> Dict[str, int]:
        """
        Process all files in the queue.
        
        Args:
            max_files: Maximum number of files to process (None for all)
            
        Returns:
            Processing statistics
        """
        queue_path = Path(self.config.queue_dir)
        json_files = list(queue_path.glob("*.json"))
        
        if max_files:
            json_files = json_files[:max_files]
        
        self.logger.info(f"Processing {len(json_files)} files from queue")
        
        for file_path in json_files:
            try:
                # Validate file
                validated_data = self.validate_queued_file(file_path)
                
                if validated_data is None:
                    continue  # Validation failed, file moved to rejected
                
                # Upload to WandB
                success = self.upload_artifact(file_path, validated_data)
                
                if not success:
                    self.logger.warning(f"Upload failed for {file_path.name}, leaving in queue")
                
            except Exception as e:
                self.stats['errors'] += 1
                self.logger.error(f"Error processing {file_path.name}: {e}")
        
        return dict(self.stats)
    
    def retry_failed_uploads(self) -> Dict[str, int]:
        """
        Retry files that failed upload (still in queue).
        
        Returns:
            Retry statistics
        """
        self.logger.info("Retrying failed uploads...")
        return self.process_queue()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status.
        
        Returns:
            Queue status information
        """
        queue_path = Path(self.config.queue_dir)
        sent_path = Path(self.config.sent_dir)
        rejected_path = Path(self.config.rejected_dir)
        
        status = {
            'queue_size': len(list(queue_path.glob("*.json"))),
            'sent_count': len(list(sent_path.glob("*.json"))),
            'rejected_count': len(list(rejected_path.glob("*.json"))),
            'processing_stats': dict(self.stats),
            'directories': {
                'queue': str(queue_path),
                'sent': str(sent_path),
                'rejected': str(rejected_path)
            }
        }
        
        return status
    
    def clean_sent_files(self, older_than_days: int = 30) -> int:
        """
        Clean old files from sent directory.
        
        Args:
            older_than_days: Remove files older than this many days
            
        Returns:
            Number of files removed
        """
        sent_path = Path(self.config.sent_dir)
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        
        removed_count = 0
        for file_path in sent_path.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                removed_count += 1
        
        self.logger.info(f"Cleaned {removed_count} old files from sent directory")
        return removed_count
    
    def requeue_rejected_file(self, filename: str) -> bool:
        """
        Move a rejected file back to queue for reprocessing.
        
        Args:
            filename: Name of rejected file
            
        Returns:
            True if successful, False otherwise
        """
        rejected_file = Path(self.config.rejected_dir) / filename
        queue_file = Path(self.config.queue_dir) / filename
        
        if not rejected_file.exists():
            self.logger.error(f"Rejected file {filename} not found")
            return False
        
        try:
            shutil.move(str(rejected_file), str(queue_file))
            
            # Remove error file if it exists
            error_file = rejected_file.with_suffix('.error.json')
            if error_file.exists():
                error_file.unlink()
            
            self.logger.info(f"Requeued {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to requeue {filename}: {e}")
            return False
    
    def get_rejected_files_info(self) -> List[Dict[str, Any]]:
        """
        Get information about rejected files.
        
        Returns:
            List of rejected file information
        """
        rejected_path = Path(self.config.rejected_dir)
        rejected_info = []
        
        for json_file in rejected_path.glob("*.json"):
            error_file = json_file.with_suffix('.error.json')
            
            info = {
                'filename': json_file.name,
                'size': json_file.stat().st_size,
                'modified': datetime.fromtimestamp(json_file.stat().st_mtime).isoformat(),
                'error': None
            }
            
            if error_file.exists():
                try:
                    with open(error_file, 'r') as f:
                        error_data = json.load(f)
                    info['error'] = error_data.get('error', 'Unknown error')
                except Exception:
                    info['error'] = 'Error file corrupted'
            
            rejected_info.append(info)
        
        return rejected_info


class ArtifactUploader:
    """
    Background uploader daemon for processing the artifact queue.
    
    Can be run as a separate process or scheduled task.
    """
    
    def __init__(self, config: ArtifactConfig = None, interval: float = 60.0):
        """
        Initialize uploader daemon.
        
        Args:
            config: Artifact configuration
            interval: Processing interval in seconds
        """
        self.manager = ArtifactManager(config)
        self.interval = interval
        self.running = False
    
    def start(self):
        """Start the uploader daemon."""
        self.running = True
        self.manager.logger.info("Artifact uploader daemon started")
        
        while self.running:
            try:
                stats = self.manager.process_queue()
                
                if stats['queued'] > 0 or stats['uploaded'] > 0:
                    self.manager.logger.info(f"Processing cycle complete: {stats}")
                
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                self.manager.logger.info("Uploader daemon stopped by user")
                break
            except Exception as e:
                self.manager.logger.error(f"Error in uploader daemon: {e}")
                time.sleep(self.interval)
    
    def stop(self):
        """Stop the uploader daemon."""
        self.running = False
        self.manager.logger.info("Artifact uploader daemon stopping...")


# Convenience functions for easy usage
def queue_experiment(data: Dict[str, Any], experiment_id: str = None, config: ArtifactConfig = None) -> str:
    """
    Convenience function to queue an experiment.
    
    Args:
        data: Experiment data
        experiment_id: Optional experiment ID
        config: Optional artifact configuration
        
    Returns:
        Artifact hash ID
    """
    manager = ArtifactManager(config)
    return manager.queue_experiment(data, experiment_id)


def process_queue(config: ArtifactConfig = None) -> Dict[str, int]:
    """
    Convenience function to process the queue.
    
    Args:
        config: Optional artifact configuration
        
    Returns:
        Processing statistics
    """
    manager = ArtifactManager(config)
    return manager.process_queue()


def get_queue_status(config: ArtifactConfig = None) -> Dict[str, Any]:
    """
    Convenience function to get queue status.
    
    Args:
        config: Optional artifact configuration
        
    Returns:
        Queue status
    """
    manager = ArtifactManager(config)
    return manager.get_queue_status()


# Export main components
__all__ = [
    'ArtifactConfig',
    'ArtifactManager', 
    'ArtifactUploader',
    'queue_experiment',
    'process_queue',
    'get_queue_status'
]
