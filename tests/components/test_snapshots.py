"""
Tests for snapshot components.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any

from src.structure_net.core import EvolutionContext, AnalysisReport, EvolutionPlan
from src.structure_net.components.orchestrators import SnapshotOrchestrator
from src.structure_net.components.strategies import SnapshotStrategy
from src.structure_net.components.metrics import SnapshotMetric
from src.structure_net.components.models import MinimalModel


class TestSnapshotOrchestrator:
    """Test SnapshotOrchestrator component."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def orchestrator(self, temp_dir):
        return SnapshotOrchestrator(
            save_dir=temp_dir,
            performance_threshold=0.02,
            max_snapshots=10,
            use_deltas=True
        )
    
    @pytest.fixture
    def model(self):
        return MinimalModel([784, 256, 128, 10], sparsity=0.9)
    
    @pytest.fixture
    def context(self, model):
        return EvolutionContext({
            'model': model,
            'epoch': 10,
            'performance': {'accuracy': 0.85, 'loss': 0.3},
            'growth_info': {'growth_occurred': True, 'connections_added': 50}
        })
    
    def test_initialization(self, orchestrator, temp_dir):
        """Test orchestrator initialization."""
        assert orchestrator.save_dir == Path(temp_dir)
        assert orchestrator.performance_threshold == 0.02
        assert orchestrator.use_deltas
        assert orchestrator.max_snapshots == 10
    
    def test_should_save_growth(self, orchestrator):
        """Test snapshot decision for growth events."""
        context = EvolutionContext({
            'epoch': 10,
            'growth_info': {'growth_occurred': True}
        })
        
        should_save, reason = orchestrator._should_save_snapshot(context)
        assert should_save
        assert reason == 'growth_event'
    
    def test_should_save_performance(self, orchestrator):
        """Test snapshot decision for performance improvement."""
        # Set up previous performance
        orchestrator.last_performance = 0.8
        
        context = EvolutionContext({
            'epoch': 10,
            'performance': {'accuracy': 0.85},  # 5% improvement
            'growth_info': {'growth_occurred': False}
        })
        
        should_save, reason = orchestrator._should_save_snapshot(context)
        assert should_save
        assert 'performance_improvement' in reason
    
    def test_should_save_milestone(self, orchestrator):
        """Test snapshot decision for milestone epochs."""
        context = EvolutionContext({
            'epoch': 100,  # Milestone epoch
            'performance': {'accuracy': 0.8},
            'growth_info': {'growth_occurred': False}
        })
        
        should_save, reason = orchestrator._should_save_snapshot(context)
        assert should_save
        assert 'milestone_epoch_100' in reason
    
    def test_run_cycle_save(self, orchestrator, context):
        """Test running cycle with snapshot save."""
        results = orchestrator.run_cycle(context)
        
        assert results['snapshot_saved']
        assert results['snapshot_id'] is not None
        assert 'growth_event' in results['reason']
        assert results['stats']['total_saved'] == 1
    
    def test_run_cycle_no_save(self, orchestrator):
        """Test running cycle without snapshot save."""
        context = EvolutionContext({
            'model': MinimalModel([784, 128, 10]),
            'epoch': 5,  # Non-milestone
            'performance': {'accuracy': 0.8},
            'growth_info': {'growth_occurred': False}
        })
        
        results = orchestrator.run_cycle(context)
        
        assert not results['snapshot_saved']
        assert results['snapshot_id'] is None
    
    def test_save_full_snapshot(self, orchestrator, model, temp_dir):
        """Test saving full snapshot."""
        context = EvolutionContext({
            'epoch': 10,
            'performance': {'accuracy': 0.85},
            'growth_info': {'growth_occurred': True}
        })
        
        snapshot_id = orchestrator._save_snapshot(model, context, 'test')
        
        # Check files created
        snapshot_dir = Path(temp_dir) / snapshot_id
        assert snapshot_dir.exists()
        assert (snapshot_dir / 'model_state.pt').exists()
        assert (snapshot_dir / 'architecture.json').exists()
        assert (snapshot_dir / 'metadata.json').exists()
    
    def test_save_delta_snapshot(self, orchestrator, model, temp_dir):
        """Test saving delta snapshot."""
        context = EvolutionContext({
            'epoch': 10,
            'performance': {'accuracy': 0.85},
            'growth_info': {'growth_occurred': True}
        })
        
        # Save first snapshot (full)
        first_id = orchestrator._save_snapshot(model, context, 'test_full')
        
        # Modify model slightly
        with torch.no_grad():
            for param in model.parameters():
                param += torch.randn_like(param) * 0.01
        
        # Save second snapshot (delta)
        context['epoch'] = 15
        second_id = orchestrator._save_snapshot(model, context, 'test_delta')
        
        # Check delta files created
        snapshot_dir = Path(temp_dir) / second_id
        assert (snapshot_dir / 'delta_state.pt').exists()
        assert (snapshot_dir / 'delta_info.json').exists()
    
    def test_load_snapshot(self, orchestrator, model, context):
        """Test loading snapshot."""
        # Save snapshot
        snapshot_id = orchestrator._save_snapshot(model, context, 'test')
        
        # Create new model with different weights
        new_model = MinimalModel([784, 256, 128, 10], sparsity=0.9)
        
        # Load snapshot
        metadata = orchestrator.load_snapshot(snapshot_id, new_model)
        
        assert metadata['snapshot_id'] == snapshot_id
        assert metadata['epoch'] == 10
    
    def test_composition_health(self, orchestrator, context):
        """Test composition health reporting."""
        # Save some snapshots
        orchestrator.run_cycle(context)
        
        health = orchestrator.get_composition_health()
        
        assert health['total_snapshots'] == 1
        assert 'disk_usage_mb' in health
        assert 'snapshot_types' in health
        assert health['health_status'] == 'healthy'


class TestSnapshotStrategy:
    """Test SnapshotStrategy component."""
    
    @pytest.fixture
    def strategy(self):
        return SnapshotStrategy(
            min_improvement=0.02,
            growth_priority=0.9,
            performance_priority=0.7
        )
    
    @pytest.fixture
    def report(self):
        report = AnalysisReport()
        report['model_stats'] = {
            'total_parameters': 100000,
            'architecture': [784, 256, 10]
        }
        report['performance_metrics'] = {
            'accuracy': 0.85,
            'loss': 0.2
        }
        report['growth_analysis'] = {
            'growth_occurred': False
        }
        return report
    
    @pytest.fixture
    def context(self):
        return EvolutionContext({'epoch': 10})
    
    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.min_improvement == 0.02
        assert strategy.growth_priority == 0.9
        assert strategy.get_strategy_type() == 'snapshot'
    
    def test_propose_plan_growth(self, strategy, report, context):
        """Test plan proposal for growth event."""
        report['growth_analysis']['growth_occurred'] = True
        
        plan = strategy.propose_plan(report, context)
        
        assert plan['type'] == 'snapshot'
        assert plan['action'] == 'save'
        assert plan['criteria_met']['growth_event']
        assert plan.priority == 0.9
    
    def test_propose_plan_performance(self, strategy, report, context):
        """Test plan proposal for performance improvement."""
        # Set previous performance
        strategy.last_performance = 0.8
        report['performance_metrics']['accuracy'] = 0.85  # 5% improvement
        
        plan = strategy.propose_plan(report, context)
        
        assert plan['type'] == 'snapshot'
        assert plan['action'] == 'save'
        assert plan['criteria_met']['performance_improvement']
    
    def test_propose_plan_no_op(self, strategy, report, context):
        """Test no-op plan when criteria not met."""
        # No growth, no significant performance change
        strategy.last_performance = 0.84  # Small change
        
        plan = strategy.propose_plan(report, context)
        
        assert plan['type'] == 'snapshot'
        assert plan['action'] == 'skip'
        assert plan.priority == 0.0
    
    def test_milestone_detection(self, strategy, report):
        """Test milestone epoch detection."""
        context = EvolutionContext({'epoch': 100})  # Milestone
        
        criteria = strategy._evaluate_criteria(report, context)
        assert criteria['milestone_epoch']
    
    def test_interval_check(self, strategy):
        """Test minimum interval checking."""
        strategy.last_snapshot_epoch = 5
        
        context = EvolutionContext({'epoch': 8})  # 3 epochs later
        assert not strategy._check_interval(context)
        
        context = EvolutionContext({'epoch': 12})  # 7 epochs later
        assert strategy._check_interval(context)


class TestSnapshotMetric:
    """Test SnapshotMetric component."""
    
    @pytest.fixture
    def metric(self):
        return SnapshotMetric()
    
    @pytest.fixture
    def model(self):
        return MinimalModel([784, 256, 10], sparsity=0.9)
    
    def test_initialization(self, metric):
        """Test metric initialization."""
        assert metric.name == "SnapshotMetric"
        assert metric.total_snapshots == 0
        
        schema = metric.get_measurement_schema()
        assert 'snapshot_count' in schema
        assert 'compression_ratio' in schema
    
    def test_analyze_basic(self, metric):
        """Test basic analysis without model."""
        context = EvolutionContext({
            'snapshot_info': {
                'total_saved': 5,
                'total_size_mb': 250.0,
                'deltas_saved': 3
            }
        })
        
        results = metric.analyze(None, context)
        
        assert results['snapshot_count'] == 5
        assert results['total_size_mb'] == 250.0
        assert results['delta_ratio'] == 0.6  # 3/5
    
    def test_analyze_with_model(self, metric, model):
        """Test analysis with model for compression calculation."""
        context = EvolutionContext({
            'snapshot_info': {
                'total_saved': 3,
                'total_size_mb': 30.0,  # Small due to compression
                'deltas_saved': 2
            }
        })
        
        results = metric.analyze(model, context)
        
        assert 'compression_ratio' in results
        assert results['compression_ratio'] > 0
        assert 'storage_efficiency' in results
    
    def test_coverage_analysis(self, metric):
        """Test snapshot coverage analysis."""
        history = [
            {'epoch': 10, 'reason': 'growth_event'},
            {'epoch': 25, 'reason': 'performance_improvement'},
            {'epoch': 50, 'reason': 'milestone_epoch_50'},
            {'epoch': 75, 'reason': 'growth_event'}
        ]
        
        context = EvolutionContext({
            'epoch': 100,
            'snapshot_info': {'total_saved': 4},
            'snapshot_history': history
        })
        
        results = metric.analyze(None, context)
        
        assert 'epoch_coverage' in results
        assert 'performance_coverage' in results
        assert 'growth_coverage' in results
        assert results['epoch_coverage'] > 0
    
    def test_efficiency_report(self, metric):
        """Test efficiency report generation."""
        # Simulate some tracking
        metric.total_snapshots = 10
        metric.total_size_mb = 500.0
        metric.delta_count = 7
        metric.full_count = 3
        
        report = metric.get_efficiency_report()
        
        assert report['total_snapshots'] == 10
        assert report['average_size_mb'] == 50.0
        assert report['delta_usage']['delta_percentage'] == 70.0
        assert 'recommendations' in report


class TestSnapshotIntegration:
    """Test integration between snapshot components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def components(self, temp_dir):
        """Create snapshot components."""
        orchestrator = SnapshotOrchestrator(temp_dir, performance_threshold=0.02)
        strategy = SnapshotStrategy(min_improvement=0.02)
        metric = SnapshotMetric()
        
        return orchestrator, strategy, metric
    
    @pytest.fixture
    def model(self):
        return MinimalModel([784, 256, 10], sparsity=0.9)
    
    def test_strategy_orchestrator_integration(self, components, model):
        """Test strategy and orchestrator working together."""
        orchestrator, strategy, metric = components
        
        # Create analysis report
        report = AnalysisReport()
        report['model_stats'] = {'total_parameters': 50000}
        report['performance_metrics'] = {'accuracy': 0.85}
        report['growth_analysis'] = {'growth_occurred': True}
        
        context = EvolutionContext({
            'model': model,
            'epoch': 10
        })
        
        # Strategy proposes snapshot
        plan = strategy.propose_plan(report, context)
        assert plan['type'] == 'snapshot'
        assert plan['action'] == 'save'
        
        # Orchestrator executes the plan
        results = orchestrator.run_cycle(context)
        assert results['snapshot_saved']
    
    def test_metric_tracking(self, components, model):
        """Test metric tracking of orchestrator actions."""
        orchestrator, strategy, metric = components
        
        context = EvolutionContext({
            'model': model,
            'epoch': 10,
            'growth_info': {'growth_occurred': True}
        })
        
        # Take snapshot
        results = orchestrator.run_cycle(context)
        
        # Analyze with metric
        metric_context = EvolutionContext({
            'snapshot_info': results['stats']
        })
        
        analysis = metric.analyze(model, metric_context)
        
        assert analysis['snapshot_count'] == 1
        assert analysis['total_size_mb'] > 0