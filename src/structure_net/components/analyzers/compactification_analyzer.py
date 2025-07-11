"""
Compactification analyzer component.

This analyzer combines multiple compactification metrics to provide
comprehensive analysis of network compression effectiveness.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import logging
import numpy as np

from src.structure_net.core import (
    BaseAnalyzer, IModel, ILayer, EvolutionContext, AnalysisReport,
    ComponentContract, ComponentVersion, Maturity,
    ResourceRequirements, ResourceLevel
)
from src.structure_net.components.metrics import (
    CompressionRatioMetric, PatchEffectivenessMetric,
    MemoryEfficiencyMetric, ReconstructionQualityMetric
)


class CompactificationAnalyzer(BaseAnalyzer):
    """
    Analyzes network compactification effectiveness comprehensively.
    
    Combines compression ratio, patch effectiveness, memory efficiency,
    and reconstruction quality metrics to provide insights into the
    quality and effectiveness of network compactification.
    """
    
    def __init__(self, quality_thresholds: Optional[Dict[str, float]] = None,
                 name: str = None):
        """
        Initialize compactification analyzer.
        
        Args:
            quality_thresholds: Quality thresholds for grading
            name: Optional custom name
        """
        super().__init__(name or "CompactificationAnalyzer")
        
        # Default quality thresholds
        self.quality_thresholds = quality_thresholds or {
            'excellent': 0.9,
            'good': 0.8,
            'acceptable': 0.7,
            'poor': 0.6
        }
        
        # Initialize metrics
        self._compression_metric = CompressionRatioMetric()
        self._patch_metric = PatchEffectivenessMetric()
        self._memory_metric = MemoryEfficiencyMetric()
        self._reconstruction_metric = ReconstructionQualityMetric()
        
        self._required_metrics = {
            "CompressionRatioMetric",
            "PatchEffectivenessMetric",
            "MemoryEfficiencyMetric",
            "ReconstructionQualityMetric"
        }
    
    @property
    def contract(self) -> ComponentContract:
        """Define the component contract."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"compact_data"},
            optional_inputs={"original_network", "compact_history"},
            provided_outputs={
                "analysis.compression_analysis",
                "analysis.patch_analysis",
                "analysis.memory_analysis",
                "analysis.quality_analysis",
                "analysis.overall_assessment",
                "analysis.recommendations",
                "analysis.trends"
            },
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, 
                         context: EvolutionContext) -> Dict[str, Any]:
        """
        Perform comprehensive compactification analysis.
        
        Args:
            model: Model to analyze (may be None for compact data analysis)
            report: Analysis report containing metric data
            context: Evolution context with compact data
            
        Returns:
            Dictionary containing analysis results
        """
        # Get compact data
        compact_data = context.get('compact_data')
        if compact_data is None:
            raise ValueError("CompactificationAnalyzer requires 'compact_data' in context")
        
        original_network = context.get('original_network')
        compact_history = context.get('compact_history', [])
        
        # Run compression analysis
        compression_analysis = self._analyze_compression(
            compact_data, original_network, report, context
        )
        
        # Run patch analysis
        patch_analysis = self._analyze_patches(
            compact_data, report, context
        )
        
        # Run memory analysis
        memory_analysis = self._analyze_memory(
            compact_data, report, context
        )
        
        # Run quality analysis
        quality_analysis = self._analyze_quality(
            compact_data, report, context
        )
        
        # Overall assessment
        overall_assessment = self._compute_overall_assessment(
            compression_analysis, patch_analysis, 
            memory_analysis, quality_analysis
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            compression_analysis, patch_analysis,
            memory_analysis, quality_analysis,
            overall_assessment
        )
        
        # Analyze trends if history available
        trends = {}
        if compact_history:
            trends = self._analyze_trends(compact_history, context)
        
        return {
            "compression_analysis": compression_analysis,
            "patch_analysis": patch_analysis,
            "memory_analysis": memory_analysis,
            "quality_analysis": quality_analysis,
            "overall_assessment": overall_assessment,
            "recommendations": recommendations,
            "trends": trends
        }
    
    def _analyze_compression(self, compact_data: Dict[str, Any],
                           original_network: Optional[nn.Module],
                           report: AnalysisReport,
                           context: EvolutionContext) -> Dict[str, Any]:
        """Analyze compression effectiveness."""
        # Create context for compression metric
        comp_context = EvolutionContext({
            'compact_data': compact_data,
            'original_network': original_network
        })
        
        # Run compression metric
        comp_key = "CompressionRatioMetric"
        if comp_key not in report.metrics:
            comp_result = self._compression_metric.analyze(None, comp_context)
            report.add_metric_data(comp_key, comp_result)
        else:
            comp_result = report.get(f"metrics.{comp_key}")
        
        # Add interpretation
        efficiency_grade = self._grade_efficiency(comp_result['efficiency_score'])
        
        return {
            **comp_result,
            "efficiency_grade": efficiency_grade,
            "compression_quality": self._assess_compression_quality(comp_result)
        }
    
    def _analyze_patches(self, compact_data: Dict[str, Any],
                        report: AnalysisReport,
                        context: EvolutionContext) -> Dict[str, Any]:
        """Analyze patch effectiveness."""
        # Create context for patch metric
        patch_context = EvolutionContext({'compact_data': compact_data})
        
        # Run patch metric
        patch_key = "PatchEffectivenessMetric"
        if patch_key not in report.metrics:
            patch_result = self._patch_metric.analyze(None, patch_context)
            report.add_metric_data(patch_key, patch_result)
        else:
            patch_result = report.get(f"metrics.{patch_key}")
        
        # Add interpretation
        patch_quality = (patch_result['avg_patch_density'] + 
                        patch_result['information_preservation']) / 2
        
        return {
            **patch_result,
            "patch_quality_score": patch_quality,
            "patch_distribution": self._analyze_patch_distribution(compact_data)
        }
    
    def _analyze_memory(self, compact_data: Dict[str, Any],
                       report: AnalysisReport,
                       context: EvolutionContext) -> Dict[str, Any]:
        """Analyze memory efficiency."""
        # Create context for memory metric
        mem_context = EvolutionContext({'compact_data': compact_data})
        
        # Run memory metric
        mem_key = "MemoryEfficiencyMetric"
        if mem_key not in report.metrics:
            mem_result = self._memory_metric.analyze(None, mem_context)
            report.add_metric_data(mem_key, mem_result)
        else:
            mem_result = report.get(f"metrics.{mem_key}")
        
        # Add memory distribution analysis
        total_memory = mem_result['total_memory']
        if total_memory > 0:
            memory_distribution = {
                "patch_percentage": (mem_result['patch_memory'] / total_memory) * 100,
                "skeleton_percentage": (mem_result['skeleton_memory'] / total_memory) * 100,
                "metadata_percentage": (mem_result['metadata_memory'] / total_memory) * 100
            }
        else:
            memory_distribution = {
                "patch_percentage": 0,
                "skeleton_percentage": 0,
                "metadata_percentage": 0
            }
        
        return {
            **mem_result,
            "memory_distribution": memory_distribution
        }
    
    def _analyze_quality(self, compact_data: Dict[str, Any],
                        report: AnalysisReport,
                        context: EvolutionContext) -> Dict[str, Any]:
        """Analyze reconstruction quality."""
        # Create context for reconstruction metric
        recon_context = EvolutionContext({'compact_data': compact_data})
        
        # Run reconstruction metric
        recon_key = "ReconstructionQualityMetric"
        if recon_key not in report.metrics:
            recon_result = self._reconstruction_metric.analyze(None, recon_context)
            report.add_metric_data(recon_key, recon_result)
        else:
            recon_result = report.get(f"metrics.{recon_key}")
        
        # Compute overall quality score
        quality_score = (recon_result['fidelity_score'] + 
                        recon_result['structural_preservation'] - 
                        recon_result['information_loss']) / 3
        quality_score = max(0, min(1, quality_score))
        
        return {
            **recon_result,
            "overall_quality_score": quality_score,
            "quality_grade": self._grade_quality(quality_score)
        }
    
    def _compute_overall_assessment(self, compression_analysis: Dict[str, Any],
                                  patch_analysis: Dict[str, Any],
                                  memory_analysis: Dict[str, Any],
                                  quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall compactification assessment."""
        # Component scores
        compression_score = compression_analysis['efficiency_score']
        patch_score = patch_analysis['patch_quality_score']
        memory_score = memory_analysis['memory_efficiency']
        quality_score = quality_analysis['overall_quality_score']
        
        # Weighted overall score
        overall_score = (
            0.3 * compression_score +
            0.2 * patch_score +
            0.2 * memory_score +
            0.3 * quality_score
        )
        
        # Assessment
        assessment = {
            "overall_score": overall_score,
            "overall_grade": self._grade_quality(overall_score),
            "component_scores": {
                "compression": compression_score,
                "patches": patch_score,
                "memory": memory_score,
                "quality": quality_score
            },
            "status": self._determine_status(overall_score),
            "efficiency_rating": self._rate_efficiency(compression_score, memory_score)
        }
        
        return assessment
    
    def _analyze_patch_distribution(self, compact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze distribution of patches."""
        if 'patches' not in compact_data:
            return {"patch_sizes": [], "size_distribution": {}}
        
        patch_sizes = []
        for patch_info in compact_data['patches']:
            if 'data' in patch_info:
                patch_sizes.append(patch_info['data'].numel())
        
        if not patch_sizes:
            return {"patch_sizes": [], "size_distribution": {}}
        
        return {
            "patch_sizes": patch_sizes,
            "size_distribution": {
                "mean": np.mean(patch_sizes),
                "std": np.std(patch_sizes),
                "min": np.min(patch_sizes),
                "max": np.max(patch_sizes),
                "total": sum(patch_sizes)
            }
        }
    
    def _analyze_trends(self, compact_history: List[Dict[str, Any]],
                       context: EvolutionContext) -> Dict[str, Any]:
        """Analyze compactification trends over time."""
        if len(compact_history) < 2:
            return {}
        
        # Extract time series
        compression_ratios = []
        quality_scores = []
        memory_efficiencies = []
        
        for compact_data in compact_history[-10:]:  # Last 10 entries
            # Quick analysis for each
            hist_context = EvolutionContext({'compact_data': compact_data})
            
            comp_result = self._compression_metric.analyze(None, hist_context)
            compression_ratios.append(comp_result['compression_ratio'])
            
            mem_result = self._memory_metric.analyze(None, hist_context)
            memory_efficiencies.append(mem_result['memory_efficiency'])
            
            qual_result = self._reconstruction_metric.analyze(None, hist_context)
            quality_scores.append(qual_result['fidelity_score'])
        
        # Compute trends
        def compute_trend(values):
            if len(values) < 2:
                return 0.0
            # Simple linear trend
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            return z[0]  # Slope
        
        return {
            "compression_trend": compute_trend(compression_ratios),
            "quality_trend": compute_trend(quality_scores),
            "memory_efficiency_trend": compute_trend(memory_efficiencies),
            "stability": 1.0 - np.std(quality_scores) if quality_scores else 0.0,
            "samples_analyzed": len(compression_ratios)
        }
    
    def _generate_recommendations(self, compression_analysis: Dict[str, Any],
                                patch_analysis: Dict[str, Any],
                                memory_analysis: Dict[str, Any],
                                quality_analysis: Dict[str, Any],
                                overall_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Overall assessment
        if overall_assessment['overall_score'] < 0.6:
            recommendations.append(
                "Overall compactification quality is below acceptable levels. "
                "Consider adjusting compression parameters."
            )
        
        # Compression recommendations
        if compression_analysis['compression_ratio'] > 0.5:
            recommendations.append(
                "Compression ratio is high (>50%). Consider more aggressive "
                "pruning or quantization strategies."
            )
        
        # Patch recommendations
        if patch_analysis['patch_count'] > 1000:
            recommendations.append(
                f"High patch count ({patch_analysis['patch_count']}). "
                "Consider consolidating smaller patches for better efficiency."
            )
        
        if patch_analysis['avg_patch_density'] < 0.1:
            recommendations.append(
                "Low average patch density. Consider removing or merging "
                "sparse patches."
            )
        
        # Memory recommendations
        memory_dist = memory_analysis['memory_distribution']
        if memory_dist['metadata_percentage'] > 20:
            recommendations.append(
                "High metadata overhead (>20%). Consider optimizing patch "
                "indexing structure."
            )
        
        # Quality recommendations
        if quality_analysis['information_loss'] > 0.3:
            recommendations.append(
                "Significant information loss detected (>30%). Consider "
                "reducing compression aggressiveness."
            )
        
        if quality_analysis['reconstruction_error'] > 0.1:
            recommendations.append(
                "High reconstruction error. Verify compactification algorithm "
                "preserves critical network features."
            )
        
        # Add positive feedback
        if overall_assessment['overall_score'] > 0.8:
            recommendations.insert(0, 
                "Excellent compactification achieved with good balance of "
                "compression and quality."
            )
        
        return recommendations
    
    def _grade_efficiency(self, score: float) -> str:
        """Convert efficiency score to grade."""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.6:
            return 'C'
        else:
            return 'D'
    
    def _grade_quality(self, score: float) -> str:
        """Convert quality score to grade."""
        if score >= self.quality_thresholds['excellent']:
            return 'Excellent'
        elif score >= self.quality_thresholds['good']:
            return 'Good'
        elif score >= self.quality_thresholds['acceptable']:
            return 'Acceptable'
        elif score >= self.quality_thresholds['poor']:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _assess_compression_quality(self, comp_result: Dict[str, Any]) -> str:
        """Assess compression quality."""
        ratio = comp_result['compression_ratio']
        if ratio < 0.1:
            return "Excellent compression"
        elif ratio < 0.3:
            return "Good compression"
        elif ratio < 0.5:
            return "Moderate compression"
        else:
            return "Limited compression"
    
    def _determine_status(self, score: float) -> str:
        """Determine compactification status."""
        if score >= 0.8:
            return "Optimal"
        elif score >= 0.6:
            return "Acceptable"
        elif score >= 0.4:
            return "Suboptimal"
        else:
            return "Poor"
    
    def _rate_efficiency(self, compression_score: float, 
                        memory_score: float) -> str:
        """Rate overall efficiency."""
        avg_efficiency = (compression_score + memory_score) / 2
        if avg_efficiency >= 0.8:
            return "Highly efficient"
        elif avg_efficiency >= 0.6:
            return "Moderately efficient"
        else:
            return "Inefficient"