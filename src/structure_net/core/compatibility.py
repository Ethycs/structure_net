"""
Compatibility management system for Structure Net components.

This module provides infrastructure for validating component compositions,
detecting compatibility issues, and maintaining a registry of available components.
"""

from typing import List, Dict, Set, Type, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from .interfaces import IComponent, ComponentContract, Maturity


class CompatibilityLevel(Enum):
    COMPATIBLE = "compatible"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CompatibilityIssue:
    level: CompatibilityLevel
    component_a: str
    component_b: str
    description: str
    suggested_fix: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.level.value.upper()}: {self.description}"


class ComponentRegistry:
    """Central registry for all available components"""
    
    def __init__(self):
        self._components: Dict[str, Type[IComponent]] = {}
        self._contracts: Dict[str, ComponentContract] = {}
        self._logger = logging.getLogger(__name__)
    
    def register(self, component_class: Type[IComponent]) -> None:
        """Register a component class"""
        try:
            # Create instance to get contract
            instance = component_class()
            contract = instance.contract
            
            self._components[contract.component_name] = component_class
            self._contracts[contract.component_name] = contract
            
            self._logger.info(f"Registered component: {contract.component_name} v{contract.version}")
            
        except Exception as e:
            self._logger.error(f"Failed to register {component_class.__name__}: {str(e)}")
            raise
    
    def get_component(self, name: str) -> Optional[Type[IComponent]]:
        """Get a component class by name"""
        return self._components.get(name)
    
    def get_contract(self, name: str) -> Optional[ComponentContract]:
        """Get a component contract by name"""
        return self._contracts.get(name)
    
    def get_available_components(self) -> Dict[str, Type[IComponent]]:
        """Get all available components"""
        return self._components.copy()
    
    def search_by_output(self, output: str) -> List[str]:
        """Find components that provide a specific output"""
        results = []
        for name, contract in self._contracts.items():
            if output in contract.provided_outputs:
                results.append(name)
        return results
    
    def search_by_maturity(self, maturity: Maturity) -> List[str]:
        """Find components with specific maturity level"""
        results = []
        for name, contract in self._contracts.items():
            if contract.maturity == maturity:
                results.append(name)
        return results
    
    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """Generate dependency graph of all components"""
        graph = {}
        
        for name, contract in self._contracts.items():
            dependencies = set()
            
            # Find components that provide required inputs
            for required_input in contract.required_inputs:
                providers = self.search_by_output(required_input)
                dependencies.update(providers)
            
            graph[name] = dependencies
        
        return graph


class CompatibilityManager:
    """Manages compatibility checking between components"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self._logger = logging.getLogger(__name__)
        self._compatibility_cache: Dict[Tuple[str, str], List[CompatibilityIssue]] = {}
    
    def validate_composition(self, components: List[IComponent]) -> List[CompatibilityIssue]:
        """Validate a complete component composition"""
        issues = []
        
        # Check pairwise compatibility
        for i, comp_a in enumerate(components):
            for comp_b in components[i+1:]:
                pair_issues = self.check_compatibility(comp_a, comp_b)
                issues.extend(pair_issues)
        
        # Check data flow completeness
        flow_issues = self._check_data_flow(components)
        issues.extend(flow_issues)
        
        # Check resource constraints
        resource_issues = self._check_resource_constraints(components)
        issues.extend(resource_issues)
        
        # Check maturity consistency
        maturity_issues = self._check_maturity_consistency(components)
        issues.extend(maturity_issues)
        
        return issues
    
    def check_compatibility(self, comp_a: IComponent, comp_b: IComponent) -> List[CompatibilityIssue]:
        """Check compatibility between two components"""
        
        # Check cache first
        cache_key = (comp_a.name, comp_b.name)
        if cache_key in self._compatibility_cache:
            return self._compatibility_cache[cache_key]
        
        issues = []
        
        try:
            contract_a = comp_a.contract
            contract_b = comp_b.contract
            
            # Version compatibility
            if not contract_a.version.is_compatible_with(contract_b.version):
                issues.append(CompatibilityIssue(
                    level=CompatibilityLevel.ERROR,
                    component_a=comp_a.name,
                    component_b=comp_b.name,
                    description=f"Version mismatch: {contract_a.version} vs {contract_b.version}",
                    suggested_fix="Ensure both components use compatible major versions"
                ))
            
            # Direct incompatibility check
            if type(comp_b) in contract_a.incompatible_with:
                issues.append(CompatibilityIssue(
                    level=CompatibilityLevel.CRITICAL,
                    component_a=comp_a.name,
                    component_b=comp_b.name,
                    description=f"{comp_a.name} is incompatible with {type(comp_b).__name__}",
                    suggested_fix="Use alternative components"
                ))
            
            if type(comp_a) in contract_b.incompatible_with:
                issues.append(CompatibilityIssue(
                    level=CompatibilityLevel.CRITICAL,
                    component_a=comp_b.name,
                    component_b=comp_a.name,
                    description=f"{comp_b.name} is incompatible with {type(comp_a).__name__}",
                    suggested_fix="Use alternative components"
                ))
            
            # Maturity level compatibility
            if contract_b.maturity not in contract_a.compatible_maturity_levels:
                issues.append(CompatibilityIssue(
                    level=CompatibilityLevel.WARNING,
                    component_a=comp_a.name,
                    component_b=comp_b.name,
                    description=f"{comp_a.name} ({contract_a.maturity.value}) may not work well with {comp_b.name} ({contract_b.maturity.value})",
                    suggested_fix="Consider using components with matching maturity levels"
                ))
            
        except Exception as e:
            issues.append(CompatibilityIssue(
                level=CompatibilityLevel.ERROR,
                component_a=comp_a.name,
                component_b=comp_b.name,
                description=f"Compatibility check failed: {str(e)}",
                suggested_fix="Ensure components properly implement contracts"
            ))
        
        # Cache results
        self._compatibility_cache[cache_key] = issues
        
        return issues
    
    def _check_data_flow(self, components: List[IComponent]) -> List[CompatibilityIssue]:
        """Check if data flow between components is complete"""
        issues = []
        
        # Collect all provided and required outputs
        all_provided = set()
        all_required = set()
        
        for comp in components:
            try:
                contract = comp.contract
                all_provided.update(contract.provided_outputs)
                all_required.update(contract.required_inputs)
            except Exception as e:
                issues.append(CompatibilityIssue(
                    level=CompatibilityLevel.ERROR,
                    component_a=comp.name,
                    component_b="system",
                    description=f"Cannot access contract: {str(e)}"
                ))
        
        # Check for missing inputs
        missing_inputs = all_required - all_provided
        if missing_inputs:
            for missing in missing_inputs:
                # Find which components need this input
                needing_components = []
                for comp in components:
                    try:
                        if missing in comp.contract.required_inputs:
                            needing_components.append(comp.name)
                    except:
                        pass
                
                issues.append(CompatibilityIssue(
                    level=CompatibilityLevel.ERROR,
                    component_a=", ".join(needing_components),
                    component_b="missing",
                    description=f"Required input '{missing}' is not provided by any component",
                    suggested_fix=f"Add a component that provides '{missing}'"
                ))
        
        return issues
    
    def _check_resource_constraints(self, components: List[IComponent]) -> List[CompatibilityIssue]:
        """Check if resource requirements are reasonable"""
        issues = []
        
        total_memory_estimate = 0
        gpu_required = False
        
        for comp in components:
            try:
                resources = comp.contract.resources
                
                # Estimate memory usage
                if resources.memory_level == ResourceLevel.LOW:
                    total_memory_estimate += 1
                elif resources.memory_level == ResourceLevel.MEDIUM:
                    total_memory_estimate += 4
                elif resources.memory_level == ResourceLevel.HIGH:
                    total_memory_estimate += 8
                elif resources.memory_level == ResourceLevel.EXTREME:
                    total_memory_estimate += 16
                
                if resources.requires_gpu:
                    gpu_required = True
                    
            except Exception:
                pass
        
        # Check total memory
        if total_memory_estimate > 32:
            issues.append(CompatibilityIssue(
                level=CompatibilityLevel.WARNING,
                component_a="composition",
                component_b="resources",
                description=f"High total memory requirement: ~{total_memory_estimate}GB",
                suggested_fix="Consider using fewer high-memory components"
            ))
        
        # Check GPU availability
        if gpu_required and not torch.cuda.is_available():
            issues.append(CompatibilityIssue(
                level=CompatibilityLevel.CRITICAL,
                component_a="composition",
                component_b="hardware",
                description="GPU required but not available",
                suggested_fix="Use CPU-compatible components or enable GPU"
            ))
        
        return issues
    
    def _check_maturity_consistency(self, components: List[IComponent]) -> List[CompatibilityIssue]:
        """Check if maturity levels are consistent"""
        issues = []
        
        maturity_levels = []
        for comp in components:
            try:
                maturity_levels.append((comp.name, comp.contract.maturity))
            except:
                pass
        
        # Count maturity levels
        stable_count = sum(1 for _, m in maturity_levels if m == Maturity.STABLE)
        experimental_count = sum(1 for _, m in maturity_levels if m == Maturity.EXPERIMENTAL)
        deprecated_count = sum(1 for _, m in maturity_levels if m == Maturity.DEPRECATED)
        
        # Warn about deprecated components
        if deprecated_count > 0:
            deprecated_names = [name for name, m in maturity_levels if m == Maturity.DEPRECATED]
            issues.append(CompatibilityIssue(
                level=CompatibilityLevel.WARNING,
                component_a=", ".join(deprecated_names),
                component_b="lifecycle",
                description="Using deprecated components",
                suggested_fix="Replace deprecated components with newer alternatives"
            ))
        
        # Warn about mixing stable and experimental
        if stable_count > 0 and experimental_count > stable_count:
            issues.append(CompatibilityIssue(
                level=CompatibilityLevel.WARNING,
                component_a="composition",
                component_b="maturity",
                description="Majority of components are experimental",
                suggested_fix="Consider using more stable components for production"
            ))
        
        return issues
    
    def suggest_fixes(self, issues: List[CompatibilityIssue]) -> Dict[str, List[str]]:
        """Suggest fixes for compatibility issues"""
        fixes = {
            "replacements": [],
            "additions": [],
            "configuration": []
        }
        
        for issue in issues:
            if issue.suggested_fix:
                if "replace" in issue.suggested_fix.lower():
                    fixes["replacements"].append(issue.suggested_fix)
                elif "add" in issue.suggested_fix.lower():
                    fixes["additions"].append(issue.suggested_fix)
                else:
                    fixes["configuration"].append(issue.suggested_fix)
        
        return fixes
    
    def find_compatible_alternatives(self, component: IComponent, target_components: List[IComponent]) -> List[str]:
        """Find alternative components that would be compatible"""
        alternatives = []
        
        # Get all available components of similar type
        all_components = self.registry.get_available_components()
        
        for name, comp_class in all_components.items():
            if name == component.name:
                continue
            
            try:
                # Check if it's the same type
                if isinstance(component, type(comp_class)):
                    # Create instance and check compatibility
                    instance = comp_class()
                    
                    # Check compatibility with all target components
                    compatible = True
                    for target in target_components:
                        issues = self.check_compatibility(instance, target)
                        if any(i.level == CompatibilityLevel.CRITICAL for i in issues):
                            compatible = False
                            break
                    
                    if compatible:
                        alternatives.append(name)
                        
            except Exception:
                pass
        
        return alternatives