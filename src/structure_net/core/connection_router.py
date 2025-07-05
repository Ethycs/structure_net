"""
Connection Router for Multi-Scale Snapshots Experiment

This module implements the connection routing logic that connects high extrema
to low extrema with controlled fan-out and load balancing.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict


class ConnectionRouter:
    """
    Routes connections from high extrema to low extrema with controlled patterns.
    
    Implements the connection routing rules:
    - Universal: High extrema → Low extrema
    - Search radius: up to 3 layers ahead
    - Connection weight initialization: small random (0.1 * randn)
    - No self-connections within same layer
    - Fan-out control: max 3-4 connections per high extrema
    """
    
    def __init__(
        self,
        max_fan_out: int = 4,  # Back to specification: 3-4 connections per high extrema
        max_search_radius: int = 3,
        connection_weight_scale: float = 0.1,
        max_incoming_per_neuron: int = 5  # Back to specification: max 5 incoming
    ):
        """
        Initialize connection router.
        
        Args:
            max_fan_out: Maximum connections per high extrema
            max_search_radius: Maximum layers ahead to search for targets
            connection_weight_scale: Scale factor for new connection weights
            max_incoming_per_neuron: Maximum incoming connections per low extrema
        """
        self.max_fan_out = max_fan_out
        self.max_search_radius = max_search_radius
        self.connection_weight_scale = connection_weight_scale
        self.max_incoming_per_neuron = max_incoming_per_neuron
        
        # Statistics
        self.routing_history = []
        self.connection_stats = defaultdict(int)
    
    def route_connections(
        self,
        extrema: Dict[int, Dict[str, List[int]]],
        layer_sizes: List[int],
        existing_connections: Optional[Dict] = None
    ) -> List[Tuple[int, int, int, int]]:
        """
        Route connections from high extrema to low extrema.
        
        Args:
            extrema: Dictionary mapping layer_idx to {'high': [...], 'low': [...]}
            layer_sizes: List of layer sizes
            existing_connections: Optional dict tracking existing connections
            
        Returns:
            List of (source_layer, source_neuron, target_layer, target_neuron) tuples
        """
        if existing_connections is None:
            existing_connections = defaultdict(lambda: defaultdict(int))
        
        new_connections = []
        routing_stats = {
            'total_high_extrema': 0,
            'total_low_extrema': 0,
            'connections_created': 0,
            'skipped_due_to_limits': 0
        }
        
        # Process each layer with high extrema
        for source_layer, layer_extrema in extrema.items():
            high_extrema = layer_extrema.get('high', [])
            routing_stats['total_high_extrema'] += len(high_extrema)
            
            # Route each high extrema
            for high_neuron in high_extrema:
                connections = self._route_single_extrema(
                    source_layer,
                    high_neuron,
                    extrema,
                    layer_sizes,
                    existing_connections
                )
                
                new_connections.extend(connections)
                routing_stats['connections_created'] += len(connections)
        
        # Count total low extrema for stats
        for layer_extrema in extrema.values():
            routing_stats['total_low_extrema'] += len(layer_extrema.get('low', []))
        
        # Record routing event
        self.routing_history.append(routing_stats)
        
        return new_connections
    
    def _route_single_extrema(
        self,
        source_layer: int,
        source_neuron: int,
        extrema: Dict[int, Dict[str, List[int]]],
        layer_sizes: List[int],
        existing_connections: Dict
    ) -> List[Tuple[int, int, int, int]]:
        """
        Route connections for a single high extrema neuron.
        
        Args:
            source_layer: Layer index of high extrema
            source_neuron: Neuron index of high extrema
            extrema: All extrema information
            layer_sizes: List of layer sizes
            existing_connections: Existing connection counts
            
        Returns:
            List of new connections for this extrema
        """
        connections = []
        targets_found = []
        
        # Search for low extrema in subsequent layers
        for target_layer in range(source_layer + 1, min(len(layer_sizes), source_layer + self.max_search_radius + 1)):
            if target_layer in extrema:
                low_extrema = extrema[target_layer].get('low', [])
                
                # Find available targets (not overloaded)
                available_targets = []
                for low_neuron in low_extrema:
                    if existing_connections[target_layer][low_neuron] < self.max_incoming_per_neuron:
                        available_targets.append((target_layer, low_neuron))
                
                targets_found.extend(available_targets)
        
        # If no low extrema found, create connections to quiet zones
        if not targets_found:
            targets_found = self._find_quiet_zones(source_layer, extrema, layer_sizes, existing_connections)
        
        # Select targets with preference for closer layers
        selected_targets = self._select_targets(targets_found, source_layer)
        
        # Create connections
        for target_layer, target_neuron in selected_targets:
            connections.append((source_layer, source_neuron, target_layer, target_neuron))
            
            # Update existing connections count
            existing_connections[target_layer][target_neuron] += 1
        
        return connections
    
    def _find_quiet_zones(
        self,
        source_layer: int,
        extrema: Dict[int, Dict[str, List[int]]],
        layer_sizes: List[int],
        existing_connections: Dict
    ) -> List[Tuple[int, int]]:
        """
        Find quiet zones (neurons that are neither high nor low extrema).
        
        Args:
            source_layer: Source layer index
            extrema: All extrema information
            layer_sizes: List of layer sizes
            existing_connections: Existing connection counts
            
        Returns:
            List of (layer, neuron) tuples for quiet zones
        """
        quiet_targets = []
        
        # Search in subsequent layers
        for target_layer in range(source_layer + 1, min(len(layer_sizes), source_layer + self.max_search_radius + 1)):
            layer_size = layer_sizes[target_layer]
            
            # Get extrema for this layer
            layer_extrema = extrema.get(target_layer, {'high': [], 'low': []})
            extrema_neurons = set(layer_extrema['high'] + layer_extrema['low'])
            
            # Find quiet neurons (not extrema, not overloaded)
            for neuron_idx in range(layer_size):
                if (neuron_idx not in extrema_neurons and 
                    existing_connections[target_layer][neuron_idx] < self.max_incoming_per_neuron):
                    quiet_targets.append((target_layer, neuron_idx))
        
        return quiet_targets
    
    def _select_targets(
        self,
        available_targets: List[Tuple[int, int]],
        source_layer: int
    ) -> List[Tuple[int, int]]:
        """
        Select targets from available options with preference for closer layers.
        
        Args:
            available_targets: List of (layer, neuron) tuples
            source_layer: Source layer index
            
        Returns:
            Selected targets (up to max_fan_out)
        """
        if not available_targets:
            return []
        
        # Sort by layer distance (prefer closer layers)
        available_targets.sort(key=lambda x: x[0] - source_layer)
        
        # Apply three-rule system for connection patterns
        selected = []
        
        # Rule 1: Distance Variation - short, medium, long connections
        distances = [1, 2, random.choice([3, 4, 5])]
        
        for distance in distances:
            target_layer = source_layer + distance
            
            # Find targets at this distance
            distance_targets = [t for t in available_targets if t[0] == target_layer]
            
            if distance_targets:
                # Select one target at this distance
                selected.append(random.choice(distance_targets))
                
                if len(selected) >= self.max_fan_out:
                    break
        
        # Rule 2: Fan-out Control - fill remaining slots if available
        remaining_slots = self.max_fan_out - len(selected)
        if remaining_slots > 0:
            # Get targets not already selected
            remaining_targets = [t for t in available_targets if t not in selected]
            
            # Randomly select remaining targets
            additional = random.sample(
                remaining_targets, 
                min(remaining_slots, len(remaining_targets))
            )
            selected.extend(additional)
        
        # Rule 3: Reciprocal Probability - handled at higher level
        
        return selected[:self.max_fan_out]
    
    def apply_load_balancing(
        self,
        connections: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Apply load balancing to prevent overconnection.
        
        Args:
            connections: List of proposed connections
            
        Returns:
            Filtered connections that respect load balancing
        """
        # Count incoming connections per neuron
        incoming_counts = defaultdict(int)
        for source_layer, source_neuron, target_layer, target_neuron in connections:
            incoming_counts[(target_layer, target_neuron)] += 1
        
        # Filter connections that would exceed limits
        balanced_connections = []
        current_counts = defaultdict(int)
        
        for connection in connections:
            source_layer, source_neuron, target_layer, target_neuron = connection
            target_key = (target_layer, target_neuron)
            
            if current_counts[target_key] < self.max_incoming_per_neuron:
                balanced_connections.append(connection)
                current_counts[target_key] += 1
        
        return balanced_connections
    
    def add_reciprocal_connections(
        self,
        connections: List[Tuple[int, int, int, int]],
        reciprocal_probability: float = 0.2
    ) -> List[Tuple[int, int, int, int]]:
        """
        Add reciprocal connections with given probability.
        
        Args:
            connections: Existing connections
            reciprocal_probability: Probability of adding reciprocal connection
            
        Returns:
            Connections with reciprocals added
        """
        reciprocal_connections = []
        
        for source_layer, source_neuron, target_layer, target_neuron in connections:
            # Add original connection
            reciprocal_connections.append((source_layer, source_neuron, target_layer, target_neuron))
            
            # Maybe add reciprocal (if layers are adjacent)
            if (target_layer == source_layer + 1 and 
                random.random() < reciprocal_probability):
                # Add reciprocal connection
                reciprocal_connections.append((target_layer, target_neuron, source_layer, source_neuron))
        
        return reciprocal_connections
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        if not self.routing_history:
            return {}
        
        total_stats = defaultdict(int)
        for stats in self.routing_history:
            for key, value in stats.items():
                total_stats[key] += value
        
        return {
            'total_routing_events': len(self.routing_history),
            'total_high_extrema_processed': total_stats['total_high_extrema'],
            'total_low_extrema_available': total_stats['total_low_extrema'],
            'total_connections_created': total_stats['connections_created'],
            'average_connections_per_event': total_stats['connections_created'] / len(self.routing_history),
            'routing_history': self.routing_history.copy()
        }
    
    def reset_stats(self):
        """Reset routing statistics."""
        self.routing_history.clear()
        self.connection_stats.clear()


class ParsimonousRouter:
    """
    Simplified router implementing the parsimonious growth recipe.
    
    This is the "80% complexity with 20% rules" version for rapid prototyping.
    """
    
    def __init__(self, max_connections_per_extrema: int = 4):
        self.max_connections = max_connections_per_extrema
    
    def parsimonious_growth(
        self,
        extrema: Dict[int, Dict[str, List[int]]],
        layer_sizes: List[int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Simple rules, rich structure - the parsimonious recipe.
        
        Args:
            extrema: Extrema information
            layer_sizes: Layer sizes
            
        Returns:
            List of connections
        """
        connections = []
        
        for source_layer, layer_extrema in extrema.items():
            high_extrema = layer_extrema.get('high', [])
            
            for high_neuron in high_extrema:
                layer_connections = []
                
                # 1. Always connect to next layer
                if source_layer + 1 < len(layer_sizes):
                    next_layer_targets = self._find_compatible_extrema(
                        source_layer + 1, extrema, layer_sizes
                    )
                    if next_layer_targets:
                        layer_connections.append((source_layer, high_neuron, source_layer + 1, random.choice(next_layer_targets)))
                
                # 2. Sometimes skip (creates depth)
                if random.random() < 0.5 and source_layer + 2 < len(layer_sizes):
                    second_layer_targets = self._find_compatible_extrema(
                        source_layer + 2, extrema, layer_sizes
                    )
                    if second_layer_targets:
                        layer_connections.append((source_layer, high_neuron, source_layer + 2, random.choice(second_layer_targets)))
                
                # 3. Rarely loop back (creates cycles)
                if random.random() < 0.1 and source_layer > 0:
                    prev_layer_targets = self._find_compatible_extrema(
                        source_layer - 1, extrema, layer_sizes
                    )
                    if prev_layer_targets:
                        layer_connections.append((source_layer, high_neuron, source_layer - 1, random.choice(prev_layer_targets)))
                
                # 4. Limit total connections
                connections.extend(layer_connections[:self.max_connections])
        
        return connections
    
    def _find_compatible_extrema(
        self,
        target_layer: int,
        extrema: Dict[int, Dict[str, List[int]]],
        layer_sizes: List[int]
    ) -> List[int]:
        """Find compatible extrema in target layer."""
        if target_layer in extrema:
            # Prefer low extrema
            low_extrema = extrema[target_layer].get('low', [])
            if low_extrema:
                return low_extrema
            
            # Fall back to high extrema if no low extrema
            high_extrema = extrema[target_layer].get('high', [])
            if high_extrema:
                return high_extrema
        
        # If no extrema, return random neurons
        layer_size = layer_sizes[target_layer]
        return list(range(min(5, layer_size)))  # Return first few neurons


# Example usage and testing
if __name__ == "__main__":
    # Test connection router
    router = ConnectionRouter()
    
    # Create sample extrema data
    extrema = {
        0: {'high': [1, 5, 10], 'low': [2, 8]},
        1: {'high': [3, 7], 'low': [0, 4, 9]},
        2: {'high': [2], 'low': [1, 6, 8]}
    }
    
    layer_sizes = [784, 256, 128, 10]
    
    # Route connections
    connections = router.route_connections(extrema, layer_sizes)
    
    print("Routed connections:")
    for conn in connections:
        print(f"Layer {conn[0]}, Neuron {conn[1]} → Layer {conn[2]}, Neuron {conn[3]}")
    
    # Apply load balancing
    balanced = router.apply_load_balancing(connections)
    print(f"\nOriginal connections: {len(connections)}")
    print(f"After load balancing: {len(balanced)}")
    
    # Add reciprocal connections
    with_reciprocals = router.add_reciprocal_connections(balanced)
    print(f"With reciprocals: {len(with_reciprocals)}")
    
    # Print statistics
    print("\nRouting statistics:")
    stats = router.get_routing_stats()
    for key, value in stats.items():
        if key != 'routing_history':
            print(f"{key}: {value}")
    
    # Test parsimonious router
    print("\n" + "="*50)
    print("Testing Parsimonious Router")
    
    parsimonious = ParsimonousRouter()
    simple_connections = parsimonious.parsimonious_growth(extrema, layer_sizes)
    
    print("Parsimonious connections:")
    for conn in simple_connections:
        print(f"Layer {conn[0]}, Neuron {conn[1]} → Layer {conn[2]}, Neuron {conn[3]}")
