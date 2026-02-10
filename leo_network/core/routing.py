"""
Routing Module

This module provides routing algorithms for LEO satellite networks,
including shortest path routing and multi-path routing strategies.
"""

import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .topology import LEOConstellation, Link
from .traffic import Packet


@dataclass
class RouteEntry:
    """Routing table entry"""
    destination: str
    next_hop: str
    path: List[str]
    metric: float  # e.g., delay, hop count
    valid: bool = True


class Router(ABC):
    """
    Abstract base class for routing algorithms
    
    All routing algorithms should inherit from this class and
    implement the compute_path method.
    """
    
    def __init__(self, constellation: LEOConstellation):
        """
        Initialize router
        
        Args:
            constellation: LEO constellation topology
        """
        self.constellation = constellation
        self.routing_table: Dict[str, Dict[str, RouteEntry]] = {}
        self.name = "BaseRouter"
    
    @abstractmethod
    def compute_path(
        self,
        source: str,
        destination: str
    ) -> Optional[List[str]]:
        """
        Compute path from source to destination
        
        Args:
            source: Source node ID
            destination: Destination node ID
            
        Returns:
            List of node IDs representing the path, or None if no path exists
        """
        pass
    
    def route_packet(self, packet: Packet) -> bool:
        """
        Route a packet by computing and assigning its path
        
        Args:
            packet: Packet to route
            
        Returns:
            True if path was found, False otherwise
        """
        path = self.compute_path(packet.source, packet.destination)
        if path:
            packet.path = path
            packet.current_hop = 0
            return True
        return False
    
    def build_routing_table(self):
        """Build complete routing table for all node pairs"""
        nodes = list(self.constellation.graph.nodes())
        
        for src in nodes:
            self.routing_table[src] = {}
            for dst in nodes:
                if src != dst:
                    path = self.compute_path(src, dst)
                    if path and len(path) > 1:
                        metric = self._calculate_path_metric(path)
                        self.routing_table[src][dst] = RouteEntry(
                            destination=dst,
                            next_hop=path[1],
                            path=path,
                            metric=metric
                        )
    
    def _calculate_path_metric(self, path: List[str]) -> float:
        """Calculate path metric (total delay by default)"""
        total_delay = 0.0
        for i in range(len(path) - 1):
            link = self.constellation.get_link(path[i], path[i+1])
            if link:
                total_delay += link.propagation_delay
        return total_delay
    
    def get_next_hop(self, current: str, destination: str) -> Optional[str]:
        """Get next hop for a destination from current node"""
        if current in self.routing_table:
            if destination in self.routing_table[current]:
                return self.routing_table[current][destination].next_hop
        return None
    
    def get_path_delay(self, path: List[str]) -> float:
        """Calculate total propagation delay for a path"""
        return self._calculate_path_metric(path)


class ShortestPathRouter(Router):
    """
    Shortest Path Router using Dijkstra's algorithm
    
    Computes shortest paths based on link propagation delay.
    """
    
    def __init__(self, constellation: LEOConstellation, weight: str = "weight"):
        """
        Initialize shortest path router
        
        Args:
            constellation: LEO constellation topology
            weight: Edge attribute to use as weight (default: propagation delay)
        """
        super().__init__(constellation)
        self.weight = weight
        self.name = "ShortestPathRouter"
    
    def compute_path(
        self,
        source: str,
        destination: str
    ) -> Optional[List[str]]:
        """Compute shortest path using Dijkstra's algorithm"""
        try:
            path = nx.shortest_path(
                self.constellation.graph,
                source=source,
                target=destination,
                weight=self.weight
            )
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


class KShortestPathsRouter(Router):
    """
    K-Shortest Paths Router
    
    Computes K shortest paths and can select among them
    based on various strategies.
    """
    
    def __init__(
        self,
        constellation: LEOConstellation,
        k: int = 3,
        weight: str = "weight"
    ):
        """
        Initialize K-shortest paths router
        
        Args:
            constellation: LEO constellation topology
            k: Number of shortest paths to compute
            weight: Edge attribute to use as weight
        """
        super().__init__(constellation)
        self.k = k
        self.weight = weight
        self.name = f"K{k}ShortestPathsRouter"
    
    def compute_k_paths(
        self,
        source: str,
        destination: str
    ) -> List[List[str]]:
        """
        Compute K shortest paths
        
        Args:
            source: Source node ID
            destination: Destination node ID
            
        Returns:
            List of K shortest paths (each path is a list of node IDs)
        """
        try:
            # Use islice to avoid computing all paths (generator)
            from itertools import islice
            path_generator = nx.shortest_simple_paths(
                self.constellation.graph,
                source=source,
                target=destination,
                weight=self.weight
            )
            # Only take first k paths from generator
            paths = list(islice(path_generator, self.k))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def compute_path(
        self,
        source: str,
        destination: str
    ) -> Optional[List[str]]:
        """Compute single shortest path (first of K paths)"""
        paths = self.compute_k_paths(source, destination)
        return paths[0] if paths else None


class ECMPRouter(Router):
    """
    Equal-Cost Multi-Path (ECMP) Router
    
    Distributes traffic across multiple equal-cost paths
    using hash-based selection.
    """
    
    def __init__(
        self,
        constellation: LEOConstellation,
        max_paths: int = 4,
        weight: str = "weight"
    ):
        """
        Initialize ECMP router
        
        Args:
            constellation: LEO constellation topology
            max_paths: Maximum number of equal-cost paths to use
            weight: Edge attribute to use as weight
        """
        super().__init__(constellation)
        self.max_paths = max_paths
        self.weight = weight
        self.name = f"ECMP{max_paths}Router"
        self.ecmp_table: Dict[Tuple[str, str], List[List[str]]] = {}
        self.rng = np.random.default_rng()
    
    def compute_ecmp_paths(
        self,
        source: str,
        destination: str
    ) -> List[List[str]]:
        """
        Compute equal-cost paths
        
        Args:
            source: Source node ID
            destination: Destination node ID
            
        Returns:
            List of equal-cost paths
        """
        key = (source, destination)
        if key in self.ecmp_table:
            return self.ecmp_table[key]
        
        try:
            # Get all shortest paths
            all_paths = list(nx.all_shortest_paths(
                self.constellation.graph,
                source=source,
                target=destination,
                weight=self.weight
            ))
            paths = all_paths[:self.max_paths]
            self.ecmp_table[key] = paths
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def compute_path(
        self,
        source: str,
        destination: str
    ) -> Optional[List[str]]:
        """Select one path randomly from equal-cost paths"""
        paths = self.compute_ecmp_paths(source, destination)
        if not paths:
            return None
        # Random selection (in practice, would use flow hash)
        idx = self.rng.integers(0, len(paths))
        return paths[idx]
    
    def compute_path_with_hash(
        self,
        source: str,
        destination: str,
        flow_hash: int
    ) -> Optional[List[str]]:
        """
        Select path based on flow hash for consistent routing
        
        Args:
            source: Source node ID
            destination: Destination node ID
            flow_hash: Hash value for flow identification
            
        Returns:
            Selected path
        """
        paths = self.compute_ecmp_paths(source, destination)
        if not paths:
            return None
        idx = flow_hash % len(paths)
        return paths[idx]


class LoadAwareRouter(Router):
    """
    Load-Aware Router
    
    Selects paths considering current link utilization
    to avoid congested links.
    """
    
    def __init__(
        self,
        constellation: LEOConstellation,
        k: int = 3,
        load_weight: float = 0.5
    ):
        """
        Initialize load-aware router
        
        Args:
            constellation: LEO constellation topology
            k: Number of candidate paths to consider
            load_weight: Weight for load in path selection (0-1)
        """
        super().__init__(constellation)
        self.k = k
        self.load_weight = load_weight
        self.name = f"LoadAwareK{k}Router"
        self.k_router = KShortestPathsRouter(constellation, k=k)
    
    def compute_path(
        self,
        source: str,
        destination: str
    ) -> Optional[List[str]]:
        """Compute path considering link loads"""
        paths = self.k_router.compute_k_paths(source, destination)
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        # Score each path based on delay and load
        best_path = None
        best_score = float('inf')
        
        for path in paths:
            delay = self._calculate_path_metric(path)
            load = self._calculate_path_load(path)
            
            # Combined score (lower is better)
            score = (1 - self.load_weight) * delay + self.load_weight * load * 100
            
            if score < best_score:
                best_score = score
                best_path = path
        
        return best_path
    
    def _calculate_path_load(self, path: List[str]) -> float:
        """Calculate average link utilization along path"""
        if len(path) < 2:
            return 0.0
        
        total_load = 0.0
        num_links = 0
        
        for i in range(len(path) - 1):
            link = self.constellation.get_link(path[i], path[i+1])
            if link:
                total_load += link.get_utilization()
                num_links += 1
        
        return total_load / num_links if num_links > 0 else 0.0


class RandomizedRouter(Router):
    """
    Randomized Multi-Path Router (for DDoS defense)
    
    Probabilistically selects from multiple candidate paths
    to increase unpredictability and distribute load.
    """
    
    def __init__(
        self,
        constellation: LEOConstellation,
        k: int = 5,
        temperature: float = 1.0
    ):
        """
        Initialize randomized router
        
        Args:
            constellation: LEO constellation topology
            k: Number of candidate paths
            temperature: Softmax temperature (higher = more random)
        """
        super().__init__(constellation)
        self.k = k
        self.temperature = temperature
        self.name = f"RandomizedK{k}Router"
        self.k_router = KShortestPathsRouter(constellation, k=k)
        self.rng = np.random.default_rng()
    
    def compute_path(
        self,
        source: str,
        destination: str
    ) -> Optional[List[str]]:
        """Compute path using probabilistic selection"""
        paths = self.k_router.compute_k_paths(source, destination)
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        # Calculate scores for each path
        scores = []
        for path in paths:
            delay = self._calculate_path_metric(path)
            load = self._calculate_path_load(path)
            # Lower score is better
            score = delay + load * 50
            scores.append(score)
        
        # Convert to probabilities using softmax
        probs = self._softmax_inverse(scores)
        
        # Probabilistic selection
        idx = self.rng.choice(len(paths), p=probs)
        return paths[idx]
    
    def _calculate_path_load(self, path: List[str]) -> float:
        """Calculate average link utilization along path"""
        if len(path) < 2:
            return 0.0
        
        total_load = 0.0
        num_links = 0
        
        for i in range(len(path) - 1):
            link = self.constellation.get_link(path[i], path[i+1])
            if link:
                total_load += link.get_utilization()
                num_links += 1
        
        return total_load / num_links if num_links > 0 else 0.0
    
    def _softmax_inverse(self, scores: List[float]) -> np.ndarray:
        """
        Convert scores to probabilities (lower score = higher probability)
        
        Args:
            scores: List of scores (lower is better)
            
        Returns:
            Probability distribution over paths
        """
        scores = np.array(scores)
        # Negate scores so lower becomes higher
        neg_scores = -scores / self.temperature
        # Subtract max for numerical stability
        neg_scores = neg_scores - np.max(neg_scores)
        exp_scores = np.exp(neg_scores)
        probs = exp_scores / np.sum(exp_scores)
        return probs
    
    def set_temperature(self, temperature: float):
        """
        Set softmax temperature
        
        Higher temperature = more random selection
        Lower temperature = more deterministic (prefer best path)
        """
        self.temperature = max(0.01, temperature)


def create_router(
    router_type: str,
    constellation: LEOConstellation,
    **kwargs
) -> Router:
    """
    Factory function to create router by type
    
    Args:
        router_type: Type of router ("shortest", "ksp", "ecmp", "load_aware", "random")
        constellation: LEO constellation topology
        **kwargs: Additional arguments for specific router types
        
    Returns:
        Router instance
    """
    router_map = {
        "shortest": ShortestPathRouter,
        "ksp": KShortestPathsRouter,
        "ecmp": ECMPRouter,
        "load_aware": LoadAwareRouter,
        "random": RandomizedRouter
    }
    
    if router_type not in router_map:
        raise ValueError(f"Unknown router type: {router_type}. "
                        f"Available: {list(router_map.keys())}")
    
    return router_map[router_type](constellation, **kwargs)
