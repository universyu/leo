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
        Route a packet by computing and assigning its path.
        
        First checks the pre-computed routing table (if built),
        then falls back to on-demand path computation.
        
        Args:
            packet: Packet to route
            
        Returns:
            True if path was found, False otherwise
        """
        src, dst = packet.source, packet.destination
        
        # Check pre-computed routing table first
        if src in self.routing_table and dst in self.routing_table[src]:
            entry = self.routing_table[src][dst]
            if entry.valid:
                packet.path = list(entry.path)  # Copy to avoid mutation
                packet.current_hop = 0
                return True
        
        # Fallback to on-demand computation
        path = self.compute_path(src, dst)
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
    
    def precompute_ground_station_routes(self):
        """
        Pre-compute routing table for all ground station pairs.
        
        Since routing algorithms are static and deterministic, and
        ground stations are fixed, we can compute all routes once
        at initialization to avoid repeated computation at runtime.
        
        This computes routes for all (src_gs, dst_gs) pairs where
        both src and dst are ground stations.
        """
        gs_nodes = [
            node for node in self.constellation.graph.nodes()
            if node.startswith("GS_")
        ]
        
        if len(gs_nodes) < 2:
            return
        
        total_pairs = len(gs_nodes) * (len(gs_nodes) - 1)
        computed = 0
        failed = 0
        
        for src in gs_nodes:
            if src not in self.routing_table:
                self.routing_table[src] = {}
            for dst in gs_nodes:
                if src == dst:
                    continue
                path = self.compute_path(src, dst)
                if path and len(path) > 1:
                    metric = self._calculate_path_metric(path)
                    self.routing_table[src][dst] = RouteEntry(
                        destination=dst,
                        next_hop=path[1],
                        path=path,
                        metric=metric
                    )
                    computed += 1
                else:
                    failed += 1
        
        print(f"  Pre-computed {computed}/{total_pairs} ground station routes "
              f"({failed} unreachable pairs)")
    
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
        self.path_cache: Dict[Tuple[str, str], List[List[str]]] = {}
    
    def compute_k_paths(
        self,
        source: str,
        destination: str
    ) -> List[List[str]]:
        """
        Compute K shortest paths (with caching)
        
        Args:
            source: Source node ID
            destination: Destination node ID
            
        Returns:
            List of K shortest paths (each path is a list of node IDs)
        """
        key = (source, destination)
        if key in self.path_cache:
            return self.path_cache[key]
        
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
            self.path_cache[key] = paths
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.path_cache[key] = []
            return []
    
    def compute_path(
        self,
        source: str,
        destination: str
    ) -> Optional[List[str]]:
        """Compute single shortest path (first of K paths)"""
        paths = self.compute_k_paths(source, destination)
        return paths[0] if paths else None


class KDSRouter(Router):
    """
    K-Disjoint Shortest Paths Router (KDS)
    
    Computes K node-disjoint or link-disjoint shortest paths to provide
    path diversity and resilience against single point failures or attacks.
    """
    
    def __init__(
        self,
        constellation: LEOConstellation,
        k: int = 3,
        disjoint_type: str = "link"  # "link" or "node"
    ):
        """
        Initialize K-Disjoint Shortest paths router
        
        Args:
            constellation: LEO constellation topology
            k: Number of disjoint paths to compute
            disjoint_type: Type of disjointness ("link" or "node")
        """
        super().__init__(constellation)
        self.k = k
        self.disjoint_type = disjoint_type
        self.name = f"KDS{k}Router"
        self.rng = np.random.default_rng()
        self.path_cache: Dict[Tuple[str, str], List[List[str]]] = {}
    
    def compute_k_disjoint_paths(
        self,
        source: str,
        destination: str
    ) -> List[List[str]]:
        """
        Compute K disjoint shortest paths using iterative removal
        
        Args:
            source: Source node ID
            destination: Destination node ID
            
        Returns:
            List of K disjoint paths
        """
        key = (source, destination)
        if key in self.path_cache:
            return self.path_cache[key]
        
        paths = []
        temp_graph = self.constellation.graph.copy()
        
        for _ in range(self.k):
            try:
                # Find shortest path in current graph
                path = nx.shortest_path(
                    temp_graph,
                    source=source,
                    target=destination,
                    weight="weight"
                )
                paths.append(path)
                
                # Remove edges/nodes to ensure disjointness
                if self.disjoint_type == "link":
                    # Remove all edges in the path
                    for i in range(len(path) - 1):
                        if temp_graph.has_edge(path[i], path[i+1]):
                            temp_graph.remove_edge(path[i], path[i+1])
                else:  # node disjoint
                    # Remove intermediate nodes (keep source and destination)
                    for node in path[1:-1]:
                        if temp_graph.has_node(node):
                            temp_graph.remove_node(node)
                            
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                break
        
        self.path_cache[key] = paths
        return paths
    
    def compute_path(
        self,
        source: str,
        destination: str
    ) -> Optional[List[str]]:
        """Select one path randomly from K disjoint paths"""
        paths = self.compute_k_disjoint_paths(source, destination)
        if not paths:
            return None
        idx = self.rng.integers(0, len(paths))
        return paths[idx]
    
    def compute_path_by_index(
        self,
        source: str,
        destination: str,
        index: int = 0
    ) -> Optional[List[str]]:
        """Select specific path by index"""
        paths = self.compute_k_disjoint_paths(source, destination)
        if not paths or index >= len(paths):
            return paths[0] if paths else None
        return paths[index]


class KDGRouter(Router):
    """
    K-Disjoint Geodiverse Paths Router (KDG)
    
    Computes K geographically diverse disjoint paths to maximize
    spatial separation between paths, providing resilience against
    localized attacks or failures.
    """
    
    def __init__(
        self,
        constellation: LEOConstellation,
        k: int = 3,
        diversity_weight: float = 0.5
    ):
        """
        Initialize K-Disjoint Geodiverse paths router
        
        Args:
            constellation: LEO constellation topology
            k: Number of geodiverse paths to compute
            diversity_weight: Weight for geographic diversity (0-1)
        """
        super().__init__(constellation)
        self.k = k
        self.diversity_weight = diversity_weight
        self.name = f"KDG{k}Router"
        self.rng = np.random.default_rng()
        self.path_cache: Dict[Tuple[str, str], List[List[str]]] = {}
    
    def _get_node_position(self, node_id: str) -> Tuple[int, int]:
        """Extract plane and satellite index from node ID"""
        # Node ID format: SAT_<plane>_<index> or GS_<name>
        if node_id.startswith("SAT_"):
            parts = node_id.split("_")
            return int(parts[1]), int(parts[2])
        return 0, 0  # Ground station at origin for simplicity
    
    def _calculate_path_diversity(
        self,
        new_path: List[str],
        existing_paths: List[List[str]]
    ) -> float:
        """
        Calculate geographic diversity score for a path
        
        Higher score means more diverse (better)
        """
        if not existing_paths:
            return 1.0
        
        total_diversity = 0.0
        new_nodes = set(new_path[1:-1])  # Exclude source/dest
        
        for existing_path in existing_paths:
            existing_nodes = set(existing_path[1:-1])
            
            # Calculate node overlap
            overlap = len(new_nodes & existing_nodes)
            max_nodes = max(len(new_nodes), len(existing_nodes), 1)
            node_diversity = 1.0 - (overlap / max_nodes)
            
            # Calculate plane diversity
            new_planes = set(self._get_node_position(n)[0] for n in new_nodes if n.startswith("SAT_"))
            exist_planes = set(self._get_node_position(n)[0] for n in existing_nodes if n.startswith("SAT_"))
            plane_overlap = len(new_planes & exist_planes)
            max_planes = max(len(new_planes), len(exist_planes), 1)
            plane_diversity = 1.0 - (plane_overlap / max_planes)
            
            total_diversity += 0.5 * node_diversity + 0.5 * plane_diversity
        
        return total_diversity / len(existing_paths)
    
    def compute_k_geodiverse_paths(
        self,
        source: str,
        destination: str
    ) -> List[List[str]]:
        """
        Compute K geographically diverse disjoint paths
        
        Uses a greedy algorithm that iteratively selects paths
        maximizing geographic diversity while maintaining disjointness.
        """
        key = (source, destination)
        if key in self.path_cache:
            return self.path_cache[key]
        
        paths = []
        temp_graph = self.constellation.graph.copy()
        
        # Get multiple candidate paths first
        try:
            from itertools import islice
            all_candidates = list(islice(
                nx.shortest_simple_paths(temp_graph, source, destination, weight="weight"),
                self.k * 5  # Get more candidates for diversity selection
            ))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        
        if not all_candidates:
            return []
        
        # Greedily select paths maximizing diversity
        used_links = set()
        
        for candidate in all_candidates:
            if len(paths) >= self.k:
                break
            
            # Check if path is link-disjoint from existing paths
            path_links = set()
            is_disjoint = True
            for i in range(len(candidate) - 1):
                link = (min(candidate[i], candidate[i+1]), max(candidate[i], candidate[i+1]))
                if link in used_links:
                    is_disjoint = False
                    break
                path_links.add(link)
            
            if not is_disjoint:
                continue
            
            # Calculate combined score (delay + diversity)
            delay = self._calculate_path_metric(candidate)
            diversity = self._calculate_path_diversity(candidate, paths)
            
            # Accept path if it has good diversity or is first path
            if not paths or diversity > 0.3:
                paths.append(candidate)
                used_links.update(path_links)
        
        self.path_cache[key] = paths
        return paths
    
    def compute_path(
        self,
        source: str,
        destination: str
    ) -> Optional[List[str]]:
        """Select one path randomly from K geodiverse paths"""
        paths = self.compute_k_geodiverse_paths(source, destination)
        if not paths:
            return None
        idx = self.rng.integers(0, len(paths))
        return paths[idx]


class KLORouter(Router):
    """
    K-Link-disjoint with Load Optimization Router (KLO)
    
    Computes K link-disjoint paths and dynamically selects
    the best path based on current network load conditions.
    Combines path diversity with intelligent load balancing.
    """
    
    def __init__(
        self,
        constellation: LEOConstellation,
        k: int = 3,
        load_threshold: float = 0.7,
        recompute_interval: int = 100
    ):
        """
        Initialize K-Link-disjoint Load Optimization router
        
        Args:
            constellation: LEO constellation topology
            k: Number of disjoint paths to maintain
            load_threshold: Utilization threshold for path switching
            recompute_interval: Packets between path cache refresh
        """
        super().__init__(constellation)
        self.k = k
        self.load_threshold = load_threshold
        self.recompute_interval = recompute_interval
        self.name = f"KLO{k}Router"
        self.kds_router = KDSRouter(constellation, k=k, disjoint_type="link")
        self.packet_count = 0
    
    def _calculate_path_load(self, path: List[str]) -> float:
        """Calculate maximum link utilization along path"""
        if len(path) < 2:
            return 0.0
        
        max_load = 0.0
        for i in range(len(path) - 1):
            link = self.constellation.get_link(path[i], path[i+1])
            if link:
                max_load = max(max_load, link.get_utilization())
        
        return max_load
    
    def _calculate_path_avg_load(self, path: List[str]) -> float:
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
    
    def compute_path(
        self,
        source: str,
        destination: str
    ) -> Optional[List[str]]:
        """
        Compute optimal path considering both disjointness and load
        
        Selects the path with lowest load among K disjoint paths.
        """
        # Periodically clear cache to adapt to load changes
        self.packet_count += 1
        if self.packet_count >= self.recompute_interval:
            self.kds_router.path_cache.clear()
            self.packet_count = 0
        
        paths = self.kds_router.compute_k_disjoint_paths(source, destination)
        if not paths:
            return None
        
        if len(paths) == 1:
            return paths[0]
        
        # Select path with minimum load
        best_path = None
        best_score = float('inf')
        
        for path in paths:
            max_load = self._calculate_path_load(path)
            avg_load = self._calculate_path_avg_load(path)
            delay = self._calculate_path_metric(path)
            
            # Combined score: prioritize avoiding congested links
            # but also consider average load and delay
            if max_load > self.load_threshold:
                # Penalize paths with congested links heavily
                score = max_load * 1000 + avg_load * 100 + delay
            else:
                score = avg_load * 100 + delay
            
            if score < best_score:
                best_score = score
                best_path = path
        
        return best_path
    
    def get_all_disjoint_paths(
        self,
        source: str,
        destination: str
    ) -> List[List[str]]:
        """Get all K disjoint paths for analysis"""
        return self.kds_router.compute_k_disjoint_paths(source, destination)


class KRandRouter(Router):
    """
    K-RAND Router (Paper's Algorithm 1)
    
    Randomly selects one of four routing algorithms (k-SP, k-DG, k-DS, k-LO)
    with weighted probability distribution for each packet routing decision.
    
    This increases the uncertainty for an attacker who knows the network
    topology, because the attacker cannot predict which algorithm will be
    used for any given packet, forcing them to cover all possible paths
    across all algorithms.
    
    The weights (a1, a2, a3) determine the selection probabilities:
      - P(k-SP) = a1
      - P(k-DG) = a2 - a1
      - P(k-DS) = a3 - a2  
      - P(k-LO) = 1 - a3
    """
    
    def __init__(
        self,
        constellation: LEOConstellation,
        k: int = 3,
        weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize K-RAND router
        
        Args:
            constellation: LEO constellation topology
            k: Number of paths for each sub-algorithm
            weights: Dict of algorithm weights, e.g. {"ksp": 0.25, "kdg": 0.25, "kds": 0.25, "klo": 0.25}
                     If None, uses equal weights (0.25 each)
            seed: Random seed for reproducibility
        """
        super().__init__(constellation)
        self.k = k
        self.name = f"KRand{k}Router"
        self.rng = np.random.default_rng(seed)
        
        # Default: equal weights for all 4 algorithms
        if weights is None:
            self.weights = {"ksp": 0.25, "kdg": 0.25, "kds": 0.25, "klo": 0.25}
        else:
            self.weights = weights
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Create sub-routers
        self.sub_routers: Dict[str, Router] = {
            "ksp": KShortestPathsRouter(constellation, k=k),
            "kdg": KDGRouter(constellation, k=k),
            "kds": KDSRouter(constellation, k=k),
            "klo": KLORouter(constellation, k=k),
        }
        
        # Pre-build algorithm selection arrays for efficiency
        self.algo_names = list(self.weights.keys())
        self.algo_probs = np.array([self.weights[name] for name in self.algo_names])
    
    def compute_path(
        self,
        source: str,
        destination: str
    ) -> Optional[List[str]]:
        """
        Compute path by randomly selecting one of the 4 algorithms.
        
        For each routing decision, generate random variable A in [0,1]
        and select algorithm based on weighted distribution (Algorithm 1).
        """
        # Random algorithm selection (Paper's Algorithm 1)
        selected_algo = self.rng.choice(self.algo_names, p=self.algo_probs)
        router = self.sub_routers[selected_algo]
        return router.compute_path(source, destination)
    
    def compute_path_with_algo(
        self,
        source: str,
        destination: str,
        algo_name: str
    ) -> Optional[List[str]]:
        """
        Compute path using a specific sub-algorithm (for analysis).
        """
        if algo_name in self.sub_routers:
            return self.sub_routers[algo_name].compute_path(source, destination)
        return None
    
    def get_all_possible_paths(
        self,
        source: str,
        destination: str
    ) -> Dict[str, List[List[str]]]:
        """
        Get all possible paths from all sub-algorithms.
        
        This represents the full path space an attacker must cover.
        
        Returns:
            Dict mapping algorithm name to list of k paths
        """
        all_paths = {}
        
        ksp = self.sub_routers["ksp"]
        if isinstance(ksp, KShortestPathsRouter):
            all_paths["ksp"] = ksp.compute_k_paths(source, destination)
        
        kds = self.sub_routers["kds"]
        if isinstance(kds, KDSRouter):
            all_paths["kds"] = kds.compute_k_disjoint_paths(source, destination)
        
        kdg = self.sub_routers["kdg"]
        if isinstance(kdg, KDGRouter):
            all_paths["kdg"] = kdg.compute_k_geodiverse_paths(source, destination)
        
        klo = self.sub_routers["klo"]
        if isinstance(klo, KLORouter):
            all_paths["klo"] = klo.get_all_disjoint_paths(source, destination)
        
        return all_paths
    
    def precompute_ground_station_routes(self):
        """
        Pre-compute routes for all sub-routers.
        
        Note: For k-RAND, we pre-compute routes for ALL sub-routers,
        but the routing table is NOT used at runtime (compute_path
        randomly selects algorithm each time). The pre-computed routes
        are mainly for analysis purposes.
        """
        print(f"  Pre-computing routes for all 4 sub-algorithms...")
        for algo_name, router in self.sub_routers.items():
            print(f"    Computing {algo_name.upper()} routes...")
            router.precompute_ground_station_routes()
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Update algorithm selection weights.
        
        Args:
            weights: New weights dict
        """
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        self.algo_names = list(self.weights.keys())
        self.algo_probs = np.array([self.weights[name] for name in self.algo_names])


def create_router(
    router_type: str,
    constellation: LEOConstellation,
    **kwargs
) -> Router:
    """
    Factory function to create router by type
    
    Args:
    router_type: Type of router ("ksp", "kds", "kdg", "klo", "krand")
        constellation: LEO constellation topology
        **kwargs: Additional arguments for specific router types
        
    Returns:
        Router instance
    """
    router_map = {
        "ksp": KShortestPathsRouter,
        "kds": KDSRouter,
        "kdg": KDGRouter,
        "klo": KLORouter,
        "krand": KRandRouter
    }
    
    if router_type not in router_map:
        raise ValueError(f"Unknown router type: {router_type}. "
                        f"Available: {list(router_map.keys())}")
    
    return router_map[router_type](constellation, **kwargs)
