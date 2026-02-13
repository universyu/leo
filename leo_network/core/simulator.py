"""
Network Simulator Module

This module provides the main simulation engine that orchestrates
the LEO satellite network simulation including topology, traffic,
routing, and statistics collection.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from tqdm import tqdm

from .topology import LEOConstellation
from .traffic import TrafficGenerator, Packet, PacketType
from .routing import Router, KShortestPathsRouter, create_router
from .statistics import StatisticsCollector, AttackCostCalculator


class Simulator:
    """
    Main simulation engine for LEO satellite network
    
    Coordinates topology, traffic generation, routing, and
    packet forwarding in a discrete-event simulation.
    """
    
    def __init__(
        self,
        constellation: LEOConstellation,
        router: Optional[Router] = None,
        time_step: float = 0.001,  # 1ms default time step
        seed: Optional[int] = None
    ):
        """
        Initialize simulator
        
        Args:
            constellation: LEO constellation topology
            router: Routing algorithm (default: shortest path)
            time_step: Simulation time step in seconds
            seed: Random seed for reproducibility
        """
        self.constellation = constellation
        self.time_step = time_step
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Set time_step for all links (for correct capacity calculation)
        for link in self.constellation.links.values():
            link.time_step = time_step
        
        # Initialize router
        if router is None:
            self.router = KShortestPathsRouter(constellation)
        else:
            self.router = router
        
        # Pre-compute routing table for all ground station pairs
        # Since routing algorithms are static and ground stations are fixed,
        # we can compute all routes once to avoid repeated computation
        if self.constellation.ground_stations:
            print(f"Pre-computing routing table for {self.router.name}...")
            self.router.precompute_ground_station_routes()
        
        # Initialize traffic generator
        self.traffic_generator = TrafficGenerator(seed=seed)
        
        # Initialize statistics collector
        self.stats = StatisticsCollector()
        
        # Target ISL for per-link throughput tracking
        self.target_isl: Optional[tuple] = None
        
        # Initialize attack cost calculator
        self.cost_calculator = AttackCostCalculator()
        
        # Simulation state
        self.current_time: float = 0.0
        self.packets_in_transit: List[Packet] = []
        self.event_callbacks: List[Callable] = []
        
        # Configuration
        self.verbose = False
        self.snapshot_interval = 0.1  # seconds
        self.last_snapshot_time = 0.0
    
    def add_normal_traffic(
        self,
        source: str,
        destination: str,
        rate: float = 100.0,
        packet_size: int = 1000,
        start_time: float = 0.0,
        duration: float = -1.0
    ):
        """
        Add normal traffic flow
        
        Args:
            source: Source node ID
            destination: Destination node ID
            rate: Traffic rate in packets per second
            packet_size: Packet size in bytes
            start_time: Flow start time
            duration: Flow duration (-1 for infinite)
        """
        flow_id = f"normal_{source}_{destination}_{len(self.traffic_generator.flows)}"
        self.traffic_generator.create_normal_flow(
            flow_id=flow_id,
            source=source,
            destination=destination,
            rate=rate,
            packet_size=packet_size,
            start_time=start_time,
            duration=duration
        )
    
    def add_attack_traffic(
        self,
        source: str,
        destination: str,
        rate: float = 1000.0,
        packet_size: int = 1000,
        start_time: float = 0.0,
        duration: float = -1.0
    ):
        """
        Add attack traffic flow
        
        Args:
            source: Source node ID
            destination: Destination node ID
            rate: Attack rate in packets per second
            packet_size: Packet size in bytes
            start_time: Attack start time
            duration: Attack duration (-1 for infinite)
        """
        flow_id = f"attack_{source}_{destination}_{len(self.traffic_generator.flows)}"
        self.traffic_generator.create_attack_flow(
            flow_id=flow_id,
            source=source,
            destination=destination,
            rate=rate,
            packet_size=packet_size,
            start_time=start_time,
            duration=duration
        )
    
    def add_random_normal_flows(
        self,
        num_flows: int,
        rate_range: tuple = (50, 200),
        packet_size: int = 1000
    ):
        """
        Add multiple random normal traffic flows between ground stations.
        
        Following the paper's traffic model, all normal traffic originates
        from ground stations and terminates at ground stations, traversing
        the satellite network via uplink -> ISL -> downlink paths.
        
        If fewer than 2 ground stations are available, falls back to
        satellite-to-satellite flows with a warning.
        
        Args:
            num_flows: Number of flows to create
            rate_range: (min, max) rate in packets per second
            packet_size: Packet size in bytes
        """
        gs_nodes = list(self.constellation.ground_stations.keys())
        
        if len(gs_nodes) < 2:
            # Fallback: not enough ground stations, warn and use satellites
            import warnings
            warnings.warn(
                f"Only {len(gs_nodes)} ground station(s) available. "
                f"Normal traffic should originate from ground stations per the paper's model. "
                f"Call constellation.add_global_ground_stations() first. "
                f"Falling back to satellite nodes.",
                UserWarning
            )
            nodes = list(self.constellation.satellites.keys())
            if gs_nodes:
                nodes.extend(gs_nodes)
            for i in range(num_flows):
                src, dst = self.rng.choice(nodes, size=2, replace=False)
                rate = self.rng.uniform(rate_range[0], rate_range[1])
                self.add_normal_traffic(
                    source=src,
                    destination=dst,
                    rate=rate,
                    packet_size=packet_size
                )
            return
        
        for i in range(num_flows):
            # Random source and destination ground stations
            src, dst = self.rng.choice(gs_nodes, size=2, replace=False)
            rate = self.rng.uniform(rate_range[0], rate_range[1])
            
            self.add_normal_traffic(
                source=src,
                destination=dst,
                rate=rate,
                packet_size=packet_size
            )
    
    def add_distributed_attack(
        self,
        num_attackers: int,
        target: str,
        total_rate: float = 5000.0,
        start_time: float = 0.0,
        duration: float = -1.0
    ):
        """
        Add distributed DDoS attack from multiple ground station sources.
        
        In a realistic DDoS model, attackers deploy bots at ground stations.
        The target should also be a ground station. If no ground stations are
        available, falls back to satellite nodes with a warning.
        
        Args:
            num_attackers: Number of attack sources (ground stations)
            target: Target node ID (should be a ground station)
            total_rate: Total attack rate (divided among attackers)
            start_time: Attack start time
            duration: Attack duration
        """
        gs_nodes = list(self.constellation.ground_stations.keys())
        
        if len(gs_nodes) < 2:
            # Fallback: no ground stations, use satellites
            import warnings
            warnings.warn(
                "No ground stations available for DDoS attack sources. "
                "Call constellation.add_global_ground_stations() first. "
                "Falling back to satellite nodes.",
                UserWarning
            )
            nodes = list(self.constellation.satellites.keys())
            if target in nodes:
                nodes.remove(target)
            if num_attackers > len(nodes):
                num_attackers = len(nodes)
            attackers = self.rng.choice(nodes, size=num_attackers, replace=False)
        else:
            # Select from ground stations
            available = [gs for gs in gs_nodes if gs != target]
            if num_attackers > len(available):
                num_attackers = len(available)
            attackers = self.rng.choice(available, size=num_attackers, replace=False)
        
        rate_per_attacker = total_rate / num_attackers
        
        for attacker in attackers:
            self.add_attack_traffic(
                source=attacker,
                destination=target,
                rate=rate_per_attacker,
                start_time=start_time,
                duration=duration
            )
    
    def run(
        self,
        duration: float,
        progress_bar: bool = True
    ) -> StatisticsCollector:
        """
        Run simulation for specified duration
        
        Args:
            duration: Simulation duration in seconds
            progress_bar: Show progress bar
            
        Returns:
            Statistics collector with results
        """
        num_steps = int(duration / self.time_step)
        
        if self.verbose:
            print(f"Starting simulation: {duration}s, {num_steps} steps")
            print(f"Topology: {self.constellation}")
            print(f"Router: {self.router.name}")
            print(f"Active flows: {len(self.traffic_generator.flows)}")
        
        iterator = range(num_steps)
        if progress_bar:
            iterator = tqdm(iterator, desc="Simulating", unit="step")
        
        for step in iterator:
            self._simulation_step()
        
        # Final snapshot
        # Calculate final window duration (time since last snapshot)
        final_window_duration = self.current_time - self.last_snapshot_time
        if final_window_duration > 0:
            self.stats.take_snapshot(self.current_time, window_duration=final_window_duration)
        
        return self.stats
    
    def _simulation_step(self):
        """Execute one simulation step"""
        # Reset link loads for this time step
        self.constellation.reset_link_loads()
        
        # Generate new packets
        new_packets = self.traffic_generator.generate_packets(
            self.current_time, self.time_step
        )
        
        # Route new packets
        for packet in new_packets:
            if self.router.route_packet(packet):
                self.packets_in_transit.append(packet)
                is_attack = (packet.packet_type == PacketType.ATTACK)
                self.stats.record_packet_sent(packet.size, is_attack=is_attack)
                # Record to cost calculator
                if is_attack:
                    self.cost_calculator.record_attack_packet_sent(packet.size)
                else:
                    self.cost_calculator.record_normal_packet_sent(packet.size)
            else:
                # No path found
                is_attack = (packet.packet_type == PacketType.ATTACK)
                self.stats.record_packet_dropped(packet.size, is_attack=is_attack)
                # Record to cost calculator
                if not is_attack:
                    self.cost_calculator.record_normal_packet_dropped(packet.size)
        
        # Process packets in transit
        self._process_packets()
        
        # Record statistics periodically
        if self.current_time - self.last_snapshot_time >= self.snapshot_interval:
            self._record_network_state()
            # Pass window duration for correct throughput calculation
            self.stats.take_snapshot(self.current_time, window_duration=self.snapshot_interval)
            self.last_snapshot_time = self.current_time
        
        # Advance time
        self.current_time += self.time_step
    
    def _process_packets(self):
        """Process all packets in transit"""
        completed_packets = []
        dropped_packets = []
        
        for packet in self.packets_in_transit:
            # Get current and next hop
            if packet.current_hop >= len(packet.path):
                # Should not happen, but safety check
                completed_packets.append(packet)
                continue
            
            current_node_id = packet.path[packet.current_hop]
            next_hop = packet.get_next_hop()
            
            if next_hop is None:
                # Reached destination
                packet.arrival_time = self.current_time
                completed_packets.append(packet)
                is_attack = (packet.packet_type == PacketType.ATTACK)
                traverses_target = self._packet_traverses_target_isl(packet)
                self.stats.record_packet_delivered(
                    packet.size,
                    delay=packet.get_delay() * 1000,  # Convert to ms
                    hop_count=packet.hops_taken,
                    is_attack=is_attack,
                    traverses_target_isl=traverses_target
                )
                self.traffic_generator.record_packet_delivery(
                    packet, self.current_time
                )
                # Record to cost calculator
                if is_attack:
                    self.cost_calculator.record_attack_packet_delivered(packet.size)
                else:
                    self.cost_calculator.record_normal_packet_delivered(packet.size)
                continue
            
            # Try to forward packet
            link = self.constellation.get_link(current_node_id, next_hop)
            if link is None:
                # No link, drop packet
                dropped_packets.append(packet)
                continue
            
            # Check link capacity
            if link.add_traffic(packet.size):
                # Successfully forwarded
                packet.advance_hop()
            else:
                # Link congested, drop packet
                dropped_packets.append(packet)
                is_attack = (packet.packet_type == PacketType.ATTACK)
                self.stats.record_packet_dropped(packet.size, is_attack=is_attack)
                self.traffic_generator.record_packet_drop(packet)
                # Record to cost calculator
                if not is_attack:
                    self.cost_calculator.record_normal_packet_dropped(packet.size)
        
        # Remove completed and dropped packets
        for packet in completed_packets + dropped_packets:
            if packet in self.packets_in_transit:
                self.packets_in_transit.remove(packet)
    
    def _record_network_state(self):
        """Record current network state for statistics"""
        # Record link utilization
        for link_id, link in self.constellation.links.items():
            self.stats.record_link_utilization(
                link_id, self.current_time, link.get_utilization()
            )
        
        # Record queue occupancy
        for sat_id, sat in self.constellation.satellites.items():
            self.stats.record_queue_occupancy(
                sat_id, self.current_time, sat.get_utilization()
            )
    
    def reset(self):
        """Reset simulation state"""
        self.current_time = 0.0
        self.packets_in_transit = []
        self.last_snapshot_time = 0.0
        self.stats.reset()
        self.cost_calculator.reset()
        self.traffic_generator.reset_all_stats()
        self.constellation.reset_all_stats()
    
    def set_target_isl(self, node_a: str, node_b: str):
        """Set target ISL to track for per-link throughput stats"""
        self.target_isl = (node_a, node_b)
        self.stats.set_target_isl(node_a, node_b)
    
    def _packet_traverses_target_isl(self, packet) -> bool:
        """Check if a packet's path traverses the target ISL"""
        if self.target_isl is None:
            return False
        a, b = self.target_isl
        path = packet.path
        for i in range(len(path) - 1):
            if (path[i] == a and path[i+1] == b) or (path[i] == b and path[i+1] == a):
                return True
        return False
    
    def set_router(self, router: Router):
        """Change routing algorithm"""
        self.router = router
    
    def get_results(self) -> Dict:
        """Get comprehensive simulation results"""
        stats_summary = self.stats.get_summary()
        
        return {
            "simulation_config": {
                "duration": self.current_time,
                "time_step": self.time_step,
                "router": self.router.name,
                "num_satellites": len(self.constellation.satellites),
                "num_links": len(self.constellation.links),
                "num_flows": len(self.traffic_generator.flows)
            },
            "statistics": stats_summary,
            "traffic": self.traffic_generator.get_aggregate_stats(),
            "network": self.constellation.get_network_stats(),
            "attack_cost": self.cost_calculator.get_summary(self.current_time),
            "throughput_percentiles": {
                "p5_pps": stats_summary["throughput"]["p5_pps"],
                "p5_mbps": stats_summary["throughput"]["p5_mbps"],
                "p10_pps": stats_summary["throughput"]["p10_pps"],
                "p10_mbps": stats_summary["throughput"]["p10_mbps"],
                "p50_pps": stats_summary["throughput"]["p50_pps"],
                "p50_mbps": stats_summary["throughput"]["p50_mbps"],
                "avg_pps": stats_summary["throughput"]["avg_pps"],
                "avg_mbps": stats_summary["throughput"]["avg_mbps"],
            },
            "normal_throughput_percentiles": {
                "p5_pps": stats_summary["normal_throughput"]["p5_pps"],
                "p5_mbps": stats_summary["normal_throughput"]["p5_mbps"],
                "p10_pps": stats_summary["normal_throughput"]["p10_pps"],
                "p10_mbps": stats_summary["normal_throughput"]["p10_mbps"],
                "p50_pps": stats_summary["normal_throughput"]["p50_pps"],
                "p50_mbps": stats_summary["normal_throughput"]["p50_mbps"],
                "avg_pps": stats_summary["normal_throughput"]["avg_pps"],
                "avg_mbps": stats_summary["normal_throughput"]["avg_mbps"],
            },
            "target_isl_normal_throughput_percentiles": stats_summary["target_isl_normal_throughput"],
        }
    
    def print_results(self):
        """Print simulation results"""
        print("\n" + "="*70)
        print("SIMULATION RESULTS")
        print("="*70)
        
        results = self.get_results()
        
        print("\n--- Configuration ---")
        for key, value in results["simulation_config"].items():
            print(f"  {key}: {value}")
        
        self.stats.print_summary()
        
        print("\n--- Throughput Percentiles (Key Metric) ---")
        tp = results["throughput_percentiles"]
        print(f"  5th Percentile:  {tp['p5_pps']:.2f} pps ({tp['p5_mbps']:.4f} Mbps) [Worst-case]")
        print(f"  10th Percentile: {tp['p10_pps']:.2f} pps ({tp['p10_mbps']:.4f} Mbps)")
        print(f"  50th Percentile: {tp['p50_pps']:.2f} pps ({tp['p50_mbps']:.4f} Mbps) [Median]")
        print(f"  Average:         {tp['avg_pps']:.2f} pps ({tp['avg_mbps']:.4f} Mbps)")
        
        print("\n--- Traffic Summary ---")
        traffic = results["traffic"]
        print(f"  Normal flows: {traffic['num_normal_flows']}")
        print(f"  Attack flows: {traffic['num_attack_flows']}")
        print(f"  Normal delivery rate: {traffic['normal_delivery_rate']:.4f}")
        print(f"  Attack delivery rate: {traffic['attack_delivery_rate']:.4f}")
        
        # Print attack cost analysis
        self.cost_calculator.print_summary(self.current_time)
    
    def set_baseline_loss_rate(self, loss_rate: float):
        """
        Set baseline loss rate for attack cost calculation
        
        Should be called with the loss rate from a no-attack simulation.
        
        Args:
            loss_rate: Normal packet loss rate without attack
        """
        self.cost_calculator.set_baseline_loss_rate(loss_rate)
    
    def get_attack_cost(self) -> float:
        """
        Get current attack cost
        
        Returns:
            Attack cost value (higher indicates better defense)
        """
        return self.cost_calculator.calculate_attack_cost(self.current_time)
    
    def get_attack_cost_summary(self) -> Dict:
        """
        Get attack cost summary
        
        Returns:
            Dictionary with attack cost metrics
        """
        return self.cost_calculator.get_summary(self.current_time)
    
    def get_5th_percentile_throughput(self) -> Dict:
        """
        Get 5th percentile throughput metrics
        
        This represents the worst-case network performance.
        95% of the time, throughput is higher than this value.
        
        Returns:
            Dictionary with 5th percentile throughput in pps and Mbps
        """
        return {
            "p5_pps": self.stats.get_5th_percentile_throughput_pps(),
            "p5_mbps": self.stats.get_5th_percentile_throughput_mbps()
        }


def run_basic_simulation(
    num_planes: int = 6,
    sats_per_plane: int = 11,
    num_normal_flows: int = 20,
    simulation_duration: float = 1.0,
    router_type: str = "ksp",
    seed: Optional[int] = 42
) -> Dict:
    """
    Convenience function to run a basic simulation
    
    Args:
        num_planes: Number of orbital planes
        sats_per_plane: Satellites per plane
        num_normal_flows: Number of normal traffic flows
        simulation_duration: Duration in seconds
        router_type: Type of router
        seed: Random seed
        
    Returns:
        Simulation results dictionary
    """
    # Create constellation
    constellation = LEOConstellation(
        num_planes=num_planes,
        sats_per_plane=sats_per_plane
    )
    
    # Add globally distributed ground stations (paper-scale)
    constellation.add_global_ground_stations()
    
    # Create router
    router = create_router(router_type, constellation)
    
    # Create simulator
    sim = Simulator(
        constellation=constellation,
        router=router,
        seed=seed
    )
    
    # Add random normal flows
    sim.add_random_normal_flows(num_normal_flows)
    
    # Run simulation
    sim.run(simulation_duration)
    
    return sim.get_results()
