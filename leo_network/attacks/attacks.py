"""
DDoS Attack Module

This module provides classes for simulating various DDoS attack patterns
in LEO satellite networks, including:
- Volumetric flooding attacks (UDP, ICMP floods)
- Reflection/Amplification attacks
- Application-layer slow attacks
- Pulsing attacks
- Coordinated multi-vector attacks

Attack metrics and cost analysis are also provided.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable
from enum import Enum
import random
import math

from ..core.traffic import TrafficGenerator, Flow, Packet, PacketType, TrafficPattern


class AttackType(Enum):
    """Types of DDoS attacks"""
    FLOODING = "flooding"                  # Volumetric flooding
    REFLECTION = "reflection"              # Reflection/Amplification
    SLOWLORIS = "slowloris"               # Application-layer slow attack
    PULSING = "pulsing"                   # On-off pulsing attack
    COORDINATED = "coordinated"           # Multi-vector coordinated
    LINK_TARGETED = "link_targeted"       # Target specific ISL links
    BOTTLENECK = "bottleneck"             # Target bottleneck links


class AttackStrategy(Enum):
    """Attack source selection strategy"""
    RANDOM = "random"                     # Random source selection
    DISTRIBUTED = "distributed"           # Evenly distributed sources
    CLUSTERED = "clustered"               # Clustered sources
    PATH_ALIGNED = "path_aligned"         # Sources aligned on target paths


@dataclass
class AttackConfig:
    """
    Configuration for a DDoS attack
    
    Attributes:
        attack_type: Type of attack
        targets: List of target node IDs
        num_attackers: Number of attack sources
        total_rate: Total attack rate (packets/s)
        packet_size: Attack packet size (bytes)
        start_time: Attack start time (s)
        duration: Attack duration (s), -1 for infinite
        strategy: Attack source selection strategy
        amplification_factor: For reflection attacks
        pulse_on_time: On duration for pulsing attacks (s)
        pulse_off_time: Off duration for pulsing attacks (s)
        ramp_up_time: Gradual ramp-up time (s)
    """
    attack_type: AttackType = AttackType.FLOODING
    targets: List[str] = field(default_factory=list)
    num_attackers: int = 10
    total_rate: float = 5000.0  # packets/s
    packet_size: int = 1000     # bytes
    start_time: float = 0.0
    duration: float = -1.0
    strategy: AttackStrategy = AttackStrategy.DISTRIBUTED
    
    # Reflection attack parameters
    amplification_factor: float = 1.0
    reflectors: List[str] = field(default_factory=list)
    
    # Pulsing attack parameters
    pulse_on_time: float = 0.1   # seconds
    pulse_off_time: float = 0.1  # seconds
    
    # Slowloris parameters
    connections_per_attacker: int = 100
    request_rate: float = 1.0    # requests per connection per second
    
    # Ramp-up for gradual attack
    ramp_up_time: float = 0.0
    
    # Link-targeted attack
    target_links: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class AttackMetrics:
    """
    Metrics for evaluating attack impact and cost
    
    Attributes:
        attack_packets_sent: Total attack packets sent
        attack_packets_delivered: Attack packets that reached targets
        attack_bandwidth_used: Total attack bandwidth (Mbps)
        normal_packets_affected: Normal packets dropped due to attack
        attack_cost: Estimated attack cost (resource units)
        attack_efficiency: Ratio of damage to cost
        max_link_utilization: Maximum link utilization during attack
        affected_links: Number of links with utilization > threshold
    """
    attack_packets_sent: int = 0
    attack_packets_delivered: int = 0
    attack_bandwidth_used: float = 0.0
    normal_packets_affected: int = 0
    attack_cost: float = 0.0
    attack_efficiency: float = 0.0
    max_link_utilization: float = 0.0
    affected_links: int = 0
    bottleneck_events: int = 0
    
    # Time series
    utilization_over_time: List[Tuple[float, float]] = field(default_factory=list)


class AttackFlow(Flow):
    """
    Extended Flow class for attack traffic with special behaviors
    """
    
    def __init__(
        self,
        flow_id: str,
        source: str,
        destination: str,
        rate: float,
        packet_size: int,
        attack_type: AttackType,
        start_time: float = 0.0,
        duration: float = -1.0,
        pulse_on: float = 0.1,
        pulse_off: float = 0.1,
        amplification: float = 1.0,
        ramp_up_time: float = 0.0
    ):
        # Use POISSON pattern instead of CONSTANT to ensure packets are generated
        # even when rate * time_step < 1 (which would result in 0 packets with CONSTANT)
        super().__init__(
            id=flow_id,
            source=source,
            destination=destination,
            rate=rate,
            packet_size=packet_size,
            flow_type=PacketType.ATTACK,
            start_time=start_time,
            duration=duration,
            pattern=TrafficPattern.POISSON
        )
        self.attack_type = attack_type
        self.pulse_on = pulse_on
        self.pulse_off = pulse_off
        self.amplification = amplification
        self.ramp_up_time = ramp_up_time
        self.pulse_state = True  # True = on, False = off
        self.pulse_timer = 0.0
        self.effective_start_time = start_time
    
    def get_current_rate(self, current_time: float) -> float:
        """Get current attack rate considering attack pattern"""
        if not self.is_active(current_time):
            return 0.0
        
        elapsed = current_time - self.start_time
        
        # Ramp-up phase
        if self.ramp_up_time > 0 and elapsed < self.ramp_up_time:
            ramp_factor = elapsed / self.ramp_up_time
            base_rate = self.rate * ramp_factor
        else:
            base_rate = self.rate
        
        # Pulsing pattern
        if self.attack_type == AttackType.PULSING:
            cycle_time = self.pulse_on + self.pulse_off
            cycle_position = elapsed % cycle_time
            if cycle_position >= self.pulse_on:
                return 0.0  # Off phase
        
        # Slowloris - lower rate but persistent
        if self.attack_type == AttackType.SLOWLORIS:
            return base_rate * 0.1  # Much lower rate
        
        return base_rate * self.amplification


class DDoSAttackGenerator:
    """
    DDoS Attack Generator for LEO satellite network simulation
    
    Generates various types of DDoS attack traffic based on
    attack configurations and target selection strategies.
    """
    
    def __init__(
        self,
        constellation,  # LEOConstellation
        traffic_generator: TrafficGenerator,
        seed: Optional[int] = None
    ):
        """
        Initialize DDoS attack generator
        
        Args:
            constellation: LEO constellation topology
            traffic_generator: Traffic generator instance
            seed: Random seed for reproducibility
        """
        self.constellation = constellation
        self.traffic_generator = traffic_generator
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)
        
        # Active attacks
        self.active_attacks: Dict[str, AttackConfig] = {}
        self.attack_flows: Dict[str, List[AttackFlow]] = {}
        self.attack_metrics: Dict[str, AttackMetrics] = {}
        
        # Attack counter
        self.attack_counter = 0
    
    def create_flooding_attack(
        self,
        targets: List[str],
        num_attackers: int = 10,
        total_rate: float = 5000.0,
        packet_size: int = 1000,
        start_time: float = 0.0,
        duration: float = -1.0,
        strategy: AttackStrategy = AttackStrategy.DISTRIBUTED,
        ramp_up_time: float = 0.0
    ) -> str:
        """
        Create a volumetric flooding attack
        
        Args:
            targets: List of target node IDs
            num_attackers: Number of attack sources
            total_rate: Total attack rate (packets/s)
            packet_size: Packet size in bytes
            start_time: Attack start time
            duration: Attack duration
            strategy: Source selection strategy
            ramp_up_time: Gradual ramp-up time
            
        Returns:
            Attack ID
        """
        config = AttackConfig(
            attack_type=AttackType.FLOODING,
            targets=targets,
            num_attackers=num_attackers,
            total_rate=total_rate,
            packet_size=packet_size,
            start_time=start_time,
            duration=duration,
            strategy=strategy,
            ramp_up_time=ramp_up_time
        )
        
        return self._deploy_attack(config)
    
    def create_reflection_attack(
        self,
        targets: List[str],
        reflectors: Optional[List[str]] = None,
        num_attackers: int = 5,
        total_rate: float = 1000.0,
        amplification_factor: float = 10.0,
        start_time: float = 0.0,
        duration: float = -1.0
    ) -> str:
        """
        Create a reflection/amplification attack
        
        The attack sends small requests to reflectors which respond
        with amplified traffic to the spoofed source (target).
        
        Args:
            targets: List of target node IDs (spoofed sources)
            reflectors: List of reflector node IDs (if None, auto-select)
            num_attackers: Number of attack sources
            total_rate: Request rate (responses will be amplified)
            amplification_factor: Response/Request size ratio
            start_time: Attack start time
            duration: Attack duration
            
        Returns:
            Attack ID
        """
        # Auto-select reflectors if not provided
        if reflectors is None:
            all_nodes = list(self.constellation.satellites.keys())
            # Exclude targets from reflectors
            available = [n for n in all_nodes if n not in targets]
            num_reflectors = min(num_attackers * 2, len(available))
            reflectors = list(self.rng.choice(
                available, size=num_reflectors, replace=False
            ))
        
        config = AttackConfig(
            attack_type=AttackType.REFLECTION,
            targets=targets,
            num_attackers=num_attackers,
            total_rate=total_rate,
            packet_size=100,  # Small request packets
            start_time=start_time,
            duration=duration,
            strategy=AttackStrategy.DISTRIBUTED,
            amplification_factor=amplification_factor,
            reflectors=reflectors
        )
        
        return self._deploy_attack(config)
    
    def create_slowloris_attack(
        self,
        targets: List[str],
        num_attackers: int = 20,
        connections_per_attacker: int = 100,
        request_rate: float = 1.0,
        start_time: float = 0.0,
        duration: float = -1.0
    ) -> str:
        """
        Create a Slowloris-style application layer attack
        
        Opens many connections and sends data very slowly to
        exhaust connection resources.
        
        Args:
            targets: List of target node IDs
            num_attackers: Number of attack sources
            connections_per_attacker: Connections per attacker
            request_rate: Request rate per connection (very low)
            start_time: Attack start time
            duration: Attack duration
            
        Returns:
            Attack ID
        """
        # Total rate is low but connections are many
        total_connections = num_attackers * connections_per_attacker
        total_rate = total_connections * request_rate
        
        config = AttackConfig(
            attack_type=AttackType.SLOWLORIS,
            targets=targets,
            num_attackers=num_attackers,
            total_rate=total_rate,
            packet_size=500,  # Small packets
            start_time=start_time,
            duration=duration,
            strategy=AttackStrategy.DISTRIBUTED,
            connections_per_attacker=connections_per_attacker,
            request_rate=request_rate
        )
        
        return self._deploy_attack(config)
    
    def create_pulsing_attack(
        self,
        targets: List[str],
        num_attackers: int = 10,
        peak_rate: float = 10000.0,
        pulse_on_time: float = 0.1,
        pulse_off_time: float = 0.1,
        start_time: float = 0.0,
        duration: float = -1.0
    ) -> str:
        """
        Create a pulsing (on-off) attack
        
        Alternates between high-rate bursts and quiet periods,
        making detection more difficult.
        
        Args:
            targets: List of target node IDs
            num_attackers: Number of attack sources
            peak_rate: Peak attack rate during on phase
            pulse_on_time: Duration of on phase
            pulse_off_time: Duration of off phase
            start_time: Attack start time
            duration: Attack duration
            
        Returns:
            Attack ID
        """
        config = AttackConfig(
            attack_type=AttackType.PULSING,
            targets=targets,
            num_attackers=num_attackers,
            total_rate=peak_rate,
            packet_size=1000,
            start_time=start_time,
            duration=duration,
            strategy=AttackStrategy.DISTRIBUTED,
            pulse_on_time=pulse_on_time,
            pulse_off_time=pulse_off_time
        )
        
        return self._deploy_attack(config)
    
    def create_coordinated_attack(
        self,
        targets: List[str],
        attack_vectors: List[AttackType],
        num_attackers_per_vector: int = 5,
        total_rate_per_vector: float = 2000.0,
        start_time: float = 0.0,
        duration: float = -1.0,
        stagger_time: float = 0.0
    ) -> str:
        """
        Create a coordinated multi-vector attack
        
        Combines multiple attack types simultaneously or in sequence.
        
        Args:
            targets: List of target node IDs
            attack_vectors: List of attack types to combine
            num_attackers_per_vector: Attackers per vector
            total_rate_per_vector: Rate per vector
            start_time: Attack start time
            duration: Attack duration
            stagger_time: Time between vector launches
            
        Returns:
            Attack ID (master)
        """
        master_id = f"coordinated_{self.attack_counter}"
        self.attack_counter += 1
        
        # Create sub-attacks for each vector
        sub_attack_ids = []
        for i, vector_type in enumerate(attack_vectors):
            vector_start = start_time + i * stagger_time
            
            if vector_type == AttackType.FLOODING:
                sub_id = self.create_flooding_attack(
                    targets=targets,
                    num_attackers=num_attackers_per_vector,
                    total_rate=total_rate_per_vector,
                    start_time=vector_start,
                    duration=duration
                )
            elif vector_type == AttackType.PULSING:
                sub_id = self.create_pulsing_attack(
                    targets=targets,
                    num_attackers=num_attackers_per_vector,
                    peak_rate=total_rate_per_vector,
                    start_time=vector_start,
                    duration=duration
                )
            elif vector_type == AttackType.SLOWLORIS:
                sub_id = self.create_slowloris_attack(
                    targets=targets,
                    num_attackers=num_attackers_per_vector,
                    start_time=vector_start,
                    duration=duration
                )
            elif vector_type == AttackType.REFLECTION:
                sub_id = self.create_reflection_attack(
                    targets=targets,
                    num_attackers=num_attackers_per_vector,
                    total_rate=total_rate_per_vector / 10,
                    amplification_factor=10.0,
                    start_time=vector_start,
                    duration=duration
                )
            else:
                continue
            
            sub_attack_ids.append(sub_id)
        
        # Store coordinated attack info
        config = AttackConfig(
            attack_type=AttackType.COORDINATED,
            targets=targets,
            start_time=start_time,
            duration=duration
        )
        self.active_attacks[master_id] = config
        self.attack_metrics[master_id] = AttackMetrics()
        
        return master_id
    
    def create_bottleneck_attack(
        self,
        target_links: List[Tuple[str, str]],
        num_attackers: int = 10,
        total_rate: float = 5000.0,
        start_time: float = 0.0,
        duration: float = -1.0
    ) -> str:
        """
        Create an attack targeting specific bottleneck links
        
        Selects attack sources and destinations to maximize
        traffic through specified links.
        
        Args:
            target_links: List of (source, target) link tuples to attack
            num_attackers: Number of attack sources
            total_rate: Total attack rate
            start_time: Attack start time
            duration: Attack duration
            
        Returns:
            Attack ID
        """
        config = AttackConfig(
            attack_type=AttackType.BOTTLENECK,
            targets=[],  # Will be computed
            num_attackers=num_attackers,
            total_rate=total_rate,
            packet_size=1000,
            start_time=start_time,
            duration=duration,
            strategy=AttackStrategy.PATH_ALIGNED,
            target_links=target_links
        )
        
        return self._deploy_attack(config)
    
    def _deploy_attack(self, config: AttackConfig) -> str:
        """
        Deploy an attack based on configuration
        
        Args:
            config: Attack configuration
            
        Returns:
            Attack ID
        """
        attack_id = f"attack_{config.attack_type.value}_{self.attack_counter}"
        self.attack_counter += 1
        
        # Store configuration
        self.active_attacks[attack_id] = config
        self.attack_metrics[attack_id] = AttackMetrics()
        self.attack_flows[attack_id] = []
        
        # Select attack sources
        attackers = self._select_attackers(config)
        
        # Create attack flows
        if config.attack_type == AttackType.REFLECTION:
            self._create_reflection_flows(attack_id, config, attackers)
        elif config.attack_type == AttackType.BOTTLENECK:
            self._create_bottleneck_flows(attack_id, config)
        else:
            self._create_standard_flows(attack_id, config, attackers)
        
        return attack_id
    
    def _select_attackers(self, config: AttackConfig) -> List[str]:
        """Select attack sources based on strategy"""
        all_nodes = list(self.constellation.satellites.keys())
        
        # Remove targets from potential attackers
        available = [n for n in all_nodes if n not in config.targets]
        
        num_attackers = min(config.num_attackers, len(available))
        
        if num_attackers == 0:
            return []
        
        if config.strategy == AttackStrategy.RANDOM:
            return list(self.rng.choice(available, size=num_attackers, replace=False))
        
        elif config.strategy == AttackStrategy.DISTRIBUTED:
            # Select attackers distributed across different planes
            planes = {}
            for node in available:
                sat = self.constellation.satellites[node]
                if sat.plane_id not in planes:
                    planes[sat.plane_id] = []
                planes[sat.plane_id].append(node)
            
            attackers = []
            plane_ids = list(planes.keys())
            max_iterations = num_attackers * 2  # Safety limit
            iterations = 0
            
            while len(attackers) < num_attackers and plane_ids and iterations < max_iterations:
                iterations += 1
                plane_id = plane_ids[iterations % len(plane_ids)]
                if planes[plane_id]:
                    attacker = self.rng.choice(planes[plane_id])
                    attackers.append(attacker)
                    planes[plane_id].remove(attacker)
                    if not planes[plane_id]:
                        plane_ids.remove(plane_id)
            
            return attackers
        
        elif config.strategy == AttackStrategy.CLUSTERED:
            # Select attackers from a few adjacent planes
            if not available:
                return []
            
            # Pick a random starting plane
            first = self.rng.choice(available)
            first_plane = self.constellation.satellites[first].plane_id
            
            # Select from this plane and adjacent planes
            clustered = [
                n for n in available
                if abs(self.constellation.satellites[n].plane_id - first_plane) <= 1
            ]
            
            return list(self.rng.choice(
                clustered, 
                size=min(num_attackers, len(clustered)), 
                replace=False
            ))
        
        elif config.strategy == AttackStrategy.PATH_ALIGNED:
            # Select attackers that will route through target links
            # This is a simplified version - in practice would use path analysis
            return list(self.rng.choice(available, size=num_attackers, replace=False))
        
        return []
    
    def _create_standard_flows(
        self, 
        attack_id: str, 
        config: AttackConfig, 
        attackers: List[str]
    ):
        """Create standard attack flows"""
        if not attackers or not config.targets:
            return
        
        rate_per_attacker = config.total_rate / len(attackers)
        
        for i, attacker in enumerate(attackers):
            # Round-robin target selection
            target = config.targets[i % len(config.targets)]
            
            flow = AttackFlow(
                flow_id=f"{attack_id}_flow_{i}",
                source=attacker,
                destination=target,
                rate=rate_per_attacker,
                packet_size=config.packet_size,
                attack_type=config.attack_type,
                start_time=config.start_time,
                duration=config.duration,
                pulse_on=config.pulse_on_time,
                pulse_off=config.pulse_off_time,
                ramp_up_time=config.ramp_up_time
            )
            
            self.attack_flows[attack_id].append(flow)
            self.traffic_generator.add_flow(flow)
    
    def _create_reflection_flows(
        self, 
        attack_id: str, 
        config: AttackConfig, 
        attackers: List[str]
    ):
        """Create reflection attack flows"""
        if not attackers or not config.reflectors or not config.targets:
            return
        
        # Attackers send to reflectors, reflectors "respond" to targets
        rate_per_attacker = config.total_rate / len(attackers)
        
        for i, attacker in enumerate(attackers):
            # Select reflector
            reflector = config.reflectors[i % len(config.reflectors)]
            target = config.targets[i % len(config.targets)]
            
            # Flow from attacker to reflector (small packets)
            trigger_flow = AttackFlow(
                flow_id=f"{attack_id}_trigger_{i}",
                source=attacker,
                destination=reflector,
                rate=rate_per_attacker,
                packet_size=config.packet_size,  # Small
                attack_type=config.attack_type,
                start_time=config.start_time,
                duration=config.duration
            )
            
            # Flow from reflector to target (amplified)
            amplified_flow = AttackFlow(
                flow_id=f"{attack_id}_amplified_{i}",
                source=reflector,
                destination=target,
                rate=rate_per_attacker,
                packet_size=int(config.packet_size * config.amplification_factor),
                attack_type=config.attack_type,
                start_time=config.start_time,
                duration=config.duration,
                amplification=config.amplification_factor
            )
            
            self.attack_flows[attack_id].extend([trigger_flow, amplified_flow])
            self.traffic_generator.add_flow(trigger_flow)
            self.traffic_generator.add_flow(amplified_flow)
    
    def _create_bottleneck_flows(self, attack_id: str, config: AttackConfig):
        """Create flows targeting specific bottleneck links"""
        if not config.target_links:
            return
        
        # For each target link, find source-destination pairs that use it
        all_nodes = list(self.constellation.satellites.keys())
        rate_per_link = config.total_rate / len(config.target_links)
        
        flow_idx = 0
        for src_node, dst_node in config.target_links:
            # Find nodes that would route through this link
            # Simplified: use nodes from planes adjacent to the link endpoints
            
            src_sat = self.constellation.satellites.get(src_node)
            dst_sat = self.constellation.satellites.get(dst_node)
            
            if not src_sat or not dst_sat:
                continue
            
            # Select attackers from "before" the link
            before_nodes = [
                n for n in all_nodes
                if self.constellation.satellites[n].plane_id <= src_sat.plane_id
            ]
            
            # Select targets "after" the link
            after_nodes = [
                n for n in all_nodes
                if self.constellation.satellites[n].plane_id >= dst_sat.plane_id
            ]
            
            if not before_nodes or not after_nodes:
                continue
            
            attackers_for_link = min(
                config.num_attackers // len(config.target_links),
                len(before_nodes)
            )
            
            selected_attackers = list(self.rng.choice(
                before_nodes, size=max(1, attackers_for_link), replace=False
            ))
            selected_targets = list(self.rng.choice(
                after_nodes, size=max(1, attackers_for_link), replace=False
            ))
            
            rate_per_attacker = rate_per_link / len(selected_attackers)
            
            for attacker, target in zip(selected_attackers, selected_targets):
                flow = AttackFlow(
                    flow_id=f"{attack_id}_flow_{flow_idx}",
                    source=attacker,
                    destination=target,
                    rate=rate_per_attacker,
                    packet_size=config.packet_size,
                    attack_type=config.attack_type,
                    start_time=config.start_time,
                    duration=config.duration
                )
                
                self.attack_flows[attack_id].append(flow)
                self.traffic_generator.add_flow(flow)
                flow_idx += 1
    
    def stop_attack(self, attack_id: str):
        """Stop an active attack"""
        if attack_id not in self.active_attacks:
            return
        
        # Remove attack flows from traffic generator
        if attack_id in self.attack_flows:
            for flow in self.attack_flows[attack_id]:
                self.traffic_generator.remove_flow(flow.id)
            del self.attack_flows[attack_id]
        
        del self.active_attacks[attack_id]
    
    def stop_all_attacks(self):
        """Stop all active attacks"""
        attack_ids = list(self.active_attacks.keys())
        for attack_id in attack_ids:
            self.stop_attack(attack_id)
    
    def get_attack_metrics(self, attack_id: str) -> Optional[AttackMetrics]:
        """Get metrics for a specific attack"""
        return self.attack_metrics.get(attack_id)
    
    def get_all_attack_metrics(self) -> Dict[str, AttackMetrics]:
        """Get metrics for all attacks"""
        return self.attack_metrics.copy()
    
    def update_metrics(self, attack_id: str, current_time: float):
        """Update attack metrics based on current state"""
        if attack_id not in self.attack_metrics:
            return
        
        metrics = self.attack_metrics[attack_id]
        
        # Calculate attack bandwidth
        if attack_id in self.attack_flows:
            total_bandwidth = 0.0
            for flow in self.attack_flows[attack_id]:
                rate = flow.get_current_rate(current_time)
                bandwidth_mbps = (rate * flow.packet_size * 8) / 1e6
                total_bandwidth += bandwidth_mbps
            metrics.attack_bandwidth_used = total_bandwidth
        
        # Update max link utilization
        max_util = 0.0
        affected = 0
        for link in self.constellation.links.values():
            util = link.get_utilization()
            max_util = max(max_util, util)
            if util > 0.8:  # 80% threshold
                affected += 1
        
        metrics.max_link_utilization = max_util
        metrics.affected_links = affected
        
        # Record time series
        metrics.utilization_over_time.append((current_time, max_util))
    
    def calculate_attack_cost(self, attack_id: str) -> float:
        """
        Calculate the cost of an attack
        
        Cost is based on:
        - Number of compromised hosts (attackers)
        - Bandwidth consumed
        - Duration of attack
        
        Returns:
            Estimated attack cost in abstract units
        """
        if attack_id not in self.active_attacks:
            return 0.0
        
        config = self.active_attacks[attack_id]
        metrics = self.attack_metrics[attack_id]
        
        # Cost components
        host_cost = config.num_attackers * 10  # Cost per compromised host
        bandwidth_cost = metrics.attack_bandwidth_used * 0.1  # Cost per Mbps
        
        # Reflection attacks have lower host cost but need reflectors
        if config.attack_type == AttackType.REFLECTION:
            host_cost *= 0.5
            bandwidth_cost *= config.amplification_factor * 0.2
        
        # Slowloris is cheaper but less effective
        if config.attack_type == AttackType.SLOWLORIS:
            host_cost *= 0.3
            bandwidth_cost *= 0.1
        
        total_cost = host_cost + bandwidth_cost
        metrics.attack_cost = total_cost
        
        return total_cost
    
    def get_attack_summary(self) -> Dict:
        """Get summary of all attacks"""
        summary = {
            "active_attacks": len(self.active_attacks),
            "total_attack_flows": sum(
                len(flows) for flows in self.attack_flows.values()
            ),
            "attacks": {}
        }
        
        for attack_id, config in self.active_attacks.items():
            metrics = self.attack_metrics.get(attack_id, AttackMetrics())
            summary["attacks"][attack_id] = {
                "type": config.attack_type.value,
                "targets": config.targets,
                "num_attackers": config.num_attackers,
                "total_rate": config.total_rate,
                "bandwidth_mbps": metrics.attack_bandwidth_used,
                "max_link_util": metrics.max_link_utilization,
                "affected_links": metrics.affected_links,
                "cost": metrics.attack_cost
            }
        
        return summary
