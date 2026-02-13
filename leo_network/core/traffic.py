"""
Traffic Generation Module

This module provides classes for generating network traffic,
including normal traffic and attack traffic for DDoS simulation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random


class PacketType(Enum):
    """Type of network packet"""
    NORMAL = "normal"
    ATTACK = "attack"


class TrafficPattern(Enum):
    """Traffic generation pattern"""
    CONSTANT = "constant"       # Constant bit rate
    POISSON = "poisson"         # Poisson arrival process
    BURSTY = "bursty"           # Bursty traffic
    ON_OFF = "on_off"           # On-off pattern


@dataclass
class Packet:
    """
    Represents a network packet
    
    Attributes:
        id: Unique packet identifier
        source: Source node ID
        destination: Destination node ID
        size: Packet size in bytes
        packet_type: Normal or attack packet
        creation_time: Time when packet was created
        path: List of node IDs representing the path
        current_hop: Current position in the path
    """
    id: int
    source: str
    destination: str
    size: int = 1000  # bytes
    packet_type: PacketType = PacketType.NORMAL
    creation_time: float = 0.0
    priority: int = 0
    
    # Runtime state
    path: List[str] = field(default_factory=list)
    current_hop: int = 0
    arrival_time: float = 0.0
    hops_taken: int = 0
    
    def __post_init__(self):
        if not self.path:
            self.path = []
    
    def get_next_hop(self) -> Optional[str]:
        """Get next node in the path"""
        if self.current_hop < len(self.path) - 1:
            return self.path[self.current_hop + 1]
        return None
    
    def advance_hop(self):
        """Move to next hop"""
        self.current_hop += 1
        self.hops_taken += 1
    
    def has_reached_destination(self) -> bool:
        """Check if packet has reached destination"""
        if not self.path:
            return False
        return self.current_hop >= len(self.path) - 1
    
    def get_delay(self) -> float:
        """Calculate end-to-end delay"""
        return self.arrival_time - self.creation_time


@dataclass
class Flow:
    """
    Represents a traffic flow (sequence of packets)
    
    Attributes:
        id: Unique flow identifier
        source: Source node ID
        destination: Destination node ID
        rate: Traffic rate in packets per second
        packet_size: Size of each packet in bytes
        flow_type: Normal or attack flow
        start_time: Flow start time
        duration: Flow duration (-1 for infinite)
    """
    id: str
    source: str
    destination: str
    rate: float = 100.0  # packets per second
    packet_size: int = 1000  # bytes
    flow_type: PacketType = PacketType.NORMAL
    start_time: float = 0.0
    duration: float = -1.0  # -1 means infinite
    pattern: TrafficPattern = TrafficPattern.POISSON
    
    # Runtime state
    packets_generated: int = 0
    packets_delivered: int = 0
    packets_dropped: int = 0
    total_delay: float = 0.0
    
    def is_active(self, current_time: float) -> bool:
        """Check if flow is active at current time"""
        if current_time < self.start_time:
            return False
        if self.duration < 0:
            return True
        return current_time < self.start_time + self.duration
    
    def get_average_delay(self) -> float:
        """Get average end-to-end delay"""
        if self.packets_delivered == 0:
            return 0.0
        return self.total_delay / self.packets_delivered
    
    def get_delivery_rate(self) -> float:
        """Get packet delivery rate"""
        if self.packets_generated == 0:
            return 0.0
        return self.packets_delivered / self.packets_generated
    
    def get_throughput_mbps(self, elapsed_time: float) -> float:
        """Get throughput in Mbps"""
        if elapsed_time <= 0:
            return 0.0
        total_bits = self.packets_delivered * self.packet_size * 8
        return total_bits / (elapsed_time * 1e6)
    
    def reset_stats(self):
        """Reset flow statistics"""
        self.packets_generated = 0
        self.packets_delivered = 0
        self.packets_dropped = 0
        self.total_delay = 0.0


class TrafficGenerator:
    """
    Traffic generator for network simulation
    
    Generates both normal traffic and attack traffic based on
    configured flows and patterns.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize traffic generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.flows: Dict[str, Flow] = {}
        self.packet_counter: int = 0
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)
        # Fractional accumulator for each flow to handle
        # low-rate flows where rate * time_step < 1
        self._flow_accumulators: Dict[str, float] = {}
    
    def add_flow(self, flow: Flow):
        """Add a traffic flow"""
        self.flows[flow.id] = flow
        self._flow_accumulators[flow.id] = 0.0
    
    def remove_flow(self, flow_id: str):
        """Remove a traffic flow"""
        if flow_id in self.flows:
            del self.flows[flow_id]
        self._flow_accumulators.pop(flow_id, None)
    
    def create_normal_flow(
        self,
        flow_id: str,
        source: str,
        destination: str,
        rate: float = 100.0,
        packet_size: int = 1000,
        start_time: float = 0.0,
        duration: float = -1.0,
        pattern: TrafficPattern = TrafficPattern.POISSON
    ) -> Flow:
        """
        Create and add a normal traffic flow
        
        Args:
            flow_id: Unique flow identifier
            source: Source node ID
            destination: Destination node ID
            rate: Traffic rate (packets/s)
            packet_size: Packet size in bytes
            start_time: Flow start time
            duration: Flow duration (-1 for infinite)
            pattern: Traffic pattern
            
        Returns:
            Created flow object
        """
        flow = Flow(
            id=flow_id,
            source=source,
            destination=destination,
            rate=rate,
            packet_size=packet_size,
            flow_type=PacketType.NORMAL,
            start_time=start_time,
            duration=duration,
            pattern=pattern
        )
        self.add_flow(flow)
        return flow
    
    def create_attack_flow(
        self,
        flow_id: str,
        source: str,
        destination: str,
        rate: float = 1000.0,
        packet_size: int = 1000,
        start_time: float = 0.0,
        duration: float = -1.0,
        pattern: TrafficPattern = TrafficPattern.CONSTANT
    ) -> Flow:
        """
        Create and add an attack traffic flow
        
        Args:
            flow_id: Unique flow identifier
            source: Source node ID
            destination: Destination node ID
            rate: Attack rate (packets/s)
            packet_size: Packet size in bytes
            start_time: Attack start time
            duration: Attack duration (-1 for infinite)
            pattern: Traffic pattern
            
        Returns:
            Created flow object
        """
        flow = Flow(
            id=flow_id,
            source=source,
            destination=destination,
            rate=rate,
            packet_size=packet_size,
            flow_type=PacketType.ATTACK,
            start_time=start_time,
            duration=duration,
            pattern=pattern
        )
        self.add_flow(flow)
        return flow
    
    def generate_packets(
        self,
        current_time: float,
        time_step: float
    ) -> List[Packet]:
        """
        Generate packets for current time step
        
        Args:
            current_time: Current simulation time
            time_step: Duration of time step
            
        Returns:
            List of generated packets
        """
        packets = []
        
        for flow in self.flows.values():
            if not flow.is_active(current_time):
                continue
            
            # Get current rate - support dynamic rate for attack flows
            if hasattr(flow, 'get_current_rate'):
                current_rate = flow.get_current_rate(current_time)
            else:
                current_rate = flow.rate
            
            # Skip if rate is zero (e.g., pulsing attack in off phase)
            if current_rate <= 0:
                continue
            
            # Calculate number of packets to generate
            num_packets = self._calculate_num_packets(
                flow.id, current_rate, time_step, flow.pattern
            )
            
            for _ in range(num_packets):
                packet = Packet(
                    id=self.packet_counter,
                    source=flow.source,
                    destination=flow.destination,
                    size=flow.packet_size,
                    packet_type=flow.flow_type,
                    creation_time=current_time + self.rng.uniform(0, time_step)
                )
                packets.append(packet)
                flow.packets_generated += 1
                self.packet_counter += 1
        
        return packets
    
    def _calculate_num_packets(
        self,
        flow_id: str,
        rate: float,
        time_step: float,
        pattern: TrafficPattern
    ) -> int:
        """
        Calculate number of packets to generate based on pattern.
        
        Uses a fractional accumulator per flow so that low-rate flows
        (where rate * time_step < 1.0) still correctly generate packets
        over multiple time steps.
        """
        expected = rate * time_step
        
        if pattern == TrafficPattern.CONSTANT:
            # Accumulate fractional packets across time steps
            accumulated = self._flow_accumulators.get(flow_id, 0.0) + expected
            num = int(accumulated)
            self._flow_accumulators[flow_id] = accumulated - num
            return num
        
        elif pattern == TrafficPattern.POISSON:
            return self.rng.poisson(expected)
        
        elif pattern == TrafficPattern.BURSTY:
            # Bursty: sometimes generate more, sometimes less
            if self.rng.random() < 0.3:
                return int(expected * 3)  # Burst
            elif self.rng.random() < 0.5:
                return 0  # Quiet period
            else:
                return int(expected)
        
        elif pattern == TrafficPattern.ON_OFF:
            # On-off: 50% chance of being on
            if self.rng.random() < 0.5:
                return int(expected * 2)
            else:
                return 0
        
        return int(expected)
    
    def record_packet_delivery(self, packet: Packet, arrival_time: float):
        """Record successful packet delivery"""
        # Find the flow this packet belongs to
        for flow in self.flows.values():
            if flow.source == packet.source and flow.destination == packet.destination:
                if flow.flow_type == packet.packet_type:
                    flow.packets_delivered += 1
                    flow.total_delay += arrival_time - packet.creation_time
                    break
    
    def record_packet_drop(self, packet: Packet):
        """Record packet drop"""
        for flow in self.flows.values():
            if flow.source == packet.source and flow.destination == packet.destination:
                if flow.flow_type == packet.packet_type:
                    flow.packets_dropped += 1
                    break
    
    def get_flow_stats(self) -> Dict:
        """Get statistics for all flows"""
        stats = {}
        for flow_id, flow in self.flows.items():
            stats[flow_id] = {
                "source": flow.source,
                "destination": flow.destination,
                "type": flow.flow_type.value,
                "rate": flow.rate,
                "packets_generated": flow.packets_generated,
                "packets_delivered": flow.packets_delivered,
                "packets_dropped": flow.packets_dropped,
                "delivery_rate": flow.get_delivery_rate(),
                "average_delay_ms": flow.get_average_delay()
            }
        return stats
    
    def get_aggregate_stats(self) -> Dict:
        """Get aggregate statistics across all flows"""
        total_generated = sum(f.packets_generated for f in self.flows.values())
        total_delivered = sum(f.packets_delivered for f in self.flows.values())
        total_dropped = sum(f.packets_dropped for f in self.flows.values())
        
        normal_flows = [f for f in self.flows.values() if f.flow_type == PacketType.NORMAL]
        attack_flows = [f for f in self.flows.values() if f.flow_type == PacketType.ATTACK]
        
        normal_generated = sum(f.packets_generated for f in normal_flows)
        normal_delivered = sum(f.packets_delivered for f in normal_flows)
        attack_generated = sum(f.packets_generated for f in attack_flows)
        attack_delivered = sum(f.packets_delivered for f in attack_flows)
        
        return {
            "total_packets_generated": total_generated,
            "total_packets_delivered": total_delivered,
            "total_packets_dropped": total_dropped,
            "overall_delivery_rate": total_delivered / total_generated if total_generated > 0 else 0.0,
            "normal_packets_generated": normal_generated,
            "normal_packets_delivered": normal_delivered,
            "normal_delivery_rate": normal_delivered / normal_generated if normal_generated > 0 else 0.0,
            "attack_packets_generated": attack_generated,
            "attack_packets_delivered": attack_delivered,
            "attack_delivery_rate": attack_delivered / attack_generated if attack_generated > 0 else 0.0,
            "num_normal_flows": len(normal_flows),
            "num_attack_flows": len(attack_flows)
        }
    
    def reset_all_stats(self):
        """Reset statistics for all flows"""
        for flow in self.flows.values():
            flow.reset_stats()
        self.packet_counter = 0
        # Reset accumulators
        for flow_id in self._flow_accumulators:
            self._flow_accumulators[flow_id] = 0.0
    
    def get_active_flows(self, current_time: float) -> List[Flow]:
        """Get list of currently active flows"""
        return [f for f in self.flows.values() if f.is_active(current_time)]
    
    def get_total_traffic_rate(self, current_time: float) -> float:
        """Get total traffic rate at current time (packets/s)"""
        active_flows = self.get_active_flows(current_time)
        return sum(f.rate for f in active_flows)
    
    def get_attack_traffic_ratio(self, current_time: float) -> float:
        """Get ratio of attack traffic to total traffic"""
        active_flows = self.get_active_flows(current_time)
        if not active_flows:
            return 0.0
        
        total_rate = sum(f.rate for f in active_flows)
        attack_rate = sum(f.rate for f in active_flows if f.flow_type == PacketType.ATTACK)
        
        return attack_rate / total_rate if total_rate > 0 else 0.0
