"""
Statistics Collection Module

This module provides classes for collecting and analyzing
simulation statistics including throughput, delay, packet loss,
and link utilization metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json


@dataclass
class TimeSeriesData:
    """Container for time series data"""
    timestamps: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    
    def add(self, timestamp: float, value: float):
        self.timestamps.append(timestamp)
        self.values.append(value)
    
    def get_average(self) -> float:
        return np.mean(self.values) if self.values else 0.0
    
    def get_max(self) -> float:
        return np.max(self.values) if self.values else 0.0
    
    def get_min(self) -> float:
        return np.min(self.values) if self.values else 0.0
    
    def get_percentile(self, p: float) -> float:
        return np.percentile(self.values, p) if self.values else 0.0
    
    def to_dataframe(self, name: str = "value") -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": self.timestamps,
            name: self.values
        })


class StatisticsCollector:
    """
    Collects and aggregates simulation statistics
    
    Tracks metrics including:
    - Throughput (packets/s, Mbps)
    - End-to-end delay (ms)
    - Packet loss rate
    - Link utilization
    - Queue occupancy
    - 5th percentile throughput (for worst-case performance analysis)
    """
    
    def __init__(self):
        """Initialize statistics collector"""
        # Packet-level statistics
        self.packets_sent: int = 0
        self.packets_delivered: int = 0
        self.packets_dropped: int = 0
        
        # Byte-level statistics
        self.bytes_sent: int = 0
        self.bytes_delivered: int = 0
        self.bytes_dropped: int = 0
        
        # Delay statistics
        self.delays: List[float] = []
        self.hop_counts: List[int] = []
        
        # Time series data
        self.throughput_series = TimeSeriesData()
        self.throughput_mbps_series = TimeSeriesData()  # Throughput in Mbps for percentile calculation
        self.delay_series = TimeSeriesData()
        self.loss_rate_series = TimeSeriesData()
        self.link_utilization_series: Dict[str, TimeSeriesData] = defaultdict(TimeSeriesData)
        self.queue_occupancy_series: Dict[str, TimeSeriesData] = defaultdict(TimeSeriesData)
        
        # Per-flow statistics
        self.flow_stats: Dict[str, Dict] = {}
        
        # Attack-specific statistics
        self.normal_packets_delivered: int = 0
        self.normal_packets_dropped: int = 0
        self.attack_packets_delivered: int = 0
        self.attack_packets_dropped: int = 0
        
        # Per-window throughput tracking for percentile calculation
        self.window_throughput_pps: List[float] = []  # Throughput samples in pps
        self.window_throughput_mbps: List[float] = []  # Throughput samples in Mbps
        self.window_bytes_delivered: int = 0  # Bytes delivered in current window
        self.window_packets_delivered: int = 0  # Packets delivered in current window
        
        # Snapshot storage
        self.snapshots: List[Dict] = []
    
    def record_packet_sent(self, packet_size: int, is_attack: bool = False):
        """Record packet sent"""
        self.packets_sent += 1
        self.bytes_sent += packet_size
    
    def record_packet_delivered(
        self,
        packet_size: int,
        delay: float,
        hop_count: int,
        is_attack: bool = False
    ):
        """Record successful packet delivery"""
        self.packets_delivered += 1
        self.bytes_delivered += packet_size
        self.delays.append(delay)
        self.hop_counts.append(hop_count)
        
        # Track per-window statistics
        self.window_bytes_delivered += packet_size
        self.window_packets_delivered += 1
        
        if is_attack:
            self.attack_packets_delivered += 1
        else:
            self.normal_packets_delivered += 1
    
    def record_packet_dropped(self, packet_size: int, is_attack: bool = False):
        """Record packet drop"""
        self.packets_dropped += 1
        self.bytes_dropped += packet_size
        
        if is_attack:
            self.attack_packets_dropped += 1
        else:
            self.normal_packets_dropped += 1
    
    def record_link_utilization(self, link_id: str, timestamp: float, utilization: float):
        """Record link utilization"""
        self.link_utilization_series[link_id].add(timestamp, utilization)
    
    def record_queue_occupancy(self, node_id: str, timestamp: float, occupancy: float):
        """Record queue occupancy"""
        self.queue_occupancy_series[node_id].add(timestamp, occupancy)
    
    def take_snapshot(self, timestamp: float, additional_data: Optional[Dict] = None, window_duration: float = 0.1):
        """
        Take a snapshot of current statistics
        
        Args:
            timestamp: Current simulation time
            additional_data: Additional data to include in snapshot
            window_duration: Duration of the current snapshot window (for throughput calculation)
        """
        # Calculate throughput for this window
        if window_duration > 0:
            window_throughput_pps = self.window_packets_delivered / window_duration
            window_throughput_mbps = (self.window_bytes_delivered * 8) / (window_duration * 1e6)
        else:
            window_throughput_pps = 0.0
            window_throughput_mbps = 0.0
        
        # Store window throughput for percentile calculation
        self.window_throughput_pps.append(window_throughput_pps)
        self.window_throughput_mbps.append(window_throughput_mbps)
        
        # Reset window counters
        self.window_packets_delivered = 0
        self.window_bytes_delivered = 0
        
        snapshot = {
            "timestamp": timestamp,
            "packets_sent": self.packets_sent,
            "packets_delivered": self.packets_delivered,
            "packets_dropped": self.packets_dropped,
            "delivery_rate": self.get_delivery_rate(),
            "avg_delay": self.get_average_delay(),
            "throughput_pps": self.get_throughput_pps(timestamp),
            "window_throughput_pps": window_throughput_pps,
            "window_throughput_mbps": window_throughput_mbps,
        }
        
        if additional_data:
            snapshot.update(additional_data)
        
        self.snapshots.append(snapshot)
        
        # Record time series
        self.throughput_series.add(timestamp, snapshot["throughput_pps"])
        self.throughput_mbps_series.add(timestamp, window_throughput_mbps)
        self.delay_series.add(timestamp, snapshot["avg_delay"])
        self.loss_rate_series.add(timestamp, 1.0 - snapshot["delivery_rate"])
    
    def get_delivery_rate(self) -> float:
        """Get overall packet delivery rate"""
        if self.packets_sent == 0:
            return 0.0
        return self.packets_delivered / self.packets_sent
    
    def get_loss_rate(self) -> float:
        """Get overall packet loss rate"""
        return 1.0 - self.get_delivery_rate()
    
    def get_average_delay(self) -> float:
        """Get average end-to-end delay in ms"""
        if not self.delays:
            return 0.0
        return np.mean(self.delays)
    
    def get_delay_percentile(self, p: float) -> float:
        """Get delay percentile"""
        if not self.delays:
            return 0.0
        return np.percentile(self.delays, p)
    
    def get_average_hop_count(self) -> float:
        """Get average hop count"""
        if not self.hop_counts:
            return 0.0
        return np.mean(self.hop_counts)
    
    def get_throughput_pps(self, elapsed_time: float) -> float:
        """Get throughput in packets per second"""
        if elapsed_time <= 0:
            return 0.0
        return self.packets_delivered / elapsed_time
    
    def get_throughput_mbps(self, elapsed_time: float) -> float:
        """Get throughput in Mbps"""
        if elapsed_time <= 0:
            return 0.0
        return (self.bytes_delivered * 8) / (elapsed_time * 1e6)
    
    def get_throughput_percentile_pps(self, p: float) -> float:
        """
        Get throughput percentile in packets per second
        
        Args:
            p: Percentile value (0-100)
            
        Returns:
            Throughput at the given percentile
        """
        if not self.window_throughput_pps:
            return 0.0
        return np.percentile(self.window_throughput_pps, p)
    
    def get_throughput_percentile_mbps(self, p: float) -> float:
        """
        Get throughput percentile in Mbps
        
        Args:
            p: Percentile value (0-100)
            
        Returns:
            Throughput at the given percentile in Mbps
        """
        if not self.window_throughput_mbps:
            return 0.0
        return np.percentile(self.window_throughput_mbps, p)
    
    def get_5th_percentile_throughput_pps(self) -> float:
        """
        Get 5th percentile throughput in packets per second
        
        This represents the worst-case throughput (bottom 5% of observations).
        95% of the time, throughput is higher than this value.
        
        Returns:
            5th percentile throughput in pps
        """
        return self.get_throughput_percentile_pps(5)
    
    def get_5th_percentile_throughput_mbps(self) -> float:
        """
        Get 5th percentile throughput in Mbps
        
        This represents the worst-case throughput (bottom 5% of observations).
        95% of the time, throughput is higher than this value.
        
        Returns:
            5th percentile throughput in Mbps
        """
        return self.get_throughput_percentile_mbps(5)
    
    def get_normal_traffic_stats(self) -> Dict:
        """Get statistics for normal (non-attack) traffic"""
        total_normal = self.normal_packets_delivered + self.normal_packets_dropped
        return {
            "delivered": self.normal_packets_delivered,
            "dropped": self.normal_packets_dropped,
            "delivery_rate": self.normal_packets_delivered / total_normal if total_normal > 0 else 0.0
        }
    
    def get_attack_traffic_stats(self) -> Dict:
        """Get statistics for attack traffic"""
        total_attack = self.attack_packets_delivered + self.attack_packets_dropped
        return {
            "delivered": self.attack_packets_delivered,
            "dropped": self.attack_packets_dropped,
            "delivery_rate": self.attack_packets_delivered / total_attack if total_attack > 0 else 0.0
        }
    
    def get_link_utilization_stats(self) -> Dict:
        """Get aggregate link utilization statistics"""
        if not self.link_utilization_series:
            return {"avg": 0.0, "max": 0.0, "min": 0.0}
        
        all_utils = []
        for series in self.link_utilization_series.values():
            all_utils.extend(series.values)
        
        if not all_utils:
            return {"avg": 0.0, "max": 0.0, "min": 0.0}
        
        return {
            "avg": np.mean(all_utils),
            "max": np.max(all_utils),
            "min": np.min(all_utils),
            "p95": np.percentile(all_utils, 95),
            "p99": np.percentile(all_utils, 99)
        }
    
    def get_summary(self) -> Dict:
        """Get comprehensive statistics summary"""
        elapsed = self.snapshots[-1]["timestamp"] if self.snapshots else 1.0
        
        return {
            "overview": {
                "total_packets_sent": self.packets_sent,
                "total_packets_delivered": self.packets_delivered,
                "total_packets_dropped": self.packets_dropped,
                "delivery_rate": self.get_delivery_rate(),
                "loss_rate": self.get_loss_rate(),
            },
            "throughput": {
                "avg_pps": self.get_throughput_pps(elapsed),
                "avg_mbps": self.get_throughput_mbps(elapsed),
                "p5_pps": self.get_5th_percentile_throughput_pps(),
                "p5_mbps": self.get_5th_percentile_throughput_mbps(),
                "p10_pps": self.get_throughput_percentile_pps(10),
                "p10_mbps": self.get_throughput_percentile_mbps(10),
                "p50_pps": self.get_throughput_percentile_pps(50),
                "p50_mbps": self.get_throughput_percentile_mbps(50),
            },
            "delay": {
                "avg_ms": self.get_average_delay(),
                "p50_ms": self.get_delay_percentile(50),
                "p95_ms": self.get_delay_percentile(95),
                "p99_ms": self.get_delay_percentile(99),
                "max_ms": np.max(self.delays) if self.delays else 0.0,
            },
            "hop_count": {
                "avg": self.get_average_hop_count(),
                "max": np.max(self.hop_counts) if self.hop_counts else 0,
            },
            "normal_traffic": self.get_normal_traffic_stats(),
            "attack_traffic": self.get_attack_traffic_stats(),
            "link_utilization": self.get_link_utilization_stats(),
        }
    
    def reset(self):
        """Reset all statistics"""
        self.packets_sent = 0
        self.packets_delivered = 0
        self.packets_dropped = 0
        self.bytes_sent = 0
        self.bytes_delivered = 0
        self.bytes_dropped = 0
        self.delays = []
        self.hop_counts = []
        self.throughput_series = TimeSeriesData()
        self.throughput_mbps_series = TimeSeriesData()
        self.delay_series = TimeSeriesData()
        self.loss_rate_series = TimeSeriesData()
        self.link_utilization_series.clear()
        self.queue_occupancy_series.clear()
        self.flow_stats.clear()
        self.normal_packets_delivered = 0
        self.normal_packets_dropped = 0
        self.attack_packets_delivered = 0
        self.attack_packets_dropped = 0
        # Reset window throughput tracking
        self.window_throughput_pps = []
        self.window_throughput_mbps = []
        self.window_bytes_delivered = 0
        self.window_packets_delivered = 0
        self.snapshots = []
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert snapshots to DataFrame"""
        return pd.DataFrame(self.snapshots)
    
    def save_to_json(self, filepath: str):
        """Save statistics to JSON file"""
        import numpy as np
        
        def convert_numpy(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        data = {
            "summary": convert_numpy(self.get_summary()),
            "snapshots": convert_numpy(self.snapshots)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self):
        """Print formatted statistics summary"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("SIMULATION STATISTICS SUMMARY")
        print("="*60)
        
        print("\n--- Overview ---")
        for key, value in summary["overview"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n--- Throughput ---")
        print(f"  Average: {summary['throughput']['avg_pps']:.2f} pps")
        print(f"  Average: {summary['throughput']['avg_mbps']:.4f} Mbps")
        print(f"  5th Percentile: {summary['throughput']['p5_pps']:.2f} pps ({summary['throughput']['p5_mbps']:.4f} Mbps)")
        print(f"  10th Percentile: {summary['throughput']['p10_pps']:.2f} pps ({summary['throughput']['p10_mbps']:.4f} Mbps)")
        print(f"  50th Percentile: {summary['throughput']['p50_pps']:.2f} pps ({summary['throughput']['p50_mbps']:.4f} Mbps)")
        
        print("\n--- Delay (ms) ---")
        print(f"  Average: {summary['delay']['avg_ms']:.2f}")
        print(f"  P50: {summary['delay']['p50_ms']:.2f}")
        print(f"  P95: {summary['delay']['p95_ms']:.2f}")
        print(f"  P99: {summary['delay']['p99_ms']:.2f}")
        print(f"  Max: {summary['delay']['max_ms']:.2f}")
        
        print("\n--- Normal Traffic ---")
        normal = summary["normal_traffic"]
        print(f"  Delivered: {normal['delivered']}")
        print(f"  Dropped: {normal['dropped']}")
        print(f"  Delivery Rate: {normal['delivery_rate']:.4f}")
        
        print("\n--- Attack Traffic ---")
        attack = summary["attack_traffic"]
        print(f"  Delivered: {attack['delivered']}")
        print(f"  Dropped: {attack['dropped']}")
        print(f"  Delivery Rate: {attack['delivery_rate']:.4f}")
        
        print("\n--- Link Utilization ---")
        util = summary["link_utilization"]
        print(f"  Average: {util['avg']:.4f}")
        print(f"  Max: {util['max']:.4f}")
        
        print("\n" + "="*60)


class AttackCostCalculator:
    """
    Attack Cost Calculator
    
    Calculates the attack cost based on the principle that:
    - Higher attack traffic required to achieve the same normal packet loss rate
      indicates higher attack cost (better defense effectiveness)
    
    Attack Cost = Attack Traffic Volume / Normal Packet Loss Rate
    
    This metric helps evaluate routing algorithms' resilience against DDoS attacks.
    A higher attack cost means the attacker needs more resources to cause the same damage.
    """
    
    def __init__(self):
        """Initialize attack cost calculator"""
        # Attack traffic metrics
        self.attack_packets_sent: int = 0
        self.attack_bytes_sent: int = 0
        self.attack_packets_delivered: int = 0
        self.attack_bytes_delivered: int = 0
        
        # Normal traffic metrics
        self.normal_packets_sent: int = 0
        self.normal_bytes_sent: int = 0
        self.normal_packets_delivered: int = 0
        self.normal_packets_dropped: int = 0
        self.normal_bytes_delivered: int = 0
        self.normal_bytes_dropped: int = 0
        
        # Time series for tracking cost evolution
        self.cost_history: List[Tuple[float, float]] = []  # (timestamp, cost)
        self.loss_rate_history: List[Tuple[float, float]] = []  # (timestamp, loss_rate)
        self.attack_rate_history: List[Tuple[float, float]] = []  # (timestamp, attack_rate_mbps)
        
        # Baseline metrics (no attack scenario)
        self.baseline_loss_rate: float = 0.0
        self.baseline_set: bool = False
    
    def record_attack_packet_sent(self, packet_size: int):
        """Record attack packet sent"""
        self.attack_packets_sent += 1
        self.attack_bytes_sent += packet_size
    
    def record_attack_packet_delivered(self, packet_size: int):
        """Record attack packet delivered"""
        self.attack_packets_delivered += 1
        self.attack_bytes_delivered += packet_size
    
    def record_normal_packet_sent(self, packet_size: int):
        """Record normal packet sent"""
        self.normal_packets_sent += 1
        self.normal_bytes_sent += packet_size
    
    def record_normal_packet_delivered(self, packet_size: int):
        """Record normal packet delivered"""
        self.normal_packets_delivered += 1
        self.normal_bytes_delivered += packet_size
    
    def record_normal_packet_dropped(self, packet_size: int):
        """Record normal packet dropped"""
        self.normal_packets_dropped += 1
        self.normal_bytes_dropped += packet_size
    
    def set_baseline_loss_rate(self, loss_rate: float):
        """
        Set baseline loss rate from no-attack scenario
        
        Args:
            loss_rate: Normal packet loss rate without attack
        """
        self.baseline_loss_rate = loss_rate
        self.baseline_set = True
    
    def get_normal_loss_rate(self) -> float:
        """
        Get current normal packet loss rate
        
        Returns:
            Loss rate as a float between 0 and 1
        """
        total_normal = self.normal_packets_delivered + self.normal_packets_dropped
        if total_normal == 0:
            return 0.0
        return self.normal_packets_dropped / total_normal
    
    def get_attack_induced_loss_rate(self) -> float:
        """
        Get attack-induced loss rate (excluding baseline)
        
        Returns:
            Attack-induced loss rate
        """
        current_loss = self.get_normal_loss_rate()
        if self.baseline_set:
            # Subtract baseline loss rate to get attack-induced portion
            induced_loss = max(0.0, current_loss - self.baseline_loss_rate)
            return induced_loss
        return current_loss
    
    def get_attack_traffic_volume_bytes(self) -> int:
        """Get total attack traffic volume in bytes"""
        return self.attack_bytes_sent
    
    def get_attack_traffic_volume_mbps(self, elapsed_time: float) -> float:
        """
        Get average attack traffic rate in Mbps
        
        Args:
            elapsed_time: Simulation elapsed time in seconds
            
        Returns:
            Attack rate in Mbps
        """
        if elapsed_time <= 0:
            return 0.0
        return (self.attack_bytes_sent * 8) / (elapsed_time * 1e6)
    
    def get_attack_traffic_pps(self, elapsed_time: float) -> float:
        """
        Get average attack traffic rate in packets per second
        
        Args:
            elapsed_time: Simulation elapsed time in seconds
            
        Returns:
            Attack rate in pps
        """
        if elapsed_time <= 0:
            return 0.0
        return self.attack_packets_sent / elapsed_time
    
    def calculate_attack_cost(self, elapsed_time: float, epsilon: float = 1e-6) -> float:
        """
        Calculate attack cost
        
        Attack Cost = Attack Traffic (Mbps) / Attack-Induced Loss Rate
        
        Higher cost means attacker needs more traffic to achieve the same damage,
        indicating better defense effectiveness.
        
        Args:
            elapsed_time: Simulation elapsed time in seconds
            epsilon: Small value to prevent division by zero
            
        Returns:
            Attack cost value (higher is better for defense)
        """
        attack_rate_mbps = self.get_attack_traffic_volume_mbps(elapsed_time)
        induced_loss_rate = self.get_attack_induced_loss_rate()
        
        if induced_loss_rate < epsilon:
            # No significant attack-induced loss
            # Return a very high cost indicating ineffective attack
            if attack_rate_mbps > 0:
                return float('inf')
            return 0.0
        
        # Cost = traffic needed per unit of damage
        cost = attack_rate_mbps / induced_loss_rate
        return cost
    
    def calculate_normalized_attack_cost(
        self, 
        elapsed_time: float,
        target_loss_rate: float = 0.1,
        epsilon: float = 1e-6
    ) -> float:
        """
        Calculate normalized attack cost for a target loss rate
        
        This estimates how much attack traffic would be needed to achieve
        a specific target normal packet loss rate.
        
        Args:
            elapsed_time: Simulation elapsed time in seconds
            target_loss_rate: Target normal packet loss rate (default 10%)
            epsilon: Small value to prevent division by zero
            
        Returns:
            Estimated attack traffic (Mbps) needed to achieve target loss rate
        """
        attack_rate_mbps = self.get_attack_traffic_volume_mbps(elapsed_time)
        induced_loss_rate = self.get_attack_induced_loss_rate()
        
        if induced_loss_rate < epsilon:
            # Cannot estimate if no induced loss
            return float('inf')
        
        # Linear extrapolation: cost_for_target = attack_rate * (target / current)
        normalized_cost = attack_rate_mbps * (target_loss_rate / induced_loss_rate)
        return normalized_cost
    
    def record_snapshot(self, timestamp: float, elapsed_time: float):
        """
        Record a snapshot of current metrics
        
        Args:
            timestamp: Current simulation time
            elapsed_time: Total elapsed time
        """
        loss_rate = self.get_normal_loss_rate()
        attack_rate = self.get_attack_traffic_volume_mbps(elapsed_time)
        cost = self.calculate_attack_cost(elapsed_time)
        
        self.loss_rate_history.append((timestamp, loss_rate))
        self.attack_rate_history.append((timestamp, attack_rate))
        if cost != float('inf'):
            self.cost_history.append((timestamp, cost))
    
    def get_cost_efficiency_ratio(self, elapsed_time: float) -> float:
        """
        Calculate cost efficiency ratio
        
        Efficiency = Normal Packets Dropped / Attack Packets Sent
        
        Lower efficiency means attack is less effective (better defense).
        
        Args:
            elapsed_time: Simulation elapsed time
            
        Returns:
            Cost efficiency ratio
        """
        if self.attack_packets_sent == 0:
            return 0.0
        return self.normal_packets_dropped / self.attack_packets_sent
    
    def get_damage_per_mbps(self, elapsed_time: float) -> float:
        """
        Calculate damage caused per Mbps of attack traffic
        
        Damage per Mbps = Induced Loss Rate / Attack Rate (Mbps)
        
        Lower value indicates better defense.
        
        Args:
            elapsed_time: Simulation elapsed time
            
        Returns:
            Damage per Mbps
        """
        attack_rate = self.get_attack_traffic_volume_mbps(elapsed_time)
        if attack_rate == 0:
            return 0.0
        return self.get_attack_induced_loss_rate() / attack_rate
    
    def get_summary(self, elapsed_time: float) -> Dict:
        """
        Get comprehensive attack cost summary
        
        Args:
            elapsed_time: Simulation elapsed time
            
        Returns:
            Dictionary with all cost metrics
        """
        return {
            "attack_traffic": {
                "packets_sent": self.attack_packets_sent,
                "bytes_sent": self.attack_bytes_sent,
                "rate_mbps": self.get_attack_traffic_volume_mbps(elapsed_time),
                "rate_pps": self.get_attack_traffic_pps(elapsed_time),
            },
            "normal_traffic": {
                "packets_sent": self.normal_packets_sent,
                "packets_delivered": self.normal_packets_delivered,
                "packets_dropped": self.normal_packets_dropped,
                "loss_rate": self.get_normal_loss_rate(),
            },
            "cost_metrics": {
                "baseline_loss_rate": self.baseline_loss_rate,
                "induced_loss_rate": self.get_attack_induced_loss_rate(),
                "attack_cost": self.calculate_attack_cost(elapsed_time),
                "normalized_cost_10pct": self.calculate_normalized_attack_cost(
                    elapsed_time, target_loss_rate=0.1
                ),
                "cost_efficiency_ratio": self.get_cost_efficiency_ratio(elapsed_time),
                "damage_per_mbps": self.get_damage_per_mbps(elapsed_time),
            }
        }
    
    def reset(self):
        """Reset all metrics"""
        self.attack_packets_sent = 0
        self.attack_bytes_sent = 0
        self.attack_packets_delivered = 0
        self.attack_bytes_delivered = 0
        self.normal_packets_sent = 0
        self.normal_bytes_sent = 0
        self.normal_packets_delivered = 0
        self.normal_packets_dropped = 0
        self.normal_bytes_delivered = 0
        self.normal_bytes_dropped = 0
        self.cost_history = []
        self.loss_rate_history = []
        self.attack_rate_history = []
        # Keep baseline if set
    
    def print_summary(self, elapsed_time: float):
        """Print formatted attack cost summary"""
        summary = self.get_summary(elapsed_time)
        
        print("\n" + "="*60)
        print("ATTACK COST ANALYSIS")
        print("="*60)
        
        print("\n--- Attack Traffic ---")
        atk = summary["attack_traffic"]
        print(f"  Packets Sent: {atk['packets_sent']:,}")
        print(f"  Bytes Sent: {atk['bytes_sent']:,}")
        print(f"  Rate: {atk['rate_mbps']:.4f} Mbps ({atk['rate_pps']:.2f} pps)")
        
        print("\n--- Normal Traffic Impact ---")
        norm = summary["normal_traffic"]
        print(f"  Packets Sent: {norm['packets_sent']:,}")
        print(f"  Packets Delivered: {norm['packets_delivered']:,}")
        print(f"  Packets Dropped: {norm['packets_dropped']:,}")
        print(f"  Loss Rate: {norm['loss_rate']:.4%}")
        
        print("\n--- Cost Metrics ---")
        cost = summary["cost_metrics"]
        print(f"  Baseline Loss Rate: {cost['baseline_loss_rate']:.4%}")
        print(f"  Attack-Induced Loss Rate: {cost['induced_loss_rate']:.4%}")
        
        attack_cost = cost['attack_cost']
        if attack_cost == float('inf'):
            print(f"  Attack Cost: ∞ (attack ineffective)")
        else:
            print(f"  Attack Cost: {attack_cost:.4f} Mbps per unit loss")
        
        normalized_cost = cost['normalized_cost_10pct']
        if normalized_cost == float('inf'):
            print(f"  Traffic for 10% Loss: ∞ (cannot estimate)")
        else:
            print(f"  Traffic for 10% Loss: {normalized_cost:.4f} Mbps")
        
        print(f"  Cost Efficiency Ratio: {cost['cost_efficiency_ratio']:.6f}")
        print(f"  Damage per Mbps: {cost['damage_per_mbps']:.6f}")
        
        print("\n" + "="*60)


def compare_attack_costs(
    results_dict: Dict[str, Dict],
    target_loss_rate: float = 0.1
) -> pd.DataFrame:
    """
    Compare attack costs across different routing algorithms
    
    Args:
        results_dict: Dictionary mapping algorithm name to its results
                     Each result should contain 'cost_calculator' summary and 'throughput_percentiles'
        target_loss_rate: Target loss rate for normalized comparison
        
    Returns:
        DataFrame comparing costs across algorithms
    """
    comparison_data = []
    
    for algo_name, results in results_dict.items():
        if 'attack_cost' in results:
            cost_data = results['attack_cost']
            row = {
                'algorithm': algo_name,
                'attack_rate_mbps': cost_data['attack_traffic']['rate_mbps'],
                'normal_loss_rate': cost_data['normal_traffic']['loss_rate'],
                'induced_loss_rate': cost_data['cost_metrics']['induced_loss_rate'],
                'attack_cost': cost_data['cost_metrics']['attack_cost'],
                f'traffic_for_{int(target_loss_rate*100)}pct_loss': 
                    cost_data['cost_metrics'].get('normalized_cost_10pct', float('inf')),
                'damage_per_mbps': cost_data['cost_metrics']['damage_per_mbps'],
            }
            
            # Add 5th percentile throughput if available
            if 'throughput_percentiles' in results:
                tp = results['throughput_percentiles']
                row['p5_throughput_pps'] = tp.get('p5_pps', 0.0)
                row['p5_throughput_mbps'] = tp.get('p5_mbps', 0.0)
                row['avg_throughput_pps'] = tp.get('avg_pps', 0.0)
                row['avg_throughput_mbps'] = tp.get('avg_mbps', 0.0)
            
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by attack cost (higher is better)
    if not df.empty:
        df = df.sort_values('attack_cost', ascending=False)
    
    return df


def compare_throughput_percentiles(
    results_dict: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Compare throughput percentiles across different routing algorithms
    
    This is useful for evaluating worst-case performance under DDoS attacks.
    Higher 5th percentile throughput indicates better performance during attack periods.
    
    Args:
        results_dict: Dictionary mapping algorithm name to its results
                     Each result should contain 'throughput_percentiles'
        
    Returns:
        DataFrame comparing throughput percentiles across algorithms
    """
    comparison_data = []
    
    for algo_name, results in results_dict.items():
        if 'throughput_percentiles' in results:
            tp = results['throughput_percentiles']
            comparison_data.append({
                'algorithm': algo_name,
                'p5_pps': tp.get('p5_pps', 0.0),
                'p5_mbps': tp.get('p5_mbps', 0.0),
                'p10_pps': tp.get('p10_pps', 0.0),
                'p10_mbps': tp.get('p10_mbps', 0.0),
                'p50_pps': tp.get('p50_pps', 0.0),
                'p50_mbps': tp.get('p50_mbps', 0.0),
                'avg_pps': tp.get('avg_pps', 0.0),
                'avg_mbps': tp.get('avg_mbps', 0.0),
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by 5th percentile throughput (higher is better for defense)
    if not df.empty:
        df = df.sort_values('p5_pps', ascending=False)
    
    return df


def print_algorithm_comparison(
    results_dict: Dict[str, Dict],
    target_loss_rate: float = 0.1
):
    """
    Print a formatted comparison of routing algorithms
    
    Args:
        results_dict: Dictionary mapping algorithm name to its results
        target_loss_rate: Target loss rate for normalized comparison
    """
    print("\n" + "="*80)
    print("ROUTING ALGORITHM COMPARISON")
    print("="*80)
    
    # Attack cost comparison
    cost_df = compare_attack_costs(results_dict, target_loss_rate)
    
    if not cost_df.empty:
        print("\n--- Attack Cost Comparison (Higher is Better for Defense) ---")
        print(f"{'Algorithm':<25} {'Attack Cost':>15} {'Loss Rate':>12} {'P5 Throughput':>15}")
        print("-" * 70)
        
        for _, row in cost_df.iterrows():
            algo = row['algorithm']
            cost = row['attack_cost']
            loss_rate = row['normal_loss_rate']
            p5_tp = row.get('p5_throughput_pps', 0.0)
            
            cost_str = f"{cost:.2f}" if cost != float('inf') else "∞"
            print(f"{algo:<25} {cost_str:>15} {loss_rate:>11.2%} {p5_tp:>14.2f} pps")
    
    # Throughput percentile comparison
    tp_df = compare_throughput_percentiles(results_dict)
    
    if not tp_df.empty:
        print("\n--- 5th Percentile Throughput Comparison (Higher is Better) ---")
        print(f"{'Algorithm':<25} {'P5 (pps)':>12} {'P5 (Mbps)':>12} {'Avg (pps)':>12} {'Avg (Mbps)':>12}")
        print("-" * 75)
        
        for _, row in tp_df.iterrows():
            algo = row['algorithm']
            print(f"{algo:<25} {row['p5_pps']:>12.2f} {row['p5_mbps']:>12.4f} {row['avg_pps']:>12.2f} {row['avg_mbps']:>12.4f}")
    
    print("\n" + "="*80)
