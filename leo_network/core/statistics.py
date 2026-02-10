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
    
    def take_snapshot(self, timestamp: float, additional_data: Optional[Dict] = None):
        """
        Take a snapshot of current statistics
        
        Args:
            timestamp: Current simulation time
            additional_data: Additional data to include in snapshot
        """
        snapshot = {
            "timestamp": timestamp,
            "packets_sent": self.packets_sent,
            "packets_delivered": self.packets_delivered,
            "packets_dropped": self.packets_dropped,
            "delivery_rate": self.get_delivery_rate(),
            "avg_delay": self.get_average_delay(),
            "throughput_pps": self.get_throughput_pps(timestamp),
        }
        
        if additional_data:
            snapshot.update(additional_data)
        
        self.snapshots.append(snapshot)
        
        # Record time series
        self.throughput_series.add(timestamp, snapshot["throughput_pps"])
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
        self.delay_series = TimeSeriesData()
        self.loss_rate_series = TimeSeriesData()
        self.link_utilization_series.clear()
        self.queue_occupancy_series.clear()
        self.flow_stats.clear()
        self.normal_packets_delivered = 0
        self.normal_packets_dropped = 0
        self.attack_packets_delivered = 0
        self.attack_packets_dropped = 0
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
