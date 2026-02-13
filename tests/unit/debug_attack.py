#!/usr/bin/env python3
"""
Debug/Test Script for DDoS Attack Simulation

This script tests the DDoS attack functionality with configurable parameters.
It demonstrates the impact of DDoS attacks on network performance.
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from leo_network import LEOConstellation, Simulator
from leo_network import DDoSAttackGenerator, AttackStrategy


def run_attack_test(
    isl_bandwidth_mbps: float = 100.0,
    num_attackers: int = 50,
    attack_rate: float = 100000.0,
    duration: float = 0.5
):
    """
    Run a DDoS attack test with configurable parameters.
    
    Args:
        isl_bandwidth_mbps: ISL link bandwidth in Mbps
        num_attackers: Number of attack sources
        attack_rate: Total attack rate in packets per second
        duration: Simulation duration in seconds
    """
    print("="*70)
    print("DDoS Attack Test")
    print("="*70)
    
    # Create constellation with specified bandwidth
    constellation = LEOConstellation(
        num_planes=6, 
        sats_per_plane=11,
        isl_bandwidth_mbps=isl_bandwidth_mbps
    )
    
    # Add globally distributed ground stations (paper-scale ~40 stations)
    constellation.add_global_ground_stations()
    
    print(f"\n--- Network Configuration ---")
    print(f"  Satellites: {len(constellation.satellites)}")
    print(f"  Links: {len(constellation.links)}")
    print(f"  ISL Bandwidth: {constellation.isl_bandwidth_mbps} Mbps")
    
    # Target satellites
    target_sats = ["SAT_2_5", "SAT_3_5"]
    
    # Create simulator
    sim = Simulator(constellation=constellation, time_step=0.001, seed=42)
    attack_gen = DDoSAttackGenerator(constellation, sim.traffic_generator, seed=42)
    
    # Add normal traffic (ground-station to ground-station per the paper)
    sim.add_random_normal_flows(num_flows=20, rate_range=(50, 200))
    
    # Create flooding attack
    attack_id = attack_gen.create_flooding_attack(
        targets=target_sats,
        num_attackers=num_attackers,
        total_rate=attack_rate,
        packet_size=1000,
        strategy=AttackStrategy.DISTRIBUTED
    )
    
    # Calculate expected attack bandwidth
    expected_attack_bw = attack_rate * 1000 * 8 / 1e6
    
    print(f"\n--- Attack Configuration ---")
    print(f"  Attack ID: {attack_id}")
    print(f"  Attackers: {num_attackers}")
    print(f"  Attack Rate: {attack_rate} pps")
    print(f"  Expected Bandwidth: {expected_attack_bw:.1f} Mbps")
    print(f"  Attack/Link ratio: {expected_attack_bw / isl_bandwidth_mbps:.1f}x")
    print(f"  Attack Flows: {len(attack_gen.attack_flows[attack_id])}")
    
    # Run simulation
    print(f"\n--- Running Simulation ({duration}s) ---")
    sim.run(duration=duration, progress_bar=True)
    
    # Get results
    results = sim.get_results()
    stats = results["statistics"]
    
    print(f"\n--- Results ---")
    print(f"  Overall Delivery: {stats['overview']['delivery_rate']:.4f}")
    print(f"  Normal Delivery:  {stats['normal_traffic']['delivery_rate']:.4f}")
    print(f"  Attack Delivery:  {stats['attack_traffic']['delivery_rate']:.4f}")
    print(f"  Normal Packets:   {stats['normal_traffic']['delivered']} delivered, {stats['normal_traffic']['dropped']} dropped")
    print(f"  Attack Packets:   {stats['attack_traffic']['delivered']} delivered, {stats['attack_traffic']['dropped']} dropped")
    
    # Link utilization
    link_utils = stats.get('link_utilization', {})
    if link_utils:
        print(f"\n--- Link Utilization ---")
        print(f"  Max: {max(link_utils.values()):.2%}")
        print(f"  Avg: {sum(link_utils.values())/len(link_utils):.2%}")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    # Run test with default parameters
    run_attack_test()
