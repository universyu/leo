#!/usr/bin/env python3
"""
Example: DDoS Attack Simulation

This script demonstrates various DDoS attack scenarios in LEO satellite networks:
1. Flooding attack
2. Reflection/Amplification attack
3. Pulsing attack
4. Coordinated multi-vector attack
5. Bottleneck-targeted attack

For each attack type, we measure the impact on normal traffic.
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np

from leo_network import (
    LEOConstellation,
    TrafficGenerator,
    ShortestPathRouter,
    Simulator,
    DDoSAttackGenerator,
    AttackType,
    AttackStrategy
)
from leo_network.core.routing import create_router


def run_attack_scenario(
    name: str,
    constellation: LEOConstellation,
    attack_generator: DDoSAttackGenerator,
    attack_func,
    attack_kwargs: dict,
    baseline_results: dict
) -> dict:
    """Run a single attack scenario and collect metrics"""
    print(f"\n{'='*60}")
    print(f"Attack Scenario: {name}")
    print(f"{'='*60}")
    
    # Create fresh simulator
    router = ShortestPathRouter(constellation)
    sim = Simulator(
        constellation=constellation,
        router=router,
        time_step=0.001,
        seed=42
    )
    
    # Recreate attack generator with new traffic generator
    attack_gen = DDoSAttackGenerator(
        constellation=constellation,
        traffic_generator=sim.traffic_generator,
        seed=42
    )
    
    # Add normal traffic (same as baseline)
    sim.add_random_normal_flows(num_flows=20, rate_range=(50, 200))
    
    # Add ground station flows
    if "GS_Beijing" in constellation.ground_stations:
        sim.add_normal_traffic("GS_Beijing", "GS_NewYork", rate=100)
        sim.add_normal_traffic("GS_London", "GS_Sydney", rate=100)
    
    # Launch attack
    print(f"Launching attack...")
    attack_id = attack_func(**attack_kwargs)
    print(f"  Attack ID: {attack_id}")
    
    # Get attack config info
    if attack_id in attack_gen.active_attacks:
        config = attack_gen.active_attacks[attack_id]
        print(f"  Attack Type: {config.attack_type.value}")
        print(f"  Attackers: {config.num_attackers}")
        print(f"  Total Rate: {config.total_rate} pps")
    
    # Run simulation
    print("Running simulation...")
    sim.run(duration=1.0, progress_bar=True)
    
    # Get results
    results = sim.get_results()
    stats = results["statistics"]
    
    # Calculate impact compared to baseline
    baseline_delivery = baseline_results["statistics"]["overview"]["delivery_rate"]
    current_delivery = stats["overview"]["delivery_rate"]
    delivery_impact = (baseline_delivery - current_delivery) / baseline_delivery * 100
    
    baseline_delay = baseline_results["statistics"]["delay"]["avg_ms"]
    current_delay = stats["delay"]["avg_ms"]
    delay_increase = (current_delay - baseline_delay) / baseline_delay * 100 if baseline_delay > 0 else 0
    
    print(f"\n--- Impact Analysis ---")
    print(f"  Delivery Rate: {current_delivery:.4f} (baseline: {baseline_delivery:.4f})")
    print(f"  Impact: {delivery_impact:.2f}% decrease")
    print(f"  Avg Delay: {current_delay:.2f} ms (baseline: {baseline_delay:.2f} ms)")
    print(f"  Delay Increase: {delay_increase:.2f}%")
    
    # Normal traffic specific stats
    normal_stats = stats["normal_traffic"]
    print(f"\n--- Normal Traffic ---")
    print(f"  Delivered: {normal_stats['delivered']}")
    print(f"  Dropped: {normal_stats['dropped']}")
    print(f"  Delivery Rate: {normal_stats['delivery_rate']:.4f}")
    
    # Attack traffic stats
    attack_stats = stats["attack_traffic"]
    print(f"\n--- Attack Traffic ---")
    print(f"  Delivered: {attack_stats['delivered']}")
    print(f"  Dropped: {attack_stats['dropped']}")
    print(f"  Delivery Rate: {attack_stats['delivery_rate']:.4f}")
    
    results["impact"] = {
        "delivery_impact_percent": delivery_impact,
        "delay_increase_percent": delay_increase,
        "normal_delivery_rate": normal_stats["delivery_rate"],
        "attack_delivery_rate": attack_stats["delivery_rate"]
    }
    
    return results


def main():
    print("="*70)
    print("DDoS Attack Simulation in LEO Satellite Network")
    print("="*70)
    
    # Create constellation
    # Using lower bandwidth (100 Mbps) to better demonstrate DDoS attack effects
    # In real scenarios, ISL bandwidth might be 10 Gbps, but attacks would also be stronger
    print("\n[1] Creating LEO Constellation...")
    constellation = LEOConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0,
        inclination_deg=53.0,
        isl_bandwidth_mbps=100.0  # Lower bandwidth to demonstrate congestion
    )
    
    # Add ground stations
    ground_stations = [
        ("GS_Beijing", 39.9, 116.4),
        ("GS_NewYork", 40.7, -74.0),
        ("GS_London", 51.5, -0.1),
        ("GS_Sydney", -33.9, 151.2),
    ]
    for gs_id, lat, lon in ground_stations:
        constellation.add_ground_station(gs_id, lat, lon)
    
    print(f"  Satellites: {len(constellation.satellites)}")
    print(f"  Links: {len(constellation.links)}")
    print(f"  Ground Stations: {len(constellation.ground_stations)}")
    
    # Run baseline (no attack)
    print("\n[2] Running Baseline (No Attack)...")
    baseline_sim = Simulator(
        constellation=constellation,
        router=ShortestPathRouter(constellation),
        time_step=0.001,
        seed=42
    )
    baseline_sim.add_random_normal_flows(num_flows=20, rate_range=(50, 200))
    baseline_sim.add_normal_traffic("GS_Beijing", "GS_NewYork", rate=100)
    baseline_sim.add_normal_traffic("GS_London", "GS_Sydney", rate=100)
    baseline_sim.run(duration=1.0, progress_bar=True)
    baseline_results = baseline_sim.get_results()
    
    print(f"  Baseline Delivery Rate: {baseline_results['statistics']['overview']['delivery_rate']:.4f}")
    print(f"  Baseline Avg Delay: {baseline_results['statistics']['delay']['avg_ms']:.2f} ms")
    
    # Define attack scenarios
    print("\n[3] Running Attack Scenarios...")
    
    # Select some targets
    target_sats = ["SAT_2_5", "SAT_3_5"]  # Central satellites
    
    all_results = {"baseline": baseline_results}
    
    # Scenario 1: Flooding Attack
    def create_flooding():
        attack_gen = DDoSAttackGenerator(
            constellation=constellation,
            traffic_generator=TrafficGenerator(seed=42),
            seed=42
        )
        return attack_gen.create_flooding_attack(
            targets=target_sats,
            num_attackers=15,
            total_rate=8000.0,
            start_time=0.0,
            duration=-1,
            strategy=AttackStrategy.DISTRIBUTED
        )
    
    # We need a different approach - create attack directly in simulator
    # Let's create a helper function
    
    # Attack scenarios with increased intensity to demonstrate congestion
    # With 100 Mbps links, we need significant traffic to cause drops
    # 50000 pps * 1000 bytes = 400 Mbps attack bandwidth
    scenarios = [
        {
            "name": "Medium Flooding (30 attackers, 50000 pps)",
            "attack_type": "flooding",
            "params": {
                "targets": target_sats,
                "num_attackers": 30,
                "total_rate": 50000.0,
                "strategy": AttackStrategy.DISTRIBUTED
            }
        },
        {
            "name": "Pulsing Attack (20 attackers, 80000 pps peak)",
            "attack_type": "pulsing", 
            "params": {
                "targets": target_sats,
                "num_attackers": 20,
                "peak_rate": 80000.0,
                "pulse_on_time": 0.05,
                "pulse_off_time": 0.05
            }
        },
        {
            "name": "Reflection Attack (10 attackers, 10x amplification)",
            "attack_type": "reflection",
            "params": {
                "targets": target_sats,
                "num_attackers": 10,
                "total_rate": 5000.0,
                "amplification_factor": 10.0
            }
        },
        {
            "name": "High-Volume Flooding (50 attackers, 100000 pps)",
            "attack_type": "flooding",
            "params": {
                "targets": target_sats,
                "num_attackers": 50,
                "total_rate": 100000.0,
                "strategy": AttackStrategy.DISTRIBUTED
            }
        }
    ]
    
    for scenario in scenarios:
        # Create fresh simulator
        router = ShortestPathRouter(constellation)
        sim = Simulator(
            constellation=constellation,
            router=router,
            time_step=0.001,
            seed=42
        )
        
        # Create attack generator with simulator's traffic generator
        attack_gen = DDoSAttackGenerator(
            constellation=constellation,
            traffic_generator=sim.traffic_generator,
            seed=42
        )
        
        # Add normal traffic
        sim.add_random_normal_flows(num_flows=20, rate_range=(50, 200))
        sim.add_normal_traffic("GS_Beijing", "GS_NewYork", rate=100)
        sim.add_normal_traffic("GS_London", "GS_Sydney", rate=100)
        
        # Launch attack
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*60}")
        
        if scenario["attack_type"] == "flooding":
            attack_id = attack_gen.create_flooding_attack(**scenario["params"])
        elif scenario["attack_type"] == "pulsing":
            attack_id = attack_gen.create_pulsing_attack(**scenario["params"])
        elif scenario["attack_type"] == "reflection":
            attack_id = attack_gen.create_reflection_attack(**scenario["params"])
        
        print(f"  Attack ID: {attack_id}")
        print(f"  Attack Flows Created: {len(attack_gen.attack_flows.get(attack_id, []))}")
        
        # Run simulation
        sim.run(duration=1.0, progress_bar=True)
        
        # Get results
        results = sim.get_results()
        stats = results["statistics"]
        
        # Calculate impact
        baseline_delivery = baseline_results["statistics"]["overview"]["delivery_rate"]
        current_delivery = stats["overview"]["delivery_rate"]
        delivery_impact = (baseline_delivery - current_delivery) / baseline_delivery * 100
        
        baseline_delay = baseline_results["statistics"]["delay"]["avg_ms"]
        current_delay = stats["delay"]["avg_ms"]
        delay_increase = (current_delay - baseline_delay) / baseline_delay * 100 if baseline_delay > 0 else 0
        
        print(f"\n--- Results ---")
        print(f"  Overall Delivery Rate: {current_delivery:.4f}")
        print(f"  Impact on Delivery: {delivery_impact:+.2f}%")
        print(f"  Avg Delay: {current_delay:.2f} ms")
        print(f"  Delay Change: {delay_increase:+.2f}%")
        
        normal_stats = stats["normal_traffic"]
        attack_stats = stats["attack_traffic"]
        print(f"  Normal Traffic Delivery: {normal_stats['delivery_rate']:.4f}")
        print(f"  Attack Traffic Delivery: {attack_stats['delivery_rate']:.4f}")
        
        results["impact"] = {
            "delivery_impact_percent": delivery_impact,
            "delay_increase_percent": delay_increase
        }
        
        all_results[scenario["name"]] = results
    
    # Generate comparison plot
    print("\n[4] Generating Comparison Plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract data for plotting
    scenario_names = ["Baseline"] + [s["name"].split("(")[0].strip() for s in scenarios]
    delivery_rates = [baseline_results["statistics"]["overview"]["delivery_rate"]]
    avg_delays = [baseline_results["statistics"]["delay"]["avg_ms"]]
    normal_delivery = [baseline_results["statistics"]["normal_traffic"]["delivery_rate"]]
    
    for scenario in scenarios:
        name = scenario["name"]
        if name in all_results:
            stats = all_results[name]["statistics"]
            delivery_rates.append(stats["overview"]["delivery_rate"])
            avg_delays.append(stats["delay"]["avg_ms"])
            normal_delivery.append(stats["normal_traffic"]["delivery_rate"])
    
    x = np.arange(len(scenario_names))
    
    # Plot 1: Overall Delivery Rate
    ax = axes[0]
    colors = ['green'] + ['red'] * len(scenarios)
    bars = ax.bar(x, delivery_rates, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel("Delivery Rate")
    ax.set_title("Overall Packet Delivery Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=delivery_rates[0], color='green', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average Delay
    ax = axes[1]
    bars = ax.bar(x, avg_delays, color=['green'] + ['orange'] * len(scenarios), 
                  alpha=0.7, edgecolor='black')
    ax.set_ylabel("Delay (ms)")
    ax.set_title("Average End-to-End Delay")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=avg_delays[0], color='green', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Normal Traffic Delivery
    ax = axes[2]
    bars = ax.bar(x, normal_delivery, color=['green'] + ['purple'] * len(scenarios),
                  alpha=0.7, edgecolor='black')
    ax.set_ylabel("Delivery Rate")
    ax.set_title("Normal Traffic Delivery Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=normal_delivery[0], color='green', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle("DDoS Attack Impact Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs("output", exist_ok=True)
    fig.savefig("output/ddos_attack_comparison.png", dpi=150)
    print("  Saved: output/ddos_attack_comparison.png")
    
    plt.close('all')
    
    # Print summary table
    print("\n" + "="*70)
    print("ATTACK IMPACT SUMMARY")
    print("="*70)
    print(f"{'Scenario':<40} {'Delivery':<12} {'Delay':<12} {'Impact':<12}")
    print("-"*70)
    print(f"{'Baseline':<40} {delivery_rates[0]:<12.4f} {avg_delays[0]:<12.2f} {'N/A':<12}")
    
    for i, scenario in enumerate(scenarios):
        name = scenario["name"].split("(")[0].strip()
        if scenario["name"] in all_results:
            dr = delivery_rates[i+1]
            delay = avg_delays[i+1]
            impact = all_results[scenario["name"]]["impact"]["delivery_impact_percent"]
            print(f"{name:<40} {dr:<12.4f} {delay:<12.2f} {impact:+.2f}%")
    
    print("="*70)
    print("\nDDoS Attack Simulation Complete!")


if __name__ == "__main__":
    main()
