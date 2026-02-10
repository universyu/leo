#!/usr/bin/env python3
"""
Example: Compare Different Routing Algorithms

This script compares the performance of different routing algorithms
under normal traffic conditions.
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from leo_network import LEOConstellation, Simulator
from leo_network.core.routing import create_router
from leo_network.core.visualization import plot_comparison
import matplotlib.pyplot as plt


def run_with_router(router_type: str, constellation: LEOConstellation, **kwargs):
    """Run simulation with specific router type"""
    # Create fresh router
    router = create_router(router_type, constellation, **kwargs)
    
    # Create simulator
    sim = Simulator(
        constellation=constellation,
        router=router,
        time_step=0.001,
        seed=42
    )
    
    # Add same traffic pattern for fair comparison
    sim.add_random_normal_flows(num_flows=30, rate_range=(50, 200))
    
    # Run simulation
    sim.run(duration=1.0, progress_bar=False)
    
    return sim.get_results()


def main():
    print("="*70)
    print("Router Comparison Experiment")
    print("="*70)
    
    # Create constellation
    print("\nCreating constellation...")
    constellation = LEOConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0
    )
    print(f"  {constellation}")
    
    # Define routers to compare
    routers_config = [
        ("Shortest Path", "shortest", {}),
        ("K-SP (k=3)", "ksp", {"k": 3}),
        ("ECMP", "ecmp", {"max_paths": 4}),
        ("Load-Aware", "load_aware", {"k": 3, "load_weight": 0.5}),
        ("Randomized", "random", {"k": 5, "temperature": 1.0}),
    ]
    
    results = {}
    
    for name, router_type, kwargs in routers_config:
        print(f"\nRunning with {name}...")
        results[name] = run_with_router(router_type, constellation, **kwargs)
        
        # Print key metrics
        stats = results[name]["statistics"]
        print(f"  Delivery Rate: {stats['overview']['delivery_rate']:.4f}")
        print(f"  Avg Delay: {stats['delay']['avg_ms']:.2f} ms")
        print(f"  Throughput: {stats['throughput']['avg_pps']:.2f} pps")
    
    # Plot comparison
    print("\nGenerating comparison plot...")
    fig = plot_comparison(
        results,
        metrics=["delivery_rate", "avg_delay_ms", "throughput_pps"],
        title="Routing Algorithm Comparison (Normal Traffic)"
    )
    
    # Save and show
    os.makedirs("output", exist_ok=True)
    fig.savefig("output/router_comparison.png", dpi=150)
    print("Saved: output/router_comparison.png")
    
    plt.show()


if __name__ == "__main__":
    main()
