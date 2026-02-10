#!/usr/bin/env python3
"""
Example: Basic LEO Network Simulation

This script demonstrates the basic usage of the LEO satellite network
simulation framework, including:
- Creating a constellation topology
- Adding traffic flows
- Running simulation with shortest path routing
- Analyzing results
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leo_network import (
    LEOConstellation,
    TrafficGenerator,
    TrafficPattern,
    ShortestPathRouter, 
    KShortestPathsRouter,
    ECMPRouter,
    LoadAwareRouter,
    RandomizedRouter,
    StatisticsCollector,
    Simulator
)
from leo_network.core.routing import create_router
from leo_network.core.visualization import (
    plot_constellation_2d,
    plot_simulation_results,
    plot_link_utilization_heatmap
)
import matplotlib.pyplot as plt


def main():
    print("="*70)
    print("LEO Satellite Network Simulation - Basic Example")
    print("="*70)
    
    # =====================================================
    # Step 1: Create LEO Constellation
    # =====================================================
    print("\n[1] Creating LEO Constellation...")
    
    # Create a small constellation for quick simulation
    # Parameters similar to Starlink (simplified)
    constellation = LEOConstellation(
        num_planes=6,           # Number of orbital planes
        sats_per_plane=11,      # Satellites per plane
        altitude_km=550.0,      # Orbital altitude
        inclination_deg=53.0,   # Orbital inclination
        isl_bandwidth_mbps=10000.0,  # 10 Gbps ISL bandwidth
        sat_capacity=10000,     # Processing capacity (pps)
        sat_buffer_size=1000    # Buffer size (packets)
    )
    
    print(f"  Created: {constellation}")
    print(f"  Total satellites: {len(constellation.satellites)}")
    print(f"  Total ISL links: {len(constellation.links)}")
    
    # Add some ground stations
    print("\n[2] Adding Ground Stations...")
    
    ground_stations = [
        ("GS_Beijing", 39.9, 116.4),
        ("GS_NewYork", 40.7, -74.0),
        ("GS_London", 51.5, -0.1),
        ("GS_Sydney", -33.9, 151.2),
    ]
    
    for gs_id, lat, lon in ground_stations:
        constellation.add_ground_station(gs_id, lat, lon)
        print(f"  Added {gs_id} at ({lat}, {lon})")
    
    # =====================================================
    # Step 2: Create Router
    # =====================================================
    print("\n[3] Creating Router...")
    
    # Use shortest path router for baseline
    router = ShortestPathRouter(constellation)
    print(f"  Router: {router.name}")
    
    # =====================================================
    # Step 3: Create Simulator
    # =====================================================
    print("\n[4] Creating Simulator...")
    
    simulator = Simulator(
        constellation=constellation,
        router=router,
        time_step=0.001,  # 1ms time step
        seed=42           # For reproducibility
    )
    
    # =====================================================
    # Step 4: Add Traffic Flows
    # =====================================================
    print("\n[5] Adding Traffic Flows...")
    
    # Add random normal traffic flows between satellites
    print("  Adding 20 random normal flows...")
    simulator.add_random_normal_flows(
        num_flows=20,
        rate_range=(50, 200),  # 50-200 packets/s per flow
        packet_size=1000
    )
    
    # Add some specific flows between ground stations
    print("  Adding ground station flows...")
    simulator.add_normal_traffic(
        source="GS_Beijing",
        destination="GS_NewYork",
        rate=100,
        packet_size=1000
    )
    simulator.add_normal_traffic(
        source="GS_London",
        destination="GS_Sydney",
        rate=100,
        packet_size=1000
    )
    
    print(f"  Total flows: {len(simulator.traffic_generator.flows)}")
    
    # =====================================================
    # Step 5: Run Simulation
    # =====================================================
    print("\n[6] Running Simulation...")
    
    simulation_duration = 1.0  # 1 second
    stats = simulator.run(
        duration=simulation_duration,
        progress_bar=True
    )
    
    # =====================================================
    # Step 6: Analyze Results
    # =====================================================
    print("\n[7] Simulation Results:")
    simulator.print_results()
    
    # =====================================================
    # Step 7: Visualization
    # =====================================================
    print("\n[8] Generating Visualizations...")
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot constellation
    fig1 = plot_constellation_2d(
        constellation,
        show_links=True,
        title="LEO Satellite Constellation (6 planes × 11 satellites)"
    )
    fig1.savefig(os.path.join(output_dir, "constellation_2d.png"), dpi=150)
    print(f"  Saved: {output_dir}/constellation_2d.png")
    
    # Plot simulation results
    fig2 = plot_simulation_results(
        stats,
        title="Simulation Results - Shortest Path Routing"
    )
    fig2.savefig(os.path.join(output_dir, "results.png"), dpi=150)
    print(f"  Saved: {output_dir}/results.png")
    
    # Plot link utilization heatmap
    fig3 = plot_link_utilization_heatmap(
        constellation, stats,
        title="Link Utilization Heatmap"
    )
    fig3.savefig(os.path.join(output_dir, "utilization.png"), dpi=150)
    print(f"  Saved: {output_dir}/utilization.png")
    
    # Save statistics to JSON
    stats.save_to_json(os.path.join(output_dir, "statistics.json"))
    print(f"  Saved: {output_dir}/statistics.json")
    
    print("\n" + "="*70)
    print("Simulation Complete!")
    print("="*70)
    
    # Close figures to free memory
    plt.close('all')
    
    # Optionally show plots (uncomment for interactive use)
    # plt.show()
    
    return simulator


if __name__ == "__main__":
    main()
