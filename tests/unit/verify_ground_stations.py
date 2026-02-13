#!/usr/bin/env python3
"""
Quick verification script for ground station and traffic model changes.
Verifies:
1. Global ground stations are created correctly (~40 stations)
2. Normal traffic flows are ground-station to ground-station
3. Simulation runs without errors
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from leo_network import LEOConstellation, Simulator, KShortestPathsRouter


def main():
    print("=" * 70)
    print("Verification: Ground Station & Traffic Model")
    print("=" * 70)

    # 1. Create constellation
    constellation = LEOConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0,
        inclination_deg=53.0,
        isl_bandwidth_mbps=100.0,
    )
    print(f"\nSatellites: {len(constellation.satellites)}")
    print(f"Ground Stations before: {len(constellation.ground_stations)}")

    # 2. Add global ground stations
    gs_ids = constellation.add_global_ground_stations()
    print(f"Ground Stations after: {len(constellation.ground_stations)}")
    print(f"Added stations: {len(gs_ids)}")
    print(f"Station IDs: {gs_ids[:5]} ... (showing first 5)")
    print(f"Total links (ISL+GSL): {len(constellation.links)}")

    # 3. Verify each ground station has at least one connected satellite
    for gs_id, gs in constellation.ground_stations.items():
        if len(gs.connected_sats) == 0:
            print(f"  WARNING: {gs_id} has NO connected satellites!")
        else:
            pass  # OK
    connected_counts = [len(gs.connected_sats) for gs in constellation.ground_stations.values()]
    print(f"\nConnected satellites per GS: min={min(connected_counts)}, max={max(connected_counts)}, avg={sum(connected_counts)/len(connected_counts):.1f}")

    # 4. Create simulator and add flows
    router = KShortestPathsRouter(constellation)
    sim = Simulator(constellation=constellation, router=router, time_step=0.001, seed=42)
    sim.add_random_normal_flows(num_flows=20, rate_range=(50, 200))

    # 5. Verify all flows are GS-to-GS
    gs_keys = set(constellation.ground_stations.keys())
    all_gs_flows = True
    for flow_id, flow in sim.traffic_generator.flows.items():
        if flow.source not in gs_keys or flow.destination not in gs_keys:
            print(f"  ERROR: Flow {flow_id} is NOT GS-to-GS: {flow.source} -> {flow.destination}")
            all_gs_flows = False
    if all_gs_flows:
        print(f"\nAll {len(sim.traffic_generator.flows)} flows are ground-station to ground-station.")
    else:
        print("\nERROR: Some flows are NOT ground-station to ground-station!")

    # 6. Run quick simulation
    print("\nRunning quick simulation (0.2s)...")
    sim.run(duration=0.2, progress_bar=True)
    results = sim.get_results()
    stats = results["statistics"]
    print(f"  Delivery Rate: {stats['overview']['delivery_rate']:.4f}")
    print(f"  Avg Delay: {stats['delay']['avg_ms']:.2f} ms")
    print(f"  Normal Traffic Delivered: {stats['normal_traffic']['delivered']}")
    print(f"  Normal Traffic Dropped: {stats['normal_traffic']['dropped']}")

    print("\n" + "=" * 70)
    print("VERIFICATION PASSED!" if all_gs_flows else "VERIFICATION FAILED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
