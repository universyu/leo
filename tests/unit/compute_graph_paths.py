#!/usr/bin/env python3
"""
Compute k=3 paths for all four routing algorithms (KSP, KDS, KDG, KLO)
for a representative GS pair, and save the results to JSON for graph visualization.
"""

import os
import sys
import json
import time

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from leo_network import LEOConstellation
from leo_network.core.routing import (
    KShortestPathsRouter,
    KDSRouter,
    KDGRouter,
    KLORouter,
)


def main():
    print("=" * 60)
    print("  Computing Graph Paths for Visualization")
    print("=" * 60)

    # Build constellation (same config as simulations)
    print("\n  Building constellation...")
    constellation = LEOConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0,
        inclination_deg=53.0,
        isl_bandwidth_mbps=100.0,
    )

    # Add ground stations (same as simulation)
    ground_stations = {
        "GS_NewYork": (40.7128, -74.0060),
        "GS_London": (51.5074, -0.1278),
        "GS_Tokyo": (35.6762, 139.6503),
        "GS_Sydney": (-33.8688, 151.2093),
        "GS_MexicoCity": (19.4326, -99.1332),
        "GS_Moscow": (55.7558, 37.6173),
        "GS_Dubai": (25.2048, 55.2708),
        "GS_Singapore": (1.3521, 103.8198),
        "GS_SaoPaulo": (-23.5505, -46.6333),
        "GS_Mumbai": (19.0760, 72.8777),
        "GS_Beijing": (39.9042, 116.4074),
        "GS_Cairo": (30.0444, 31.2357),
        "GS_Istanbul": (41.0082, 28.9784),
        "GS_Paris": (48.8566, 2.3522),
        "GS_Berlin": (52.5200, 13.4050),
        "GS_Rome": (41.9028, 12.4964),
        "GS_Madrid": (40.4168, -3.7038),
        "GS_Stockholm": (59.3293, 18.0686),
        "GS_Delhi": (28.6139, 77.2090),
        "GS_Bangkok": (13.7563, 100.5018),
        "GS_Seoul": (37.5665, 126.9780),
        "GS_Toronto": (43.6532, -79.3832),
        "GS_Chicago": (41.8781, -87.6298),
        "GS_LosAngeles": (34.0522, -118.2437),
        "GS_Miami": (25.7617, -80.1918),
        "GS_Houston": (29.7604, -95.3698),
        "GS_Seattle": (47.6062, -122.3321),
        "GS_Johannesburg": (-26.2041, 28.0473),
        "GS_Nairobi": (-1.2921, 36.8219),
        "GS_Lima": (-12.0464, -77.0428),
        "GS_Santiago": (-33.4489, -70.6693),
        "GS_Auckland": (-36.8485, 174.7633),
        "GS_Bogota": (4.7110, -74.0721),
        "GS_TelAviv": (32.0853, 34.7818),
    }

    for gs_name, (lat, lon) in ground_stations.items():
        constellation.add_ground_station(
            gs_id=gs_name, latitude=lat, longitude=lon
        )

    print(f"  Satellites: {len(constellation.satellites)}")
    print(f"  Ground stations: {len(constellation.ground_stations)}")
    print(f"  Links: {len(constellation.links)}")

    # Target ISL
    target_src = "SAT_4_2"
    target_dst = "SAT_4_3"

    # Choose a representative GS pair that goes through the target ISL
    # (MexicoCity -> Istanbul is one of the highest-traffic pairs)
    src_gs = "GS_MexicoCity"
    dst_gs = "GS_Istanbul"

    # Create routers
    k = 3
    routers = {
        "KSP": KShortestPathsRouter(constellation, k=k),
        "KDS": KDSRouter(constellation, k=k, disjoint_type="link"),
        "KDG": KDGRouter(constellation, k=k, diversity_weight=0.5),
        "KLO": KLORouter(constellation, k=k, load_threshold=0.7),
    }

    results = {
        "constellation": {
            "num_planes": 6,
            "sats_per_plane": 11,
            "total_sats": 66,
        },
        "target_isl": {
            "source": target_src,
            "target": target_dst,
        },
        "gs_pair": {
            "source": src_gs,
            "destination": dst_gs,
        },
        "algorithms": {},
    }

    # Collect satellite positions for layout
    sat_positions = {}
    for sat_id, sat in constellation.satellites.items():
        sat_positions[sat_id] = {
            "plane": sat.plane_id,
            "index": sat.sat_id,
            "lat": sat.position[0],
            "lon": sat.position[1],
        }
    results["sat_positions"] = sat_positions

    # Collect all ISL edges (for drawing the grid)
    all_isls = []
    seen = set()
    for link_id, link in constellation.links.items():
        if link.source.startswith("SAT_") and link.target.startswith("SAT_"):
            key = (min(link.source, link.target), max(link.source, link.target))
            if key not in seen:
                seen.add(key)
                all_isls.append(list(key))
    results["all_isls"] = all_isls

    # Collect GSL edges for the source and destination GS
    gs_connections = {}
    for gs_name in [src_gs, dst_gs]:
        gs = constellation.ground_stations[gs_name]
        gs_connections[gs_name] = {
            "connected_sats": list(gs.connected_sats),
            "lat": gs.position[0],
            "lon": gs.position[1],
        }
    results["gs_connections"] = gs_connections

    # Compute paths for each algorithm
    for algo_name, router in routers.items():
        print(f"\n  Computing {algo_name} paths: {src_gs} -> {dst_gs} ...")
        t0 = time.time()

        if algo_name == "KSP":
            paths = router.compute_k_paths(src_gs, dst_gs)
        elif algo_name == "KDS":
            paths = router.compute_k_disjoint_paths(src_gs, dst_gs)
        elif algo_name == "KDG":
            paths = router.compute_k_geodiverse_paths(src_gs, dst_gs)
        elif algo_name == "KLO":
            paths = router.get_all_disjoint_paths(src_gs, dst_gs)
        else:
            paths = []

        elapsed = time.time() - t0

        # Analyze each path
        path_data = []
        for i, path in enumerate(paths):
            # Extract ISL edges on this path
            edges = []
            goes_through_target = False
            for j in range(len(path) - 1):
                a, b = path[j], path[j + 1]
                edges.append([a, b])
                if (a == target_src and b == target_dst) or (a == target_dst and b == target_src):
                    goes_through_target = True

            path_data.append({
                "path_index": i,
                "nodes": path,
                "edges": edges,
                "hop_count": len(path) - 1,
                "goes_through_target": goes_through_target,
            })

        # Count paths through target
        paths_through = sum(1 for p in path_data if p["goes_through_target"])

        results["algorithms"][algo_name] = {
            "num_paths": len(path_data),
            "paths_through_target": paths_through,
            "compute_time_s": round(elapsed, 4),
            "paths": path_data,
        }

        print(f"    Found {len(path_data)} paths ({paths_through} through target ISL)")
        print(f"    Compute time: {elapsed:.4f}s")
        for i, p in enumerate(path_data):
            through_str = " *** THROUGH TARGET ***" if p["goes_through_target"] else ""
            print(f"    Path {i}: {p['hop_count']} hops{through_str}")

    # Save to JSON
    output_path = os.path.join(PROJECT_ROOT, "output", "graph_paths_data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
