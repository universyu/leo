"""
Compute Routing Table Data for Visualization (All 4 Algorithms)

This script computes the routing table for a representative ground station
(GS_MexicoCity) to all other ground stations using each of the four
routing algorithms: KSP, KDS, KDG, KLO.

Saves results to:
  output/routing_table_ksp.json
  output/routing_table_kds.json
  output/routing_table_kdg.json
  output/routing_table_klo.json
"""

import json
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from leo_network.core.topology import LEOConstellation
from leo_network.core.routing import (
    KShortestPathsRouter,
    KDSRouter,
    KDGRouter,
    KLORouter,
)


GROUND_STATIONS = {
    "GS_MexicoCity": (19.4326, -99.1332),
    "GS_Istanbul": (41.0082, 28.9784),
    "GS_London": (51.5074, -0.1278),
    "GS_Paris": (48.8566, 2.3522),
    "GS_Tokyo": (35.6762, 139.6503),
    "GS_Sydney": (-33.8688, 151.2093),
    "GS_NewYork": (40.7128, -74.0060),
    "GS_Chicago": (41.8781, -87.6298),
    "GS_Houston": (29.7604, -95.3698),
    "GS_Toronto": (43.6532, -79.3832),
    "GS_SaoPaulo": (-23.5505, -46.6333),
    "GS_BuenosAires": (-34.6037, -58.3816),
    "GS_Cairo": (30.0444, 31.2357),
    "GS_Dubai": (25.2048, 55.2708),
    "GS_Mumbai": (19.0760, 72.8777),
    "GS_Singapore": (1.3521, 103.8198),
    "GS_Seoul": (37.5665, 126.9780),
    "GS_Beijing": (39.9042, 116.4074),
    "GS_Moscow": (55.7558, 37.6173),
    "GS_Berlin": (52.5200, 13.4050),
    "GS_Rome": (41.9028, 12.4964),
    "GS_Madrid": (40.4168, -3.7038),
    "GS_Johannesburg": (-26.2041, 28.0473),
    "GS_Lagos": (6.5244, 3.3792),
    "GS_Nairobi": (-1.2921, 36.8219),
    "GS_Lima": (-12.0464, -77.0428),
    "GS_Bogota": (4.7110, -74.0721),
    "GS_LosAngeles": (34.0522, -118.2437),
    "GS_SanFrancisco": (37.7749, -122.4194),
    "GS_Denver": (39.7392, -104.9903),
    "GS_Miami": (25.7617, -80.1918),
    "GS_TelAviv": (32.0853, 34.7818),
}


def _get_k_paths(router, algo_name, src, dst):
    """Get k paths from a router using the algorithm-specific method."""
    if algo_name == "KSP":
        return router.compute_k_paths(src, dst)
    elif algo_name == "KDS":
        return router.compute_k_disjoint_paths(src, dst)
    elif algo_name == "KDG":
        return router.compute_k_geodiverse_paths(src, dst)
    elif algo_name == "KLO":
        return router.get_all_disjoint_paths(src, dst)
    return []


def compute_routing_table_for_algo(constellation, router, algo_name, algo_desc, src_gs):
    """Compute routing table for one algorithm and return the result dict."""
    print(f"\n  Computing {algo_name} ({algo_desc}) routing table from {src_gs}...")
    t0 = time.time()

    gs_connections = {}
    for gs_id, gs in constellation.ground_stations.items():
        gs_connections[gs_id] = {
            "lat": gs.position[0],
            "lon": gs.position[1],
            "connected_sats": list(gs.connected_sats),
        }

    routing_entries = []
    dst_list = sorted([gs for gs in GROUND_STATIONS if gs != src_gs])

    for dst_gs in dst_list:
        paths = _get_k_paths(router, algo_name, src_gs, dst_gs)
        if not paths:
            continue

        # Primary path (first / best)
        primary = paths[0]
        hop_count = len(primary) - 1
        delay_ms = router.get_path_delay(primary)
        next_hop = primary[1] if len(primary) > 1 else "N/A"

        # Link details for primary path
        links_detail = []
        for i in range(len(primary) - 1):
            src_node, dst_node = primary[i], primary[i + 1]
            link = constellation.get_link(src_node, dst_node)
            link_type = "GSL" if (src_node.startswith("GS_") or dst_node.startswith("GS_")) else "ISL"
            links_detail.append({
                "from": src_node,
                "to": dst_node,
                "type": link_type,
                "delay_ms": round(link.propagation_delay, 3) if link else 0,
            })

        entry = {
            "destination": dst_gs,
            "next_hop": next_hop,
            "path": primary,
            "hop_count": hop_count,
            "total_delay_ms": round(delay_ms, 3),
            "num_k_paths": len(paths),
            "links": links_detail,
        }

        # Alternative paths
        alt_paths = []
        for idx, p in enumerate(paths[1:], start=1):
            alt_paths.append({
                "path_index": idx,
                "nodes": p,
                "hop_count": len(p) - 1,
                "delay_ms": round(router.get_path_delay(p), 3),
                "next_hop": p[1] if len(p) > 1 else "N/A",
            })
        entry["alternative_paths"] = alt_paths
        routing_entries.append(entry)

    # Sort by delay
    routing_entries.sort(key=lambda e: e["total_delay_ms"])
    elapsed = time.time() - t0

    # Satellite positions
    sat_positions = {}
    for sat_id, sat in constellation.satellites.items():
        parts = sat_id.split("_")
        sat_positions[sat_id] = {
            "plane": int(parts[1]),
            "index": int(parts[2]),
            "lat": sat.position[0],
            "lon": sat.position[1],
        }

    n = len(routing_entries)
    result = {
        "algorithm": algo_name,
        "algorithm_desc": algo_desc,
        "source_gs": src_gs,
        "source_position": {
            "lat": GROUND_STATIONS[src_gs][0],
            "lon": GROUND_STATIONS[src_gs][1],
        },
        "constellation": {
            "num_planes": 6,
            "sats_per_plane": 11,
            "total_sats": 66,
            "total_gs": len(GROUND_STATIONS),
        },
        "gs_connections": gs_connections,
        "routing_table": routing_entries,
        "sat_positions": sat_positions,
        "summary": {
            "total_destinations": n,
            "avg_hop_count": round(sum(e["hop_count"] for e in routing_entries) / max(n, 1), 2),
            "avg_delay_ms": round(sum(e["total_delay_ms"] for e in routing_entries) / max(n, 1), 3),
            "min_delay_ms": routing_entries[0]["total_delay_ms"] if n else 0,
            "max_delay_ms": routing_entries[-1]["total_delay_ms"] if n else 0,
            "compute_time_s": round(elapsed, 4),
        },
    }

    print(f"    Destinations: {n}")
    print(f"    Avg hops: {result['summary']['avg_hop_count']}")
    print(f"    Avg delay: {result['summary']['avg_delay_ms']} ms")
    print(f"    Min / Max delay: {result['summary']['min_delay_ms']} / {result['summary']['max_delay_ms']} ms")
    print(f"    Compute time: {elapsed:.4f}s")

    return result


def main():
    print("=" * 60)
    print("  Computing Routing Tables for All 4 Algorithms")
    print("=" * 60)

    # Build constellation
    constellation = LEOConstellation(num_planes=6, sats_per_plane=11)
    for gs_name, (lat, lon) in GROUND_STATIONS.items():
        constellation.add_ground_station(gs_id=gs_name, latitude=lat, longitude=lon)

    print(f"  Satellites: {len(constellation.satellites)}")
    print(f"  Ground stations: {len(constellation.ground_stations)}")

    src_gs = "GS_MexicoCity"
    output_dir = os.path.join(os.path.dirname(__file__), "../../output")
    os.makedirs(output_dir, exist_ok=True)

    # Define all 4 algorithms
    algorithms = [
        ("KSP", "K-Shortest Paths",
         KShortestPathsRouter(constellation, k=3)),
        ("KDS", "K-Disjoint Shortest",
         KDSRouter(constellation, k=3, disjoint_type="link")),
        ("KDG", "K-Disjoint Geodiverse",
         KDGRouter(constellation, k=3, diversity_weight=0.5)),
        ("KLO", "K-Limited-Overlap",
         KLORouter(constellation, k=3, load_threshold=0.7)),
    ]

    for algo_name, algo_desc, router in algorithms:
        result = compute_routing_table_for_algo(
            constellation, router, algo_name, algo_desc, src_gs
        )

        out_path = os.path.join(output_dir, f"routing_table_{algo_name.lower()}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    ✅ Saved: {out_path}")

    print(f"\n{'=' * 60}")
    print("  ✅ All 4 routing table JSON files saved!")
    print("=" * 60)


if __name__ == "__main__":
    main()
