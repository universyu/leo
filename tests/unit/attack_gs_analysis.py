#!/usr/bin/env python3
"""
Attack Ground Station Analysis for Target ISL: SAT_4_2 <-> SAT_4_3

For each of the 4 routing algorithms (KSP, KDS, KDG, KLO):
1. Find ALL GS pairs whose k-paths pass through SAT_4_2 <-> SAT_4_3
2. Identify which SOURCE ground stations the attacker needs bots at
3. Show exactly which GS -> GS communications are affected
4. Save results to JSON

Logic:
- The attacker wants to congest ISL SAT_4_2 <-> SAT_4_3
- To do this, attacker places bots at ground stations and sends traffic
  to destinations such that the routing algorithm routes the traffic
  through the target ISL
- For each algorithm, the attacker only needs to use GS pairs whose
  routes pass through the target ISL
"""

import sys
import os
import json
import time
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from leo_network import LEOConstellation
from leo_network.core.routing import (
    KShortestPathsRouter, KDSRouter, KDGRouter, KLORouter
)
from leo_network.core.topology import LinkType


def get_k_paths(router, src, dst):
    """Get k paths from a router, using the appropriate method."""
    if isinstance(router, KShortestPathsRouter):
        return router.compute_k_paths(src, dst)
    elif isinstance(router, KDSRouter):
        return router.compute_k_disjoint_paths(src, dst)
    elif isinstance(router, KDGRouter):
        return router.compute_k_geodiverse_paths(src, dst)
    elif isinstance(router, KLORouter):
        return router.get_all_disjoint_paths(src, dst)
    return []


def path_uses_isl(path, target_a, target_b):
    """Check if a path passes through the target ISL link."""
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if (a == target_a and b == target_b) or (a == target_b and b == target_a):
            return True
    return False


def main():
    TARGET_A = "SAT_4_2"
    TARGET_B = "SAT_4_3"
    K = 3

    print("=" * 80)
    print(f"  Attack GS Analysis: Target ISL = {TARGET_A} <-> {TARGET_B}")
    print("=" * 80)

    # Create constellation
    constellation = LEOConstellation(
        num_planes=6, sats_per_plane=11,
        altitude_km=550.0, inclination_deg=53.0,
        isl_bandwidth_mbps=100.0
    )
    constellation.add_global_ground_stations()

    gs_nodes = sorted([n for n in constellation.graph.nodes() if n.startswith("GS_")])
    print(f"  Ground stations: {len(gs_nodes)}")
    print(f"  Total GS pairs: {len(gs_nodes) * (len(gs_nodes) - 1)}")

    # Create routers
    routers = {
        "KSP": KShortestPathsRouter(constellation, k=K),
        "KDS": KDSRouter(constellation, k=K),
        "KDG": KDGRouter(constellation, k=K),
        "KLO": KLORouter(constellation, k=K),
    }

    # Result container
    results = {
        "target_isl": f"{TARGET_A} <-> {TARGET_B}",
        "k": K,
        "total_gs": len(gs_nodes),
        "total_gs_pairs": len(gs_nodes) * (len(gs_nodes) - 1),
        "algorithms": {}
    }

    for algo_name, router in routers.items():
        print(f"\n{'─' * 70}")
        print(f"  Algorithm: {algo_name}")
        print(f"{'─' * 70}")

        t0 = time.time()

        # For each GS pair, check all k-paths
        # attack_sources[gs] = list of destinations where gs->dst passes through target
        attack_sources = defaultdict(list)
        # affected_pairs: list of (src, dst, num_paths_through, total_paths)
        affected_pairs = []
        # source GS that attacker needs bots at
        source_gs_set = set()
        # destination GS that the attack traffic is addressed to
        dest_gs_set = set()

        total_paths_through = 0
        total_paths = 0

        for src in gs_nodes:
            for dst in gs_nodes:
                if src == dst:
                    continue

                paths = get_k_paths(router, src, dst)
                total_paths += len(paths)

                through_count = 0
                through_paths_detail = []
                for path in paths:
                    if path_uses_isl(path, TARGET_A, TARGET_B):
                        through_count += 1
                        through_paths_detail.append(
                            " -> ".join(path)
                        )

                if through_count > 0:
                    total_paths_through += through_count
                    source_gs_set.add(src)
                    dest_gs_set.add(dst)
                    attack_sources[src].append({
                        "destination": dst,
                        "paths_through_target": through_count,
                        "total_paths": len(paths),
                        "hit_ratio": through_count / len(paths),
                        "paths": through_paths_detail,
                    })
                    affected_pairs.append({
                        "source": src,
                        "destination": dst,
                        "paths_through_target": through_count,
                        "total_paths": len(paths),
                        "hit_ratio": through_count / len(paths),
                    })

        elapsed = time.time() - t0

        # Sort affected pairs by hit_ratio descending
        affected_pairs.sort(key=lambda x: (-x["hit_ratio"], -x["paths_through_target"]))

        # Count how many attack destinations each source GS has
        source_summary = []
        for gs in sorted(source_gs_set):
            dests = attack_sources[gs]
            total_through = sum(d["paths_through_target"] for d in dests)
            source_summary.append({
                "ground_station": gs,
                "num_attack_destinations": len(dests),
                "total_paths_through_target": total_through,
                "destinations": [d["destination"] for d in dests],
            })
        source_summary.sort(key=lambda x: -x["total_paths_through_target"])

        # Print summary
        print(f"  Computation time: {elapsed:.2f}s")
        print(f"  Affected GS pairs: {len(affected_pairs)} / {len(gs_nodes)*(len(gs_nodes)-1)}")
        print(f"  Total k-paths through target: {total_paths_through} / {total_paths}")
        print(f"  P(through target): {total_paths_through/max(1,total_paths):.4f}")
        print(f"  Source GS needed (attacker bot locations): {len(source_gs_set)}")
        print(f"  Destination GS (attack targets): {len(dest_gs_set)}")

        print(f"\n  === Source Ground Stations (where attacker places bots) ===")
        print(f"  {'Rank':<5} {'Ground Station':<22} {'# Destinations':<16} {'Paths Through'}")
        print(f"  {'─' * 60}")
        for i, s in enumerate(source_summary, 1):
            print(f"  {i:<5} {s['ground_station']:<22} {s['num_attack_destinations']:<16} {s['total_paths_through_target']}")

        print(f"\n  === Top 20 Most Effective Attack GS Pairs ===")
        print(f"  {'Rank':<5} {'Source':<20} {'Destination':<20} {'Through/Total':<16} {'Hit Ratio'}")
        print(f"  {'─' * 75}")
        for i, p in enumerate(affected_pairs[:20], 1):
            print(f"  {i:<5} {p['source']:<20} {p['destination']:<20} "
                  f"{p['paths_through_target']}/{p['total_paths']:<12} {p['hit_ratio']:.1%}")

        # Store results
        results["algorithms"][algo_name] = {
            "affected_pairs_count": len(affected_pairs),
            "total_paths_through": total_paths_through,
            "total_paths": total_paths,
            "p_through": total_paths_through / max(1, total_paths),
            "source_gs_count": len(source_gs_set),
            "dest_gs_count": len(dest_gs_set),
            "source_gs_list": sorted(list(source_gs_set)),
            "dest_gs_list": sorted(list(dest_gs_set)),
            "source_summary": source_summary,
            "affected_pairs": affected_pairs,
            "computation_time_s": elapsed,
        }

    # =========================================================================
    # Cross-algorithm comparison
    # =========================================================================
    print(f"\n\n{'#' * 80}")
    print(f"  CROSS-ALGORITHM COMPARISON")
    print(f"{'#' * 80}")

    print(f"\n  {'Algorithm':<10} {'Affected Pairs':<16} {'Source GS':<12} {'Dest GS':<12} "
          f"{'P(through)':<14} {'Cost(Mbps)'}")
    print(f"  {'─' * 80}")

    isl_bw = 100.0  # Mbps
    for algo in ["KSP", "KDS", "KDG", "KLO"]:
        d = results["algorithms"][algo]
        p = d["p_through"]
        cost = isl_bw / p if p > 0 else float('inf')
        print(f"  {algo:<10} {d['affected_pairs_count']:<16} {d['source_gs_count']:<12} "
              f"{d['dest_gs_count']:<12} {p:<14.4f} {cost:.1f}")

    # Union of source GS across all algorithms (what k-RAND attacker needs)
    all_source_gs = set()
    all_dest_gs = set()
    all_affected_pairs = set()
    for algo in ["KSP", "KDS", "KDG", "KLO"]:
        d = results["algorithms"][algo]
        all_source_gs.update(d["source_gs_list"])
        all_dest_gs.update(d["dest_gs_list"])
        for p in d["affected_pairs"]:
            all_affected_pairs.add((p["source"], p["destination"]))

    print(f"\n  k-RAND (union of all algorithms):")
    print(f"    Source GS needed: {len(all_source_gs)} (vs individual algorithms above)")
    print(f"    Dest GS needed:   {len(all_dest_gs)}")
    print(f"    Total affected pairs: {len(all_affected_pairs)}")

    results["k_RAND"] = {
        "source_gs_count": len(all_source_gs),
        "source_gs_list": sorted(list(all_source_gs)),
        "dest_gs_count": len(all_dest_gs),
        "dest_gs_list": sorted(list(all_dest_gs)),
        "affected_pairs_count": len(all_affected_pairs),
    }

    # Show which GS are unique to certain algorithms
    print(f"\n  === Source GS Uniqueness Analysis ===")
    for algo in ["KSP", "KDS", "KDG", "KLO"]:
        algo_gs = set(results["algorithms"][algo]["source_gs_list"])
        other_gs = set()
        for other_algo in ["KSP", "KDS", "KDG", "KLO"]:
            if other_algo != algo:
                other_gs.update(results["algorithms"][other_algo]["source_gs_list"])
        unique_gs = algo_gs - other_gs
        if unique_gs:
            print(f"  {algo} unique source GS: {sorted(unique_gs)}")
        else:
            print(f"  {algo}: no unique source GS (all shared with other algorithms)")

    # Detailed per-algorithm source GS comparison
    print(f"\n  === Per-Algorithm Source GS Bot Placement ===")
    for algo in ["KSP", "KDS", "KDG", "KLO"]:
        gs_list = results["algorithms"][algo]["source_gs_list"]
        print(f"\n  [{algo}] Attacker needs bots at {len(gs_list)} ground stations:")
        for gs in gs_list:
            # Find total paths through for this GS
            summary = next((s for s in results["algorithms"][algo]["source_summary"]
                          if s["ground_station"] == gs), None)
            if summary:
                print(f"    📍 {gs} → sends to {summary['num_attack_destinations']} destinations, "
                      f"{summary['total_paths_through_target']} paths through target")

    # Save to JSON
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "output", "attack_gs_analysis.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Results saved to: {output_path}")

    print("\n  Done!")


if __name__ == "__main__":
    main()
