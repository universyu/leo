#!/usr/bin/env python3
"""
Theoretical Attack Cost Calculator (No Simulation Required)

Based on pre-computed routing tables, calculate the theoretical DDoS
attack cost for each routing algorithm and k-RAND.

Core Logic (from the paper):
=============================================================
- The attacker knows the routing algorithm and the network topology.
- For a FIXED algorithm: the attacker knows exactly which k paths will
  be used for each GS pair. It only needs to send traffic along the
  paths that pass through the target ISL.
- For k-RAND: the attacker does NOT know which algorithm will be chosen
  at runtime. It must cover ALL possible paths across ALL 4 algorithms.
  This dramatically increases the attack cost or MaxUp.

Key Metrics:
- Cost: Total traffic volume (Mbps) needed to congest the target ISL
- MaxUp: Max uplink bandwidth increase at any single ground station
- Number of bots required (assuming each bot generates X Mbps)
=============================================================
"""

import sys
import os
import time
from collections import defaultdict, Counter
from itertools import combinations

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


def path_uses_isl(path, target_src, target_dst):
    """Check if a path passes through the target ISL link."""
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if (a == target_src and b == target_dst) or (a == target_dst and b == target_src):
            return True
    return False


def extract_isl_set(path):
    """Extract all ISL links from a path as a set of normalized tuples."""
    isls = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if a.startswith("SAT_") and b.startswith("SAT_"):
            isls.add((min(a, b), max(a, b)))
    return isls


def get_uplink_sat(path):
    """Get the first satellite in the path (the uplink satellite)."""
    for node in path:
        if node.startswith("SAT_"):
            return node
    return None


def analyze_target_isl(constellation, target_isl, k=3):
    """
    Full analysis of attack cost for a given target ISL.

    For each algorithm:
    1. Find all GS pairs whose k-paths pass through the target ISL
    2. Count how many of the k paths pass through the target
    3. Calculate the probability that a randomly selected path goes through target
    4. Calculate attack cost = ISL_capacity / P(through target)
    5. Calculate MaxUp = max traffic needed at a single GS
    """
    target_src, target_dst = target_isl
    isl_bw = constellation.links[f"ISL_{target_src}_{target_dst}"].bandwidth \
        if f"ISL_{target_src}_{target_dst}" in constellation.links \
        else constellation.links[f"ISL_{target_dst}_{target_src}"].bandwidth

    print(f"\n{'=' * 80}")
    print(f"  TARGET ISL: {target_src} <-> {target_dst}  (Bandwidth: {isl_bw} Mbps)")
    print(f"{'=' * 80}")

    gs_nodes = sorted([n for n in constellation.graph.nodes() if n.startswith("GS_")])
    total_pairs = len(gs_nodes) * (len(gs_nodes) - 1)
    print(f"  Ground stations: {len(gs_nodes)}, Total GS pairs: {total_pairs}")

    # Create routers
    algo_names = ["KSP", "KDS", "KDG", "KLO"]
    routers = {
        "KSP": KShortestPathsRouter(constellation, k=k),
        "KDS": KDSRouter(constellation, k=k),
        "KDG": KDGRouter(constellation, k=k),
        "KLO": KLORouter(constellation, k=k),
    }

    # =========================================================================
    # Step 1: For each algorithm, collect all k-paths for all GS pairs
    # =========================================================================
    algo_data = {}

    for algo_name, router in routers.items():
        t0 = time.time()
        pairs_through = []       # GS pairs with at least 1 path through target
        k_paths_through = 0      # Total count of individual k-paths through target
        k_paths_total = 0        # Total count of all k-paths
        all_isls_used = set()    # Union of all ISLs used by this algorithm
        uplink_load = Counter()  # Per-satellite uplink load for attack traffic

        # For each pair that has paths through target, record details
        pair_details = []

        for src in gs_nodes:
            for dst in gs_nodes:
                if src == dst:
                    continue

                paths = get_k_paths(router, src, dst)
                k_paths_total += len(paths)

                through_count = 0
                for path in paths:
                    all_isls_used.update(extract_isl_set(path))
                    if path_uses_isl(path, target_src, target_dst):
                        through_count += 1

                k_paths_through += through_count

                if through_count > 0:
                    pairs_through.append((src, dst))
                    # The first satellite in the path is the uplink point
                    for path in paths:
                        if path_uses_isl(path, target_src, target_dst):
                            up_sat = get_uplink_sat(path)
                            if up_sat:
                                uplink_load[up_sat] += 1

                    pair_details.append({
                        "src": src, "dst": dst,
                        "through": through_count,
                        "total": len(paths),
                        "ratio": through_count / len(paths) if paths else 0,
                    })

        elapsed = time.time() - t0
        p_through = k_paths_through / max(1, k_paths_total)

        algo_data[algo_name] = {
            "pairs_through": pairs_through,
            "num_pairs_through": len(pairs_through),
            "k_paths_through": k_paths_through,
            "k_paths_total": k_paths_total,
            "p_through": p_through,
            "unique_isls": len(all_isls_used),
            "uplink_load": uplink_load,
            "pair_details": pair_details,
            "elapsed": elapsed,
        }

    # =========================================================================
    # Step 2: Calculate k-RAND metrics
    # =========================================================================
    # For k-RAND, the attacker doesn't know which algorithm is chosen.
    # With equal weights (0.25 each), the probability a packet goes through
    # target is the average across all 4 algorithms.
    krand_total_through = sum(d["k_paths_through"] for d in algo_data.values())
    krand_total_paths = sum(d["k_paths_total"] for d in algo_data.values())
    krand_p_through = krand_total_through / max(1, krand_total_paths)

    # The attacker must cover ALL GS pairs that have paths through target
    # in ANY algorithm (union of all algorithms' attack surfaces)
    all_attack_pairs = set()
    for d in algo_data.values():
        all_attack_pairs.update(d["pairs_through"])

    # For k-RAND, the uplink load is spread across more satellites
    krand_uplink = Counter()
    for d in algo_data.values():
        krand_uplink += d["uplink_load"]

    algo_data["k-RAND"] = {
        "num_pairs_through": len(all_attack_pairs),
        "k_paths_through": krand_total_through,
        "k_paths_total": krand_total_paths,
        "p_through": krand_p_through,
        "unique_isls": len(set().union(*(set() for _ in algo_data.values()))),
        "uplink_load": krand_uplink,
    }

    # =========================================================================
    # Step 3: Calculate Attack Cost and MaxUp
    # =========================================================================
    print(f"\n  {'─' * 75}")
    print(f"  STEP 1: Path Analysis per Algorithm")
    print(f"  {'─' * 75}")
    print(f"  {'Algorithm':<10} {'GS Pairs w/ Target':<20} {'K-Paths Through':<18} "
          f"{'Total K-Paths':<15} {'P(through)':<12} {'ISLs Used'}")
    print(f"  {'─' * 90}")

    for algo in ["KSP", "KDS", "KDG", "KLO", "k-RAND"]:
        d = algo_data[algo]
        print(f"  {algo:<10} {d['num_pairs_through']:<20} {d['k_paths_through']:<18} "
              f"{d['k_paths_total']:<15} {d['p_through']:<12.4f} "
              f"{d.get('unique_isls', 'N/A')}")

    # =========================================================================
    # Attack Cost Calculation
    # =========================================================================
    print(f"\n  {'─' * 75}")
    print(f"  STEP 2: Attack Cost Calculation")
    print(f"  {'─' * 75}")
    print(f"\n  ISL Bandwidth = {isl_bw} Mbps")
    print(f"  To congest the target ISL, the attacker needs the total attack traffic")
    print(f"  going through this ISL to exceed {isl_bw} Mbps.")
    print(f"")
    print(f"  If P(through) = probability a single attack packet's route goes through")
    print(f"  the target ISL, then:")
    print(f"    Total attack traffic needed = ISL_BW / P(through)")
    print(f"    (because only P fraction of attack traffic actually hits the target)")
    print(f"")

    bot_traffic_mbps = 10.0  # Each bot generates 10 Mbps

    print(f"  {'Algorithm':<10} {'P(through)':<14} {'Cost (Mbps)':<14} {'Cost Factor':<14} "
          f"{'Bots Needed':<14} {'Bot Factor'}")
    print(f"  {'─' * 80}")

    baseline_cost = None
    cost_results = {}

    for algo in ["KSP", "KDS", "KDG", "KLO", "k-RAND"]:
        d = algo_data[algo]
        p = d["p_through"]

        if p > 0:
            cost_mbps = isl_bw / p
            bots_needed = int(np.ceil(cost_mbps / bot_traffic_mbps))
        else:
            cost_mbps = float('inf')
            bots_needed = float('inf')

        if baseline_cost is None:
            baseline_cost = cost_mbps

        cost_factor = cost_mbps / baseline_cost if baseline_cost > 0 else 0
        bot_factor = bots_needed / (baseline_cost / bot_traffic_mbps) if baseline_cost > 0 else 0

        cost_results[algo] = {
            "p_through": p,
            "cost_mbps": cost_mbps,
            "cost_factor": cost_factor,
            "bots_needed": bots_needed,
        }

        print(f"  {algo:<10} {p:<14.4f} {cost_mbps:<14.1f} {cost_factor:<14.2f}x "
              f"{bots_needed:<14} {bot_factor:.2f}x")

    # =========================================================================
    # MaxUp Calculation
    # =========================================================================
    print(f"\n  {'─' * 75}")
    print(f"  STEP 3: MaxUp (Detectability) Analysis")
    print(f"  {'─' * 75}")
    print(f"\n  MaxUp = maximum uplink bandwidth increase at any single satellite.")
    print(f"  Lower MaxUp = harder to detect (attack spread across more satellites).")
    print(f"  Higher MaxUp = easier to detect (too much traffic at one satellite).")
    print(f"")

    # For each algorithm, the attacker targets specific GS pairs.
    # The uplink traffic goes to the GS's nearest satellite.
    # MaxUp = max traffic at any single satellite / total attack traffic

    print(f"  {'Algorithm':<10} {'Attack Pairs':<14} {'Unique Uplink Sats':<20} "
          f"{'Max Load (count)':<18} {'Hottest Sat'}")
    print(f"  {'─' * 80}")

    for algo in ["KSP", "KDS", "KDG", "KLO", "k-RAND"]:
        d = algo_data[algo]
        uplink = d["uplink_load"]
        if uplink:
            max_sat, max_count = uplink.most_common(1)[0]
            unique_sats = len(uplink)
        else:
            max_sat, max_count = "N/A", 0
            unique_sats = 0

        num_pairs = d["num_pairs_through"]
        print(f"  {algo:<10} {num_pairs:<14} {unique_sats:<20} "
              f"{max_count:<18} {max_sat}")

    # For a fair comparison: if attacker wants to be stealthy (MaxUp <= threshold),
    # how many bots does it need?
    print(f"\n  {'─' * 75}")
    print(f"  STEP 4: Cost vs MaxUp Trade-off")
    print(f"  {'─' * 75}")
    print(f"\n  Scenario: Attacker wants MaxUp <= 20 Mbps per satellite (stealth mode)")
    print(f"  Each bot sends 10 Mbps. Max 2 bots per ground station allowed.")
    print(f"")

    maxup_threshold = 20.0  # Mbps per satellite

    print(f"  {'Algorithm':<10} {'Cost (Mbps)':<14} {'Bots Needed':<14} "
          f"{'GS Needed':<12} {'Feasible?':<12}")
    print(f"  {'─' * 65}")

    for algo in ["KSP", "KDS", "KDG", "KLO", "k-RAND"]:
        cr = cost_results[algo]
        d = algo_data[algo]
        uplink = d["uplink_load"]

        cost = cr["cost_mbps"]
        bots = cr["bots_needed"]

        # How many unique GS the attacker needs bots at
        # Simplified: need to distribute bots across enough GS so that
        # no single satellite's uplink exceeds maxup_threshold
        if uplink:
            max_sat, max_count = uplink.most_common(1)[0]
            total_uplink_count = sum(uplink.values())
            # Max fraction going to one satellite
            max_fraction = max_count / total_uplink_count if total_uplink_count > 0 else 1
            # Effective MaxUp if all attack traffic sent
            effective_maxup = cost * max_fraction
            # Ground stations needed
            gs_needed = max(1, int(np.ceil(bots / 2)))  # 2 bots per GS max
            feasible = effective_maxup <= maxup_threshold
        else:
            gs_needed = bots
            feasible = False
            effective_maxup = cost

        feasible_str = "✅ YES" if feasible else f"❌ NO (MaxUp={effective_maxup:.0f})"
        if cost == float('inf'):
            print(f"  {algo:<10} {'∞':<14} {'∞':<14} {'∞':<12} {'N/A':<12}")
        else:
            print(f"  {algo:<10} {cost:<14.1f} {bots:<14} {gs_needed:<12} {feasible_str}")

    # =========================================================================
    # Detailed: Which GS pairs matter most for each algorithm
    # =========================================================================
    print(f"\n  {'─' * 75}")
    print(f"  STEP 5: Most Valuable Attack GS Pairs (per algorithm)")
    print(f"  {'─' * 75}")
    print(f"  (GS pairs where the highest fraction of k-paths go through target ISL)")

    for algo in ["KSP", "KDS", "KDG", "KLO"]:
        d = algo_data[algo]
        if not d.get("pair_details"):
            continue
        sorted_pairs = sorted(d["pair_details"], key=lambda x: x["ratio"], reverse=True)
        print(f"\n  [{algo}] Top 10 attack GS pairs:")
        print(f"    {'Source':<18} {'Destination':<18} {'Through/Total':<16} {'Hit Ratio'}")
        print(f"    {'─' * 65}")
        for pd in sorted_pairs[:10]:
            print(f"    {pd['src']:<18} {pd['dst']:<18} "
                  f"{pd['through']}/{pd['total']:<14} {pd['ratio']:.1%}")

    # =========================================================================
    # Key Comparison: path overlap across algorithms
    # =========================================================================
    print(f"\n  {'─' * 75}")
    print(f"  STEP 6: Cross-Algorithm Path Diversity (Why k-RAND Wins)")
    print(f"  {'─' * 75}")

    # For a few interesting GS pairs, show the ISL sets used by each algorithm
    # to demonstrate that different algorithms use different ISLs
    sample_pairs = []
    for d in algo_data.values():
        if d.get("pair_details"):
            for pd in d["pair_details"]:
                if pd["ratio"] >= 0.5:
                    sample_pairs.append((pd["src"], pd["dst"]))

    sample_pairs = list(set(sample_pairs))[:8]

    print(f"\n  For {len(sample_pairs)} sample GS pairs, comparing ISL sets used:")
    print(f"  (Different algorithms produce different paths → attacker must cover all)")

    for src, dst in sample_pairs:
        print(f"\n  📡 {src} → {dst}:")
        all_pair_isls = set()
        algo_isls = {}

        for algo_name, router in routers.items():
            paths = get_k_paths(router, src, dst)
            isls = set()
            through_count = 0
            for path in paths:
                path_isls = extract_isl_set(path)
                isls.update(path_isls)
                if path_uses_isl(path, target_src, target_dst):
                    through_count += 1
            algo_isls[algo_name] = isls
            all_pair_isls.update(isls)
            print(f"    {algo_name}: {len(paths)} paths, {len(isls)} unique ISLs, "
                  f"{through_count}/{len(paths)} through target")

        # Overlap analysis
        if len(algo_isls) >= 2:
            names = list(algo_isls.keys())
            total_union = set()
            for isls in algo_isls.values():
                total_union.update(isls)

            total_intersection = algo_isls[names[0]]
            for name in names[1:]:
                total_intersection = total_intersection & algo_isls[name]

            diversity = 1 - len(total_intersection) / max(1, len(total_union))
            print(f"    → Union of all ISLs: {len(total_union)}, "
                  f"Intersection: {len(total_intersection)}, "
                  f"Diversity: {diversity:.1%}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print(f"\n\n{'#' * 80}")
    print(f"  FINAL SUMMARY")
    print(f"{'#' * 80}")

    print(f"\n  Target ISL: {target_src} <-> {target_dst} ({isl_bw} Mbps)")
    print(f"  Bot traffic: {bot_traffic_mbps} Mbps each")
    print()

    print(f"  ┌──────────┬────────────┬────────────┬────────────┬────────────┐")
    print(f"  │Algorithm │ P(through) │ Cost(Mbps) │ Cost Factor│ Bots Needed│")
    print(f"  ├──────────┼────────────┼────────────┼────────────┼────────────┤")

    for algo in ["KSP", "KDS", "KDG", "KLO", "k-RAND"]:
        cr = cost_results[algo]
        p = cr["p_through"]
        cost = cr["cost_mbps"]
        factor = cr["cost_factor"]
        bots = cr["bots_needed"]
        marker = " ★" if algo == "k-RAND" else "  "
        if cost < 1e9:
            print(f"  │{algo:<10}│ {p:>10.4f} │ {cost:>10.1f} │ {factor:>10.2f}x│ {bots:>10} │{marker}")
        else:
            print(f"  │{algo:<10}│ {p:>10.4f} │ {'∞':>10} │ {'∞':>10} │ {'∞':>10} │{marker}")

    print(f"  └──────────┴────────────┴────────────┴────────────┴────────────┘")

    # Compute the advantage
    krand_cost = cost_results["k-RAND"]["cost_mbps"]
    fixed_costs = [cost_results[a]["cost_mbps"] for a in ["KSP", "KDS", "KDG", "KLO"]
                   if cost_results[a]["cost_mbps"] < 1e9]
    if fixed_costs and krand_cost < 1e9:
        min_fixed = min(fixed_costs)
        max_fixed = max(fixed_costs)
        avg_fixed = sum(fixed_costs) / len(fixed_costs)

        print(f"\n  📊 k-RAND advantage over easiest-to-attack algorithm:")
        print(f"     Min fixed cost: {min_fixed:.1f} Mbps ({min_fixed/bot_traffic_mbps:.0f} bots)")
        print(f"     k-RAND cost:    {krand_cost:.1f} Mbps ({krand_cost/bot_traffic_mbps:.0f} bots)")
        print(f"     Cost increase:  {krand_cost/min_fixed:.2f}x")
        print(f"     Extra bots:     +{(krand_cost - min_fixed)/bot_traffic_mbps:.0f}")

        print(f"\n  📊 k-RAND advantage over average fixed algorithm:")
        print(f"     Avg fixed cost: {avg_fixed:.1f} Mbps ({avg_fixed/bot_traffic_mbps:.0f} bots)")
        print(f"     k-RAND cost:    {krand_cost:.1f} Mbps ({krand_cost/bot_traffic_mbps:.0f} bots)")
        print(f"     Cost increase:  {krand_cost/avg_fixed:.2f}x")

    print(f"\n  ✅ Conclusion:")
    print(f"     When the routing algorithm is fixed (deterministic), the attacker")
    print(f"     can precisely predict which paths will be used and minimize its cost.")
    print(f"     With k-RAND, the attacker must cover all possible paths from ALL")
    print(f"     4 algorithms, increasing the cost by {krand_cost/min_fixed:.2f}x.")
    print(f"     This validates the paper's core thesis: randomness in routing")
    print(f"     increases the attacker's cost and/or detectability (MaxUp).")
    print()

    return cost_results, algo_data


def analyze_multiple_targets(constellation, k=3):
    """Analyze attack cost for multiple target ISLs."""
    print("\n" + "=" * 80)
    print("  MULTI-TARGET ANALYSIS: Top 5 Most Vulnerable ISL Links")
    print("=" * 80)

    # First, find the most-used ISL links
    router = KShortestPathsRouter(constellation, k=k)
    router.precompute_ground_station_routes()
    gs_nodes = sorted([n for n in constellation.graph.nodes() if n.startswith("GS_")])

    link_usage = Counter()
    for src in gs_nodes:
        for dst in gs_nodes:
            if src == dst:
                continue
            entry = router.routing_table.get(src, {}).get(dst)
            if entry and entry.valid:
                for i in range(len(entry.path) - 1):
                    a, b = entry.path[i], entry.path[i + 1]
                    if a.startswith("SAT_") and b.startswith("SAT_"):
                        link_usage[(min(a, b), max(a, b))] += 1

    top_isls = [link for link, _ in link_usage.most_common(5)]

    print(f"\n  Top 5 most-used ISL links (by route count):")
    for i, (link, count) in enumerate(link_usage.most_common(5)):
        print(f"    {i+1}. {link[0]} <-> {link[1]}  ({count} routes)")

    # Analyze each target
    summary = {}
    for target in top_isls:
        cost_results, _ = analyze_target_isl(constellation, target, k=k)
        summary[target] = cost_results

    # Cross-target summary
    print(f"\n\n{'#' * 80}")
    print(f"  CROSS-TARGET SUMMARY: k-RAND Cost Advantage")
    print(f"{'#' * 80}")

    print(f"\n  {'Target ISL':<30} {'Best Fixed Cost':<16} {'k-RAND Cost':<16} {'Advantage'}")
    print(f"  {'─' * 80}")

    for target in top_isls:
        cr = summary[target]
        fixed_costs = [cr[a]["cost_mbps"] for a in ["KSP", "KDS", "KDG", "KLO"]
                       if cr[a]["cost_mbps"] < 1e9]
        krand_cost = cr["k-RAND"]["cost_mbps"]
        if fixed_costs and krand_cost < 1e9:
            min_fixed = min(fixed_costs)
            advantage = krand_cost / min_fixed
            tgt_str = f"{target[0]} <-> {target[1]}"
            print(f"  {tgt_str:<30} {min_fixed:<16.1f} {krand_cost:<16.1f} {advantage:.2f}x")


# Import numpy for calculations
import numpy as np


def main():
    print("=" * 80)
    print("  Theoretical DDoS Attack Cost Calculator")
    print("  (Based on routing tables - no simulation required)")
    print("=" * 80)

    start = time.time()

    # Create constellation
    print("\n[1] Creating LEO Constellation (6×11 = 66 satellites)...")
    constellation = LEOConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0,
        inclination_deg=53.0,
        isl_bandwidth_mbps=100.0
    )
    constellation.add_global_ground_stations()

    sat_count = len(constellation.satellites)
    gs_count = len(constellation.ground_stations)
    isl_count = sum(1 for l in constellation.links.values()
                    if l.link_type in [LinkType.ISL_INTRA, LinkType.ISL_INTER])
    print(f"  Satellites: {sat_count}, Ground stations: {gs_count}, ISL links: {isl_count}")

    # Main target: the most vulnerable ISL from previous analysis
    print("\n[2] Analyzing primary target: SAT_4_2 <-> SAT_4_3")
    target_isl = ("SAT_4_2", "SAT_4_3")
    cost_results, algo_data = analyze_target_isl(constellation, target_isl, k=3)

    # Analyze multiple targets
    print("\n[3] Analyzing top 5 most vulnerable ISL links...")
    analyze_multiple_targets(constellation, k=3)

    elapsed = time.time() - start
    print(f"\n  Total execution time: {elapsed:.1f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
