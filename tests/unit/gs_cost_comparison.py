#!/usr/bin/env python3
"""
Ground Station Cost Comparison Across Algorithms

Reads the attack_gs_analysis.json and produces a unified view:
1. The FULL SET of ground stations involved across all 4 algorithms
2. For each GS: which algorithms need it, and what is its cost in each algorithm
3. Per-algorithm total cost and cross-algorithm union cost

"Cost" for a source GS in a given algorithm is defined as:
  - The number of attack destinations (how many dst GS pairs use the target ISL)
  - The total k-paths through the target ISL (weighted contribution)
  - The effective attack traffic (Mbps) this GS must inject

The key insight is:
  - For a FIXED algorithm, the attacker only attacks the GS set of THAT algorithm
  - For k-RAND, the attacker must attack the UNION of all algorithms' GS sets
"""

import json
import os
from collections import defaultdict

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "output")
INPUT_FILE = os.path.join(DATA_DIR, "attack_gs_analysis.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "gs_cost_comparison.json")

ISL_BW = 100.0  # Mbps - target ISL bandwidth


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    algos = ["KSP", "KDS", "KDG", "KLO"]

    # =========================================================================
    # 1. Build unified GS set and per-GS per-algorithm info
    # =========================================================================
    # gs_info[gs_name][algo] = { num_destinations, total_paths_through, destinations }
    gs_info = defaultdict(lambda: {})

    for algo in algos:
        algo_data = data["algorithms"][algo]
        for summary in algo_data["source_summary"]:
            gs = summary["ground_station"]
            gs_info[gs][algo] = {
                "num_destinations": summary["num_attack_destinations"],
                "total_paths_through": summary["total_paths_through_target"],
                "destinations": summary["destinations"],
            }

    all_gs = sorted(gs_info.keys())

    # =========================================================================
    # 2. Per-algorithm: compute attack cost per GS
    # =========================================================================
    # For each algorithm, the attacker sends traffic from bot GS to specific
    # destinations. Each (src, dst) pair has some k-paths, of which some go
    # through the target ISL. The fraction that hits the target is hit_ratio.
    #
    # To fill the target ISL to capacity (100 Mbps), the total attack traffic
    # sent = ISL_BW / P(through). Each GS's share of cost is proportional to
    # the paths it contributes through the target ISL.

    # First, build per-GS per-algo detailed pair info from the affected_pairs
    gs_pair_info = defaultdict(lambda: defaultdict(list))

    for algo in algos:
        for pair in data["algorithms"][algo]["affected_pairs"]:
            src = pair["source"]
            gs_pair_info[src][algo].append({
                "destination": pair["destination"],
                "paths_through": pair["paths_through_target"],
                "total_paths": pair["total_paths"],
                "hit_ratio": pair["hit_ratio"],
            })

    # =========================================================================
    # 3. Compute cost metrics
    # =========================================================================
    algo_totals = {}
    for algo in algos:
        algo_data = data["algorithms"][algo]
        p_through = algo_data["p_through"]
        total_cost_mbps = ISL_BW / p_through if p_through > 0 else float('inf')
        total_paths_through = algo_data["total_paths_through"]

        algo_totals[algo] = {
            "p_through": p_through,
            "total_cost_mbps": total_cost_mbps,
            "total_paths_through": total_paths_through,
            "source_gs_count": algo_data["source_gs_count"],
            "affected_pairs_count": algo_data["affected_pairs_count"],
        }

    # Per-GS cost = (GS's paths through target / total paths through target) * total cost
    gs_cost_table = []

    for gs in all_gs:
        row = {
            "ground_station": gs,
            "algorithms_involved": [],
            "per_algorithm": {},
        }

        for algo in algos:
            if algo in gs_info[gs]:
                info = gs_info[gs][algo]
                total_through = algo_totals[algo]["total_paths_through"]
                total_cost = algo_totals[algo]["total_cost_mbps"]

                # This GS's share: how much of the total attack traffic must
                # originate from this GS
                gs_share = info["total_paths_through"] / total_through if total_through > 0 else 0
                gs_cost_mbps = gs_share * total_cost

                row["algorithms_involved"].append(algo)
                row["per_algorithm"][algo] = {
                    "num_destinations": info["num_destinations"],
                    "paths_through_target": info["total_paths_through"],
                    "share_of_total": round(gs_share, 4),
                    "cost_mbps": round(gs_cost_mbps, 2),
                    "destinations": info["destinations"],
                }
            else:
                row["per_algorithm"][algo] = None

        # k-RAND cost: since attacker doesn't know the algorithm, it must
        # send enough traffic to cover the WORST case across all algorithms
        # For this GS, the k-RAND cost = max of its cost across all algorithms
        krand_costs = [row["per_algorithm"][a]["cost_mbps"]
                       for a in algos if row["per_algorithm"][a] is not None]
        row["krand_max_cost_mbps"] = max(krand_costs) if krand_costs else 0

        gs_cost_table.append(row)

    # Sort by number of algorithms involved (desc), then by max cost (desc)
    gs_cost_table.sort(key=lambda x: (-len(x["algorithms_involved"]),
                                       -x["krand_max_cost_mbps"]))

    # =========================================================================
    # 4. Print results
    # =========================================================================
    print("=" * 120)
    print("  GROUND STATION ATTACK COST COMPARISON")
    print(f"  Target ISL: {data['target_isl']}  |  ISL Bandwidth: {ISL_BW} Mbps")
    print("=" * 120)

    # Algorithm totals
    print(f"\n  {'─' * 90}")
    print(f"  Algorithm Totals:")
    print(f"  {'─' * 90}")
    print(f"  {'Algorithm':<10} {'Source GS':<12} {'Affected Pairs':<16} {'P(through)':<14} "
          f"{'Total Cost (Mbps)':<20} {'Bots @10Mbps'}")
    print(f"  {'─' * 90}")
    for algo in algos:
        t = algo_totals[algo]
        bots = int(t["total_cost_mbps"] / 10) + 1
        print(f"  {algo:<10} {t['source_gs_count']:<12} {t['affected_pairs_count']:<16} "
              f"{t['p_through']:<14.4f} {t['total_cost_mbps']:<20.1f} {bots}")

    # Full GS set
    print(f"\n\n  {'═' * 110}")
    print(f"  FULL GROUND STATION SET: {len(all_gs)} stations")
    print(f"  {'═' * 110}")

    # Header
    print(f"\n  {'Ground Station':<22} {'Algos':<14} ", end="")
    for algo in algos:
        print(f"│ {algo} Cost(Mbps) ", end="")
    print(f"│ {'k-RAND Max':>12}")
    print(f"  {'─' * 22} {'─' * 14} ", end="")
    for _ in algos:
        print(f"┼{'─' * 15} ", end="")
    print(f"┼{'─' * 13}")

    for row in gs_cost_table:
        gs = row["ground_station"]
        algos_str = ",".join(row["algorithms_involved"])
        print(f"  {gs:<22} {algos_str:<14} ", end="")
        for algo in algos:
            pa = row["per_algorithm"][algo]
            if pa is not None:
                print(f"│ {pa['cost_mbps']:>11.1f}    ", end="")
            else:
                print(f"│       —        ", end="")
        print(f"│ {row['krand_max_cost_mbps']:>11.1f}")

    # =========================================================================
    # 5. Show which GS are unique to certain algorithms
    # =========================================================================
    print(f"\n\n  {'═' * 90}")
    print(f"  ALGORITHM-SPECIFIC GS ANALYSIS")
    print(f"  {'═' * 90}")

    algo_gs_sets = {}
    for algo in algos:
        algo_gs_sets[algo] = set(data["algorithms"][algo]["source_gs_list"])

    union_all = set()
    for s in algo_gs_sets.values():
        union_all.update(s)

    intersection_all = algo_gs_sets[algos[0]].copy()
    for algo in algos[1:]:
        intersection_all &= algo_gs_sets[algo]

    print(f"\n  Union (all algorithms):        {len(union_all)} GS")
    print(f"  Intersection (all algorithms): {len(intersection_all)} GS")

    print(f"\n  GS in ALL 4 algorithms ({len(intersection_all)}):")
    for gs in sorted(intersection_all):
        print(f"    ✅ {gs}")

    # GS unique to specific algorithms
    for algo in algos:
        others = set()
        for other in algos:
            if other != algo:
                others.update(algo_gs_sets[other])
        unique = algo_gs_sets[algo] - others
        if unique:
            print(f"\n  GS ONLY in {algo} ({len(unique)}):")
            for gs in sorted(unique):
                print(f"    🔸 {gs}")

    # GS NOT in certain algorithms
    print(f"\n  Per-algorithm membership:")
    for gs in sorted(union_all):
        membership = [algo for algo in algos if gs in algo_gs_sets[algo]]
        missing = [algo for algo in algos if gs not in algo_gs_sets[algo]]
        if missing:
            print(f"    {gs:<22} IN: {','.join(membership):<20} NOT IN: {','.join(missing)}")

    # =========================================================================
    # 6. Detailed per-GS view: destinations and cost per algorithm
    # =========================================================================
    print(f"\n\n  {'═' * 100}")
    print(f"  DETAILED PER-GS VIEW (sorted by k-RAND max cost)")
    print(f"  {'═' * 100}")

    gs_cost_table.sort(key=lambda x: -x["krand_max_cost_mbps"])

    for row in gs_cost_table[:15]:  # Top 15
        gs = row["ground_station"]
        print(f"\n  📍 {gs}")
        print(f"     Involved in: {', '.join(row['algorithms_involved'])}")
        print(f"     k-RAND max cost: {row['krand_max_cost_mbps']:.1f} Mbps")

        for algo in algos:
            pa = row["per_algorithm"][algo]
            if pa is not None:
                print(f"     [{algo}] {pa['num_destinations']} dests, "
                      f"{pa['paths_through_target']} paths, "
                      f"share={pa['share_of_total']:.1%}, "
                      f"cost={pa['cost_mbps']:.1f} Mbps")
                print(f"            → {', '.join(pa['destinations'])}")
            else:
                print(f"     [{algo}] — (not involved)")

    # =========================================================================
    # 7. Summary: total cost comparison
    # =========================================================================
    print(f"\n\n  {'#' * 90}")
    print(f"  FINAL SUMMARY: ATTACK COST COMPARISON")
    print(f"  {'#' * 90}")

    # For each fixed algorithm: sum of all GS costs = total cost
    print(f"\n  For a FIXED algorithm, attacker only needs to cover that algorithm's GS set:")
    for algo in algos:
        t = algo_totals[algo]
        print(f"    {algo}: {t['source_gs_count']} GS, total cost = {t['total_cost_mbps']:.1f} Mbps")

    # For k-RAND: attacker needs to cover the UNION of all GS sets
    # and the cost = sum of max(cost across algos) for each GS
    krand_total = sum(row["krand_max_cost_mbps"] for row in gs_cost_table)
    print(f"\n  For k-RAND, attacker must cover {len(union_all)} GS (union of all):")
    print(f"    k-RAND: {len(union_all)} GS, total cost = {krand_total:.1f} Mbps")

    # Another way to think about k-RAND cost:
    # Since attacker doesn't know the algorithm, the expected cost is the
    # average of the four algorithms' costs (if uniform probability)
    avg_cost = sum(algo_totals[a]["total_cost_mbps"] for a in algos) / len(algos)
    max_cost = max(algo_totals[a]["total_cost_mbps"] for a in algos)
    min_cost = min(algo_totals[a]["total_cost_mbps"] for a in algos)

    print(f"\n  Alternative k-RAND cost models:")
    print(f"    Min fixed cost (easiest to attack):  {min_cost:.1f} Mbps")
    print(f"    Max fixed cost (hardest to attack):  {max_cost:.1f} Mbps")
    print(f"    Avg fixed cost:                      {avg_cost:.1f} Mbps")
    print(f"    k-RAND worst-case (must cover all):  {krand_total:.1f} Mbps")
    print(f"    k-RAND advantage vs min fixed:       {krand_total/min_cost:.2f}x")
    print(f"    k-RAND advantage vs avg fixed:       {krand_total/avg_cost:.2f}x")

    # =========================================================================
    # 8. Save results to JSON
    # =========================================================================
    output = {
        "target_isl": data["target_isl"],
        "isl_bandwidth_mbps": ISL_BW,
        "algorithm_totals": algo_totals,
        "all_ground_stations": all_gs,
        "total_gs_count": len(all_gs),
        "union_gs_count": len(union_all),
        "intersection_gs_count": len(intersection_all),
        "intersection_gs_list": sorted(list(intersection_all)),
        "union_gs_list": sorted(list(union_all)),
        "per_gs_cost_table": gs_cost_table,
        "krand_total_cost_mbps": krand_total,
        "per_algo_gs_sets": {algo: sorted(list(s)) for algo, s in algo_gs_sets.items()},
        "gs_not_in_all_algos": {
            gs: {
                "in": [a for a in algos if gs in algo_gs_sets[a]],
                "not_in": [a for a in algos if gs not in algo_gs_sets[a]],
            }
            for gs in sorted(union_all)
            if any(gs not in algo_gs_sets[a] for a in algos)
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Results saved to: {OUTPUT_FILE}")
    print("  Done!")


if __name__ == "__main__":
    main()
