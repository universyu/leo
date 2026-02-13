#!/usr/bin/env python3
"""
Full Ground Station Attack Cost Table

Generates a complete table of all 38 ground stations with their attack cost
under each algorithm (KSP, KDS, KDG, KLO) and k-RAND.
Stations not involved in an algorithm's attack have cost = 0.
k-RAND = max(KSP, KDS, KDG, KLO) for each station.
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "output")
INPUT_FILE = os.path.join(DATA_DIR, "gs_cost_comparison.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "gs_full_cost_table.json")

# All 38 ground stations in the project
ALL_38_STATIONS = [
    "GS_NewYork", "GS_LosAngeles", "GS_Chicago", "GS_Toronto",
    "GS_Houston", "GS_Seattle", "GS_MexicoCity", "GS_Miami",
    "GS_London", "GS_Paris", "GS_Berlin", "GS_Moscow",
    "GS_Rome", "GS_Madrid", "GS_Istanbul", "GS_Stockholm",
    "GS_Beijing", "GS_Shanghai", "GS_Tokyo", "GS_Seoul",
    "GS_Mumbai", "GS_Delhi", "GS_Bangkok", "GS_Singapore",
    "GS_SaoPaulo", "GS_BuenosAires", "GS_Lima", "GS_Bogota",
    "GS_Santiago",
    "GS_Cairo", "GS_Lagos", "GS_Nairobi", "GS_Johannesburg",
    "GS_Sydney", "GS_Melbourne", "GS_Auckland",
    "GS_Dubai", "GS_TelAviv",
]

ALGOS = ["KSP", "KDS", "KDG", "KLO"]


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # Build lookup from existing per_gs_cost_table
    existing = {}
    for row in data["per_gs_cost_table"]:
        existing[row["ground_station"]] = row

    # Build the full table
    table = []
    for gs in sorted(ALL_38_STATIONS):
        row = {"ground_station": gs}
        costs = {}
        for algo in ALGOS:
            if gs in existing and existing[gs]["per_algorithm"].get(algo) is not None:
                costs[algo] = existing[gs]["per_algorithm"][algo]["cost_mbps"]
            else:
                costs[algo] = 0.0
        row["KSP_cost"] = costs["KSP"]
        row["KDS_cost"] = costs["KDS"]
        row["KDG_cost"] = costs["KDG"]
        row["KLO_cost"] = costs["KLO"]
        row["kRAND_cost"] = round(max(costs.values()), 2)
        table.append(row)

    # Compute totals
    totals = {algo: round(sum(r[f"{algo}_cost"] for r in table), 2) for algo in ALGOS}
    totals["kRAND"] = round(sum(r["kRAND_cost"] for r in table), 2)
    totals["kRAND_vs_min_fixed"] = round(totals["kRAND"] / min(totals[a] for a in ALGOS), 2)
    totals["kRAND_vs_max_fixed"] = round(totals["kRAND"] / max(totals[a] for a in ALGOS), 2)

    # Count how many stations have non-zero cost per algo
    counts = {algo: sum(1 for r in table if r[f"{algo}_cost"] > 0) for algo in ALGOS}
    counts["kRAND"] = sum(1 for r in table if r["kRAND_cost"] > 0)

    # Print table
    print("=" * 110)
    print(f"  FULL GROUND STATION ATTACK COST TABLE  (Target: SAT_4_2 <-> SAT_4_3)")
    print("=" * 110)
    print(f"\n  {'Ground Station':<22} {'KSP':>10} {'KDS':>10} {'KDG':>10} {'KLO':>10} {'k-RAND':>10}")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")

    for r in table:
        ksp = f"{r['KSP_cost']:.1f}" if r['KSP_cost'] > 0 else "0"
        kds = f"{r['KDS_cost']:.1f}" if r['KDS_cost'] > 0 else "0"
        kdg = f"{r['KDG_cost']:.1f}" if r['KDG_cost'] > 0 else "0"
        klo = f"{r['KLO_cost']:.1f}" if r['KLO_cost'] > 0 else "0"
        krand = f"{r['kRAND_cost']:.1f}" if r['kRAND_cost'] > 0 else "0"
        print(f"  {r['ground_station']:<22} {ksp:>10} {kds:>10} {kdg:>10} {klo:>10} {krand:>10}")

    print(f"  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")
    print(f"  {'TOTAL':<22} {totals['KSP']:>10.1f} {totals['KDS']:>10.1f} "
          f"{totals['KDG']:>10.1f} {totals['KLO']:>10.1f} {totals['kRAND']:>10.1f}")
    print(f"  {'Stations involved':<22} {counts['KSP']:>10} {counts['KDS']:>10} "
          f"{counts['KDG']:>10} {counts['KLO']:>10} {counts['kRAND']:>10}")

    print(f"\n  k-RAND advantage vs easiest (KSP):  {totals['kRAND_vs_min_fixed']}x")
    print(f"  k-RAND advantage vs hardest (KDS):  {totals['kRAND_vs_max_fixed']}x")

    # Save JSON
    output = {
        "target_isl": "SAT_4_2 <-> SAT_4_3",
        "total_ground_stations": len(ALL_38_STATIONS),
        "algorithms": ALGOS + ["kRAND"],
        "cost_table": table,
        "totals": totals,
        "stations_involved_count": counts,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
