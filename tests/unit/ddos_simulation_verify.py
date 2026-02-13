#!/usr/bin/env python3
"""
DDoS Attack Simulation Verification

For each routing algorithm (KSP, KDS, KDG, KLO, k-RAND):
1. Run baseline simulation (no attack) to get normal performance
2. Run attack simulation where attacker sends traffic from affected GS pairs
   with the exact cost (Mbps) calculated by gs_cost_comparison.py
3. Record: normal delivery rate, 5th percentile throughput, avg delay, etc.
4. Save all results to JSON

The attack model:
- For each algorithm, read the affected_pairs from attack_gs_analysis.json
- Convert each GS's cost_mbps into attack flow rate (pps)
- Attack flows go from source GS → destination GS, following the routing
- The router will route them through the network (some through target ISL)
- Measure the impact on normal traffic

For k-RAND:
- The attacker must cover ALL algorithms' GS pairs (union)
- Each GS's attack cost = max(cost across all 4 algorithms) [Method A]
"""

import sys
import os
import json
import time
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from leo_network import LEOConstellation
from leo_network.core.routing import (
    KShortestPathsRouter, KDSRouter, KDGRouter, KLORouter, KRandRouter
)
from leo_network.core.traffic import TrafficGenerator, Flow, PacketType, TrafficPattern
from leo_network.core.simulator import Simulator

# ============================================================================
# Configuration
# ============================================================================
ISL_BW = 100.0          # Mbps - target ISL bandwidth
PACKET_SIZE = 1000       # bytes per packet
NUM_NORMAL_FLOWS = 20    # number of background normal flows
NORMAL_RATE_RANGE = (50, 200)  # pps range for normal flows
SIM_DURATION = 2.0       # seconds
TIME_STEP = 0.001        # 1ms
SEED = 42
TARGET_ISL_A = "SAT_4_2"
TARGET_ISL_B = "SAT_4_3"

DATA_DIR = os.path.join(project_root, "output")
ANALYSIS_FILE = os.path.join(DATA_DIR, "attack_gs_analysis.json")
COST_FILE = os.path.join(DATA_DIR, "gs_cost_comparison.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "ddos_simulation_results.json")


def mbps_to_pps(mbps, packet_size_bytes=PACKET_SIZE):
    """Convert Mbps to packets per second"""
    bits_per_packet = packet_size_bytes * 8
    return (mbps * 1e6) / bits_per_packet


def create_constellation():
    """Create the standard constellation"""
    constellation = LEOConstellation(
        num_planes=6, sats_per_plane=11,
        altitude_km=550.0, inclination_deg=53.0,
        isl_bandwidth_mbps=ISL_BW
    )
    constellation.add_global_ground_stations()
    return constellation


def build_attack_flows_for_algo(algo_name, cost_data, analysis_data):
    """
    Build list of (source, destination, rate_pps) for a given algorithm.

    Each affected GS pair gets attack traffic proportional to its contribution.
    The total attack traffic for the algorithm = total_cost_mbps.
    Each source GS's share = its cost_mbps (from cost_data).
    We distribute each GS's cost evenly across its attack destinations.
    """
    attack_flows = []

    algo_totals = cost_data["algorithm_totals"][algo_name]
    total_cost_mbps = algo_totals["total_cost_mbps"]

    # Get per-GS cost info
    for gs_entry in cost_data["per_gs_cost_table"]:
        algo_info = gs_entry["per_algorithm"].get(algo_name)
        if algo_info is None:
            continue

        gs_cost_mbps = algo_info["cost_mbps"]
        destinations = algo_info["destinations"]

        if not destinations or gs_cost_mbps <= 0:
            continue

        # Distribute this GS's cost evenly across its destinations
        per_dest_mbps = gs_cost_mbps / len(destinations)
        per_dest_pps = mbps_to_pps(per_dest_mbps)

        for dst in destinations:
            attack_flows.append({
                "source": gs_entry["ground_station"],
                "destination": dst,
                "rate_pps": per_dest_pps,
                "rate_mbps": per_dest_mbps,
            })

    return attack_flows, total_cost_mbps


def build_attack_flows_for_krand(cost_data):
    """
    Build attack flows for k-RAND scenario.

    The attacker must cover ALL algorithms. For each GS, the cost =
    max(cost across all 4 algorithms) [Method A].

    For destinations, we take the union of destinations across all algorithms
    for each GS, and distribute the max cost evenly.
    """
    algos = ["KSP", "KDS", "KDG", "KLO"]
    attack_flows = []
    total_krand_cost = 0.0

    for gs_entry in cost_data["per_gs_cost_table"]:
        gs_name = gs_entry["ground_station"]

        # Find max cost and union of destinations
        max_cost = 0.0
        all_dests = set()

        for algo in algos:
            algo_info = gs_entry["per_algorithm"].get(algo)
            if algo_info is None:
                continue
            if algo_info["cost_mbps"] > max_cost:
                max_cost = algo_info["cost_mbps"]
            all_dests.update(algo_info["destinations"])

        if max_cost <= 0 or not all_dests:
            continue

        total_krand_cost += max_cost
        per_dest_mbps = max_cost / len(all_dests)
        per_dest_pps = mbps_to_pps(per_dest_mbps)

        for dst in sorted(all_dests):
            attack_flows.append({
                "source": gs_name,
                "destination": dst,
                "rate_pps": per_dest_pps,
                "rate_mbps": per_dest_mbps,
            })

    return attack_flows, total_krand_cost


def run_simulation(router_name, router, constellation, attack_flows, seed=SEED):
    """
    Run a single simulation with given router and attack flows.

    Returns dict with all metrics.
    """
    sim = Simulator(
        constellation=constellation,
        router=router,
        time_step=TIME_STEP,
        seed=seed
    )

    # Set target ISL for per-link throughput tracking
    sim.set_target_isl(TARGET_ISL_A, TARGET_ISL_B)

    # Add normal background traffic
    sim.add_random_normal_flows(
        num_flows=NUM_NORMAL_FLOWS,
        rate_range=NORMAL_RATE_RANGE,
        packet_size=PACKET_SIZE
    )

    # Add attack traffic flows
    for i, af in enumerate(attack_flows):
        flow_id = f"attack_{af['source']}_{af['destination']}_{i}"
        sim.traffic_generator.create_attack_flow(
            flow_id=flow_id,
            source=af["source"],
            destination=af["destination"],
            rate=af["rate_pps"],
            packet_size=PACKET_SIZE,
            start_time=0.0,
            duration=-1.0,
            pattern=TrafficPattern.CONSTANT
        )

    # Run
    print(f"    Running simulation ({SIM_DURATION}s, {int(SIM_DURATION/TIME_STEP)} steps)...")
    t0 = time.time()
    sim.run(duration=SIM_DURATION, progress_bar=True)
    elapsed = time.time() - t0

    # Collect results
    results = sim.get_results()
    stats = results["statistics"]

    # Normal traffic stats
    normal_stats = stats["normal_traffic"]
    attack_stats = stats["attack_traffic"]

    # Throughput percentiles - ALL traffic (for reference)
    tp = results["throughput_percentiles"]
    ntp = results["normal_throughput_percentiles"]
    titp = results["target_isl_normal_throughput_percentiles"]

    return {
        "router": router_name,
        "simulation_time_s": round(elapsed, 2),
        "num_attack_flows": len(attack_flows),
        "total_attack_rate_mbps": sum(af["rate_mbps"] for af in attack_flows),

        # Overall
        "overall_delivery_rate": round(stats["overview"]["delivery_rate"], 6),
        "overall_loss_rate": round(stats["overview"]["loss_rate"], 6),
        "total_packets_sent": stats["overview"]["total_packets_sent"],
        "total_packets_delivered": stats["overview"]["total_packets_delivered"],
        "total_packets_dropped": stats["overview"]["total_packets_dropped"],

        # Normal traffic
        "normal_delivery_rate": round(normal_stats["delivery_rate"], 6),
        "normal_delivered": normal_stats["delivered"],
        "normal_dropped": normal_stats["dropped"],

        # Attack traffic
        "attack_delivery_rate": round(attack_stats["delivery_rate"], 6),
        "attack_delivered": attack_stats["delivered"],
        "attack_dropped": attack_stats["dropped"],

        # Delay
        "avg_delay_ms": round(stats["delay"]["avg_ms"], 4),
        "p50_delay_ms": round(stats["delay"]["p50_ms"], 4),
        "p95_delay_ms": round(stats["delay"]["p95_ms"], 4),
        "p99_delay_ms": round(stats["delay"]["p99_ms"], 4),
        "max_delay_ms": round(stats["delay"]["max_ms"], 4),

        # Throughput percentiles - ALL traffic (for reference)
        "p5_throughput_pps": round(tp["p5_pps"], 4),
        "p5_throughput_mbps": round(tp["p5_mbps"], 6),
        "avg_throughput_pps": round(tp["avg_pps"], 4),
        "avg_throughput_mbps": round(tp["avg_mbps"], 6),

        # Throughput percentiles - NORMAL traffic only (all links)
        "normal_p5_throughput_pps": round(ntp["p5_pps"], 4),
        "normal_p5_throughput_mbps": round(ntp["p5_mbps"], 6),
        "normal_p10_throughput_pps": round(ntp["p10_pps"], 4),
        "normal_p10_throughput_mbps": round(ntp["p10_mbps"], 6),
        "normal_p50_throughput_pps": round(ntp["p50_pps"], 4),
        "normal_p50_throughput_mbps": round(ntp["p50_mbps"], 6),
        "normal_avg_throughput_pps": round(ntp["avg_pps"], 4),
        "normal_avg_throughput_mbps": round(ntp["avg_mbps"], 6),

        # Throughput percentiles - NORMAL traffic on TARGET ISL only
        "target_isl": titp.get("target_isl", "N/A"),
        "target_isl_normal_packets": titp.get("total_normal_packets", 0),
        "target_isl_normal_p5_pps": round(titp["p5_pps"], 4),
        "target_isl_normal_p5_mbps": round(titp["p5_mbps"], 6),
        "target_isl_normal_p10_pps": round(titp["p10_pps"], 4),
        "target_isl_normal_p10_mbps": round(titp["p10_mbps"], 6),
        "target_isl_normal_p50_pps": round(titp["p50_pps"], 4),
        "target_isl_normal_p50_mbps": round(titp["p50_mbps"], 6),
        "target_isl_normal_avg_pps": round(titp["avg_pps"], 4),
        "target_isl_normal_avg_mbps": round(titp["avg_mbps"], 6),

        # Link utilization
        "avg_link_utilization": round(stats["link_utilization"]["avg"], 6),
        "max_link_utilization": round(stats["link_utilization"]["max"], 6),
    }


def main():
    print("=" * 80)
    print("  DDoS ATTACK SIMULATION VERIFICATION")
    print("  Target ISL: SAT_4_2 <-> SAT_4_3  |  ISL BW: 100 Mbps")
    print("=" * 80)

    # Load pre-computed data
    print("\n[1] Loading pre-computed attack analysis data...")
    with open(ANALYSIS_FILE, "r") as f:
        analysis_data = json.load(f)
    with open(COST_FILE, "r") as f:
        cost_data = json.load(f)
    print("    ✅ Loaded attack_gs_analysis.json and gs_cost_comparison.json")

    # =========================================================================
    # Step 1: Run baseline (no attack) for each algorithm
    # =========================================================================
    print("\n[2] Running BASELINE simulations (no attack)...")

    algo_configs = {
        "KSP": lambda c: KShortestPathsRouter(c, k=3),
        "KDS": lambda c: KDSRouter(c, k=3),
        "KDG": lambda c: KDGRouter(c, k=3),
        "KLO": lambda c: KLORouter(c, k=3),
        "kRAND": lambda c: KRandRouter(c, k=3, seed=SEED),
    }

    baseline_results = {}
    for algo_name, router_factory in algo_configs.items():
        print(f"\n  --- Baseline: {algo_name} ---")
        constellation = create_constellation()
        router = router_factory(constellation)
        result = run_simulation(
            router_name=f"{algo_name}_baseline",
            router=router,
            constellation=constellation,
            attack_flows=[],  # No attack
            seed=SEED
        )
        baseline_results[algo_name] = result
        print(f"    Delivery Rate: {result['normal_delivery_rate']:.4f}")
        print(f"    P5 Throughput (all-links normal): {result['normal_p5_throughput_pps']:.2f} pps "
              f"({result['normal_p5_throughput_mbps']:.4f} Mbps)")
        print(f"    P5 Throughput (target ISL normal): {result['target_isl_normal_p5_pps']:.2f} pps "
              f"({result['target_isl_normal_p5_mbps']:.4f} Mbps) "
              f"[{result['target_isl_normal_packets']} pkts on ISL]")
        print(f"    Avg Delay: {result['avg_delay_ms']:.2f} ms")

    # =========================================================================
    # Step 2: Run attack simulations for each fixed algorithm
    # =========================================================================
    print("\n\n[3] Running ATTACK simulations...")

    attack_results = {}

    # Fixed algorithms: KSP, KDS, KDG, KLO
    for algo_name in ["KSP", "KDS", "KDG", "KLO"]:
        print(f"\n  --- Attack: {algo_name} ---")

        # Build attack flows for this algorithm
        attack_flows, total_cost = build_attack_flows_for_algo(
            algo_name, cost_data, analysis_data
        )
        print(f"    Attack flows: {len(attack_flows)}")
        print(f"    Total attack cost: {total_cost:.1f} Mbps "
              f"({mbps_to_pps(total_cost):.0f} pps)")

        # Create fresh constellation and router
        constellation = create_constellation()
        router = algo_configs[algo_name](constellation)

        result = run_simulation(
            router_name=f"{algo_name}_attack",
            router=router,
            constellation=constellation,
            attack_flows=attack_flows,
            seed=SEED
        )
        attack_results[algo_name] = result

        # Compare with baseline
        bl = baseline_results[algo_name]
        print(f"    Normal Delivery: {result['normal_delivery_rate']:.4f} "
              f"(baseline: {bl['normal_delivery_rate']:.4f})")
        print(f"    P5 All-links (normal): {result['normal_p5_throughput_pps']:.2f} pps "
              f"(baseline: {bl['normal_p5_throughput_pps']:.2f} pps)")
        print(f"    P5 Target ISL (normal): {result['target_isl_normal_p5_pps']:.2f} pps "
              f"(baseline: {bl['target_isl_normal_p5_pps']:.2f} pps) "
              f"[{result['target_isl_normal_packets']} pkts on ISL]")
        print(f"    Avg Delay: {result['avg_delay_ms']:.2f} ms "
              f"(baseline: {bl['avg_delay_ms']:.2f} ms)")

    # k-RAND attack
    print(f"\n  --- Attack: kRAND ---")
    attack_flows_krand, total_cost_krand = build_attack_flows_for_krand(cost_data)
    print(f"    Attack flows: {len(attack_flows_krand)}")
    print(f"    Total attack cost: {total_cost_krand:.1f} Mbps "
          f"({mbps_to_pps(total_cost_krand):.0f} pps)")

    constellation = create_constellation()
    router = algo_configs["kRAND"](constellation)

    result = run_simulation(
        router_name="kRAND_attack",
        router=router,
        constellation=constellation,
        attack_flows=attack_flows_krand,
        seed=SEED
    )
    attack_results["kRAND"] = result

    bl = baseline_results["kRAND"]
    print(f"    Normal Delivery: {result['normal_delivery_rate']:.4f} "
          f"(baseline: {bl['normal_delivery_rate']:.4f})")
    print(f"    P5 All-links (normal): {result['normal_p5_throughput_pps']:.2f} pps "
          f"(baseline: {bl['normal_p5_throughput_pps']:.2f} pps)")
    print(f"    P5 Target ISL (normal): {result['target_isl_normal_p5_pps']:.2f} pps "
          f"(baseline: {bl['target_isl_normal_p5_pps']:.2f} pps) "
          f"[{result['target_isl_normal_packets']} pkts on ISL]")
    print(f"    Avg Delay: {result['avg_delay_ms']:.2f} ms "
          f"(baseline: {bl['avg_delay_ms']:.2f} ms)")

    # =========================================================================
    # Step 3: Summary and comparison
    # =========================================================================
    print("\n\n" + "=" * 100)
    print("  SIMULATION RESULTS SUMMARY")
    print("=" * 100)

    # Header
    print(f"\n  {'Algorithm':<10} │ {'Attack Cost':>12} │ {'Normal DR':>10} │ "
          f"{'DR Drop':>8} │ {'NormP5(pps)':>11} │ {'ISLP5(pps)':>10} │ "
          f"{'ISLPkts':>8} │ {'Avg Delay':>10} │ {'Max Link':>9}")
    print(f"  {'─'*10} ┼ {'─'*12} ┼ {'─'*10} ┼ {'─'*8} ┼ {'─'*11} ┼ "
          f"{'─'*10} ┼ {'─'*8} ┼ {'─'*10} ┼ {'─'*9}")

    algos_order = ["KSP", "KDS", "KDG", "KLO", "kRAND"]
    calculated_costs = {
        "KSP": cost_data["algorithm_totals"]["KSP"]["total_cost_mbps"],
        "KDS": cost_data["algorithm_totals"]["KDS"]["total_cost_mbps"],
        "KDG": cost_data["algorithm_totals"]["KDG"]["total_cost_mbps"],
        "KLO": cost_data["algorithm_totals"]["KLO"]["total_cost_mbps"],
        "kRAND": cost_data["krand_total_cost_mbps"],
    }

    for algo in algos_order:
        ar = attack_results[algo]
        bl = baseline_results[algo]
        calc_cost = calculated_costs[algo]
        dr_drop = bl["normal_delivery_rate"] - ar["normal_delivery_rate"]

        print(f"  {algo:<10} │ {calc_cost:>10.1f} Mb │ {ar['normal_delivery_rate']:>10.4f} │ "
              f"{dr_drop:>+8.4f} │ {ar['normal_p5_throughput_pps']:>11.2f} │ "
              f"{ar['target_isl_normal_p5_pps']:>10.2f} │ "
              f"{ar['target_isl_normal_packets']:>8d} │ {ar['avg_delay_ms']:>8.2f}ms │ "
              f"{ar['max_link_utilization']:>9.4f}")

    # =========================================================================
    # Step 4: Save results to JSON
    # =========================================================================
    output = {
        "config": {
            "target_isl": "SAT_4_2 <-> SAT_4_3",
            "isl_bandwidth_mbps": ISL_BW,
            "packet_size_bytes": PACKET_SIZE,
            "num_normal_flows": NUM_NORMAL_FLOWS,
            "normal_rate_range_pps": list(NORMAL_RATE_RANGE),
            "simulation_duration_s": SIM_DURATION,
            "time_step_s": TIME_STEP,
            "seed": SEED,
        },
        "calculated_costs_mbps": calculated_costs,
        "baseline": baseline_results,
        "attack": attack_results,
        "comparison": {},
    }

    for algo in algos_order:
        ar = attack_results[algo]
        bl = baseline_results[algo]
        output["comparison"][algo] = {
            "calculated_cost_mbps": round(calculated_costs[algo], 2),
            "simulated_attack_mbps": round(ar["total_attack_rate_mbps"], 2),
            # All-links normal P5 throughput
            "baseline_normal_dr": round(bl["normal_delivery_rate"], 6),
            "attack_normal_dr": round(ar["normal_delivery_rate"], 6),
            "dr_drop": round(bl["normal_delivery_rate"] - ar["normal_delivery_rate"], 6),
            "baseline_normal_p5_throughput_pps": round(bl["normal_p5_throughput_pps"], 4),
            "attack_normal_p5_throughput_pps": round(ar["normal_p5_throughput_pps"], 4),
            "baseline_normal_p5_throughput_mbps": round(bl["normal_p5_throughput_mbps"], 6),
            "attack_normal_p5_throughput_mbps": round(ar["normal_p5_throughput_mbps"], 6),
            "normal_p5_throughput_drop_pps": round(bl["normal_p5_throughput_pps"] - ar["normal_p5_throughput_pps"], 4),
            # Target ISL normal P5 throughput
            "baseline_target_isl_normal_p5_pps": round(bl["target_isl_normal_p5_pps"], 4),
            "attack_target_isl_normal_p5_pps": round(ar["target_isl_normal_p5_pps"], 4),
            "baseline_target_isl_normal_p5_mbps": round(bl["target_isl_normal_p5_mbps"], 6),
            "attack_target_isl_normal_p5_mbps": round(ar["target_isl_normal_p5_mbps"], 6),
            "target_isl_normal_p5_drop_pps": round(bl["target_isl_normal_p5_pps"] - ar["target_isl_normal_p5_pps"], 4),
            "baseline_target_isl_normal_packets": bl["target_isl_normal_packets"],
            "attack_target_isl_normal_packets": ar["target_isl_normal_packets"],
            # Delay
            "baseline_avg_delay_ms": round(bl["avg_delay_ms"], 4),
            "attack_avg_delay_ms": round(ar["avg_delay_ms"], 4),
            "delay_increase_ms": round(ar["avg_delay_ms"] - bl["avg_delay_ms"], 4),
        }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Results saved to: {OUTPUT_FILE}")
    print("  Done!")


if __name__ == "__main__":
    main()
