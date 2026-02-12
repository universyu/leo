#!/usr/bin/env python3
"""
Targeted ISL Attack Simulation

This script analyzes the vulnerability of four K-path routing algorithms
(KSP, KDS, KDG, KLO) and launches targeted attacks against
the most vulnerable ISL link for each algorithm.

For each router:
1. Analyze the routing algorithm's characteristics to find its weakest ISL link
2. Launch a targeted high-volume attack to congest/block that specific link
3. Measure the impact on normal traffic delivery, latency, and throughput

Attack strategies per router:
- KSP: Attack the link appearing on ALL K shortest paths for the most pairs (path overlap bottleneck)
- KDS: Attack the link appearing in most disjoint path sets (static cache weakness)
- KDG: Attack the critical inter-plane bridge link (geodiverse paths still need plane crossings)
- KLO: Attack the link causing maximum load cascade (force re-routing exhaustion)
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np

from leo_network import (
    LEOConstellation,
    Simulator,
    DDoSAttackGenerator,
    AttackType,
    AttackStrategy,
    KShortestPathsRouter,
    KDSRouter,
    KDGRouter,
    KLORouter,
    print_algorithm_comparison,
)
from leo_network.core.routing import create_router


# ========================= Configuration =========================
NUM_PLANES = 6
SATS_PER_PLANE = 11
ISL_BANDWIDTH_MBPS = 100.0       # Lower bandwidth to observe congestion clearly
NUM_NORMAL_FLOWS = 30
NORMAL_RATE_RANGE = (50, 200)
SIMULATION_DURATION = 1.0        # seconds
TIME_STEP = 0.001                # 1 ms
SEED = 42

# Attack parameters
NUM_ATTACKERS = 40
ATTACK_TOTAL_RATE = 100000.0     # packets/s (enough to saturate 100 Mbps links)
ATTACK_PACKET_SIZE = 1500        # bytes (MTU-sized for maximum bandwidth impact)
NUM_SAMPLE_PAIRS = 300           # Number of src-dst pairs to sample for vulnerability analysis


def create_constellation() -> LEOConstellation:
    """Create a standard LEO constellation for testing"""
    constellation = LEOConstellation(
        num_planes=NUM_PLANES,
        sats_per_plane=SATS_PER_PLANE,
        altitude_km=550.0,
        inclination_deg=53.0,
        isl_bandwidth_mbps=ISL_BANDWIDTH_MBPS,
    )
    return constellation


def run_baseline(constellation: LEOConstellation) -> dict:
    """Run baseline simulation without attacks"""
    print("\n" + "=" * 70)
    print("BASELINE: No Attack")
    print("=" * 70)

    results = {}
    router_configs = [
        ("KSP (k=3)", "ksp", {"k": 3}),
        ("KDS (k=3)", "kds", {"k": 3, "disjoint_type": "link"}),
        ("KDG (k=3)", "kdg", {"k": 3, "diversity_weight": 0.5}),
        ("KLO (k=3)", "klo", {"k": 3, "load_threshold": 0.7}),
    ]

    for name, router_type, kwargs in router_configs:
        # Create fresh constellation for each test (to reset link loads)
        const = create_constellation()
        router = create_router(router_type, const, **kwargs)
        sim = Simulator(constellation=const, router=router, time_step=TIME_STEP, seed=SEED)
        sim.add_random_normal_flows(num_flows=NUM_NORMAL_FLOWS, rate_range=NORMAL_RATE_RANGE)
        sim.run(duration=SIMULATION_DURATION, progress_bar=False)
        res = sim.get_results()
        results[name] = res

        stats = res["statistics"]
        print(f"  {name:<20} Delivery: {stats['overview']['delivery_rate']:.4f}  "
              f"Delay: {stats['delay']['avg_ms']:.2f} ms  "
              f"P5 Throughput: {res['throughput_percentiles']['p5_pps']:.2f} pps")

    return results


def analyze_and_attack_router(
    router_name: str,
    router_type: str,
    router_kwargs: dict,
    baseline_loss_rate: float,
) -> dict:
    """
    For a given router:
    1. Find the most vulnerable ISL link
    2. Launch a targeted congestion attack on that link
    3. Return results
    """
    print(f"\n{'=' * 70}")
    print(f"TARGETED ATTACK: {router_name}")
    print(f"{'=' * 70}")

    # Step 1: Create constellation and router for vulnerability analysis
    const_analysis = create_constellation()
    router_analysis = create_router(router_type, const_analysis, **router_kwargs)

    # Create a temporary attack generator for analysis
    from leo_network.core.traffic import TrafficGenerator
    temp_tg = TrafficGenerator(seed=SEED)
    analyzer = DDoSAttackGenerator(
        constellation=const_analysis,
        traffic_generator=temp_tg,
        seed=SEED,
    )

    # Step 2: Find the most vulnerable ISL link
    print(f"\n  [Phase 1] Analyzing {router_name} vulnerability...")
    src_node, dst_node, count, analysis = analyzer.find_vulnerable_isl_for_router(
        router_analysis, num_sample_pairs=NUM_SAMPLE_PAIRS
    )

    print(f"  Strategy:        {analysis.get('strategy', 'N/A')}")
    print(f"  Target ISL:      {src_node} -> {dst_node}")
    print(f"  Traversal Count: {count} / {NUM_SAMPLE_PAIRS} sampled pairs")
    print(f"  Reason:          {analysis.get('reason', 'N/A')}")
    print(f"  Top 5 links:")
    for link, cnt in analysis.get("top_5_links", []):
        print(f"    {link[0]} -> {link[1]}: {cnt}")

    # Step 3: Create fresh constellation and run targeted attack
    print(f"\n  [Phase 2] Launching targeted ISL congestion attack...")
    const_attack = create_constellation()
    router_attack = create_router(router_type, const_attack, **router_kwargs)
    sim = Simulator(
        constellation=const_attack,
        router=router_attack,
        time_step=TIME_STEP,
        seed=SEED,
    )
    sim.set_baseline_loss_rate(baseline_loss_rate)

    # Add normal traffic
    sim.add_random_normal_flows(num_flows=NUM_NORMAL_FLOWS, rate_range=NORMAL_RATE_RANGE)

    # Create attack generator
    attack_gen = DDoSAttackGenerator(
        constellation=const_attack,
        traffic_generator=sim.traffic_generator,
        seed=SEED,
    )

    # Launch targeted attack
    attack_id = attack_gen.create_targeted_isl_congestion_attack(
        target_link=(src_node, dst_node),
        router=router_attack,
        num_attackers=NUM_ATTACKERS,
        total_rate=ATTACK_TOTAL_RATE,
        packet_size=ATTACK_PACKET_SIZE,
    )

    num_attack_flows = len(attack_gen.attack_flows.get(attack_id, []))
    print(f"  Attack ID:       {attack_id}")
    print(f"  Attack Flows:    {num_attack_flows}")
    print(f"  Total Rate:      {ATTACK_TOTAL_RATE:.0f} pps")
    print(f"  Packet Size:     {ATTACK_PACKET_SIZE} bytes")
    print(f"  Attack BW:       {ATTACK_TOTAL_RATE * ATTACK_PACKET_SIZE * 8 / 1e6:.2f} Mbps")
    print(f"  Link Bandwidth:  {ISL_BANDWIDTH_MBPS:.0f} Mbps")

    # Run simulation
    print(f"\n  [Phase 3] Running simulation...")
    sim.run(duration=SIMULATION_DURATION, progress_bar=True)
    results = sim.get_results()

    # Print results
    stats = results["statistics"]
    print(f"\n  --- Results for {router_name} ---")
    print(f"  Overall Delivery Rate:  {stats['overview']['delivery_rate']:.4f}")
    print(f"  Normal Traffic Delivery: {stats['normal_traffic']['delivery_rate']:.4f}")
    print(f"  Attack Traffic Delivery: {stats['attack_traffic']['delivery_rate']:.4f}")
    print(f"  Avg Delay:              {stats['delay']['avg_ms']:.2f} ms")
    print(f"  P95 Delay:              {stats['delay']['p95_ms']:.2f} ms")
    print(f"  P99 Delay:              {stats['delay']['p99_ms']:.2f} ms")
    print(f"  P5 Throughput:          {results['throughput_percentiles']['p5_pps']:.2f} pps")

    # Attack cost
    cost_data = results["attack_cost"]
    cost_metrics = cost_data["cost_metrics"]
    attack_cost = cost_metrics["attack_cost"]
    if attack_cost == float("inf"):
        print(f"  Attack Cost:            INF (attack ineffective)")
    else:
        print(f"  Attack Cost:            {attack_cost:.2f}")
    print(f"  Normal Loss Rate:       {cost_data['normal_traffic']['loss_rate']:.4%}")
    print(f"  Induced Loss Rate:      {cost_metrics['induced_loss_rate']:.4%}")

    # Check if target link is congested
    target_link_id_fwd = f"ISL_{src_node}_{dst_node}"
    target_link_id_bwd = f"ISL_{dst_node}_{src_node}"
    for lid in [target_link_id_fwd, target_link_id_bwd]:
        link = const_attack.links.get(lid)
        if link:
            util = link.get_utilization()
            print(f"  Target Link [{lid}] Utilization: {util:.4f} ({'BLOCKED' if util >= 1.0 else 'CONGESTED' if util > 0.8 else 'OK'})")

    # Store analysis info in results
    results["attack_analysis"] = {
        "router_name": router_name,
        "target_isl": (src_node, dst_node),
        "strategy": analysis.get("strategy", "N/A"),
        "reason": analysis.get("reason", "N/A"),
        "traversal_count": count,
        "num_attack_flows": num_attack_flows,
    }

    return results


def plot_comparison_chart(baseline_results: dict, attack_results: dict):
    """Generate comparison charts"""
    router_names = list(attack_results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Row 1: Baseline vs Attack ---

    # 1. Normal Traffic Delivery Rate
    ax = axes[0][0]
    baseline_delivery = [baseline_results[name]["statistics"]["normal_traffic"]["delivery_rate"]
                         for name in router_names]
    attack_delivery = [attack_results[name]["statistics"]["normal_traffic"]["delivery_rate"]
                       for name in router_names]
    x = np.arange(len(router_names))
    width = 0.35
    ax.bar(x - width / 2, baseline_delivery, width, label="Baseline", color="green", alpha=0.7)
    ax.bar(x + width / 2, attack_delivery, width, label="Under Attack", color="red", alpha=0.7)
    ax.set_ylabel("Delivery Rate")
    ax.set_title("Normal Traffic Delivery Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(router_names, rotation=15, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Average Delay
    ax = axes[0][1]
    baseline_delay = [baseline_results[name]["statistics"]["delay"]["avg_ms"]
                      for name in router_names]
    attack_delay = [attack_results[name]["statistics"]["delay"]["avg_ms"]
                    for name in router_names]
    ax.bar(x - width / 2, baseline_delay, width, label="Baseline", color="green", alpha=0.7)
    ax.bar(x + width / 2, attack_delay, width, label="Under Attack", color="orange", alpha=0.7)
    ax.set_ylabel("Delay (ms)")
    ax.set_title("Average End-to-End Delay")
    ax.set_xticks(x)
    ax.set_xticklabels(router_names, rotation=15, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 3. 5th Percentile Throughput
    ax = axes[0][2]
    baseline_p5 = [baseline_results[name]["throughput_percentiles"]["p5_pps"]
                   for name in router_names]
    attack_p5 = [attack_results[name]["throughput_percentiles"]["p5_pps"]
                 for name in router_names]
    ax.bar(x - width / 2, baseline_p5, width, label="Baseline", color="green", alpha=0.7)
    ax.bar(x + width / 2, attack_p5, width, label="Under Attack", color="purple", alpha=0.7)
    ax.set_ylabel("Throughput (pps)")
    ax.set_title("5th Percentile Throughput (Worst-Case)")
    ax.set_xticks(x)
    ax.set_xticklabels(router_names, rotation=15, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # --- Row 2: Attack-specific metrics ---

    # 4. Attack Cost (higher = better defense)
    ax = axes[1][0]
    attack_costs = []
    for name in router_names:
        cost = attack_results[name]["attack_cost"]["cost_metrics"]["attack_cost"]
        attack_costs.append(cost if cost != float("inf") else 0)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(router_names)))
    bars = ax.bar(x, attack_costs, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Attack Cost")
    ax.set_title("Attack Cost (Higher = Better Defense)")
    ax.set_xticks(x)
    ax.set_xticklabels(router_names, rotation=15, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 5. Induced Loss Rate (lower = better defense)
    ax = axes[1][1]
    induced_loss = [attack_results[name]["attack_cost"]["cost_metrics"]["induced_loss_rate"]
                    for name in router_names]
    bars = ax.bar(x, [l * 100 for l in induced_loss],
                  color=["red" if l > 0.05 else "orange" if l > 0.02 else "green" for l in induced_loss],
                  alpha=0.7, edgecolor="black")
    ax.set_ylabel("Induced Loss Rate (%)")
    ax.set_title("Attack-Induced Normal Packet Loss Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(router_names, rotation=15, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 6. Delivery Rate Drop (baseline - attack)
    ax = axes[1][2]
    delivery_drop = []
    for i, name in enumerate(router_names):
        drop = baseline_delivery[i] - attack_delivery[i]
        delivery_drop.append(drop * 100)  # Convert to percentage
    bars = ax.bar(x, delivery_drop,
                  color=["darkred" if d > 5 else "orange" if d > 2 else "green" for d in delivery_drop],
                  alpha=0.7, edgecolor="black")
    ax.set_ylabel("Delivery Rate Drop (%)")
    ax.set_title("Normal Traffic Delivery Rate Drop")
    ax.set_xticks(x)
    ax.set_xticklabels(router_names, rotation=15, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Targeted ISL Attack: Router Vulnerability Comparison\n"
        f"(Attack: {ATTACK_TOTAL_RATE:.0f} pps × {ATTACK_PACKET_SIZE}B = "
        f"{ATTACK_TOTAL_RATE * ATTACK_PACKET_SIZE * 8 / 1e6:.0f} Mbps, "
        f"Link BW: {ISL_BANDWIDTH_MBPS:.0f} Mbps)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    fig.savefig("output/targeted_isl_attack_comparison.png", dpi=150)
    print("\n  Saved: output/targeted_isl_attack_comparison.png")
    plt.close("all")


def print_summary_table(baseline_results: dict, attack_results: dict):
    """Print a formatted summary table"""
    print("\n" + "=" * 100)
    print("TARGETED ISL ATTACK — SUMMARY TABLE")
    print("=" * 100)

    header = (f"{'Router':<16} {'Target ISL':<24} {'Strategy':<30} "
              f"{'Normal DR':>10} {'DR Drop':>8} {'Atk Cost':>10} {'P5 TP':>10}")
    print(header)
    print("-" * 100)

    for name, res in attack_results.items():
        analysis = res.get("attack_analysis", {})
        target = analysis.get("target_isl", ("?", "?"))
        strategy = analysis.get("strategy", "?")
        normal_dr = res["statistics"]["normal_traffic"]["delivery_rate"]
        baseline_dr = baseline_results[name]["statistics"]["normal_traffic"]["delivery_rate"]
        dr_drop = (baseline_dr - normal_dr) * 100

        cost = res["attack_cost"]["cost_metrics"]["attack_cost"]
        cost_str = f"{cost:.2f}" if cost != float("inf") else "INF"

        p5 = res["throughput_percentiles"]["p5_pps"]

        target_str = f"{target[0]}->{target[1]}"
        print(f"{name:<16} {target_str:<24} {strategy:<30} "
              f"{normal_dr:>10.4f} {dr_drop:>+7.2f}% {cost_str:>10} {p5:>10.2f}")

    print("=" * 100)

    # Print attack reasoning for each router
    print("\n" + "=" * 100)
    print("ATTACK STRATEGY REASONING")
    print("=" * 100)
    for name, res in attack_results.items():
        analysis = res.get("attack_analysis", {})
        print(f"\n  {name}:")
        print(f"    Target:   {analysis.get('target_isl', '?')}")
        print(f"    Strategy: {analysis.get('strategy', '?')}")
        print(f"    Reason:   {analysis.get('reason', '?')}")
        print(f"    Flows crossing target: {analysis.get('traversal_count', '?')}/{NUM_SAMPLE_PAIRS}")
        print(f"    Attack flows created:  {analysis.get('num_attack_flows', '?')}")


def main():
    print("=" * 70)
    print("Targeted ISL Attack Simulation")
    print("Analyzing 4 K-Path Routing Algorithms' Vulnerability to ISL Congestion")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Constellation:    {NUM_PLANES} planes × {SATS_PER_PLANE} sats = {NUM_PLANES * SATS_PER_PLANE} satellites")
    print(f"  ISL Bandwidth:    {ISL_BANDWIDTH_MBPS} Mbps")
    print(f"  Normal Flows:     {NUM_NORMAL_FLOWS}")
    print(f"  Attack Rate:      {ATTACK_TOTAL_RATE:.0f} pps × {ATTACK_PACKET_SIZE}B = "
          f"{ATTACK_TOTAL_RATE * ATTACK_PACKET_SIZE * 8 / 1e6:.0f} Mbps")
    print(f"  Simulation:       {SIMULATION_DURATION}s, step={TIME_STEP*1000}ms")

    # Phase 1: Run baselines
    print("\n" + "#" * 70)
    print("# PHASE 1: Baseline Simulation (No Attack)")
    print("#" * 70)
    baseline_results = run_baseline(create_constellation())

    # Phase 2: Targeted attacks
    print("\n" + "#" * 70)
    print("# PHASE 2: Targeted ISL Congestion Attacks")
    print("#" * 70)

    router_configs = [
        ("KSP (k=3)", "ksp", {"k": 3}),
        ("KDS (k=3)", "kds", {"k": 3, "disjoint_type": "link"}),
        ("KDG (k=3)", "kdg", {"k": 3, "diversity_weight": 0.5}),
        ("KLO (k=3)", "klo", {"k": 3, "load_threshold": 0.7}),
    ]

    attack_results = {}
    for name, router_type, kwargs in router_configs:
        baseline_loss = 1.0 - baseline_results[name]["statistics"]["normal_traffic"]["delivery_rate"]
        result = analyze_and_attack_router(
            router_name=name,
            router_type=router_type,
            router_kwargs=kwargs,
            baseline_loss_rate=baseline_loss,
        )
        attack_results[name] = result

    # Phase 3: Comparison
    print("\n" + "#" * 70)
    print("# PHASE 3: Comparison & Visualization")
    print("#" * 70)

    print_summary_table(baseline_results, attack_results)

    # Use the built-in comparison if available
    print_algorithm_comparison(attack_results)

    # Generate plots
    plot_comparison_chart(baseline_results, attack_results)

    print("\n" + "=" * 70)
    print("Targeted ISL Attack Simulation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
