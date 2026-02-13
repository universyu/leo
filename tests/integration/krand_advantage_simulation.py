#!/usr/bin/env python3
"""
k-RAND Advantage Verification Simulation

This script validates the paper's core thesis: adding randomness in routing
algorithm selection (k-RAND) increases the attacker's cost and/or MaxUp,
making DDoS attacks on LEO satellite ISL links harder to execute.

Experiment Design:
==================
1. Target ISL Selection: Pick the network's most vulnerable ISL link
2. Attack Model: Attacker knows the routing algorithm + topology,
   selects GS pairs whose routes pass through the target ISL
3. Comparison: For each algorithm (KSP, KDS, KDG, KLO, k-RAND),
   measure how much attack traffic is needed to congest the target ISL

Key Metrics (from the paper):
- Cost: Total attack traffic volume (Gbps) needed to congest the target ISL
- MaxUp: Maximum uplink bandwidth change at any single ground station
- Normal traffic delivery rate under attack
- 5th percentile throughput (worst-case network performance)
"""

import sys
import os
import time
import numpy as np
from collections import Counter, defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from leo_network import (
    LEOConstellation,
    TrafficGenerator,
    KShortestPathsRouter,
    KDSRouter,
    KDGRouter,
    KLORouter,
    Simulator,
    DDoSAttackGenerator,
    AttackType,
    AttackStrategy,
    create_router,
)
from leo_network.core.routing import KRandRouter
from leo_network.core.topology import LinkType


# =============================================================================
# Part 1: Theoretical Analysis - Path Diversity & Attack Cost
# =============================================================================

def analyze_path_diversity(constellation, target_isl, k=3):
    """
    For a given target ISL link, analyze how many GS pairs' routes
    pass through it under each algorithm, and compute the theoretical
    attack cost and MaxUp.

    The key insight: if an algorithm has more diverse paths, fewer of
    them will go through the target ISL, so the attacker needs to send
    more traffic to ensure congestion.
    """
    print("\n" + "=" * 70)
    print("  PART 1: Path Diversity & Theoretical Attack Cost Analysis")
    print("=" * 70)

    link_src, link_dst = target_isl
    print(f"\n  Target ISL: {link_src} <-> {link_dst}")

    # Get all GS nodes
    gs_nodes = sorted([n for n in constellation.graph.nodes() if n.startswith("GS_")])
    print(f"  Ground stations: {len(gs_nodes)}")

    # Create routers
    routers = {
        "KSP": KShortestPathsRouter(constellation, k=k),
        "KDS": KDSRouter(constellation, k=k),
        "KDG": KDGRouter(constellation, k=k),
        "KLO": KLORouter(constellation, k=k),
    }

    # Pre-compute routes
    for name, router in routers.items():
        router.precompute_ground_station_routes()

    results = {}

    for algo_name, router in routers.items():
        # For each GS pair, check if its route passes through the target ISL
        pairs_through_target = []
        pairs_not_through = []
        all_isl_links_used = set()

        for src in gs_nodes:
            for dst in gs_nodes:
                if src == dst:
                    continue
                entry = router.routing_table.get(src, {}).get(dst)
                if entry is None or not entry.valid:
                    continue

                path = entry.path
                passes_target = False
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    if a.startswith("SAT_") and b.startswith("SAT_"):
                        all_isl_links_used.add((min(a, b), max(a, b)))
                    if (a == link_src and b == link_dst) or (a == link_dst and b == link_src):
                        passes_target = True

                if passes_target:
                    pairs_through_target.append((src, dst))
                else:
                    pairs_not_through.append((src, dst))

        # Now compute attack-related metrics
        # For k-path algorithms, get ALL k paths and check how many go through target
        all_k_paths_through = 0
        all_k_paths_total = 0
        unique_isls_across_k_paths = set()

        for src in gs_nodes:
            for dst in gs_nodes:
                if src == dst:
                    continue
                # Get all k paths
                if isinstance(router, KShortestPathsRouter):
                    paths = router.compute_k_paths(src, dst)
                elif isinstance(router, KDSRouter):
                    paths = router.compute_k_disjoint_paths(src, dst)
                elif isinstance(router, KDGRouter):
                    paths = router.compute_k_geodiverse_paths(src, dst)
                elif isinstance(router, KLORouter):
                    paths = router.get_all_disjoint_paths(src, dst)
                else:
                    paths = []

                for path in paths:
                    all_k_paths_total += 1
                    for i in range(len(path) - 1):
                        a, b = path[i], path[i + 1]
                        if a.startswith("SAT_") and b.startswith("SAT_"):
                            unique_isls_across_k_paths.add((min(a, b), max(a, b)))
                        if (a == link_src and b == link_dst) or (a == link_dst and b == link_src):
                            all_k_paths_through += 1
                            break

        # Probability a random k-path goes through the target ISL
        p_through = all_k_paths_through / max(1, all_k_paths_total)

        results[algo_name] = {
            "pairs_through": len(pairs_through_target),
            "pairs_total": len(pairs_through_target) + len(pairs_not_through),
            "k_paths_through": all_k_paths_through,
            "k_paths_total": all_k_paths_total,
            "p_through": p_through,
            "unique_isls": len(unique_isls_across_k_paths),
            "pairs_through_list": pairs_through_target,
        }

    # k-RAND: union of all paths across all algorithms
    krand_all_paths = set()
    krand_through = 0
    krand_total = 0
    for algo_name, res in results.items():
        krand_through += res["k_paths_through"]
        krand_total += res["k_paths_total"]

    # k-RAND path diversity = union of all ISLs used across all algorithms
    krand_unique_isls = set()
    for res in results.values():
        # We need to recompute this properly
        pass
    # Simpler: sum unique ISLs
    for algo_name, router in routers.items():
        for src in gs_nodes:
            for dst in gs_nodes:
                if src == dst:
                    continue
                if isinstance(router, KShortestPathsRouter):
                    paths = router.compute_k_paths(src, dst)
                elif isinstance(router, KDSRouter):
                    paths = router.compute_k_disjoint_paths(src, dst)
                elif isinstance(router, KDGRouter):
                    paths = router.compute_k_geodiverse_paths(src, dst)
                elif isinstance(router, KLORouter):
                    paths = router.get_all_disjoint_paths(src, dst)
                else:
                    paths = []
                for path in paths:
                    for i in range(len(path) - 1):
                        a, b = path[i], path[i + 1]
                        if a.startswith("SAT_") and b.startswith("SAT_"):
                            krand_unique_isls.add((min(a, b), max(a, b)))

    # Average probability for k-RAND (weighted equally)
    p_through_krand = krand_through / max(1, krand_total)

    results["k-RAND"] = {
        "pairs_through": sum(r["pairs_through"] for r in results.values()) // len(results),
        "pairs_total": list(results.values())[0]["pairs_total"],
        "k_paths_through": krand_through,
        "k_paths_total": krand_total,
        "p_through": p_through_krand,
        "unique_isls": len(krand_unique_isls),
        "pairs_through_list": [],
    }

    # Print results
    print(f"\n  {'Algorithm':<10} {'Routes Through':<16} {'K-Paths Through':<18} "
          f"{'Total K-Paths':<15} {'P(through)':<12} {'Unique ISLs':<12}")
    print(f"  {'-' * 85}")

    isl_bandwidth_mbps = 100.0  # From the constellation config
    for algo_name in ["KSP", "KDS", "KDG", "KLO", "k-RAND"]:
        r = results[algo_name]
        print(f"  {algo_name:<10} {r['pairs_through']:<16} {r['k_paths_through']:<18} "
              f"{r['k_paths_total']:<15} {r['p_through']:<12.4f} {r['unique_isls']:<12}")

    # Theoretical attack cost
    print(f"\n  --- Theoretical Attack Cost (to congest target ISL at {isl_bandwidth_mbps} Mbps) ---")
    print(f"  {'Algorithm':<10} {'P(through target)':<20} {'Required Traffic':<20} {'Cost Factor':<15}")
    print(f"  {'-' * 65}")

    baseline_cost = None
    for algo_name in ["KSP", "KDS", "KDG", "KLO", "k-RAND"]:
        r = results[algo_name]
        p = r["p_through"]
        if p > 0:
            # To congest the target ISL, attacker needs to send enough traffic
            # such that traffic * p >= ISL capacity
            required_traffic = isl_bandwidth_mbps / p
        else:
            required_traffic = float('inf')

        if baseline_cost is None:
            baseline_cost = required_traffic

        cost_factor = required_traffic / baseline_cost if baseline_cost > 0 else 0

        print(f"  {algo_name:<10} {p:<20.4f} {required_traffic:<20.1f} Mbps  {cost_factor:<15.2f}x")

    return results


# =============================================================================
# Part 2: Per-GS-Pair Path Coverage Analysis
# =============================================================================

def analyze_per_pair_coverage(constellation, target_isl, k=3, sample_pairs=50):
    """
    For a sample of GS pairs, show how many of the k paths from each
    algorithm pass through the target ISL. This illustrates why k-RAND
    is harder to attack: the attacker must cover paths from ALL algorithms.
    """
    print("\n" + "=" * 70)
    print("  PART 2: Per-GS-Pair Path Coverage (Why k-RAND Is Harder to Attack)")
    print("=" * 70)

    link_src, link_dst = target_isl
    gs_nodes = sorted([n for n in constellation.graph.nodes() if n.startswith("GS_")])
    rng = np.random.default_rng(42)

    routers = {
        "KSP": KShortestPathsRouter(constellation, k=k),
        "KDS": KDSRouter(constellation, k=k),
        "KDG": KDGRouter(constellation, k=k),
        "KLO": KLORouter(constellation, k=k),
    }

    # Find GS pairs whose routes go through the target ISL (for at least one algo)
    interesting_pairs = []
    for src in gs_nodes:
        for dst in gs_nodes:
            if src == dst:
                continue
            for name, router in routers.items():
                if isinstance(router, KShortestPathsRouter):
                    paths = router.compute_k_paths(src, dst)
                elif isinstance(router, KDSRouter):
                    paths = router.compute_k_disjoint_paths(src, dst)
                elif isinstance(router, KDGRouter):
                    paths = router.compute_k_geodiverse_paths(src, dst)
                elif isinstance(router, KLORouter):
                    paths = router.get_all_disjoint_paths(src, dst)
                else:
                    paths = []

                for path in paths:
                    for i in range(len(path) - 1):
                        a, b = path[i], path[i + 1]
                        if (a == link_src and b == link_dst) or (a == link_dst and b == link_src):
                            interesting_pairs.append((src, dst))
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

    # Deduplicate
    interesting_pairs = list(set(interesting_pairs))
    rng.shuffle(interesting_pairs)
    selected_pairs = interesting_pairs[:min(sample_pairs, len(interesting_pairs))]

    print(f"\n  Showing {len(selected_pairs)} GS pairs whose routes touch target ISL")
    print(f"  For each pair, showing how many of the k={k} paths pass through the target ISL")
    print()
    print(f"  {'GS Pair':<40} {'KSP':<10} {'KDS':<10} {'KDG':<10} {'KLO':<10} {'Total Paths':<12} {'Union ISLs'}")
    print(f"  {'-' * 105}")

    total_ksp_through = 0
    total_kds_through = 0
    total_kdg_through = 0
    total_klo_through = 0
    total_unique_isls_all = set()

    for src, dst in selected_pairs[:20]:  # Show first 20
        pair_str = f"{src} -> {dst}"
        algo_through = {}
        pair_isls = set()

        for name, router in routers.items():
            if isinstance(router, KShortestPathsRouter):
                paths = router.compute_k_paths(src, dst)
            elif isinstance(router, KDSRouter):
                paths = router.compute_k_disjoint_paths(src, dst)
            elif isinstance(router, KDGRouter):
                paths = router.compute_k_geodiverse_paths(src, dst)
            elif isinstance(router, KLORouter):
                paths = router.get_all_disjoint_paths(src, dst)
            else:
                paths = []

            through_count = 0
            for path in paths:
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    if a.startswith("SAT_") and b.startswith("SAT_"):
                        pair_isls.add((min(a, b), max(a, b)))
                    if (a == link_src and b == link_dst) or (a == link_dst and b == link_src):
                        through_count += 1
                        break

            algo_through[name] = f"{through_count}/{len(paths)}"
            if name == "KSP":
                total_ksp_through += through_count
            elif name == "KDS":
                total_kds_through += through_count
            elif name == "KDG":
                total_kdg_through += through_count
            elif name == "KLO":
                total_klo_through += through_count

        total_paths = sum(
            len(router.compute_k_paths(src, dst)) if isinstance(router, KShortestPathsRouter)
            else len(router.compute_k_disjoint_paths(src, dst)) if isinstance(router, KDSRouter)
            else len(router.compute_k_geodiverse_paths(src, dst)) if isinstance(router, KDGRouter)
            else len(router.get_all_disjoint_paths(src, dst))
            for router in routers.values()
        )
        total_unique_isls_all.update(pair_isls)

        print(f"  {pair_str:<40} {algo_through['KSP']:<10} {algo_through['KDS']:<10} "
              f"{algo_through['KDG']:<10} {algo_through['KLO']:<10} {total_paths:<12} {len(pair_isls)}")

    print(f"\n  Key Insight: For k-RAND, the attacker must cover paths from ALL 4 algorithms.")
    print(f"  Even if KSP has 3/3 paths through the target, KDS might have only 1/3,")
    print(f"  meaning the attacker's probability of hitting the target drops to ~(3+1+x+y)/(3+3+3+3)")


# =============================================================================
# Part 3: Actual Packet-Level Simulation
# =============================================================================

def run_simulation_with_targeted_attack(
    constellation,
    router,
    target_isl,
    attack_rate,
    num_normal_flows=30,
    normal_rate_range=(50, 200),
    num_attackers=30,
    sim_duration=1.0,
    seed=42
):
    """
    Run a complete simulation with normal traffic + targeted ISL attack.

    The attacker sends traffic specifically designed to pass through
    the target ISL link, simulating a knowledgeable adversary.
    """
    sim = Simulator(
        constellation=constellation,
        router=router,
        time_step=0.001,
        seed=seed
    )

    # Add normal traffic between random GS pairs
    sim.add_random_normal_flows(
        num_flows=num_normal_flows,
        rate_range=normal_rate_range
    )

    # Create attack generator
    attack_gen = DDoSAttackGenerator(
        constellation=constellation,
        traffic_generator=sim.traffic_generator,
        seed=seed
    )

    # Create targeted attack: find GS pairs whose routes pass through target ISL
    attack_id = attack_gen.create_targeted_isl_congestion_attack(
        target_link=target_isl,
        router=router,
        num_attackers=num_attackers,
        total_rate=attack_rate,
        packet_size=1000,
        start_time=0.0,
        duration=-1.0
    )

    # Run simulation
    sim.run(duration=sim_duration, progress_bar=False)

    # Get results
    results = sim.get_results()
    stats = results["statistics"]

    return {
        "delivery_rate": stats["overview"]["delivery_rate"],
        "normal_delivery_rate": stats["normal_traffic"]["delivery_rate"],
        "attack_delivery_rate": stats["attack_traffic"]["delivery_rate"],
        "avg_delay_ms": stats["delay"]["avg_ms"],
        "normal_delivered": stats["normal_traffic"]["delivered"],
        "normal_dropped": stats["normal_traffic"]["dropped"],
        "attack_delivered": stats["attack_traffic"]["delivered"],
        "attack_dropped": stats["attack_traffic"]["dropped"],
        "p5_throughput_mbps": results["throughput_percentiles"]["p5_mbps"],
        "attack_cost": results["attack_cost"],
    }


def run_comparative_simulations(constellation, target_isl, k=3):
    """
    Run comparative simulations across all 5 routing algorithms
    with varying attack intensities.
    """
    print("\n" + "=" * 70)
    print("  PART 3: Packet-Level Simulation - Attack Impact Comparison")
    print("=" * 70)

    link_src, link_dst = target_isl
    print(f"\n  Target ISL: {link_src} <-> {link_dst}")

    # Attack rate levels (packets per second)
    # With 1000-byte packets: 10000 pps = 80 Mbps, 50000 pps = 400 Mbps
    attack_rates = [5000, 10000, 20000, 40000, 60000, 80000]
    attack_mbps = [r * 1000 * 8 / 1e6 for r in attack_rates]

    algo_configs = [
        ("KSP", {"router_type": "ksp", "k": k}),
        ("KDS", {"router_type": "kds", "k": k}),
        ("KDG", {"router_type": "kdg", "k": k}),
        ("KLO", {"router_type": "klo", "k": k}),
        ("k-RAND", {"router_type": "krand", "k": k, "seed": 42}),
    ]

    all_results = {}

    for algo_name, algo_kwargs in algo_configs:
        print(f"\n  --- Running simulations for {algo_name} ---")
        all_results[algo_name] = {"rates": [], "normal_delivery": [], "avg_delay": [],
                                   "p5_throughput": []}

        for rate in attack_rates:
            rate_mbps = rate * 1000 * 8 / 1e6
            print(f"    Attack rate: {rate} pps ({rate_mbps:.0f} Mbps)...", end=" ")

            # Create fresh constellation for each run
            fresh_constellation = LEOConstellation(
                num_planes=6, sats_per_plane=11,
                altitude_km=550.0, inclination_deg=53.0,
                isl_bandwidth_mbps=100.0
            )
            fresh_constellation.add_global_ground_stations()

            router = create_router(algo_kwargs["router_type"], fresh_constellation,
                                   **{k_: v_ for k_, v_ in algo_kwargs.items()
                                      if k_ != "router_type"})

            result = run_simulation_with_targeted_attack(
                constellation=fresh_constellation,
                router=router,
                target_isl=target_isl,
                attack_rate=rate,
                num_normal_flows=30,
                sim_duration=0.5,
                seed=42
            )

            all_results[algo_name]["rates"].append(rate_mbps)
            all_results[algo_name]["normal_delivery"].append(result["normal_delivery_rate"])
            all_results[algo_name]["avg_delay"].append(result["avg_delay_ms"])
            all_results[algo_name]["p5_throughput"].append(result["p5_throughput_mbps"])

            print(f"Normal DR={result['normal_delivery_rate']:.4f}, "
                  f"Delay={result['avg_delay_ms']:.1f}ms")

    return all_results, attack_mbps


# =============================================================================
# Part 4: Baseline (No Attack) Simulation
# =============================================================================

def run_baseline_simulations(k=3):
    """Run baseline simulations (no attack) for all algorithms."""
    print("\n" + "=" * 70)
    print("  PART 4: Baseline Performance (No Attack)")
    print("=" * 70)

    algo_configs = [
        ("KSP", {"router_type": "ksp", "k": k}),
        ("KDS", {"router_type": "kds", "k": k}),
        ("KDG", {"router_type": "kdg", "k": k}),
        ("KLO", {"router_type": "klo", "k": k}),
        ("k-RAND", {"router_type": "krand", "k": k, "seed": 42}),
    ]

    baselines = {}

    for algo_name, algo_kwargs in algo_configs:
        print(f"  Running baseline for {algo_name}...", end=" ")

        constellation = LEOConstellation(
            num_planes=6, sats_per_plane=11,
            altitude_km=550.0, inclination_deg=53.0,
            isl_bandwidth_mbps=100.0
        )
        constellation.add_global_ground_stations()

        router = create_router(algo_kwargs["router_type"], constellation,
                               **{k_: v_ for k_, v_ in algo_kwargs.items()
                                  if k_ != "router_type"})

        sim = Simulator(
            constellation=constellation,
            router=router,
            time_step=0.001,
            seed=42
        )
        sim.add_random_normal_flows(num_flows=30, rate_range=(50, 200))
        sim.run(duration=0.5, progress_bar=False)

        results = sim.get_results()
        stats = results["statistics"]
        baselines[algo_name] = {
            "delivery_rate": stats["normal_traffic"]["delivery_rate"],
            "avg_delay_ms": stats["delay"]["avg_ms"],
            "p5_throughput": results["throughput_percentiles"]["p5_mbps"],
        }

        print(f"DR={baselines[algo_name]['delivery_rate']:.4f}, "
              f"Delay={baselines[algo_name]['avg_delay_ms']:.1f}ms")

    return baselines


# =============================================================================
# Part 5: Plotting & Summary
# =============================================================================

def plot_results(all_results, attack_mbps, baselines, target_isl, output_dir="output"):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    algo_colors = {
        "KSP": "#e74c3c",
        "KDS": "#3498db",
        "KDG": "#2ecc71",
        "KLO": "#f39c12",
        "k-RAND": "#9b59b6",
    }
    algo_markers = {
        "KSP": "o", "KDS": "s", "KDG": "^", "KLO": "D", "k-RAND": "*",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Normal Traffic Delivery Rate vs Attack Intensity
    ax = axes[0]
    for algo_name in ["KSP", "KDS", "KDG", "KLO", "k-RAND"]:
        r = all_results[algo_name]
        linewidth = 3 if algo_name == "k-RAND" else 1.5
        markersize = 12 if algo_name == "k-RAND" else 7
        ax.plot(r["rates"], r["normal_delivery"],
                color=algo_colors[algo_name],
                marker=algo_markers[algo_name],
                linewidth=linewidth,
                markersize=markersize,
                label=algo_name,
                zorder=10 if algo_name == "k-RAND" else 5)

    ax.set_xlabel("Attack Traffic (Mbps)", fontsize=12)
    ax.set_ylabel("Normal Traffic Delivery Rate", fontsize=12)
    ax.set_title("Normal Traffic Delivery Under DDoS Attack", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Plot 2: Average Delay vs Attack Intensity
    ax = axes[1]
    for algo_name in ["KSP", "KDS", "KDG", "KLO", "k-RAND"]:
        r = all_results[algo_name]
        linewidth = 3 if algo_name == "k-RAND" else 1.5
        markersize = 12 if algo_name == "k-RAND" else 7
        ax.plot(r["rates"], r["avg_delay"],
                color=algo_colors[algo_name],
                marker=algo_markers[algo_name],
                linewidth=linewidth,
                markersize=markersize,
                label=algo_name,
                zorder=10 if algo_name == "k-RAND" else 5)

    ax.set_xlabel("Attack Traffic (Mbps)", fontsize=12)
    ax.set_ylabel("Average Delay (ms)", fontsize=12)
    ax.set_title("Network Delay Under DDoS Attack", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: 5th Percentile Throughput (worst-case)
    ax = axes[2]
    for algo_name in ["KSP", "KDS", "KDG", "KLO", "k-RAND"]:
        r = all_results[algo_name]
        linewidth = 3 if algo_name == "k-RAND" else 1.5
        markersize = 12 if algo_name == "k-RAND" else 7
        ax.plot(r["rates"], r["p5_throughput"],
                color=algo_colors[algo_name],
                marker=algo_markers[algo_name],
                linewidth=linewidth,
                markersize=markersize,
                label=algo_name,
                zorder=10 if algo_name == "k-RAND" else 5)

    ax.set_xlabel("Attack Traffic (Mbps)", fontsize=12)
    ax.set_ylabel("5th Percentile Throughput (Mbps)", fontsize=12)
    ax.set_title("Worst-Case Throughput Under DDoS Attack", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    link_str = f"{target_isl[0]}↔{target_isl[1]}"
    fig.suptitle(f"k-RAND vs Fixed Routing: DDoS Attack Resilience\n"
                 f"Target ISL: {link_str}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "krand_advantage_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot: {path}")
    plt.close('all')

    # Bar chart: delivery rate at max attack
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    algo_names = ["KSP", "KDS", "KDG", "KLO", "k-RAND"]
    # Take the result at the highest attack rate
    max_rate_idx = -1
    delivery_at_max = [all_results[a]["normal_delivery"][max_rate_idx] for a in algo_names]
    colors_bar = [algo_colors[a] for a in algo_names]

    bars = ax2.bar(algo_names, delivery_at_max, color=colors_bar, alpha=0.8, edgecolor='black')
    # Highlight k-RAND
    bars[-1].set_edgecolor('#9b59b6')
    bars[-1].set_linewidth(3)

    ax2.set_ylabel("Normal Traffic Delivery Rate", fontsize=12)
    max_rate_mbps = all_results["KSP"]["rates"][max_rate_idx]
    ax2.set_title(f"Normal Delivery Rate at Max Attack ({max_rate_mbps:.0f} Mbps)\n"
                  f"Higher = Better Defense", fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, delivery_at_max):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add baseline reference
    for i, algo in enumerate(algo_names):
        if algo in baselines:
            ax2.plot([i - 0.3, i + 0.3],
                     [baselines[algo]["delivery_rate"]] * 2,
                     'k--', linewidth=1, alpha=0.5)

    path2 = os.path.join(output_dir, "krand_advantage_bar.png")
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {path2}")
    plt.close('all')


def print_summary(all_results, baselines, attack_mbps, theoretical_results):
    """Print comprehensive summary table."""
    print("\n\n" + "#" * 70)
    print("  FINAL SUMMARY: k-RAND Advantage Verification")
    print("#" * 70)

    algo_names = ["KSP", "KDS", "KDG", "KLO", "k-RAND"]

    # Summary table
    print(f"\n  {'Metric':<30}", end="")
    for algo in algo_names:
        print(f" {algo:>10}", end="")
    print()
    print(f"  {'-' * 80}")

    # Baseline delivery
    print(f"  {'Baseline Delivery Rate':<30}", end="")
    for algo in algo_names:
        if algo in baselines:
            print(f" {baselines[algo]['delivery_rate']:>10.4f}", end="")
        else:
            print(f" {'N/A':>10}", end="")
    print()

    # Delivery at each attack level
    for i, rate in enumerate(attack_mbps):
        print(f"  {f'DR @ {rate:.0f} Mbps attack':<30}", end="")
        for algo in algo_names:
            val = all_results[algo]["normal_delivery"][i]
            print(f" {val:>10.4f}", end="")
        print()

    # Theoretical P(through target)
    print(f"\n  {'Theoretical P(through ISL)':<30}", end="")
    for algo in algo_names:
        if algo in theoretical_results:
            p = theoretical_results[algo]["p_through"]
            print(f" {p:>10.4f}", end="")
        else:
            print(f" {'N/A':>10}", end="")
    print()

    # Theoretical cost factor
    costs = {}
    baseline_p = theoretical_results.get("KSP", {}).get("p_through", 1)
    for algo in algo_names:
        if algo in theoretical_results:
            p = theoretical_results[algo]["p_through"]
            if p > 0:
                costs[algo] = (1.0 / p) / (1.0 / baseline_p) if baseline_p > 0 else 0
            else:
                costs[algo] = float('inf')

    print(f"  {'Theoretical Cost Factor':<30}", end="")
    for algo in algo_names:
        if algo in costs:
            print(f" {costs[algo]:>10.2f}x", end="")
        else:
            print(f" {'N/A':>10}", end="")
    print()

    # Find the attack rate where delivery drops below 0.9 for each algo
    print(f"\n  {'Rate for DR < 0.9 (Mbps)':<30}", end="")
    for algo in algo_names:
        threshold_rate = "N/A"
        for i, dr in enumerate(all_results[algo]["normal_delivery"]):
            if dr < 0.9:
                threshold_rate = f"{all_results[algo]['rates'][i]:.0f}"
                break
        if threshold_rate == "N/A":
            threshold_rate = f">{all_results[algo]['rates'][-1]:.0f}"
        print(f" {threshold_rate:>10}", end="")
    print()

    print(f"\n  {'Rate for DR < 0.5 (Mbps)':<30}", end="")
    for algo in algo_names:
        threshold_rate = "N/A"
        for i, dr in enumerate(all_results[algo]["normal_delivery"]):
            if dr < 0.5:
                threshold_rate = f"{all_results[algo]['rates'][i]:.0f}"
                break
        if threshold_rate == "N/A":
            threshold_rate = f">{all_results[algo]['rates'][-1]:.0f}"
        print(f" {threshold_rate:>10}", end="")
    print()

    # Conclusion
    print(f"\n\n  {'=' * 70}")
    print(f"  CONCLUSION")
    print(f"  {'=' * 70}")

    # Compare k-RAND with best fixed algorithm at highest attack rate
    max_idx = -1
    krand_dr = all_results["k-RAND"]["normal_delivery"][max_idx]
    best_fixed = max(all_results[a]["normal_delivery"][max_idx] for a in ["KSP", "KDS", "KDG", "KLO"])
    worst_fixed = min(all_results[a]["normal_delivery"][max_idx] for a in ["KSP", "KDS", "KDG", "KLO"])

    print(f"\n  At maximum attack intensity ({all_results['KSP']['rates'][max_idx]:.0f} Mbps):")
    print(f"    k-RAND normal delivery:      {krand_dr:.4f}")
    print(f"    Best fixed algo delivery:    {best_fixed:.4f}")
    print(f"    Worst fixed algo delivery:   {worst_fixed:.4f}")

    if krand_dr > worst_fixed:
        improvement = (krand_dr - worst_fixed) / worst_fixed * 100 if worst_fixed > 0 else float('inf')
        print(f"\n  ✅ k-RAND improves over worst fixed algorithm by {improvement:.1f}%")
    else:
        print(f"\n  ⚠️ k-RAND did not outperform all fixed algorithms at this attack level")

    krand_p = theoretical_results.get("k-RAND", {}).get("p_through", 0)
    ksp_p = theoretical_results.get("KSP", {}).get("p_through", 0)
    if ksp_p > 0 and krand_p > 0:
        cost_multiplier = ksp_p / krand_p
        print(f"\n  📊 Theoretical attack cost multiplier for k-RAND: {cost_multiplier:.2f}x")
        print(f"     (Attacker needs {cost_multiplier:.2f}x more traffic to achieve same congestion)")

    print(f"\n  Key findings per the paper's theory:")
    print(f"    1. Randomness ↑ → Attacker uncertainty ↑ → Cost ↑ or MaxUp ↑")
    print(f"    2. k-RAND forces attacker to cover ALL possible routing paths")
    print(f"    3. Fixed algorithms allow targeted attacks with minimal resources")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  k-RAND Advantage Verification Simulation")
    print("  Paper: Random Routing for DDoS Defense in LEO Satellite Networks")
    print("=" * 70)

    start_time = time.time()

    # Create constellation
    print("\n[1] Creating LEO Constellation...")
    constellation = LEOConstellation(
        num_planes=6,
        sats_per_plane=11,
        altitude_km=550.0,
        inclination_deg=53.0,
        isl_bandwidth_mbps=100.0
    )
    constellation.add_global_ground_stations()

    print(f"  Satellites: {len(constellation.satellites)}")
    print(f"  Ground Stations: {len(constellation.ground_stations)}")
    print(f"  Links: {len(constellation.links)}")

    # Target ISL: the most vulnerable link from our previous analysis
    target_isl = ("SAT_4_2", "SAT_4_3")
    print(f"\n  Attack Target ISL: {target_isl[0]} <-> {target_isl[1]}")

    # Part 1: Theoretical analysis
    print("\n[2] Running theoretical analysis...")
    theoretical_results = analyze_path_diversity(constellation, target_isl, k=3)

    # Part 2: Per-pair coverage
    print("\n[3] Analyzing per-pair path coverage...")
    analyze_per_pair_coverage(constellation, target_isl, k=3)

    # Part 4: Baseline (run before attack simulations)
    print("\n[4] Running baseline simulations (no attack)...")
    baselines = run_baseline_simulations(k=3)

    # Part 3: Actual packet simulation
    print("\n[5] Running attack simulations...")
    all_results, attack_mbps = run_comparative_simulations(constellation, target_isl, k=3)

    # Part 5: Plot and summarize
    print("\n[6] Generating plots and summary...")
    plot_results(all_results, attack_mbps, baselines, target_isl)
    print_summary(all_results, baselines, attack_mbps, theoretical_results)

    elapsed = time.time() - start_time
    print(f"\n  Total execution time: {elapsed:.1f}s")
    print("\n  Simulation complete!")


if __name__ == "__main__":
    main()
