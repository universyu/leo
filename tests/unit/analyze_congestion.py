"""
Congestion Analysis: Find the most congested ISL links and
the most vulnerable ground station pairs for each routing algorithm.

For each of the 4 routing algorithms (ksp, kds, kdg, klo):
1. Pre-compute routes for all 38*37=1406 GS pairs
2. Count how many routes traverse each ISL link
3. Identify the top bottleneck ISL links
4. Identify which GS pairs share the most bottleneck links
"""

import sys
import time
from collections import defaultdict, Counter
sys.path.insert(0, '/Users/selenli/Desktop/graduate_code')

from leo_network import LEOConstellation
from leo_network.core.routing import create_router
from leo_network.core.topology import LinkType


def analyze_router(constellation, router_type, k=3):
    """Analyze congestion for a given routing algorithm."""
    
    print(f"\n{'='*70}")
    print(f"  Routing Algorithm: {router_type.upper()} (k={k})")
    print(f"{'='*70}")
    
    # Create router and pre-compute routes
    router = create_router(router_type, constellation, k=k)
    
    start = time.time()
    router.precompute_ground_station_routes()
    elapsed = time.time() - start
    print(f"  Pre-computation time: {elapsed:.2f}s")
    
    gs_nodes = sorted([
        node for node in constellation.graph.nodes()
        if node.startswith("GS_")
    ])
    
    # ----------------------------------------------------------------
    # 1. Count how many routes pass through each ISL link
    # ----------------------------------------------------------------
    link_usage = Counter()       # (src, dst) -> count of routes using this link
    link_type_map = {}           # (src, dst) -> LinkType
    gs_pair_links = {}           # (gs_src, gs_dst) -> list of ISL links used
    
    total_routes = 0
    for src in gs_nodes:
        for dst in gs_nodes:
            if src == dst:
                continue
            entry = router.routing_table.get(src, {}).get(dst)
            if entry is None or not entry.valid:
                continue
            
            total_routes += 1
            path = entry.path
            isl_links_in_path = []
            
            for i in range(len(path) - 1):
                node_a, node_b = path[i], path[i+1]
                # Normalize link direction for counting
                link_key = (min(node_a, node_b), max(node_a, node_b))
                
                # Only count ISL links (SAT <-> SAT), not GSL
                if node_a.startswith("SAT_") and node_b.startswith("SAT_"):
                    link_usage[link_key] += 1
                    isl_links_in_path.append(link_key)
                    
                    # Record link type
                    if link_key not in link_type_map:
                        link = constellation.get_link(node_a, node_b)
                        if link:
                            link_type_map[link_key] = link.link_type
            
            gs_pair_links[(src, dst)] = isl_links_in_path
    
    print(f"\n  Total valid routes: {total_routes}")
    print(f"  Total unique ISL links used: {len(link_usage)}")
    
    # ----------------------------------------------------------------
    # 2. Top 15 most used ISL links (bottleneck links)
    # ----------------------------------------------------------------
    print(f"\n  --- Top 15 Bottleneck ISL Links ---")
    print(f"  {'Rank':<5} {'Link':<30} {'Type':<12} {'Route Count':<12} {'% of Routes'}")
    print(f"  {'-'*75}")
    
    top_links = link_usage.most_common(15)
    bottleneck_links = set()
    for rank, (link_key, count) in enumerate(top_links, 1):
        lt = link_type_map.get(link_key, "unknown")
        lt_str = lt.value if hasattr(lt, 'value') else str(lt)
        pct = count / total_routes * 100
        bottleneck_links.add(link_key)
        print(f"  {rank:<5} {link_key[0]} <-> {link_key[1]:<10} {lt_str:<12} {count:<12} {pct:.1f}%")
    
    # ----------------------------------------------------------------
    # 3. ISL usage distribution statistics
    # ----------------------------------------------------------------
    if link_usage:
        counts = list(link_usage.values())
        print(f"\n  --- ISL Usage Statistics ---")
        print(f"  Total ISL links in network: {sum(1 for lid, l in constellation.links.items() if l.link_type in [LinkType.ISL_INTRA, LinkType.ISL_INTER])}")
        print(f"  ISL links actually used:    {len(counts)}")
        print(f"  Max usage count:            {max(counts)}")
        print(f"  Min usage count:            {min(counts)}")
        print(f"  Mean usage count:           {sum(counts)/len(counts):.1f}")
        print(f"  Median usage count:         {sorted(counts)[len(counts)//2]}")
        
        # Distribution buckets
        buckets = [(1, 10), (11, 50), (51, 100), (101, 200), (201, 500), (501, 1000), (1001, 9999)]
        print(f"\n  Usage Distribution:")
        for lo, hi in buckets:
            cnt = sum(1 for c in counts if lo <= c <= hi)
            if cnt > 0:
                print(f"    {lo:>4}-{hi:<4} routes: {cnt} ISL links")
    
    # ----------------------------------------------------------------
    # 4. Intra-plane vs Inter-plane ISL usage comparison
    # ----------------------------------------------------------------
    intra_usage = []
    inter_usage = []
    for link_key, count in link_usage.items():
        lt = link_type_map.get(link_key)
        if lt == LinkType.ISL_INTRA:
            intra_usage.append(count)
        elif lt == LinkType.ISL_INTER:
            inter_usage.append(count)
    
    print(f"\n  --- Intra-plane vs Inter-plane ISL ---")
    if intra_usage:
        print(f"  Intra-plane ISL: {len(intra_usage)} links, "
              f"avg usage={sum(intra_usage)/len(intra_usage):.1f}, "
              f"max={max(intra_usage)}")
    if inter_usage:
        print(f"  Inter-plane ISL: {len(inter_usage)} links, "
              f"avg usage={sum(inter_usage)/len(inter_usage):.1f}, "
              f"max={max(inter_usage)}")
    
    # ----------------------------------------------------------------
    # 5. Find GS pairs whose routes pass through the most bottleneck links
    # ----------------------------------------------------------------
    # Score each GS pair by how many of its ISL links are in the top bottleneck set
    top5_links = set(lk for lk, _ in link_usage.most_common(5))
    
    gs_pair_bottleneck_score = {}
    gs_pair_total_load = {}
    
    for (src, dst), isl_list in gs_pair_links.items():
        # Count how many bottleneck links this route passes through
        bottleneck_count = sum(1 for l in isl_list if l in top5_links)
        gs_pair_bottleneck_score[(src, dst)] = bottleneck_count
        
        # Sum total load across all ISL links in this route
        total_load = sum(link_usage.get(l, 0) for l in isl_list)
        gs_pair_total_load[(src, dst)] = total_load
    
    # Sort by bottleneck count first, then by total load
    sorted_pairs = sorted(
        gs_pair_bottleneck_score.items(),
        key=lambda x: (x[1], gs_pair_total_load.get(x[0], 0)),
        reverse=True
    )
    
    print(f"\n  --- Top 15 Most Congestion-Prone GS Pairs ---")
    print(f"  (Routes that pass through the most heavily-loaded ISL links)")
    print(f"  {'Rank':<5} {'Source':<18} {'Destination':<18} {'Bottleneck ISLs':<16} {'Total Load Score':<16} {'Hops'}")
    print(f"  {'-'*85}")
    
    for rank, ((src, dst), bn_count) in enumerate(sorted_pairs[:15], 1):
        entry = router.routing_table.get(src, {}).get(dst)
        hops = len(entry.path) - 1 if entry else "N/A"
        load_score = gs_pair_total_load.get((src, dst), 0)
        print(f"  {rank:<5} {src:<18} {dst:<18} {bn_count:<16} {load_score:<16} {hops}")
    
    # ----------------------------------------------------------------
    # 6. Satellite-level analysis: which satellites are transit hotspots
    # ----------------------------------------------------------------
    sat_transit_count = Counter()
    for (src, dst), isl_list in gs_pair_links.items():
        entry = router.routing_table.get(src, {}).get(dst)
        if entry:
            # Count intermediate satellites (exclude first and last hops)
            for node in entry.path[1:-1]:
                if node.startswith("SAT_"):
                    sat_transit_count[node] += 1
    
    print(f"\n  --- Top 10 Transit Hotspot Satellites ---")
    print(f"  {'Rank':<5} {'Satellite':<15} {'Transit Count':<15} {'% of Routes'}")
    print(f"  {'-'*50}")
    for rank, (sat, count) in enumerate(sat_transit_count.most_common(10), 1):
        pct = count / total_routes * 100
        print(f"  {rank:<5} {sat:<15} {count:<15} {pct:.1f}%")
    
    return {
        "router_type": router_type,
        "total_routes": total_routes,
        "link_usage": link_usage,
        "top_links": top_links,
        "gs_pair_total_load": gs_pair_total_load,
        "sat_transit_count": sat_transit_count,
    }


def compare_algorithms(results):
    """Compare congestion patterns across all algorithms."""
    print(f"\n\n{'#'*70}")
    print(f"  CROSS-ALGORITHM COMPARISON")
    print(f"{'#'*70}")
    
    # Compare max ISL usage
    print(f"\n  {'Algorithm':<10} {'Max ISL Load':<14} {'Mean ISL Load':<15} {'Hottest Link':<35} {'Top Satellite'}")
    print(f"  {'-'*90}")
    
    for r in results:
        usage = r["link_usage"]
        if not usage:
            continue
        counts = list(usage.values())
        max_link = usage.most_common(1)[0]
        top_sat = r["sat_transit_count"].most_common(1)[0] if r["sat_transit_count"] else ("N/A", 0)
        
        link_str = f"{max_link[0][0]}<->{max_link[0][1]}"
        print(f"  {r['router_type'].upper():<10} {max(counts):<14} {sum(counts)/len(counts):<15.1f} {link_str:<35} {top_sat[0]}({top_sat[1]})")
    
    # Compare which links are consistently bottlenecks
    print(f"\n  --- Consistently Bottleneck Links (appear in top 5 of all algorithms) ---")
    top5_sets = []
    for r in results:
        top5 = set(lk for lk, _ in r["link_usage"].most_common(5))
        top5_sets.append(top5)
    
    if top5_sets:
        common_bottlenecks = top5_sets[0]
        for s in top5_sets[1:]:
            common_bottlenecks = common_bottlenecks & s
        
        if common_bottlenecks:
            for link in common_bottlenecks:
                loads = []
                for r in results:
                    loads.append(f"{r['router_type'].upper()}:{r['link_usage'].get(link, 0)}")
                print(f"    {link[0]} <-> {link[1]}  [{', '.join(loads)}]")
        else:
            print(f"    (No link appears in the top 5 of ALL algorithms)")
    
    print()


def main():
    print("=" * 70)
    print("  LEO Satellite Network Congestion Analysis")
    print("  Analyzing which ISL links and GS pairs are most congestion-prone")
    print("=" * 70)
    
    # Create constellation
    constellation = LEOConstellation(num_planes=6, sats_per_plane=11)
    constellation.add_global_ground_stations()
    
    gs_count = len(constellation.ground_stations)
    sat_count = len(constellation.satellites)
    isl_count = sum(
        1 for l in constellation.links.values()
        if l.link_type in [LinkType.ISL_INTRA, LinkType.ISL_INTER]
    )
    gsl_count = sum(
        1 for l in constellation.links.values()
        if l.link_type == LinkType.GSL
    )
    
    print(f"\n  Network Topology:")
    print(f"    Satellites:     {sat_count}")
    print(f"    Ground stations: {gs_count}")
    print(f"    ISL links:      {isl_count}")
    print(f"    GSL links:      {gsl_count}")
    print(f"    Total GS pairs: {gs_count * (gs_count - 1)}")
    
    # Analyze each algorithm
    results = []
    for algo in ["ksp", "kds", "kdg", "klo"]:
        result = analyze_router(constellation, algo, k=3)
        results.append(result)
    
    # Cross-algorithm comparison
    compare_algorithms(results)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
