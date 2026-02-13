"""
Test pre-computed routing table for ground station pairs.
Verifies that all 4 routing algorithms can pre-compute routes
and that lookup is consistent with on-demand computation.
"""

import sys
import time
sys.path.insert(0, '/Users/selenli/Desktop/graduate_code')

from leo_network import LEOConstellation, KShortestPathsRouter, KDSRouter, KDGRouter, KLORouter
from leo_network.core.routing import create_router


def test_precompute():
    print("=" * 60)
    print("Testing Pre-computed Routing Tables")
    print("=" * 60)
    
    # Create constellation with ground stations
    constellation = LEOConstellation(num_planes=6, sats_per_plane=11)
    constellation.add_global_ground_stations()
    
    gs_nodes = list(constellation.ground_stations.keys())
    num_pairs = len(gs_nodes) * (len(gs_nodes) - 1)
    print(f"\nGround stations: {len(gs_nodes)}")
    print(f"Total GS pairs:  {num_pairs}")
    
    # Test each router
    routers = {
        "ksp": ("K-SP", {"k": 3}),
        "kds": ("K-DS", {"k": 3}),
        "kdg": ("K-DG", {"k": 3}),
        "klo": ("K-LO", {"k": 3}),
    }
    
    for router_type, (name, kwargs) in routers.items():
        print(f"\n--- {name} ({router_type}) ---")
        
        router = create_router(router_type, constellation, **kwargs)
        
        # Time the pre-computation
        start = time.time()
        router.precompute_ground_station_routes()
        elapsed = time.time() - start
        
        # Check routing table
        cached_count = sum(
            len(dsts) for dsts in router.routing_table.values()
        )
        print(f"  Cached routes:  {cached_count}")
        print(f"  Pre-compute time: {elapsed:.2f}s")
        
        # Verify a few routes by comparing table vs on-demand
        test_pairs = [(gs_nodes[0], gs_nodes[-1]),
                      (gs_nodes[1], gs_nodes[-2])]
        
        all_match = True
        for src, dst in test_pairs:
            # From routing table
            table_path = router.routing_table.get(src, {}).get(dst)
            if table_path is None:
                print(f"  WARNING: No route in table for {src} -> {dst}")
                all_match = False
                continue
            
            # On-demand compute (should match)
            on_demand_path = router.compute_path(src, dst)
            if on_demand_path is None:
                print(f"  WARNING: No on-demand route for {src} -> {dst}")
                all_match = False
                continue
            
            # For deterministic routers (ksp, kds, kdg), paths should be same
            # For klo, path may vary due to load optimization
            if table_path.path == on_demand_path:
                pass  # Match
            else:
                # KLO may pick different path from the k set
                if router_type == "klo":
                    pass  # Expected
                else:
                    print(f"  MISMATCH: {src} -> {dst}")
                    print(f"    Table:    {table_path.path[:3]}...")
                    print(f"    OnDemand: {on_demand_path[:3]}...")
                    all_match = False
        
        if all_match:
            print(f"  ✓ Route consistency verified")
        
        # Show sample route
        src, dst = gs_nodes[0], gs_nodes[-1]
        entry = router.routing_table.get(src, {}).get(dst)
        if entry:
            print(f"  Sample route: {src} -> {dst}")
            print(f"    Path length: {len(entry.path)} hops")
            print(f"    Delay:       {entry.metric:.4f}s")
            print(f"    Next hop:    {entry.next_hop}")
    
    print("\n" + "=" * 60)
    print("Pre-computation test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_precompute()
