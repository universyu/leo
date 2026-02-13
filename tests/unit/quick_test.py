#!/usr/bin/env python3
"""
Quick Test Script

Run this to verify the installation and basic functionality.
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def test_topology():
    """Test topology creation"""
    print("Testing topology module...")
    from leo_network import LEOConstellation
    
    constellation = LEOConstellation(
        num_planes=4,
        sats_per_plane=6,
        altitude_km=550.0
    )
    
    assert len(constellation.satellites) == 24, "Wrong number of satellites"
    assert len(constellation.links) > 0, "No links created"
    assert constellation.graph.number_of_nodes() == 24, "Graph nodes mismatch"
    
    print(f"  ✓ Created constellation with {len(constellation.satellites)} satellites")
    print(f"  ✓ Created {len(constellation.links)} ISL links")
    return True


def test_traffic():
    """Test traffic generation"""
    print("Testing traffic module...")
    from leo_network import TrafficGenerator, TrafficPattern
    
    gen = TrafficGenerator(seed=42)
    
    # Create test flow
    flow = gen.create_normal_flow(
        flow_id="test_flow",
        source="SAT_0_0",
        destination="SAT_1_1",
        rate=100
    )
    
    # Generate packets
    packets = gen.generate_packets(current_time=0.0, time_step=0.1)
    
    assert len(packets) > 0, "No packets generated"
    print(f"  ✓ Generated {len(packets)} packets")
    return True


def test_routing():
    """Test routing algorithms"""
    print("Testing routing module...")
    from leo_network import LEOConstellation
    from leo_network import (
        KShortestPathsRouter,
        create_router
    )
    
    constellation = LEOConstellation(num_planes=4, sats_per_plane=6)
    
    # Test k-shortest path
    router = KShortestPathsRouter(constellation)
    path = router.compute_path("SAT_0_0", "SAT_2_3")
    
    assert path is not None, "No path found"
    assert path[0] == "SAT_0_0", "Path start wrong"
    assert path[-1] == "SAT_2_3", "Path end wrong"
    
    print(f"  ✓ K-Shortest path (k=1 default): {len(path)} hops")
    
    # Test K-shortest paths
    k_router = KShortestPathsRouter(constellation, k=3)
    k_paths = k_router.compute_k_paths("SAT_0_0", "SAT_2_3")
    
    assert len(k_paths) > 0, "No K-paths found"
    print(f"  ✓ K-shortest paths: found {len(k_paths)} paths")
    
    # Test factory function
    for router_type in ["ksp", "kds", "kdg", "klo"]:
        r = create_router(router_type, constellation)
        assert r is not None, f"Failed to create {router_type} router"
    
    print("  ✓ All router types created successfully")
    return True


def test_simulation():
    """Test simulation"""
    print("Testing simulation...")
    from leo_network import LEOConstellation
    from leo_network import Simulator
    
    constellation = LEOConstellation(num_planes=4, sats_per_plane=6)
    
    sim = Simulator(
        constellation=constellation,
        time_step=0.01,
        seed=42
    )
    
    # Add traffic
    sim.add_normal_traffic(
        source="SAT_0_0",
        destination="SAT_2_3",
        rate=100
    )
    
    # Run short simulation
    stats = sim.run(duration=0.1, progress_bar=False)
    
    assert stats.packets_sent > 0, "No packets sent"
    print(f"  ✓ Simulation completed: {stats.packets_sent} packets sent")
    print(f"  ✓ Delivery rate: {stats.get_delivery_rate():.4f}")
    
    return True


def test_statistics():
    """Test statistics collection"""
    print("Testing statistics module...")
    from leo_network import StatisticsCollector
    
    stats = StatisticsCollector()
    
    # Record some data
    for i in range(100):
        stats.record_packet_sent(1000)
        if i % 10 != 0:  # 90% delivery
            stats.record_packet_delivered(1000, delay=5.0, hop_count=3)
        else:
            stats.record_packet_dropped(1000)
    
    assert abs(stats.get_delivery_rate() - 0.9) < 0.01, "Wrong delivery rate"
    print(f"  ✓ Delivery rate calculation: {stats.get_delivery_rate():.2f}")
    
    summary = stats.get_summary()
    assert "overview" in summary, "Summary missing overview"
    assert "delay" in summary, "Summary missing delay"
    print("  ✓ Statistics summary generated")
    
    return True


def main():
    print("="*60)
    print("LEO Network Simulation Framework - Quick Test")
    print("="*60 + "\n")
    
    tests = [
        ("Topology", test_topology),
        ("Traffic", test_traffic),
        ("Routing", test_routing),
        ("Statistics", test_statistics),
        ("Simulation", test_simulation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"  → {name}: PASSED\n")
        except Exception as e:
            failed += 1
            print(f"  → {name}: FAILED - {e}\n")
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ All tests passed! The framework is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python examples/basic_simulation.py")
        print("  2. Run: python tests/integration/router_comparison.py")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
