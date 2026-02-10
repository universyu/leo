#!/usr/bin/env python3
"""Debug script for attack flow generation"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from leo_network import LEOConstellation
from leo_network import TrafficGenerator, PacketType, TrafficPattern
from leo_network import DDoSAttackGenerator, AttackStrategy, AttackFlow

# Create constellation
constellation = LEOConstellation(num_planes=6, sats_per_plane=11)

print(f"Satellites: {len(constellation.satellites)}")

# Create traffic generator
traffic_gen = TrafficGenerator(seed=42)

# Create attack generator
attack_gen = DDoSAttackGenerator(constellation, traffic_gen, seed=42)

# Create flooding attack
target_sats = ["SAT_2_5", "SAT_3_5"]
attack_id = attack_gen.create_flooding_attack(
    targets=target_sats,
    num_attackers=5,
    total_rate=1000.0,
    strategy=AttackStrategy.DISTRIBUTED
)

print(f"\nAttack ID: {attack_id}")
print(f"Flows in traffic_gen: {len(traffic_gen.flows)}")

# Check one flow detail
flow = list(traffic_gen.flows.values())[0]
print(f"\n--- First Flow Debug ---")
print(f"  ID: {flow.id}")
print(f"  Rate: {flow.rate}")
print(f"  Pattern: {flow.pattern}")
print(f"  is_active(0.0): {flow.is_active(0.0)}")
print(f"  hasattr get_current_rate: {hasattr(flow, 'get_current_rate')}")

# Test _calculate_num_packets manually
time_step = 0.001
expected = flow.rate * time_step
print(f"\n--- Manual Calculation ---")
print(f"  rate * time_step = {flow.rate} * {time_step} = {expected}")
print(f"  int(expected) = {int(expected)}")

# For constant pattern, expected = 0.2, int(0.2) = 0
# This is the bug! The rate per attacker (200 pps) * 0.001s = 0.2 packets
# int(0.2) = 0

# Test with Poisson
import numpy as np
rng = np.random.default_rng(42)
for i in range(10):
    n = rng.poisson(expected)
    print(f"  Poisson sample {i}: {n}")

print("\n--- ISSUE FOUND ---")
print(f"With rate={flow.rate} pps and time_step={time_step}s:")
print(f"  Expected packets per step: {expected}")
print(f"  CONSTANT pattern: int({expected}) = {int(expected)} --> 0 packets!")
print(f"  POISSON pattern: poisson({expected}) has ~{1-np.exp(-expected):.1%} chance of >= 1 packet")
print("\nThe AttackFlow uses CONSTANT pattern but rate*time_step < 1, so no packets generated!")
print("\nSolution: Either use POISSON pattern or increase time_step or increase rate")

# Check attack flows details
print("\n--- Attack Flow Details ---")
for i, flow in enumerate(attack_gen.attack_flows[attack_id]):
    print(f"Flow {i}: {flow.id}")
    print(f"  Type: {type(flow).__name__}")
    print(f"  Source: {flow.source}")
    print(f"  Destination: {flow.destination}")
    print(f"  Rate: {flow.rate}")
    print(f"  Start time: {flow.start_time}")
    print(f"  Duration: {flow.duration}")
    print(f"  Flow type: {flow.flow_type}")
    print(f"  Is active at t=0: {flow.is_active(0.0)}")
    if hasattr(flow, 'get_current_rate'):
        print(f"  get_current_rate(0.0): {flow.get_current_rate(0.0)}")
    print(f"  Pattern: {flow.pattern}")
    print()

# Check if flows are in traffic generator
print("\n--- Flows in Traffic Generator ---")
for fid, flow in traffic_gen.flows.items():
    is_attack = 'attack' in fid
    print(f"  {fid}: type={flow.flow_type.value}, rate={flow.rate}, is_active(0)={flow.is_active(0.0)}")

# Try generating packets directly
print("\n--- Generating Packets at t=0 ---")
packets = traffic_gen.generate_packets(0.0, 0.001)
attack_packets = [p for p in packets if p.packet_type == PacketType.ATTACK]
normal_packets = [p for p in packets if p.packet_type == PacketType.NORMAL]
print(f"Total packets generated: {len(packets)}")
print(f"Attack packets: {len(attack_packets)}")
print(f"Normal packets: {len(normal_packets)}")

# Try generating more times
print("\n--- Generating Packets over 0.1s ---")
total_attack = 0
total_normal = 0
for step in range(100):
    t = step * 0.001
    packets = traffic_gen.generate_packets(t, 0.001)
    for p in packets:
        if p.packet_type == PacketType.ATTACK:
            total_attack += 1
        else:
            total_normal += 1

print(f"Total attack packets in 0.1s: {total_attack}")
print(f"Total normal packets in 0.1s: {total_normal}")
print(f"Expected attack packets (rate=1000, 0.1s): ~100")
