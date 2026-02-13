# LEO Satellite Network DDoS Simulation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive simulation framework for modeling **Low Earth Orbit (LEO) satellite networks** and evaluating **DDoS attack impacts**. This project provides tools for researchers and engineers to study network resilience, test defense mechanisms, and analyze attack patterns in satellite communication systems.

## 🌟 Features

### Core Network Simulation
- **Walker Constellation Modeling**: Configurable orbital planes, satellites per plane, altitude, and inclination
- **Inter-Satellite Links (ISL)**: Automatic generation of intra-plane and inter-plane links
- **Ground Station Support**: Flexible ground station placement with Ground-to-Satellite Links (GSL)
- **Multiple Routing Algorithms**: Shortest path, K-shortest paths, ECMP, load-aware, and randomized routing

### DDoS Attack Simulation
- **7 Attack Types**: Flooding, Reflection, Slowloris, Pulsing, Coordinated, Link-targeted, Bottleneck
- **4 Attack Strategies**: Random, Distributed, Clustered, Path-aligned source selection
- **Realistic Traffic Modeling**: Configurable attack rates, packet sizes, and timing patterns
- **Impact Analysis**: Separate tracking of normal vs. attack traffic performance

### Analysis & Visualization
- **14 Publication-Ready Figures**: Automated generation of all thesis/paper figures
- **Comprehensive Statistics**: Throughput, latency (avg/P50/P95/P99), packet loss, link utilization
- **Attack Cost Metrics**: Measures attack efficiency (attack traffic / normal packet loss rate)
- **5th Percentile Throughput**: Evaluates worst-case network performance under attack
- **Graph-Theoretic Visualization**: 4 K-routing algorithms visualized as graph topologies
- **Routing Table Visualization**: Complete routing table display for all 4 K-routing algorithms

---

## 📁 Project Structure

```
graduate_code/
├── leo_network/                    # Core simulation library
│   ├── __init__.py                # Unified API exports
│   ├── core/                      # 🛰️ Basic network communication module
│   │   ├── __init__.py
│   │   ├── topology.py            # LEO constellation & link modeling
│   │   ├── traffic.py             # Traffic generation (normal flows)
│   │   ├── routing.py             # Routing algorithms (KSP, KDS, KDG, KLO)
│   │   ├── statistics.py          # Statistics collection & analysis
│   │   ├── simulator.py           # Simulation engine
│   │   └── visualization.py       # Base plotting & visualization tools
│   ├── attacks/                   # ⚔️ DDoS attack module
│   │   ├── __init__.py
│   │   └── attacks.py             # Attack generators & strategies
│   └── defense/                   # 🛡️ Defense module (reserved)
│
├── tests/                         # 🧪 Test & computation scripts
│   ├── unit/                      # Unit tests & data computation
│   │   ├── quick_test.py                # Quick functionality verification
│   │   ├── compute_graph_paths.py       # Compute K-paths for graph visualization (→ Fig 7~10)
│   │   ├── compute_routing_table.py     # Compute routing tables for 4 algorithms (→ Fig 11~14)
│   │   ├── ddos_simulation_verify.py    # DDoS simulation result verification
│   │   ├── attack_gs_analysis.py        # Ground station attack analysis
│   │   ├── gs_cost_comparison.py        # Ground station cost comparison
│   │   ├── gs_full_cost_table.py        # Full ground station cost table
│   │   ├── optimize_krand_weights.py    # k-RAND weight optimization
│   │   ├── test_precompute_routes.py    # Route precomputation tests
│   │   └── verify_ground_stations.py    # Ground station verification
│   └── integration/               # Integration tests & simulations
│       ├── krand_advantage_simulation.py  # k-RAND advantage simulation (→ Fig 1~6 data)
│       ├── ddos_attack_simulation.py      # Full DDoS simulation scenarios
│       ├── targeted_isl_attack.py         # Targeted ISL attack simulation
│       └── router_comparison.py           # Routing algorithm comparison
│
├── examples/                      # 📚 Example scripts
│   └── basic_simulation.py        # Getting started example
│
├── output/                        # � Intermediate data (JSON)
│   ├── graph_paths_data.json              # K-paths data for graph visualization
│   ├── routing_table_ksp.json             # KSP routing table
│   ├── routing_table_kds.json             # KDS routing table
│   ├── routing_table_kdg.json             # KDG routing table
│   ├── routing_table_klo.json             # KLO routing table
│   └── routing_table_data.json            # Combined routing table data
│
├── result/                        # 📊 Final outputs (figures & stats)
│   ├── generate_visualizations.py         # 🎨 Master visualization script (generates all 14 figures)
│   ├── fig1_attack_cost_comparison.png    # Attack cost comparison (bar chart)
│   ├── fig2_network_topology_attack.png   # LEO 6×11 network topology with attack
│   ├── fig3_p5_throughput_comparison.png   # P5 throughput comparison
│   ├── fig4_vulnerability_analysis.png    # Vulnerability analysis flow
│   ├── fig5_attacked_gs_paths.png         # GS paths through target ISL (horizontal, split)
│   ├── fig6_attacked_gs_traffic.png       # GS attack traffic (horizontal, split)
│   ├── fig7_graph_ksp.png                 # KSP algorithm graph visualization
│   ├── fig8_graph_kds.png                 # KDS algorithm graph visualization
│   ├── fig9_graph_kdg.png                 # KDG algorithm graph visualization
│   ├── fig10_graph_klo.png                # KLO algorithm graph visualization
│   ├── fig11_routing_table_ksp.png        # KSP routing table visualization
│   ├── fig12_routing_table_kds.png        # KDS routing table visualization
│   ├── fig13_routing_table_kdg.png        # KDG routing table visualization
│   ├── fig14_routing_table_klo.png        # KLO routing table visualization
│   ├── code_lines_visual.png              # Code statistics dashboard
│   └── code_line_count.json               # Code line count data
│
├── count_lines.py                 # 📏 Code line counter & dashboard generator
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd graduate_code

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| networkx | ≥ 3.0 | Graph algorithms & network modeling |
| numpy | ≥ 1.24.0 | Numerical computation |
| matplotlib | ≥ 3.7.0 | Plotting & visualization |
| pandas | ≥ 2.0.0 | Data analysis |
| seaborn | ≥ 0.12.0 | Statistical visualization |
| tqdm | ≥ 4.65.0 | Progress bars |

### Verify Installation

```bash
# Run quick test to verify everything works
python tests/unit/quick_test.py
```

---

## 📊 Figure Generation Pipeline

The project generates **14 publication-ready figures** through a three-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Stage 1: Run Simulations                       │
│                                                                     │
│  krand_advantage_simulation.py  → output/ (DDoS simulation data)   │
│  compute_graph_paths.py         → output/graph_paths_data.json     │
│  compute_routing_table.py       → output/routing_table_*.json      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Stage 2: Generate Figures                        │
│                                                                     │
│  result/generate_visualizations.py  → result/fig1 ~ fig14.png      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Stage 3: Code Statistics                         │
│                                                                     │
│  count_lines.py  → result/code_lines_visual.png                    │
│                  → result/code_line_count.json                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Execution

```bash
# Step 1: Generate simulation data (results saved to output/ as JSON)
python tests/integration/krand_advantage_simulation.py
python tests/unit/compute_graph_paths.py
python tests/unit/compute_routing_table.py

# Step 2: Generate all 14 figures (reads JSON from output/, saves PNG to result/)
python result/generate_visualizations.py

# Step 3: Generate code statistics dashboard (optional)
python count_lines.py
```

### Figure Catalog

| Figure | Filename | Description |
|--------|----------|-------------|
| Fig 1 | `fig1_attack_cost_comparison.png` | Attack cost comparison across 5 routing algorithms (bar chart) |
| Fig 2 | `fig2_network_topology_attack.png` | LEO 6×11 constellation topology with attack annotations |
| Fig 3 | `fig3_p5_throughput_comparison.png` | 5th percentile throughput comparison across algorithms |
| Fig 4 | `fig4_vulnerability_analysis.png` | Network vulnerability analysis flow diagram |
| Fig 5 | `fig5_attacked_gs_paths.png` | Ground station paths through target ISL (horizontal split view) |
| Fig 6 | `fig6_attacked_gs_traffic.png` | Ground station attack traffic volume (horizontal split view) |
| Fig 7 | `fig7_graph_ksp.png` | KSP (K-Shortest Paths) graph-theoretic visualization |
| Fig 8 | `fig8_graph_kds.png` | KDS (K-Disjoint Shortest) graph-theoretic visualization |
| Fig 9 | `fig9_graph_kdg.png` | KDG (K-Disjoint Geodiverse) graph-theoretic visualization |
| Fig 10 | `fig10_graph_klo.png` | KLO (K-Limited-Overlap) graph-theoretic visualization |
| Fig 11 | `fig11_routing_table_ksp.png` | KSP complete routing table visualization |
| Fig 12 | `fig12_routing_table_kds.png` | KDS complete routing table visualization |
| Fig 13 | `fig13_routing_table_kdg.png` | KDG complete routing table visualization |
| Fig 14 | `fig14_routing_table_klo.png` | KLO complete routing table visualization |
| Extra | `code_lines_visual.png` | Code line statistics dashboard |

---

## 📖 Usage Guide

### 1. Basic Network Simulation

```python
from leo_network import LEOConstellation, Simulator, KShortestPathsRouter

# Create a Walker constellation (6 planes × 11 satellites = 66 satellites)
constellation = LEOConstellation(
    num_planes=6,
    sats_per_plane=11,
    altitude_km=550.0,
    inclination_deg=53.0,
    isl_bandwidth_mbps=1000.0
)

# Initialize simulator with routing algorithm
router = KShortestPathsRouter(constellation)
sim = Simulator(constellation=constellation, router=router, seed=42)

# Add normal traffic flows
sim.add_random_normal_flows(num_flows=50, rate_range=(100, 500))

# Run simulation for 1 second
sim.run(duration=1.0)

# View results
sim.print_results()
stats = sim.get_results()
print(f"Delivery Rate: {stats['statistics']['delivery_rate']:.2%}")
print(f"Average Latency: {stats['statistics']['avg_latency_ms']:.2f} ms")
```

### 2. DDoS Attack Simulation

```python
from leo_network import (
    LEOConstellation,
    Simulator,
    DDoSAttackGenerator,
    AttackType,
    AttackStrategy
)

# Create constellation with lower bandwidth to observe congestion
constellation = LEOConstellation(
    num_planes=6,
    sats_per_plane=11,
    isl_bandwidth_mbps=100.0
)

# Initialize simulator and attack generator
sim = Simulator(constellation=constellation, seed=42)
attack_gen = DDoSAttackGenerator(
    constellation=constellation,
    traffic_generator=sim.traffic_generator,
    seed=42
)

# Add baseline normal traffic
sim.add_random_normal_flows(num_flows=30)

# Launch flooding attack on target satellites
attack_gen.create_flooding_attack(
    targets=["SAT_2_5", "SAT_3_5"],
    num_attackers=50,
    total_rate=100000.0,
    strategy=AttackStrategy.DISTRIBUTED
)

# Run simulation
sim.run(duration=0.5)

# Analyze impact
results = sim.get_results()
normal_stats = results['statistics']['normal_traffic']
attack_stats = results['statistics']['attack_traffic']

print(f"Normal Traffic Delivery: {normal_stats['delivery_rate']:.2%}")
print(f"Attack Traffic Delivery: {attack_stats['delivery_rate']:.2%}")
```

### 3. Comparing Routing Algorithms Under Attack

```python
from leo_network import (
    LEOConstellation, Simulator, DDoSAttackGenerator,
    KShortestPathsRouter, KDSRouter, KDGRouter, KLORouter
)

results = {}
for RouterClass in [KShortestPathsRouter, KDSRouter, KDGRouter, KLORouter]:
    constellation = LEOConstellation(num_planes=6, sats_per_plane=11)
    router = RouterClass(constellation)
    sim = Simulator(constellation=constellation, router=router, seed=42)
    attack_gen = DDoSAttackGenerator(constellation, sim.traffic_generator, seed=42)

    sim.add_random_normal_flows(num_flows=30)
    attack_gen.create_flooding_attack(
        targets=["SAT_2_5"], num_attackers=30, total_rate=50000.0
    )

    sim.run(duration=0.5)
    results[RouterClass.__name__] = sim.get_results()
```

---

## 🔧 API Reference

### Core Module (`leo_network.core`)

#### LEOConstellation
```python
LEOConstellation(
    num_planes: int = 6,           # Number of orbital planes
    sats_per_plane: int = 11,      # Satellites per plane
    altitude_km: float = 550.0,    # Orbital altitude
    inclination_deg: float = 53.0, # Orbital inclination
    isl_bandwidth_mbps: float = 1000.0,  # ISL bandwidth
    isl_delay_ms: float = 5.0      # ISL propagation delay
)
```

#### Routing Algorithms

| Class | Algorithm | Description |
|-------|-----------|-------------|
| `KShortestPathsRouter` | k-SP | K shortest simple paths — path diversity baseline |
| `KDSRouter` | k-DS | K-Disjoint Shortest paths — failure resilience via disjoint paths |
| `KDGRouter` | k-DG | K-Disjoint Geodiverse paths — geographic separation against localized attacks |
| `KLORouter` | k-LO | K-Limited-Overlap paths — adaptive load balancing with controlled overlap |

```python
from leo_network import create_router

# KSP: K-Shortest Paths
ksp_router = create_router("ksp", constellation, k=3)

# KDS: K-Disjoint Shortest Paths
kds_router = create_router("kds", constellation, k=3, disjoint_type="link")

# KDG: K-Disjoint Geodiverse Paths
kdg_router = create_router("kdg", constellation, k=3, diversity_weight=0.5)

# KLO: K-Limited-Overlap with Load Optimization
klo_router = create_router("klo", constellation, k=3, load_threshold=0.7)
```

#### Simulator
```python
sim = Simulator(constellation, router=None, seed=None)
sim.add_random_normal_flows(num_flows, rate_range=(100, 1000))
sim.run(duration=1.0)
sim.get_results() -> dict
sim.print_results()
sim.get_attack_cost() -> dict
sim.get_attack_cost_summary() -> dict
sim.get_5th_percentile_throughput() -> dict
```

### Attacks Module (`leo_network.attacks`)

#### DDoSAttackGenerator
```python
attack_gen = DDoSAttackGenerator(constellation, traffic_generator, seed=None)

# Attack creation methods
attack_gen.create_flooding_attack(targets, num_attackers, total_rate, strategy)
attack_gen.create_reflection_attack(targets, reflectors, amplification_factor)
attack_gen.create_pulsing_attack(targets, on_duration, off_duration, peak_rate)
attack_gen.create_slowloris_attack(targets, num_connections, rate_per_conn)
attack_gen.create_coordinated_attack(targets, attack_configs)
attack_gen.create_link_targeted_attack(target_links, num_attackers, rate)
attack_gen.create_bottleneck_attack(bottleneck_nodes, num_attackers, rate)
```

#### Attack Types & Strategies

| Attack Type | Description |
|-------------|-------------|
| `FLOODING` | High-volume UDP/ICMP floods |
| `REFLECTION` | Amplification attacks using reflectors |
| `SLOWLORIS` | Slow application-layer attacks |
| `PULSING` | Intermittent on-off attack pattern |
| `COORDINATED` | Multi-vector synchronized attacks |
| `LINK_TARGETED` | Attacks targeting specific ISL links |
| `BOTTLENECK` | Attacks on network bottleneck nodes |

| Attack Strategy | Description |
|-----------------|-------------|
| `RANDOM` | Random source selection |
| `DISTRIBUTED` | Evenly distributed sources |
| `CLUSTERED` | Geographically clustered sources |
| `PATH_ALIGNED` | Sources aligned along target paths |

---

## 🧪 Testing

```bash
# Quick functionality verification
python tests/unit/quick_test.py

# DDoS simulation verification
python tests/unit/ddos_simulation_verify.py

# Ground station analysis
python tests/unit/attack_gs_analysis.py
python tests/unit/gs_cost_comparison.py

# Full DDoS simulation
python tests/integration/ddos_attack_simulation.py

# Targeted ISL attack simulation
python tests/integration/targeted_isl_attack.py

# Routing algorithm comparison
python tests/integration/router_comparison.py
```

---

## �️ Roadmap

- [x] **Core Network Simulation**: Walker constellation, ISL, GSL modeling
- [x] **DDoS Attack Simulation**: 7 attack types, 4 attack strategies
- [x] **Attack Cost Metrics**: Attack efficiency and defense effectiveness measurement
- [x] **5th Percentile Throughput**: Worst-case performance evaluation
- [x] **Advanced K-Routing Algorithms**: KSP, KDS, KDG, KLO disjoint routing
- [x] **14 Publication Figures**: Automated figure generation pipeline
- [x] **Graph-Theoretic Visualization**: Algorithm path visualization on network graphs
- [x] **Routing Table Visualization**: Complete routing table display for 4 algorithms
- [ ] **Dynamic Topology**: Time-varying satellite positions and link handoffs
- [ ] **Parallel Simulation**: Multi-threaded execution for large constellations
- [ ] **Machine Learning Integration**: ML-based attack detection and routing

---

## 📚 References

1. Handley, M. (2018). "Delay is Not an Option: Low Latency Routing in Space"
2. Bhattacherjee, D., et al. (2019). "Network Topology Design at 27,000 km/hour"
3. Giuliari, L., et al. (2020). "Internet Backbones in Space"

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.
