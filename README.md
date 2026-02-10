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
- **Comprehensive Statistics**: Throughput, latency (avg/P50/P95/P99), packet loss, link utilization
- **Attack Cost Metrics**: Measures attack efficiency (attack traffic / normal packet loss rate)
- **5th Percentile Throughput**: Evaluates worst-case network performance under attack
- **Comparative Analysis**: Before/after attack impact measurement
- **Visualization Tools**: Network topology, traffic heatmaps, performance charts

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
│   │   ├── routing.py             # Routing algorithms
│   │   ├── statistics.py          # Statistics collection & analysis
│   │   ├── simulator.py           # Simulation engine
│   │   └── visualization.py       # Plotting & visualization tools
│   └── attacks/                   # ⚔️ DDoS attack module
│       ├── __init__.py
│       └── attacks.py             # Attack generators & strategies
├── tests/                         # 🧪 Test suite
│   ├── unit/                      # Unit tests
│   │   ├── quick_test.py          # Quick functionality verification
│   │   ├── debug_attack.py        # Attack module debugging
│   │   └── debug_flows.py         # Traffic flow debugging
│   └── integration/               # Integration tests
│       ├── ddos_attack_simulation.py   # Full DDoS simulation scenarios
│       └── router_comparison.py        # Routing algorithm comparison
├── examples/                      # 📚 Example scripts
│   └── basic_simulation.py        # Getting started example
├── output/                        # 📊 Generated outputs (charts, logs)
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

### Verify Installation

```bash
# Run quick test to verify everything works
python tests/unit/quick_test.py
```

### Run Your First Simulation

```bash
# Basic network simulation without attacks
python examples/basic_simulation.py

# DDoS attack simulation
python tests/integration/ddos_attack_simulation.py
```

---

## 📖 Usage Guide

### 1. Basic Network Simulation

```python
from leo_network import LEOConstellation, Simulator, ShortestPathRouter

# Create a Walker constellation (6 planes × 11 satellites = 66 satellites)
constellation = LEOConstellation(
    num_planes=6,
    sats_per_plane=11,
    altitude_km=550.0,
    inclination_deg=53.0,
    isl_bandwidth_mbps=1000.0
)

# Initialize simulator with routing algorithm
router = ShortestPathRouter(constellation.graph)
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
    isl_bandwidth_mbps=100.0  # Lower bandwidth for visible attack impact
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
    targets=["SAT_2_5", "SAT_3_5"],      # Target satellites
    num_attackers=50,                      # Number of attack sources
    total_rate=100000.0,                   # Total attack rate (pps)
    strategy=AttackStrategy.DISTRIBUTED    # Source distribution strategy
)

# Run simulation
sim.run(duration=0.5)

# Analyze impact
results = sim.get_results()
normal_stats = results['statistics']['normal_traffic']
attack_stats = results['statistics']['attack_traffic']

print(f"Normal Traffic Delivery: {normal_stats['delivery_rate']:.2%}")
print(f"Attack Traffic Delivery: {attack_stats['delivery_rate']:.2%}")
print(f"Average Latency Impact: +{normal_stats['latency_increase_ms']:.1f} ms")

# Get attack cost metrics
attack_cost = results['attack_cost']['cost_metrics']
print(f"Attack Cost: {attack_cost['attack_cost']:.2f}")
print(f"Normal Packet Loss Rate: {attack_cost['normal_loss_rate']:.2%}")

# Get 5th percentile throughput (worst-case performance)
tp = results['throughput_percentiles']
print(f"5th Percentile Throughput: {tp['p5_pps']:.2f} pps")
```

### 3. Comparing Routing Algorithms Under Attack

```python
from leo_network import (
    LEOConstellation,
    Simulator,
    DDoSAttackGenerator,
    ShortestPathRouter,
    ECMPRouter,
    LoadAwareRouter,
    AttackStrategy
)
from leo_network.core.visualization import plot_comparison

results = {}

for RouterClass in [ShortestPathRouter, ECMPRouter, LoadAwareRouter]:
    # Fresh constellation for each test
    constellation = LEOConstellation(num_planes=6, sats_per_plane=11)
    router = RouterClass(constellation.graph)
    sim = Simulator(constellation=constellation, router=router, seed=42)
    attack_gen = DDoSAttackGenerator(constellation, sim.traffic_generator, seed=42)
    
    # Add traffic and attacks
    sim.add_random_normal_flows(num_flows=30)
    attack_gen.create_flooding_attack(
        targets=["SAT_2_5"],
        num_attackers=30,
        total_rate=50000.0
    )
    
    sim.run(duration=0.5)
    results[RouterClass.__name__] = sim.get_results()

# Visualize comparison
plot_comparison(results, output_dir="output/")
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
| Class | Description | Best For |
|-------|-------------|----------|
| `ShortestPathRouter` | Dijkstra shortest path | Baseline, low latency |
| `KShortestPathsRouter` | K alternative paths | Path diversity |
| `ECMPRouter` | Equal-cost multi-path | Load distribution |
| `LoadAwareRouter` | Congestion-aware routing | High traffic scenarios |
| `RandomizedRouter` | Randomized path selection | DDoS mitigation |
| `KDSRouter` | K-Disjoint Shortest paths | Failure resilience |
| `KDGRouter` | K-Disjoint Geodiverse paths | Localized attack defense |
| `KLORouter` | K-Link-disjoint Load Optimized | Adaptive load balancing |

##### Advanced Disjoint Path Routers

```python
from leo_network import create_router, KDSRouter, KDGRouter, KLORouter

# KDS: K-Disjoint Shortest Paths
# Computes K node/link-disjoint shortest paths for failure resilience
kds_router = create_router("kds", constellation, k=3, disjoint_type="link")
# Or: kds_router = KDSRouter(constellation, k=3, disjoint_type="node")

# KDG: K-Disjoint Geodiverse Paths  
# Maximizes geographic separation between paths for localized attack defense
kdg_router = create_router("kdg", constellation, k=3, diversity_weight=0.5)
# Or: kdg_router = KDGRouter(constellation, k=3, diversity_weight=0.5)

# KLO: K-Link-disjoint with Load Optimization
# Dynamically selects best path based on current network load
klo_router = create_router("klo", constellation, k=3, load_threshold=0.7)
# Or: klo_router = KLORouter(constellation, k=3, load_threshold=0.7, recompute_interval=100)
```

| Router | Parameters | Description |
|--------|------------|-------------|
| `KDSRouter` | `k`: Number of paths<br>`disjoint_type`: "link" or "node" | Link-disjoint avoids shared edges; Node-disjoint avoids shared intermediate nodes |
| `KDGRouter` | `k`: Number of paths<br>`diversity_weight`: 0.0-1.0 | Higher weight prioritizes geographic diversity over path length |
| `KLORouter` | `k`: Number of paths<br>`load_threshold`: 0.0-1.0<br>`recompute_interval`: packets | Switches paths when link utilization exceeds threshold |

#### Simulator
```python
sim = Simulator(constellation, router=None, seed=None)
sim.add_random_normal_flows(num_flows, rate_range=(100, 1000))
sim.run(duration=1.0)
sim.get_results() -> dict
sim.print_results()

# Attack cost analysis
sim.get_attack_cost() -> dict           # Get attack cost metrics
sim.get_attack_cost_summary() -> dict   # Get detailed cost summary
sim.get_5th_percentile_throughput() -> dict  # Get worst-case throughput
```

#### Attack Cost Metrics
| Metric | Description | Formula |
|--------|-------------|---------|
| `attack_cost` | Attack efficiency measure | Attack Traffic (Mbps) / Normal Loss Rate |
| `normalized_cost` | Cost to achieve target loss | Attack Traffic × (Target Rate / Actual Rate) |
| `induced_loss_rate` | Loss caused by attack | Current Loss - Baseline Loss |

#### 5th Percentile Throughput
The 5th percentile throughput represents the worst-case network performance:
- 95% of time windows have throughput **above** this value
- Lower values indicate more severe attack impact
- Used to evaluate network resilience under DDoS attacks

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

#### Attack Types (`AttackType`)
| Type | Description |
|------|-------------|
| `FLOODING` | High-volume UDP/ICMP floods |
| `REFLECTION` | Amplification attacks using reflectors |
| `SLOWLORIS` | Slow application-layer attacks |
| `PULSING` | Intermittent on-off attack pattern |
| `COORDINATED` | Multi-vector synchronized attacks |
| `LINK_TARGETED` | Attacks targeting specific ISL links |
| `BOTTLENECK` | Attacks on network bottleneck nodes |

#### Attack Strategies (`AttackStrategy`)
| Strategy | Description |
|----------|-------------|
| `RANDOM` | Random source selection |
| `DISTRIBUTED` | Evenly distributed sources |
| `CLUSTERED` | Geographically clustered sources |
| `PATH_ALIGNED` | Sources aligned along target paths |

---

## 🧪 Testing

### Run All Tests

```bash
# Unit tests
python tests/unit/quick_test.py
python tests/unit/debug_attack.py
python tests/unit/debug_flows.py

# Integration tests
python tests/integration/ddos_attack_simulation.py
python tests/integration/router_comparison.py
```

### Expected Test Results

| Test Scenario | Overall Delivery | Normal Traffic | Attack Impact |
|--------------|------------------|----------------|---------------|
| Baseline (no attack) | 99.34% | 99.34% | N/A |
| Medium Flooding | 89.82% | 99.57% | +9.59% latency |
| Pulsing Attack | 70.15% | 97.72% | +29.39% latency |
| Reflection Attack | 79.56% | 98.55% | +19.91% latency |
| High-rate Flooding | 58.47% | 96.71% | +41.15% latency |

### Routing Algorithm Comparison (Under Attack)

| Algorithm | Attack Cost | Normal Loss Rate | 5% Throughput |
|-----------|-------------|------------------|---------------|
| ShortestPath | 11444.96 | 3.50% | Baseline |
| ECMP | 11615.78 | 3.45% | +1.5% |
| Randomized | 11791.77 | 3.40% | +2.3% |
| KDS (k=3) | 12150.32 | 3.30% | +3.1% |
| KDG (k=3) | 12380.45 | 3.24% | +3.8% |
| KLO (k=3) | 12520.18 | 3.20% | +4.2% |

> **Note**: Higher attack cost = better defense (attacker needs more traffic to cause same damage)

---

## 📊 Output Examples

Running simulations generates output in the `output/` directory:

- `topology_*.png` - Network topology visualization
- `traffic_heatmap_*.png` - Link utilization heatmaps
- `latency_distribution_*.png` - Latency CDF plots
- `attack_comparison_*.png` - Before/after attack metrics
- `simulation_results_*.json` - Detailed statistics in JSON format

---

## 🛠️ Configuration

### Environment Variables

```bash
export LEO_SIM_OUTPUT_DIR="./output"     # Output directory
export LEO_SIM_LOG_LEVEL="INFO"          # Logging level
export LEO_SIM_RANDOM_SEED="42"          # Default random seed
```

### Constellation Presets

```python
from leo_network.core.topology import STARLINK_SHELL1, ONEWEB, IRIDIUM

# Use predefined constellation configurations
constellation = LEOConstellation(**STARLINK_SHELL1)
```

---

## 🗺️ Roadmap

- [x] **Attack Cost Metrics**: Measure attack efficiency and defense effectiveness
- [x] **5th Percentile Throughput**: Worst-case performance evaluation
- [x] **Advanced Defense Algorithms**: KDS, KDG, KLO disjoint routing algorithms
- [ ] **K-Bottleneck Minimize**: Advanced bottleneck-aware routing
- [ ] **Dynamic Topology**: Time-varying satellite positions and link handoffs
- [ ] **Parallel Simulation**: Multi-threaded execution for large constellations
- [ ] **Machine Learning Integration**: ML-based attack detection and routing
- [ ] **Real Orbital Data**: TLE-based satellite position calculation
- [ ] **GUI Dashboard**: Web-based visualization and control interface

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
