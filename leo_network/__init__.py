"""
LEO Satellite Network Simulation Framework

A comprehensive framework for simulating LEO satellite network communications
and DDoS attack scenarios.

Modules:
- core: Base network communication components (topology, traffic, routing, etc.)
- attacks: DDoS attack simulation components
"""

# Import from core module
from .core import (
    # Topology
    Satellite,
    GroundStation,
    Link,
    LinkType,
    LEOConstellation,
    # Traffic
    Packet,
    PacketType,
    Flow,
    TrafficPattern,
    TrafficGenerator,
    # Routing
    Router,
    KShortestPathsRouter,
    KDSRouter,
    KDGRouter,
    KLORouter,
    KRandRouter,
    create_router,
    # Statistics
    StatisticsCollector,
    TimeSeriesData,
    AttackCostCalculator,
    compare_attack_costs,
    compare_throughput_percentiles,
    print_algorithm_comparison,
    # Simulator
    Simulator,
    # Visualization
    plot_constellation_2d,
    plot_constellation_3d,
    plot_simulation_results,
    plot_link_utilization_heatmap,
    plot_comparison,
    save_all_plots
)

# Import from attacks module
from .attacks import (
    AttackType,
    AttackStrategy,
    AttackConfig,
    AttackMetrics,
    AttackFlow,
    DDoSAttackGenerator
)

__version__ = "0.2.0"

__all__ = [
    # Core - Topology
    'Satellite',
    'GroundStation',
    'Link',
    'LinkType',
    'LEOConstellation',
    # Core - Traffic
    'Packet',
    'PacketType',
    'Flow',
    'TrafficPattern',
    'TrafficGenerator',
    # Core - Routing
    'Router',
    'KShortestPathsRouter',
    'KDSRouter',
    'KDGRouter',
    'KLORouter',
    'KRandRouter',
    'create_router',
    # Core - Statistics
    'StatisticsCollector',
    'TimeSeriesData',
    'AttackCostCalculator',
    'compare_attack_costs',
    'compare_throughput_percentiles',
    'print_algorithm_comparison',
    # Core - Simulator
    'Simulator',
    # Core - Visualization
    'plot_constellation_2d',
    'plot_constellation_3d',
    'plot_simulation_results',
    'plot_link_utilization_heatmap',
    'plot_comparison',
    'save_all_plots',
    # Attacks
    'AttackType',
    'AttackStrategy',
    'AttackConfig',
    'AttackMetrics',
    'AttackFlow',
    'DDoSAttackGenerator'
]
