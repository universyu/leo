"""
LEO Satellite Network - Core Module

This module contains the core components for LEO satellite network simulation:
- Topology: Satellite constellation and link modeling
- Traffic: Traffic generation and packet handling
- Routing: Various routing algorithms
- Statistics: Data collection and analysis
- Simulator: Main simulation engine
- Visualization: Plotting and visualization tools
"""

from .topology import (
    Satellite,
    GroundStation,
    Link,
    LinkType,
    LEOConstellation
)

from .traffic import (
    Packet,
    PacketType,
    Flow,
    TrafficPattern,
    TrafficGenerator
)

from .routing import (
    Router,
    ShortestPathRouter,
    KShortestPathsRouter,
    ECMPRouter,
    LoadAwareRouter,
    RandomizedRouter,
    create_router
)

from .statistics import (
    StatisticsCollector,
    TimeSeriesData,
    AttackCostCalculator,
    compare_attack_costs,
    compare_throughput_percentiles,
    print_algorithm_comparison
)

from .simulator import Simulator

from .visualization import (
    plot_constellation_2d,
    plot_constellation_3d,
    plot_simulation_results,
    plot_link_utilization_heatmap,
    plot_comparison,
    save_all_plots
)

__all__ = [
    # Topology
    'Satellite',
    'GroundStation', 
    'Link',
    'LinkType',
    'LEOConstellation',
    # Traffic
    'Packet',
    'PacketType',
    'Flow',
    'TrafficPattern',
    'TrafficGenerator',
    # Routing
    'Router',
    'ShortestPathRouter',
    'KShortestPathsRouter',
    'ECMPRouter',
    'LoadAwareRouter',
    'RandomizedRouter',
    'create_router',
    # Statistics
    'StatisticsCollector',
    'TimeSeriesData',
    'AttackCostCalculator',
    'compare_attack_costs',
    'compare_throughput_percentiles',
    'print_algorithm_comparison',
    # Simulator
    'Simulator',
    # Visualization
    'plot_constellation_2d',
    'plot_constellation_3d',
    'plot_simulation_results',
    'plot_link_utilization_heatmap',
    'plot_comparison',
    'save_all_plots'
]
