"""
Visualization Module

Provides functions for visualizing LEO satellite network topology,
simulation results, and network performance metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

from .topology import LEOConstellation, LinkType
from .statistics import StatisticsCollector


def plot_constellation_2d(
    constellation: LEOConstellation,
    figsize: Tuple[int, int] = (14, 8),
    show_links: bool = True,
    highlight_nodes: Optional[List[str]] = None,
    title: str = "LEO Satellite Constellation"
) -> plt.Figure:
    """
    Plot 2D projection of constellation (lat/lon)
    
    Args:
        constellation: LEO constellation to plot
        figsize: Figure size
        show_links: Whether to show ISL links
        highlight_nodes: List of node IDs to highlight
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot satellites by plane with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, constellation.num_planes))
    
    for plane_idx in range(constellation.num_planes):
        plane_sats = [
            sat for sat in constellation.satellites.values()
            if sat.plane_id == plane_idx
        ]
        
        lons = [sat.position[1] for sat in plane_sats]
        lats = [sat.position[0] for sat in plane_sats]
        
        ax.scatter(
            lons, lats,
            c=[colors[plane_idx]],
            s=50,
            label=f"Plane {plane_idx}",
            zorder=3
        )
    
    # Plot ISL links
    if show_links:
        for link in constellation.links.values():
            if link.link_type in [LinkType.ISL_INTRA, LinkType.ISL_INTER]:
                src = constellation.satellites.get(link.source)
                dst = constellation.satellites.get(link.target)
                
                if src and dst:
                    # Handle wrap-around for longitude
                    lon1, lon2 = src.position[1], dst.position[1]
                    lat1, lat2 = src.position[0], dst.position[0]
                    
                    if abs(lon2 - lon1) > 180:
                        continue  # Skip wrap-around links for clarity
                    
                    color = 'lightblue' if link.link_type == LinkType.ISL_INTRA else 'lightgreen'
                    ax.plot(
                        [lon1, lon2], [lat1, lat2],
                        c=color, alpha=0.3, linewidth=0.5, zorder=1
                    )
    
    # Highlight specific nodes
    if highlight_nodes:
        for node_id in highlight_nodes:
            if node_id in constellation.satellites:
                sat = constellation.satellites[node_id]
                ax.scatter(
                    [sat.position[1]], [sat.position[0]],
                    c='red', s=200, marker='*', zorder=4,
                    edgecolors='black', linewidths=1
                )
    
    # Plot ground stations
    for gs in constellation.ground_stations.values():
        ax.scatter(
            [gs.position[1]], [gs.position[0]],
            c='orange', s=100, marker='^', zorder=4,
            edgecolors='black', linewidths=1, label='Ground Station'
        )
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_constellation_3d(
    constellation: LEOConstellation,
    figsize: Tuple[int, int] = (12, 10),
    show_links: bool = True,
    title: str = "LEO Satellite Constellation (3D)"
) -> plt.Figure:
    """
    Plot 3D visualization of constellation
    
    Args:
        constellation: LEO constellation to plot
        figsize: Figure size
        show_links: Whether to show ISL links
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    earth_radius = 6371
    
    # Plot Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.3, color='lightblue')
    
    # Convert satellite positions to Cartesian
    def to_cartesian(lat, lon, alt):
        r = earth_radius + alt
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)
        return x, y, z
    
    # Plot satellites by plane
    colors = plt.cm.tab10(np.linspace(0, 1, constellation.num_planes))
    
    for plane_idx in range(constellation.num_planes):
        plane_sats = [
            sat for sat in constellation.satellites.values()
            if sat.plane_id == plane_idx
        ]
        
        positions = [to_cartesian(*sat.position) for sat in plane_sats]
        xs, ys, zs = zip(*positions)
        
        ax.scatter(
            xs, ys, zs,
            c=[colors[plane_idx]],
            s=30,
            label=f"Plane {plane_idx}"
        )
    
    # Plot ISL links
    if show_links:
        for link in constellation.links.values():
            if link.link_type in [LinkType.ISL_INTRA]:
                src = constellation.satellites.get(link.source)
                dst = constellation.satellites.get(link.target)
                
                if src and dst:
                    pos1 = to_cartesian(*src.position)
                    pos2 = to_cartesian(*dst.position)
                    
                    ax.plot(
                        [pos1[0], pos2[0]],
                        [pos1[1], pos2[1]],
                        [pos1[2], pos2[2]],
                        c='gray', alpha=0.2, linewidth=0.5
                    )
    
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_simulation_results(
    stats: StatisticsCollector,
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Simulation Results"
) -> plt.Figure:
    """
    Plot simulation results time series
    
    Args:
        stats: Statistics collector with results
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Throughput over time
    ax = axes[0, 0]
    if stats.throughput_series.timestamps:
        ax.plot(
            stats.throughput_series.timestamps,
            stats.throughput_series.values,
            'b-', linewidth=1
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput (pps)")
    ax.set_title("Throughput Over Time")
    ax.grid(True, alpha=0.3)
    
    # Delay over time
    ax = axes[0, 1]
    if stats.delay_series.timestamps:
        ax.plot(
            stats.delay_series.timestamps,
            stats.delay_series.values,
            'g-', linewidth=1
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average Delay (ms)")
    ax.set_title("Delay Over Time")
    ax.grid(True, alpha=0.3)
    
    # Loss rate over time
    ax = axes[1, 0]
    if stats.loss_rate_series.timestamps:
        ax.plot(
            stats.loss_rate_series.timestamps,
            stats.loss_rate_series.values,
            'r-', linewidth=1
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Loss Rate")
    ax.set_title("Packet Loss Rate Over Time")
    ax.grid(True, alpha=0.3)
    
    # Delay distribution
    ax = axes[1, 1]
    if stats.delays:
        ax.hist(stats.delays, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Delay Distribution")
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_link_utilization_heatmap(
    constellation: LEOConstellation,
    stats: StatisticsCollector,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Link Utilization Heatmap"
) -> plt.Figure:
    """
    Plot link utilization as heatmap
    
    Args:
        constellation: LEO constellation
        stats: Statistics collector with utilization data
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get average utilization per link
    link_utils = {}
    for link_id, series in stats.link_utilization_series.items():
        if series.values:
            link_utils[link_id] = np.mean(series.values)
    
    if not link_utils:
        ax.text(0.5, 0.5, "No utilization data", ha='center', va='center')
        return fig
    
    # Create matrix for intra-plane ISL utilization
    num_planes = constellation.num_planes
    sats_per_plane = constellation.sats_per_plane
    
    utilization_matrix = np.zeros((num_planes, sats_per_plane))
    
    for link_id, util in link_utils.items():
        # Parse link ID to get satellite info
        parts = link_id.split('_')
        if len(parts) >= 3 and parts[0] == "ISL":
            src_parts = parts[1].split('_')
            if len(src_parts) >= 3:
                try:
                    plane_idx = int(src_parts[1])
                    sat_idx = int(src_parts[2])
                    utilization_matrix[plane_idx, sat_idx] = max(
                        utilization_matrix[plane_idx, sat_idx], util
                    )
                except (ValueError, IndexError):
                    continue
    
    im = ax.imshow(utilization_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xlabel("Satellite Index")
    ax.set_ylabel("Orbital Plane")
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Utilization")
    
    plt.tight_layout()
    return fig


def plot_comparison(
    results_dict: Dict[str, Dict],
    metrics: List[str] = ["delivery_rate", "avg_delay_ms", "throughput_pps"],
    figsize: Tuple[int, int] = (12, 4),
    title: str = "Router Comparison"
) -> plt.Figure:
    """
    Plot comparison of different routing algorithms
    
    Args:
        results_dict: Dictionary mapping router names to results
        metrics: List of metrics to compare
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    router_names = list(results_dict.keys())
    x = np.arange(len(router_names))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = []
        
        for name in router_names:
            result = results_dict[name]
            # Navigate to find the metric value
            if "statistics" in result:
                stats = result["statistics"]
                if metric == "delivery_rate":
                    values.append(stats.get("overview", {}).get(metric, 0))
                elif metric == "avg_delay_ms":
                    values.append(stats.get("delay", {}).get("avg_ms", 0))
                elif metric == "throughput_pps":
                    values.append(stats.get("throughput", {}).get("avg_pps", 0))
                else:
                    values.append(0)
            else:
                values.append(0)
        
        bars = ax.bar(x, values, color=plt.cm.Set2(np.linspace(0, 1, len(values))))
        
        ax.set_xlabel("Router")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(router_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8
            )
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def save_all_plots(
    constellation: LEOConstellation,
    stats: StatisticsCollector,
    output_dir: str = ".",
    prefix: str = "sim"
):
    """
    Save all standard plots to files
    
    Args:
        constellation: LEO constellation
        stats: Statistics collector
        output_dir: Output directory
        prefix: Filename prefix
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2D constellation
    fig = plot_constellation_2d(constellation)
    fig.savefig(os.path.join(output_dir, f"{prefix}_constellation_2d.png"), dpi=150)
    plt.close(fig)
    
    # Simulation results
    fig = plot_simulation_results(stats)
    fig.savefig(os.path.join(output_dir, f"{prefix}_results.png"), dpi=150)
    plt.close(fig)
    
    # Link utilization heatmap
    fig = plot_link_utilization_heatmap(constellation, stats)
    fig.savefig(os.path.join(output_dir, f"{prefix}_utilization.png"), dpi=150)
    plt.close(fig)
    
    print(f"Plots saved to {output_dir}/")
