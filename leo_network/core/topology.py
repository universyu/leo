"""
LEO Satellite Network Topology Module

This module provides classes for modeling LEO satellite constellations,
including satellite nodes, inter-satellite links (ISL), and ground stations.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import math


class NodeType(Enum):
    """Type of network node"""
    SATELLITE = "satellite"
    GROUND_STATION = "ground_station"
    USER_TERMINAL = "user_terminal"


class LinkType(Enum):
    """Type of network link"""
    ISL_INTRA = "isl_intra"       # Intra-plane ISL (same orbital plane)
    ISL_INTER = "isl_inter"       # Inter-plane ISL (between orbital planes)
    GSL = "gsl"                   # Ground-Satellite Link
    USER = "user"                 # User-Satellite Link


@dataclass
class Satellite:
    """
    Represents a LEO satellite node
    
    Attributes:
        id: Unique identifier
        plane_id: Orbital plane index
        sat_id: Satellite index within the plane
        position: (latitude, longitude, altitude) in degrees/km
        capacity: Processing capacity (packets/s)
        buffer_size: Queue buffer size (packets)
    """
    id: str
    plane_id: int
    sat_id: int
    position: Tuple[float, float, float] = (0.0, 0.0, 550.0)  # lat, lon, alt(km)
    capacity: int = 10000  # packets per second
    buffer_size: int = 1000  # packets
    
    # Runtime state
    queue: List = field(default_factory=list)
    queue_length: int = 0
    processed_packets: int = 0
    dropped_packets: int = 0
    
    def __post_init__(self):
        self.queue = []
        
    def enqueue(self, packet) -> bool:
        """Add packet to queue, return False if dropped"""
        if self.queue_length >= self.buffer_size:
            self.dropped_packets += 1
            return False
        self.queue.append(packet)
        self.queue_length += 1
        return True
    
    def dequeue(self) -> Optional[object]:
        """Remove and return packet from queue"""
        if self.queue_length > 0:
            self.queue_length -= 1
            self.processed_packets += 1
            return self.queue.pop(0)
        return None
    
    def get_utilization(self) -> float:
        """Return buffer utilization ratio"""
        return self.queue_length / self.buffer_size if self.buffer_size > 0 else 0.0
    
    def reset_stats(self):
        """Reset statistics"""
        self.processed_packets = 0
        self.dropped_packets = 0


@dataclass
class GroundStation:
    """
    Represents a ground station
    
    Attributes:
        id: Unique identifier  
        position: (latitude, longitude) in degrees
        connected_sats: List of currently connected satellite IDs
    """
    id: str
    position: Tuple[float, float]  # lat, lon
    connected_sats: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.connected_sats = []


@dataclass 
class Link:
    """
    Represents a network link (ISL or GSL)
    
    Attributes:
        id: Unique identifier
        source: Source node ID
        target: Target node ID
        link_type: Type of link
        bandwidth: Link bandwidth (Mbps)
        propagation_delay: One-way propagation delay (ms)
        current_load: Current traffic load (bits transmitted in current time step)
        time_step: Simulation time step in seconds (used for capacity calculation)
    """
    id: str
    source: str
    target: str
    link_type: LinkType
    bandwidth: float = 10000.0  # Mbps (10 Gbps default for ISL)
    propagation_delay: float = 5.0  # ms
    time_step: float = 0.001  # Default time step 1ms
    
    # Runtime state
    current_load: float = 0.0  # bits transmitted in current time step
    total_bytes: int = 0
    dropped_bytes: int = 0
    
    def get_capacity_bits(self) -> float:
        """Get link capacity in bits for current time step"""
        return self.bandwidth * 1e6 * self.time_step  # Mbps * 1e6 * seconds = bits
    
    def get_utilization(self) -> float:
        """Return link utilization ratio"""
        capacity = self.get_capacity_bits()
        return self.current_load / capacity if capacity > 0 else 0.0
    
    def add_traffic(self, bytes_count: int, packet_size_bits: int = 8000) -> bool:
        """
        Add traffic to link
        Returns False if link is congested (utilization > 1.0)
        """
        traffic_bits = bytes_count * 8  # Convert bytes to bits
        self.current_load += traffic_bits
        self.total_bytes += bytes_count
        
        if self.get_utilization() > 1.0:
            self.dropped_bytes += bytes_count
            return False
        return True
    
    def reset_load(self):
        """Reset current load (call at each time step)"""
        self.current_load = 0.0
    
    def reset_stats(self):
        """Reset all statistics"""
        self.current_load = 0.0
        self.total_bytes = 0
        self.dropped_bytes = 0


class LEOConstellation:
    """
    LEO Satellite Constellation Model
    
    Implements a Walker Star constellation topology with configurable parameters.
    Supports both intra-plane and inter-plane ISL connections.
    """
    
    def __init__(
        self,
        num_planes: int = 6,
        sats_per_plane: int = 11,
        altitude_km: float = 550.0,
        inclination_deg: float = 53.0,
        isl_bandwidth_mbps: float = 10000.0,
        sat_capacity: int = 10000,
        sat_buffer_size: int = 1000
    ):
        """
        Initialize LEO constellation
        
        Args:
            num_planes: Number of orbital planes
            sats_per_plane: Number of satellites per plane
            altitude_km: Orbital altitude in km
            inclination_deg: Orbital inclination in degrees
            isl_bandwidth_mbps: ISL bandwidth in Mbps
            sat_capacity: Satellite processing capacity (packets/s)
            sat_buffer_size: Satellite queue buffer size (packets)
        """
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.altitude_km = altitude_km
        self.inclination_deg = inclination_deg
        self.isl_bandwidth_mbps = isl_bandwidth_mbps
        self.sat_capacity = sat_capacity
        self.sat_buffer_size = sat_buffer_size
        
        # Network components
        self.satellites: Dict[str, Satellite] = {}
        self.ground_stations: Dict[str, GroundStation] = {}
        self.links: Dict[str, Link] = {}
        
        # NetworkX graph for routing
        self.graph: nx.DiGraph = nx.DiGraph()
        
        # Build the constellation
        self._build_constellation()
        
    def _build_constellation(self):
        """Build the satellite constellation with ISL connections"""
        # Create satellites
        self._create_satellites()
        # Create intra-plane ISL (along orbit)
        self._create_intra_plane_isl()
        # Create inter-plane ISL (between adjacent planes)
        self._create_inter_plane_isl()
        # Build NetworkX graph
        self._build_graph()
        
    def _create_satellites(self):
        """Create all satellites in the constellation"""
        earth_radius_km = 6371.0
        orbit_radius_km = earth_radius_km + self.altitude_km
        
        for plane_idx in range(self.num_planes):
            # Calculate the longitude offset for this plane (RAAN)
            raan_deg = (360.0 / self.num_planes) * plane_idx
            
            for sat_idx in range(self.sats_per_plane):
                # Calculate satellite position
                # Mean anomaly determines position along orbit
                mean_anomaly_deg = (360.0 / self.sats_per_plane) * sat_idx
                
                # Simplified position calculation (static snapshot)
                # In reality, this would be time-dependent
                lat, lon = self._orbital_to_geographic(
                    raan_deg, 
                    self.inclination_deg, 
                    mean_anomaly_deg
                )
                
                sat_id = f"SAT_{plane_idx}_{sat_idx}"
                satellite = Satellite(
                    id=sat_id,
                    plane_id=plane_idx,
                    sat_id=sat_idx,
                    position=(lat, lon, self.altitude_km),
                    capacity=self.sat_capacity,
                    buffer_size=self.sat_buffer_size
                )
                self.satellites[sat_id] = satellite
                
    def _orbital_to_geographic(
        self, 
        raan_deg: float, 
        inclination_deg: float, 
        mean_anomaly_deg: float
    ) -> Tuple[float, float]:
        """
        Convert orbital elements to geographic coordinates (simplified)
        
        Args:
            raan_deg: Right Ascension of Ascending Node
            inclination_deg: Orbital inclination
            mean_anomaly_deg: Mean anomaly (position along orbit)
            
        Returns:
            (latitude, longitude) in degrees
        """
        # Simplified calculation - in production use SGP4/SDP4
        raan = math.radians(raan_deg)
        inc = math.radians(inclination_deg)
        ma = math.radians(mean_anomaly_deg)
        
        # Argument of latitude
        u = ma
        
        # Position in orbital plane
        x_orb = math.cos(u)
        y_orb = math.sin(u)
        
        # Transform to Earth-fixed frame (simplified, no Earth rotation)
        x = x_orb * math.cos(raan) - y_orb * math.cos(inc) * math.sin(raan)
        y = x_orb * math.sin(raan) + y_orb * math.cos(inc) * math.cos(raan)
        z = y_orb * math.sin(inc)
        
        # Convert to lat/lon
        lat = math.degrees(math.asin(z))
        lon = math.degrees(math.atan2(y, x))
        
        return lat, lon
    
    def _create_intra_plane_isl(self):
        """Create ISL links within the same orbital plane"""
        for plane_idx in range(self.num_planes):
            for sat_idx in range(self.sats_per_plane):
                # Connect to next satellite in the same plane (ring topology)
                next_sat_idx = (sat_idx + 1) % self.sats_per_plane
                
                src_id = f"SAT_{plane_idx}_{sat_idx}"
                dst_id = f"SAT_{plane_idx}_{next_sat_idx}"
                
                # Calculate propagation delay based on distance
                delay = self._calculate_isl_delay(src_id, dst_id)
                
                # Forward link
                link_id_fwd = f"ISL_{src_id}_{dst_id}"
                self.links[link_id_fwd] = Link(
                    id=link_id_fwd,
                    source=src_id,
                    target=dst_id,
                    link_type=LinkType.ISL_INTRA,
                    bandwidth=self.isl_bandwidth_mbps,
                    propagation_delay=delay
                )
                
                # Backward link
                link_id_bwd = f"ISL_{dst_id}_{src_id}"
                self.links[link_id_bwd] = Link(
                    id=link_id_bwd,
                    source=dst_id,
                    target=src_id,
                    link_type=LinkType.ISL_INTRA,
                    bandwidth=self.isl_bandwidth_mbps,
                    propagation_delay=delay
                )
                
    def _create_inter_plane_isl(self):
        """Create ISL links between adjacent orbital planes"""
        for plane_idx in range(self.num_planes):
            next_plane_idx = (plane_idx + 1) % self.num_planes
            
            # Skip inter-plane ISL at the seam (counter-rotating planes)
            # This is a simplification - in reality, seam ISL has special handling
            if next_plane_idx == 0 and self.num_planes > 2:
                continue
                
            for sat_idx in range(self.sats_per_plane):
                src_id = f"SAT_{plane_idx}_{sat_idx}"
                dst_id = f"SAT_{next_plane_idx}_{sat_idx}"
                
                delay = self._calculate_isl_delay(src_id, dst_id)
                
                # Forward link
                link_id_fwd = f"ISL_{src_id}_{dst_id}"
                self.links[link_id_fwd] = Link(
                    id=link_id_fwd,
                    source=src_id,
                    target=dst_id,
                    link_type=LinkType.ISL_INTER,
                    bandwidth=self.isl_bandwidth_mbps,
                    propagation_delay=delay
                )
                
                # Backward link
                link_id_bwd = f"ISL_{dst_id}_{src_id}"
                self.links[link_id_bwd] = Link(
                    id=link_id_bwd,
                    source=dst_id,
                    target=src_id,
                    link_type=LinkType.ISL_INTER,
                    bandwidth=self.isl_bandwidth_mbps,
                    propagation_delay=delay
                )
    
    def _calculate_isl_delay(self, src_id: str, dst_id: str) -> float:
        """
        Calculate propagation delay between two satellites
        
        Args:
            src_id: Source satellite ID
            dst_id: Destination satellite ID
            
        Returns:
            Propagation delay in milliseconds
        """
        src_sat = self.satellites[src_id]
        dst_sat = self.satellites[dst_id]
        
        # Convert to Cartesian coordinates
        earth_radius_km = 6371.0
        
        def to_cartesian(lat, lon, alt):
            r = earth_radius_km + alt
            lat_rad = math.radians(lat)
            lon_rad = math.radians(lon)
            x = r * math.cos(lat_rad) * math.cos(lon_rad)
            y = r * math.cos(lat_rad) * math.sin(lon_rad)
            z = r * math.sin(lat_rad)
            return x, y, z
        
        src_pos = to_cartesian(*src_sat.position)
        dst_pos = to_cartesian(*dst_sat.position)
        
        # Calculate distance
        distance_km = math.sqrt(
            (src_pos[0] - dst_pos[0])**2 +
            (src_pos[1] - dst_pos[1])**2 +
            (src_pos[2] - dst_pos[2])**2
        )
        
        # Light speed = 299,792 km/s
        speed_of_light_km_per_ms = 299.792
        delay_ms = distance_km / speed_of_light_km_per_ms
        
        return delay_ms
    
    def _build_graph(self):
        """Build NetworkX graph from satellites and links"""
        self.graph.clear()
        
        # Add satellite nodes
        for sat_id, sat in self.satellites.items():
            self.graph.add_node(
                sat_id,
                type=NodeType.SATELLITE,
                plane=sat.plane_id,
                sat_idx=sat.sat_id,
                position=sat.position
            )
        
        # Add ground station nodes
        for gs_id, gs in self.ground_stations.items():
            self.graph.add_node(
                gs_id,
                type=NodeType.GROUND_STATION,
                position=gs.position
            )
        
        # Add links as edges
        for link_id, link in self.links.items():
            self.graph.add_edge(
                link.source,
                link.target,
                link_id=link_id,
                weight=link.propagation_delay,
                bandwidth=link.bandwidth,
                link_type=link.link_type
            )
    
    def add_ground_station(
        self, 
        gs_id: str, 
        latitude: float, 
        longitude: float,
        gsl_bandwidth_mbps: float = 1000.0
    ):
        """
        Add a ground station and connect to visible satellites
        
        Args:
            gs_id: Ground station identifier
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            gsl_bandwidth_mbps: Ground-satellite link bandwidth
        """
        gs = GroundStation(id=gs_id, position=(latitude, longitude))
        self.ground_stations[gs_id] = gs
        
        # Find visible satellites and create GSL
        visible_sats = self._find_visible_satellites(latitude, longitude)
        
        for sat_id in visible_sats:
            gs.connected_sats.append(sat_id)
            
            # Calculate GSL delay
            sat = self.satellites[sat_id]
            distance_km = self._calculate_ground_sat_distance(
                latitude, longitude, sat.position
            )
            delay_ms = distance_km / 299.792
            
            # Create bidirectional GSL
            link_up = Link(
                id=f"GSL_{gs_id}_{sat_id}",
                source=gs_id,
                target=sat_id,
                link_type=LinkType.GSL,
                bandwidth=gsl_bandwidth_mbps,
                propagation_delay=delay_ms
            )
            self.links[link_up.id] = link_up
            
            link_down = Link(
                id=f"GSL_{sat_id}_{gs_id}",
                source=sat_id,
                target=gs_id,
                link_type=LinkType.GSL,
                bandwidth=gsl_bandwidth_mbps,
                propagation_delay=delay_ms
            )
            self.links[link_down.id] = link_down
        
        # Rebuild graph
        self._build_graph()
        
    def _find_visible_satellites(
        self, 
        latitude: float, 
        longitude: float, 
        min_elevation_deg: float = 25.0
    ) -> List[str]:
        """
        Find satellites visible from a ground location
        
        Args:
            latitude: Ground station latitude
            longitude: Ground station longitude
            min_elevation_deg: Minimum elevation angle
            
        Returns:
            List of visible satellite IDs
        """
        visible = []
        earth_radius_km = 6371.0
        
        for sat_id, sat in self.satellites.items():
            sat_lat, sat_lon, sat_alt = sat.position
            
            # Calculate angular distance on Earth's surface
            lat1 = math.radians(latitude)
            lat2 = math.radians(sat_lat)
            dlon = math.radians(sat_lon - longitude)
            
            # Great circle angular distance
            cos_d = math.sin(lat1) * math.sin(lat2) + \
                    math.cos(lat1) * math.cos(lat2) * math.cos(dlon)
            cos_d = max(-1, min(1, cos_d))  # Clamp for numerical stability
            d = math.acos(cos_d)
            
            # Calculate elevation angle
            orbit_radius = earth_radius_km + sat_alt
            sin_el = (math.cos(d) * orbit_radius - earth_radius_km) / \
                     math.sqrt(earth_radius_km**2 + orbit_radius**2 - 
                              2 * earth_radius_km * orbit_radius * math.cos(d))
            
            if sin_el >= math.sin(math.radians(min_elevation_deg)):
                visible.append(sat_id)
        
        return visible
    
    def _calculate_ground_sat_distance(
        self,
        gs_lat: float,
        gs_lon: float,
        sat_position: Tuple[float, float, float]
    ) -> float:
        """Calculate distance between ground station and satellite"""
        earth_radius_km = 6371.0
        sat_lat, sat_lon, sat_alt = sat_position
        
        # Ground station position
        gs_r = earth_radius_km
        gs_x = gs_r * math.cos(math.radians(gs_lat)) * math.cos(math.radians(gs_lon))
        gs_y = gs_r * math.cos(math.radians(gs_lat)) * math.sin(math.radians(gs_lon))
        gs_z = gs_r * math.sin(math.radians(gs_lat))
        
        # Satellite position
        sat_r = earth_radius_km + sat_alt
        sat_x = sat_r * math.cos(math.radians(sat_lat)) * math.cos(math.radians(sat_lon))
        sat_y = sat_r * math.cos(math.radians(sat_lat)) * math.sin(math.radians(sat_lon))
        sat_z = sat_r * math.sin(math.radians(sat_lat))
        
        distance = math.sqrt(
            (gs_x - sat_x)**2 + (gs_y - sat_y)**2 + (gs_z - sat_z)**2
        )
        return distance
    
    def get_link(self, src: str, dst: str) -> Optional[Link]:
        """Get link between two nodes"""
        link_id = f"ISL_{src}_{dst}"
        if link_id in self.links:
            return self.links[link_id]
        link_id = f"GSL_{src}_{dst}"
        if link_id in self.links:
            return self.links[link_id]
        return None
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors of a node"""
        return list(self.graph.neighbors(node_id))
    
    def get_total_nodes(self) -> int:
        """Get total number of nodes"""
        return len(self.satellites) + len(self.ground_stations)
    
    def get_total_links(self) -> int:
        """Get total number of links"""
        return len(self.links)
    
    def reset_all_stats(self):
        """Reset statistics for all nodes and links"""
        for sat in self.satellites.values():
            sat.reset_stats()
        for link in self.links.values():
            link.reset_stats()
    
    def reset_link_loads(self):
        """Reset current load on all links (call each time step)"""
        for link in self.links.values():
            link.reset_load()
    
    def get_network_stats(self) -> Dict:
        """Get overall network statistics"""
        total_processed = sum(s.processed_packets for s in self.satellites.values())
        total_dropped_nodes = sum(s.dropped_packets for s in self.satellites.values())
        total_link_bytes = sum(l.total_bytes for l in self.links.values())
        total_dropped_bytes = sum(l.dropped_bytes for l in self.links.values())
        
        avg_utilization = np.mean([
            l.get_utilization() for l in self.links.values()
        ]) if self.links else 0.0
        
        max_utilization = max([
            l.get_utilization() for l in self.links.values()
        ]) if self.links else 0.0
        
        return {
            "total_satellites": len(self.satellites),
            "total_ground_stations": len(self.ground_stations),
            "total_links": len(self.links),
            "total_processed_packets": total_processed,
            "total_dropped_packets_at_nodes": total_dropped_nodes,
            "total_link_bytes": total_link_bytes,
            "total_dropped_bytes_at_links": total_dropped_bytes,
            "avg_link_utilization": avg_utilization,
            "max_link_utilization": max_utilization
        }
    
    def __repr__(self):
        return (f"LEOConstellation(planes={self.num_planes}, "
                f"sats_per_plane={self.sats_per_plane}, "
                f"total_sats={len(self.satellites)}, "
                f"total_links={len(self.links)})")
