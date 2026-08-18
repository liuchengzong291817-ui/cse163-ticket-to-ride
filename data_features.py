"""
Chengzong Liu
CSE 163 A
This module loads Ticket to Ride: Europe data and derives graph features for
each destination ticket.
"""

import heapq
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


EARTH_RADIUS_KM = 6371.0
REQUIRED_CITY_COLUMNS = {"City", "Longitude", "Latitude"}
REQUIRED_ROUTE_COLUMNS = {
    "Source",
    "Target",
    "Carriages",
    "Colored",
    "Tunnel",
    "Engine",
}
REQUIRED_DESTINATION_COLUMNS = {"Source", "Target", "Points"}


def canonical_edge(source: str, target: str) -> Tuple[str, str]:
    """Return an undirected edge with its two city names in sorted order."""
    return tuple(sorted((source, target)))


def load_data(
        data_directory: str) -> Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load cities, routes, and destinations from CSV files in data_directory.
    Return the three cleaned DataFrames in that order.
    """
    directory = Path(data_directory)
    cities = pd.read_csv(directory / "cities.csv")
    routes = pd.read_csv(directory / "routes.csv")
    destinations = pd.read_csv(directory / "destinations.csv")

    cities = cities.copy()
    routes = routes.copy()
    destinations = destinations.copy()

    cities["City"] = cities["City"].str.strip()
    routes["Source"] = routes["Source"].str.strip()
    routes["Target"] = routes["Target"].str.strip()
    destinations["Source"] = destinations["Source"].str.strip()
    destinations["Target"] = destinations["Target"].str.strip()
    return cities, routes, destinations


def validate_data(cities: pd.DataFrame,
                  routes: pd.DataFrame,
                  destinations: pd.DataFrame) -> None:
    """
    Validate required columns, missing values, positive values, unique routes,
    and endpoint membership for the three input DataFrames. Raise ValueError
    when any requirement is not satisfied.
    """
    if not REQUIRED_CITY_COLUMNS.issubset(cities.columns):
        raise ValueError("cities.csv is missing one or more required columns")
    if not REQUIRED_ROUTE_COLUMNS.issubset(routes.columns):
        raise ValueError("routes.csv is missing one or more required columns")
    if not REQUIRED_DESTINATION_COLUMNS.issubset(destinations.columns):
        raise ValueError(
            "destinations.csv is missing one or more required columns"
        )
    if cities.isna().any().any():
        raise ValueError("cities.csv contains missing values")
    if routes.isna().any().any():
        raise ValueError("routes.csv contains missing values")
    if destinations.isna().any().any():
        raise ValueError("destinations.csv contains missing values")
    if cities["City"].duplicated().any():
        raise ValueError("cities.csv contains duplicate city names")
    if (routes["Carriages"] <= 0).any():
        raise ValueError("route carriage counts must be positive")
    if (destinations["Points"] <= 0).any():
        raise ValueError("destination point values must be positive")

    route_edges = routes.apply(
        lambda row: canonical_edge(row["Source"], row["Target"]),
        axis=1,
    )
    if route_edges.duplicated().any():
        raise ValueError("routes.csv contains duplicate undirected routes")

    known_cities = set(cities["City"])
    route_endpoints = set(routes["Source"]) | set(routes["Target"])
    destination_endpoints = (
        set(destinations["Source"]) | set(destinations["Target"])
    )
    if not route_endpoints.issubset(known_cities):
        raise ValueError("a route endpoint is missing from cities.csv")
    if not destination_endpoints.issubset(known_cities):
        raise ValueError("a destination endpoint is missing from cities.csv")


def build_graph(routes: pd.DataFrame) -> Dict[
        str, List[Tuple[str, int]]]:
    """
    Build and return an undirected adjacency list from the routes DataFrame.
    Each neighbor entry contains a city name and its route carriage cost.
    """
    graph: Dict[str, List[Tuple[str, int]]] = {}
    for row in routes.itertuples(index=False):
        graph.setdefault(row.Source, []).append(
            (row.Target, int(row.Carriages))
        )
        graph.setdefault(row.Target, []).append(
            (row.Source, int(row.Carriages))
        )
    return graph


def dijkstra(graph: Dict[str, List[Tuple[str, int]]],
             source: str,
             target: str,
             excluded_edge: Optional[Tuple[str, str]] = None
             ) -> Tuple[float, List[str], int]:
    """
    Return the minimum carriage cost, one minimum path, and the number of
    equally short paths from source to target. Ignore excluded_edge when it is
    provided, and return infinity, an empty path, and zero if no path exists.
    """
    distances = {source: 0.0}
    path_counts = {source: 1}
    previous: Dict[str, str] = {}
    queue = [(0.0, source)]

    while queue:
        current_distance, city = heapq.heappop(queue)
        if current_distance != distances[city]:
            continue

        for neighbor, carriage_cost in graph.get(city, []):
            edge = canonical_edge(city, neighbor)
            if excluded_edge is not None and edge == excluded_edge:
                continue

            new_distance = current_distance + carriage_cost
            known_distance = distances.get(neighbor, math.inf)
            if new_distance < known_distance:
                distances[neighbor] = new_distance
                path_counts[neighbor] = path_counts[city]
                previous[neighbor] = city
                heapq.heappush(queue, (new_distance, neighbor))
            elif new_distance == known_distance:
                path_counts[neighbor] += path_counts[city]

    if target not in distances:
        return math.inf, [], 0

    path = [target]
    while path[-1] != source:
        path.append(previous[path[-1]])
    path.reverse()
    return distances[target], path, path_counts[target]


def path_edges(path: List[str]) -> Tuple[Tuple[str, str], ...]:
    """Return the canonical undirected edges used by a sequence of cities."""
    return tuple(
        canonical_edge(path[index], path[index + 1])
        for index in range(len(path) - 1)
    )


def haversine_distance(longitude1: float,
                       latitude1: float,
                       longitude2: float,
                       latitude2: float) -> float:
    """
    Return the great-circle distance in kilometers between two longitude and
    latitude coordinate pairs.
    """
    longitude1_radians = math.radians(longitude1)
    latitude1_radians = math.radians(latitude1)
    longitude2_radians = math.radians(longitude2)
    latitude2_radians = math.radians(latitude2)

    longitude_difference = longitude2_radians - longitude1_radians
    latitude_difference = latitude2_radians - latitude1_radians
    haversine_value = (
        math.sin(latitude_difference / 2) ** 2
        + math.cos(latitude1_radians)
        * math.cos(latitude2_radians)
        * math.sin(longitude_difference / 2) ** 2
    )
    central_angle = 2 * math.asin(math.sqrt(haversine_value))
    return EARTH_RADIUS_KM * central_angle


def _route_lookup(routes: pd.DataFrame) -> Dict[Tuple[str, str], dict]:
    """Return route attributes indexed by canonical undirected edge."""
    lookup = {}
    for row in routes.to_dict("records"):
        lookup[canonical_edge(row["Source"], row["Target"])] = row
    return lookup


def _coordinate_lookup(
        cities: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Return longitude and latitude coordinates indexed by city name."""
    return {
        row.City: (float(row.Longitude), float(row.Latitude))
        for row in cities.itertuples(index=False)
    }


def _detour_features(graph: Dict[str, List[Tuple[str, int]]],
                     source: str,
                     target: str,
                     shortest_cost: float,
                     edges: Tuple[Tuple[str, str], ...]
                     ) -> Tuple[float, bool]:
    """
    Return the largest finite extra cost after removing a path edge and whether
    any path edge disconnects the destination endpoints.
    """
    finite_detours = []
    has_cut_edge = False
    for edge in edges:
        detour_cost, _, _ = dijkstra(
            graph,
            source,
            target,
            excluded_edge=edge,
        )
        if math.isinf(detour_cost):
            has_cut_edge = True
        else:
            finite_detours.append(detour_cost - shortest_cost)

    maximum_detour = max(finite_detours, default=0.0)
    return maximum_detour, has_cut_edge


def build_feature_table(cities: pd.DataFrame,
                        routes: pd.DataFrame,
                        destinations: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all three datasets and return one row of geographic and graph
    features for every destination ticket.
    """
    validate_data(cities, routes, destinations)
    graph = build_graph(routes)
    route_lookup = _route_lookup(routes)
    coordinate_lookup = _coordinate_lookup(cities)
    records = []

    for ticket in destinations.itertuples(index=False):
        shortest_cost, path, equal_path_count = dijkstra(
            graph,
            ticket.Source,
            ticket.Target,
        )
        edges = path_edges(path)
        source_longitude, source_latitude = coordinate_lookup[ticket.Source]
        target_longitude, target_latitude = coordinate_lookup[ticket.Target]
        detour_cost, has_cut_edge = _detour_features(
            graph,
            ticket.Source,
            ticket.Target,
            shortest_cost,
            edges,
        )
        edge_attributes = [route_lookup[edge] for edge in edges]
        edge_count = len(edge_attributes)
        tunnel_count = sum(bool(row["Tunnel"]) for row in edge_attributes)
        ferry_count = sum(int(row["Engine"]) > 0 for row in edge_attributes)

        records.append({
            "Source": ticket.Source,
            "Target": ticket.Target,
            "Points": int(ticket.Points),
            "source_longitude": source_longitude,
            "source_latitude": source_latitude,
            "target_longitude": target_longitude,
            "target_latitude": target_latitude,
            "shortest_trains": int(shortest_cost),
            "path_nodes": tuple(path),
            "path_edges": edges,
            "equal_shortest_paths": int(equal_path_count),
            "straight_line_km": haversine_distance(
                source_longitude,
                source_latitude,
                target_longitude,
                target_latitude,
            ),
            "points_per_train": ticket.Points / shortest_cost,
            "max_detour_trains": detour_cost,
            "has_cut_edge": has_cut_edge,
            "mean_edge_length": shortest_cost / edge_count,
            "intermediate_cities": max(len(path) - 2, 0),
            "tunnel_fraction": tunnel_count / edge_count,
            "ferry_fraction": ferry_count / edge_count,
        })

    features = pd.DataFrame(records)
    high_value_cutoff = features["Points"].quantile(0.75)
    high_value_rows = features[features["Points"] >= high_value_cutoff]
    edge_counts = Counter(
        edge
        for edges in high_value_rows["path_edges"]
        for edge in edges
    )

    overlaps = []
    for row in features.itertuples(index=False):
        is_high_value = row.Points >= high_value_cutoff
        overlap_total = sum(
            edge_counts[edge] - int(is_high_value)
            for edge in row.path_edges
        )
        overlaps.append(overlap_total / len(row.path_edges))
    features["high_value_edge_overlap"] = overlaps
    return features


def important_edges(features: pd.DataFrame,
                    count: int = 5) -> List[Tuple[str, str]]:
    """
    Return the count most frequently used shortest-path edges in features.
    Ties are broken deterministically by the canonical edge names.
    """
    edge_counts = Counter(
        edge
        for edges in features["path_edges"]
        for edge in edges
    )
    ranked = sorted(
        edge_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [edge for edge, _ in ranked[:count]]


def edges_in_feature_table(features: pd.DataFrame) -> Set[
        Tuple[str, str]]:
    """Return the set of every shortest-path edge present in features."""
    return {
        edge
        for edges in features["path_edges"]
        for edge in edges
    }
