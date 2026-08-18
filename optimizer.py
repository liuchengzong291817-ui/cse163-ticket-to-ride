"""
Chengzong Liu
CSE 163 A
This module builds and evaluates shared Ticket to Ride route networks under
a 45-train budget, including matched learned-value and blocked-edge tests.
"""

import heapq
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from data_features import canonical_edge


Edge = Tuple[str, str]
ROUTE_SCORE_BY_LENGTH = {
    1: 1,
    2: 2,
    3: 4,
    4: 7,
    6: 15,
    8: 21,
}
TRAIN_COST_WEIGHT = 3.0


def _route_lookup(routes: pd.DataFrame) -> Dict[Edge, Dict[str, int]]:
    """Return train cost and route points for each undirected route."""
    lookup = {}
    for row in routes.itertuples(index=False):
        length = int(row.Carriages)
        if length not in ROUTE_SCORE_BY_LENGTH:
            raise ValueError("a route has an unsupported carriage count")
        edge = canonical_edge(row.Source, row.Target)
        lookup[edge] = {
            "length": length,
            "route_points": ROUTE_SCORE_BY_LENGTH[length],
        }
    return lookup


def _adjacency(route_lookup: Dict[Edge, Dict[str, int]]) -> Dict[
        str, List[str]]:
    """Return an undirected adjacency list from a route lookup."""
    graph = defaultdict(list)
    for source, target in route_lookup:
        graph[source].append(target)
        graph[target].append(source)
    for city in graph:
        graph[city].sort()
    return dict(graph)


def _marginal_path(
        graph: Dict[str, List[str]],
        route_lookup: Dict[Edge, Dict[str, int]],
        source: str,
        target: str,
        built_edges: Set[Edge],
        blocked_edges: Set[Edge]) -> Tuple[float, List[str]]:
    """
    Return the least additional-train cost and one deterministic path.
    Already-built edges have zero marginal cost and blocked edges are absent.
    """
    distances = {source: 0.0}
    previous: Dict[str, str] = {}
    queue = [(0.0, source)]

    while queue:
        distance, city = heapq.heappop(queue)
        if distance != distances[city]:
            continue
        for neighbor in graph.get(city, []):
            edge = canonical_edge(city, neighbor)
            if edge in blocked_edges:
                continue
            edge_cost = 0 if edge in built_edges else (
                route_lookup[edge]["length"]
            )
            new_distance = distance + edge_cost
            if new_distance < distances.get(neighbor, math.inf):
                distances[neighbor] = new_distance
                previous[neighbor] = city
                heapq.heappush(queue, (new_distance, neighbor))

    if target not in distances:
        return math.inf, []

    path = [target]
    while path[-1] != source:
        path.append(previous[path[-1]])
    path.reverse()
    return distances[target], path


def _path_edges(path: Sequence[str]) -> Set[Edge]:
    """Return canonical edges from a path represented by city names."""
    return {
        canonical_edge(path[index], path[index + 1])
        for index in range(len(path) - 1)
    }


def _edge_dependence(
        graph: Dict[str, List[str]],
        route_lookup: Dict[Edge, Dict[str, int]],
        destinations: pd.DataFrame) -> Dict[Edge, float]:
    """
    Return normalized destination-path usage in the interval [0, 1].
    This target-independent score is high for edges on many ticket paths.
    """
    counts = Counter()
    for ticket in destinations.itertuples(index=False):
        _, path = _marginal_path(
            graph,
            route_lookup,
            ticket.Source,
            ticket.Target,
            set(),
            set(),
        )
        counts.update(_path_edges(path))

    largest_count = max(counts.values(), default=0)
    if largest_count == 0:
        return {edge: 0.0 for edge in route_lookup}
    return {
        edge: counts[edge] / largest_count
        for edge in route_lookup
    }


def _validate_values(
        destinations: pd.DataFrame,
        ticket_values: Sequence[float]) -> np.ndarray:
    """Return a finite numeric value vector with one entry per ticket."""
    values = np.asarray(ticket_values, dtype=float)
    if values.ndim != 1 or len(values) != len(destinations):
        raise ValueError("ticket_values must contain one value per ticket")
    if not np.isfinite(values).all():
        raise ValueError("ticket_values must all be finite")
    return values


def optimize_network(
        routes: pd.DataFrame,
        destinations: pd.DataFrame,
        ticket_values: Sequence[float],
        budget: int = 45,
        risk_weight: float = 0.0,
        blocked_edges: Optional[Set[Edge]] = None) -> Dict[str, object]:
    """
    Greedily build a shared route network using ticket_values.

    At every step, choose the feasible ticket with the largest original-style
    net value: ticket value plus newly claimed route points, less three points
    per marginal train and a route-dependence penalty. Return selected ticket
    row indices, unique built edges, trains used, and the objective inputs. The
    input tickets are treated as already accepted candidates, so the planner
    keeps connecting the best remaining candidate until none fits the budget;
    negative net value affects ranking but is not an optional-draw stop rule.
    Deterministic tie-breaking makes matched comparisons reproducible.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if risk_weight < 0:
        raise ValueError("risk_weight must be nonnegative")

    values = _validate_values(destinations, ticket_values)
    route_lookup = _route_lookup(routes)
    graph = _adjacency(route_lookup)
    blocked = set() if blocked_edges is None else set(blocked_edges)
    if not blocked.issubset(route_lookup):
        raise ValueError("blocked_edges contains an unknown route")
    dependence = _edge_dependence(graph, route_lookup, destinations)

    built_edges: Set[Edge] = set()
    selected_indices = []
    remaining_indices = set(range(len(destinations)))
    trains_used = 0

    while remaining_indices:
        candidates = []
        for index in sorted(remaining_indices):
            ticket = destinations.iloc[index]
            unused_cost, path = _marginal_path(
                graph,
                route_lookup,
                ticket["Source"],
                ticket["Target"],
                built_edges,
                blocked,
            )
            if not path:
                continue

            complete_path_edges = _path_edges(path)
            new_edges = complete_path_edges - built_edges
            marginal_cost = sum(
                route_lookup[edge]["length"] for edge in new_edges
            )
            if marginal_cost != unused_cost:
                raise RuntimeError("marginal path cost is inconsistent")
            if trains_used + marginal_cost > budget:
                continue

            added_route_points = sum(
                route_lookup[edge]["route_points"] for edge in new_edges
            )
            mean_dependence = np.mean([
                dependence[edge] for edge in complete_path_edges
            ])
            risk_penalty = (
                risk_weight * values[index] * mean_dependence
            )
            gain = values[index] + added_route_points - risk_penalty
            net_value = gain - TRAIN_COST_WEIGHT * marginal_cost
            if marginal_cost == 0:
                efficiency = math.inf
            else:
                efficiency = gain / marginal_cost

            candidates.append((
                net_value,
                efficiency,
                -marginal_cost,
                -index,
                index,
                new_edges,
            ))

        if not candidates:
            break
        _, _, _, _, index, new_edges = max(candidates)

        marginal_cost = sum(
            route_lookup[edge]["length"] for edge in new_edges
        )
        built_edges.update(new_edges)
        selected_indices.append(index)
        remaining_indices.remove(index)
        trains_used += marginal_cost

    return {
        "selected_indices": selected_indices,
        "unselected_indices": sorted(remaining_indices),
        "built_edges": built_edges,
        "trains_used": trains_used,
        "candidate_count": len(destinations),
        "budget": budget,
        "risk_weight": risk_weight,
        "blocked_edges": blocked,
        "predicted_ticket_value": float(values[selected_indices].sum()),
    }


def _connected(
        built_edges: Set[Edge],
        source: str,
        target: str,
        blocked_edges: Set[Edge]) -> bool:
    """Return whether two cities remain connected in the claimed network."""
    graph = defaultdict(list)
    for edge in built_edges - blocked_edges:
        first, second = edge
        graph[first].append(second)
        graph[second].append(first)

    visited = {source}
    pending = [source]
    while pending:
        city = pending.pop()
        for neighbor in graph[city]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return target in visited


def evaluate_plan(
        plan: Dict[str, object],
        routes: pd.DataFrame,
        destinations: pd.DataFrame,
        stress_edges: Optional[Sequence[Edge]] = None) -> Dict[str, float]:
    """
    Return actual ticket, route, efficiency, and post-hoc robustness metrics.
    Stress testing removes one edge at a time without allowing repairs.
    """
    route_lookup = _route_lookup(routes)
    selected_indices = list(plan["selected_indices"])
    built_edges = set(plan["built_edges"])
    trains_used = int(plan["trains_used"])
    ticket_points = float(
        destinations.iloc[selected_indices]["Points"].sum()
    )
    candidate_ticket_points = float(destinations["Points"].sum())
    route_points = float(sum(
        route_lookup[edge]["route_points"] for edge in built_edges
    ))
    total_points = ticket_points + route_points
    connected_count = sum(
        _connected(
            built_edges,
            destinations.iloc[index]["Source"],
            destinations.iloc[index]["Target"],
            set(),
        )
        for index in selected_indices
    )
    if connected_count != len(selected_indices):
        raise RuntimeError("a selected ticket is not connected by the plan")
    ticket_efficiency = 0.0
    total_efficiency = 0.0
    if trains_used > 0:
        ticket_efficiency = ticket_points / trains_used
        total_efficiency = total_points / trains_used
    metrics = {
        "trains_used": float(trains_used),
        "tickets_connected": float(len(selected_indices)),
        "candidate_tickets": float(len(destinations)),
        "unconnected_candidates": float(
            len(destinations) - len(selected_indices)
        ),
        "ticket_coverage_rate": len(selected_indices) / len(destinations),
        "candidate_ticket_points": candidate_ticket_points,
        "unconnected_candidate_points": (
            candidate_ticket_points - ticket_points
        ),
        "printed_points_coverage_rate": (
            ticket_points / candidate_ticket_points
        ),
        "predicted_ticket_value": float(
            plan["predicted_ticket_value"]
        ),
        "printed_ticket_points": ticket_points,
        "route_points": route_points,
        "total_points": total_points,
        "ticket_points_per_train": ticket_efficiency,
        "total_points_per_train": total_efficiency,
        "points_per_train": total_efficiency,
    }

    edges_to_test = [] if stress_edges is None else list(stress_edges)
    if edges_to_test:
        retained_points = []
        for blocked_edge in edges_to_test:
            retained = 0.0
            for index in selected_indices:
                ticket = destinations.iloc[index]
                if _connected(
                        built_edges,
                        ticket["Source"],
                        ticket["Target"],
                        {blocked_edge}):
                    retained += float(ticket["Points"])
            retained_points.append(retained)

        metrics["mean_blocked_ticket_points"] = float(
            np.mean(retained_points)
        )
        metrics["worst_blocked_ticket_points"] = float(
            np.min(retained_points)
        )
        mean_retention = 0.0
        worst_retention = 0.0
        if ticket_points > 0:
            mean_retention = np.mean(retained_points) / ticket_points
            worst_retention = np.min(retained_points) / ticket_points
        metrics["mean_blocked_retention"] = float(mean_retention)
        metrics["worst_blocked_retention"] = float(worst_retention)
    return metrics


def blocked_edge_stress_test(
        routes: pd.DataFrame,
        destinations: pd.DataFrame,
        ticket_values: Sequence[float],
        stress_edges: Sequence[Edge],
        budget: int = 45,
        risk_weight: float = 0.0) -> pd.DataFrame:
    """
    Compare post-hoc damage with adaptive re-optimization for each blocked
    edge. The same value vector, budget, and tie-breaking are used throughout.
    """
    base_plan = optimize_network(
        routes,
        destinations,
        ticket_values,
        budget=budget,
        risk_weight=risk_weight,
    )
    base_points = evaluate_plan(
        base_plan,
        routes,
        destinations,
    )["printed_ticket_points"]
    records = []

    for blocked_edge in stress_edges:
        post_hoc = evaluate_plan(
            base_plan,
            routes,
            destinations,
            stress_edges=[blocked_edge],
        )
        adaptive_plan = optimize_network(
            routes,
            destinations,
            ticket_values,
            budget=budget,
            risk_weight=risk_weight,
            blocked_edges={blocked_edge},
        )
        adaptive = evaluate_plan(
            adaptive_plan,
            routes,
            destinations,
        )
        records.append({
            "blocked_source": blocked_edge[0],
            "blocked_target": blocked_edge[1],
            "base_ticket_points": base_points,
            "post_hoc_ticket_points": (
                post_hoc["mean_blocked_ticket_points"]
            ),
            "post_hoc_retention": post_hoc["mean_blocked_retention"],
            "adaptive_ticket_points": adaptive["printed_ticket_points"],
            "adaptive_total_points": adaptive["total_points"],
            "adaptive_trains_used": adaptive["trains_used"],
        })
    return pd.DataFrame(records)


def _matched_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Return paired baseline-versus-learned summaries for trial metrics."""
    metrics = [
        "printed_ticket_points",
        "route_points",
        "total_points",
        "ticket_points_per_train",
        "total_points_per_train",
        "tickets_connected",
        "ticket_coverage_rate",
        "printed_points_coverage_rate",
        "trains_used",
    ]
    optional_metrics = [
        "mean_blocked_ticket_points",
        "worst_blocked_ticket_points",
        "mean_blocked_retention",
        "worst_blocked_retention",
    ]
    metrics.extend([
        metric for metric in optional_metrics if metric in results.columns
    ])
    records = []
    for metric in metrics:
        paired = results.pivot(
            index="trial",
            columns="method",
            values=metric,
        )
        baseline = paired["Original baseline"]
        learned = paired["Learned revised"]
        difference = learned - baseline
        records.append({
            "metric": metric,
            "baseline_median": baseline.median(),
            "learned_median": learned.median(),
            "median_paired_difference": difference.median(),
            "mean_paired_difference": difference.mean(),
            "learned_greater_rate": (difference > 0).mean(),
            "equal_rate": (difference == 0).mean(),
            "baseline_q1": baseline.quantile(0.25),
            "baseline_q3": baseline.quantile(0.75),
            "learned_q1": learned.quantile(0.25),
            "learned_q3": learned.quantile(0.75),
        })
    return pd.DataFrame(records)


def learned_route_values(
        routes: pd.DataFrame,
        destinations: pd.DataFrame,
        out_of_fold_predictions: Sequence[float]) -> np.ndarray:
    """
    Return printed points plus an out-of-fold map-value premium.

    The premium is predicted printed points minus the ticket's minimum train
    cost. Printed points are known when a card is visible; the cross-fitted
    premium replaces the previous fixed assumption about structural value.
    """
    predictions = _validate_values(destinations, out_of_fold_predictions)
    route_lookup = _route_lookup(routes)
    graph = _adjacency(route_lookup)
    shortest_costs = []
    for ticket in destinations.itertuples(index=False):
        cost, _ = _marginal_path(
            graph,
            route_lookup,
            ticket.Source,
            ticket.Target,
            set(),
            set(),
        )
        shortest_costs.append(cost)
    return (
        destinations["Points"].to_numpy(dtype=float)
        + predictions
        - np.asarray(shortest_costs)
    )


def run_matched_simulations(
        routes: pd.DataFrame,
        destinations: pd.DataFrame,
        out_of_fold_predictions: Sequence[float],
        stress_edges: Optional[Sequence[Edge]] = None,
        trials: int = 250,
        pool_size: int = 12,
        budget: int = 45,
        learned_risk_weight: float = 0.20,
        random_seed: int = 163) -> Tuple[
            pd.DataFrame,
            pd.DataFrame,
            Dict[str, Dict[str, object]]]:
    """
    Run deterministic matched candidate-pool simulations.

    Every trial samples one already-accepted candidate pool and sends that
    identical pool through
    the original printed-point heuristic and revised learned-value heuristic.
    Both methods use the same optimizer, budget, and tie-breaking. The revised
    method uses the fixed learned_risk_weight for its dependence penalty.
    Return trial-level results, paired summaries, and full-deck plans.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if pool_size <= 0 or pool_size > len(destinations):
        raise ValueError("pool_size must be between 1 and the deck size")

    predictions = _validate_values(
        destinations,
        out_of_fold_predictions,
    )
    learned = learned_route_values(routes, destinations, predictions)
    baseline = destinations["Points"].to_numpy(dtype=float)
    random = np.random.default_rng(random_seed)
    records = []

    for trial in range(trials):
        global_indices = np.sort(
            random.choice(len(destinations), size=pool_size, replace=False)
        )
        pool = destinations.iloc[global_indices].reset_index(drop=True)
        methods = {
            "Original baseline": (
                baseline[global_indices],
                0.0,
            ),
            "Learned revised": (
                learned[global_indices],
                learned_risk_weight,
            ),
        }

        for method, (values, method_risk_weight) in methods.items():
            plan = optimize_network(
                routes,
                pool,
                values,
                budget=budget,
                risk_weight=method_risk_weight,
            )
            metrics = evaluate_plan(
                plan,
                routes,
                pool,
                stress_edges=stress_edges,
            )
            selected_global = tuple(
                int(global_indices[index])
                for index in plan["selected_indices"]
            )
            records.append({
                "trial": trial,
                "method": method,
                "candidate_indices": tuple(
                    int(index) for index in global_indices
                ),
                "selected_global_indices": selected_global,
                "built_edges": tuple(sorted(plan["built_edges"])),
                **metrics,
            })

    results = pd.DataFrame(records)
    summary = _matched_summary(results)
    full_deck_plans = {
        "Original baseline": optimize_network(
            routes,
            destinations,
            baseline,
            budget=budget,
            risk_weight=0.0,
        ),
        "Learned revised": optimize_network(
            routes,
            destinations,
            learned,
            budget=budget,
            risk_weight=learned_risk_weight,
        ),
    }
    return results, summary, full_deck_plans
