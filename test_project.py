"""
Chengzong Liu
CSE 163 A
This program tests data preparation, graph features, machine-learning
evaluation, and optimization for the Ticket to Ride final project.
"""

import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from data_features import (build_feature_table, build_graph, canonical_edge,
                           dijkstra, edges_in_feature_table,
                           haversine_distance, important_edges, load_data,
                           validate_data)
from modeling import (MODEL_FEATURES, MODEL_NAMES, best_prediction_column,
                      evaluate_models, model_matrix)
from optimizer import (blocked_edge_stress_test, evaluate_plan,
                       optimize_network, run_matched_simulations)


DATA_DIRECTORY = Path(__file__).parent / "data"
Graph = Dict[str, List[Tuple[str, int]]]
Edge = Tuple[str, str]


def _load_full_features() -> Tuple[pd.DataFrame,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   pd.DataFrame]:
    """Load the three project datasets and return them with their features."""
    cities, routes, destinations = load_data(str(DATA_DIRECTORY))
    features = build_feature_table(cities, routes, destinations)
    return cities, routes, destinations, features


def _endpoints_connected(edges: Set[Edge],
                         source: str,
                         target: str) -> bool:
    """Return whether source and target are connected by the given edges."""
    graph: Dict[str, List[str]] = {}
    for first, second in edges:
        graph.setdefault(first, []).append(second)
        graph.setdefault(second, []).append(first)

    visited = {source}
    pending = [source]
    while pending:
        city = pending.pop()
        for neighbor in graph.get(city, []):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return target in visited


def test_dijkstra() -> None:
    """Test minimum cost, equal paths, edge exclusion, and no-path output."""
    graph: Graph = {
        "A": [("B", 1), ("C", 1)],
        "B": [("A", 1), ("D", 1)],
        "C": [("A", 1), ("D", 1)],
        "D": [("B", 1), ("C", 1)],
        "X": [],
    }

    cost, path, path_count = dijkstra(graph, "A", "D")
    assert cost == 2
    assert path == ["A", "B", "D"]
    assert path_count == 2

    cost, path, path_count = dijkstra(
        graph,
        "A",
        "D",
        excluded_edge=canonical_edge("A", "B"),
    )
    assert cost == 2
    assert path == ["A", "C", "D"]
    assert path_count == 1

    cost, path, path_count = dijkstra(graph, "A", "X")
    assert math.isinf(cost)
    assert path == []
    assert path_count == 0


def test_haversine_distance() -> None:
    """Test identical coordinates, symmetry, and a known equator distance."""
    assert haversine_distance(12.5, 47.2, 12.5, 47.2) == 0
    forward = haversine_distance(0, 0, 1, 0)
    backward = haversine_distance(1, 0, 0, 0)
    assert math.isclose(forward, 111.19492664455873, rel_tol=1e-12)
    assert math.isclose(forward, backward, rel_tol=1e-12)


def test_data_validation() -> None:
    """Test successful validation and rejection of an undirected duplicate."""
    cities, routes, destinations = load_data(str(DATA_DIRECTORY))
    validate_data(cities, routes, destinations)

    invalid_routes = pd.concat([
        routes,
        routes.iloc[[0]].rename(
            columns={"Source": "Target", "Target": "Source"}
        ),
    ], ignore_index=True)
    try:
        validate_data(cities, invalid_routes, destinations)
        assert False, "validate_data should reject a duplicate route"
    except ValueError:
        pass


def test_full_data_features() -> None:
    """Test raw-data sizes and the published Part 2 feature invariants."""
    cities, routes, destinations, features = _load_full_features()
    assert cities.shape == (47, 3)
    assert routes.shape == (90, 6)
    assert destinations.shape == (46, 3)
    assert features.shape == (46, 20)
    assert int(features.isna().sum().sum()) == 0

    graph = build_graph(routes)
    assert len(graph) == 47
    assert len(edges_in_feature_table(features)) == 69
    assert (features["shortest_trains"] > 0).all()
    assert features["tunnel_fraction"].between(0, 1).all()
    assert features["ferry_fraction"].between(0, 1).all()

    equal_path_counts = features["equal_shortest_paths"].value_counts()
    assert equal_path_counts.to_dict() == {1: 35, 2: 10, 3: 1}
    assert int((features["Points"] == features["shortest_trains"]).sum()) == 39
    assert math.isclose(
        features["Points"].corr(features["shortest_trains"]),
        0.9952670603978323,
        rel_tol=1e-12,
    )
    assert math.isclose(
        features["straight_line_km"].mean(),
        1329.3789414359333,
        rel_tol=1e-12,
    )
    assert math.isclose(
        features["max_detour_trains"].mean(),
        3.3043478260869565,
        rel_tol=1e-12,
    )
    assert math.isclose(
        features["high_value_edge_overlap"].mean(),
        1.0336438923395446,
        rel_tol=1e-12,
    )

    cut_tickets = set(
        features.loc[
            features["has_cut_edge"], ["Source", "Target"]
        ].itertuples(index=False, name=None)
    )
    assert cut_tickets == {
        ("Edinburgh", "Paris"),
        ("Edinburgh", "Athina"),
    }
    assert important_edges(features, count=2) == [
        ("Berlin", "Frankfurt"),
        ("Frankfurt", "Munchen"),
    ]


def test_feature_examples() -> None:
    """Test hand-checkable efficient, fragile, and overlapping tickets."""
    unused_cities, unused_routes, unused_destinations, features = (
        _load_full_features()
    )
    indexed = features.set_index(["Source", "Target"])

    best = indexed.loc[("Bruxelles", "Danzic")]
    assert best["shortest_trains"] == 8
    assert best["points_per_train"] == 1.125

    worst = indexed.loc[("Palermo", "Constantinople")]
    assert worst["shortest_trains"] == 10
    assert worst["points_per_train"] == 0.8

    fragile = indexed.loc[("Frankfurt", "Khobenhaven")]
    assert fragile["max_detour_trains"] == 19

    shared = indexed.loc[("London", "Berlin")]
    assert shared["high_value_edge_overlap"] == 3


def test_model_leakage_safeguards() -> None:
    """Test that predictors omit every feature derived from ticket points."""
    unused_cities, unused_routes, unused_destinations, features = (
        _load_full_features()
    )
    prohibited = {"Points", "points_per_train", "high_value_edge_overlap"}
    assert prohibited.isdisjoint(MODEL_FEATURES)

    predictors, response = model_matrix(features)
    changed = features.copy()
    changed["Points"] = changed["Points"] + 100
    changed["points_per_train"] = changed["points_per_train"] + 100
    changed["high_value_edge_overlap"] = (
        changed["high_value_edge_overlap"] + 100
    )
    changed_predictors, changed_response = model_matrix(changed)

    pd.testing.assert_frame_equal(predictors, changed_predictors)
    assert (changed_response == response + 100).all()
    assert list(predictors.columns) == MODEL_FEATURES
    assert predictors["has_cut_edge"].dtype.kind in "iu"


def test_fast_nested_cross_validation() -> None:
    """Test nested CV outputs and complete out-of-fold prediction coverage."""
    unused_cities, unused_routes, unused_destinations, features = (
        _load_full_features()
    )
    metrics, predictions, parameters, importance = evaluate_models(
        features,
        random_seed=163,
        outer_splits=3,
        outer_repeats=1,
        ridge_alphas=(0.1, 1.0),
        forest_trees=(10,),
        forest_depths=(3,),
        forest_leaf_sizes=(1, 2),
    )

    assert set(metrics["model"]) == set(MODEL_NAMES)
    assert len(parameters) == 6
    assert set(importance["model"]) == {"Ridge", "Random forest"}
    assert set(importance["feature"]) == set(MODEL_FEATURES)
    assert len(importance) == 2 * len(MODEL_FEATURES)
    assert int(importance.isna().sum().sum()) == 0
    assert len(predictions) == len(features)
    prediction_columns = [
        "mean_baseline_prediction",
        "ridge_prediction",
        "random_forest_prediction",
    ]
    assert int(predictions[prediction_columns].isna().sum().sum()) == 0
    assert np.isfinite(predictions[prediction_columns].to_numpy()).all()
    assert np.isfinite(
        metrics[["pooled_oof_mae", "pooled_oof_r2"]].to_numpy()
    ).all()
    assert best_prediction_column(metrics) in predictions.columns
    ridge_mae = metrics.loc[
        metrics["model"] == "Ridge", "pooled_oof_mae"
    ].iloc[0]
    baseline_mae = metrics.loc[
        metrics["model"] == "Mean baseline", "pooled_oof_mae"
    ].iloc[0]
    assert ridge_mae < baseline_mae


def test_optimizer_budget_and_reproducibility() -> None:
    """Test budget feasibility, connectivity, and deterministic selection."""
    unused_cities, routes, destinations, features = _load_full_features()
    values = features["Points"].to_numpy(dtype=float)
    first_plan = optimize_network(
        routes,
        destinations,
        values,
        budget=45,
        risk_weight=0.25,
    )
    second_plan = optimize_network(
        routes,
        destinations,
        values,
        budget=45,
        risk_weight=0.25,
    )
    assert first_plan == second_plan

    selected = list(first_plan["selected_indices"])
    built_edges = set(first_plan["built_edges"])
    route_costs = {
        canonical_edge(row.Source, row.Target): int(row.Carriages)
        for row in routes.itertuples(index=False)
    }
    assert len(selected) == len(set(selected))
    assert all(0 <= index < len(destinations) for index in selected)
    assert sum(route_costs[edge] for edge in built_edges) == (
        first_plan["trains_used"]
    )
    assert 0 < first_plan["trains_used"] <= 45

    for index in selected:
        ticket = destinations.iloc[index]
        assert _endpoints_connected(
            built_edges,
            ticket["Source"],
            ticket["Target"],
        )


def test_matched_optimizer_scenarios() -> None:
    """Test matched fixed-value and learned-value optimizer constraints."""
    unused_cities, routes, destinations, features = _load_full_features()
    fixed_values = features["Points"].to_numpy(dtype=float)
    learned_values = fixed_values + np.linspace(-0.25, 0.25, len(features))
    fixed_plan = optimize_network(
        routes,
        destinations,
        fixed_values,
        budget=45,
        risk_weight=0.2,
    )
    learned_plan = optimize_network(
        routes,
        destinations,
        learned_values,
        budget=45,
        risk_weight=0.2,
    )

    assert fixed_plan["budget"] == learned_plan["budget"] == 45
    assert fixed_plan["risk_weight"] == learned_plan["risk_weight"] == 0.2
    assert fixed_plan["blocked_edges"] == learned_plan["blocked_edges"]
    for plan in [fixed_plan, learned_plan]:
        metrics = evaluate_plan(plan, routes, destinations)
        selected = list(plan["selected_indices"])
        expected_ticket_points = float(
            destinations.iloc[selected]["Points"].sum()
        )
        assert metrics["printed_ticket_points"] == expected_ticket_points
        assert metrics["tickets_connected"] == len(selected)
        assert metrics["total_points"] == (
            metrics["printed_ticket_points"] + metrics["route_points"]
        )
        assert metrics["points_per_train"] > 0


def test_blocked_edge_robustness() -> None:
    """Test blocked-route avoidance and bounded post-hoc stress metrics."""
    unused_cities, routes, destinations, features = _load_full_features()
    values = features["Points"].to_numpy(dtype=float)
    stress_edges = important_edges(features, count=2)
    blocked = stress_edges[0]
    blocked_plan = optimize_network(
        routes,
        destinations,
        values,
        budget=45,
        risk_weight=0.25,
        blocked_edges={blocked},
    )
    assert blocked not in blocked_plan["built_edges"]
    assert blocked_plan["trains_used"] <= 45

    base_plan = optimize_network(
        routes,
        destinations,
        values,
        budget=45,
        risk_weight=0.25,
    )
    stressed = evaluate_plan(
        base_plan,
        routes,
        destinations,
        stress_edges=stress_edges,
    )
    assert 0 <= stressed["worst_blocked_retention"] <= 1
    assert 0 <= stressed["mean_blocked_retention"] <= 1
    assert stressed["worst_blocked_ticket_points"] <= (
        stressed["mean_blocked_ticket_points"]
    )
    assert stressed["mean_blocked_ticket_points"] <= (
        stressed["printed_ticket_points"]
    )

    comparison = blocked_edge_stress_test(
        routes,
        destinations,
        values,
        stress_edges,
        budget=45,
        risk_weight=0.25,
    )
    assert len(comparison) == len(stress_edges)
    assert comparison["post_hoc_retention"].between(0, 1).all()
    assert (comparison["adaptive_trains_used"] <= 45).all()
    assert (
        comparison["post_hoc_ticket_points"]
        <= comparison["base_ticket_points"]
    ).all()


def test_matched_simulation_reproducibility() -> None:
    """Test identical trial pools, bounded metrics, and seeded repetition."""
    unused_cities, routes, destinations, features = _load_full_features()
    out_of_fold_predictions = (
        features["Points"].to_numpy(dtype=float)
        + np.linspace(-0.5, 0.5, len(features))
    )
    stress_edges = important_edges(features, count=1)
    first = run_matched_simulations(
        routes,
        destinations,
        out_of_fold_predictions,
        stress_edges=stress_edges,
        trials=4,
        pool_size=10,
        budget=45,
        learned_risk_weight=0.2,
        random_seed=163,
    )
    second = run_matched_simulations(
        routes,
        destinations,
        out_of_fold_predictions,
        stress_edges=stress_edges,
        trials=4,
        pool_size=10,
        budget=45,
        learned_risk_weight=0.2,
        random_seed=163,
    )
    first_results, first_summary, first_plans = first
    second_results, second_summary, second_plans = second

    pd.testing.assert_frame_equal(first_results, second_results)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert first_plans == second_plans
    assert len(first_results) == 8
    assert set(first_results["method"]) == {
        "Original baseline",
        "Learned revised",
    }
    candidate_counts = first_results.groupby("trial")[
        "candidate_indices"
    ].nunique()
    assert (candidate_counts == 1).all()
    assert (first_results["trains_used"] <= 45).all()
    assert first_results["mean_blocked_retention"].between(0, 1).all()
    assert first_results["worst_blocked_retention"].between(0, 1).all()
    assert int(first_results.isna().sum().sum()) == 0
    assert int(first_summary.isna().sum().sum()) == 0

    for plan in first_plans.values():
        assert plan["trains_used"] <= 45


def main() -> None:
    """Run every project test."""
    test_dijkstra()
    test_haversine_distance()
    test_data_validation()
    test_full_data_features()
    test_feature_examples()
    test_model_leakage_safeguards()
    test_fast_nested_cross_validation()
    test_optimizer_budget_and_reproducibility()
    test_matched_optimizer_scenarios()
    test_blocked_edge_robustness()
    test_matched_simulation_reproducibility()
    print("All tests passed!")


if __name__ == "__main__":
    main()
