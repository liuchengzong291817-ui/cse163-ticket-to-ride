"""
Chengzong Liu
CSE 163 A
This program reproduces every table and figure for the Ticket to Ride:
Europe final-project report.
"""

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_features import (build_feature_table, canonical_edge,
                           important_edges, load_data)
from modeling import best_prediction_column, evaluate_models
from optimizer import (blocked_edge_stress_test, learned_route_values,
                       run_matched_simulations)


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
FIGURE_DIRECTORY = PROJECT_DIRECTORY / "figures"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"
RANDOM_SEED = 163
TRIAL_COUNT = 250
POOL_SIZE = 12
TRAIN_BUDGET = 45
LEARNED_RISK_WEIGHT = 0.20


def _prepare_directories() -> None:
    """Create the figure and table output directories when needed."""
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _display_name(feature: str) -> str:
    """Return a readable plot label for a snake-case feature name."""
    labels = {
        "shortest_trains": "Minimum trains",
        "straight_line_km": "Straight-line distance",
        "equal_shortest_paths": "Equal shortest paths",
        "max_detour_trains": "Maximum detour",
        "has_cut_edge": "Cut-edge indicator",
        "mean_edge_length": "Mean edge length",
        "intermediate_cities": "Intermediate cities",
        "tunnel_fraction": "Tunnel fraction",
        "ferry_fraction": "Ferry fraction",
    }
    return labels.get(feature, feature.replace("_", " ").title())


def _serialize_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    """Return a CSV-friendly copy of the destination feature table."""
    result = features.copy()
    result["path_nodes"] = result["path_nodes"].apply(
        lambda path: " -> ".join(path)
    )
    result["path_edges"] = result["path_edges"].apply(
        lambda edges: " | ".join(
            f"{source}--{target}" for source, target in edges
        )
    )
    return result


def _serialize_optimizer_trials(results: pd.DataFrame) -> pd.DataFrame:
    """Return a CSV-friendly copy of matched optimizer trial results."""
    serialized = results.copy()
    serialized["candidate_indices"] = serialized[
        "candidate_indices"
    ].apply(lambda values: ",".join(str(value) for value in values))
    serialized["selected_global_indices"] = serialized[
        "selected_global_indices"
    ].apply(lambda values: ",".join(str(value) for value in values))
    serialized["built_edges"] = serialized["built_edges"].apply(
        lambda edges: " | ".join(
            f"{source}--{target}" for source, target in edges
        )
    )
    return serialized


def _dataset_summary(
        cities: pd.DataFrame,
        routes: pd.DataFrame,
        destinations: pd.DataFrame,
        features: pd.DataFrame) -> pd.DataFrame:
    """Return row, column, and missing-value counts for all project tables."""
    names = [
        "cities.csv",
        "routes.csv",
        "destinations.csv",
        "combined feature table",
    ]
    tables = [cities, routes, destinations, features]
    return pd.DataFrame({
        "dataset": names,
        "rows": [len(table) for table in tables],
        "columns": [len(table.columns) for table in tables],
        "missing_values": [
            int(table.isna().sum().sum()) for table in tables
        ],
    })


def _rq1_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Return seven-number summaries for the principal RQ1 variables."""
    columns = {
        "Points": "Printed points",
        "shortest_trains": "Minimum trains",
        "points_per_train": "Ticket points per train",
        "straight_line_km": "Straight-line kilometers",
        "equal_shortest_paths": "Equal shortest paths",
        "max_detour_trains": "Maximum detour trains",
        "high_value_edge_overlap": "High-value edge overlap",
    }
    records = []
    for column, label in columns.items():
        values = features[column]
        records.append({
            "variable": label,
            "mean": values.mean(),
            "standard_deviation": values.std(),
            "minimum": values.min(),
            "first_quartile": values.quantile(0.25),
            "median": values.median(),
            "third_quartile": values.quantile(0.75),
            "maximum": values.max(),
        })
    return pd.DataFrame(records)


def _save_figure(figure: plt.Figure, file_name: str) -> None:
    """Save a tightly cropped report figure at print resolution."""
    figure.savefig(
        FIGURE_DIRECTORY / file_name,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def plot_eda_relationships(features: pd.DataFrame) -> None:
    """Plot ticket value, minimum cost, efficiency, and detour fragility."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    palette = {False: "#2a6f97", True: "#d1495b"}
    sns.scatterplot(
        data=features,
        x="shortest_trains",
        y="Points",
        hue="has_cut_edge",
        palette=palette,
        s=65,
        edgecolor="white",
        linewidth=0.5,
        ax=axes[0],
    )
    minimum = min(features["shortest_trains"].min(),
                  features["Points"].min())
    maximum = max(features["shortest_trains"].max(),
                  features["Points"].max())
    axes[0].plot(
        [minimum, maximum],
        [minimum, maximum],
        color="#666666",
        linestyle="--",
        linewidth=1,
        label="Equal value and cost",
    )
    correlation = features["Points"].corr(features["shortest_trains"])
    axes[0].text(
        0.04,
        0.92,
        f"Pearson r = {correlation:.3f}",
        transform=axes[0].transAxes,
    )
    axes[0].set_title("Printed Value Closely Tracks Minimum Cost")
    axes[0].set_xlabel("Minimum trains on a shortest path")
    axes[0].set_ylabel("Printed destination points")
    legend = axes[0].get_legend()
    if legend is not None:
        legend.set_title("Contains cut edge")

    sns.scatterplot(
        data=features,
        x="max_detour_trains",
        y="points_per_train",
        hue="has_cut_edge",
        palette=palette,
        s=65,
        edgecolor="white",
        linewidth=0.5,
        legend=False,
        ax=axes[1],
    )
    axes[1].axhline(1, color="#666666", linestyle="--", linewidth=1)
    axes[1].set_title("Efficiency Varies Little as Detour Cost Changes")
    axes[1].set_xlabel("Largest finite detour after removing a path edge")
    axes[1].set_ylabel("Printed ticket points per minimum train")
    figure.suptitle(
        "Ticket Value, Resource Cost, and Route Fragility",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_figure(figure, "eda_relationships.png")


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    """Plot repeated-CV MAE and pooled out-of-fold R-squared by model."""
    ordered = metrics.sort_values("mean_fold_mae")
    colors = ["#3a7d44", "#457b9d", "#b56576"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(
        ordered["model"],
        ordered["mean_fold_mae"],
        yerr=ordered["std_fold_mae"],
        color=colors,
        capsize=4,
    )
    axes[0].set_title("Repeated Outer-Fold MAE")
    axes[0].set_ylabel("Mean absolute error (points)")
    axes[0].tick_params(axis="x", rotation=18)

    axes[1].bar(
        ordered["model"],
        ordered["pooled_oof_r2"],
        color=colors,
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_title("Averaged Out-of-Fold R-squared")
    axes[1].set_ylabel("R-squared")
    axes[1].tick_params(axis="x", rotation=18)
    figure.suptitle(
        "Leakage-Free Model Performance",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_figure(figure, "model_comparison.png")


def plot_oof_predictions(
        features: pd.DataFrame,
        predictions: pd.DataFrame,
        metrics: pd.DataFrame) -> None:
    """Plot held-out ticket predictions from the lowest-MAE model."""
    prediction_column = best_prediction_column(metrics)
    best_model = metrics.iloc[0]["model"]
    figure, axis = plt.subplots(figsize=(6.5, 5.4))
    axis.scatter(
        features["Points"],
        predictions[prediction_column],
        s=55,
        color="#2a6f97",
        edgecolor="white",
        linewidth=0.5,
    )
    minimum = features["Points"].min() - 1
    maximum = features["Points"].max() + 1
    axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        color="#555555",
        linewidth=1,
    )
    residuals = (
        predictions[prediction_column] - features["Points"]
    ).abs()
    for index in residuals.nlargest(3).index:
        ticket = features.loc[index]
        axis.annotate(
            f"{ticket['Source']}–{ticket['Target']}",
            (
                ticket["Points"],
                predictions.loc[index, prediction_column],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlim(minimum, maximum)
    axis.set_ylim(minimum, maximum)
    axis.set_xlabel("Printed destination points")
    axis.set_ylabel("Averaged out-of-fold prediction")
    axis.set_title(f"{best_model}: Out-of-Fold Prediction by Ticket")
    figure.tight_layout()
    _save_figure(figure, "oof_predictions.png")


def plot_feature_importance(importance: pd.DataFrame) -> None:
    """Plot outer-fold Ridge coefficients and forest permutation importance."""
    ridge = importance[importance["model"] == "Ridge"].copy()
    forest = importance[
        importance["model"] == "Random forest"
    ].copy()
    ridge = ridge.sort_values("mean_importance")
    forest = forest.sort_values("mean_importance")
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.0))

    axes[0].barh(
        [_display_name(value) for value in ridge["feature"]],
        ridge["mean_importance"],
        xerr=ridge["std_importance"],
        color="#457b9d",
        capsize=2,
    )
    axes[0].axvline(0, color="#555555", linewidth=0.8)
    axes[0].set_title("Ridge Standardized Coefficients")
    axes[0].set_xlabel("Mean coefficient across outer folds")

    axes[1].barh(
        [_display_name(value) for value in forest["feature"]],
        forest["mean_importance"],
        xerr=forest["std_importance"],
        color="#d08c60",
        capsize=2,
    )
    axes[1].axvline(0, color="#555555", linewidth=0.8)
    axes[1].set_title("Random-Forest Held-Out Permutation Importance")
    axes[1].set_xlabel("Increase in held-out MAE after permutation")
    figure.suptitle(
        "Map Features That Drive Ticket-Point Predictions",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_figure(figure, "feature_importance.png")


def plot_optimizer_comparison(results: pd.DataFrame) -> None:
    """Plot matched optimizer efficiency and printed-point distributions."""
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    methods = ["Original baseline", "Learned revised"]
    colors = ["#457b9d", "#e07a5f"]
    efficiency_data = [
        results.loc[
            results["method"] == method,
            "total_points_per_train",
        ]
        for method in methods
    ]
    efficiency_boxes = axes[0].boxplot(
        efficiency_data,
        tick_labels=methods,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        orientation="vertical",
    )
    for box, color in zip(efficiency_boxes["boxes"], colors):
        box.set_facecolor(color)
    axes[0].set_title("Total Points per Train")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Total points / trains used")
    axes[0].tick_params(axis="x", rotation=12)

    points_data = [
        results.loc[
            results["method"] == method,
            "printed_ticket_points",
        ]
        for method in methods
    ]
    point_boxes = axes[1].boxplot(
        points_data,
        tick_labels=methods,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        orientation="vertical",
    )
    for box, color in zip(point_boxes["boxes"], colors):
        box.set_facecolor(color)
    axes[1].set_title("Printed Ticket Points Connected")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Printed ticket points")
    axes[1].tick_params(axis="x", rotation=12)
    figure.suptitle(
        "Original and Learned Optimizers on 250 Matched Pools",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_figure(figure, "optimizer_comparison.png")


def plot_optimizer_robustness(robustness: pd.DataFrame) -> None:
    """Plot post-hoc retention and adaptive total points by blocked edge."""
    data = robustness.copy()
    data["edge"] = (
        data["blocked_source"] + "–" + data["blocked_target"]
    )
    palette = {
        "Original baseline": "#457b9d",
        "Learned revised": "#e07a5f",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    sns.barplot(
        data=data,
        x="edge",
        y="post_hoc_retention",
        hue="method",
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_title("Post-Hoc Ticket-Point Retention")
    axes[0].set_xlabel("Blocked important edge")
    axes[0].set_ylabel("Fraction of ticket points retained")
    axes[0].tick_params(axis="x", rotation=32)
    axes[0].set_ylim(0, 1.05)

    sns.barplot(
        data=data,
        x="edge",
        y="adaptive_total_points",
        hue="method",
        palette=palette,
        ax=axes[1],
    )
    axes[1].set_title("Adaptive Re-Optimization")
    axes[1].set_xlabel("Blocked important edge")
    axes[1].set_ylabel("Total points after replanning")
    axes[1].tick_params(axis="x", rotation=32)
    legend = axes[1].get_legend()
    if legend is not None:
        legend.remove()
    axes[0].legend(title="Method", loc="lower left", fontsize=8)
    figure.suptitle(
        "Robustness to Five Frequently Used Edges",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_figure(figure, "optimizer_robustness.png")


def _draw_network(
        axis: plt.Axes,
        cities: pd.DataFrame,
        routes: pd.DataFrame,
        selected_edges: Sequence[Tuple[str, str]],
        title: str) -> None:
    """Draw one selected route network over the complete Europe graph."""
    coordinates = cities.set_index("City")[[
        "Longitude", "Latitude"
    ]].to_dict("index")
    selected = set(selected_edges)
    for route in routes.itertuples(index=False):
        edge = canonical_edge(route.Source, route.Target)
        source = coordinates[route.Source]
        target = coordinates[route.Target]
        is_selected = edge in selected
        axis.plot(
            [source["Longitude"], target["Longitude"]],
            [source["Latitude"], target["Latitude"]],
            color="#d1495b" if is_selected else "#c8c8c8",
            linewidth=2.3 if is_selected else 0.55,
            alpha=0.95 if is_selected else 0.55,
            zorder=2 if is_selected else 1,
        )
    axis.scatter(
        cities["Longitude"],
        cities["Latitude"],
        s=10,
        color="#333333",
        zorder=3,
    )
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_aspect("equal", adjustable="datalim")


def plot_selected_networks(
        cities: pd.DataFrame,
        routes: pd.DataFrame,
        plans: Dict[str, Dict[str, object]]) -> None:
    """Plot full-deck networks selected by the two optimizer objectives."""
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for axis, method in zip(axes, [
            "Original baseline", "Learned revised"]):
        plan = plans[method]
        title = (
            f"{method}\n{plan['trains_used']} trains, "
            f"{len(plan['selected_indices'])} tickets"
        )
        _draw_network(
            axis,
            cities,
            routes,
            plan["built_edges"],
            title,
        )
    figure.suptitle(
        "Illustrative Full-Deck Route Networks",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save_figure(figure, "selected_networks.png")


def _blocked_edge_results(
        routes: pd.DataFrame,
        destinations: pd.DataFrame,
        predictions: Sequence[float],
        stress_edges: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    """Return full-deck blocked-edge results for both optimizer methods."""
    baseline_values = destinations["Points"].to_numpy(dtype=float)
    learned_values = learned_route_values(
        routes,
        destinations,
        predictions,
    )
    method_inputs = {
        "Original baseline": (baseline_values, 0.0),
        "Learned revised": (learned_values, LEARNED_RISK_WEIGHT),
    }
    tables = []
    for method, (values, risk_weight) in method_inputs.items():
        table = blocked_edge_stress_test(
            routes,
            destinations,
            values,
            stress_edges,
            budget=TRAIN_BUDGET,
            risk_weight=risk_weight,
        )
        table.insert(0, "method", method)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def run_analysis() -> None:
    """Run the complete reproducible feature, model, and optimizer analysis."""
    _prepare_directories()
    sns.set_theme(style="whitegrid", context="notebook")
    cities, routes, destinations = load_data(str(DATA_DIRECTORY))
    features = build_feature_table(cities, routes, destinations)

    metrics, predictions, parameters, importance = evaluate_models(
        features,
        random_seed=RANDOM_SEED,
        outer_splits=5,
        outer_repeats=5,
        ridge_alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
        forest_trees=(100, 300),
        forest_depths=(None, 3, 5),
        forest_leaf_sizes=(1, 2, 4),
    )
    prediction_column = best_prediction_column(metrics)
    stress_edges = important_edges(features, count=5)
    optimizer_results, optimizer_summary, plans = (
        run_matched_simulations(
            routes,
            destinations,
            predictions[prediction_column],
            stress_edges=stress_edges,
            trials=TRIAL_COUNT,
            pool_size=POOL_SIZE,
            budget=TRAIN_BUDGET,
            learned_risk_weight=LEARNED_RISK_WEIGHT,
            random_seed=RANDOM_SEED,
        )
    )
    robustness = _blocked_edge_results(
        routes,
        destinations,
        predictions[prediction_column],
        stress_edges,
    )

    _serialize_feature_table(features).to_csv(
        OUTPUT_DIRECTORY / "feature_table.csv", index=False
    )
    _dataset_summary(cities, routes, destinations, features).to_csv(
        OUTPUT_DIRECTORY / "dataset_summary.csv", index=False
    )
    _rq1_summary(features).to_csv(
        OUTPUT_DIRECTORY / "rq1_summary.csv", index=False
    )
    metrics.to_csv(OUTPUT_DIRECTORY / "model_metrics.csv", index=False)
    predictions.to_csv(
        OUTPUT_DIRECTORY / "oof_predictions.csv", index=False
    )
    parameters.to_csv(
        OUTPUT_DIRECTORY / "selected_hyperparameters.csv", index=False
    )
    importance.to_csv(
        OUTPUT_DIRECTORY / "feature_importance.csv", index=False
    )
    _serialize_optimizer_trials(optimizer_results).to_csv(
        OUTPUT_DIRECTORY / "optimizer_trials.csv", index=False
    )
    optimizer_summary.to_csv(
        OUTPUT_DIRECTORY / "optimizer_summary.csv", index=False
    )
    robustness.to_csv(
        OUTPUT_DIRECTORY / "blocked_edge_robustness.csv", index=False
    )

    plot_eda_relationships(features)
    plot_model_comparison(metrics)
    plot_oof_predictions(features, predictions, metrics)
    plot_feature_importance(importance)
    plot_optimizer_comparison(optimizer_results)
    plot_optimizer_robustness(robustness)
    plot_selected_networks(cities, routes, plans)
    print("Analysis complete: tables are in outputs/ and figures are in "
          "figures/.")


def main() -> None:
    """Run the final-project analysis from the included raw CSV files."""
    run_analysis()


if __name__ == "__main__":
    main()
