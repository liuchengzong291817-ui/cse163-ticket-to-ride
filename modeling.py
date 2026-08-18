"""
Chengzong Liu
CSE 163 A
This module evaluates models that predict Ticket to Ride destination points
from map-derived features without using target-derived predictors.
"""

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_FEATURES = [
    "shortest_trains",
    "straight_line_km",
    "equal_shortest_paths",
    "max_detour_trains",
    "has_cut_edge",
    "mean_edge_length",
    "intermediate_cities",
    "tunnel_fraction",
    "ferry_fraction",
]
MODEL_NAMES = ["Mean baseline", "Ridge", "Random forest"]


def model_matrix(features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return the target-independent predictor matrix and printed-point response
    from a destination feature table.
    """
    predictors = features[MODEL_FEATURES].copy()
    predictors["has_cut_edge"] = predictors["has_cut_edge"].astype(int)
    response = features["Points"].astype(float)
    return predictors, response


def _ridge_estimator() -> Pipeline:
    """Return a standardized Ridge-regression pipeline."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge()),
    ])


def _forest_estimator(random_seed: int) -> RandomForestRegressor:
    """Return a reproducible random-forest regressor."""
    return RandomForestRegressor(
        random_state=random_seed,
        n_jobs=1,
    )


def evaluate_models(
        features: pd.DataFrame,
        random_seed: int = 163,
        outer_splits: int = 5,
        outer_repeats: int = 5,
        ridge_alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
        forest_trees: Sequence[int] = (200,),
        forest_depths: Sequence[object] = (None, 3, 5),
        forest_leaf_sizes: Sequence[int] = (1, 2, 4),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a mean baseline, Ridge, and random forest with repeated nested
    cross-validation. Return metric summaries, averaged out-of-fold
    predictions, selected hyperparameters, and held-out feature importance.
    """
    predictors, response = model_matrix(features)
    splitter = RepeatedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=random_seed,
    )
    prediction_sums = {
        model_name: np.zeros(len(features), dtype=float)
        for model_name in MODEL_NAMES
    }
    prediction_counts = np.zeros(len(features), dtype=int)
    fold_records = []
    parameter_records = []
    importance_records = []

    for fold_number, (train_indices, test_indices) in enumerate(
            splitter.split(predictors), start=1):
        train_x = predictors.iloc[train_indices]
        train_y = response.iloc[train_indices]
        test_x = predictors.iloc[test_indices]
        test_y = response.iloc[test_indices]
        inner_folds = min(4, len(train_indices))
        inner_splitter = KFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=random_seed + fold_number,
        )

        baseline = DummyRegressor(strategy="mean")
        ridge_search = GridSearchCV(
            _ridge_estimator(),
            {"ridge__alpha": list(ridge_alphas)},
            scoring="neg_mean_absolute_error",
            cv=inner_splitter,
        )
        forest_search = GridSearchCV(
            _forest_estimator(random_seed + fold_number),
            {
                "n_estimators": list(forest_trees),
                "max_depth": list(forest_depths),
                "min_samples_leaf": list(forest_leaf_sizes),
            },
            scoring="neg_mean_absolute_error",
            cv=inner_splitter,
            n_jobs=1,
        )
        estimators = {
            "Mean baseline": baseline,
            "Ridge": ridge_search,
            "Random forest": forest_search,
        }

        for model_name, estimator in estimators.items():
            estimator.fit(train_x, train_y)
            predictions = estimator.predict(test_x)
            prediction_sums[model_name][test_indices] += predictions
            fold_records.append({
                "fold": fold_number,
                "model": model_name,
                "mae": mean_absolute_error(test_y, predictions),
                "r2": r2_score(test_y, predictions),
            })

        ridge_coefficients = (
            ridge_search.best_estimator_.named_steps["ridge"].coef_
        )
        for feature, coefficient in zip(
                MODEL_FEATURES, ridge_coefficients):
            importance_records.append({
                "fold": fold_number,
                "model": "Ridge",
                "feature": feature,
                "importance": coefficient,
                "measure": "standardized coefficient",
            })

        held_out_importance = permutation_importance(
            forest_search.best_estimator_,
            test_x,
            test_y,
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=random_seed + fold_number,
            n_jobs=1,
        )
        for feature, importance in zip(
                MODEL_FEATURES, held_out_importance.importances_mean):
            importance_records.append({
                "fold": fold_number,
                "model": "Random forest",
                "feature": feature,
                "importance": importance,
                "measure": "held-out MAE increase",
            })

        prediction_counts[test_indices] += 1
        parameter_records.append({
            "fold": fold_number,
            "model": "Ridge",
            **ridge_search.best_params_,
        })
        parameter_records.append({
            "fold": fold_number,
            "model": "Random forest",
            **forest_search.best_params_,
        })

    fold_metrics = pd.DataFrame(fold_records)
    prediction_table = features[["Source", "Target", "Points"]].copy()
    for model_name in MODEL_NAMES:
        column_name = model_name.lower().replace(" ", "_") + "_prediction"
        prediction_table[column_name] = (
            prediction_sums[model_name] / prediction_counts
        )

    metric_records = []
    for model_name in MODEL_NAMES:
        model_folds = fold_metrics[fold_metrics["model"] == model_name]
        prediction_column = (
            model_name.lower().replace(" ", "_") + "_prediction"
        )
        averaged_predictions = prediction_table[prediction_column]
        metric_records.append({
            "model": model_name,
            "mean_fold_mae": model_folds["mae"].mean(),
            "std_fold_mae": model_folds["mae"].std(),
            "mean_fold_r2": model_folds["r2"].mean(),
            "std_fold_r2": model_folds["r2"].std(),
            "pooled_oof_mae": mean_absolute_error(
                response,
                averaged_predictions,
            ),
            "pooled_oof_r2": r2_score(response, averaged_predictions),
        })

    metrics = pd.DataFrame(metric_records).sort_values("pooled_oof_mae")
    parameters = pd.DataFrame(parameter_records)
    importance_by_fold = pd.DataFrame(importance_records)
    importance = (
        importance_by_fold
        .groupby(["model", "feature", "measure"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
        )
    )
    return (
        metrics.reset_index(drop=True),
        prediction_table,
        parameters,
        importance,
    )


def best_prediction_column(metrics: pd.DataFrame) -> str:
    """Return the out-of-fold prediction column for the lowest-MAE model."""
    best_model = metrics.iloc[0]["model"]
    return best_model.lower().replace(" ", "_") + "_prediction"
