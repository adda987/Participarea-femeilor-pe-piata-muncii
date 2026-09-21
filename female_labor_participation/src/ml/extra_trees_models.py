"""Specificații pentru Extra Trees."""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.ml.utils import ModelSpec, RANDOM_STATE


ET_GRID = {
    "model__n_estimators": [200, 400, 600, 800],
    "model__max_depth": [None, 3, 5, 8, 12],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", "log2", 0.5, 1.0],
}


def extra_trees_specs() -> list[ModelSpec]:
    """Returnează modelele Extra Trees."""

    return [
        ModelSpec(
            name="Extra Trees",
            family="Extra Trees",
            task="regression",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1))]),
            param_distributions=ET_GRID,
            search_iter=18,
            scoring="neg_root_mean_squared_error",
            model_filename="extra_trees_regressor.joblib",
        ),
        ModelSpec(
            name="Extra Trees",
            family="Extra Trees",
            task="classification",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"))]),
            param_distributions={**ET_GRID, "model__class_weight": ["balanced", None]},
            search_iter=18,
            scoring="f1_macro",
            model_filename="extra_trees_classifier.joblib",
        ),
    ]
