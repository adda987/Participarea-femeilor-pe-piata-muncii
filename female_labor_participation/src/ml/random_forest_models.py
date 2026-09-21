"""Specificații pentru Random Forest."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.ml.utils import ModelSpec, RANDOM_STATE


RF_GRID = {
    "model__n_estimators": [200, 400, 600, 800],
    "model__max_depth": [None, 3, 5, 8, 12],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", "log2", 0.5, 1.0],
}


def random_forest_specs() -> list[ModelSpec]:
    """Returnează modelele Random Forest."""

    return [
        ModelSpec(
            name="Random Forest",
            family="Random Forest",
            task="regression",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))]),
            param_distributions=RF_GRID,
            search_iter=18,
            scoring="neg_root_mean_squared_error",
            model_filename="random_forest_regressor.joblib",
        ),
        ModelSpec(
            name="Random Forest",
            family="Random Forest",
            task="classification",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"))]),
            param_distributions={**RF_GRID, "model__class_weight": ["balanced", None]},
            search_iter=18,
            scoring="f1_macro",
            model_filename="random_forest_classifier.joblib",
        ),
    ]

