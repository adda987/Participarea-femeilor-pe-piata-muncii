"""Specificații pentru Gradient Boosting."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.ml.utils import ModelSpec, RANDOM_STATE


GB_GRID_REGRESSION = {
    "model__n_estimators": [50, 100, 200, 300, 500],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__max_depth": [1, 2, 3, 4],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__subsample": [0.6, 0.8, 1.0],
    "model__loss": ["squared_error", "huber"],
}

GB_GRID_CLASSIFICATION = {
    "model__n_estimators": [50, 100, 200, 300, 500],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__max_depth": [1, 2, 3, 4],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__subsample": [0.6, 0.8, 1.0],
}


def gradient_boosting_specs() -> list[ModelSpec]:
    """Returnează modelele Gradient Boosting."""

    return [
        ModelSpec(
            name="Gradient Boosting",
            family="Gradient Boosting",
            task="regression",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
            param_distributions=GB_GRID_REGRESSION,
            search_iter=18,
            scoring="neg_root_mean_squared_error",
            model_filename="gradient_boosting_regressor.joblib",
        ),
        ModelSpec(
            name="Gradient Boosting",
            family="Gradient Boosting",
            task="classification",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
            param_distributions=GB_GRID_CLASSIFICATION,
            search_iter=18,
            scoring="f1_macro",
            model_filename="gradient_boosting_classifier.joblib",
        ),
    ]

