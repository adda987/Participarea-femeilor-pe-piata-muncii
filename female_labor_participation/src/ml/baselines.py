"""Modele baseline pentru comparația Machine Learning."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.utils import ModelSpec, RANDOM_STATE


def regression_baselines() -> list[ModelSpec]:
    """Returnează baseline-urile de regresie recalibrate pe experimentul ML."""

    return [
        ModelSpec(
            name="Dummy Mean",
            family="Baseline",
            task="regression",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="mean"))]),
            scoring="neg_root_mean_squared_error",
            model_filename="dummy_mean_regressor.joblib",
        ),
        ModelSpec(
            name="Linear Regression",
            family="Baseline",
            task="regression",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LinearRegression())]),
            scoring="neg_root_mean_squared_error",
            model_filename="linear_regression.joblib",
        ),
        ModelSpec(
            name="Ridge",
            family="Baseline",
            task="regression",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(random_state=RANDOM_STATE))]),
            param_distributions={"model__alpha": [0.01, 0.1, 1, 10, 100]},
            search_iter=5,
            scoring="neg_root_mean_squared_error",
            model_filename="ridge_regression.joblib",
        ),
        ModelSpec(
            name="LASSO",
            family="Baseline",
            task="regression",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Lasso(random_state=RANDOM_STATE, max_iter=20000))]),
            param_distributions={"model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1]},
            search_iter=6,
            scoring="neg_root_mean_squared_error",
            model_filename="lasso_regression.joblib",
        ),
    ]


def classification_baselines() -> list[ModelSpec]:
    """Returnează baseline-urile de clasificare recalibrate pe experimentul ML."""

    return [
        ModelSpec(
            name="Dummy Classifier",
            family="Baseline",
            task="classification",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent"))]),
            scoring="f1_macro",
            model_filename="dummy_classifier.joblib",
        ),
        ModelSpec(
            name="Logistic Regression",
            family="Baseline",
            task="classification",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
            param_distributions={"model__C": [0.1, 1, 10, 50]},
            search_iter=4,
            scoring="f1_macro",
            model_filename="logistic_regression.joblib",
        ),
    ]
