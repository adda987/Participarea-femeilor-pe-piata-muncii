"""Specificații pentru modelele Support Vector Machines."""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from src.ml.utils import ModelSpec, RANDOM_STATE


def svm_specs() -> list[ModelSpec]:
    """Returnează SVR și SVC cu grilele cerute."""

    return [
        ModelSpec(
            name="SVR",
            family="Support Vector Machines",
            task="regression",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVR())]),
            param_distributions={
                "model__C": [0.1, 1, 10, 50, 100],
                "model__epsilon": [0.01, 0.05, 0.1, 0.25, 0.5],
                "model__kernel": ["linear", "rbf", "poly"],
                "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "model__degree": [2, 3],
            },
            search_iter=24,
            scoring="neg_root_mean_squared_error",
            model_filename="svr.joblib",
        ),
        ModelSpec(
            name="SVC",
            family="Support Vector Machines",
            task="classification",
            estimator=Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVC(probability=True, random_state=RANDOM_STATE))]),
            param_distributions={
                "model__C": [0.1, 1, 10, 50, 100],
                "model__kernel": ["linear", "rbf", "poly"],
                "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "model__degree": [2, 3],
            },
            search_iter=24,
            scoring="f1_macro",
            model_filename="svc.joblib",
        ),
    ]

