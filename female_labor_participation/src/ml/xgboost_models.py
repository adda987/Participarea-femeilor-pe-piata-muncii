"""Specificații pentru XGBoost."""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.ml.utils import ModelSpec, RANDOM_STATE


XGB_GRID = {
    "model__n_estimators": [100, 200, 400, 600],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__max_depth": [2, 3, 4, 5],
    "model__min_child_weight": [1, 3, 5],
    "model__subsample": [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__reg_alpha": [0, 0.01, 0.1, 1],
    "model__reg_lambda": [0.1, 1, 10],
}


def xgboost_specs() -> list[ModelSpec]:
    """Returnează modelele XGBoost."""

    return [
        ModelSpec(
            name="XGBoost",
            family="XGBoost",
            task="regression",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
                ]
            ),
            param_distributions=XGB_GRID,
            search_iter=18,
            scoring="neg_root_mean_squared_error",
            model_filename="xgb_regressor.joblib",
        ),
        ModelSpec(
            name="XGBoost",
            family="XGBoost",
            task="classification",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            objective="multi:softprob",
                            num_class=3,
                            eval_metric="mlogloss",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            verbosity=0,
                        ),
                    ),
                ]
            ),
            param_distributions=XGB_GRID,
            search_iter=18,
            scoring="f1_macro",
            model_filename="xgb_classifier.joblib",
        ),
    ]

