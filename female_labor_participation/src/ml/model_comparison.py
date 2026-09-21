"""Orchestratorul complet pentru pagina Machine Learning."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from itertools import product
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, f1_score, mean_squared_error, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV

from src.ml.baselines import classification_baselines, regression_baselines
from src.ml.classification_target import CLASS_LABELS, CODE_TO_CLASS, build_classification_target
from src.ml.data_preparation import FEATURE_LABELS, MLExperimentData, load_ml_dataset
from src.ml.extra_trees_models import extra_trees_specs
from src.ml.gradient_boosting_models import gradient_boosting_specs
from src.ml.interpretation import (
    TREE_FAMILIES,
    compute_permutation_importance,
    consensus_table,
    generate_partial_dependence,
    generate_shap_classification,
    generate_shap_regression,
    native_feature_importance,
)
from src.ml.metrics import classification_metric_row, metric_frame, regression_metric_row
from src.ml.plots import (
    absolute_error_distribution,
    actual_vs_predicted,
    brier_scores,
    class_distribution,
    complexity_performance,
    confusion_heatmap,
    correct_incorrect_confidence,
    cv_comparison,
    dataset_shift,
    error_distribution,
    generalization_map,
    importance_bar,
    importance_heatmap,
    metric_bar,
    performance_by_class,
    performance_by_country,
    performance_by_year,
    permutation_bar,
    predictions_over_time,
    pr_multiclass,
    probability_distribution,
    residual_plot,
    shock_errors,
    simple_line,
    small_multiple_actual_predicted,
    target_distribution,
    timeline_train_test,
    train_vs_test_rmse,
    roc_multiclass,
)
from src.ml.random_forest_models import random_forest_specs
from src.ml.svm_models import svm_specs
from src.ml.utils import (
    BURGUNDY,
    GOLD,
    MLOutputPaths,
    ModelSpec,
    RANDOM_STATE,
    TEAL,
    apply_project_layout,
    ensure_output_dirs,
    ml_paths,
    save_figure,
    save_metadata,
    save_model,
    save_table,
)
from src.ml.validation import build_validation_design, evaluate_group_cv, evaluate_temporal_cv
from src.ml.xgboost_models import xgboost_specs


warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._classification")
warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation")

REGRESSION_MODELS = ["SVR", "Random Forest", "Gradient Boosting", "XGBoost", "Extra Trees"]
CLASSIFICATION_MODELS = ["SVC", "Random Forest", "Gradient Boosting", "XGBoost", "Extra Trees"]


def _x_y(bundle: MLExperimentData) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Returnează X/y train-test pentru regresie."""

    x_train = bundle.train[bundle.feature_columns]
    y_train = bundle.train["female_lfpr"]
    x_test = bundle.test[bundle.feature_columns]
    y_test = bundle.test["female_lfpr"]
    return x_train, y_train, x_test, y_test


def _param_grid_size(grid: dict[str, list[Any]] | None) -> int:
    """Numără combinațiile posibile ale unui grid."""

    if not grid:
        return 0
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _fit_spec(spec: ModelSpec, x_train, y_train, temporal_cv) -> tuple[Any, dict[str, Any], float, float]:
    """Antrenează un model, cu tuning temporal dacă are grid."""

    start = time.perf_counter()
    if spec.param_distributions and spec.search_iter > 0:
        n_iter = min(spec.search_iter, _param_grid_size(spec.param_distributions))
        search = RandomizedSearchCV(
            estimator=clone(spec.estimator),
            param_distributions=spec.param_distributions,
            n_iter=n_iter,
            scoring=spec.scoring,
            cv=temporal_cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            error_score=np.nan,
        )
        with parallel_backend("threading"):
            search.fit(x_train, y_train)
        estimator = search.best_estimator_
        best_params = search.best_params_
        cv_score = float(search.best_score_)
        if spec.scoring.startswith("neg_"):
            cv_score = -cv_score
    else:
        estimator = clone(spec.estimator)
        with parallel_backend("threading"):
            estimator.fit(x_train, y_train)
        best_params = {}
        cv_score = evaluate_temporal_cv(estimator, x_train, y_train, temporal_cv, spec.scoring)
    fit_time = time.perf_counter() - start
    return estimator, best_params, fit_time, cv_score


def _model_stem(model_name: str) -> str:
    """Transformă numele modelului în stem pentru fișiere."""

    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("ă", "a")
        .replace("â", "a")
        .replace("î", "i")
        .replace("ș", "s")
        .replace("ț", "t")
    )


def _save_xgb_native(estimator, filename: str, paths: MLOutputPaths) -> None:
    """Salvează modelul XGBoost în format nativ când există."""

    model = estimator.named_steps.get("model") if hasattr(estimator, "named_steps") else estimator
    if hasattr(model, "save_model"):
        model.save_model(paths.models / filename)


def _predict_proba_aligned(estimator, x_test, labels: list[int]) -> np.ndarray:
    """Returnează probabilități aliniate la ordinea claselor 0,1,2."""

    proba = estimator.predict_proba(x_test)
    classes = getattr(estimator.named_steps.get("model"), "classes_", labels) if hasattr(estimator, "named_steps") else getattr(estimator, "classes_", labels)
    aligned = np.zeros((len(x_test), len(labels)))
    for index, label in enumerate(classes):
        if int(label) in labels:
            aligned[:, labels.index(int(label))] = proba[:, index]
    return aligned


def _hyperparameter_rows(task: str, model_name: str, params: dict[str, Any]) -> list[dict[str, str]]:
    """Pregătește hyperparametrii selectați pentru export."""

    if not params:
        return [{"Task": task, "Model": model_name, "Parametru": "Fără tuning", "Valoare": "Specificație implicită controlată"}]
    rows = []
    for key, value in params.items():
        rows.append({"Task": task, "Model": model_name, "Parametru": key.replace("model__", ""), "Valoare": str(value)})
    return rows


def _model_card_row(task: str, spec: ModelSpec, params: dict[str, Any], train_n: int, test_n: int, metric_name: str, metric_value: float, fit_time: float) -> dict[str, Any]:
    """Construiește cardul metodologic al unui model."""

    warning = "Importanța predictivă nu reprezintă automat efect economic cauzal."
    if task == "classification":
        warning = "Clasificarea este exploratorie, iar pragurile sunt tertile calculate exclusiv pe training."
    return {
        "Familie": spec.family,
        "Model": spec.name,
        "Task": "Regresie" if task == "regression" else "Clasificare",
        "Tip model": type(spec.estimator.named_steps["model"]).__name__ if hasattr(spec.estimator, "named_steps") else type(spec.estimator).__name__,
        "Hyperparametri selectați": "; ".join(f"{key.replace('model__', '')}={value}" for key, value in params.items()) if params else "Specificație implicită controlată",
        "Număr features": len(FEATURE_LABELS),
        "Train N": train_n,
        "Test N": test_n,
        "Metrică principală": metric_name,
        "Valoare metrică": metric_value,
        "Timp de antrenare sec": fit_time,
        "Avertisment": warning,
    }


def _dataset_shift(bundle: MLExperimentData) -> pd.DataFrame:
    """Calculează diferența standardizată între train și test pentru fiecare feature."""

    rows = []
    for feature in bundle.feature_columns:
        train_values = bundle.train[feature].dropna()
        test_values = bundle.test[feature].dropna()
        pooled_sd = pd.concat([train_values, test_values]).std()
        rows.append(
            {
                "Feature raw": feature,
                "Feature": FEATURE_LABELS.get(feature, feature),
                "Media train": train_values.mean(),
                "Media test": test_values.mean(),
                "Diferență standardizată": (test_values.mean() - train_values.mean()) / pooled_sd if pooled_sd and np.isfinite(pooled_sd) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _learning_curve(estimator, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, years: pd.Series, task: str, model_name: str) -> pd.DataFrame:
    """Construiește o learning curve temporală pe subseturi expanding-window."""

    rows = []
    candidate_years = [2008, 2010, 2012, 2014, 2016, 2018]
    for end_year in candidate_years:
        mask = years <= end_year
        if mask.sum() < 30:
            continue
        model = clone(estimator)
        with parallel_backend("threading"):
            model.fit(x_train.loc[mask], y_train.loc[mask])
        train_pred = model.predict(x_train.loc[mask])
        test_pred = model.predict(x_test)
        if task == "regression":
            train_metric = mean_squared_error(y_train.loc[mask], train_pred) ** 0.5
            test_metric = mean_squared_error(y_test, test_pred) ** 0.5
            metric = "RMSE"
        else:
            train_metric = f1_score(y_train.loc[mask], train_pred, average="macro", zero_division=0)
            test_metric = f1_score(y_test, test_pred, average="macro", zero_division=0)
            metric = "F1 Macro"
        rows.append({"Model": model_name, "Task": task, "Train până la": end_year, "Train": train_metric, "Test": test_metric, "Metrică": metric})
    return pd.DataFrame(rows)


def _validation_curve(estimator, x_train: pd.DataFrame, y_train: pd.Series, temporal_cv, task: str, model_name: str, param_name: str, values: Iterable[Any]) -> pd.DataFrame:
    """Evaluează câteva valori pentru un hyperparametru."""

    scoring = "neg_root_mean_squared_error" if task == "regression" else "f1_macro"
    rows = []
    for value in values:
        model = clone(estimator)
        try:
            model.set_params(**{param_name: value})
            score = evaluate_temporal_cv(model, x_train, y_train, temporal_cv, scoring)
        except Exception:
            score = np.nan
        rows.append({"Model": model_name, "Task": task, "Parametru": param_name.replace("model__", ""), "Valoare": str(value), "Scor": score})
    return pd.DataFrame(rows)


def _svm_heatmap(estimator, x_train, y_train, temporal_cv, task: str, filename: str) -> pd.DataFrame:
    """Heatmap C x gamma pentru kernel RBF."""

    c_values = [0.1, 1, 10, 50, 100]
    gamma_values = ["scale", "auto", 0.001, 0.01, 0.1]
    scoring = "neg_root_mean_squared_error" if task == "regression" else "f1_macro"
    rows = []
    for c_value, gamma in product(c_values, gamma_values):
        model = clone(estimator)
        model.set_params(model__kernel="rbf", model__C=c_value, model__gamma=gamma)
        try:
            score = evaluate_temporal_cv(model, x_train, y_train, temporal_cv, scoring)
        except Exception:
            score = np.nan
        rows.append({"C": c_value, "gamma": str(gamma), "Scor": score})
    data = pd.DataFrame(rows)
    matrix = data.pivot_table(index="gamma", columns="C", values="Scor")
    fig = px.imshow(matrix, text_auto=".3f", color_continuous_scale=[[0, BURGUNDY], [0.5, GOLD], [1, TEAL]], aspect="auto")
    title = "Heatmap C × gamma pentru kernel RBF"
    save_figure(apply_project_layout(fig, title), filename)
    return data


def _xgb_training_history(estimator, x_train, y_train, years: pd.Series, task: str, paths: MLOutputPaths, filename: str) -> pd.DataFrame:
    """Antrenează XGBoost pe development/validation intern fără a atinge testul."""

    if not hasattr(estimator, "named_steps"):
        return pd.DataFrame()
    dev_mask = years <= 2016
    val_mask = (years >= 2017) & (years <= 2018)
    if dev_mask.sum() < 30 or val_mask.sum() == 0:
        return pd.DataFrame()
    imputer = clone(estimator.named_steps["imputer"])
    model = clone(estimator.named_steps["model"])
    x_dev = imputer.fit_transform(x_train.loc[dev_mask])
    x_val = imputer.transform(x_train.loc[val_mask])
    y_dev = y_train.loc[dev_mask]
    y_val = y_train.loc[val_mask]
    try:
        model.set_params(early_stopping_rounds=20)
        model.fit(x_dev, y_dev, eval_set=[(x_dev, y_dev), (x_val, y_val)], verbose=False)
    except Exception:
        model = clone(estimator.named_steps["model"])
        model.fit(x_dev, y_dev, eval_set=[(x_dev, y_dev), (x_val, y_val)], verbose=False)
    history = model.evals_result()
    metric_key = "rmse" if task == "regression" else "mlogloss"
    train_values = history.get("validation_0", {}).get(metric_key, [])
    val_values = history.get("validation_1", {}).get(metric_key, [])
    data = pd.DataFrame({"Rundă": range(1, len(train_values) + 1), "Train": train_values, "Validation": val_values})
    if not data.empty:
        long_data = data.melt(id_vars="Rundă", var_name="Eșantion", value_name="Loss")
        fig = px.line(long_data, x="Rundă", y="Loss", color="Eșantion", color_discrete_map={"Train": TEAL, "Validation": BURGUNDY})
        save_figure(apply_project_layout(fig, "Train vs validation loss pentru XGBoost"), filename, paths)
    return data


def _class_metrics_by_model(y_true: pd.Series, predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculează F1/precision/recall pe clasă pentru fiecare model."""

    rows = []
    for model_name in predictions["Model"].unique():
        data = predictions.loc[predictions["Model"].eq(model_name)]
        y_pred = data["Predicție cod"].astype(int)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
        for code, label in CODE_TO_CLASS.items():
            rows.append(
                {
                    "Model": model_name,
                    "Clasă": label,
                    "Precision": precision[code],
                    "Recall": recall[code],
                    "F1": f1[code],
                    "Support": support[code],
                }
            )
    return pd.DataFrame(rows)


def _brier_scores(proba_frame: pd.DataFrame) -> pd.DataFrame:
    """Calculează Brier score one-vs-rest pe clasă."""

    rows = []
    for model_name in proba_frame["Model"].unique():
        data = proba_frame.loc[proba_frame["Model"].eq(model_name)]
        real_codes = data["Real cod"].astype(int)
        for code, label in CODE_TO_CLASS.items():
            rows.append(
                {
                    "Model": model_name,
                    "Clasă": label,
                    "Brier Score": brier_score_loss((real_codes == code).astype(int), data[label]),
                }
            )
    return pd.DataFrame(rows)


def _shock_error_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Agregă erorile pentru perioadele de șoc definite."""

    data = predictions.copy()
    data["Perioadă"] = np.select(
        [data["Year"].eq(2019), data["Year"].eq(2020), data["Year"].eq(2021), data["Year"].isin([2022, 2023])],
        ["2019", "2020", "2021", "2022–2023"],
        default="Altă perioadă",
    )
    return (
        data.groupby(["Model", "Perioadă"], as_index=False)
        .agg(RMSE=("Reziduu", lambda values: float(np.sqrt(np.mean(np.square(values))))), MAE=("Eroare absolută", "mean"))
        .sort_values(["Model", "Perioadă"])
    )


def _make_summary_tables(bundle: MLExperimentData, class_target, validation_design) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tabele introductive pentru experiment."""

    experiment_summary = pd.DataFrame(
        [
            {"Indicator": "Observații totale utilizabile", "Valoare": len(bundle.data), "Detaliu": "Target female_lfpr disponibil"},
            {"Indicator": "Train", "Valoare": len(bundle.train), "Detaliu": "2001–2018"},
            {"Indicator": "Test", "Valoare": len(bundle.test), "Detaliu": "2019–2023"},
            {"Indicator": "Număr features", "Valoare": len(bundle.feature_columns), "Detaliu": "Fără Year, Country sau ISO3"},
            {"Indicator": "Număr țări", "Valoare": bundle.data["ISO3"].nunique(), "Detaliu": "Economii europene individuale"},
            {"Indicator": "Perioadă", "Valoare": "2001–2023", "Detaliu": "Frecvență anuală"},
            {"Indicator": "Observații eliminate target lipsă", "Valoare": bundle.dropped_target_missing, "Detaliu": "Target-ul nu este imputat"},
        ]
    )
    class_distribution_table = pd.concat([class_target.distribution_train, class_target.distribution_test], ignore_index=True)
    temporal_cv_table = pd.DataFrame(validation_design.temporal_folds)
    return experiment_summary, class_distribution_table, temporal_cv_table


def run_ml_analysis() -> dict[str, Any]:
    """Rulează complet experimentul Machine Learning și salvează outputurile."""

    paths = ensure_output_dirs(ml_paths())
    for stale_figure in paths.figures.glob("*.json"):
        stale_figure.unlink()
    bundle = load_ml_dataset()
    x_train, y_train_reg, x_test, y_test_reg = _x_y(bundle)
    class_target = build_classification_target(y_train_reg, y_test_reg)
    y_train_cls = class_target.y_train_codes
    y_test_cls = class_target.y_test_codes
    validation_design = build_validation_design(bundle.train["Year"], bundle.train["ISO3"])

    experiment_summary, class_distribution_table, temporal_cv_table = _make_summary_tables(bundle, class_target, validation_design)
    shift = _dataset_shift(bundle)
    save_table(experiment_summary, "experiment_summary.csv", paths)
    save_table(class_distribution_table, "classification_distribution.csv", paths)
    save_table(temporal_cv_table, "temporal_cv_folds.csv", paths)
    save_table(shift, "dataset_shift.csv", paths)
    timeline_train_test("train_test_timeline.json")
    target_distribution(bundle.train, bundle.test, "target_distribution.json")
    class_distribution(class_distribution_table, "class_distribution.json")
    dataset_shift(shift, "dataset_shift.json")

    regression_specs = [*regression_baselines(), *(spec for spec in svm_specs() if spec.task == "regression"), *(spec for spec in random_forest_specs() if spec.task == "regression"), *(spec for spec in gradient_boosting_specs() if spec.task == "regression"), *(spec for spec in xgboost_specs() if spec.task == "regression"), *(spec for spec in extra_trees_specs() if spec.task == "regression")]
    classification_specs = [*classification_baselines(), *(spec for spec in svm_specs() if spec.task == "classification"), *(spec for spec in random_forest_specs() if spec.task == "classification"), *(spec for spec in gradient_boosting_specs() if spec.task == "classification"), *(spec for spec in xgboost_specs() if spec.task == "classification"), *(spec for spec in extra_trees_specs() if spec.task == "classification")]

    fitted_regression: dict[str, Any] = {}
    fitted_classification: dict[str, Any] = {}
    regression_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    regression_prediction_rows: list[pd.DataFrame] = []
    classification_prediction_rows: list[pd.DataFrame] = []
    proba_rows: list[pd.DataFrame] = []
    hyperparameter_rows: list[dict[str, str]] = []
    model_card_rows: list[dict[str, Any]] = []
    learning_rows: list[pd.DataFrame] = []
    validation_rows: list[pd.DataFrame] = []
    permutation_rows: list[pd.DataFrame] = []
    native_importance_rows: list[pd.DataFrame] = []
    warnings: list[str] = []
    if class_target.warning:
        warnings.append(class_target.warning)

    for spec in regression_specs:
        estimator, params, fit_time, temporal_cv_rmse = _fit_spec(spec, x_train, y_train_reg, validation_design.temporal_cv)
        fitted_regression[spec.name] = estimator
        group_cv_rmse = evaluate_group_cv(estimator, x_train, y_train_reg, bundle.train["ISO3"], validation_design.group_cv, "neg_root_mean_squared_error")
        pred_train = estimator.predict(x_train)
        pred_test = estimator.predict(x_test)
        regression_rows.append(
            regression_metric_row(
                spec.name,
                y_train_reg,
                pred_train,
                y_test_reg,
                pred_test,
                temporal_cv_rmse,
                group_cv_rmse,
                fit_time,
                spec.family,
            )
        )
        regression_prediction_rows.append(
            pd.DataFrame(
                {
                    "Model": spec.name,
                    "Year": bundle.test["Year"].to_numpy(),
                    "country": bundle.test["country"].to_numpy(),
                    "ISO3": bundle.test["ISO3"].to_numpy(),
                    "Real": y_test_reg.to_numpy(),
                    "Predicție": pred_test,
                    "Reziduu": y_test_reg.to_numpy() - pred_test,
                    "Eroare absolută": np.abs(y_test_reg.to_numpy() - pred_test),
                }
            )
        )
        hyperparameter_rows.extend(_hyperparameter_rows("Regresie", spec.name, params))
        model_card_rows.append(_model_card_row("regression", spec, params, len(x_train), len(x_test), "RMSE Test", regression_rows[-1]["RMSE Test"], fit_time))
        save_model(estimator, spec.model_filename, paths)
        if spec.name == "XGBoost":
            _save_xgb_native(estimator, "xgb_regressor.json", paths)
            history = _xgb_training_history(estimator, x_train, y_train_reg, bundle.train["Year"], "regression", paths, "xgb_regression_train_validation_loss.json")
            save_table(history, "xgb_regression_training_history.csv", paths)
        if spec.name in REGRESSION_MODELS:
            learning_rows.append(_learning_curve(estimator, x_train, y_train_reg, x_test, y_test_reg, bundle.train["Year"], "regression", spec.name))
            if spec.name == "SVR":
                validation_rows.append(_validation_curve(estimator, x_train, y_train_reg, validation_design.temporal_cv, "regression", spec.name, "model__C", [0.1, 1, 10, 50, 100]))
                save_table(_svm_heatmap(estimator, x_train, y_train_reg, validation_design.temporal_cv, "regression", "svr_c_gamma_heatmap.json"), "svr_c_gamma_scores.csv", paths)
            elif spec.name in {"Random Forest", "Extra Trees"}:
                validation_rows.append(_validation_curve(estimator, x_train, y_train_reg, validation_design.temporal_cv, "regression", spec.name, "model__max_depth", [3, 5, 8, 12, None]))
            elif spec.name in {"Gradient Boosting", "XGBoost"}:
                validation_rows.append(_validation_curve(estimator, x_train, y_train_reg, validation_design.temporal_cv, "regression", spec.name, "model__learning_rate", [0.01, 0.03, 0.05, 0.1, 0.2]))
            permutation_rows.append(compute_permutation_importance(estimator, x_test, y_test_reg, bundle.feature_columns, spec.name, "regression", "neg_root_mean_squared_error"))
            native_importance_rows.append(native_feature_importance(estimator, bundle.feature_columns, spec.name, "regression"))

    for spec in classification_specs:
        estimator, params, fit_time, temporal_cv_f1 = _fit_spec(spec, x_train, y_train_cls, validation_design.temporal_cv)
        fitted_classification[spec.name] = estimator
        group_cv_f1 = evaluate_group_cv(estimator, x_train, y_train_cls, bundle.train["ISO3"], validation_design.group_cv, "f1_macro")
        pred_train = estimator.predict(x_train)
        pred_test = estimator.predict(x_test)
        proba_test = _predict_proba_aligned(estimator, x_test, [0, 1, 2])
        classification_rows.append(
            classification_metric_row(
                spec.name,
                y_train_cls,
                pred_train,
                y_test_cls,
                pred_test,
                proba_test,
                temporal_cv_f1,
                group_cv_f1,
                fit_time,
                spec.family,
                [0, 1, 2],
            )
        )
        classification_prediction_rows.append(
            pd.DataFrame(
                {
                    "Model": spec.name,
                    "Year": bundle.test["Year"].to_numpy(),
                    "country": bundle.test["country"].to_numpy(),
                    "ISO3": bundle.test["ISO3"].to_numpy(),
                    "Real cod": y_test_cls.to_numpy(),
                    "Predicție cod": pred_test,
                    "Real": [CODE_TO_CLASS[int(value)] for value in y_test_cls],
                    "Predicție": [CODE_TO_CLASS[int(value)] for value in pred_test],
                    "Corect": y_test_cls.to_numpy() == pred_test,
                }
            )
        )
        proba_rows.append(
            pd.DataFrame(
                {
                    "Model": spec.name,
                    "Year": bundle.test["Year"].to_numpy(),
                    "country": bundle.test["country"].to_numpy(),
                    "Real": [CODE_TO_CLASS[int(value)] for value in y_test_cls],
                    "Real cod": y_test_cls.to_numpy(),
                    "Predicție": [CODE_TO_CLASS[int(value)] for value in pred_test],
                    "Predicție cod": pred_test,
                    **{label: proba_test[:, code] for code, label in CODE_TO_CLASS.items()},
                }
            )
        )
        hyperparameter_rows.extend(_hyperparameter_rows("Clasificare", spec.name, params))
        model_card_rows.append(_model_card_row("classification", spec, params, len(x_train), len(x_test), "F1 Macro", classification_rows[-1]["F1 Macro"], fit_time))
        save_model(estimator, spec.model_filename, paths)
        if spec.name == "XGBoost":
            _save_xgb_native(estimator, "xgb_classifier.json", paths)
            history = _xgb_training_history(estimator, x_train, y_train_cls, bundle.train["Year"], "classification", paths, "xgb_classification_train_validation_loss.json")
            save_table(history, "xgb_classification_training_history.csv", paths)
        if spec.name in CLASSIFICATION_MODELS:
            learning_rows.append(_learning_curve(estimator, x_train, y_train_cls, x_test, y_test_cls, bundle.train["Year"], "classification", spec.name))
            if spec.name == "SVC":
                validation_rows.append(_validation_curve(estimator, x_train, y_train_cls, validation_design.temporal_cv, "classification", spec.name, "model__C", [0.1, 1, 10, 50, 100]))
                save_table(_svm_heatmap(estimator, x_train, y_train_cls, validation_design.temporal_cv, "classification", "svc_c_gamma_heatmap.json"), "svc_c_gamma_scores.csv", paths)
            elif spec.name in {"Random Forest", "Extra Trees"}:
                validation_rows.append(_validation_curve(estimator, x_train, y_train_cls, validation_design.temporal_cv, "classification", spec.name, "model__max_depth", [3, 5, 8, 12, None]))
            elif spec.name in {"Gradient Boosting", "XGBoost"}:
                validation_rows.append(_validation_curve(estimator, x_train, y_train_cls, validation_design.temporal_cv, "classification", spec.name, "model__learning_rate", [0.01, 0.03, 0.05, 0.1, 0.2]))
            permutation_rows.append(compute_permutation_importance(estimator, x_test, y_test_cls, bundle.feature_columns, spec.name, "classification", "f1_macro"))
            native_importance_rows.append(native_feature_importance(estimator, bundle.feature_columns, spec.name, "classification"))

    regression_leaderboard = metric_frame(regression_rows).sort_values("RMSE Test").reset_index(drop=True)
    classification_leaderboard = metric_frame(classification_rows).sort_values("F1 Macro", ascending=False).reset_index(drop=True)
    regression_predictions = pd.concat(regression_prediction_rows, ignore_index=True)
    classification_predictions = pd.concat(classification_prediction_rows, ignore_index=True)
    classification_proba = pd.concat(proba_rows, ignore_index=True)
    hyperparameters = pd.DataFrame(hyperparameter_rows)
    model_cards = pd.DataFrame(model_card_rows)
    learning_curves = pd.concat(learning_rows, ignore_index=True)
    validation_curves = pd.concat(validation_rows, ignore_index=True)
    permutation_importance_all = pd.concat(permutation_rows, ignore_index=True)
    native_importance_all = pd.concat(native_importance_rows, ignore_index=True) if native_importance_rows else pd.DataFrame()
    consensus = consensus_table(permutation_importance_all)
    class_metrics = _class_metrics_by_model(y_test_cls, classification_predictions)
    brier = _brier_scores(classification_proba)

    performance_year = (
        regression_predictions.groupby(["Model", "Year"], as_index=False)
        .agg(RMSE=("Reziduu", lambda values: float(np.sqrt(np.mean(np.square(values))))), MAE=("Eroare absolută", "mean"))
    )
    performance_country = (
        regression_predictions.groupby(["Model", "country"], as_index=False)
        .agg(**{"MAE mediu": ("Eroare absolută", "mean"), "RMSE": ("Reziduu", lambda values: float(np.sqrt(np.mean(np.square(values)))))})
    )
    shock_table = _shock_error_table(regression_predictions)

    save_table(regression_leaderboard, "regression_leaderboard.csv", paths)
    save_table(classification_leaderboard, "classification_leaderboard.csv", paths)
    save_table(regression_leaderboard.loc[regression_leaderboard["Familie"].eq("Baseline")], "regression_baselines.csv", paths)
    save_table(classification_leaderboard.loc[classification_leaderboard["Familie"].eq("Baseline")], "classification_baselines.csv", paths)
    save_table(regression_predictions, "regression_predictions.csv", paths)
    save_table(classification_predictions, "classification_predictions.csv", paths)
    save_table(classification_proba, "classification_probabilities.csv", paths)
    save_table(hyperparameters, "hyperparameters.csv", paths)
    save_table(model_cards, "model_cards.csv", paths)
    save_table(learning_curves, "learning_curves.csv", paths)
    save_table(validation_curves, "validation_curves.csv", paths)
    save_table(permutation_importance_all.loc[permutation_importance_all["Task"].eq("regression")], "permutation_importance_regression.csv", paths)
    save_table(permutation_importance_all.loc[permutation_importance_all["Task"].eq("classification")], "permutation_importance_classification.csv", paths)
    save_table(native_importance_all, "native_feature_importance.csv", paths)
    save_table(consensus, "importance_consensus.csv", paths)
    save_table(class_metrics, "classification_performance_by_class.csv", paths)
    save_table(brier, "classification_brier_scores.csv", paths)
    save_table(performance_year, "performance_by_year.csv", paths)
    save_table(performance_country, "performance_by_country.csv", paths)
    save_table(shock_table, "shock_period_errors.csv", paths)

    for filename, model_name in [
        ("svr_metrics.csv", "SVR"),
        ("rf_regression_metrics.csv", "Random Forest"),
        ("gb_regression_metrics.csv", "Gradient Boosting"),
        ("xgb_regression_metrics.csv", "XGBoost"),
        ("extra_trees_regression_metrics.csv", "Extra Trees"),
    ]:
        save_table(regression_leaderboard.loc[regression_leaderboard["Model"].eq(model_name)], filename, paths)
    for filename, model_name in [
        ("svc_metrics.csv", "SVC"),
        ("rf_classification_metrics.csv", "Random Forest"),
        ("gb_classification_metrics.csv", "Gradient Boosting"),
        ("xgb_classification_metrics.csv", "XGBoost"),
        ("extra_trees_classification_metrics.csv", "Extra Trees"),
    ]:
        save_table(classification_leaderboard.loc[classification_leaderboard["Model"].eq(model_name)], filename, paths)

    for model_name in REGRESSION_MODELS:
        stem = _model_stem(model_name)
        actual_vs_predicted(regression_predictions, model_name, f"{stem}_actual_vs_predicted.json")
        residual_plot(regression_predictions, model_name, f"{stem}_residuals.json")
        error_distribution(regression_predictions, model_name, f"{stem}_error_distribution.json")
        absolute_error_distribution(regression_predictions, model_name, f"{stem}_absolute_error_distribution.json")
        predictions_over_time(regression_predictions, model_name, f"{stem}_predictions_over_time.json")
        permutation_bar(permutation_importance_all, model_name, "regression", f"{stem}_permutation_importance_regression.json")
        if not native_importance_all.empty:
            importance_bar(native_importance_all, model_name, "regression", f"{stem}_feature_importance_regression.json", f"{model_name}: importanță nativă a predictorilor")
        model_lc = learning_curves.loc[(learning_curves["Model"].eq(model_name)) & (learning_curves["Task"].eq("regression"))]
        if not model_lc.empty:
            lc_long = model_lc.melt(id_vars=["Train până la", "Model", "Task", "Metrică"], value_vars=["Train", "Test"], var_name="Eșantion", value_name="Scor")
            fig = px.line(lc_long, x="Train până la", y="Scor", color="Eșantion", markers=True, color_discrete_map={"Train": TEAL, "Test": BURGUNDY})
            save_figure(apply_project_layout(fig, f"{model_name}: Learning Curve"), f"{stem}_learning_curve_regression.json", paths)
        model_vc = validation_curves.loc[(validation_curves["Model"].eq(model_name)) & (validation_curves["Task"].eq("regression"))]
        if not model_vc.empty:
            simple_line(model_vc, "Valoare", "Scor", GOLD, f"{model_name}: Validation Curve", f"{stem}_validation_curve_regression.json", "Scor validare temporală")

    for model_name in CLASSIFICATION_MODELS:
        stem = _model_stem(model_name)
        data = classification_predictions.loc[classification_predictions["Model"].eq(model_name)]
        proba = classification_proba.loc[classification_proba["Model"].eq(model_name)]
        confusion_heatmap(data["Real cod"], data["Predicție cod"], [0, 1, 2], CLASS_LABELS, f"{stem}_confusion_matrix.json", False, f"{model_name}: matrice de confuzie")
        confusion_heatmap(data["Real cod"], data["Predicție cod"], [0, 1, 2], CLASS_LABELS, f"{stem}_confusion_matrix_normalized.json", True, f"{model_name}: matrice de confuzie normalizată")
        roc_multiclass(data["Real cod"], proba[CLASS_LABELS].to_numpy(), [0, 1, 2], CLASS_LABELS, f"{stem}_roc_ovr.json", f"{model_name}: curbe ROC OvR")
        pr_multiclass(data["Real cod"], proba[CLASS_LABELS].to_numpy(), [0, 1, 2], CLASS_LABELS, f"{stem}_precision_recall.json", f"{model_name}: curbe Precision–Recall")
        probability_distribution(classification_proba, model_name, f"{stem}_probability_distribution.json")
        correct_incorrect_confidence(classification_proba, model_name, f"{stem}_correct_incorrect_confidence.json")
        permutation_bar(permutation_importance_all, model_name, "classification", f"{stem}_permutation_importance_classification.json")
        if not native_importance_all.empty:
            importance_bar(native_importance_all, model_name, "classification", f"{stem}_feature_importance_classification.json", f"{model_name}: importanță nativă a predictorilor")
        model_lc = learning_curves.loc[(learning_curves["Model"].eq(model_name)) & (learning_curves["Task"].eq("classification"))]
        if not model_lc.empty:
            lc_long = model_lc.melt(id_vars=["Train până la", "Model", "Task", "Metrică"], value_vars=["Train", "Test"], var_name="Eșantion", value_name="Scor")
            fig = px.line(lc_long, x="Train până la", y="Scor", color="Eșantion", markers=True, color_discrete_map={"Train": TEAL, "Test": BURGUNDY})
            save_figure(apply_project_layout(fig, f"{model_name}: Learning Curve"), f"{stem}_learning_curve_classification.json", paths)
        model_vc = validation_curves.loc[(validation_curves["Model"].eq(model_name)) & (validation_curves["Task"].eq("classification"))]
        if not model_vc.empty:
            simple_line(model_vc, "Valoare", "Scor", GOLD, f"{model_name}: Validation Curve", f"{stem}_validation_curve_classification.json", "Scor validare temporală")

    importance_heatmap(permutation_importance_all, "regression", "permutation_heatmap_regression.json", "Importanța predictorilor în modelele de regresie")
    importance_heatmap(permutation_importance_all, "classification", "permutation_heatmap_classification.json", "Importanța predictorilor în modelele de clasificare")
    metric_bar(regression_leaderboard, "RMSE Test", "RMSE Test – regresie", "leaderboard_regression_rmse.json")
    metric_bar(regression_leaderboard, "MAE Test", "MAE Test – regresie", "leaderboard_regression_mae.json")
    metric_bar(regression_leaderboard, "R² Test", "R² Test – regresie", "leaderboard_regression_r2.json")
    metric_bar(regression_leaderboard, "Generalization Gap", "Generalization Gap – regresie", "leaderboard_regression_gap.json")
    train_vs_test_rmse(regression_leaderboard, "leaderboard_train_vs_test_rmse.json")
    cv_comparison(regression_leaderboard, "leaderboard_regression_cv_comparison.json", "regression")
    small_multiple_actual_predicted(regression_predictions.loc[regression_predictions["Model"].isin(REGRESSION_MODELS)], "leaderboard_actual_vs_predicted_all.json")
    performance_by_year(performance_year.loc[performance_year["Model"].isin(REGRESSION_MODELS)], "performance_by_year_heatmap.json")
    performance_by_country(performance_country.loc[performance_country["Model"].eq(regression_leaderboard.iloc[0]["Model"])], "performance_by_country_bar.json")
    shock_errors(shock_table.loc[shock_table["Model"].isin(REGRESSION_MODELS)], "shock_period_errors.json")

    metric_bar(classification_leaderboard, "F1 Macro", "Macro F1 – clasificare", "leaderboard_classification_f1.json")
    metric_bar(classification_leaderboard, "Balanced Accuracy", "Balanced Accuracy – clasificare", "leaderboard_classification_balanced_accuracy.json")
    metric_bar(classification_leaderboard, "ROC-AUC OvR", "ROC-AUC OvR – clasificare", "leaderboard_classification_roc_auc.json")
    metric_bar(classification_leaderboard, "Log Loss", "Log Loss – clasificare", "leaderboard_classification_log_loss.json")
    cv_comparison(classification_leaderboard, "leaderboard_classification_cv_comparison.json", "classification")
    performance_by_class(class_metrics, "classification_performance_by_class.json")
    brier_scores(brier, "classification_brier_scores.json")

    complexity_rows = []
    complexity_map = {
        "Linear Regression": "Redusă",
        "Ridge": "Redusă",
        "LASSO": "Redusă",
        "Logistic Regression": "Redusă",
        "SVR": "Medie",
        "SVC": "Medie",
        "Random Forest": "Ridicată",
        "Gradient Boosting": "Ridicată",
        "XGBoost": "Ridicată",
        "Extra Trees": "Ridicată",
    }
    for _, row in regression_leaderboard.iterrows():
        complexity_rows.append({"Model": row["Model"], "Task": "Regresie", "Complexitate": complexity_map.get(row["Model"], "Redusă"), "Metrică test": row["RMSE Test"], "Fit time sec": row["Fit time sec"]})
    for _, row in classification_leaderboard.iterrows():
        complexity_rows.append({"Model": row["Model"], "Task": "Clasificare", "Complexitate": complexity_map.get(row["Model"], "Redusă"), "Metrică test": row["F1 Macro"], "Fit time sec": row["Fit time sec"]})
    complexity = pd.DataFrame(complexity_rows)
    save_table(complexity, "complexity_performance.csv", paths)
    complexity_performance(complexity, "complexity_performance.json")
    generalization_map(regression_leaderboard, "RMSE Train", "RMSE Test", "generalization_map_regression.json", "Generalization map – regresie")
    generalization_map(classification_leaderboard, "F1 Train", "F1 Macro", "generalization_map_classification.json", "Generalization map – clasificare")

    shap_warnings = []
    shap_regression = pd.DataFrame()
    shap_classification = pd.DataFrame()
    tree_regression = regression_leaderboard.loc[regression_leaderboard["Familie"].isin(TREE_FAMILIES)]
    tree_classification = classification_leaderboard.loc[classification_leaderboard["Familie"].isin(TREE_FAMILIES)]
    best_tree_regression = tree_regression.iloc[0]["Model"] if not tree_regression.empty else None
    best_tree_classification = tree_classification.iloc[0]["Model"] if not tree_classification.empty else None
    for model_name in tree_regression["Model"].tolist():
        try:
            shap_regression = generate_shap_regression(fitted_regression[model_name], x_test, y_test_reg, paths, "regression")
            best_tree_regression = model_name
            break
        except Exception as error:
            shap_warnings.append(f"SHAP regresie indisponibil pentru {model_name}: {error}")
    for model_name in tree_classification["Model"].tolist():
        try:
            shap_classification = generate_shap_classification(fitted_classification[model_name], x_test, paths, "classification")
            best_tree_classification = model_name
            break
        except Exception as error:
            shap_warnings.append(f"SHAP clasificare indisponibil pentru {model_name}: {error}")
    if not shap_regression.empty:
        save_table(shap_regression, "shap_global_regression.csv", paths)
        top_features = shap_regression["Feature raw"].head(3).tolist()
        pdp_files = generate_partial_dependence(fitted_regression[best_tree_regression], x_train, top_features, paths)
    else:
        save_table(pd.DataFrame(columns=["Feature raw", "Feature", "Mean |SHAP|"]), "shap_global_regression.csv", paths)
        pdp_files = []
    if not shap_classification.empty:
        save_table(shap_classification, "shap_global_classification.csv", paths)
    else:
        save_table(pd.DataFrame(columns=["Feature raw", "Feature", "Mean |SHAP|"]), "shap_global_classification.csv", paths)
    warnings.extend(shap_warnings)

    best_regression = regression_leaderboard.iloc[0]
    best_classification = classification_leaderboard.iloc[0]
    stable_regression = regression_leaderboard.sort_values(["Temporal CV RMSE", "Group CV RMSE"]).iloc[0]
    important_features = consensus.head(5)["Feature"].tolist()
    error_year = performance_year.loc[performance_year["Model"].eq(best_regression["Model"])].sort_values("RMSE", ascending=False).head(1)
    common_confusions = (
        classification_predictions.loc[classification_predictions["Real"].ne(classification_predictions["Predicție"])]
        .groupby(["Real", "Predicție"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .head(3)
    )
    if common_confusions.empty:
        confusion_text = "Nu există confuzii dominante în setul de test pentru modelele evaluate."
    else:
        confusion_text = "; ".join(f"{row.Real} → {row.Predicție} ({row['size']})" for _, row in common_confusions.iterrows())
    final_conclusions = {
        "regression": (
            f"{best_regression['Model']} a obținut cea mai redusă eroare RMSE pe perioada de test "
            f"({best_regression['RMSE Test']:.3f}). Rezultatul este evaluat împreună cu MAE, R², diferența train-test și validarea temporală."
        ),
        "regression_stability": (
            f"{stable_regression['Model']} are cea mai favorabilă combinație descriptivă între validarea temporală și validarea pe țări. "
            "Această informație completează, dar nu înlocuiește evaluarea pe holdout temporal."
        ),
        "features": "Predictorii care apar frecvent importanți sunt: " + ", ".join(important_features) + ". Importanța este predictivă, nu cauzală.",
        "periods": (
            f"Pentru modelul favorizat, cea mai mare eroare anuală agregată din test apare în {int(error_year.iloc[0]['Year'])}."
            if not error_year.empty
            else "Nu a putut fi identificat un an dominant al erorii."
        ),
        "classification": (
            f"{best_classification['Model']} a obținut cea mai ridicată valoare F1 Macro pe setul de test "
            f"({best_classification['F1 Macro']:.3f}), indicând cel mai bun echilibru predictiv între clase în această evaluare."
        ),
        "classification_confusions": f"Confuziile cele mai frecvente sunt: {confusion_text}",
        "general": "Rezultatele predictive completează analiza econometrică, dar nu înlocuiesc interpretarea structurală și economică a coeficienților.",
    }

    metadata = {
        "train_start": 2001,
        "train_end": 2018,
        "test_start": 2019,
        "test_end": 2023,
        "train_n": len(bundle.train),
        "test_n": len(bundle.test),
        "number_features": len(bundle.feature_columns),
        "number_countries": int(bundle.data["ISO3"].nunique()),
        "q33_class_threshold": class_target.q33,
        "q67_class_threshold": class_target.q67,
        "classification_distribution_train": class_target.distribution_train.to_dict(orient="records"),
        "classification_distribution_test": class_target.distribution_test.to_dict(orient="records"),
        "best_regression_model": str(best_regression["Model"]),
        "best_classification_model": str(best_classification["Model"]),
        "best_regression_rmse": float(best_regression["RMSE Test"]),
        "best_classification_f1_macro": float(best_classification["F1 Macro"]),
        "temporal_cv_folds": validation_design.temporal_folds,
        "group_cv_folds": validation_design.group_folds,
        "best_tree_regression_for_shap": best_tree_regression,
        "best_tree_classification_for_shap": best_tree_classification,
        "partial_dependence_files": pdp_files,
        "warnings": warnings,
        "final_conclusions": final_conclusions,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_metadata(metadata, paths)
    return metadata


if __name__ == "__main__":
    result = run_ml_analysis()
    print(f"Analiza Machine Learning a fost finalizată. Model regresie: {result['best_regression_model']}; model clasificare: {result['best_classification_model']}.")
