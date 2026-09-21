"""Metrici pentru regresie și clasificare."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def smape(y_true, y_pred) -> float:
    """Calculează sMAPE cu protecție la împărțire la zero."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    valid = denominator > np.finfo(float).eps
    if not np.any(valid):
        return np.nan
    return float(np.mean(200 * np.abs(y_pred[valid] - y_true[valid]) / denominator[valid]))


def regression_metric_row(
    model_name: str,
    y_train,
    pred_train,
    y_test,
    pred_test,
    temporal_cv_rmse: float,
    group_cv_rmse: float,
    fit_time: float,
    family: str,
) -> dict[str, float | str]:
    """Returnează metricile standard pentru regresie."""

    rmse_train = float(mean_squared_error(y_train, pred_train) ** 0.5)
    rmse_test = float(mean_squared_error(y_test, pred_test) ** 0.5)
    return {
        "Model": model_name,
        "Familie": family,
        "RMSE Train": rmse_train,
        "RMSE Test": rmse_test,
        "MAE Train": float(mean_absolute_error(y_train, pred_train)),
        "MAE Test": float(mean_absolute_error(y_test, pred_test)),
        "MedAE Test": float(median_absolute_error(y_test, pred_test)),
        "R² Train": float(r2_score(y_train, pred_train)),
        "R² Test": float(r2_score(y_test, pred_test)),
        "MAPE Test": float(mean_absolute_percentage_error(y_test, pred_test) * 100),
        "sMAPE Test": smape(y_test, pred_test),
        "Generalization Gap": rmse_test - rmse_train,
        "RMSE Test / RMSE Train": rmse_test / rmse_train if rmse_train > 0 else np.nan,
        "Temporal CV RMSE": temporal_cv_rmse,
        "Group CV RMSE": group_cv_rmse,
        "Fit time sec": fit_time,
    }


def classification_metric_row(
    model_name: str,
    y_train,
    pred_train,
    y_test,
    pred_test,
    proba_test,
    temporal_cv_f1: float,
    group_cv_f1: float,
    fit_time: float,
    family: str,
    labels: list[int],
) -> dict[str, float | str]:
    """Returnează metricile standard pentru clasificare."""

    row: dict[str, float | str] = {
        "Model": model_name,
        "Familie": family,
        "Accuracy": float(accuracy_score(y_test, pred_test)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, pred_test)),
        "Precision Macro": float(precision_score(y_test, pred_test, average="macro", zero_division=0)),
        "Recall Macro": float(recall_score(y_test, pred_test, average="macro", zero_division=0)),
        "F1 Macro": float(f1_score(y_test, pred_test, average="macro", zero_division=0)),
        "F1 Weighted": float(f1_score(y_test, pred_test, average="weighted", zero_division=0)),
        "F1 Train": float(f1_score(y_train, pred_train, average="macro", zero_division=0)),
        "F1 Train - F1 Test": float(f1_score(y_train, pred_train, average="macro", zero_division=0) - f1_score(y_test, pred_test, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, pred_test)),
        "Cohen Kappa": float(cohen_kappa_score(y_test, pred_test)),
        "Temporal CV F1 Macro": temporal_cv_f1,
        "Group CV F1 Macro": group_cv_f1,
        "Fit time sec": fit_time,
    }
    try:
        row["ROC-AUC OvR"] = float(roc_auc_score(y_test, proba_test, labels=labels, multi_class="ovr", average="macro"))
    except Exception as error:
        row["ROC-AUC OvR"] = np.nan
        row["ROC-AUC motiv"] = str(error)
    try:
        row["Log Loss"] = float(log_loss(y_test, proba_test, labels=labels))
    except Exception as error:
        row["Log Loss"] = np.nan
        row["Log Loss motiv"] = str(error)
    return row


def metric_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    """Transformă rândurile de metrici într-un DataFrame curat."""

    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
