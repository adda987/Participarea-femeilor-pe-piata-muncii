"""Interpretabilitate: permutation importance, SHAP și PDP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.inspection import partial_dependence, permutation_importance

from src.ml.data_preparation import FEATURE_LABELS
from src.ml.utils import BURGUNDY, GOLD, TEAL, MLOutputPaths, apply_project_layout, save_figure, set_matplotlib_cache


TREE_FAMILIES = {"Random Forest", "Gradient Boosting", "XGBoost", "Extra Trees"}


def compute_permutation_importance(estimator, x_test, y_test, feature_columns: list[str], model_name: str, task: str, scoring: str) -> pd.DataFrame:
    """Calculează importanța prin permutare pe setul de test."""

    result = permutation_importance(estimator, x_test, y_test, n_repeats=7, random_state=42, scoring=scoring, n_jobs=1)
    data = pd.DataFrame(
        {
            "Model": model_name,
            "Task": task,
            "Feature raw": feature_columns,
            "Feature": [FEATURE_LABELS.get(feature, feature) for feature in feature_columns],
            "Importanță": result.importances_mean,
            "Std": result.importances_std,
        }
    )
    max_abs = data["Importanță"].abs().max()
    data["Importanță normalizată"] = data["Importanță"].abs() / max_abs if max_abs and np.isfinite(max_abs) else 0
    data["Rang"] = data["Importanță"].rank(ascending=False, method="min")
    return data


def native_feature_importance(estimator, feature_columns: list[str], model_name: str, task: str) -> pd.DataFrame:
    """Extrage importanța nativă pentru modele tree-based."""

    model = estimator.named_steps.get("model") if hasattr(estimator, "named_steps") else estimator
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return pd.DataFrame(columns=["Model", "Task", "Feature raw", "Feature", "Importanță"])
    return pd.DataFrame(
        {
            "Model": model_name,
            "Task": task,
            "Feature raw": feature_columns,
            "Feature": [FEATURE_LABELS.get(feature, feature) for feature in feature_columns],
            "Importanță": np.asarray(values, dtype=float),
        }
    )


def consensus_table(importance: pd.DataFrame) -> pd.DataFrame:
    """Calculează consensul importanțelor între modele."""

    if importance.empty:
        return pd.DataFrame(columns=["Feature", "Rang mediu regresie", "Rang mediu clasificare", "Top 5 regresie", "Top 5 clasificare"])
    rows = []
    for feature in sorted(importance["Feature"].unique()):
        reg = importance.loc[(importance["Feature"].eq(feature)) & (importance["Task"].eq("regression"))]
        cls = importance.loc[(importance["Feature"].eq(feature)) & (importance["Task"].eq("classification"))]
        rows.append(
            {
                "Feature": feature,
                "Rang mediu regresie": reg["Rang"].mean() if not reg.empty else np.nan,
                "Rang mediu clasificare": cls["Rang"].mean() if not cls.empty else np.nan,
                "Top 5 regresie": int((reg["Rang"] <= 5).sum()) if not reg.empty else 0,
                "Top 5 clasificare": int((cls["Rang"] <= 5).sum()) if not cls.empty else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["Top 5 regresie", "Top 5 clasificare", "Rang mediu regresie"], ascending=[False, False, True])


def _processed_features(estimator, x_data: pd.DataFrame) -> pd.DataFrame:
    """Aplică doar preprocesarea pipeline-ului pentru interpretare."""

    if hasattr(estimator, "named_steps") and "imputer" in estimator.named_steps:
        values = estimator.named_steps["imputer"].transform(x_data)
        return pd.DataFrame(values, columns=x_data.columns, index=x_data.index)
    return x_data.copy()


def _tree_model(estimator) -> Any:
    """Extrage modelul final din pipeline."""

    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        return estimator.named_steps["model"]
    return estimator


def generate_shap_regression(estimator, x_test: pd.DataFrame, y_test: pd.Series, paths: MLOutputPaths, prefix: str) -> pd.DataFrame:
    """Generează outputurile SHAP pentru regresie tree-based."""

    set_matplotlib_cache(paths)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    x_processed = _processed_features(estimator, x_test)
    model = _tree_model(estimator)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_processed)
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 0]
    global_importance = pd.DataFrame(
        {
            "Feature raw": x_processed.columns,
            "Feature": [FEATURE_LABELS.get(feature, feature) for feature in x_processed.columns],
            "Mean |SHAP|": np.abs(values).mean(axis=0),
        }
    ).sort_values("Mean |SHAP|", ascending=False)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(paths.shap / f"{prefix}_beeswarm.png", dpi=170, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(paths.shap / f"{prefix}_global_bar.png", dpi=170, bbox_inches="tight")
    plt.close()

    for feature in global_importance["Feature raw"].head(3):
        plt.figure()
        shap.plots.scatter(shap_values[:, feature], show=False)
        plt.tight_layout()
        plt.savefig(paths.shap / f"{prefix}_dependence_{feature}.png", dpi=170, bbox_inches="tight")
        plt.close()

    quantile_targets = {
        "low": y_test.quantile(0.1),
        "median": y_test.quantile(0.5),
        "high": y_test.quantile(0.9),
    }
    for label, target_value in quantile_targets.items():
        position = int(np.argmin(np.abs(y_test.to_numpy() - target_value)))
        plt.figure()
        shap.plots.waterfall(shap_values[position], show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(paths.shap / f"{prefix}_waterfall_{label}.png", dpi=170, bbox_inches="tight")
        plt.close()

    return global_importance


def generate_shap_classification(estimator, x_test: pd.DataFrame, paths: MLOutputPaths, prefix: str) -> pd.DataFrame:
    """Generează output SHAP global pentru clasificare tree-based."""

    set_matplotlib_cache(paths)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    x_processed = _processed_features(estimator, x_test)
    model = _tree_model(estimator)
    explainer = shap.TreeExplainer(model)
    try:
        shap_values = explainer(x_processed)
        values = shap_values.values
    except Exception:
        values = explainer.shap_values(x_processed)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 3:
        class_values = np.abs(values).mean(axis=0)
        if class_values.shape[0] != len(x_processed.columns) and class_values.shape[1] == len(x_processed.columns):
            class_values = class_values.T
        mean_abs = class_values.mean(axis=1)
    else:
        mean_abs = np.abs(values).mean(axis=0)
        class_values = None
    importance_data = {
        "Feature raw": x_processed.columns,
        "Feature": [FEATURE_LABELS.get(feature, feature) for feature in x_processed.columns],
        "Mean |SHAP|": mean_abs,
    }
    if class_values is not None and class_values.ndim == 2 and class_values.shape[1] >= 3:
        labels = ["Participare scăzută", "Participare medie", "Participare ridicată"]
        for index, label in enumerate(labels[: class_values.shape[1]]):
            importance_data[f"Mean |SHAP| {label}"] = class_values[:, index]
    global_importance = pd.DataFrame(importance_data).sort_values("Mean |SHAP|", ascending=False)

    plt.figure()
    plot_data = global_importance.head(12).sort_values("Mean |SHAP|", ascending=True)
    plt.barh(plot_data["Feature"], plot_data["Mean |SHAP|"], color=BURGUNDY)
    plt.xlabel("Mean |SHAP|")
    plt.title("SHAP global pentru clasificare")
    plt.tight_layout()
    plt.savefig(paths.shap / f"{prefix}_classification_global_bar.png", dpi=170, bbox_inches="tight")
    plt.close()

    return global_importance


def generate_partial_dependence(estimator, x_train: pd.DataFrame, top_features: list[str], paths: MLOutputPaths) -> list[str]:
    """Generează PDP-uri Plotly pentru cei mai importanți predictori."""

    output_names: list[str] = []
    pdp_data = x_train.copy()
    numeric_medians = pdp_data.median(numeric_only=True)
    pdp_data = pdp_data.fillna(numeric_medians)
    for feature in top_features[:3]:
        try:
            result = partial_dependence(estimator, pdp_data, features=[feature], grid_resolution=24, kind="average")
            grid = result["grid_values"][0]
            average = result["average"][0]
        except Exception:
            continue
        grid = np.asarray(grid, dtype=float)
        average = np.asarray(average, dtype=float)
        valid = np.isfinite(grid) & np.isfinite(average)
        if not valid.any():
            continue
        grid = grid[valid]
        average = average[valid]
        data = pd.DataFrame({"Valoare predictor": grid, "Predicție medie": average})
        fig = px.line(data, x="Valoare predictor", y="Predicție medie", markers=True, color_discrete_sequence=[TEAL])
        filename = f"partial_dependence_{feature}.json"
        save_figure(apply_project_layout(fig, f"Relația predictivă marginală estimată: {FEATURE_LABELS.get(feature, feature)}"), filename, paths)
        output_names.append(filename)
    return output_names
