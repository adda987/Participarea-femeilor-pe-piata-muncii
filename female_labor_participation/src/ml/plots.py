"""Grafice Plotly pentru experimentul Machine Learning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from src.ml.classification_target import CLASS_LABELS
from src.ml.utils import BURGUNDY, CLASS_COLORS, GOLD, IVORY, MUTED, NAVY, PLUM, TEAL, apply_project_layout, save_figure


MODEL_COLORS = [TEAL, BURGUNDY, GOLD, PLUM, NAVY, MUTED, "#2F8F88", "#8A334E", "#5B4A86"]


def timeline_train_test(filename: str) -> None:
    """Grafic timeline pentru split-ul temporal."""

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[18], y=["Experiment"], base=[2001], orientation="h", marker_color=TEAL, name="Train 2001–2018"))
    fig.add_trace(go.Bar(x=[5], y=["Experiment"], base=[2019], orientation="h", marker_color=BURGUNDY, name="Test 2019–2023"))
    fig.add_vline(x=2018.5, line_color=GOLD, line_width=3)
    fig.update_layout(barmode="overlay", xaxis_range=[2000.5, 2023.5], height=260)
    fig.update_xaxes(title="An", tickmode="linear", dtick=2)
    fig.update_yaxes(showticklabels=False)
    save_figure(apply_project_layout(fig, "Separarea temporală a datelor"), filename)


def target_distribution(train: pd.DataFrame, test: pd.DataFrame, filename: str) -> None:
    """Histogramă comparativă pentru target."""

    data = pd.concat(
        [
            train.assign(Eșantion="Train"),
            test.assign(Eșantion="Test"),
        ],
        ignore_index=True,
    )
    fig = px.histogram(
        data,
        x="female_lfpr",
        color="Eșantion",
        marginal="box",
        nbins=24,
        opacity=0.72,
        color_discrete_map={"Train": TEAL, "Test": BURGUNDY},
    )
    fig.update_layout(bargap=0.06)
    fig.update_xaxes(title="Participarea femeilor la forța de muncă (%)")
    fig.update_yaxes(title="Observații")
    save_figure(apply_project_layout(fig, "Distribuția target-ului de regresie"), filename)


def class_distribution(distribution: pd.DataFrame, filename: str) -> None:
    """Distribuția claselor în train și test."""

    fig = px.bar(
        distribution,
        x="Clasă",
        y="Observații",
        color="Clasă",
        facet_col="Eșantion",
        text="Observații",
        color_discrete_map=CLASS_COLORS,
        category_orders={"Clasă": CLASS_LABELS},
    )
    fig.update_yaxes(title="Observații")
    save_figure(apply_project_layout(fig, "Distribuția claselor de participare"), filename)


def dataset_shift(shift: pd.DataFrame, filename: str) -> None:
    """Bar chart pentru diferențele standardizate train-test."""

    fig = px.bar(
        shift.sort_values("Diferență standardizată"),
        x="Diferență standardizată",
        y="Feature",
        orientation="h",
        color="Diferență standardizată",
        color_continuous_scale=[[0, BURGUNDY], [0.5, IVORY], [1, TEAL]],
    )
    fig.update_layout(coloraxis_colorbar={"title": "Dif. std."})
    save_figure(apply_project_layout(fig, "Schimbări ale distribuției predictorilor între train și test"), filename)


def actual_vs_predicted(predictions: pd.DataFrame, model: str, filename: str) -> None:
    """Actual vs Predicted pentru regresie."""

    data = predictions.loc[predictions["Model"].eq(model)]
    fig = px.scatter(data, x="Real", y="Predicție", hover_data=["country", "Year"], color="Year", color_continuous_scale=[TEAL, GOLD, BURGUNDY])
    min_value = float(np.nanmin([data["Real"].min(), data["Predicție"].min()]))
    max_value = float(np.nanmax([data["Real"].max(), data["Predicție"].max()]))
    fig.add_trace(go.Scatter(x=[min_value, max_value], y=[min_value, max_value], mode="lines", line={"color": NAVY, "dash": "dash"}, name="Predicție perfectă"))
    fig.update_xaxes(title="Valori reale")
    fig.update_yaxes(title="Predicții")
    save_figure(apply_project_layout(fig, f"{model}: valori reale vs predicții"), filename)


def residual_plot(predictions: pd.DataFrame, model: str, filename: str) -> None:
    """Reziduuri vs predicții."""

    data = predictions.loc[predictions["Model"].eq(model)]
    fig = px.scatter(data, x="Predicție", y="Reziduu", hover_data=["country", "Year"], color="Year", color_continuous_scale=[TEAL, GOLD, BURGUNDY])
    fig.add_hline(y=0, line_color=GOLD, line_width=2)
    fig.update_xaxes(title="Predicție")
    fig.update_yaxes(title="Reziduu")
    save_figure(apply_project_layout(fig, f"{model}: reziduuri vs predicții"), filename)


def error_distribution(predictions: pd.DataFrame, model: str, filename: str) -> None:
    """Distribuția erorilor."""

    data = predictions.loc[predictions["Model"].eq(model)]
    fig = px.histogram(data, x="Reziduu", nbins=22, marginal="box", color_discrete_sequence=[TEAL])
    fig.add_vline(x=0, line_color=GOLD, line_width=2)
    save_figure(apply_project_layout(fig, f"{model}: distribuția erorilor"), filename)


def absolute_error_distribution(predictions: pd.DataFrame, model: str, filename: str) -> None:
    """Distribuția erorii absolute."""

    data = predictions.loc[predictions["Model"].eq(model)]
    fig = px.histogram(data, x="Eroare absolută", nbins=20, marginal="rug", color_discrete_sequence=[BURGUNDY])
    save_figure(apply_project_layout(fig, f"{model}: distribuția erorii absolute"), filename)


def predictions_over_time(predictions: pd.DataFrame, model: str, filename: str) -> None:
    """Predicții și valori reale pe anii de test."""

    data = predictions.loc[predictions["Model"].eq(model)]
    summary = data.groupby("Year", as_index=False)[["Real", "Predicție"]].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary["Year"], y=summary["Real"], mode="lines+markers", line={"color": NAVY, "width": 3}, name="Real"))
    fig.add_trace(go.Scatter(x=summary["Year"], y=summary["Predicție"], mode="lines+markers", line={"color": TEAL, "width": 3, "dash": "dash"}, name="Predicție"))
    fig.update_xaxes(title="An")
    fig.update_yaxes(title="Participare feminină (%)")
    save_figure(apply_project_layout(fig, f"{model}: predicții în timp pe setul test"), filename)


def simple_line(data: pd.DataFrame, x: str, y: str, color: str, title: str, filename: str, y_title: str | None = None) -> None:
    """Linie generică pentru learning/validation curves."""

    fig = px.line(data, x=x, y=y, markers=True, color_discrete_sequence=[color])
    fig.update_yaxes(title=y_title or y)
    save_figure(apply_project_layout(fig, title), filename)


def metric_bar(data: pd.DataFrame, metric: str, title: str, filename: str) -> None:
    """Bar chart pentru o metrică de leaderboard."""

    plot_data = data.sort_values(metric, ascending=True if metric not in {"R² Test", "F1 Macro", "Balanced Accuracy", "ROC-AUC OvR"} else False)
    fig = px.bar(plot_data, x=metric, y="Model", orientation="h", color="Familie", color_discrete_sequence=MODEL_COLORS, text=metric)
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    save_figure(apply_project_layout(fig, title), filename)


def train_vs_test_rmse(data: pd.DataFrame, filename: str) -> None:
    """Compară RMSE train și test."""

    plot_data = data[["Model", "RMSE Train", "RMSE Test"]].melt(id_vars="Model", var_name="Metrică", value_name="RMSE")
    fig = px.bar(plot_data, x="Model", y="RMSE", color="Metrică", barmode="group", color_discrete_map={"RMSE Train": TEAL, "RMSE Test": BURGUNDY})
    fig.update_xaxes(tickangle=-35)
    save_figure(apply_project_layout(fig, "Train vs Test RMSE"), filename)


def cv_comparison(data: pd.DataFrame, filename: str, task: str) -> None:
    """Compară validarea temporală și pe țări."""

    if task == "regression":
        columns = ["Temporal CV RMSE", "Group CV RMSE"]
        value_name = "RMSE"
    else:
        columns = ["Temporal CV F1 Macro", "Group CV F1 Macro"]
        value_name = "F1 Macro"
    plot_data = data[["Model", *columns]].melt(id_vars="Model", var_name="Validare", value_name=value_name)
    fig = px.bar(plot_data, x="Model", y=value_name, color="Validare", barmode="group", color_discrete_sequence=[TEAL, BURGUNDY])
    fig.update_xaxes(tickangle=-35)
    save_figure(apply_project_layout(fig, "Validare temporală vs validare pe țări"), filename)


def permutation_bar(importance: pd.DataFrame, model: str, task: str, filename: str) -> None:
    """Importanța prin permutare pentru un model."""

    data = importance.loc[(importance["Model"].eq(model)) & (importance["Task"].eq(task))].sort_values("Importanță", ascending=True).tail(12)
    if data.empty:
        return
    fig = px.bar(data, x="Importanță", y="Feature", orientation="h", color_discrete_sequence=[TEAL if task == "regression" else BURGUNDY])
    save_figure(apply_project_layout(fig, f"{model}: importanță prin permutare"), filename)


def importance_bar(importance: pd.DataFrame, model: str, task: str, filename: str, title: str) -> None:
    """Importanță nativă pentru arbori."""

    data = importance.loc[(importance["Model"].eq(model)) & (importance["Task"].eq(task))].sort_values("Importanță", ascending=True).tail(12)
    if data.empty:
        return
    fig = px.bar(data, x="Importanță", y="Feature", orientation="h", color_discrete_sequence=[GOLD])
    save_figure(apply_project_layout(fig, title), filename)


def importance_heatmap(importance: pd.DataFrame, task: str, filename: str, title: str) -> None:
    """Heatmap pentru importanța normalizată între modele."""

    data = importance.loc[importance["Task"].eq(task)].pivot_table(index="Feature", columns="Model", values="Importanță normalizată", fill_value=0)
    fig = px.imshow(data, color_continuous_scale=[[0, IVORY], [0.5, GOLD], [1, TEAL]], aspect="auto")
    fig.update_layout(coloraxis_colorbar={"title": "Importanță"})
    save_figure(apply_project_layout(fig, title), filename)


def confusion_heatmap(y_true, y_pred, labels: list[int], label_names: list[str], filename: str, normalized: bool, title: str) -> None:
    """Matrice de confuzie."""

    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalized else None)
    fig = px.imshow(
        matrix,
        x=label_names,
        y=label_names,
        text_auto=".2f" if normalized else True,
        color_continuous_scale=[[0, IVORY], [0.5, GOLD], [1, TEAL]],
        aspect="auto",
    )
    fig.update_xaxes(title="Clasă prezisă")
    fig.update_yaxes(title="Clasă reală")
    save_figure(apply_project_layout(fig, title), filename)


def roc_multiclass(y_true, proba, labels: list[int], label_names: list[str], filename: str, title: str) -> None:
    """Curbe ROC one-vs-rest."""

    y_bin = label_binarize(y_true, classes=labels)
    fig = go.Figure()
    for index, label in enumerate(labels):
        if y_bin[:, index].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, index], proba[:, index])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{label_names[index]} AUC={auc(fpr, tpr):.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": MUTED, "dash": "dash"}, name="Benchmark"))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    save_figure(apply_project_layout(fig, title), filename)


def pr_multiclass(y_true, proba, labels: list[int], label_names: list[str], filename: str, title: str) -> None:
    """Curbe Precision–Recall one-vs-rest."""

    y_bin = label_binarize(y_true, classes=labels)
    fig = go.Figure()
    for index, label in enumerate(labels):
        if y_bin[:, index].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, index], proba[:, index])
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=label_names[index]))
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    save_figure(apply_project_layout(fig, title), filename)


def probability_distribution(proba_frame: pd.DataFrame, model: str, filename: str) -> None:
    """Distribuția probabilităților prezise."""

    data = proba_frame.loc[proba_frame["Model"].eq(model)]
    long_data = data.melt(id_vars=["Model", "Year", "country", "Real", "Predicție"], value_vars=CLASS_LABELS, var_name="Clasă", value_name="Probabilitate")
    fig = px.histogram(long_data, x="Probabilitate", color="Clasă", nbins=20, opacity=0.72, color_discrete_map=CLASS_COLORS)
    save_figure(apply_project_layout(fig, f"{model}: distribuția probabilităților prezise"), filename)


def correct_incorrect_confidence(proba_frame: pd.DataFrame, model: str, filename: str) -> None:
    """Încredere prezisă pentru predicții corecte și incorecte."""

    data = proba_frame.loc[proba_frame["Model"].eq(model)].copy()
    data["Status"] = np.where(data["Real"].eq(data["Predicție"]), "Corect", "Incorect")
    data["Încredere"] = data[CLASS_LABELS].max(axis=1)
    fig = px.box(data, x="Status", y="Încredere", color="Status", points="all", color_discrete_map={"Corect": TEAL, "Incorect": BURGUNDY})
    save_figure(apply_project_layout(fig, f"{model}: predicții corecte vs incorecte"), filename)


def performance_by_year(performance: pd.DataFrame, filename: str) -> None:
    """Heatmap RMSE pe ani de test."""

    data = performance.pivot_table(index="Model", columns="Year", values="RMSE", fill_value=np.nan)
    fig = px.imshow(data, color_continuous_scale=[[0, TEAL], [0.5, GOLD], [1, BURGUNDY]], aspect="auto")
    fig.update_layout(coloraxis_colorbar={"title": "RMSE"})
    save_figure(apply_project_layout(fig, "Stabilitatea performanței predictive în perioada de test"), filename)


def performance_by_country(performance: pd.DataFrame, filename: str) -> None:
    """Bar chart pentru erori medii pe țară."""

    data = performance.sort_values("MAE mediu", ascending=True)
    fig = px.bar(data.tail(18), x="MAE mediu", y="country", orientation="h", color="MAE mediu", color_continuous_scale=[[0, TEAL], [0.5, GOLD], [1, BURGUNDY]])
    save_figure(apply_project_layout(fig, "Economii cu cele mai mari erori medii absolute pe test"), filename)


def shock_errors(data: pd.DataFrame, filename: str) -> None:
    """Erori agregate pe perioade de șoc."""

    fig = px.bar(data, x="Perioadă", y="RMSE", color="Model", barmode="group", color_discrete_sequence=MODEL_COLORS)
    save_figure(apply_project_layout(fig, "Erori predictive în perioade de șoc"), filename)


def complexity_performance(data: pd.DataFrame, filename: str) -> None:
    """Complexitate vs performanță predictivă."""

    fig = px.scatter(data, x="Complexitate", y="Metrică test", color="Task", size="Fit time sec", text="Model", color_discrete_map={"Regresie": TEAL, "Clasificare": BURGUNDY})
    fig.update_traces(textposition="top center")
    save_figure(apply_project_layout(fig, "Complexitatea modelului și performanța predictivă"), filename)


def generalization_map(data: pd.DataFrame, x: str, y: str, filename: str, title: str) -> None:
    """Scatter train/test pentru diagnostic de generalizare."""

    fig = px.scatter(data, x=x, y=y, color="Familie", text="Model", size="Fit time sec", color_discrete_sequence=MODEL_COLORS)
    fig.update_traces(textposition="top center")
    save_figure(apply_project_layout(fig, title), filename)


def small_multiple_actual_predicted(predictions: pd.DataFrame, filename: str) -> None:
    """Small multiples actual vs predicted pentru modelele principale."""

    data = predictions.copy()
    fig = px.scatter(data, x="Real", y="Predicție", facet_col="Model", facet_col_wrap=3, color="Year", color_continuous_scale=[TEAL, GOLD, BURGUNDY])
    save_figure(apply_project_layout(fig, "Actual vs Predicted pentru modelele principale"), filename)


def performance_by_class(class_metrics: pd.DataFrame, filename: str) -> None:
    """Metrici de clasificare pe clasă."""

    fig = px.bar(class_metrics, x="Clasă", y="F1", color="Model", barmode="group", color_discrete_sequence=MODEL_COLORS)
    save_figure(apply_project_layout(fig, "Performanță pe clasă"), filename)


def brier_scores(data: pd.DataFrame, filename: str) -> None:
    """Brier score pe clasă pentru calibrare."""

    fig = px.bar(data, x="Model", y="Brier Score", color="Clasă", barmode="group", color_discrete_map=CLASS_COLORS)
    fig.update_xaxes(tickangle=-35)
    save_figure(apply_project_layout(fig, "Diagnostic de calibrare: Brier score pe clasă"), filename)
