"""Clustering și PCA pentru economiile europene."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import os
from typing import Any
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.correlation_analysis import (
    DEPENDENT_SLUG,
    COUNTRY_FIELD,
    ISO3_FIELD,
    prepare_correlation_dataset,
    variable_label,
)
from src.geo_config import CLUSTER_COLORS, GEOMETRY_ISO_COLUMN, country_label
from src.utils import PALETTE


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

CLUSTER_VARIABLES: list[str] = [
    DEPENDENT_SLUG,
    "gdp_pc_ppp",
    "female_unemployment",
    "fertility",
    "urbanization",
    "female_services_employment",
    "female_vulnerable_employment",
    "internet_use",
    "female_tertiary_education",
    "women_parliament",
    "control_corruption",
]

RADAR_VARIABLES: list[str] = [
    DEPENDENT_SLUG,
    "gdp_pc_ppp",
    "fertility",
    "internet_use",
    "female_tertiary_education",
    "control_corruption",
]

MIN_VARIABLE_COUNTRY_COVERAGE = 0.70
REDUNDANCY_CORRELATION_THRESHOLD = 0.98
NEAR_CONSTANT_STD_THRESHOLD = 1e-8
RANDOM_STATE = 42
KMEANS_N_INIT = 20


@dataclass(frozen=True)
class ClusteringAnalysisResult:
    """Rezultatele complete ale clustering-ului și PCA."""

    countries: pd.DataFrame
    standardized_countries: pd.DataFrame
    selected_variables: list[str]
    selection_notes: list[str]
    evaluation: pd.DataFrame
    optimal_k: int
    profiles_original: pd.DataFrame
    profiles_standardized: pd.DataFrame
    pca_scores: pd.DataFrame
    pca_loadings: pd.DataFrame
    explained_variance_ratio: tuple[float, float]
    linkage_matrix: np.ndarray


def _with_country_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    """Adaugă denumiri românești pe baza codului ISO3."""

    labelled = dataset.copy()
    labelled[COUNTRY_FIELD] = [
        country_label(iso3, fallback)
        for iso3, fallback in zip(labelled[ISO3_FIELD].astype(str), labelled[COUNTRY_FIELD].astype(str))
    ]
    return labelled


def country_level_cluster_candidates(dataframe: pd.DataFrame | None) -> pd.DataFrame:
    """Agregă variabilele candidate la nivel de țară pentru perioada 2001-2023."""

    dataset = prepare_correlation_dataset(dataframe)
    if dataset.empty:
        return pd.DataFrame(columns=[ISO3_FIELD, COUNTRY_FIELD, *CLUSTER_VARIABLES])

    country_means = (
        dataset.groupby([ISO3_FIELD, COUNTRY_FIELD], as_index=False)[CLUSTER_VARIABLES]
        .mean(numeric_only=True)
        .sort_values(COUNTRY_FIELD)
        .reset_index(drop=True)
    )
    return _with_country_labels(country_means)


def _select_variables(country_means: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Selectează variabilele finale pentru clustering și documentează deciziile."""

    selected: list[str] = []
    notes: list[str] = []
    coverage = country_means[CLUSTER_VARIABLES].notna().mean()
    standard_deviation = country_means[CLUSTER_VARIABLES].std(skipna=True)

    for slug in CLUSTER_VARIABLES:
        if coverage[slug] < MIN_VARIABLE_COUNTRY_COVERAGE:
            notes.append(f"{variable_label(slug)} exclusă: acoperire sub pragul de 70% la nivel de țară.")
            continue
        if not isfinite(float(standard_deviation[slug])) or float(standard_deviation[slug]) <= NEAR_CONSTANT_STD_THRESHOLD:
            notes.append(f"{variable_label(slug)} exclusă: variație insuficientă între țări.")
            continue
        selected.append(slug)

    dropped_for_redundancy: set[str] = set()
    if len(selected) > 1:
        correlations = country_means[selected].corr(method="pearson").abs()
        for index, first in enumerate(selected):
            if first in dropped_for_redundancy:
                continue
            for second in selected[index + 1 :]:
                if second in dropped_for_redundancy:
                    continue
                value = correlations.loc[first, second]
                if pd.notna(value) and float(value) >= REDUNDANCY_CORRELATION_THRESHOLD:
                    first_coverage = coverage[first]
                    second_coverage = coverage[second]
                    drop_slug = second if second_coverage <= first_coverage else first
                    dropped_for_redundancy.add(drop_slug)
                    notes.append(
                        f"{variable_label(drop_slug)} exclusă doar pentru clustering: corelație Pearson peste 0,98 cu o variabilă mai bine acoperită."
                    )

    selected = [slug for slug in selected if slug not in dropped_for_redundancy]
    if not notes:
        notes.append("Variabilele candidate au trecut verificările de acoperire, variație și redundanță pentru analiza exploratorie.")
    return selected, notes


def _evaluate_kmeans(values: np.ndarray, max_k: int = 6) -> pd.DataFrame:
    """Calculează inertia, Silhouette Score și Calinski-Harabasz pentru k=2...6."""

    rows: list[dict[str, float | int]] = []
    upper_k = min(max_k, values.shape[0] - 1)
    for k in range(2, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            labels = model.fit_predict(values)
            silhouette = float(silhouette_score(values, labels))
            calinski_harabasz = float(calinski_harabasz_score(values, labels))
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": silhouette,
                "calinski_harabasz": calinski_harabasz,
            }
        )
    return pd.DataFrame(rows)


def _choose_k(evaluation: pd.DataFrame) -> int:
    """Alege numărul final de clustere pe baza Silhouette Score."""

    if evaluation.empty:
        return 2
    best_row = evaluation.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(best_row["k"])


def _cluster_profiles(countries: pd.DataFrame, standardized: pd.DataFrame, selected_variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculează profilurile clusterelor în valori originale și standardizate."""

    original_rows: list[dict[str, object]] = []
    for cluster_id, group in countries.sort_values([COUNTRY_FIELD]).groupby("cluster_id", sort=True):
        row: dict[str, object] = {
            "Cluster": f"Cluster {int(cluster_id)}",
            "Nr. țări": int(group.shape[0]),
            "Țări": ", ".join(group[COUNTRY_FIELD].astype(str).sort_values().tolist()),
        }
        for slug in selected_variables:
            row[variable_label(slug)] = float(group[slug].mean())
        original_rows.append(row)

    standardized_profile = (
        standardized.groupby("Cluster", as_index=False)[selected_variables]
        .mean(numeric_only=True)
        .sort_values("Cluster")
        .reset_index(drop=True)
    )
    standardized_profile = standardized_profile.rename(columns={slug: variable_label(slug) for slug in selected_variables})
    return pd.DataFrame(original_rows), standardized_profile


def _pca_results(standardized: pd.DataFrame, selected_variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    """Calculează PCA pe variabilele standardizate folosite în clustering."""

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    values = standardized[selected_variables].to_numpy(dtype=float)
    scores = pca.fit_transform(values)
    pca_scores = standardized[[ISO3_FIELD, COUNTRY_FIELD, "cluster_id", "Cluster"]].copy()
    pca_scores["PC1"] = scores[:, 0]
    pca_scores["PC2"] = scores[:, 1]

    loadings = pd.DataFrame(
        {
            "Variabilă": [variable_label(slug) for slug in selected_variables],
            "Loading PC1": pca.components_[0],
            "Loading PC2": pca.components_[1],
        }
    )
    loadings["_ordine"] = loadings[["Loading PC1", "Loading PC2"]].abs().max(axis=1)
    loadings = loadings.sort_values("_ordine", ascending=False).drop(columns="_ordine").reset_index(drop=True)
    explained = (float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1]))
    return pca_scores, loadings, explained


def prepare_clustering_analysis(dataframe: pd.DataFrame | None) -> ClusteringAnalysisResult:
    """Pregătește clustering-ul K-Means, dendrograma și PCA."""

    country_means = country_level_cluster_candidates(dataframe)
    selected_variables, notes = _select_variables(country_means)
    complete = country_means.dropna(subset=selected_variables).copy()
    excluded_countries = sorted(set(country_means[COUNTRY_FIELD]) - set(complete[COUNTRY_FIELD]))
    if excluded_countries:
        notes.append(f"Țări excluse din clustering din cauza valorilor lipsă în variabilele finale: {', '.join(excluded_countries)}.")
    notes.append("Nu s-au imputat valori; mediile pe țară sunt calculate doar din observațiile disponibile.")

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(complete[selected_variables].to_numpy(dtype=float))
    standardized = complete[[ISO3_FIELD, COUNTRY_FIELD]].copy()
    standardized[selected_variables] = scaled_values

    evaluation = _evaluate_kmeans(scaled_values)
    optimal_k = _choose_k(evaluation)
    final_model = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        labels = final_model.fit_predict(scaled_values) + 1

    complete["cluster_id"] = labels
    complete["Cluster"] = [f"Cluster {int(label)}" for label in labels]
    standardized["cluster_id"] = labels
    standardized["Cluster"] = [f"Cluster {int(label)}" for label in labels]
    complete = complete.sort_values(["cluster_id", COUNTRY_FIELD]).reset_index(drop=True)
    standardized = standardized.sort_values(["cluster_id", COUNTRY_FIELD]).reset_index(drop=True)

    profiles_original, profiles_standardized = _cluster_profiles(complete, standardized, selected_variables)
    pca_scores, pca_loadings, explained = _pca_results(standardized, selected_variables)
    hierarchy = linkage(standardized[selected_variables].to_numpy(dtype=float), method="ward", metric="euclidean")

    notes.append(f"Numărul final de clustere este selectat prin valoarea maximă Silhouette Score; soluția rămâne exploratorie.")
    return ClusteringAnalysisResult(
        countries=complete,
        standardized_countries=standardized,
        selected_variables=selected_variables,
        selection_notes=notes,
        evaluation=evaluation,
        optimal_k=optimal_k,
        profiles_original=profiles_original,
        profiles_standardized=profiles_standardized,
        pca_scores=pca_scores,
        pca_loadings=pca_loadings,
        explained_variance_ratio=explained,
        linkage_matrix=hierarchy,
    )


def _theme_figure(fig: go.Figure, title: str, height: int) -> go.Figure:
    """Aplică tema vizuală a proiectului unei figuri Plotly."""

    fig.update_layout(
        title={"text": title or ""},
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, Source Sans Pro, Arial, sans-serif", "color": PALETTE["ink"]},
        title_font={"family": "Georgia, Times New Roman, serif", "color": PALETTE["navy"], "size": 21},
        margin={"l": 42, "r": 22, "t": 62, "b": 46},
        hoverlabel={"bgcolor": PALETTE["navy"], "font_color": "#FFFFFF"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"color": PALETTE["ink"]}},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(7,26,47,0.08)", zeroline=False, tickfont={"color": PALETTE["ink"]}, title_font={"color": PALETTE["navy"]})
    fig.update_yaxes(showgrid=True, gridcolor="rgba(7,26,47,0.08)", zeroline=False, tickfont={"color": PALETTE["ink"]}, title_font={"color": PALETTE["navy"]})
    return fig


def elbow_method_figure(evaluation: pd.DataFrame) -> go.Figure:
    """Construiește graficul Elbow Method."""

    fig = go.Figure(
        go.Scatter(
            x=evaluation["k"].tolist(),
            y=evaluation["inertia"].tolist(),
            mode="lines+markers",
            line={"color": PALETTE["navy"], "width": 3},
            marker={"color": PALETTE["gold"], "size": 9},
            hovertemplate="k=%{x}<br>Inertia: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Număr clustere")
    fig.update_yaxes(title="Inertia")
    return _theme_figure(fig, "Metoda Elbow", 330)


def silhouette_score_figure(evaluation: pd.DataFrame) -> go.Figure:
    """Construiește graficul Silhouette Score."""

    fig = go.Figure(
        go.Scatter(
            x=evaluation["k"].tolist(),
            y=evaluation["silhouette"].tolist(),
            mode="lines+markers",
            line={"color": PALETTE["teal"], "width": 3},
            marker={"color": PALETTE["burgundy"], "size": 9},
            hovertemplate="k=%{x}<br>Silhouette Score: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Număr clustere")
    fig.update_yaxes(title="Silhouette Score")
    return _theme_figure(fig, "Silhouette Score", 330)


def _cluster_hover(row: pd.Series) -> str:
    """Construiește hover-ul pentru harta clusterelor."""

    fields = [
        (DEPENDENT_SLUG, "Participarea medie", "%"),
        ("gdp_pc_ppp", "PIB/locuitor mediu", ""),
        ("fertility", "Fertilitatea medie", ""),
        ("internet_use", "Internet mediu", "%"),
        ("control_corruption", "Controlul corupției mediu", ""),
    ]
    lines = [f"<b>{row[COUNTRY_FIELD]}</b>", str(row["Cluster"])]
    for slug, label, suffix in fields:
        if slug not in row:
            continue
        value = row[slug]
        if pd.notna(value):
            lines.append(f"{label}: {float(value):,.2f}{suffix}")
    return "<br>".join(lines)


def cluster_map_figure(result: ClusteringAnalysisResult, geojson: dict[str, Any]) -> go.Figure:
    """Construiește harta clusterelor K-Means."""

    plotting = result.countries.copy()
    plotting["hover"] = plotting.apply(_cluster_hover, axis=1)
    fig = go.Figure()
    for cluster_id, group in plotting.groupby("cluster_id", sort=True):
        color = CLUSTER_COLORS[(int(cluster_id) - 1) % len(CLUSTER_COLORS)]
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                featureidkey=f"properties.{GEOMETRY_ISO_COLUMN}",
                locations=group[ISO3_FIELD].astype(str).tolist(),
                z=[1] * len(group),
                zmin=0,
                zmax=1,
                colorscale=[[0.0, color], [1.0, color]],
                marker_line_color="#FFFFFF",
                marker_line_width=0.6,
                showscale=False,
                showlegend=True,
                name=f"Cluster {int(cluster_id)}",
                text=group["hover"].tolist(),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(255,255,255,0)",
        showcountries=True,
        countrycolor="#FFFFFF",
        countrywidth=0.6,
        lataxis={"range": [34, 72]},
        lonaxis={"range": [-25, 45]},
    )
    fig.update_layout(
        title="Analiza cluster",
        height=500,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, Source Sans Pro, Arial, sans-serif", "color": PALETTE["ink"]},
        title_font={"family": "Georgia, Times New Roman, serif", "color": PALETTE["navy"], "size": 21},
        margin={"l": 8, "r": 8, "t": 56, "b": 8},
        hoverlabel={"bgcolor": PALETTE["navy"], "font_color": "#FFFFFF"},
        legend={"orientation": "h", "x": 0, "y": -0.02, "font": {"color": PALETTE["ink"]}},
    )
    return fig


def cluster_profile_heatmap(result: ClusteringAnalysisResult) -> go.Figure:
    """Construiește heatmap-ul profilurilor standardizate pe clustere."""

    profile = result.profiles_standardized.set_index("Cluster")
    values = profile.to_numpy(dtype=float)
    text = np.empty(values.shape, dtype=object)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            text[row_index, column_index] = f"{values[row_index, column_index]:.2f}"
    max_abs = float(np.nanmax(np.abs(values))) if values.size else 1.0
    max_abs = max(max_abs, 1.0)
    fig = go.Figure(
        go.Heatmap(
            z=values.tolist(),
            x=profile.columns.tolist(),
            y=profile.index.tolist(),
            zmin=-max_abs,
            zmax=max_abs,
            zmid=0,
            colorscale=[
                [0.0, PALETTE["burgundy"]],
                [0.5, PALETTE["ivory"]],
                [1.0, PALETTE["teal"]],
            ],
            xgap=1,
            ygap=1,
            text=text.tolist(),
            texttemplate="%{text}",
            textfont={"size": 10, "color": PALETTE["ink"]},
            colorbar={"title": "Z-score"},
            hovertemplate="%{y}<br>%{x}<br>Z-score: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(tickangle=-30, tickfont={"size": 10, "color": PALETTE["ink"]})
    return _theme_figure(fig, "Profilul clusterelor pe variabile standardizate", 430)


def dendrogram_figure(result: ClusteringAnalysisResult) -> go.Figure:
    """Construiește dendrograma clustering-ului ierarhic Ward."""

    labels = result.standardized_countries.sort_values([COUNTRY_FIELD])[COUNTRY_FIELD].tolist()
    values = result.standardized_countries.sort_values([COUNTRY_FIELD])[result.selected_variables].to_numpy(dtype=float)
    hierarchy = linkage(values, method="ward", metric="euclidean")
    dendro = dendrogram(hierarchy, labels=labels, no_plot=True)
    colors = {
        "C0": PALETTE["navy"],
        "C1": PALETTE["teal"],
        "C2": PALETTE["burgundy"],
        "C3": PALETTE["gold"],
        "C4": PALETTE["plum"],
    }
    fig = go.Figure()
    for xs, ys, color_key in zip(dendro["icoord"], dendro["dcoord"], dendro["color_list"]):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": colors.get(color_key, PALETTE["muted"]), "width": 2},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    tickvals = [5 + 10 * index for index in range(len(dendro["ivl"]))]
    ticktext = [f"<b>{label}</b>" if label == "România" else label for label in dendro["ivl"]]
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=-45, title="")
    fig.update_yaxes(title="Distanță Ward")
    themed = _theme_figure(fig, "", 570)
    themed.update_layout(title={"text": ""}, margin={"l": 52, "r": 20, "t": 24, "b": 150})
    return themed


def pca_scatter_figure(result: ClusteringAnalysisResult) -> go.Figure:
    """Construiește scatter plot-ul PC1-PC2 colorat după cluster."""

    scores = result.pca_scores.copy()
    fig = go.Figure()
    for cluster_id, group in scores.groupby("cluster_id", sort=True):
        color = CLUSTER_COLORS[(int(cluster_id) - 1) % len(CLUSTER_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=group["PC1"].tolist(),
                y=group["PC2"].tolist(),
                mode="markers+text",
                text=group[COUNTRY_FIELD].tolist(),
                textposition="top center",
                marker={"color": color, "size": 11, "line": {"color": "#FFFFFF", "width": 1}},
                name=f"Cluster {int(cluster_id)}",
                customdata=group[[COUNTRY_FIELD, "Cluster"]].astype(str).to_numpy(),
                hovertemplate=(
                    "Țară: %{customdata[0]}<br>"
                    "%{customdata[1]}<br>"
                    "PC1: %{x:.2f}<br>"
                    "PC2: %{y:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_xaxes(title=f"PC1 ({result.explained_variance_ratio[0] * 100:.1f}%)")
    fig.update_yaxes(title=f"PC2 ({result.explained_variance_ratio[1] * 100:.1f}%)")
    themed = _theme_figure(fig, "", 520)
    themed.update_layout(title={"text": ""}, margin={"l": 42, "r": 22, "t": 24, "b": 46})
    return themed


def format_cluster_profile_table(result: ClusteringAnalysisResult) -> pd.DataFrame:
    """Formatează profilul clusterelor pentru afișare compactă."""

    display = result.profiles_original.copy()
    numeric_columns = [column for column in display.columns if column not in {"Cluster", "Nr. țări", "Țări"}]
    for column in numeric_columns:
        display[column] = display[column].map(lambda value: f"{float(value):,.2f}" if pd.notna(value) else "—")
    return display


def format_pca_loadings_table(result: ClusteringAnalysisResult) -> pd.DataFrame:
    """Formatează tabelul loadings PCA pentru PC1 și PC2."""

    display = result.pca_loadings.copy()
    display["Loading PC1"] = display["Loading PC1"].map(lambda value: f"{float(value):.3f}")
    display["Loading PC2"] = display["Loading PC2"].map(lambda value: f"{float(value):.3f}")
    return display
