"""Pagina pentru analiza descriptivă."""

from __future__ import annotations

from html import escape
from math import isfinite

import pandas as pd
import streamlit as st

from components.empty_states import render_section_header
from components.html import render_html
from components.sidebar import configure_page, inject_global_css, render_sidebar
from components.tables import render_light_table
from src.correlation_analysis import (
    SCATTER_RELATIONSHIPS,
    correlations_with_dependent,
    labelled_correlation_matrix,
    prepare_correlation_dataset,
)
from src.clustering_analysis import (
    cluster_map_figure,
    dendrogram_figure,
    elbow_method_figure,
    format_cluster_profile_table,
    format_pca_loadings_table,
    pca_scatter_figure,
    prepare_clustering_analysis,
    silhouette_score_figure,
)
from src.descriptive_charts import (
    correlation_heatmap,
    correlation_scatter_plot,
    extremes_2023_chart,
    general_box_plot,
    histogram_with_density,
    kurtosis_chart,
    missing_by_country_bar,
    missing_by_variable_bar,
    representative_years_box_plot,
    skewness_chart,
)
from src.descriptive_statistics import (
    dependent_variable_sample,
    dependent_variable_statistics,
    format_number,
    format_percent,
    interpret_kurtosis,
    interpret_skewness,
)
from src.data_loader import get_session_dataset
from src.geographic_analysis import (
    participation_map,
    prepare_geographic_analysis,
)
from src.missing_values_analysis import (
    general_missing_summary,
    missing_by_country,
    missing_by_variable,
    prepare_missing_dataset,
)


CARD_COLORS = {
    "teal": "#0E6A68",
    "gold": "#C99A45",
    "navy": "#071A2F",
    "plum": "#6E2447",
    "burgundy": "#8A334E",
    "muted": "#687083",
}


def _render_stat_card(label: str, value: str, accent: str, caption: str | None = None) -> None:
    """Afișează un card statistic compact."""

    color = CARD_COLORS.get(accent, CARD_COLORS["gold"])
    caption_html = f"<p>{escape(caption)}</p>" if caption else ""
    render_html(
        f"""
        <div class="metric-card" style="border-top-color: {color}; min-height: 118px;">
            <span class="metric-label">{escape(label)}</span>
            <strong>{escape(value)}</strong>
            {caption_html}
        </div>
        """
    )


def _render_card_row(cards: list[dict[str, str]], columns: int) -> None:
    """Afișează un rând de carduri statistice."""

    cols = st.columns(columns)
    for col, card in zip(cols, cards):
        with col:
            _render_stat_card(
                card["label"],
                card["value"],
                card["accent"],
                card.get("caption"),
            )


def _render_centered_two_card_row(cards: list[dict[str, str]]) -> None:
    """Afișează două carduri centrate pe același rând."""

    cols = st.columns([1, 2, 2, 1])
    for col, card in zip(cols[1:3], cards):
        with col:
            _render_stat_card(
                card["label"],
                card["value"],
                card["accent"],
                card.get("caption"),
            )


def _render_note_card(title: str, body: str | None = None, accent: str = "gold", items: list[str] | None = None) -> None:
    """Afișează un card metodologic compact."""

    color = CARD_COLORS.get(accent, CARD_COLORS["gold"])
    body_html = f"<p>{escape(body)}</p>" if body else ""
    items_html = ""
    if items:
        items_html = "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"
    render_html(
        f"""
        <div class="info-panel" style="border-top-color: {color}; min-height: 0;">
            <h3>{escape(title)}</h3>
            {body_html}
            {items_html}
        </div>
        """
    )


def _format_float(value: float | int | None, decimals: int = 2) -> str:
    """Formatează un coeficient numeric."""

    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not isfinite(numeric):
        return "—"
    return f"{numeric:.{decimals}f}"


def _format_p_value(value: float | int | None) -> str:
    """Formatează p-value pentru tabele compacte."""

    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not isfinite(numeric):
        return "—"
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.3f}"


def _format_dependent_correlation_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Pregătește tabelul corelațiilor cu variabila dependentă."""

    display = dataframe.copy()
    display["Pearson r"] = display["Pearson r"].map(_format_float)
    display["p-value Pearson"] = display["p-value Pearson"].map(_format_p_value)
    display["Spearman ρ"] = display["Spearman ρ"].map(_format_float)
    display["p-value Spearman"] = display["p-value Spearman"].map(_format_p_value)
    display["N"] = display["N"].astype(int).astype(str)
    return display.rename(columns={"p-value Pearson": "p-value (Pearson)", "p-value Spearman": "p-value (Spearman)"})


configure_page("Analiza descriptivă")
inject_global_css()
render_sidebar("Analiza descriptivă")

render_section_header("Analiza descriptivă")

dataframe = get_session_dataset()
sample = dependent_variable_sample(dataframe)
stats = dependent_variable_statistics(sample)
correlation_dataset = prepare_correlation_dataset(dataframe)
missing_dataset = prepare_missing_dataset(dataframe)

tabs = st.tabs(
    [
        "Statistici descriptive",
        "Distribuții și valori extreme",
        "Valori lipsă",
        "Corelații",
        "Hartă europeană",
    ]
)

with tabs[0]:
    render_section_header("Statistici descriptive")

    render_section_header("Indicatorii tendinței centrale")
    _render_centered_two_card_row(
        [
            {"label": "Media", "value": format_number(stats["mean"]), "accent": "teal"},
            {"label": "Mediana", "value": format_number(stats["median"]), "accent": "teal"},
        ],
    )

    render_section_header("Indicatorii dispersiei")
    _render_card_row(
        [
            {"label": "Varianță", "value": format_number(stats["variance"]), "accent": "gold"},
            {"label": "Deviație standard", "value": format_number(stats["std"]), "accent": "gold"},
            {"label": "Amplitudine", "value": format_number(stats["range"]), "accent": "gold"},
            {"label": "Interval interquartilic – IQR", "value": format_number(stats["iqr"]), "accent": "gold"},
            {"label": "Coeficient de variație", "value": stats["cv"].label, "accent": "gold"},
        ],
        columns=5,
    )

    render_section_header("Indicatorii poziției")
    _render_card_row(
        [
            {"label": "Minim", "value": format_number(stats["min"]), "accent": "navy"},
            {"label": "Q1", "value": format_number(stats["q1"]), "accent": "navy"},
            {"label": "Mediană / Q2", "value": format_number(stats["median"]), "accent": "navy"},
            {"label": "Q3", "value": format_number(stats["q3"]), "accent": "navy"},
            {"label": "Maxim", "value": format_number(stats["max"]), "accent": "navy"},
        ],
        columns=5,
    )

    render_section_header("Forma distribuției")
    _render_card_row(
        [
            {
                "label": "Skewness",
                "value": format_number(stats["skewness"]),
                "accent": "burgundy",
                "caption": interpret_skewness(stats["skewness"]),
            },
            {
                "label": "KURTOSIS",
                "value": format_number(stats["kurtosis"]),
                "accent": "plum",
                "caption": interpret_kurtosis(stats["kurtosis"]),
            },
        ],
        columns=2,
    )
    shape_left, shape_right = st.columns(2)
    with shape_left:
        st.plotly_chart(skewness_chart(sample, stats), use_container_width=True)
    with shape_right:
        st.plotly_chart(kurtosis_chart(sample, stats), use_container_width=True)

    render_section_header("Calitatea observațiilor")
    _render_card_row(
        [
            {"label": "Observații valide", "value": format_number(stats["valid_count"], decimals=0), "accent": "muted"},
            {"label": "Valori lipsă", "value": format_number(stats["missing_count"], decimals=0), "accent": "muted"},
            {"label": "Procent valori lipsă", "value": format_percent(stats["missing_percent"]), "accent": "muted"},
        ],
        columns=3,
    )

with tabs[1]:
    render_section_header("Box plot")

    row_one_left, row_one_right = st.columns([0.64, 0.36])
    with row_one_left:
        st.plotly_chart(histogram_with_density(sample, stats), use_container_width=True)

    with row_one_right:
        st.plotly_chart(general_box_plot(sample), use_container_width=True)

    st.plotly_chart(representative_years_box_plot(sample), use_container_width=True)

    st.plotly_chart(extremes_2023_chart(sample), use_container_width=True)

with tabs[2]:
    render_section_header("Valori lipsă")

    missing_summary = general_missing_summary(missing_dataset)
    variable_missing = missing_by_variable(missing_dataset)
    country_missing = missing_by_country(missing_dataset)

    _render_card_row(
        [
            {"label": "Număr total de celule analizate", "value": format_number(missing_summary["total_cells"], decimals=0), "accent": "navy"},
            {"label": "Valori disponibile", "value": format_number(missing_summary["available"], decimals=0), "accent": "teal"},
            {"label": "Valori lipsă", "value": format_number(missing_summary["missing"], decimals=0), "accent": "burgundy"},
            {"label": "Procent total valori lipsă", "value": format_percent(missing_summary["missing_percent"]), "accent": "gold"},
        ],
        columns=4,
    )

    missing_row = st.columns(2)
    with missing_row[0]:
        st.plotly_chart(missing_by_variable_bar(variable_missing), use_container_width=True)
    with missing_row[1]:
        st.plotly_chart(missing_by_country_bar(country_missing), use_container_width=True)

with tabs[3]:
    render_section_header("Corelații")

    pearson_matrix = labelled_correlation_matrix(correlation_dataset, method="pearson")
    spearman_matrix = labelled_correlation_matrix(correlation_dataset, method="spearman")

    render_section_header("Corelația Pearson", "Măsoară intensitatea relațiilor liniare dintre variabile.")
    st.plotly_chart(correlation_heatmap(pearson_matrix, "Corelația Pearson"), use_container_width=True)

    render_section_header("Corelația Spearman", "Surprinde relații monotone și este mai puțin sensibilă la valori extreme.")
    st.plotly_chart(correlation_heatmap(spearman_matrix, "Corelația Spearman"), use_container_width=True)

    render_section_header("Relația cu variabila dependentă")
    dependent_correlations = correlations_with_dependent(correlation_dataset)
    render_light_table(
        _format_dependent_correlation_table(dependent_correlations),
        height=410,
    )

    render_section_header("Scatter plot-uri prestabilite")
    scatter_rows = [st.columns(2), st.columns(2)]
    for index, relationship in enumerate(SCATTER_RELATIONSHIPS):
        with scatter_rows[index // 2][index % 2]:
            st.plotly_chart(
                correlation_scatter_plot(correlation_dataset, relationship["x"], relationship["title"]),
                use_container_width=True,
            )

with tabs[4]:
    geographic_result = prepare_geographic_analysis(dataframe)
    clustering_result = prepare_clustering_analysis(dataframe)

    render_section_header("Participarea femeilor pe piața muncii-2023")
    st.plotly_chart(
        participation_map(
            geographic_result.geojson,
            geographic_result.annual_2023,
            "",
            geographic_result.shared_min,
            geographic_result.shared_max,
        ),
        use_container_width=True,
    )
    render_section_header("Evoluția participării feminine în Europa")
    comparison_columns = st.columns(2)
    with comparison_columns[0]:
        st.plotly_chart(
            participation_map(
                geographic_result.geojson,
                geographic_result.annual_2001,
                "2001",
                geographic_result.shared_min,
                geographic_result.shared_max,
                height=430,
            ),
            use_container_width=True,
        )
    with comparison_columns[1]:
        st.plotly_chart(
            participation_map(
                geographic_result.geojson,
                geographic_result.annual_2023,
                "2023",
                geographic_result.shared_min,
                geographic_result.shared_max,
                height=430,
            ),
            use_container_width=True,
        )

    render_section_header("Analiza cluster")
    evaluation_columns = st.columns(2)
    with evaluation_columns[0]:
        st.plotly_chart(elbow_method_figure(clustering_result.evaluation), use_container_width=True)
    with evaluation_columns[1]:
        st.plotly_chart(silhouette_score_figure(clustering_result.evaluation), use_container_width=True)

    cluster_layout = st.columns([0.6, 0.4])
    with cluster_layout[0]:
        st.plotly_chart(cluster_map_figure(clustering_result, geographic_result.geojson), use_container_width=True)
    with cluster_layout[1]:
        render_light_table(format_cluster_profile_table(clustering_result), height=500)

    render_section_header("Dendrogramă")
    st.plotly_chart(dendrogram_figure(clustering_result), use_container_width=True)

    render_section_header("Proiecție PCA pe primele două componente principale")
    pca_columns = st.columns([0.65, 0.35])
    with pca_columns[0]:
        st.plotly_chart(pca_scatter_figure(clustering_result), use_container_width=True)
    with pca_columns[1]:
        render_light_table(format_pca_loadings_table(clustering_result), height=520)
