"""Pagina pentru analiza transversală."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from components.html import render_html
from components.empty_states import render_empty_state, render_info_panel, render_section_header, render_warning_box
from components.metric_cards import render_metric_grid
from components.sidebar import configure_page, inject_global_css, render_sidebar
from components.tables import render_light_table
from src.r_analysis_bridge import CrossSectionalResults, load_cross_sectional_results


configure_page("Analiza transversală")
inject_global_css()
render_sidebar("Analiza transversală")


def _format_value(value: object, digits: int = 3) -> str:
    """Formatează strict pentru afișare valorile deja calculate în R."""

    if value == "" or pd.isna(value):
        return "—"
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    if abs(numeric) >= 100:
        return f"{numeric:,.1f}"
    return f"{numeric:.{digits}f}"


def _format_p(value: object) -> str:
    """Formatează p-value pentru afișare."""

    if value == "" or pd.isna(value):
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.4f}"


def _clean_text(value: object) -> str:
    """Curăță o valoare textuală înainte de randarea în carduri."""

    if value == "" or pd.isna(value):
        return "—"
    return str(value)


def _metric(table: pd.DataFrame, indicator: str, digits: int = 3) -> str:
    """Extrage o metrică dintr-un tabel produs de R."""

    if table.empty or "Indicator" not in table.columns or "Valoare" not in table.columns:
        return "—"
    row = table.loc[table["Indicator"].astype(str).eq(indicator)]
    if row.empty:
        return "—"
    return _format_value(row.iloc[0]["Valoare"], digits=digits)


def _coefficient(table: pd.DataFrame, term: str, column: str, digits: int = 3) -> str:
    """Extrage o valoare din tabelul de coeficienți produs de R."""

    if table.empty or "Termen" not in table.columns or column not in table.columns:
        return "—"
    row = table.loc[table["Termen"].astype(str).eq(term)]
    if row.empty:
        return "—"
    if column == "p-value":
        return _format_p(row.iloc[0][column])
    return _format_value(row.iloc[0][column], digits=digits)


def _display_table(dataframe: pd.DataFrame, height: int = 360) -> None:
    """Afișează un tabel compact, cu valorile lipsă curățate."""

    if dataframe.empty:
        render_empty_state("Tabel indisponibil", "Analiza R nu a returnat rânduri pentru această componentă.")
        return
    display = dataframe.copy()
    for column in display.columns:
        if column == "p-value":
            display[column] = display[column].map(_format_p)
        elif pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(_format_value)
        else:
            display[column] = display[column].map(lambda value: "—" if value == "" or pd.isna(value) else str(value))
    render_light_table(display, height=height)


def _display_tests(dataframe: pd.DataFrame) -> None:
    """Afișează testele econometrice în carduri, nu în tabel."""

    if dataframe.empty:
        render_empty_state("Teste indisponibile", "Analiza R nu a returnat teste pentru această componentă.")
        return

    h0_column = "Ipoteza nulă" if "Ipoteza nulă" in dataframe.columns else "H0"
    conclusion_column = "Concluzie" if "Concluzie" in dataframe.columns else "Interpretare"
    cards: list[str] = []
    for _, row in dataframe.iterrows():
        p_value = row.get("p-value", "")
        try:
            numeric_p = float(p_value)
        except (TypeError, ValueError):
            numeric_p = None
        tone = "neutral"
        if numeric_p is not None and pd.notna(numeric_p):
            tone = "alert" if numeric_p < 0.05 else "ok"

        category_html = ""
        if "Categorie" in dataframe.columns and _clean_text(row.get("Categorie", "")) != "—":
            category_html = f'<span class="test-card-category">{escape(_clean_text(row.get("Categorie")))}</span>'

        cards.append(
            f"""
            <article class="test-card tone-{tone}">
                <div class="test-card-top">
                    <strong>{escape(_clean_text(row.get("Test", "Test")))}</strong>
                    {category_html}
                </div>
                <p class="test-card-h0">{escape(_clean_text(row.get(h0_column, "—")))}</p>
                <div class="test-card-stats">
                    <span>Statistică <b>{escape(_format_value(row.get("Statistică", "")))}</b></span>
                    <span>p-value <b>{escape(_format_p(row.get("p-value", "")))}</b></span>
                </div>
                <p class="test-card-conclusion">{escape(_clean_text(row.get(conclusion_column, "—")))}</p>
            </article>
            """
        )

    render_html(
        f"""
        <div class="test-card-grid">
            {''.join(cards)}
        </div>
        """
    )


def _figure(results: CrossSectionalResults, name: str) -> Path | None:
    """Returnează calea unei figuri produse de R."""

    return results.figures.get(name)


def _show_figure(results: CrossSectionalResults, name: str, caption: str | None = None) -> None:
    """Afișează o figură PNG sau o stare goală dacă lipsește."""

    path = _figure(results, name)
    if path is None:
        render_empty_state("Grafic indisponibil", f"Figura {name} nu a fost generată de scriptul R.")
        return
    st.image(str(path), caption=caption, use_column_width=True)


def _show_figure_grid(results: CrossSectionalResults, names: Iterable[str], columns: int = 2) -> None:
    """Afișează graficele într-o grilă stabilă."""

    names = list(names)
    for start in range(0, len(names), columns):
        cols = st.columns(columns)
        for index, figure_name in enumerate(names[start : start + columns]):
            with cols[index]:
                _show_figure(results, figure_name)


@st.cache_data(show_spinner=False)
def _load_results() -> CrossSectionalResults:
    """Încarcă rezultatele R cu reutilizarea outputurilor existente."""

    return load_cross_sectional_results(force=False)


render_html(
    """
    <section class="cross-page-title">
        <h1>Analiza transversală</h1>
    </section>
    """
)

try:
    results = _load_results()
except RuntimeError as error:
    render_warning_box(str(error), title="Analiza transversală nu poate fi încărcată")
    st.stop()

metadata = results.metadata
tables = results.tables
conclusions = metadata.get("concluzii", {})

tabs = st.tabs(
    [
        "Regresie simplă",
        "Regresie multiplă",
        "Diagnostic și validare",
        "Extensii și predicții",
        "Ridge & LASSO",
    ]
)

with tabs[0]:
    render_section_header("Regresia simplă", "Model: Participarea femeilor ~ Inflație")

    simple_coefficients = tables["simple_coefficients.csv"]
    simple_metrics = tables["simple_model_metrics.csv"]
    render_metric_grid(
        [
            {"label": "R²", "value": _metric(simple_metrics, "R²"), "accent": "teal"},
            {"label": "R² ajustat", "value": _metric(simple_metrics, "R² ajustat"), "accent": "gold"},
            {"label": "Coeficient Inflație", "value": _coefficient(simple_coefficients, "Inflație", "Coeficient"), "accent": "burgundy"},
            {"label": "p-value", "value": _coefficient(simple_coefficients, "Inflație", "p-value"), "accent": "plum"},
            {"label": "N", "value": _metric(simple_metrics, "N", digits=0), "accent": "navy"},
        ],
        columns=5,
    )

    scatter_col, conclusion_col = st.columns([0.66, 0.34])
    with scatter_col:
        _show_figure(results, "simple_scatter_regression.png")
    with conclusion_col:
        render_info_panel("Concluzie", conclusions.get("model_simplu", ""), accent="burgundy")

    render_section_header("Coeficienții modelului simplu")
    _display_table(simple_coefficients, height=160)

    render_section_header("Grafice de diagnostic")
    _show_figure_grid(
        results,
        [
            "simple_qq_plot.png",
            "simple_residual_histogram.png",
            "simple_residual_boxplot.png",
            "simple_residuals_predictor.png",
        ],
        columns=2,
    )

    render_section_header("Teste pentru modelul simplu")
    _display_tests(tables["simple_tests.csv"])

    render_section_header("Predicții pentru niveluri reprezentative ale inflației")
    _display_table(tables["simple_predictions.csv"], height=260)

with tabs[1]:
    render_section_header("Modelul de regresie multiplă")

    multiple_metrics = tables["multiple_model_metrics.csv"]
    render_metric_grid(
        [
            {"label": "N", "value": _metric(multiple_metrics, "N", digits=0), "accent": "navy"},
            {"label": "R²", "value": _metric(multiple_metrics, "R²"), "accent": "teal"},
            {"label": "R² ajustat", "value": _metric(multiple_metrics, "R² ajustat"), "accent": "gold"},
            {"label": "AIC", "value": _metric(multiple_metrics, "AIC"), "accent": "burgundy"},
            {"label": "BIC", "value": _metric(multiple_metrics, "BIC"), "accent": "plum"},
        ],
        columns=5,
    )
    _show_figure(results, "multiple_coefficient_plot.png")

    render_section_header("Coeficienții modelului multiplu")
    _display_table(tables["multiple_coefficients.csv"], height=430)

    render_section_header("Matricea de corelație Pearson")
    _show_figure(results, "multiple_correlation_heatmap.png")

with tabs[2]:
    render_section_header("Diagnostic și validarea ipotezelor modelului")

    multiple_tests = tables["multiple_tests.csv"]
    for category in [
        "Heteroscedasticitate",
        "Normalitate",
        "Autocorelare",
        "Multicoliniaritate",
        "Specificare",
        "Forma reziduurilor",
    ]:
        subset = multiple_tests.loc[multiple_tests["Categorie"].astype(str).eq(category)] if not multiple_tests.empty else pd.DataFrame()
        if subset.empty:
            continue
        render_section_header(category)
        _display_tests(subset)

    render_section_header("Multicoliniaritate")
    vif_col, vif_plot_col = st.columns([0.44, 0.56])
    with vif_col:
        _display_table(tables["multiple_vif.csv"], height=430)
    with vif_plot_col:
        _show_figure(results, "multiple_vif_bar.png")

    render_section_header("Grafice de diagnostic")
    _show_figure_grid(
        results,
        [
            "multiple_qq_plot.png",
            "multiple_residual_histogram.png",
            "multiple_scale_location.png",
            "multiple_leverage_studentized.png",
            "multiple_observed_fitted.png",
        ],
        columns=2,
    )

with tabs[3]:
    render_section_header("Extensii ale modelului și predicții")

    render_section_header("Prag de digitalizare")
    render_metric_grid(
        [
            {"label": "Prag median", "value": _format_value(metadata.get("prag_digitalizare")), "caption": "utilizarea internetului", "accent": "teal"},
            {"label": "An analizat", "value": str(metadata.get("anul_analizat", 2023)), "caption": "secțiune transversală", "accent": "gold"},
        ],
        columns=2,
    )
    render_info_panel(
        "Interpretare",
        "Variabila dummy explorează existența unui posibil efect de prag al digitalizării și nu reprezintă o clasificare oficială a economiilor.",
        accent="teal",
    )
    _display_table(tables["dummy_counts.csv"], height=150)
    _display_table(tables["dummy_coefficients.csv"], height=360)

    render_section_header("Teste pentru modelul cu dummy")
    _display_tests(tables["dummy_tests.csv"])

    render_section_header("Interacțiunea Internet × Educație")
    render_info_panel(
        "Ipoteză exploratorie",
        "Asocierea digitalizării cu participarea femeilor poate depinde de nivelul capitalului uman feminin.",
        accent="gold",
    )
    inter1_col, inter1_table_col = st.columns([0.6, 0.4])
    with inter1_col:
        _show_figure(results, "interaction1_plot.png")
    with inter1_table_col:
        _display_table(tables["interaction1_coefficients.csv"], height=430)

    render_section_header("Interacțiunea PIB × Controlul corupției")
    render_info_panel(
        "Ipoteză exploratorie",
        "Asocierea nivelului de dezvoltare economică cu participarea femeilor poate varia în funcție de calitatea instituțională.",
        accent="burgundy",
    )
    inter2_col, inter2_table_col = st.columns([0.6, 0.4])
    with inter2_col:
        _show_figure(results, "interaction2_plot.png")
    with inter2_table_col:
        _display_table(tables["interaction2_coefficients.csv"], height=360)

    render_section_header("Relația neliniară cu PIB")
    _display_table(tables["nonlinear_gdp_coefficients.csv"], height=320)

    render_section_header("Predicții pe scenarii data-driven")
    scenario_col, scenario_table_col = st.columns([0.58, 0.42])
    with scenario_col:
        _show_figure(results, "scenario_predictions.png")
    with scenario_table_col:
        _display_table(tables["scenario_predictions.csv"], height=250)

with tabs[4]:
    render_section_header(
        "Regularizare și comparația modelelor",
        "Ridge și LASSO sunt utilizate ca instrumente predictive și de regularizare, nu ca substitut automat pentru interpretarea econometrică.",
    )

    render_metric_grid(
        [
            {"label": "Train", "value": str(metadata.get("train_n", "—")), "caption": "80% din eșantion", "accent": "teal"},
            {"label": "Test", "value": str(metadata.get("test_n", "—")), "caption": "20% din eșantion", "accent": "gold"},
            {"label": "Lambda Ridge", "value": _format_value(metadata.get("lambda_ridge", {}).get("lambda_min")), "caption": "lambda.min", "accent": "plum"},
            {"label": "Lambda LASSO", "value": _format_value(metadata.get("lambda_lasso", {}).get("lambda_min")), "caption": "lambda.min", "accent": "burgundy"},
        ],
        columns=4,
    )

    render_section_header("Tabel comparativ final")
    render_info_panel("Interpretare", "RMSE mai mic indică o eroare predictivă mai redusă pe setul de test.", accent="teal")
    _display_table(tables["ml_metrics.csv"], height=180)
    render_info_panel("Concluzie", conclusions.get("ridge_lasso", ""), accent="burgundy")

    render_section_header("Validare încrucișată și trasee de coeficienți")
    _show_figure_grid(
        results,
        [
            "ridge_cv_error.png",
            "lasso_cv_error.png",
            "ridge_coefficient_paths.png",
            "lasso_coefficient_paths.png",
        ],
        columns=2,
    )

    render_section_header("Valori estimate vs observate")
    _show_figure_grid(
        results,
        [
            "ols_predicted_actual.png",
            "ridge_predicted_actual.png",
            "lasso_predicted_actual.png",
        ],
        columns=3,
    )

    render_section_header("Compararea performanței")
    _show_figure_grid(results, ["ml_rmse_comparison.png", "ml_mae_comparison.png"], columns=2)

    render_section_header("Coeficienții LASSO nenuli")
    lasso_col, lasso_table_col = st.columns([0.56, 0.44])
    with lasso_col:
        _show_figure(results, "lasso_nonzero_coefficients.png")
    with lasso_table_col:
        render_info_panel(
            "Interpretare",
            "Variabilele păstrate de LASSO în această partiționare au contribuție predictivă în modelul regularizat.",
            accent="gold",
        )
        _display_table(tables["lasso_nonzero_coefficients.csv"], height=320)
