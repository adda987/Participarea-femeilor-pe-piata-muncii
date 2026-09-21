"""Pagina pentru analiza econometrică panel."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from components.empty_states import render_empty_state, render_info_panel, render_section_header, render_warning_box
from components.html import render_html
from components.metric_cards import render_metric_grid
from components.sidebar import configure_page, inject_global_css, render_sidebar
from components.tables import render_light_table
from src.r_panel_bridge import PanelResults, load_panel_results


PAGE_NAME = "Analiză econometrică panel"


configure_page(PAGE_NAME)
inject_global_css()
render_sidebar(PAGE_NAME)


def _format_value(value: object, digits: int = 3) -> str:
    """Formatează valorile calculate deja în R pentru afișare."""

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
    """Curăță o valoare textuală înainte de randarea HTML."""

    if value == "" or pd.isna(value):
        return "—"
    return str(value)


def _structure_value(table: pd.DataFrame, indicator: str) -> str:
    """Extrage o valoare din panel_structure.csv."""

    if table.empty:
        return "—"
    row = table.loc[table["Indicator"].astype(str).eq(indicator)]
    if row.empty:
        return "—"
    return _clean_text(row.iloc[0]["Valoare"])


def _model_metric(table: pd.DataFrame, model: str, column: str, digits: int = 3) -> str:
    """Extrage o metrică pentru un model panel."""

    if table.empty or "Model" not in table.columns or column not in table.columns:
        return "—"
    row = table.loc[table["Model"].astype(str).eq(model)]
    if row.empty:
        return "—"
    return _format_value(row.iloc[0][column], digits=digits)


def _display_table(dataframe: pd.DataFrame, height: int = 360) -> None:
    """Afișează tabelele R într-o formă curată."""

    if dataframe.empty:
        render_empty_state("Tabel indisponibil", "Scriptul R nu a returnat rânduri pentru această componentă.")
        return
    display = dataframe.copy()
    for column in display.columns:
        if column == "p-value" or column.endswith(" p") or column.startswith("p "):
            display[column] = display[column].map(_format_p)
        elif pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(_format_value)
        else:
            display[column] = display[column].map(_clean_text)
    render_light_table(display, height=height)


def _display_tests(dataframe: pd.DataFrame) -> None:
    """Afișează testele panel în carduri."""

    if dataframe.empty:
        render_empty_state("Teste indisponibile", "Scriptul R nu a returnat teste pentru această componentă.")
        return
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
                <p class="test-card-h0">{escape(_clean_text(row.get("Ipoteza nulă", "—")))}</p>
                <div class="test-card-stats">
                    <span>Statistică <b>{escape(_format_value(row.get("Statistică", "")))}</b></span>
                    <span>p-value <b>{escape(_format_p(row.get("p-value", "")))}</b></span>
                </div>
                <p class="test-card-conclusion">{escape(_clean_text(row.get("Interpretare", "—")))}</p>
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


def _show_figure(results: PanelResults, name: str, caption: str | None = None) -> None:
    """Afișează o figură produsă de R."""

    path: Path | None = results.figures.get(name)
    if path is None:
        render_empty_state("Grafic indisponibil", f"Figura {name} nu a fost generată de scriptul R.")
        return
    st.image(str(path), caption=caption, use_column_width=True)


def _show_figure_grid(results: PanelResults, names: Iterable[str], columns: int = 2) -> None:
    """Afișează mai multe figuri într-un grid."""

    names = list(names)
    for start in range(0, len(names), columns):
        cols = st.columns(columns)
        for index, figure_name in enumerate(names[start : start + columns]):
            with cols[index]:
                _show_figure(results, figure_name)


def _selected_rows(dataframe: pd.DataFrame, category: str) -> pd.DataFrame:
    """Filtrează testele după categorie pentru carduri."""

    if dataframe.empty or "Categorie" not in dataframe.columns:
        return pd.DataFrame()
    return dataframe.loc[dataframe["Categorie"].astype(str).eq(category)]


@st.cache_data(show_spinner=False)
def _load_results() -> PanelResults:
    """Încarcă rezultatele panel generate de R."""

    return load_panel_results(force=False)


render_html(
    """
    <section class="panel-page-title">
        <h1>Modele panel: diferențe între țări și evoluții în timp</h1>
        <p>Analiza exploatează simultan variația dintre economiile europene și evoluția fiecărei economii în perioada 2001–2023.</p>
    </section>
    """
)

try:
    results = _load_results()
except RuntimeError as error:
    render_warning_box(str(error), title="Analiza panel nu poate fi încărcată")
    st.stop()

metadata = results.metadata
tables = results.tables
notes = metadata.get("notes", {})

tabs = st.tabs(
    [
        "Structura panelului",
        "Heterogenitate",
        "Modele Pooled, FE și RE",
        "Selecția modelului",
        "Diagnostic panel",
        "Model robust și concluzii",
    ]
)

with tabs[0]:
    render_section_header("Structura bazei panel")
    structure = tables["panel_structure.csv"]
    render_metric_grid(
        [
            {"label": "Număr de țări", "value": _structure_value(structure, "Număr de țări"), "accent": "teal"},
            {"label": "Perioadă", "value": _structure_value(structure, "Perioadă"), "accent": "gold"},
            {"label": "Observații country-year", "value": _structure_value(structure, "Număr observații country-year"), "accent": "navy"},
            {"label": "Variabile candidate", "value": _structure_value(structure, "Număr variabile candidate"), "accent": "burgundy"},
            {"label": "Structură", "value": _structure_value(structure, "Panel echilibrat / neechilibrat"), "accent": "plum"},
        ],
        columns=5,
    )
    if not metadata.get("balanced", False):
        render_warning_box("Panelul este neechilibrat: numărul observațiilor temporale diferă între unele economii.")

    _show_figure(results, "panel_coverage_heatmap.png")

    render_section_header("Variația within și between")
    variation_col, table_col = st.columns([0.58, 0.42])
    with variation_col:
        _show_figure(results, "within_between_variation.png")
    with table_col:
        _display_table(tables["within_between_variation.csv"], height=520)

with tabs[1]:
    render_section_header("Heterogenitatea între economii și în timp")
    _show_figure(results, "country_mean_ci.png")
    _show_figure(results, "year_mean_ci.png")
    _show_figure(results, "female_lfpr_spaghetti.png")
    _show_figure(results, "representative_years_distribution.png")

with tabs[2]:
    render_section_header("Estimarea modelelor panel")
    model_metrics = tables["panel_model_metrics.csv"]
    render_metric_grid(
        [
            {"label": "Pooled OLS", "value": _model_metric(model_metrics, "Pooled OLS", "R² relevant"), "caption": "R²", "accent": "gold"},
            {"label": "Fixed Effects", "value": _model_metric(model_metrics, "Fixed Effects", "R² relevant"), "caption": "R² within", "accent": "teal"},
            {"label": "Random Effects", "value": _model_metric(model_metrics, "Random Effects", "R² relevant"), "caption": "R²", "accent": "burgundy"},
            {"label": "Two-Way FE", "value": _model_metric(model_metrics, "Two-Way Fixed Effects", "R² relevant"), "caption": "R² within", "accent": "plum"},
        ],
        columns=4,
    )
    note_cols = st.columns(2)
    with note_cols[0]:
        render_info_panel("Fixed Effects", notes.get("fe", ""), accent="teal")
    with note_cols[1]:
        render_info_panel("Random Effects", notes.get("re", ""), accent="burgundy")

    _show_figure(results, "pooled_fe_re_coefficient_plot.png")
    render_section_header("Tabel comparativ al coeficienților")
    _display_table(tables["model_comparison.csv"], height=430)

    render_section_header("Indicatori globali ai modelelor")
    _display_table(model_metrics, height=220)

    render_section_header("Observed vs Fitted")
    _show_figure_grid(results, ["observed_fitted_pooled.png", "observed_fitted_fe.png", "observed_fitted_re.png"], columns=3)

with tabs[3]:
    render_section_header("Selecția specificației panel")
    _display_tests(tables["selection_tests.csv"])
    selected = tables["selected_model.csv"]
    if not selected.empty:
        render_info_panel("Specificația favorizată", _clean_text(selected.iloc[0].get("Interpretare", "")), accent="gold")

    render_section_header("Efecte temporale comune")
    time_cols = st.columns([0.58, 0.42])
    with time_cols[0]:
        _show_figure(results, "time_fixed_effects.png")
    with time_cols[1]:
        _display_table(tables["time_fixed_effects.csv"], height=430)

with tabs[4]:
    render_section_header("Diagnosticarea modelului panel")
    diagnostics = tables["panel_diagnostics.csv"]
    card_rows = pd.concat(
        [
            _selected_rows(diagnostics, "Heteroscedasticitate").head(1),
            _selected_rows(diagnostics, "Autocorelare").head(2),
            _selected_rows(diagnostics, "Dependență transversală").head(1),
            _selected_rows(diagnostics, "Normalitate").head(2),
        ],
        ignore_index=True,
    )
    _display_tests(card_rows)
    render_info_panel("Normalitatea reziduurilor", notes.get("normality", ""), accent="gold")

    render_section_header("Grafice reziduale")
    _show_figure_grid(
        results,
        [
            "panel_qq_plot.png",
            "panel_residual_histogram.png",
            "panel_residual_boxplot.png",
            "panel_residuals_time.png",
            "panel_residuals_by_country.png",
            "panel_residuals_year_mean.png",
            "panel_observed_fitted_final.png",
        ],
        columns=2,
    )

    render_section_header("Multicoliniaritate")
    vif_cols = st.columns([0.42, 0.58])
    with vif_cols[0]:
        _display_table(tables["vif_panel.csv"], height=430)
    with vif_cols[1]:
        _show_figure(results, "vif_panel_bar.png")

    render_section_header("Tabel central de diagnostic")
    _display_table(diagnostics, height=520)

with tabs[5]:
    render_section_header("Model final și inferență robustă")
    render_info_panel("Erori standard robuste", notes.get("robust", ""), accent="teal")
    render_metric_grid(
        [
            {"label": "Model final", "value": metadata.get("selected_model", "—"), "accent": "gold"},
            {"label": "Inferență robustă", "value": metadata.get("robust_method_used", "—"), "accent": "teal"},
            {"label": "Heteroscedasticitate", "value": "Da" if metadata.get("heteroskedasticity_detected") else "Nu", "accent": "burgundy"},
            {"label": "Autocorelare", "value": "Da" if metadata.get("serial_correlation_detected") else "Nu", "accent": "plum"},
        ],
        columns=4,
    )

    render_section_header("Comparația inferenței")
    _display_table(tables["robust_inference_comparison.csv"], height=460)
    _show_figure(results, "robust_coefficient_plot.png")

    render_section_header("Efecte fixe estimate")
    render_info_panel("Notă de interpretare", notes.get("fixed_effects", ""), accent="gold")
    fixed_cols = st.columns(2)
    with fixed_cols[0]:
        _show_figure(results, "country_fixed_effects.png")
    with fixed_cols[1]:
        _show_figure(results, "time_fixed_effects.png")

    render_section_header("Modele tematice și stabilitatea coeficienților")
    _display_table(tables["thematic_models_summary.csv"], height=250)
    _show_figure(results, "coefficient_stability_specs.png")

    render_section_header("Analiză pe subperioade")
    _display_table(tables["subperiod_models_summary.csv"], height=180)
    _show_figure(results, "coefficient_stability_subperiods.png")

    render_section_header("Interpretarea coeficienților")
    _display_table(tables["panel_coefficient_interpretations.csv"], height=420)

    render_section_header("Concluzii ale analizei panel")
    final_summary = tables["panel_final_summary.csv"]
    if not final_summary.empty:
        render_info_panel("Sinteză", _clean_text(final_summary.iloc[0].get("Text", "")), accent="burgundy")

    render_section_header("Succesiunea metodologică")
    flow_steps = ["Pooled OLS", "FE / RE", "F test + LM", "Hausman", "One-way / Two-way", "Diagnostic", "Robust SE", "Model final"]
    flow_cols = st.columns(len(flow_steps) * 2 - 1)
    for index, step in enumerate(flow_steps):
        with flow_cols[index * 2]:
            st.markdown(f"**{step}**")
        if index < len(flow_steps) - 1:
            with flow_cols[index * 2 + 1]:
                st.markdown("↓")
