"""Pagina pentru serii de timp și dinamică temporală."""

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
from src.r_time_series_bridge import TimeSeriesResults, load_time_series_results


PAGE_NAME = "Serii de timp și dinamică temporală"


configure_page(PAGE_NAME)
inject_global_css()
render_sidebar(PAGE_NAME)


def _format_value(value: object, digits: int = 3) -> str:
    """Formatează valorile calculate în R pentru afișare."""

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
    """Curăță valorile textuale pentru carduri."""

    if value == "" or pd.isna(value):
        return "—"
    return str(value)


def _table_value(table: pd.DataFrame, key: str, value_column: str = "Valoare", key_column: str = "Indicator", digits: int = 3) -> str:
    """Extrage o valoare pregătită în R dintr-un tabel tip indicator-valoare."""

    if table.empty or key_column not in table.columns or value_column not in table.columns:
        return "—"
    row = table.loc[table[key_column].astype(str).eq(key)]
    if row.empty:
        return "—"
    return _format_value(row.iloc[0][value_column], digits=digits)


def _table_detail(table: pd.DataFrame, key: str, detail_column: str = "Detaliu", key_column: str = "Indicator") -> str | None:
    """Extrage detaliul textual asociat unei valori."""

    if table.empty or key_column not in table.columns or detail_column not in table.columns:
        return None
    row = table.loc[table[key_column].astype(str).eq(key)]
    if row.empty:
        return None
    return _clean_text(row.iloc[0][detail_column])


def _display_table(dataframe: pd.DataFrame, height: int = 320) -> None:
    """Afișează un tabel compact fără calcule statistice suplimentare."""

    if dataframe.empty:
        render_empty_state("Tabel indisponibil", "Scriptul R nu a returnat rânduri pentru această componentă.")
        return
    display = dataframe.copy()
    for column in display.columns:
        if column in {"p-value", "p-value aproximativ"}:
            display[column] = display[column].map(_format_p)
        elif pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(_format_value)
        else:
            display[column] = display[column].map(_clean_text)
    render_light_table(display, height=height)


def _show_figure(results: TimeSeriesResults, name: str, caption: str | None = None) -> None:
    """Afișează o figură produsă de R."""

    path: Path | None = results.figures.get(name)
    if path is None:
        render_empty_state("Grafic indisponibil", f"Figura {name} nu a fost generată de scriptul R.")
        return
    st.image(str(path), caption=caption, use_column_width=True)


def _show_figure_grid(results: TimeSeriesResults, names: Iterable[str], columns: int = 2) -> None:
    """Afișează mai multe figuri într-o grilă."""

    names = list(names)
    for start in range(0, len(names), columns):
        cols = st.columns(columns)
        for index, figure_name in enumerate(names[start : start + columns]):
            with cols[index]:
                _show_figure(results, figure_name)


def _stationarity_cards(dataframe: pd.DataFrame) -> None:
    """Afișează ADF, PP și KPSS ca trei carduri metodologice."""

    if dataframe.empty:
        render_empty_state("Teste indisponibile", "Scriptul R nu a returnat testele de staționaritate.")
        return

    cards: list[str] = []
    for test_name, tone in [("ADF", "teal"), ("PP", "burgundy"), ("KPSS", "gold")]:
        row = dataframe.loc[dataframe["Test"].astype(str).eq(test_name)].head(1)
        if row.empty:
            continue
        record = row.iloc[0]
        cards.append(
            f"""
            <article class="test-card tone-neutral">
                <div class="test-card-top">
                    <strong>{escape(test_name)}</strong>
                    <span class="test-card-category">{escape(_clean_text(record.get("Specificație", "—")))}</span>
                </div>
                <p class="test-card-h0">{escape(_clean_text(record.get("H0", "—")))}</p>
                <div class="test-card-stats">
                    <span>Statistică <b>{escape(_format_value(record.get("Statistică", "")))}</b></span>
                    <span>Decizie <b>{escape(_clean_text(record.get("Decizie", "—")))}</b></span>
                </div>
                <p class="test-card-conclusion">{escape(_clean_text(record.get("Concluzie", "—")))}</p>
            </article>
            """
        )

    render_html(
        f"""
        <div class="test-card-grid time-series-test-grid">
            {''.join(cards)}
        </div>
        """
    )


def _diagnostic_card(table: pd.DataFrame, indicator: str, title: str, accent: str = "gold") -> None:
    """Afișează o concluzie de diagnostic produsă în R."""

    if table.empty or "Indicator" not in table.columns:
        render_empty_state(title, "Diagnostic indisponibil.")
        return
    row = table.loc[table["Indicator"].astype(str).eq(indicator)]
    if row.empty:
        render_empty_state(title, "Diagnostic indisponibil.")
        return
    render_info_panel(title, _clean_text(row.iloc[0].get("Concluzie", "")), accent=accent)


def _best_forecast_row(table: pd.DataFrame) -> pd.Series | None:
    """Returnează primul rând din tabelul sortat de R după RMSE."""

    if table.empty:
        return None
    return table.iloc[0]


@st.cache_data(show_spinner=False)
def _load_results() -> TimeSeriesResults:
    """Încarcă rezultatele R ale analizei temporale."""

    return load_time_series_results(force=False)


render_html(
    """
    <section class="panel-page-title">
        <h1>Serii de timp și dinamica participării feminine</h1>
        <p>Staționaritate, modele univariate, netezire exponențială, prognoză și relații dinamice pentru economiile europene, 2001–2023.</p>
    </section>
    """
)

try:
    results = _load_results()
except RuntimeError as error:
    render_warning_box(str(error), title="Analiza temporală nu poate fi încărcată")
    st.stop()

metadata = results.metadata
tables = results.tables
series_summary = tables["series_summary.csv"]

tabs = st.tabs(
    [
        "Evoluția seriei și trendul",
        "Staționaritate și diferențiere",
        "ARMA și ARIMA",
        "Netezire exponențială",
        "Prognoză și comparația modelelor",
        "Dinamică multivariată",
    ]
)

with tabs[0]:
    render_section_header("Evoluția participării feminine în timp")
    render_metric_grid(
        [
            {
                "label": "Ani",
                "value": _table_value(series_summary, "Ani", digits=0),
                "caption": _table_detail(series_summary, "Ani"),
                "accent": "teal",
            },
            {
                "label": "Media perioadei",
                "value": _table_value(series_summary, "Media perioadei"),
                "caption": _table_detail(series_summary, "Media perioadei"),
                "accent": "gold",
            },
            {
                "label": "Minim",
                "value": _table_value(series_summary, "Minim"),
                "caption": _table_detail(series_summary, "Minim"),
                "accent": "burgundy",
            },
            {
                "label": "Maxim",
                "value": _table_value(series_summary, "Maxim"),
                "caption": _table_detail(series_summary, "Maxim"),
                "accent": "plum",
            },
            {
                "label": "Trend anual estimat",
                "value": _table_value(series_summary, "Trend anual estimat"),
                "caption": _table_detail(series_summary, "Trend anual estimat"),
                "accent": "navy",
            },
        ],
        columns=5,
    )

    _show_figure(results, "flfp_europe_evolution.png")

    render_section_header("Trend determinist")
    trend_col, table_col = st.columns([0.64, 0.36])
    with trend_col:
        _show_figure(results, "flfp_trend_linear.png")
    with table_col:
        _display_table(tables["trend_model.csv"], height=240)

    render_section_header("Corelograma seriei originale")
    _show_figure_grid(results, ["acf_level.png", "pacf_level.png"], columns=2)

with tabs[1]:
    render_section_header("Analiza staționarității")
    render_info_panel(
        "Ipoteze testate",
        "ADF și Phillips–Perron pornesc de la H0: seria are rădăcină unitară. KPSS are ipoteza nulă inversă: seria este staționară.",
        accent="plum",
    )
    _stationarity_cards(tables["stationarity_level.csv"])

    render_section_header("Tabel central de staționaritate")
    _display_table(tables["stationarity_level.csv"], height=280)

    render_section_header("Diferențiere de ordinul I")
    diff_col, shock_col = st.columns([0.62, 0.38])
    with diff_col:
        _show_figure(results, "flfp_diff1.png")
    with shock_col:
        _display_table(tables["diff_shocks.csv"], height=270)

    render_section_header("Corelograma seriei diferențiate")
    _show_figure_grid(results, ["acf_diff1.png", "pacf_diff1.png"], columns=2)

    render_section_header("Nivel vs diferența I")
    _display_table(tables["integration_summary.csv"], height=230)

with tabs[2]:
    candidate_cols = st.columns(2)
    with candidate_cols[0]:
        render_section_header("Modele ARMA candidate")
        _display_table(tables["arma_candidates.csv"], height=260)
    with candidate_cols[1]:
        render_section_header("Modele ARIMA candidate")
        _display_table(tables["arima_candidates.csv"], height=260)

    _show_figure(results, "arima_information_criteria.png")

    render_section_header("Modelul ARIMA selectat")
    render_metric_grid(
        [
            {"label": "Model selectat", "value": metadata.get("selected_arima", "—"), "accent": "teal"},
            {"label": "Verificare auto.arima", "value": metadata.get("auto_arima_check", "—"), "accent": "gold"},
            {"label": "Ordin de integrare", "value": metadata.get("estimated_integration_order", "—"), "accent": "burgundy"},
        ],
        columns=3,
    )
    _display_table(tables["selected_arima_coefficients.csv"], height=170)

    render_section_header("Diagnostic reziduuri")
    _show_figure_grid(
        results,
        [
            "arima_residuals_time.png",
            "arima_residuals_acf.png",
            "arima_residuals_qq.png",
            "arima_residuals_histogram.png",
            "arima_observed_fitted.png",
            "arima_residuals_boxplot.png",
        ],
        columns=2,
    )
    diagnostic_cols = st.columns([0.42, 0.58])
    with diagnostic_cols[0]:
        _diagnostic_card(tables["arima_diagnostics.csv"], "Concluzie zgomot alb", "Reziduurile se comportă ca zgomot alb?", accent="teal")
    with diagnostic_cols[1]:
        _display_table(tables["arima_diagnostics.csv"], height=270)

with tabs[3]:
    render_section_header("Modele de netezire exponențială")
    smoothing = tables["smoothing_models.csv"]
    smoothing_cards = []
    for model_name, accent in [("SES", "teal"), ("Holt", "burgundy"), ("ETS", "gold")]:
        row = smoothing.loc[smoothing["Model"].astype(str).eq(model_name)].head(1) if not smoothing.empty else pd.DataFrame()
        value = _format_value(row.iloc[0]["AICc"]) if not row.empty and "AICc" in row.columns else "—"
        caption = "AICc" if model_name != "ETS" else _clean_text(row.iloc[0].get("Model ETS ales", "AICc")) if not row.empty else "AICc"
        smoothing_cards.append({"label": model_name, "value": value, "caption": caption, "accent": accent})
    render_metric_grid(smoothing_cards, columns=3)

    _show_figure(results, "smoothing_comparison.png")
    render_section_header("Modele individuale")
    _show_figure_grid(results, ["smoothing_ses.png", "smoothing_holt.png", "smoothing_ets.png"], columns=3)
    render_section_header("Parametrii netezirii")
    _display_table(smoothing, height=180)

with tabs[4]:
    render_section_header("Evaluarea capacității predictive")
    _show_figure(results, "forecast_arima_holdout.png")
    _show_figure(results, "forecast_model_comparison.png")

    best_row = _best_forecast_row(tables["forecast_accuracy.csv"])
    render_metric_grid(
        [
            {"label": "Model favorabil", "value": _clean_text(best_row.get("Model", "—")) if best_row is not None else "—", "accent": "gold"},
            {"label": "RMSE", "value": _format_value(best_row.get("RMSE", "")) if best_row is not None else "—", "accent": "teal"},
            {"label": "MAE", "value": _format_value(best_row.get("MAE", "")) if best_row is not None else "—", "accent": "burgundy"},
            {"label": "MAPE", "value": _format_value(best_row.get("MAPE", "")) if best_row is not None else "—", "accent": "plum"},
            {"label": "MASE", "value": _format_value(best_row.get("MASE", "")) if best_row is not None else "—", "accent": "navy"},
            {"label": "Theil U2", "value": _format_value(best_row.get("Theil U2", "")) if best_row is not None else "—", "accent": "gold"},
        ],
        columns=6,
    )

    render_section_header("Comparația metricilor de forecast")
    metric_cols = st.columns(2)
    with metric_cols[0]:
        _show_figure(results, "forecast_rmse_comparison.png")
    with metric_cols[1]:
        _show_figure(results, "forecast_mae_comparison.png")
    _display_table(tables["forecast_accuracy.csv"], height=260)

    render_section_header("Prognoză exploratorie 2024–2026")
    render_info_panel(
        "Limitare",
        "Prognoza 2024–2026 reprezintă o extensie statistică a seriei și nu include explicit șocuri economice viitoare sau modificări structurale.",
        accent="burgundy",
    )
    _show_figure(results, "forecast_future_2024_2026.png")
    _display_table(tables["future_forecast.csv"], height=170)

with tabs[5]:
    render_section_header("Dinamică multivariată")
    render_info_panel(
        "Extensie multivariată",
        "Această secțiune extinde analiza univariată prin serii europene agregate neponderat pentru participarea feminină, inflație, creștere PIB și șomaj feminin.",
        accent="teal",
    )

    _show_figure(results, "multivariate_standardized.png")

    render_section_header("Staționaritate multivariată")
    _display_table(tables["multivariate_stationarity.csv"], height=200)

    render_section_header("FLFP și inflație")
    _show_figure_grid(
        results,
        [
            "multivariate_flfp_inflation_standardized.png",
            "multivariate_ccf_flfp_inflation.png",
            "multivariate_scatter_flfp_inflation.png",
        ],
        columns=2,
    )

    render_section_header("Granger și model multivariat")
    model_cols = st.columns(2)
    with model_cols[0]:
        _display_table(tables["granger_results.csv"], height=180)
    with model_cols[1]:
        _display_table(tables["var_summary.csv"], height=220)

    if "multivariate_irf_inflation_to_flfp.png" in results.figures:
        render_section_header("Impulse response")
        _show_figure(results, "multivariate_irf_inflation_to_flfp.png")
