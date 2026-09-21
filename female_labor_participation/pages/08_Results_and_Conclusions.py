"""Pagina pentru rezultate și concluzii."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st

from components.empty_states import render_empty_state, render_section_header
from components.html import render_html
from components.metric_cards import render_metric_grid
from components.sidebar import configure_page, inject_global_css, render_sidebar


configure_page("Rezultate și concluzii")
inject_global_css()
render_sidebar("Rezultate și concluzii")


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def _read_csv(relative_path: str) -> pd.DataFrame:
    """Încarcă un tabel de rezultate, dacă există."""

    path = OUTPUTS / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(relative_path: str) -> dict:
    """Încarcă metadata de rezultate, dacă există."""

    path = OUTPUTS / relative_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _format_value(value: object, digits: int = 3) -> str:
    """Formatează compact valorile deja calculate."""

    if value == "" or value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}".replace(",", " ")
    if abs(numeric) >= 100:
        return f"{numeric:,.1f}"
    return f"{numeric:.{digits}f}"


def _format_p(value: object) -> str:
    """Formatează p-value."""

    if value == "" or value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.4f}"


def _metric(table: pd.DataFrame, indicator: str, digits: int = 3) -> str:
    """Extrage o metrică dintr-un tabel Indicator-Valoare."""

    if table.empty or "Indicator" not in table.columns or "Valoare" not in table.columns:
        return "—"
    row = table.loc[table["Indicator"].astype(str).eq(indicator)]
    if row.empty:
        return "—"
    return _format_value(row.iloc[0]["Valoare"], digits=digits)


def _coef(table: pd.DataFrame, term: str, column: str = "Coeficient", digits: int = 3) -> str:
    """Extrage un coeficient dintr-un tabel econometric."""

    if table.empty or "Termen" not in table.columns or column not in table.columns:
        return "—"
    row = table.loc[table["Termen"].astype(str).eq(term)]
    if row.empty:
        return "—"
    return _format_value(row.iloc[0][column], digits=digits)


def _coef_p(table: pd.DataFrame, term: str) -> str:
    """Extrage p-value pentru un coeficient."""

    if table.empty or "Termen" not in table.columns or "p-value" not in table.columns:
        return "—"
    row = table.loc[table["Termen"].astype(str).eq(term)]
    if row.empty:
        return "—"
    return _format_p(row.iloc[0]["p-value"])


def _top_items(table: pd.DataFrame, column: str, limit: int = 5) -> str:
    """Returnează primele elemente dintr-un tabel ca listă textuală."""

    if table.empty or column not in table.columns:
        return "—"
    return ", ".join(table[column].astype(str).head(limit).tolist())


def _significant_terms(table: pd.DataFrame, limit: int = 6) -> str:
    """Rezumat textual pentru coeficienții semnificativi."""

    if table.empty or not {"Termen", "Coeficient", "p-value"}.issubset(table.columns):
        return "—"
    significant = table.loc[table["p-value"] < 0.05].head(limit)
    if significant.empty:
        return "—"
    terms = []
    for _, row in significant.iterrows():
        beta = float(row["Coeficient"])
        sign = "+" if beta > 0 else ""
        terms.append(f"{row['Termen']} (β={sign}{_format_value(beta, 3)})")
    return ", ".join(terms)


def _render_result_card(title: str, body: str, meta: str, accent: str = "gold") -> None:
    """Afișează un card de concluzie."""

    render_html(
        f"""
        <article class="result-card tone-{escape(accent)}">
            <span>{escape(meta)}</span>
            <h3>{escape(title)}</h3>
            <p>{escape(body)}</p>
        </article>
        """
    )


def _render_signal(title: str, verdict: str, detail: str, tone: str = "teal") -> str:
    """Construiește un card de convergență între metode."""

    return f"""
    <article class="signal-card tone-{escape(tone)}">
        <span>{escape(verdict)}</span>
        <h3>{escape(title)}</h3>
        <p>{escape(detail)}</p>
    </article>
    """


def _render_visual(path: Path, title: str, caption: str) -> None:
    """Afișează o figură reprezentativă într-un cadru editorial."""

    if not path.exists():
        return
    if title or caption:
        title_html = f"<h3>{escape(title)}</h3>" if title else ""
        caption_html = f"<p>{escape(caption)}</p>" if caption else ""
        render_html(
            f"""
            <div class="results-visual-caption">
                {title_html}
                {caption_html}
            </div>
            """
        )
    st.image(str(path), use_column_width=True)


def _render_plotly(path: Path, title: str, caption: str) -> None:
    """Afișează o figură Plotly reprezentativă."""

    if not path.exists():
        return
    render_html(
        f"""
        <div class="results-visual-caption">
            <h3>{escape(title)}</h3>
            <p>{escape(caption)}</p>
        </div>
        """
    )
    fig = pio.from_json(path.read_text(encoding="utf-8"))
    fig.update_layout(height=440)
    st.plotly_chart(fig, use_container_width=True)


cs_metadata = _read_json("cross_sectional/metadata/analysis_metadata.json")
panel_metadata = _read_json("panel/metadata/panel_analysis.json")
time_metadata = _read_json("time_series/metadata/time_series_analysis.json")
ml_metadata = _read_json("machine_learning/metadata/ml_analysis.json")

simple_metrics = _read_csv("cross_sectional/tables/simple_model_metrics.csv")
simple_coefficients = _read_csv("cross_sectional/tables/simple_coefficients.csv")
multiple_metrics = _read_csv("cross_sectional/tables/multiple_model_metrics.csv")
robust_hc3 = _read_csv("cross_sectional/tables/robust_hc3_coefficients.csv")
multiple_tests = _read_csv("cross_sectional/tables/multiple_tests.csv")

dk_coefficients = _read_csv("panel/tables/driscoll_kraay_coefficients.csv")
panel_diagnostics = _read_csv("panel/tables/panel_diagnostics.csv")

trend_model = _read_csv("time_series/tables/trend_model.csv")
forecast_accuracy = _read_csv("time_series/tables/forecast_accuracy.csv")
granger_results = _read_csv("time_series/tables/granger_results.csv")

regression_leaderboard = _read_csv("machine_learning/tables/regression_leaderboard.csv")
classification_leaderboard = _read_csv("machine_learning/tables/classification_leaderboard.csv")
importance_consensus = _read_csv("machine_learning/tables/importance_consensus.csv")
shap_regression = _read_csv("machine_learning/tables/shap_global_regression.csv")


if not any(
    [
        cs_metadata,
        panel_metadata,
        time_metadata,
        ml_metadata,
        not regression_leaderboard.empty,
    ]
):
    render_empty_state(
        "Rezultatele nu sunt disponibile",
        "Pagina se construiește din outputurile generate de modulele econometrice, de serii de timp și Machine Learning.",
    )
    st.stop()


inflation_beta = _coef(simple_coefficients, "Inflație")
inflation_p = _coef_p(simple_coefficients, "Inflație")
simple_r2 = _metric(simple_metrics, "R²")
multiple_adj_r2 = _metric(multiple_metrics, "R² ajustat")
unemployment_hc3 = _coef(robust_hc3, "Șomaj feminin")
unemployment_hc3_p = _coef_p(robust_hc3, "Șomaj feminin")
reset_row = (
    multiple_tests.loc[multiple_tests["Test"].astype(str).str.contains("Ramsey", na=False)].head(1)
    if not multiple_tests.empty and "Test" in multiple_tests.columns
    else pd.DataFrame()
)
reset_p = _format_p(reset_row.iloc[0]["p-value"]) if not reset_row.empty else "—"

trend_beta = _format_value(trend_model.iloc[0]["Beta"], 4) if not trend_model.empty and "Beta" in trend_model.columns else "—"
trend_r2 = _format_value(trend_model.iloc[0]["R²"], 3) if not trend_model.empty and "R²" in trend_model.columns else "—"
best_forecast = forecast_accuracy.sort_values("RMSE").head(1) if not forecast_accuracy.empty and "RMSE" in forecast_accuracy.columns else pd.DataFrame()
best_forecast_model = str(best_forecast.iloc[0]["Model"]) if not best_forecast.empty else time_metadata.get("selected_forecast_model", "—")
best_forecast_rmse = _format_value(best_forecast.iloc[0]["RMSE"], 3) if not best_forecast.empty else "—"
granger_text = time_metadata.get(
    "multivariate_conclusion",
    "Testele Granger sunt interpretate ca valoare predictivă temporală incrementală, nu ca relație cauzală structurală.",
)

best_regression = regression_leaderboard.head(1)
best_classification = classification_leaderboard.head(1)
best_reg_model = str(best_regression.iloc[0]["Model"]) if not best_regression.empty else ml_metadata.get("best_regression_model", "—")
best_reg_rmse = _format_value(best_regression.iloc[0]["RMSE Test"], 3) if not best_regression.empty else _format_value(ml_metadata.get("best_regression_rmse"), 3)
best_reg_r2 = _format_value(best_regression.iloc[0]["R² Test"], 3) if not best_regression.empty and "R² Test" in best_regression.columns else "—"
best_cls_model = str(best_classification.iloc[0]["Model"]) if not best_classification.empty else ml_metadata.get("best_classification_model", "—")
best_cls_f1 = _format_value(best_classification.iloc[0]["F1 Macro"], 3) if not best_classification.empty else _format_value(ml_metadata.get("best_classification_f1_macro"), 3)
top_predictors = _top_items(importance_consensus, "Feature", 5)
top_shap = _top_items(shap_regression, "Feature", 5)

panel_significant = dk_coefficients.loc[dk_coefficients["p-value"] < 0.05] if not dk_coefficients.empty and "p-value" in dk_coefficients.columns else pd.DataFrame()
panel_terms = _significant_terms(panel_significant)


render_html(
    f"""
    <section class="results-hero">
        <h1>Rezultate și concluzii</h1>
    </section>
    """
)

render_metric_grid(
    [
        {
            "label": "Trend anual",
            "value": f"+{trend_beta}",
            "caption": f"puncte procentuale; R²={trend_r2}",
            "accent": "plum",
        },
        {
            "label": "ML regresie",
            "value": best_reg_model,
            "caption": f"RMSE={best_reg_rmse}; R²={best_reg_r2}",
            "accent": "burgundy",
        },
    ],
    columns=2,
)

render_section_header("Concluzia centrală")

render_html(
    f"""
    <div class="results-thesis">
        <div class="results-thesis-mark">01</div>
        <div>
            <h2 class="results-thesis-title">Participarea feminină are o evoluție ascendentă, dar determinanții ei nu se reduc la un singur factor.</h2>
            <p>
                Seria agregată europeană indică un trend crescător în perioada 2001–2023. În același timp,
                modelele arată că relațiile diferă între analiza dintre țări, variația în interiorul țărilor
                și predicția pe setul de test. Dimensiunile recurente sunt structura pieței muncii,
                șomajul feminin, urbanizarea, calitatea instituțională și vulnerabilitatea ocupării.
            </p>
        </div>
    </div>
    """
)

render_section_header("Rezultate pe metode")

method_cols = st.columns(2)
with method_cols[0]:
    _render_result_card(
        "Analiza transversală, 2023",
        (
            f"În regresia simplă, inflația are asociere negativă cu participarea feminină "
            f"(β={inflation_beta}, p={inflation_p}, R²={simple_r2}). În modelul multiplu, "
            f"după erori robuste HC3, șomajul feminin rămâne predictor semnificativ "
            f"(β={unemployment_hc3}, p={unemployment_hc3_p}). Testul Ramsey RESET "
            f"(p={reset_p}) sugerează prudență privind specificarea."
        ),
        "Secțiune transversală",
        "teal",
    )
with method_cols[1]:
    _render_result_card(
        "Econometria datelor panel",
        (
            f"Testele de selecție favorizează modelul {panel_metadata.get('selected_model', '—')}. "
            f"Cu inferență {panel_metadata.get('robust_method_used', '—')}, coeficienții semnificativi includ: "
            f"{panel_terms}. Diagnosticele indică heteroscedasticitate și autocorelare, de aceea interpretarea "
            f"se bazează pe erori robuste."
        ),
        "Panel 2001–2023",
        "gold",
    )

method_cols = st.columns(2)
with method_cols[0]:
    _render_result_card(
        "Dinamica temporală",
        (
            f"Seria agregată are trend crescător, cu un coeficient anual estimat de {trend_beta} puncte procentuale. "
            f"Modelul de prognoză cu cea mai mică eroare RMSE pe holdout este {best_forecast_model} "
            f"(RMSE={best_forecast_rmse}). {granger_text}"
        ),
        "Serii de timp",
        "plum",
    )
with method_cols[1]:
    _render_result_card(
        "Machine Learning",
        (
            f"{best_reg_model} obține cea mai mică eroare de regresie pe test (RMSE={best_reg_rmse}), "
            f"iar {best_cls_model} are cea mai bună clasificare după F1 Macro ({best_cls_f1}). "
            f"Predictorii recurenți ca importanță sunt: {top_predictors}."
        ),
        "Predicție",
        "burgundy",
    )

render_section_header("Convergența rezultatelor")

signals = [
    _render_signal(
        "Inflația",
        "semnal limitat",
        "Apare negativă și semnificativă în regresia simplă 2023, dar nu rămâne determinant robust în specificațiile multiple, panel sau temporale.",
        "gold",
    ),
    _render_signal(
        "Șomajul feminin",
        "semnal recurent",
        "Este semnificativ în modelul transversal robust și în modelul panel Driscoll–Kraay, însă semnul diferă între perspectiva între țări și cea within.",
        "teal",
    ),
    _render_signal(
        "Urbanizarea",
        "semnal structural",
        "Devine semnificativă în panel și apare între predictorii importanți în Machine Learning, ceea ce susține relevanța dimensiunii teritoriale.",
        "plum",
    ),
    _render_signal(
        "Instituții și vulnerabilitate",
        "semnal predictiv puternic",
        f"Controlul corupției și ocuparea vulnerabilă sunt printre predictorii cei mai importanți în consensul ML; SHAP evidențiază: {top_shap}.",
        "burgundy",
    ),
]
render_html(f'<div class="signal-grid">{"".join(signals)}</div>')

render_section_header("Vizualizări reprezentative")

visual_cols = st.columns(2)
with visual_cols[0]:
    _render_visual(
        OUTPUTS / "time_series/figures/flfp_europe_evolution.png",
        "",
        "",
    )
with visual_cols[1]:
    _render_visual(
        OUTPUTS / "panel/figures/robust_coefficient_plot.png",
        "",
        "",
    )

visual_cols = st.columns(2)
with visual_cols[0]:
    _render_visual(
        OUTPUTS / "cross_sectional/figures/simple_scatter_regression.png",
        "Relația transversală cu inflația",
        "Regresia simplă pentru 2023 surprinde asocierea negativă dintre inflație și participarea feminină.",
    )
with visual_cols[1]:
    _render_plotly(
        OUTPUTS / "machine_learning/figures/permutation_heatmap_regression.json",
        "Importanța predictorilor în modelele ML",
        "Importanța este predictivă și completează, fără a înlocui, interpretarea econometrică.",
    )

render_section_header("Implicații economice")

implication_cols = st.columns(3)
with implication_cols[0]:
    _render_result_card(
        "Piața muncii contează prin structură, nu doar prin nivel",
        "Șomajul feminin, ocuparea vulnerabilă și ponderea serviciilor indică faptul că participarea nu poate fi separată de calitatea și compoziția ocupării.",
        "Implicație",
        "teal",
    )
with implication_cols[1]:
    _render_result_card(
        "Instituțiile apar mai ales ca semnal predictiv",
        "Controlul corupției este important în Machine Learning și apare semnificativ în unele specificații, dar interpretarea sa depinde de model și de tipul variației analizate.",
        "Implicație",
        "gold",
    )
with implication_cols[2]:
    _render_result_card(
        "Digitalizarea nu acționează izolat",
        "Utilizarea internetului este relevantă în interpretarea SHAP, dar rezultatele econometrice nu susțin tratarea ei ca determinant structural singular.",
        "Implicație",
        "plum",
    )

render_section_header("Limitări")

limits = [
    _render_signal(
        "Cauzalitate",
        "nu este demonstrată",
        "Regresiile și modelele predictive descriu asocieri statistice; nu identifică automat efecte cauzale structurale.",
        "burgundy",
    ),
    _render_signal(
        "Analiza transversală",
        "un singur an",
        "Rezultatele pentru 2023 descriu diferențe între țări și sunt sensibile la specificare, lucru semnalat de testul Ramsey RESET.",
        "gold",
    ),
    _render_signal(
        "Panel",
        "diagnostice active",
        "Heteroscedasticitatea și autocorelarea impun inferență robustă; coeficienții panel reflectă variația în timp în interiorul economiilor.",
        "teal",
    ),
    _render_signal(
        "Machine Learning",
        "predicție, nu explicație cauzală",
        "Importanțele, SHAP și scorurile pe test arată utilitate predictivă, dar nu trebuie citite ca efecte economice directe.",
        "plum",
    ),
]
render_html(f'<div class="signal-grid limit-grid">{"".join(limits)}</div>')

render_html(
    """
    <div class="final-conclusion-band">
        <span>Concluzie finală</span>
        <p>
            Rezultatele susțin o interpretare integrată: participarea femeilor pe piața muncii în Europa a crescut
            în perioada analizată, iar diferențele observate sunt legate de condiții macroeconomice, instituționale,
            structurale și demografice. Cel mai solid mesaj metodologic este complementaritatea dintre econometrie
            și Machine Learning: prima clarifică tipul de asociere și condițiile de inferență, iar al doilea arată
            care variabile au valoare predictivă ridicată.
        </p>
    </div>
    """
)
