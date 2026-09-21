"""Pagina Streamlit pentru experimentul Machine Learning."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.io as pio
import streamlit as st

from components.html import render_html
from components.metric_cards import render_metric_grid
from components.sidebar import configure_page, inject_global_css, render_sidebar
from components.tables import render_light_table
from src.ml_analysis_bridge import MLResults, load_ml_results


configure_page("Machine Learning")
inject_global_css()
render_sidebar("Machine Learning")


@st.cache_data(show_spinner=False)
def _load_results() -> MLResults:
    """Încarcă rezultatele persistate ale pipeline-ului ML."""

    return load_ml_results()


def _format_value(value: object, digits: int = 3) -> str:
    """Formatare compactă pentru metrici."""

    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", " ")
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _summary_value(results: MLResults, indicator: str) -> str:
    """Extrage o valoare din sumarul experimentului."""

    summary = results.tables.get("experiment_summary", pd.DataFrame())
    if summary.empty:
        return "n/a"
    match = summary.loc[summary["Indicator"].eq(indicator)]
    if match.empty:
        return "n/a"
    return _format_value(match.iloc[0]["Valoare"])


def _section(title: str, subtitle: str | None = None) -> None:
    """Antet editorial de secțiune."""

    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    render_html(
        f"""
        <div class="section-header">
            <span class="section-eyebrow">Machine Learning</span>
            <h2>{title}</h2>
            {subtitle_html}
        </div>
        """
    )


def _model_intro(title: str, body: str, accent: str = "gold") -> None:
    """Card metodologic scurt."""

    render_html(
        f"""
        <div class="ml-note-card accent-{accent}">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """
    )


def _model_card(results: MLResults, model: str, task: str) -> None:
    """Afișează cardul metodologic al unui model."""

    cards = results.tables.get("model_cards", pd.DataFrame())
    if cards.empty:
        return
    match = cards.loc[cards["Model"].eq(model) & cards["Task"].eq(task)]
    if match.empty:
        return
    row = match.iloc[0]
    render_html(
        f"""
        <div class="ml-model-card">
            <span>{row['Task']}</span>
            <h3>{row['Model']}</h3>
            <p>{row['Tip model']}</p>
            <div class="ml-model-meta">
                <b>{int(row['Train N'])}</b><small>train</small>
                <b>{int(row['Test N'])}</b><small>test</small>
                <b>{_format_value(row['Valoare metrică'])}</b><small>{row['Metrică principală']}</small>
                <b>{_format_value(row['Timp de antrenare sec'], 2)}s</b><small>fit time</small>
            </div>
        </div>
        """
    )


def _metric_cards(rows: list[dict[str, str]]) -> None:
    """Wrapper pentru cardurile metrice."""

    render_metric_grid(rows, columns=min(4, max(1, len(rows))))


def _metrics_from_table(table: pd.DataFrame, specs: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    """Construiește carduri metrice din primul rând al unui tabel."""

    if table.empty:
        return []
    row = table.iloc[0]
    return [
        {
            "label": label,
            "value": _format_value(row[column]),
            "caption": caption,
            "accent": accent,
        }
        for column, label, accent, caption in specs
        if column in table.columns
    ]


def _table(results: MLResults, name: str, columns: Iterable[str] | None = None, height: int | None = None) -> None:
    """Afișează un tabel dacă există."""

    data = results.tables.get(name, pd.DataFrame())
    if data.empty:
        st.info("Tabelul nu este disponibil în outputurile curente.")
        return
    if columns is not None:
        visible = [column for column in columns if column in data.columns]
        data = data[visible]
    render_light_table(data, height=height)


def _figure(results: MLResults, name: str, height: int | None = None) -> None:
    """Afișează un grafic Plotly salvat ca JSON."""

    path = results.figures.get(name)
    if path is None:
        st.info("Graficul nu este disponibil în outputurile curente.")
        return
    fig = pio.from_json(path.read_text(encoding="utf-8"))
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


def _figure_grid(results: MLResults, names: list[str], height: int | None = None) -> None:
    """Afișează graficele în rânduri de câte două."""

    for start in range(0, len(names), 2):
        cols = st.columns(2)
        for index, name in enumerate(names[start : start + 2]):
            with cols[index]:
                _figure(results, name, height=height)


def _image(path: Path) -> None:
    """Afișează o imagine SHAP."""

    st.image(str(path), use_column_width=True)


def _shap_gallery(results: MLResults, keys: list[str]) -> None:
    """Galerie pentru imaginile SHAP disponibile."""

    available = [results.shap_figures[key] for key in keys if key in results.shap_figures]
    if not available:
        st.info("Imaginile SHAP nu sunt disponibile pentru această selecție.")
        return
    for start in range(0, len(available), 2):
        cols = st.columns(2)
        for index, path in enumerate(available[start : start + 2]):
            with cols[index]:
                _image(path)


def _hyperparameters(results: MLResults, model: str, task: str) -> None:
    """Afișează hyperparametrii selectați pentru model."""

    params = results.tables.get("hyperparameters", pd.DataFrame())
    if params.empty:
        return
    data = params.loc[params["Model"].eq(model) & params["Task"].eq(task)]
    if data.empty:
        return
    tones = ["teal", "gold", "burgundy", "plum", "navy"]
    cards = []
    for index, row in data.reset_index(drop=True).iterrows():
        tone = tones[index % len(tones)]
        param = escape(str(row["Parametru"]))
        value = escape(str(row["Valoare"]))
        cards.append(
            f"""
            <div class="ml-hyperparam-card tone-{tone}">
                <span>{param}</span>
                <strong>{value}</strong>
            </div>
            """
        )
    render_html(
        f"""
        <div class="ml-hyperparams">
            <div class="ml-hyperparams-head">
                <span>{escape(task)}</span>
                <h3>Hyperparametrii selectați</h3>
            </div>
            <div class="ml-hyperparam-grid">
                {''.join(cards)}
            </div>
        </div>
        """
    )


def _model_tab(
    results: MLResults,
    title: str,
    reg_model: str,
    cls_model: str,
    reg_metrics_name: str,
    cls_metrics_name: str,
    stem: str,
    intro: str | None,
    cls_stem: str | None = None,
    extra_reg_figures: list[str] | None = None,
    extra_cls_figures: list[str] | None = None,
) -> None:
    """Structura comună pentru taburile dedicate familiilor de modele."""

    classification_stem = cls_stem or stem
    _section(title, intro)
    cols = st.columns(2)
    with cols[0]:
        _model_card(results, reg_model, "Regresie")
    with cols[1]:
        _model_card(results, cls_model, "Clasificare")

    st.markdown("### Regresie")
    reg_metrics = results.tables.get(reg_metrics_name, pd.DataFrame())
    _metric_cards(
        _metrics_from_table(
            reg_metrics,
            [
                ("RMSE Test", "RMSE test", "teal", "holdout 2019–2023"),
                ("MAE Test", "MAE test", "gold", "eroare absolută"),
                ("R² Test", "R² test", "plum", "variație explicată predictiv"),
                ("Generalization Gap", "Gap", "burgundy", "RMSE test minus train"),
            ],
        )
    )
    _figure_grid(
        results,
        [
            f"{stem}_actual_vs_predicted",
            f"{stem}_residuals",
            f"{stem}_error_distribution",
            f"{stem}_absolute_error_distribution",
            f"{stem}_learning_curve_regression",
            f"{stem}_validation_curve_regression",
            f"{stem}_permutation_importance_regression",
            f"{stem}_predictions_over_time",
            *(extra_reg_figures or []),
        ],
    )
    _hyperparameters(results, reg_model, "Regresie")

    st.markdown("### Clasificare")
    cls_metrics = results.tables.get(cls_metrics_name, pd.DataFrame())
    _metric_cards(
        _metrics_from_table(
            cls_metrics,
            [
                ("Accuracy", "Accuracy", "teal", "test 2019–2023"),
                ("Balanced Accuracy", "Balanced acc.", "gold", "corecție dezechilibru"),
                ("F1 Macro", "F1 Macro", "plum", "metrică principală"),
                ("ROC-AUC OvR", "ROC-AUC", "burgundy", "one-vs-rest"),
            ],
        )
    )
    _figure_grid(
        results,
        [
            f"{classification_stem}_confusion_matrix",
            f"{classification_stem}_confusion_matrix_normalized",
            f"{classification_stem}_roc_ovr",
            f"{classification_stem}_precision_recall",
            f"{classification_stem}_learning_curve_classification",
            f"{classification_stem}_validation_curve_classification",
            f"{classification_stem}_permutation_importance_classification",
            f"{classification_stem}_probability_distribution",
            f"{classification_stem}_correct_incorrect_confidence",
            *(extra_cls_figures or []),
        ],
    )


try:
    results = _load_results()
except Exception as error:
    st.error("Pagina Machine Learning nu poate încărca rezultatele calculate.")
    st.exception(error)
    st.stop()


render_html(
    """
    <section class="panel-page-title">
        <h1>Machine Learning: predicție, clasificare și interpretabilitate</h1>
        <p>Evaluarea relațiilor predictive dintre factorii economici, demografici, structurali, digitali și instituționali și participarea femeilor pe piața muncii.</p>
    </section>
    """
)

tabs = st.tabs(
    [
        "Pregătirea experimentului",
        "Support Vector Machines",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "Interpretabilitatea modelelor",
        "Comparația finală",
    ]
)

with tabs[0]:
    _metric_cards(
        [
            {"label": "Observații utilizabile", "value": _summary_value(results, "Observații totale utilizabile"), "caption": "target disponibil", "accent": "teal"},
            {"label": "Train", "value": _summary_value(results, "Train"), "caption": "2001–2018", "accent": "gold"},
            {"label": "Test", "value": _summary_value(results, "Test"), "caption": "2019–2023", "accent": "burgundy"},
            {"label": "Features", "value": _summary_value(results, "Număr features"), "caption": "fără țară, ISO3 sau an", "accent": "plum"},
            {"label": "Țări", "value": _summary_value(results, "Număr țări"), "caption": "economii europene", "accent": "navy"},
            {"label": "Perioadă", "value": _summary_value(results, "Perioadă"), "caption": "date anuale", "accent": "teal"},
        ]
    )
    cols = st.columns(2)
    with cols[0]:
        _model_intro("Regresie", "Target continuu: rata participării femeilor la forța de muncă, 15–64 ani.", "teal")
    with cols[1]:
        _model_intro(
            "Clasificare",
            f"Trei clase construite prin tertile calculate exclusiv pe training: Q33={_format_value(results.metadata.get('q33_class_threshold'))}, Q67={_format_value(results.metadata.get('q67_class_threshold'))}.",
            "burgundy",
        )
    _figure_grid(results, ["target_distribution", "class_distribution", "dataset_shift"])

with tabs[1]:
    _model_tab(
        results,
        "Support Vector Machines",
        "SVR",
        "SVC",
        "svr_metrics",
        "svc_metrics",
        "svr",
        None,
        cls_stem="svc",
        extra_reg_figures=["svr_c_gamma_heatmap"],
        extra_cls_figures=["svc_c_gamma_heatmap"],
    )

with tabs[2]:
    _model_tab(
        results,
        "Random Forest",
        "Random Forest",
        "Random Forest",
        "rf_regression_metrics",
        "rf_classification_metrics",
        "random_forest",
        None,
        extra_reg_figures=["random_forest_feature_importance_regression"],
        extra_cls_figures=["random_forest_feature_importance_classification"],
    )

with tabs[3]:
    _model_tab(
        results,
        "Gradient Boosting",
        "Gradient Boosting",
        "Gradient Boosting",
        "gb_regression_metrics",
        "gb_classification_metrics",
        "gradient_boosting",
        None,
        extra_reg_figures=["gradient_boosting_feature_importance_regression"],
        extra_cls_figures=["gradient_boosting_feature_importance_classification"],
    )

with tabs[4]:
    _model_tab(
        results,
        "XGBoost",
        "XGBoost",
        "XGBoost",
        "xgb_regression_metrics",
        "xgb_classification_metrics",
        "xgboost",
        None,
        extra_reg_figures=["xgboost_feature_importance_regression", "xgb_regression_train_validation_loss"],
        extra_cls_figures=["xgboost_feature_importance_classification", "xgb_classification_train_validation_loss"],
    )
    st.markdown("### SHAP preview")
    _shap_gallery(results, ["regression_beeswarm", "regression_global_bar"])

with tabs[5]:
    _section("Ce învață modelele din date?", "Compararea importanței predictorilor și interpretarea relațiilor predictive.")
    _figure_grid(results, ["permutation_heatmap_regression", "permutation_heatmap_classification"])
    st.markdown("### Predictori cu importanță predictivă stabilă între modele")
    _table(results, "importance_consensus", height=360)
    st.markdown("### SHAP regresie")
    _table(results, "shap_global_regression", height=300)
    _shap_gallery(
        results,
        [
            "regression_beeswarm",
            "regression_global_bar",
            "regression_dependence_control_corruption",
            "regression_dependence_female_vulnerable",
            "regression_dependence_internet",
            "regression_waterfall_low",
            "regression_waterfall_median",
            "regression_waterfall_high",
        ],
    )
    st.markdown("### SHAP clasificare")
    _table(results, "shap_global_classification", height=260)
    _shap_gallery(results, ["classification_classification_global_bar"])
    st.markdown("### Relația predictivă marginală estimată")
    pdp_files = [Path(name).stem for name in results.metadata.get("partial_dependence_files", [])]
    if pdp_files:
        _figure_grid(results, pdp_files)
    else:
        st.info("Partial Dependence nu este disponibil în outputurile curente.")

with tabs[6]:
    _section("Comparația performanței predictive")
    reg_model = results.metadata.get("best_regression_model", "n/a")
    cls_model = results.metadata.get("best_classification_model", "n/a")
    _metric_cards(
        [
            {"label": "Regresie", "value": str(reg_model), "caption": f"RMSE test {_format_value(results.metadata.get('best_regression_rmse'))}", "accent": "teal"},
            {"label": "Clasificare", "value": str(cls_model), "caption": f"F1 Macro {_format_value(results.metadata.get('best_classification_f1_macro'))}", "accent": "burgundy"},
            {"label": "Modele salvate", "value": _format_value(len(results.models)), "caption": "pipeline-uri și modele native", "accent": "gold"},
            {"label": "Grafice exportate", "value": _format_value(len(results.figures)), "caption": "Plotly JSON și SHAP", "accent": "plum"},
        ]
    )
    st.markdown("### Regresie")
    _table(
        results,
        "regression_leaderboard",
        [
            "Model",
            "RMSE Train",
            "RMSE Test",
            "MAE Test",
            "MedAE Test",
            "R² Test",
            "MAPE Test",
            "sMAPE Test",
            "Generalization Gap",
            "Temporal CV RMSE",
            "Group CV RMSE",
            "Fit time sec",
        ],
        height=360,
    )
    _figure_grid(
        results,
        [
            "leaderboard_regression_rmse",
            "leaderboard_regression_mae",
            "leaderboard_regression_r2",
            "leaderboard_train_vs_test_rmse",
            "leaderboard_regression_gap",
            "leaderboard_regression_cv_comparison",
            "leaderboard_actual_vs_predicted_all",
            "performance_by_year_heatmap",
            "performance_by_country_bar",
            "shock_period_errors",
            "generalization_map_regression",
            "complexity_performance",
        ],
    )
    st.markdown("### Clasificare")
    _table(
        results,
        "classification_leaderboard",
        [
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "Precision Macro",
            "Recall Macro",
            "F1 Macro",
            "F1 Weighted",
            "MCC",
            "Cohen Kappa",
            "ROC-AUC OvR",
            "Log Loss",
            "Temporal CV F1 Macro",
            "Group CV F1 Macro",
            "Fit time sec",
        ],
        height=340,
    )
    _figure_grid(
        results,
        [
            "leaderboard_classification_f1",
            "leaderboard_classification_balanced_accuracy",
            "leaderboard_classification_roc_auc",
            "leaderboard_classification_log_loss",
            "leaderboard_classification_cv_comparison",
            "classification_performance_by_class",
            "classification_brier_scores",
        ],
    )
    st.markdown("### Concluzii predictive")
    conclusions = results.metadata.get("final_conclusions", {})
    for key in ["regression", "regression_stability", "features", "periods", "classification", "classification_confusions", "general"]:
        if conclusions.get(key):
            st.markdown(f"- {conclusions[key]}")
