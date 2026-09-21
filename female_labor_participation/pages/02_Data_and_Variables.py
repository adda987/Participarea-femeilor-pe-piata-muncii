"""Pagina pentru date și variabile."""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from components.empty_states import render_empty_state, render_section_header
from components.html import render_html
from components.sidebar import configure_page, inject_global_css, render_sidebar
from components.tables import render_dataframe_table
from src.data_cleaning import coerce_year_column, standardize_column_names
from src.data_loader import find_project_dataset, load_local_dataset
from src.data_validation import infer_country_year_columns, panel_quality_summary
from src.utils import get_project_paths


VARIABLE_CARDS: list[dict[str, str]] = [
    {
        "theme": "Variabila analizată",
        "name": "Rata participării femeilor la forța de muncă, 15–64 ani",
        "code": "SL.TLF.ACTI.FE.ZS",
        "source": "World Development Indicators",
        "definition": "Ponderea femeilor de 15–64 ani care participă la forța de muncă.",
        "frequency": "Anuală",
        "type": "Indicator procentual",
        "role": "Variabilă dependentă",
        "notes": "Măsoară fenomenul central al cercetării.",
    },
    {
        "theme": "Factori macroeconomici",
        "name": "PIB real pe locuitor, PPP",
        "code": "NY.GDP.PCAP.PP.KD",
        "source": "World Development Indicators",
        "definition": "PIB pe locuitor exprimat la paritatea puterii de cumpărare, în dolari internaționali constanți.",
        "frequency": "Anuală",
        "type": "Indicator macroeconomic",
        "role": "Variabilă explicativă",
        "notes": "Proxy pentru nivelul dezvoltării economice.",
    },
    {
        "theme": "Factori macroeconomici",
        "name": "Creșterea PIB real",
        "code": "NY.GDP.MKTP.KD.ZG",
        "source": "World Development Indicators",
        "definition": "Rata anuală de creștere a PIB real.",
        "frequency": "Anuală",
        "type": "Indicator macroeconomic",
        "role": "Variabilă explicativă",
        "notes": "Surprinde dinamica activității economice.",
    },
    {
        "theme": "Factori macroeconomici",
        "name": "Inflația, prețurile de consum",
        "code": "FP.CPI.TOTL.ZG",
        "source": "World Development Indicators",
        "definition": "Rata anuală a inflației măsurată prin indicele prețurilor de consum.",
        "frequency": "Anuală",
        "type": "Indicator macroeconomic",
        "role": "Variabilă explicativă",
        "notes": "Reflectă presiuni asupra puterii de cumpărare.",
    },
    {
        "theme": "Factori macroeconomici",
        "name": "Șomajul femeilor",
        "code": "SL.UEM.TOTL.FE.ZS",
        "source": "World Development Indicators",
        "definition": "Ponderea femeilor șomere în forța de muncă feminină.",
        "frequency": "Anuală",
        "type": "Indicator al pieței muncii",
        "role": "Variabilă explicativă",
        "notes": "Controlează condițiile specifice pieței muncii feminine.",
    },
    {
        "theme": "Demografie și capital uman",
        "name": "Rata totală a fertilității",
        "code": "SP.DYN.TFRT.IN",
        "source": "World Development Indicators",
        "definition": "Numărul mediu de copii născuți de o femeie de-a lungul vieții.",
        "frequency": "Anuală",
        "type": "Indicator demografic",
        "role": "Variabilă explicativă",
        "notes": "Relevantă pentru constrângerile familiale și oferta de muncă.",
    },
    {
        "theme": "Demografie și capital uman",
        "name": "Urbanizarea",
        "code": "SP.URB.TOTL.IN.ZS",
        "source": "World Development Indicators",
        "definition": "Ponderea populației care locuiește în zone urbane.",
        "frequency": "Anuală",
        "type": "Indicator structural",
        "role": "Variabilă explicativă",
        "notes": "Aproximează accesul la servicii, educație și oportunități de muncă.",
    },
    {
        "theme": "Demografie și capital uman",
        "name": "Rata de dependență a copiilor",
        "code": "SP.POP.DPND.YG",
        "source": "World Development Indicators",
        "definition": "Raportul populației tinere dependente la populația de vârstă activă.",
        "frequency": "Anuală",
        "type": "Indicator demografic",
        "role": "Variabilă explicativă",
        "notes": "Indică presiunea potențială a responsabilităților de îngrijire.",
    },
    {
        "theme": "Demografie și capital uman",
        "name": "Înscrierea femeilor în învățământul terțiar",
        "code": "SE.TER.ENRR.FE",
        "source": "World Development Indicators",
        "definition": "Rata brută de înscriere a femeilor în învățământul terțiar.",
        "frequency": "Anuală",
        "type": "Indicator educațional",
        "role": "Variabilă explicativă",
        "notes": "Proxy pentru capitalul uman feminin.",
    },
    {
        "theme": "Structura pieței muncii",
        "name": "Femei ocupate în servicii",
        "code": "SL.SRV.EMPL.FE.ZS",
        "source": "World Development Indicators",
        "definition": "Ponderea femeilor ocupate în sectorul serviciilor.",
        "frequency": "Anuală",
        "type": "Indicator sectorial",
        "role": "Variabilă explicativă",
        "notes": "Reflectă structura sectorială a ocupării feminine.",
    },
    {
        "theme": "Structura pieței muncii",
        "name": "Ocuparea vulnerabilă a femeilor",
        "code": "SL.EMP.VULN.FE.ZS",
        "source": "World Development Indicators",
        "definition": "Ponderea femeilor în forme vulnerabile de ocupare.",
        "frequency": "Anuală",
        "type": "Indicator al calității ocupării",
        "role": "Variabilă explicativă",
        "notes": "Aproximează precaritatea ocupării feminine.",
    },
    {
        "theme": "Digitalizare și instituții",
        "name": "Utilizarea internetului",
        "code": "IT.NET.USER.ZS",
        "source": "World Development Indicators",
        "definition": "Ponderea populației care utilizează internetul.",
        "frequency": "Anuală",
        "type": "Indicator digital",
        "role": "Variabilă explicativă",
        "notes": "Proxy pentru accesul la infrastructură și competențe digitale.",
    },
    {
        "theme": "Digitalizare și instituții",
        "name": "Femei în parlamentele naționale",
        "code": "SG.GEN.PARL.ZS",
        "source": "World Development Indicators",
        "definition": "Ponderea locurilor parlamentare ocupate de femei.",
        "frequency": "Anuală",
        "type": "Indicator instituțional",
        "role": "Variabilă explicativă",
        "notes": "Reflectă reprezentarea politică a femeilor.",
    },
    {
        "theme": "Digitalizare și instituții",
        "name": "Controlul corupției",
        "code": "GOV_WGI_CC_EST",
        "source": "Worldwide Governance Indicators / World Bank",
        "definition": "Estimare a controlului corupției în cadrul indicatorilor de guvernanță.",
        "frequency": "Anuală",
        "type": "Indicator instituțional",
        "role": "Variabilă explicativă",
        "notes": "Surprinde calitatea mediului instituțional.",
    },
]

THEME_ACCENTS = {
    "Variabila analizată": "teal",
    "Factori macroeconomici": "gold",
    "Demografie și capital uman": "plum",
    "Structura pieței muncii": "violet",
    "Digitalizare și instituții": "burgundy",
}


def _column_for_code(dataframe: pd.DataFrame | None, code: str) -> str | None:
    if dataframe is None:
        return None
    matches = [column for column in dataframe.columns if f"[{code}]" in str(column)]
    return matches[0] if matches else None


def _missing_percent(dataframe: pd.DataFrame | None, code: str) -> str:
    column = _column_for_code(dataframe, code)
    if dataframe is None:
        return "Baza nu este încărcată"
    if column is None:
        return "Coloană absentă"
    if dataframe.empty:
        return "Nu există rânduri"
    percent = pd.to_numeric(dataframe[column], errors="coerce").isna().mean() * 100
    return f"{percent:.1f}%"


def _render_page_title() -> None:
    render_html(
        """
        <section class="data-page-title">
            <h1>Date și variabile</h1>
        </section>
        """
    )


def _render_variable_card(variable: dict[str, str], dataframe: pd.DataFrame | None) -> None:
    missing = _missing_percent(dataframe, variable["code"])
    accent = THEME_ACCENTS.get(variable["theme"], "gold")
    render_html(
        f"""
        <article class="variable-card accent-{accent}">
            <div class="variable-card-header">
                <div>
                    <span class="variable-theme">{escape(variable["theme"])}</span>
                    <h3>{escape(variable["name"])}</h3>
                </div>
                <code>{escape(variable["code"])}</code>
            </div>
            <p>{escape(variable["definition"])}</p>
            <div class="variable-meta-grid">
                <div><span>Sursă</span><strong>{escape(variable["source"])}</strong></div>
                <div><span>Frecvență</span><strong>{escape(variable["frequency"])}</strong></div>
                <div><span>Tipul variabilei</span><strong>{escape(variable["type"])}</strong></div>
                <div><span>Rolul în model</span><strong>{escape(variable["role"])}</strong></div>
                <div><span>Procent valori lipsă</span><strong>{missing}</strong></div>
                <div><span>Observații</span><strong>{escape(variable["notes"])}</strong></div>
            </div>
        </article>
        """
    )


def _render_variable_cards(dataframe: pd.DataFrame | None) -> None:
    render_section_header("Date")
    for start in range(0, len(VARIABLE_CARDS), 2):
        cols = st.columns(2, gap="large")
        for col, variable in zip(cols, VARIABLE_CARDS[start : start + 2]):
            with col:
                _render_variable_card(variable, dataframe)


def _render_preview_metric(label: str, value: str, accent: str) -> None:
    render_html(
        f"""
        <article class="data-preview-metric accent-{accent}">
            <span>{escape(label)}</span>
            <strong>{escape(value)}</strong>
        </article>
        """
    )


configure_page("Date și variabile")
inject_global_css()
render_sidebar("Date și variabile")
_render_page_title()

paths = get_project_paths()
detected_dataset = find_project_dataset()
dataframe = None
error_message = None
source_label = None

if detected_dataset is not None:
    try:
        dataframe = load_local_dataset(str(detected_dataset))
        dataframe = standardize_column_names(dataframe)
        source_label = str(detected_dataset.relative_to(paths.root))
    except Exception as exc:
        error_message = f"Fișierul detectat nu a putut fi citit: {exc}"

if dataframe is not None:
    inferred_country, inferred_year = infer_country_year_columns(dataframe)
    if inferred_year:
        dataframe = coerce_year_column(dataframe, inferred_year)
    st.session_state["panel_data"] = dataframe
    st.session_state["data_source_label"] = source_label
else:
    inferred_country = None
    inferred_year = None
    st.session_state.pop("panel_data", None)
    st.session_state.pop("data_source_label", None)

if error_message:
    st.error(error_message)

tab_preview, tab_data = st.tabs(["Previzualizare", "Date"])

with tab_preview:
    if dataframe is None:
        render_empty_state(
            "Previzualizare fără rânduri",
            "Configurația curentă nu a returnat un tabel pentru afișare.",
        )
    else:
        summary = panel_quality_summary(dataframe, inferred_country, inferred_year)
        metric_cols = st.columns(4)
        metrics = [
            ("Observații", str(summary["rows"]), "teal"),
            ("Coloane", str(summary["columns"]), "gold"),
            ("Țări", str(summary["countries"]), "plum"),
            ("Ani", str(summary["years"]), "violet"),
        ]
        for col, (label, value, accent) in zip(metric_cols, metrics):
            with col:
                _render_preview_metric(label, value, accent)
        render_section_header("Primele rânduri")
        render_dataframe_table(dataframe.head(20), empty_message="Nu există rânduri pentru previzualizare.", height=380)

with tab_data:
    _render_variable_cards(dataframe)
