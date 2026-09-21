"""Configurarea paginii, stilul global și navigarea laterală."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.html import render_css, render_html
from src.utils import PERIOD, PROJECT_SHORT_TITLE, get_project_paths


NAV_ITEMS = [
    {"key": "Acasă", "label": "Acasă", "path": "app.py"},
    {"key": "Literatura de specialitate", "label": "Literatura de specialitate", "path": "pages/01_Literature_Review.py"},
    {"key": "Date și variabile", "label": "Date și variabile", "path": "pages/02_Data_and_Variables.py"},
    {"key": "Analiza descriptivă", "label": "Analiza descriptivă", "path": "pages/03_Descriptive_Analysis.py"},
    {"key": "Analiza transversală", "label": "Analiza transversală", "path": "pages/04_Cross_Sectional_Analysis.py"},
    {"key": "Serii de timp și dinamică temporală", "label": "Serii de timp și dinamică temporală", "path": "pages/06_Time_Dynamics.py"},
    {"key": "Machine Learning", "label": "Machine Learning", "path": "pages/07_Machine_Learning.py"},
    {"key": "Rezultate și concluzii", "label": "Rezultate și concluzii", "path": "pages/08_Results_and_Conclusions.py"},
]


def configure_page(page_title: str) -> None:
    """Configurează pagina Streamlit cu o așezare lată."""

    st.set_page_config(
        page_title=f"{page_title} · {PROJECT_SHORT_TITLE}",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_global_css() -> None:
    """Încarcă stilul vizual comun."""

    css_path: Path = get_project_paths().css
    if css_path.exists():
        render_css(css_path.read_text(encoding="utf-8"))


def render_sidebar(current_page: str, show_period: bool = True) -> None:
    """Afișează navigarea laterală simplificată."""

    period_html = f"<p>{PERIOD}</p>" if show_period else ""
    with st.sidebar:
        render_html(
            f"""
            <div class="sidebar-brand">
                <div class="ads-monogram">
                    <span>ADS</span>
                    <i class="dot d1"></i><i class="dot d2"></i><i class="dot d3"></i>
                </div>
                <div>
                    <h1>{PROJECT_SHORT_TITLE}</h1>
                    {period_html}
                </div>
            </div>
            """
        )

        render_html('<div class="sidebar-nav-title">Navigare</div>')
        for item in NAV_ITEMS:
            if item["key"] == current_page:
                render_html(f'<div class="nav-active">{item["label"]}</div>')
            else:
                try:
                    st.page_link(item["path"], label=item["label"])
                except Exception:
                    render_html(f'<div class="nav-fallback">{item["label"]}</div>')
