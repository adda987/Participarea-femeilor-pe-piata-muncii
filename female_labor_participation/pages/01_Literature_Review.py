"""Pagina pentru literatura de specialitate."""

from __future__ import annotations

import streamlit as st

from components.article_cards import afiseaza_articol
from components.empty_states import render_empty_state
from components.html import render_html
from components.sidebar import configure_page, inject_global_css, render_sidebar
from src.literature_data import ARTICOLE


configure_page("Literatura de specialitate")
inject_global_css()
render_sidebar("Literatura de specialitate")

render_html(
    """
    <section class="literature-page-title">
        <h1>Literatura de specialitate</h1>
    </section>
    """
)

if not ARTICOLE:
    render_empty_state(
        "Nu există articole definite în backend",
        "Adaugă articole noi în lista ARTICOLE din src/literature_data.py.",
    )
else:
    for articol in ARTICOLE:
        afiseaza_articol(articol)
