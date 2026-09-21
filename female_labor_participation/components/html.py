"""Utilitare sigure pentru randarea blocurilor HTML controlate."""

from __future__ import annotations

from textwrap import dedent

import streamlit as st


def normalize_html(markup: str) -> str:
    """Elimină indentarea care poate transforma HTML-ul în bloc Markdown de cod."""

    normalized = dedent(markup).strip()
    return "\n".join(line.rstrip() for line in normalized.splitlines() if line.strip())


def render_html(markup: str) -> None:
    """Randează HTML custom într-un singur apel Streamlit."""

    st.markdown(normalize_html(markup), unsafe_allow_html=True)


def render_css(css: str) -> None:
    """Injectează CSS-ul aplicației fără să îl afișeze ca text brut."""

    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)
