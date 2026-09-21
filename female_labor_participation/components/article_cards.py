"""Componente vizuale pentru articolele de literatură."""

from __future__ import annotations

from typing import Any

import streamlit as st

from components.html import render_html


def _lista(items: list[str]) -> None:
    """Afișează o listă compactă în markdown."""

    for item in items:
        st.markdown(f"- {item}")


def _afiseaza_badgeuri(articol: dict[str, Any]) -> None:
    """Afișează badge-urile principale ale articolului."""

    render_html(
        f"""
        <div class="article-badge-grid">
            <span>{articol["tara"]}</span>
            <span>{articol["perioada"]}</span>
            <span>{articol["metoda"]}</span>
        </div>
        """
    )


def _afiseaza_card_compact(titlu: str, continut: str, accent: str = "gold") -> None:
    """Afișează un card compact pentru taburile articolului."""

    render_html(
        f"""
        <div class="literature-compact-card accent-{accent}">
            <h3>{titlu}</h3>
            <p>{continut}</p>
        </div>
        """
    )


def _afiseaza_metric_compact(titlu: str, valoare: str, descriere: str, accent: str = "gold") -> None:
    """Afișează o metrică compactă pentru rezultatele articolului."""

    render_html(
        f"""
        <div class="literature-compact-metric accent-{accent}">
            <span>{titlu}</span>
            <strong>{valoare}</strong>
            <p>{descriere}</p>
        </div>
        """
    )


def afiseaza_articol(articol: dict[str, Any]) -> None:
    """Afișează un articol într-un card extensibil reutilizabil."""

    with st.expander(f"{articol['titlu']} · {articol['autori']} · {articol['an']}", expanded=True):
        st.markdown(f"### {articol['titlu']}")
        st.caption(f"{articol['autori']} · {articol['an']}")
        _afiseaza_badgeuri(articol)

        tab_date, tab_metoda, tab_rezultate = st.tabs(
            [
                "Date și variabile",
                "Metodologie",
                "Rezultate",
            ]
        )

        with tab_date:
            c1, c2, c3 = st.columns(3)
            with c1:
                _afiseaza_card_compact("Țară", articol["tara"], accent="plum")
            with c2:
                _afiseaza_card_compact("Perioadă", articol["perioada"], accent="gold")
            with c3:
                _afiseaza_card_compact("Frecvență", articol["frecventa"], accent="teal")

            c4, c5 = st.columns(2)
            with c4:
                _afiseaza_card_compact("Sursa datelor", articol["sursa_date"], accent="violet")
            with c5:
                _afiseaza_card_compact("Variabila explicativă", "; ".join(articol["variabile_explicative"]), accent="gold")

            st.markdown("**Definiții ale participării feminine**")
            _lista(articol["variabile_dependente"])

        with tab_metoda:
            c1, c2 = st.columns(2)
            with c1:
                _afiseaza_card_compact("Metodologie principală", articol["metoda"], accent="teal")
            with c2:
                _afiseaza_card_compact("Model selectat", articol["model"], accent="gold")

            st.markdown("**Teste și metode utilizate**")
            render_html(
                """
                <div class="method-badge-grid">
                    <span>ADF</span>
                    <span>ARDL(3,4)</span>
                    <span>Bounds Test</span>
                    <span>ECM</span>
                    <span>Breusch–Godfrey</span>
                    <span>Breusch–Pagan–Godfrey</span>
                    <span>Ramsey RESET</span>
                    <span>CUSUM</span>
                </div>
                """
            )
            _lista(articol["teste"])

        with tab_rezultate:
            metric_cols = st.columns(2)
            for col, coeficient in zip(metric_cols, articol["coeficienti"]):
                with col:
                    _afiseaza_metric_compact(coeficient["grup"], coeficient["valoare"], "coeficient estimat pe termen lung", accent="gold")

            st.markdown("Valorile reprezintă coeficienții estimați pe termen lung. Ele sunt tratate ca asocieri econometrice, nu ca efecte cauzale demonstrate.")
            _afiseaza_card_compact(
                "Termen scurt",
                "Efectul inflației nu este semnificativ statistic pe termen scurt.",
                accent="plum",
            )
            st.markdown("**Rezultate importante**")
            _lista(articol["rezultate"])
