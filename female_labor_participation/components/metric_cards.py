"""Carduri metrice și componente pentru parcursul cercetării."""

from __future__ import annotations

import streamlit as st

from components.html import render_html


def render_metric_card(label: str, value: str, caption: str | None = None, accent: str = "gold") -> None:
    """Afișează un card metric compact."""

    caption_html = f"<p>{caption}</p>" if caption else ""
    render_html(
        f"""
        <div class="metric-card accent-{accent}">
            <span class="metric-label">{label}</span>
            <strong>{value}</strong>
            {caption_html}
        </div>
        """
    )


def render_metric_grid(metrics: list[dict[str, str]], columns: int = 4) -> None:
    """Afișează o grilă de carduri metrice."""

    if not metrics:
        return

    for start in range(0, len(metrics), columns):
        cols = st.columns(columns)
        for index, metric in enumerate(metrics[start : start + columns]):
            with cols[index]:
                render_metric_card(
                    label=metric.get("label", "Metric"),
                    value=metric.get("value", "De completat"),
                    caption=metric.get("caption"),
                    accent=metric.get("accent", "gold"),
                )


def render_status_badge(label: str, tone: str = "gold") -> str:
    """Returnează marcajul vizual al unui status."""

    return f'<span class="status-badge tone-{tone}">{label}</span>'


def render_research_journey(steps: list[dict[str, str]]) -> None:
    """Afișează parcursul cercetării ca secvență vizuală."""

    cards = []
    for index, step in enumerate(steps, start=1):
        cards.append(
            f"""
            <div class="journey-step">
                <div class="journey-index">{index:02d}</div>
                <h4>{step["step"]}</h4>
                <span>{step["status"]}</span>
            </div>
            """
        )

    render_html(
        f"""
        <div class="journey-grid">
            {''.join(cards)}
        </div>
        """
    )


def render_pillar_cards(pillars: list[dict[str, str]]) -> None:
    """Afișează cardurile pilonilor metodologici."""

    cols = st.columns(4)
    for col, pillar in zip(cols, pillars):
        with col:
            render_html(
                f"""
                <div class="pillar-card accent-{pillar.get("accent", "gold")}">
                    <h3>{pillar["title"]}</h3>
                    <p>{pillar["description"]}</p>
                </div>
                """
            )
