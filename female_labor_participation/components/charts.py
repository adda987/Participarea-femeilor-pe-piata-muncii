"""Plotly chart helpers and non-data placeholders."""

from __future__ import annotations

from typing import Iterable

import plotly.graph_objects as go
import streamlit as st

from components.html import render_html
from src.utils import PALETTE


def apply_figure_theme(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    """Apply the app's visual language to a Plotly figure."""

    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, Source Sans Pro, Arial, sans-serif", "color": PALETTE["ink"]},
        title_font={"family": "Georgia, Times New Roman, serif", "color": PALETTE["navy"], "size": 21},
        margin={"l": 32, "r": 24, "t": 58 if title else 28, "b": 38},
        colorway=[PALETTE["navy"], PALETTE["teal"], PALETTE["gold"], PALETTE["plum"], PALETTE["violet"]],
        hoverlabel={"bgcolor": PALETTE["navy"], "font_color": "#FFFFFF"},
        legend={"font": {"color": PALETTE["ink"]}},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(7,26,47,0.08)", zeroline=False, tickfont={"color": PALETTE["ink"]}, title_font={"color": PALETTE["navy"]})
    fig.update_yaxes(showgrid=True, gridcolor="rgba(7,26,47,0.08)", zeroline=False, tickfont={"color": PALETTE["ink"]}, title_font={"color": PALETTE["navy"]})
    return fig


def render_timeline(markers: Iterable[dict[str, str]]) -> None:
    """Render a conceptual historical timeline with crisis markers."""

    marker_list = list(markers)
    x_positions = list(range(len(marker_list)))
    colors = [PALETTE.get(marker.get("tone", "gold"), PALETTE["gold"]) for marker in marker_list]
    labels = [f'{marker["year"]}<br>{marker["label"]}' for marker in marker_list]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=[1] * len(marker_list),
            mode="lines+markers+text",
            marker={"size": 18, "color": colors, "line": {"color": "#FFFFFF", "width": 2}},
            line={"color": PALETTE["gold"], "width": 4, "shape": "spline"},
            text=labels,
            textposition="top center",
            hoverinfo="text",
            hovertext=[f'{marker["year"]}: {marker["label"]}' for marker in marker_list],
        )
    )
    fig.update_yaxes(visible=False, range=[0.88, 1.2])
    fig.update_xaxes(visible=False, range=[-0.25, len(marker_list) - 0.75])
    fig = apply_figure_theme(fig, height=260)
    st.plotly_chart(fig, use_container_width=True)


def render_placeholder_chart(title: str, message: str = "Graficul este pregătit pentru etapa metodologică relevantă.") -> None:
    """Render an empty Plotly chart that keeps the page structure stable."""

    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 15, "color": PALETTE["ink"]},
        align="center",
    )
    fig.update_xaxes(showticklabels=False, title=None, range=[0, 1])
    fig.update_yaxes(showticklabels=False, title=None, range=[0, 1])
    fig = apply_figure_theme(fig, title=title, height=330)
    st.plotly_chart(fig, use_container_width=True)


def render_network_motif() -> None:
    """Render a subtle data-network motif for the home page."""

    render_html(
        """
        <div class="network-motif" aria-hidden="true">
            <span class="node node-a"></span>
            <span class="node node-b"></span>
            <span class="node node-c"></span>
            <span class="node node-d"></span>
            <span class="line line-ab"></span>
            <span class="line line-bc"></span>
            <span class="line line-cd"></span>
            <span class="gold-trace"></span>
        </div>
        """
    )
