"""Stări goale, antete de secțiune și note reutilizabile."""

from __future__ import annotations

from html import escape

from components.html import render_html


def render_section_header(title: str, subtitle: str | None = None, eyebrow: str | None = None) -> None:
    """Afișează un antet de secțiune coerent."""

    eyebrow_html = f'<span class="section-eyebrow">{escape(eyebrow)}</span>' if eyebrow else ""
    subtitle_html = f'<p class="section-subtitle">{escape(subtitle)}</p>' if subtitle else ""
    render_html(
        f"""
        <div class="section-header">
            {eyebrow_html}
            <h2>{escape(title)}</h2>
            {subtitle_html}
        </div>
        """
    )


def render_empty_state(
    title: str = "Secțiune pregătită",
    message: str = "Conținutul va fi completat în etapa metodologică relevantă.",
    detail: str | None = None,
) -> None:
    """Afișează o stare goală elegantă pentru secțiuni în lucru."""

    detail_html = f'<p class="empty-detail">{detail}</p>' if detail else ""
    render_html(
        f"""
        <div class="empty-state">
            <div class="empty-state-mark">
                <span></span><span></span><span></span>
            </div>
            <div>
                <h3>{title}</h3>
                <p>{message}</p>
                {detail_html}
            </div>
        </div>
        """
    )


def render_warning_box(message: str, title: str = "Notă metodologică") -> None:
    """Afișează o notă metodologică discretă."""

    render_html(
        f"""
        <div class="warning-box">
            <strong>{title}</strong>
            <p>{message}</p>
        </div>
        """
    )


def render_info_panel(title: str, body: str, accent: str = "gold") -> None:
    """Afișează un panou informativ."""

    render_html(
        f"""
        <div class="info-panel accent-{accent}">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """
    )


def render_tbd_list(items: list[str], title: str = "Elemente planificate") -> None:
    """Afișează o listă de elemente planificate."""

    items_html = "".join(f"<li>{item}</li>" for item in items)
    render_html(
        f"""
        <div class="tbd-list">
            <h3>{title}</h3>
            <ul>{items_html}</ul>
        </div>
        """
    )
