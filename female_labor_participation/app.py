"""Pagina principală a aplicației de cercetare."""

from __future__ import annotations

import streamlit as st

from components.html import normalize_html, render_html
from components.sidebar import configure_page, inject_global_css, render_sidebar
from src.utils import PROJECT_TITLE_RO


ANALYSIS_SECTIONS = [
    {
        "number": "01",
        "title": "Literatura de specialitate",
        "description": "Studii, ipoteze, metode econometrice, rezultate și limite identificate în cercetările anterioare.",
        "accent": "teal",
    },
    {
        "number": "02",
        "title": "Date și variabile",
        "description": "Indicatorii utilizați, definițiile, sursele, acoperirea temporală și calitatea bazei de date.",
        "accent": "gold",
    },
    {
        "number": "03",
        "title": "Analiza descriptivă",
        "description": "Tendință centrală, dispersie, distribuții, valori extreme, corelații, valori lipsă și analiză geografică.",
        "accent": "plum",
    },
    {
        "number": "04",
        "title": "Analiza transversală",
        "description": "Diferențele dintre economiile europene și relațiile dintre participarea feminină și determinanții selectați.",
        "accent": "burgundy",
    },
    {
        "number": "05",
        "title": "Dinamica temporală",
        "description": "Evoluția în timp, persistența, rupturile structurale și efectele perioadelor de criză.",
        "accent": "teal",
    },
    {
        "number": "06",
        "title": "Machine Learning",
        "description": "Modele predictive, regularizare, validare și interpretarea importanței caracteristicilor.",
        "accent": "gold",
    },
    {
        "number": "07",
        "title": "Rezultate și concluzii",
        "description": "Sinteza rezultatelor, implicații economice, limite și direcții viitoare.",
        "accent": "navy",
    },
]

DATA_SUMMARY = [
    ("Perioada analizată", "2001–2023"),
    ("Frecvența", "Anuală"),
]

INDICATOR_GROUPS = [
    {
        "title": "Factori macroeconomici",
        "accent": "gold",
        "indicators": [
            ("PIB real pe locuitor, PPP", "NY.GDP.PCAP.PP.KD", None),
            ("Creșterea PIB real", "NY.GDP.MKTP.KD.ZG", None),
            ("Inflația, prețurile de consum", "FP.CPI.TOTL.ZG", None),
            ("Șomajul femeilor", "SL.UEM.TOTL.FE.ZS", None),
        ],
    },
    {
        "title": "Demografie și capital uman",
        "accent": "plum",
        "indicators": [
            ("Rata totală a fertilității", "SP.DYN.TFRT.IN", None),
            ("Urbanizarea", "SP.URB.TOTL.IN.ZS", None),
            ("Rata de dependență a copiilor", "SP.POP.DPND.YG", None),
            ("Înscrierea femeilor în învățământul terțiar", "SE.TER.ENRR.FE", None),
        ],
    },
    {
        "title": "Structura pieței muncii",
        "accent": "teal",
        "indicators": [
            ("Femei ocupate în servicii", "SL.SRV.EMPL.FE.ZS", None),
            ("Ocuparea vulnerabilă a femeilor", "SL.EMP.VULN.FE.ZS", None),
        ],
    },
    {
        "title": "Digitalizare și instituții",
        "accent": "burgundy",
        "indicators": [
            ("Utilizarea internetului", "IT.NET.USER.ZS", None),
            ("Femei în parlamentele naționale", "SG.GEN.PARL.ZS", None),
            ("Controlul corupției", "GOV_WGI_CC_EST", "Worldwide Governance Indicators / World Bank"),
        ],
    },
]

TIMELINE_MARKERS = [
    ("2001", "Începutul perioadei analizate", "teal"),
    ("2008–2009", "Criza financiară și economică globală", "burgundy"),
    ("2020", "Pandemia COVID-19", "plum"),
    ("2022–2023", "Șocul inflaționist și energetic", "gold"),
    ("2023", "Finalul perioadei analizate", "navy"),
]


def render_section_header(title: str, subtitle: str | None = None, wide_subtitle: bool = False) -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    extra_class = " wide-subtitle" if wide_subtitle else ""
    render_html(
        f"""
        <div class="home-section-header{extra_class}">
            <h2>{title}</h2>
            {subtitle_html}
        </div>
        """
    )


def render_analysis_card(numar: str, titlu: str, descriere: str, accent: str) -> None:
    render_html(
        f"""
        <article class="analysis-card accent-{accent}">
            <div class="analysis-number">{numar}</div>
            <h3>{titlu}</h3>
            <p>{descriere}</p>
        </article>
        """
    )


def render_data_mini_card(label: str, value: str) -> None:
    render_html(
        f"""
        <div class="data-mini-card">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
        """
    )


def indicator_row_html(name: str, code: str, source: str | None = None) -> str:
    source_html = f"<small>{source}</small>" if source else ""
    return normalize_html(
        f"""
        <div class="indicator-row">
            <span>{name}</span>
            <code>{code}</code>
            {source_html}
        </div>
        """
    )


def render_indicator_theme_card(title: str, accent: str, indicators: list[tuple[str, str, str | None]]) -> None:
    rows = "\n".join(indicator_row_html(name, code, source) for name, code, source in indicators)
    render_html(
        f"""
        <article class="indicator-card accent-{accent}">
            <h3>{title}</h3>
            <div class="indicator-list">{rows}</div>
        </article>
        """
    )


def timeline_item_html(year: str, label: str, tone: str) -> str:
    return normalize_html(
        f"""
        <article class="timeline-item timeline-tone-{tone}">
            <span class="timeline-dot"></span>
            <strong>{year}</strong>
            <p>{label}</p>
        </article>
        """
    )


def render_hero() -> None:
    render_html(
        f"""
        <section class="hero-panel hero-panel-compact home-hero">
            <h1>{PROJECT_TITLE_RO}</h1>
            <div class="hero-subtitle">Factori macroeconomici, demografici, structurali, digitali și instituționali</div>
        </section>
        """
    )


def render_research_summary() -> None:
    render_section_header("Despre cercetare")
    render_html(
        """
        <section class="home-research-summary">
            <p>
                Analiza urmărește modul în care condițiile macroeconomice, caracteristicile demografice,
                educația, structura pieței muncii, digitalizarea și calitatea instituțională se asociază
                cu participarea femeilor la forța de muncă în perioada 2001–2023. Cercetarea combină
                analiza descriptivă, comparațiile transversale, econometria datelor panel, dinamica temporală
                și Machine Learning.
            </p>
        </section>
        """
    )


def render_analysis_contents() -> None:
    render_section_header("Cuprinsul analizei")
    for start in range(0, len(ANALYSIS_SECTIONS), 4):
        cols = st.columns(4)
        for col, section in zip(cols, ANALYSIS_SECTIONS[start : start + 4]):
            with col:
                render_analysis_card(
                    section["number"],
                    section["title"],
                    section["description"],
                    section["accent"],
                )


def render_data_summary() -> None:
    cols = st.columns(len(DATA_SUMMARY))
    for col, (label, value) in zip(cols, DATA_SUMMARY):
        with col:
            render_data_mini_card(label, value)


def render_dependent_indicator() -> None:
    render_html(
        """
        <article class="indicator-card indicator-dependent accent-teal">
            <div class="indicator-card-header">
                <h3>Participarea femeilor</h3>
                <span class="dependent-badge">Variabilă dependentă</span>
            </div>
            <div class="indicator-row featured">
                <span>Rata participării femeilor la forța de muncă, 15–64 ani</span>
                <code>SL.TLF.ACTI.FE.ZS</code>
            </div>
            <p class="indicator-source">Sursă: World Development Indicators</p>
        </article>
        """
    )


def render_indicator_groups() -> None:
    for start in range(0, len(INDICATOR_GROUPS), 2):
        cols = st.columns(2, gap="large")
        for col, group in zip(cols, INDICATOR_GROUPS[start : start + 2]):
            with col:
                render_indicator_theme_card(group["title"], group["accent"], group["indicators"])


def render_data_and_indicators() -> None:
    render_section_header("Date și indicatori")
    render_data_summary()
    render_dependent_indicator()

    render_indicator_groups()


def render_context_timeline() -> None:
    markers_html = "\n".join(timeline_item_html(year, label, tone) for year, label, tone in TIMELINE_MARKERS)
    render_section_header(
        "Cronologie contextuală",
        "Perioada analizată surprinde mai multe șocuri economice și sociale majore, relevante pentru interpretarea evoluției participării femeilor pe piața muncii.",
        wide_subtitle=True,
    )
    render_html(
        f"""
        <section class="home-timeline">
            <div class="timeline-track">
                {markers_html}
            </div>
        </section>
        """
    )


configure_page("Acasă")
inject_global_css()
render_sidebar("Acasă")

render_hero()
render_research_summary()
render_analysis_contents()
render_data_and_indicators()
render_context_timeline()
