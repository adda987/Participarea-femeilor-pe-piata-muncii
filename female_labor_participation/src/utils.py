"""Project-wide constants and small utility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_TITLE_RO = "Determinanții participării femeilor pe piața muncii în economiile europene, 2001–2023"
PROJECT_SHORT_TITLE = "Participarea femeilor pe piața muncii"
AUTHOR = "Andreea-Daniela Sfetcu"
PERIOD = "2001–2023"
GEOGRAPHY = "Economii europene"


PALETTE: dict[str, str] = {
    "navy": "#071A2F",
    "navy_soft": "#10223A",
    "plum": "#6E2447",
    "burgundy": "#8A334E",
    "teal": "#0E6A68",
    "teal_soft": "#2F8F88",
    "gold": "#C99A45",
    "gold_soft": "#E6C879",
    "violet": "#5B4A86",
    "ivory": "#F7F2E8",
    "warm_gray": "#E9E2D6",
    "white": "#FFFFFF",
    "ink": "#172033",
    "muted": "#687083",
}


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical project paths used by pages and modules."""

    root: Path
    assets: Path
    data_raw: Path
    data_intermediate: Path
    data_processed: Path
    outputs: Path
    reports: Path
    literature: Path
    logo: Path
    css: Path


def get_project_paths() -> ProjectPaths:
    """Return the project path layout resolved from this file."""

    root = Path(__file__).resolve().parents[1]
    assets = root / "assets"
    return ProjectPaths(
        root=root,
        assets=assets,
        data_raw=root / "data" / "raw",
        data_intermediate=root / "data" / "intermediate",
        data_processed=root / "data" / "processed",
        outputs=root / "outputs",
        reports=root / "outputs" / "reports",
        literature=root / "literature",
        logo=assets / "logo_ads.png",
        css=assets / "styles.css",
    )


def planned_data_sources() -> list[dict[str, str]]:
    """Describe planned data-source families without claiming data availability."""

    return [
        {"Sursă": "Eurostat", "Acoperire": "piața muncii, demografie, educație, digitalizare", "Stadiu": "Planificat"},
        {"Sursă": "Banca Mondială / WDI", "Acoperire": "indicatori macroeconomici și structurali", "Stadiu": "Planificat"},
        {"Sursă": "OCDE", "Acoperire": "politici familiale, fiscalitate, instituții", "Stadiu": "Planificat"},
        {"Sursă": "OIM", "Acoperire": "indicatori comparabili ai ocupării și participării", "Stadiu": "Planificat"},
        {"Sursă": "Indicatorii de guvernanță mondială", "Acoperire": "guvernanță, controlul corupției", "Stadiu": "Planificat"},
    ]


def methodology_pillars() -> list[dict[str, str]]:
    """Return the four main methodological pillars requested for the home page."""

    return [
        {
            "title": "Analiză transversală",
            "description": "Comparații între economii europene pentru ani-cheie și blocuri tematice de predictori.",
            "accent": "plum",
        },
        {
            "title": "Econometria datelor panel",
            "description": "Modele pooled, efecte fixe, efecte aleatorii, diagnostice și verificări de robustețe.",
            "accent": "violet",
        },
        {
            "title": "Dinamică temporală și șocuri",
            "description": "Evoluții în timp, rupturi structurale și comparații pre/post crize majore.",
            "accent": "gold",
        },
        {
            "title": "Machine Learning",
            "description": "Modele predictive, validare temporală, interpretabilitate și comparație cu econometria.",
            "accent": "teal",
        },
    ]


def research_journey_steps() -> list[dict[str, str]]:
    """Return the planned research journey with non-result statuses."""

    return [
        {"step": "Definirea temei", "status": "etapă deschisă"},
        {"step": "Literatura de specialitate", "status": "de completat"},
        {"step": "Extragerea datelor", "status": "de completat"},
        {"step": "Curățarea datelor", "status": "de completat"},
        {"step": "Analiza descriptivă", "status": "de completat"},
        {"step": "Econometrie", "status": "de completat"},
        {"step": "Serii de timp", "status": "de completat"},
        {"step": "Machine Learning", "status": "de completat"},
        {"step": "Interpretare", "status": "de completat"},
        {"step": "Concluzii", "status": "de completat"},
    ]


def crisis_markers() -> list[dict[str, str]]:
    """Return major timeline markers used as context, not estimated effects."""

    return [
        {"year": "2001", "label": "începutul perioadei", "tone": "teal"},
        {"year": "2008–2009", "label": "criza financiară și economică", "tone": "plum"},
        {"year": "2020", "label": "pandemia COVID-19", "tone": "violet"},
        {"year": "2022–2023", "label": "șoc inflaționist și energetic", "tone": "gold"},
        {"year": "2023", "label": "finalul perioadei analizate", "tone": "navy"},
    ]


def metric_placeholders() -> list[dict[str, str]]:
    """Return home-dashboard metrics with non-fabricated placeholder values."""

    return [
        {"label": "Țări", "value": "De completat", "caption": "Baza de date va stabili acoperirea", "accent": "teal"},
        {"label": "Ani", "value": "De completat", "caption": "Perioada vizată: 2001–2023", "accent": "gold"},
        {"label": "Variabile candidate", "value": "De completat", "caption": "Dicționarul datelor este pregătit", "accent": "violet"},
        {"label": "Observații țară–an", "value": "De completat", "caption": "Va rezulta după integrarea datelor", "accent": "plum"},
        {"label": "Valori lipsă", "value": "De completat", "caption": "Va fi calculat din setul de date", "accent": "burgundy"},
        {"label": "Articole documentate", "value": "De completat", "caption": "Matricea manuală este pregătită", "accent": "teal"},
        {"label": "Stadiul general", "value": "Fază inițială", "caption": "Arhitectură pregătită pentru extindere", "accent": "gold"},
    ]


def literature_categories() -> list[str]:
    """Controlled category list for the literature review matrix."""

    return [
        "Inflație și factori macroeconomici",
        "Dezvoltare economică",
        "Fertilitate și demografie",
        "Educație",
        "Urbanizare",
        "Structura pieței muncii",
        "Digitalizare",
        "Instituții și guvernanță",
        "Politici familiale",
        "Criza financiară",
        "Pandemia COVID-19",
        "Modele panel",
        "Serii de timp",
        "Machine Learning",
    ]


def literature_schema() -> list[str]:
    """Return the literature matrix fields expected by the app."""

    return [
        "Titlul articolului",
        "Autor(i)",
        "An",
        "Țară sau eșantion",
        "Perioada analizată",
        "Frecvența datelor",
        "Variabila dependentă",
        "Variabile explicative",
        "Sursele de date",
        "Metodologia",
        "Întrebarea de cercetare",
        "Ipoteza",
        "Rezultatele principale",
        "Rezultatele pe termen scurt",
        "Rezultatele pe termen lung",
        "Teste / diagnostic",
        "Limitările declarate de autori",
        "Limitările identificate personal",
        "Observații critice",
        "Relevanța pentru proiect",
        "Categoria articolului",
        "Statusul lecturii",
        "Statusul notițelor",
        "Idei extrase",
        "Citarea bibliografică",
    ]


def legacy_literature_schema() -> list[str]:
    """Return older literature columns for compatibility with existing notes."""

    return [
        "Titlul articolului",
        "Autori",
        "An",
        "Țară sau eșantion",
        "Perioada analizată",
        "Frecvența datelor",
        "Metodologia",
        "Sursele de date",
        "Categoria de cercetare",
        "Statusul lecturii",
        "Statusul notițelor",
        "Întrebarea de cercetare",
        "Ipoteza",
        "Eșantionul",
        "Variabila dependentă",
        "Variabilele explicative",
        "Ecuațiile",
        "Testele aplicate",
        "Rezultatele principale",
        "Rezultatele pe termen scurt",
        "Rezultatele pe termen lung",
        "Testele de diagnostic",
        "Limitările declarate de autori",
        "Limitările identificate personal",
        "Observații critice",
        "Relevanța pentru proiect",
        "Idei preluate",
        "Citarea bibliografică",
    ]


def data_dictionary_template() -> list[dict[str, Any]]:
    """Returnează un dicționar editabil pentru variabilele planificate."""

    variable_names = [
        ("Rata participării femeilor pe piața muncii", "FLFP", "Variabilă dependentă", "participare pe piața muncii"),
        ("PIB pe locuitor", "GDP_PC", "Variabilă explicativă", "dezvoltare economică"),
        ("Rata inflației", "INFL", "Variabilă explicativă", "condiții macroeconomice"),
        ("Rata șomajului", "UNEMP", "Variabilă explicativă", "ciclu economic"),
        ("Rata șomajului în rândul femeilor", "F_UNEMP", "Variabilă explicativă", "piața muncii"),
        ("Rata fertilității", "FERT", "Variabilă explicativă", "demografie"),
        ("Populația de 65 de ani și peste", "POP65", "Variabilă explicativă", "structură demografică"),
        ("Educație terțiară în rândul femeilor", "F_TERT", "Variabilă explicativă", "educație"),
        ("Rata urbanizării", "URBAN", "Variabilă explicativă", "urbanizare"),
        ("Ocuparea în sectorul serviciilor", "SERV_EMP", "Variabilă explicativă", "structură sectorială"),
        ("Ocuparea cu timp parțial", "PART_TIME", "Variabilă explicativă", "structura muncii"),
        ("Utilizarea internetului", "INTERNET", "Variabilă explicativă", "digitalizare"),
        ("Competențe digitale", "DIG_SKILLS", "Variabilă explicativă", "capital uman digital"),
        ("Calitatea instituțională", "INST_QUAL", "Variabilă explicativă", "instituții"),
        ("Controlul corupției", "CORR_CTRL", "Variabilă explicativă", "guvernanță"),
        ("Statul de drept", "RULE_LAW", "Variabilă explicativă", "instituții"),
        ("Acoperirea serviciilor de îngrijire a copiilor", "CHILDCARE", "Variabilă explicativă", "politici familiale"),
        ("Concediul parental", "PARENT_LEAVE", "Variabilă explicativă", "politici familiale"),
        ("Povara fiscală a celui de-al doilea salariat", "SECOND_TAX", "Variabilă explicativă", "fiscalitate"),
        ("Variabilă indicator pentru criza financiară", "D_2008_2009", "Variabilă indicator", "șoc economic"),
        ("Variabilă indicator pentru pandemia COVID-19", "D_2020", "Variabilă indicator", "șoc pandemic"),
        ("Variabilă indicator pentru șocul inflaționist și energetic", "D_2022_2023", "Variabilă indicator", "șoc inflaționist și energetic"),
        ("Variabila dependentă decalată", "L1_FLFP", "Decalaj temporal", "dinamică temporală"),
        ("Termen de interacțiune", "INTERACTIUNE_DEFINITA_ULTERIOR", "Interacțiune", "ipoteză compusă"),
    ]

    return [
        {
            "denumirea variabilei": name,
            "denumirea scurtă": short,
            "codul indicatorului": "De completat",
            "definiția": "De completat",
            "unitatea de măsură": "De completat",
            "sursa": "De completat",
            "perioada disponibilă": "De completat",
            "frecvența": "anuală, de confirmat",
            "tipul variabilei": theme,
            "rolul în model": role,
            "transformarea aplicată": "De completat",
            "semnul economic așteptat": "De completat",
            "procentul valorilor lipsă": "De completat",
            "observații metodologice": "Observațiile metodologice vor fi stabilite în etapa de documentare.",
        }
        for name, short, role, theme in variable_names
    ]


def normalize_columns(columns: list[str]) -> list[str]:
    """Normalize column names for light validation and display."""

    return [str(col).strip() for col in columns]
