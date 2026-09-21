"""Structuri pentru dinamica temporală și analiza șocurilor."""

from __future__ import annotations


def shock_periods() -> list[dict[str, str]]:
    """Returnează perioadele de șoc pentru analiza vizuală."""

    return [
        {"Perioadă": "2008–2009", "Descriere": "criza financiară și economică", "Stadiu": "de analizat"},
        {"Perioadă": "2020", "Descriere": "pandemia COVID-19", "Stadiu": "de analizat"},
        {"Perioadă": "2020–2021", "Descriere": "perioadă pandemică extinsă", "Stadiu": "de analizat"},
        {"Perioadă": "2022–2023", "Descriere": "șoc inflaționist și energetic", "Stadiu": "de analizat"},
    ]


def time_diagnostics_catalog() -> list[str]:
    """Returnează diagnosticele și modelele temporale planificate."""

    return [
        "teste de staționaritate",
        "ADF",
        "Phillips–Perron",
        "KPSS",
        "teste cu rupturi structurale",
        "ACF",
        "PACF",
        "modele ETS",
        "ARIMA",
        "modele ARDL, numai dacă datele permit",
        "prognoză",
        "intervale de predicție",
        "separare temporală între antrenare și testare",
        "evaluare cu origine mobilă",
    ]


def temporal_sections() -> list[str]:
    """Returnează secțiunile temporale planificate."""

    return [
        "evoluția mediei europene",
        "evoluția medianei europene",
        "evoluția pe regiuni",
        "comparația unor țări selectate",
        "trend",
        "sezonalitate, numai dacă frecvența datelor o permite",
        "persistență",
        "laguri",
        "rupturi structurale",
        "comparații pre-criză și post-criză",
        "modele de intervenție",
    ]
