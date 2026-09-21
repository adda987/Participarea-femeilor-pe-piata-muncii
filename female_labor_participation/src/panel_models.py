"""Structuri pentru econometria datelor panel."""

from __future__ import annotations


def panel_tabs() -> list[str]:
    """Returnează etichetele taburilor de econometrie panel."""

    return [
        "MCMMP grupat",
        "Efecte fixe",
        "Efecte aleatorii",
        "Efecte fixe pe țară și an",
        "Compararea modelelor",
        "Diagnostice",
        "Verificări de robustețe",
    ]


def progressive_model_specs() -> list[str]:
    """Returnează specificațiile progresive planificate."""

    return [
        "model macroeconomic",
        "model demografic și educațional",
        "model privind structura pieței muncii",
        "model digital și instituțional",
        "model privind politicile familiale și fiscalitatea",
        "model general",
        "model parcimonios final",
    ]


def panel_diagnostics() -> list[str]:
    """Returnează testele și diagnosticele planificate pentru date panel."""

    return [
        "F-test",
        "Breusch–Pagan LM",
        "Hausman",
        "heteroscedasticitate",
        "autocorelare",
        "testul Wooldridge",
        "dependență transversală Pesaran CD",
        "multicoliniaritate",
        "VIF",
        "erori standard clusterizate pe țară",
        "erori Driscoll–Kraay",
        "sensibilitatea coeficienților",
        "eliminarea observațiilor influente",
        "analiza pe subperioade",
        "analiza pe regiuni",
    ]


def panel_output_fields() -> list[str]:
    """Returnează câmpurile așteptate pentru tabelele viitoare."""

    return [
        "ecuația modelului",
        "variabila dependentă",
        "predictorii selectați",
        "coeficienți",
        "erori standard",
        "statistici t sau z",
        "valori p",
        "intervale de încredere",
        "R pătrat în interiorul țărilor",
        "R pătrat între țări",
        "R pătrat total",
        "numărul observațiilor",
        "numărul țărilor",
        "numărul anilor",
        "efecte fixe de țară",
        "efecte fixe de an",
    ]
