"""Funcții suport pentru analiza transversală."""

from __future__ import annotations

import pandas as pd


PREDEFINED_YEARS = [2007, 2009, 2019, 2020, 2023]


def model_blocks() -> list[dict[str, str]]:
    """Returnează blocurile de modele transversale planificate."""

    return [
        {"name": "Model macroeconomic", "description": "Inflație, dezvoltare economică, șomaj și ciclul economic."},
        {"name": "Model demografic", "description": "Fertilitate, structură pe vârste, urbanizare și educație."},
        {"name": "Model structural", "description": "Structura ocupării, servicii, part-time și organizarea muncii."},
        {"name": "Model digital și instituțional", "description": "Digitalizare, calitatea instituțiilor și guvernanță."},
        {"name": "Model de politici", "description": "Politici familiale, concediu parental, fiscalitatea celui de-al doilea salariat."},
        {"name": "Model complet", "description": "Specificație integrată după validarea datelor și a coliniarității."},
    ]


def filter_year(dataframe: pd.DataFrame | None, year_col: str | None, year: int) -> pd.DataFrame:
    """Returnează observațiile aferente unui an selectat."""

    if dataframe is None or dataframe.empty or not year_col or year_col not in dataframe.columns:
        return pd.DataFrame()
    years = pd.to_numeric(dataframe[year_col], errors="coerce")
    return dataframe.loc[years.eq(year)].copy()


def has_sufficient_observations(observation_count: int, predictor_count: int, buffer: int = 5) -> bool:
    """Verifică un prag conservator înaintea unei estimări viitoare."""

    if predictor_count <= 0:
        return False
    return observation_count >= predictor_count + buffer


def diagnostics_catalog() -> list[str]:
    """Returnează diagnosticele planificate pentru modelele transversale."""

    return [
        "VIF",
        "număr de condiționare",
        "heteroscedasticitate",
        "normalitatea reziduurilor",
        "observații influente",
        "pârghie statistică",
        "distanța Cook",
        "intervale de încredere",
        "prognoze",
        "comparație Ridge și Lasso",
    ]
