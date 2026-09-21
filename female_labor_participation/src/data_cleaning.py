"""Funcții de pregătire și curățare a datelor."""

from __future__ import annotations

import pandas as pd


def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Returnează o copie cu denumiri de coloane curățate de spații inutile."""

    cleaned = dataframe.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    return cleaned


def coerce_year_column(dataframe: pd.DataFrame, year_col: str) -> pd.DataFrame:
    """Returnează o copie în care coloana anului este numerică."""

    cleaned = dataframe.copy()
    if year_col in cleaned.columns:
        cleaned[year_col] = pd.to_numeric(cleaned[year_col], errors="coerce").astype("Int64")
    return cleaned


def build_transformation_plan() -> list[dict[str, str]]:
    """Returnează transformări pregătite, fără aplicare prematură."""

    return [
        {"Transformare": "logaritmare", "Stadiu": "de completat", "Observații": "Aplicabilă doar variabilelor strict pozitive."},
        {"Transformare": "standardizare", "Stadiu": "de completat", "Observații": "Utilă pentru comparații și modele predictive."},
        {"Transformare": "decalaje temporale", "Stadiu": "de completat", "Observații": "Necesită structură panel validă."},
        {"Transformare": "termeni de interacțiune", "Stadiu": "de completat", "Observații": "Vor fi definiți pe baza ipotezelor."},
        {"Transformare": "indicatori pentru crize", "Stadiu": "de completat", "Observații": "Marcaje pentru 2008–2009, 2020 și 2022–2023."},
    ]
