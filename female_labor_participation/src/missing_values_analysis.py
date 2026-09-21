"""Diagnostic pentru valorile lipsă în eșantionul european."""

from __future__ import annotations

import pandas as pd

from src.correlation_analysis import (
    ANALYSIS_SLUGS,
    COUNTRY_FIELD,
    DEPENDENT_SLUG,
    SLUG_LABELS,
    YEAR_FIELD,
    prepare_correlation_dataset,
    variable_label,
)
from src.europe_config import ANALYSIS_END_YEAR, ANALYSIS_START_YEAR


YEARS = list(range(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR + 1))
MISSING_INVESTIGATION_THRESHOLD = 0.20


def _missing_signal(percent: float) -> str:
    """Returnează semnalul vizual asociat procentului de missing."""

    if percent < 0.05:
        return "Acoperire foarte bună"
    if percent < 0.15:
        return "De verificat"
    if percent < 0.25:
        return "Atenție"
    return "Acoperire problematică"


def prepare_missing_dataset(dataframe: pd.DataFrame | None) -> pd.DataFrame:
    """Pregătește datasetul european pentru diagnosticul valorilor lipsă."""

    return prepare_correlation_dataset(dataframe)


def general_missing_summary(dataset: pd.DataFrame) -> dict[str, int | float]:
    """Calculează rezumatul general al valorilor lipsă."""

    values = dataset[ANALYSIS_SLUGS]
    total_cells = int(values.shape[0] * values.shape[1])
    missing = int(values.isna().sum().sum())
    available = int(total_cells - missing)
    missing_percent = float(missing / total_cells) if total_cells else 0.0
    return {
        "total_cells": total_cells,
        "available": available,
        "missing": missing,
        "missing_percent": missing_percent,
    }


def missing_by_variable(dataset: pd.DataFrame) -> pd.DataFrame:
    """Calculează missingness pe variabilă."""

    rows: list[dict[str, object]] = []
    total = len(dataset)
    for slug in ANALYSIS_SLUGS:
        available_mask = dataset[slug].notna()
        available = int(available_mask.sum())
        missing = int(total - available)
        missing_percent = float(missing / total) if total else 0.0
        available_years = dataset.loc[available_mask, YEAR_FIELD].dropna()
        rows.append(
            {
                "slug": slug,
                "Variabilă": variable_label(slug),
                "Total posibile": total,
                "Disponibile": available,
                "Lipsă": missing,
                "% lipsă": missing_percent,
                "Primul an": int(available_years.min()) if not available_years.empty else None,
                "Ultimul an": int(available_years.max()) if not available_years.empty else None,
                "Semnal": _missing_signal(missing_percent),
            }
        )
    return pd.DataFrame(rows).sort_values("% lipsă", ascending=False).reset_index(drop=True)


def missing_by_country(dataset: pd.DataFrame) -> pd.DataFrame:
    """Calculează missingness pe țară."""

    rows: list[dict[str, object]] = []
    for country, group in dataset.groupby(COUNTRY_FIELD, sort=True):
        values = group[ANALYSIS_SLUGS]
        total = int(values.shape[0] * values.shape[1])
        missing = int(values.isna().sum().sum())
        available = int(total - missing)
        missing_percent = float(missing / total) if total else 0.0
        rows.append(
            {
                "Țară": country,
                "Total posibile": total,
                "Disponibile": available,
                "Lipsă": missing,
                "% lipsă": missing_percent,
                "Semnal": _missing_signal(missing_percent),
            }
        )
    return pd.DataFrame(rows).sort_values("% lipsă", ascending=False).reset_index(drop=True)


def country_variable_coverage(dataset: pd.DataFrame) -> pd.DataFrame:
    """Construiește matricea țară x variabilă cu procentul anilor disponibili."""

    rows: list[dict[str, object]] = []
    possible_years = len(YEARS)
    for country, group in dataset.groupby(COUNTRY_FIELD, sort=True):
        row: dict[str, object] = {"Țară": country}
        for slug in ANALYSIS_SLUGS:
            row[variable_label(slug)] = float(group[slug].notna().sum() / possible_years)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Țară")


def missing_over_time(dataset: pd.DataFrame) -> pd.DataFrame:
    """Calculează procentul valorilor lipsă pe an."""

    rows: list[dict[str, object]] = []
    for year in YEARS:
        group = dataset.loc[dataset[YEAR_FIELD].astype("Int64") == year]
        values = group[ANALYSIS_SLUGS]
        total = int(values.shape[0] * values.shape[1])
        missing = int(values.isna().sum().sum())
        rows.append(
            {
                "An": year,
                "% valori lipsă": float(missing / total) if total else 0.0,
                "Valori lipsă": missing,
                "Total": total,
            }
        )
    return pd.DataFrame(rows)


def dependent_variable_missing_summary(dataset: pd.DataFrame) -> dict[str, object]:
    """Analizează separat valorile lipsă pentru variabila dependentă."""

    series = dataset[DEPENDENT_SLUG]
    missing_mask = series.isna()
    countries = sorted(dataset.loc[missing_mask, COUNTRY_FIELD].dropna().unique().tolist())
    years = sorted(int(year) for year in dataset.loc[missing_mask, YEAR_FIELD].dropna().unique().tolist())
    total = int(len(series))
    missing = int(missing_mask.sum())
    available = int(total - missing)
    return {
        "available": available,
        "missing": missing,
        "missing_percent": float(missing / total) if total else 0.0,
        "countries": countries,
        "years": years,
    }


def dependent_availability_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    """Construiește matricea țară x an pentru disponibilitatea variabilei dependente."""

    working = dataset[[COUNTRY_FIELD, YEAR_FIELD, DEPENDENT_SLUG]].copy()
    working["disponibil"] = working[DEPENDENT_SLUG].notna().astype(int)
    matrix = working.pivot_table(index=COUNTRY_FIELD, columns=YEAR_FIELD, values="disponibil", aggfunc="max")
    return matrix.reindex(columns=YEARS).fillna(0).astype(int)


def missing_gap_summary(dataset: pd.DataFrame) -> dict[str, int]:
    """Numără golurile izolate și secvențele consecutive de valori lipsă."""

    isolated = 0
    two_year = 0
    three_plus = 0
    for _, group in dataset.groupby(COUNTRY_FIELD, sort=True):
        indexed = group.set_index(YEAR_FIELD).reindex(YEARS)
        for slug in ANALYSIS_SLUGS:
            missing_flags = indexed[slug].isna().tolist()
            run_length = 0
            for is_missing in missing_flags + [False]:
                if is_missing:
                    run_length += 1
                    continue
                if run_length == 1:
                    isolated += 1
                elif run_length == 2:
                    two_year += 1
                elif run_length >= 3:
                    three_plus += 1
                run_length = 0
    return {
        "isolated": isolated,
        "two_year": two_year,
        "three_plus": three_plus,
    }


def investigation_candidates(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Returnează variabilele și țările peste pragul orientativ de missingness."""

    variables = missing_by_variable(dataset)
    countries = missing_by_country(dataset)
    return {
        "variables": variables.loc[variables["% lipsă"] > MISSING_INVESTIGATION_THRESHOLD, ["Variabilă", "% lipsă"]].reset_index(drop=True),
        "countries": countries.loc[countries["% lipsă"] > MISSING_INVESTIGATION_THRESHOLD, ["Țară", "% lipsă"]].reset_index(drop=True),
    }

