"""Utilitare de validare pentru baze de date panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


COUNTRY_CANDIDATES = ["Country", "country", "Country Name", "country_name", "Țară", "Tara", "ISO3", "geo"]
YEAR_CANDIDATES = ["Year", "year", "An", "an", "TIME_PERIOD", "time", "Time"]


@dataclass(frozen=True)
class ValidationMessage:
    """Un mesaj de validare afișat în raportul de calitate."""

    severity: str
    title: str
    detail: str


def infer_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    """Identifică o coloană pornind de la denumiri probabile."""

    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]
    return None


def infer_country_year_columns(dataframe: pd.DataFrame) -> tuple[str | None, str | None]:
    """Identifică automat coloanele pentru țară și an."""

    return infer_column(dataframe.columns, COUNTRY_CANDIDATES), infer_column(dataframe.columns, YEAR_CANDIDATES)


def validate_required_columns(dataframe: pd.DataFrame, country_col: str | None, year_col: str | None) -> list[ValidationMessage]:
    """Verifică existența coloanelor pentru țară și an."""

    messages: list[ValidationMessage] = []
    if country_col is None:
        messages.append(
            ValidationMessage(
                "error",
                "Coloana pentru țară nu a fost identificată",
                "Verifică denumirea coloanei de țară din fișierul sursă.",
            )
        )
    if year_col is None:
        messages.append(
            ValidationMessage(
                "error",
                "Coloana pentru an nu a fost identificată",
                "Verifică denumirea coloanei de an din fișierul sursă.",
            )
        )
    if not messages:
        messages.append(
            ValidationMessage(
                "success",
                "Structură minimă identificată",
                "Coloanele pentru țară și an sunt disponibile pentru validarea panelului.",
            )
        )
    return messages


def detect_duplicates(dataframe: pd.DataFrame, country_col: str, year_col: str) -> pd.DataFrame:
    """Returnează rândurile cu duplicate țară–an."""

    if country_col not in dataframe.columns or year_col not in dataframe.columns:
        return pd.DataFrame()
    duplicated_mask = dataframe.duplicated(subset=[country_col, year_col], keep=False)
    return dataframe.loc[duplicated_mask].sort_values([country_col, year_col])


def years_missing_by_country(dataframe: pd.DataFrame, country_col: str, year_col: str) -> pd.DataFrame:
    """Identifică anii lipsă în intervalul observat al fiecărei țări."""

    if country_col not in dataframe.columns or year_col not in dataframe.columns:
        return pd.DataFrame(columns=["Țară", "Ani lipsă"])

    working = dataframe[[country_col, year_col]].dropna().copy()
    working[year_col] = pd.to_numeric(working[year_col], errors="coerce")
    working = working.dropna(subset=[year_col])
    if working.empty:
        return pd.DataFrame(columns=["Țară", "Ani lipsă"])

    rows: list[dict[str, object]] = []
    for country, group in working.groupby(country_col):
        years = sorted(group[year_col].astype(int).unique().tolist())
        if not years:
            continue
        expected = set(range(min(years), max(years) + 1))
        missing = sorted(expected.difference(years))
        if missing:
            rows.append({"Țară": country, "Ani lipsă": ", ".join(str(year) for year in missing)})
    return pd.DataFrame(rows)


def coverage_by_country(dataframe: pd.DataFrame, country_col: str, year_col: str) -> pd.DataFrame:
    """Calculează acoperirea temporală pe țări."""

    if country_col not in dataframe.columns or year_col not in dataframe.columns:
        return pd.DataFrame()
    return (
        dataframe.groupby(country_col)[year_col]
        .agg(["nunique", "min", "max"])
        .rename(columns={"nunique": "ani disponibili", "min": "primul an", "max": "ultimul an"})
        .reset_index()
        .rename(columns={country_col: "Țară"})
        .sort_values("ani disponibili", ascending=False)
    )


def coverage_by_year(dataframe: pd.DataFrame, country_col: str, year_col: str) -> pd.DataFrame:
    """Calculează acoperirea pe țări pentru fiecare an."""

    if country_col not in dataframe.columns or year_col not in dataframe.columns:
        return pd.DataFrame()
    return (
        dataframe.groupby(year_col)[country_col]
        .nunique()
        .rename("țări disponibile")
        .reset_index()
        .rename(columns={year_col: "An"})
        .sort_values("An")
    )


def panel_quality_summary(dataframe: pd.DataFrame, country_col: str | None, year_col: str | None) -> dict[str, object]:
    """Returnează un rezumat al calității panelului fără estimări de model."""

    summary: dict[str, object] = {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "countries": "Nu a fost identificat",
        "years": "Nu a fost identificat",
        "duplicates": "Nu poate fi calculat",
    }
    if country_col and country_col in dataframe.columns:
        summary["countries"] = int(dataframe[country_col].nunique(dropna=True))
    if year_col and year_col in dataframe.columns:
        summary["years"] = int(dataframe[year_col].nunique(dropna=True))
    if country_col and year_col and country_col in dataframe.columns and year_col in dataframe.columns:
        summary["duplicates"] = int(dataframe.duplicated(subset=[country_col, year_col]).sum())
    return summary


def build_quality_report(dataframe: pd.DataFrame, country_col: str | None, year_col: str | None) -> str:
    """Generează un raport text privind calitatea datelor."""

    summary = panel_quality_summary(dataframe, country_col, year_col)
    messages = validate_required_columns(dataframe, country_col, year_col)
    lines = [
        "Raport privind calitatea datelor",
        "=================================",
        "",
        f"Rânduri: {summary['rows']}",
        f"Coloane: {summary['columns']}",
        f"Țări disponibile: {summary['countries']}",
        f"Ani disponibili: {summary['years']}",
        f"Duplicate țară-an: {summary['duplicates']}",
        "",
        "Mesaje de validare:",
    ]
    for message in messages:
        lines.append(f"- [{message.severity.upper()}] {message.title}: {message.detail}")

    if country_col and year_col and country_col in dataframe.columns and year_col in dataframe.columns:
        missing_years = years_missing_by_country(dataframe, country_col, year_col)
        lines.extend(["", "Ani lipsă pe țări:"])
        if missing_years.empty:
            lines.append("- Nu au fost identificați ani lipsă în intervalele observate.")
        else:
            for _, row in missing_years.iterrows():
                lines.append(f"- {row['Țară']}: {row['Ani lipsă']}")
    return "\n".join(lines)
