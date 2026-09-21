"""Funcții pentru analiza valorilor lipsă."""

from __future__ import annotations

import pandas as pd


def missing_value_summary(dataframe: pd.DataFrame | None) -> pd.DataFrame:
    """Returnează numărul și procentul valorilor lipsă pentru fiecare coloană."""

    if dataframe is None or dataframe.empty:
        return pd.DataFrame(columns=["variabilă", "valori lipsă", "procent valori lipsă"])

    missing_count = dataframe.isna().sum()
    missing_percent = (missing_count / len(dataframe) * 100).round(2)
    summary = pd.DataFrame(
        {
            "variabilă": missing_count.index,
            "valori lipsă": missing_count.values,
            "procent valori lipsă": missing_percent.values,
        }
    )
    return summary.sort_values("procent valori lipsă", ascending=False)


def variables_with_missing_values(dataframe: pd.DataFrame | None) -> list[str]:
    """Returnează variabilele care conțin cel puțin o valoare lipsă."""

    if dataframe is None or dataframe.empty:
        return []
    return dataframe.columns[dataframe.isna().any()].astype(str).tolist()
