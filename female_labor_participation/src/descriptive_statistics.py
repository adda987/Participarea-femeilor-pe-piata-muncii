"""Funcții pentru statistici descriptive."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from src.europe_config import (
    ANALYSIS_END_YEAR,
    ANALYSIS_START_YEAR,
    COUNTRY_CODE_COLUMN,
    COUNTRY_NAME_COLUMN,
    EUROPE_COUNTRY_CODES,
    YEAR_COLUMN,
)


DEPENDENT_VARIABLE_COLUMN = (
    "Labor force participation rate, female (% of female population ages 15-64) "
    "(modeled ILO estimate) [SL.TLF.ACTI.FE.ZS]"
)
DEPENDENT_VARIABLE_LABEL = "Rata participării femeilor la forța de muncă, 15–64 ani"

COUNTRY_FIELD = "țară"
ISO3_FIELD = "cod_iso3"
YEAR_FIELD = "an"
VALUE_FIELD = "valoare"

CV_NEAR_ZERO_THRESHOLD = 0.1
MODE_MIN_SHARE = 0.01
SKEWNESS_SYMMETRY_THRESHOLD = 0.1
KURTOSIS_PRACTICAL_THRESHOLD = 0.5
REPRESENTATIVE_YEARS = [2001, 2007, 2009, 2019, 2020, 2023]


@dataclass(frozen=True)
class ModeResult:
    """Rezultatul calculului modului pentru o variabilă numerică."""

    status: str
    value: float | None = None


@dataclass(frozen=True)
class VariationResult:
    """Coeficient de variație cu protecție pentru medii apropiate de zero."""

    value: float | None
    label: str
    is_relevant: bool


@dataclass(frozen=True)
class OutlierSummary:
    """Rezumatul observațiilor identificate prin regula 1,5 x IQR."""

    lower_limit: float
    upper_limit: float
    count: int
    observations: pd.DataFrame


def numeric_columns(dataframe: pd.DataFrame | None) -> list[str]:
    """Returnează coloanele numerice disponibile pentru analiză."""

    if dataframe is None or dataframe.empty:
        return []
    return dataframe.select_dtypes(include=[np.number]).columns.astype(str).tolist()


def clean_numeric_series(values: pd.Series | list[float] | np.ndarray) -> pd.Series:
    """Returnează valorile numerice valide, fără NaN."""

    series = pd.Series(values)
    return pd.to_numeric(series, errors="coerce").dropna()


def mean_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează media."""

    series = clean_numeric_series(values)
    return float(series.mean()) if not series.empty else np.nan


def median_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează mediana."""

    series = clean_numeric_series(values)
    return float(series.median()) if not series.empty else np.nan


def mode_value(values: pd.Series | list[float] | np.ndarray) -> ModeResult:
    """Calculează modul fără a forța o valoare nerelevantă pentru date continue."""

    series = clean_numeric_series(values)
    if series.empty:
        return ModeResult("Fără mod relevant")

    counts = series.value_counts(dropna=True)
    max_count = int(counts.max())
    if max_count <= 1 or (max_count / len(series)) < MODE_MIN_SHARE:
        return ModeResult("Fără mod relevant")

    modes = counts[counts == max_count].index.astype(float).tolist()
    if len(modes) == 1:
        return ModeResult("Mod unic", float(modes[0]))
    return ModeResult("Mod multiplu")


def variance_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează varianța eșantionului, cu ddof=1."""

    series = clean_numeric_series(values)
    return float(series.var(ddof=1)) if len(series) > 1 else np.nan


def standard_deviation_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează deviația standard a eșantionului, cu ddof=1."""

    series = clean_numeric_series(values)
    return float(series.std(ddof=1)) if len(series) > 1 else np.nan


def minimum_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează minimul."""

    series = clean_numeric_series(values)
    return float(series.min()) if not series.empty else np.nan


def maximum_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează maximul."""

    series = clean_numeric_series(values)
    return float(series.max()) if not series.empty else np.nan


def range_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează amplitudinea."""

    min_val = minimum_value(values)
    max_val = maximum_value(values)
    return float(max_val - min_val) if isfinite(min_val) and isfinite(max_val) else np.nan


def q1_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează primul quartil."""

    series = clean_numeric_series(values)
    return float(series.quantile(0.25)) if not series.empty else np.nan


def q3_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează al treilea quartil."""

    series = clean_numeric_series(values)
    return float(series.quantile(0.75)) if not series.empty else np.nan


def iqr_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează intervalul interquartilic."""

    q1 = q1_value(values)
    q3 = q3_value(values)
    return float(q3 - q1) if isfinite(q1) and isfinite(q3) else np.nan


def coefficient_of_variation(values: pd.Series | list[float] | np.ndarray) -> VariationResult:
    """Calculează CV = deviație standard / |medie|, când media este interpretabilă."""

    mean = mean_value(values)
    std = standard_deviation_value(values)
    if not isfinite(mean) or abs(mean) < CV_NEAR_ZERO_THRESHOLD:
        return VariationResult(None, "Nerelevant – media este apropiată de zero", False)
    if not isfinite(std):
        return VariationResult(None, "Nerelevant", False)
    value = float(std / abs(mean))
    return VariationResult(value, format_percent(value), True)


def skewness_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează asimetria distribuției."""

    series = clean_numeric_series(values)
    return float(series.skew()) if len(series) > 2 else np.nan


def kurtosis_value(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează excess kurtosis, convenția Fisher: distribuția normală are 0."""

    series = clean_numeric_series(values)
    return float(series.kurt()) if len(series) > 3 else np.nan


def valid_observation_count(values: pd.Series | list[float] | np.ndarray) -> int:
    """Calculează numărul observațiilor valide."""

    return int(clean_numeric_series(values).shape[0])


def missing_observation_count(values: pd.Series | list[float] | np.ndarray) -> int:
    """Calculează numărul valorilor lipsă."""

    series = pd.Series(values)
    return int(pd.to_numeric(series, errors="coerce").isna().sum())


def missing_observation_percent(values: pd.Series | list[float] | np.ndarray) -> float:
    """Calculează procentul valorilor lipsă."""

    series = pd.Series(values)
    if series.empty:
        return np.nan
    return float(missing_observation_count(series) / len(series))


def filter_european_sample(dataframe: pd.DataFrame | None) -> pd.DataFrame:
    """Filtrează datasetul la țări europene individuale și perioada 2001-2023."""

    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    required_columns = {COUNTRY_CODE_COLUMN, COUNTRY_NAME_COLUMN, YEAR_COLUMN}
    if not required_columns.issubset(dataframe.columns):
        return pd.DataFrame()

    working = dataframe.copy()
    working[YEAR_COLUMN] = pd.to_numeric(working[YEAR_COLUMN], errors="coerce")
    country_codes = working[COUNTRY_CODE_COLUMN].astype(str).str.strip()
    years = working[YEAR_COLUMN]
    mask = (
        country_codes.isin(EUROPE_COUNTRY_CODES)
        & years.between(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR, inclusive="both")
    )
    return working.loc[mask].copy()


def dependent_variable_sample(dataframe: pd.DataFrame | None) -> pd.DataFrame:
    """Returnează eșantionul standardizat pentru variabila dependentă principală."""

    filtered = filter_european_sample(dataframe)
    if filtered.empty or DEPENDENT_VARIABLE_COLUMN not in filtered.columns:
        return pd.DataFrame(columns=[COUNTRY_FIELD, ISO3_FIELD, YEAR_FIELD, VALUE_FIELD])

    sample = filtered[[COUNTRY_NAME_COLUMN, COUNTRY_CODE_COLUMN, YEAR_COLUMN, DEPENDENT_VARIABLE_COLUMN]].copy()
    sample = sample.rename(
        columns={
            COUNTRY_NAME_COLUMN: COUNTRY_FIELD,
            COUNTRY_CODE_COLUMN: ISO3_FIELD,
            YEAR_COLUMN: YEAR_FIELD,
            DEPENDENT_VARIABLE_COLUMN: VALUE_FIELD,
        }
    )
    sample[YEAR_FIELD] = pd.to_numeric(sample[YEAR_FIELD], errors="coerce").astype("Int64")
    sample[VALUE_FIELD] = pd.to_numeric(sample[VALUE_FIELD], errors="coerce")
    return sample.sort_values([COUNTRY_FIELD, YEAR_FIELD]).reset_index(drop=True)


def valid_dependent_observations(sample: pd.DataFrame) -> pd.DataFrame:
    """Returnează observațiile valide pentru variabila dependentă."""

    if sample.empty or VALUE_FIELD not in sample.columns:
        return pd.DataFrame(columns=sample.columns)
    return sample.dropna(subset=[VALUE_FIELD]).copy()


def descriptive_summary(dataframe: pd.DataFrame | None) -> pd.DataFrame:
    """Calculează statisticile descriptive pentru coloanele numerice."""

    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    numeric_df = dataframe.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    stats = numeric_df.describe(percentiles=[0.25, 0.5, 0.75]).T
    stats = stats.rename(
        columns={
            "count": "număr de observații",
            "mean": "medie",
            "std": "deviație standard",
            "min": "minim",
            "25%": "Q1",
            "50%": "mediană",
            "75%": "Q3",
            "max": "maxim",
        }
    )
    stats["coeficient de variație"] = np.where(
        stats["medie"].abs() >= CV_NEAR_ZERO_THRESHOLD,
        stats["deviație standard"] / stats["medie"].abs(),
        np.nan,
    )
    stats = stats[
        [
            "număr de observații",
            "medie",
            "mediană",
            "deviație standard",
            "minim",
            "Q1",
            "Q3",
            "maxim",
            "coeficient de variație",
        ]
    ]
    return stats.reset_index(names="variabilă").round(4)


def dependent_variable_statistics(sample: pd.DataFrame) -> dict[str, Any]:
    """Calculează indicatorii descriptivi pentru variabila dependentă."""

    values = sample[VALUE_FIELD] if VALUE_FIELD in sample.columns else pd.Series(dtype=float)
    valid_values = clean_numeric_series(values)
    mode = mode_value(valid_values)
    cv = coefficient_of_variation(valid_values)
    q1 = q1_value(valid_values)
    q3 = q3_value(valid_values)
    iqr = iqr_value(valid_values)
    return {
        "mean": mean_value(valid_values),
        "median": median_value(valid_values),
        "mode": mode,
        "variance": variance_value(valid_values),
        "std": standard_deviation_value(valid_values),
        "min": minimum_value(valid_values),
        "max": maximum_value(valid_values),
        "range": range_value(valid_values),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "cv": cv,
        "skewness": skewness_value(valid_values),
        "kurtosis": kurtosis_value(valid_values),
        "valid_count": valid_observation_count(values),
        "missing_count": missing_observation_count(values),
        "missing_percent": missing_observation_percent(values),
    }


def outlier_summary(sample: pd.DataFrame) -> OutlierSummary:
    """Identifică observațiile peste limitele regulii 1,5 x IQR."""

    valid = valid_dependent_observations(sample)
    values = valid[VALUE_FIELD] if VALUE_FIELD in valid.columns else pd.Series(dtype=float)
    q1 = q1_value(values)
    q3 = q3_value(values)
    iqr = iqr_value(values)
    if not all(isfinite(value) for value in [q1, q3, iqr]):
        return OutlierSummary(np.nan, np.nan, 0, valid.iloc[0:0].copy())
    lower_limit = float(q1 - 1.5 * iqr)
    upper_limit = float(q3 + 1.5 * iqr)
    mask = (valid[VALUE_FIELD] < lower_limit) | (valid[VALUE_FIELD] > upper_limit)
    return OutlierSummary(lower_limit, upper_limit, int(mask.sum()), valid.loc[mask].copy())


def extremes_for_year(sample: pd.DataFrame, year: int = 2023, limit: int = 5) -> dict[str, pd.DataFrame]:
    """Returnează cele mai ridicate și scăzute valori observate într-un an."""

    valid = valid_dependent_observations(sample)
    if valid.empty:
        empty = valid.iloc[0:0].copy()
        return {"top": empty, "bottom": empty}
    year_data = valid.loc[valid[YEAR_FIELD].astype(int) == year].copy()
    if year_data.empty:
        empty = valid.iloc[0:0].copy()
        return {"top": empty, "bottom": empty}
    return {
        "top": year_data.nlargest(limit, VALUE_FIELD),
        "bottom": year_data.nsmallest(limit, VALUE_FIELD),
    }


def format_number(value: float | int | None, decimals: int = 2) -> str:
    """Formatează un număr pentru carduri."""

    if value is None or not isfinite(float(value)):
        return "—"
    return f"{float(value):,.{decimals}f}".replace(",", " ")


def format_percent(value: float | int | None, decimals: int = 2) -> str:
    """Formatează o proporție ca procent."""

    if value is None or not isfinite(float(value)):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def format_mode(mode: ModeResult, decimals: int = 2) -> str:
    """Formatează rezultatul modului."""

    if mode.value is not None:
        return format_number(mode.value, decimals)
    return mode.status


def interpret_skewness(skewness: float) -> str:
    """Generează interpretarea automată a asimetriei."""

    if not isfinite(skewness):
        return ""
    if abs(skewness) <= SKEWNESS_SYMMETRY_THRESHOLD:
        return "Distribuție aproximativ simetrică."
    if skewness > 0:
        return "Asimetrie pozitivă: coada distribuției este orientată spre valorile ridicate."
    return "Asimetrie negativă: coada distribuției este orientată spre valorile reduse."


def interpret_kurtosis(kurtosis: float) -> str:
    """Generează interpretarea automată a curtozei, folosind excess kurtosis."""

    if not isfinite(kurtosis):
        return ""
    if kurtosis > KURTOSIS_PRACTICAL_THRESHOLD:
        return "Excess kurtosis pozitivă: distribuția sugerează cozi relativ mai grele decât distribuția normală."
    if kurtosis < -KURTOSIS_PRACTICAL_THRESHOLD:
        return "Excess kurtosis negativă: distribuția sugerează cozi relativ mai ușoare decât distribuția normală."
    return "Curtoza este apropiată de cea a distribuției normale, în convenția excess kurtosis."


def distribution_interpretation(stats: dict[str, Any]) -> str:
    """Generează o interpretare scurtă pentru histogramă."""

    mean = stats.get("mean", np.nan)
    median = stats.get("median", np.nan)
    skewness = stats.get("skewness", np.nan)
    std = stats.get("std", np.nan)
    if not all(isfinite(float(value)) for value in [mean, median, skewness, std]) or std == 0:
        return ""

    if abs(mean - median) < 0.1 * std:
        return "Media și mediana sunt apropiate, ceea ce indică o distribuție relativ echilibrată în eșantion."
    if median > mean:
        return "Mediana este mai mare decât media, sugerând o ușoară orientare a distribuției spre valori reduse."
    return "Media este mai mare decât mediana, sugerând o ușoară orientare a distribuției spre valori ridicate."


def representative_years_interpretation(sample: pd.DataFrame) -> str:
    """Compară mediana distribuției între 2001 și 2023."""

    valid = valid_dependent_observations(sample)
    selected = valid[valid[YEAR_FIELD].isin(REPRESENTATIVE_YEARS)]
    medians = selected.groupby(YEAR_FIELD)[VALUE_FIELD].median()
    if 2001 not in medians.index or 2023 not in medians.index:
        return ""
    diff = float(medians.loc[2023] - medians.loc[2001])
    if abs(diff) < 0.5:
        return "Mediana din 2023 este apropiată de cea din 2001, diferența fiind sub 0,5 puncte procentuale."
    direction = "mai ridicată" if diff > 0 else "mai redusă"
    return f"Mediana din 2023 este {direction} decât cea din 2001 cu {format_number(abs(diff))} puncte procentuale."


def extremes_2023_interpretation(sample: pd.DataFrame) -> str:
    """Descrie intervalul valorilor observate în 2023."""

    extremes = extremes_for_year(sample, year=2023, limit=1)
    top = extremes["top"]
    bottom = extremes["bottom"]
    if top.empty or bottom.empty:
        return ""
    top_row = top.iloc[0]
    bottom_row = bottom.iloc[0]
    return (
        f"În 2023, valorile observate variază între {format_number(bottom_row[VALUE_FIELD])} "
        f"în {bottom_row[COUNTRY_FIELD]} și {format_number(top_row[VALUE_FIELD])} în {top_row[COUNTRY_FIELD]}."
    )


def correlation_matrix(dataframe: pd.DataFrame | None, method: str = "pearson") -> pd.DataFrame:
    """Calculează matricea de corelații Pearson sau Spearman."""

    if dataframe is None or dataframe.empty:
        return pd.DataFrame()
    numeric_df = dataframe.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(method=method)

