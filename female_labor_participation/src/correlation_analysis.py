"""Analize de corelație pentru eșantionul european."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from src.descriptive_statistics import (
    DEPENDENT_VARIABLE_COLUMN,
    filter_european_sample,
)
from src.europe_config import COUNTRY_CODE_COLUMN, COUNTRY_NAME_COLUMN, YEAR_COLUMN

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - fallback pentru medii fără scipy
    scipy_stats = None


COUNTRY_FIELD = "țară"
ISO3_FIELD = "cod_iso3"
YEAR_FIELD = "an"
DEPENDENT_SLUG = "female_lfpr"

ANALYSIS_VARIABLES: dict[str, dict[str, str]] = {
    DEPENDENT_VARIABLE_COLUMN: {
        "slug": DEPENDENT_SLUG,
        "label": "Participarea femeilor",
    },
    "GDP per capita, PPP (constant 2021 international $) [NY.GDP.PCAP.PP.KD]": {
        "slug": "gdp_pc_ppp",
        "label": "PIB/locuitor PPP",
    },
    "GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]": {
        "slug": "gdp_growth",
        "label": "Creștere PIB",
    },
    "Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]": {
        "slug": "inflation",
        "label": "Inflație",
    },
    "Unemployment, female (% of female labor force) (modeled ILO estimate) [SL.UEM.TOTL.FE.ZS]": {
        "slug": "female_unemployment",
        "label": "Șomaj feminin",
    },
    "Fertility rate, total (births per woman) [SP.DYN.TFRT.IN]": {
        "slug": "fertility",
        "label": "Fertilitate",
    },
    "Urban population (% of total population) [SP.URB.TOTL.IN.ZS]": {
        "slug": "urbanization",
        "label": "Urbanizare",
    },
    "Employment in services, female (% of female employment) (modeled ILO estimate) [SL.SRV.EMPL.FE.ZS]": {
        "slug": "female_services_employment",
        "label": "Femei în servicii",
    },
    "Age dependency ratio, young (% of working-age population) [SP.POP.DPND.YG]": {
        "slug": "child_dependency",
        "label": "Dependența copiilor",
    },
    "Vulnerable employment, female (% of female employment) (modeled ILO estimate) [SL.EMP.VULN.FE.ZS]": {
        "slug": "female_vulnerable_employment",
        "label": "Ocupare vulnerabilă",
    },
    "Individuals using the Internet (% of population) [IT.NET.USER.ZS]": {
        "slug": "internet_use",
        "label": "Utilizare internet",
    },
    "Proportion of seats held by women in national parliaments (%) [SG.GEN.PARL.ZS]": {
        "slug": "women_parliament",
        "label": "Femei în parlament",
    },
    "School enrollment, tertiary, female (% gross) [SE.TER.ENRR.FE]": {
        "slug": "female_tertiary_education",
        "label": "Educație terțiară",
    },
    "Control of Corruption - Governance estimate (approx. -2.5 to +2.5) [GOV_WGI_CC_EST]": {
        "slug": "control_corruption",
        "label": "Controlul corupției",
    },
}

SLUG_LABELS: dict[str, str] = {config["slug"]: config["label"] for config in ANALYSIS_VARIABLES.values()}
ANALYSIS_SLUGS: list[str] = [config["slug"] for config in ANALYSIS_VARIABLES.values()]
PREDICTOR_SLUGS: list[str] = [slug for slug in ANALYSIS_SLUGS if slug != DEPENDENT_SLUG]

SCATTER_RELATIONSHIPS: list[dict[str, str]] = [
    {"x": "gdp_pc_ppp", "title": "Participarea femeilor vs PIB/locuitor PPP"},
    {"x": "inflation", "title": "Participarea femeilor vs Inflație"},
    {"x": "fertility", "title": "Participarea femeilor vs Fertilitate"},
    {"x": "internet_use", "title": "Participarea femeilor vs Utilizarea internetului"},
]


@dataclass(frozen=True)
class CorrelationResult:
    """Rezultatul unei corelații pairwise."""

    coefficient: float
    p_value: float
    n: int


def variable_label(slug: str) -> str:
    """Returnează eticheta scurtă a unei variabile."""

    return SLUG_LABELS.get(slug, slug)


def analysis_variable_labels() -> list[str]:
    """Returnează etichetele variabilelor analizate."""

    return [variable_label(slug) for slug in ANALYSIS_SLUGS]


def prepare_correlation_dataset(dataframe: pd.DataFrame | None) -> pd.DataFrame:
    """Pregătește eșantionul european 2001-2023 cu variabile numerice redenumite."""

    filtered = filter_european_sample(dataframe)
    if filtered.empty:
        return pd.DataFrame(columns=[COUNTRY_FIELD, ISO3_FIELD, YEAR_FIELD, *ANALYSIS_SLUGS])

    available_columns = [column for column in ANALYSIS_VARIABLES if column in filtered.columns]
    rename_map = {column: ANALYSIS_VARIABLES[column]["slug"] for column in available_columns}
    working = filtered[[COUNTRY_NAME_COLUMN, COUNTRY_CODE_COLUMN, YEAR_COLUMN, *available_columns]].copy()
    working = working.rename(
        columns={
            COUNTRY_NAME_COLUMN: COUNTRY_FIELD,
            COUNTRY_CODE_COLUMN: ISO3_FIELD,
            YEAR_COLUMN: YEAR_FIELD,
            **rename_map,
        }
    )
    working[YEAR_FIELD] = pd.to_numeric(working[YEAR_FIELD], errors="coerce").astype("Int64")
    for slug in ANALYSIS_SLUGS:
        if slug in working.columns:
            working[slug] = pd.to_numeric(working[slug], errors="coerce")
        else:
            working[slug] = np.nan
    return working[[COUNTRY_FIELD, ISO3_FIELD, YEAR_FIELD, *ANALYSIS_SLUGS]].reset_index(drop=True)


def labelled_correlation_matrix(dataset: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Calculează matricea de corelații cu etichete scurte."""

    matrix = dataset[ANALYSIS_SLUGS].corr(method=method)
    return matrix.rename(index=SLUG_LABELS, columns=SLUG_LABELS)


def _pairwise_correlation(dataset: pd.DataFrame, x_slug: str, y_slug: str, method: str) -> CorrelationResult:
    """Calculează o corelație pairwise cu observațiile comune disponibile."""

    pair = dataset[[x_slug, y_slug]].dropna()
    if len(pair) < 3 or pair[x_slug].nunique() < 2 or pair[y_slug].nunique() < 2:
        return CorrelationResult(np.nan, np.nan, int(len(pair)))

    def standardize(values: np.ndarray) -> np.ndarray:
        centered = values - np.nanmean(values)
        std = np.nanstd(centered)
        return centered / std

    if method == "pearson":
        x_values = standardize(pair[x_slug].to_numpy(dtype=float))
        y_values = standardize(pair[y_slug].to_numpy(dtype=float))
    else:
        x_ranks = pair[x_slug].rank(method="average").to_numpy(dtype=float)
        y_ranks = pair[y_slug].rank(method="average").to_numpy(dtype=float)
        x_values = standardize(x_ranks)
        y_values = standardize(y_ranks)

    coefficient = float(np.corrcoef(x_values, y_values)[0, 1])
    if scipy_stats is None or not isfinite(coefficient):
        return CorrelationResult(coefficient, np.nan, int(len(pair)))

    denominator = max(1 - coefficient**2, np.finfo(float).eps)
    t_statistic = abs(coefficient) * np.sqrt((len(pair) - 2) / denominator)
    p_value = float(2 * scipy_stats.t.sf(t_statistic, df=len(pair) - 2))
    return CorrelationResult(float(coefficient), float(p_value), int(len(pair)))


def correlations_with_dependent(dataset: pd.DataFrame) -> pd.DataFrame:
    """Calculează Pearson, Spearman, p-values și N pentru predictorii disponibili."""

    rows: list[dict[str, object]] = []
    for slug in PREDICTOR_SLUGS:
        pearson = _pairwise_correlation(dataset, DEPENDENT_SLUG, slug, "pearson")
        spearman = _pairwise_correlation(dataset, DEPENDENT_SLUG, slug, "spearman")
        rows.append(
            {
                "Variabilă": variable_label(slug),
                "Pearson r": pearson.coefficient,
                "p-value Pearson": pearson.p_value,
                "Spearman ρ": spearman.coefficient,
                "p-value Spearman": spearman.p_value,
                "N": pearson.n,
                "_abs_pearson": abs(pearson.coefficient) if isfinite(pearson.coefficient) else -1,
            }
        )
    result = pd.DataFrame(rows).sort_values("_abs_pearson", ascending=False)
    return result.drop(columns=["_abs_pearson"]).reset_index(drop=True)


def high_predictor_correlations(dataset: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
    """Identifică perechile de predictori cu |r| peste prag."""

    corr = dataset[PREDICTOR_SLUGS].corr(method="pearson")
    rows: list[dict[str, object]] = []
    for i, first in enumerate(PREDICTOR_SLUGS):
        for second in PREDICTOR_SLUGS[i + 1 :]:
            coefficient = corr.loc[first, second]
            if pd.notna(coefficient) and abs(float(coefficient)) >= threshold:
                rows.append(
                    {
                        "Variabila 1": variable_label(first),
                        "Variabila 2": variable_label(second),
                        "Pearson r": float(coefficient),
                    }
                )
    result = pd.DataFrame(rows, columns=["Variabila 1", "Variabila 2", "Pearson r"])
    if result.empty:
        return result
    return result.sort_values("Pearson r", key=lambda series: series.abs(), ascending=False).reset_index(drop=True)


def pooled_within_between_correlations(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculează matricele Pearson pooled, within și between pentru date panel."""

    pooled = dataset[ANALYSIS_SLUGS].corr(method="pearson").rename(index=SLUG_LABELS, columns=SLUG_LABELS)

    numeric = dataset[[COUNTRY_FIELD, *ANALYSIS_SLUGS]].copy()
    country_means = numeric.groupby(COUNTRY_FIELD)[ANALYSIS_SLUGS].transform("mean")
    demeaned = numeric[ANALYSIS_SLUGS] - country_means
    within = demeaned.corr(method="pearson").rename(index=SLUG_LABELS, columns=SLUG_LABELS)

    between_means = numeric.groupby(COUNTRY_FIELD)[ANALYSIS_SLUGS].mean()
    between = between_means.corr(method="pearson").rename(index=SLUG_LABELS, columns=SLUG_LABELS)

    return {"Pooled": pooled, "Within": within, "Between": between}
