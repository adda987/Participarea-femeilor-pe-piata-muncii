"""Pregătirea datelor pentru experimentul Machine Learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_project_paths
from src.ml.utils import TEST_END, TEST_START, TRAIN_END, TRAIN_START


VARIABLES = [
    ("female_lfpr", "SL.TLF.ACTI.FE.ZS", "Participarea femeilor la forța de muncă, 15–64 ani"),
    ("gdp_pc_ppp", "NY.GDP.PCAP.PP.KD", "PIB real pe locuitor, PPP"),
    ("gdp_growth", "NY.GDP.MKTP.KD.ZG", "Creșterea PIB real"),
    ("inflation", "FP.CPI.TOTL.ZG", "Inflația, prețurile de consum"),
    ("female_unemployment", "SL.UEM.TOTL.FE.ZS", "Șomajul femeilor"),
    ("fertility", "SP.DYN.TFRT.IN", "Rata totală a fertilității"),
    ("urbanization", "SP.URB.TOTL.IN.ZS", "Urbanizarea"),
    ("female_services", "SL.SRV.EMPL.FE.ZS", "Femei ocupate în servicii"),
    ("child_dependency", "SP.POP.DPND.YG", "Rata de dependență a copiilor"),
    ("female_vulnerable", "SL.EMP.VULN.FE.ZS", "Ocuparea vulnerabilă a femeilor"),
    ("internet", "IT.NET.USER.ZS", "Utilizarea internetului"),
    ("women_parliament", "SG.GEN.PARL.ZS", "Femei în parlamentele naționale"),
    ("female_tertiary", "SE.TER.ENRR.FE", "Înscrierea femeilor în învățământul terțiar"),
    ("control_corruption", "GOV_WGI_CC_EST", "Controlul corupției"),
]

FEATURE_COLUMNS = [
    "log_gdp_pc",
    "gdp_growth",
    "inflation",
    "female_unemployment",
    "fertility",
    "urbanization",
    "female_services",
    "child_dependency",
    "female_vulnerable",
    "internet",
    "women_parliament",
    "female_tertiary",
    "control_corruption",
]

FEATURE_LABELS = {
    "log_gdp_pc": "Log PIB/locuitor PPP",
    "gdp_growth": "Creștere PIB",
    "inflation": "Inflație",
    "female_unemployment": "Șomaj feminin",
    "fertility": "Fertilitate",
    "urbanization": "Urbanizare",
    "female_services": "Femei în servicii",
    "child_dependency": "Dependența copiilor",
    "female_vulnerable": "Ocupare vulnerabilă",
    "internet": "Utilizare internet",
    "women_parliament": "Femei în parlament",
    "female_tertiary": "Educație terțiară feminină",
    "control_corruption": "Controlul corupției",
}


@dataclass(frozen=True)
class MLExperimentData:
    """Seturile utilizate în experimentul ML."""

    data: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    feature_labels: dict[str, str]
    dropped_target_missing: int
    dataset_path: Path
    country_config_path: Path


def find_column_by_code(column_names: list[str], code: str) -> str:
    """Identifică o coloană WDI după codul indicatorului."""

    matches = [name for name in column_names if f"[{code}]" in name]
    if not matches:
        raise RuntimeError(f"Nu am găsit coloana pentru codul World Bank: {code}")
    return matches[0]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convertește o coloană la numeric, tratând markerii WDI."""

    return pd.to_numeric(series.replace("..", pd.NA), errors="coerce")


def load_ml_dataset() -> MLExperimentData:
    """Încarcă datele reale și construiește split-ul temporal ML."""

    paths = get_project_paths()
    dataset_path = paths.root / "data" / "P_Data_Extract_From_World_Development_Indicators.xlsx"
    country_config_path = paths.root / "config" / "europe_countries.csv"
    if not dataset_path.exists():
        raise RuntimeError(f"Fișierul Excel nu există: {dataset_path}")
    if not country_config_path.exists():
        raise RuntimeError(f"Fișierul config/europe_countries.csv nu există: {country_config_path}")

    raw_data = pd.read_excel(dataset_path, na_values=["", "NA", ".."])
    europe_countries = pd.read_csv(country_config_path)
    column_names = list(raw_data.columns)
    mapped = {slug: find_column_by_code(column_names, code) for slug, code, _ in VARIABLES}

    data = pd.DataFrame(
        {
            "Year": _coerce_numeric(raw_data["Time"]).astype("Int64"),
            "country_original": raw_data["Country Name"].astype(str),
            "ISO3": raw_data["Country Code"].astype(str),
        }
    )
    for slug, _, _ in VARIABLES:
        data[slug] = _coerce_numeric(raw_data[mapped[slug]])

    data = (
        data.loc[
            data["Year"].notna()
            & data["ISO3"].isin(europe_countries["iso3"])
            & data["Year"].between(TRAIN_START, TEST_END)
        ]
        .copy()
        .merge(europe_countries.rename(columns={"iso3": "ISO3"}), on="ISO3", how="left")
    )
    data["country"] = data["country"].fillna(data["country_original"])
    data["Year"] = data["Year"].astype(int)
    data["log_gdp_pc"] = np.where(data["gdp_pc_ppp"] > 0, np.log(data["gdp_pc_ppp"]), np.nan)

    before_target = len(data)
    data = data.loc[data["female_lfpr"].notna()].copy()
    dropped_target_missing = before_target - len(data)
    data = data.sort_values(["Year", "ISO3"]).reset_index(drop=True)

    train = data.loc[data["Year"].between(TRAIN_START, TRAIN_END)].copy()
    test = data.loc[data["Year"].between(TEST_START, TEST_END)].copy()
    if train.empty or test.empty:
        raise RuntimeError("Split-ul temporal 2001–2018 / 2019–2023 nu poate fi construit din datele disponibile.")

    return MLExperimentData(
        data=data,
        train=train,
        test=test,
        feature_columns=FEATURE_COLUMNS,
        feature_labels=FEATURE_LABELS,
        dropped_target_missing=dropped_target_missing,
        dataset_path=dataset_path,
        country_config_path=country_config_path,
    )
