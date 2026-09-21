"""Utilitare pentru citirea fișierelor CSV și Excel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils import get_project_paths


SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
PREFERRED_DATASET_NAMES = [
    "P_Data_Extract_From_World_Development_Indicators.xlsx",
    "panel_dataset.xlsx",
    "baza_date_panel.xlsx",
    "date_panel.xlsx",
    "dataset.xlsx",
    "panel_dataset.xls",
    "baza_date_panel.xls",
    "panel_dataset.csv",
]


def _configured_dataset_path() -> Path | None:
    """Returnează calea configurată în data/dataset_path.txt, dacă există."""

    paths = get_project_paths()
    config_file = paths.root / "data" / "dataset_path.txt"
    if not config_file.exists():
        return None
    raw_value = config_file.read_text(encoding="utf-8").strip()
    if not raw_value:
        return None
    configured = Path(raw_value)
    if not configured.is_absolute():
        configured = paths.root / configured
    if configured.exists() and configured.suffix.lower() in SUPPORTED_EXTENSIONS:
        return configured
    return None


def find_project_dataset() -> Path | None:
    """Găsește automat baza de date din folderele proiectului."""

    paths = get_project_paths()
    configured = _configured_dataset_path()
    if configured:
        return configured

    search_directories = [paths.root / "data", paths.data_processed, paths.data_raw]
    named_candidates = [directory / name for directory in search_directories for name in PREFERRED_DATASET_NAMES]
    for candidate in named_candidates:
        if candidate.exists():
            return candidate

    for extension in [".xlsx", ".xls", ".csv"]:
        for directory in search_directories:
            matches = sorted(directory.glob(f"*{extension}"))
            if matches:
                return matches[0]
    return None


def find_default_dataset() -> Path | None:
    """Păstrează compatibilitatea cu apelurile existente."""

    return find_project_dataset()


def _coerce_wdi_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Curăță markerul WDI pentru valori lipsă și convertește coloanele numerice."""

    cleaned = dataframe.copy().replace("..", pd.NA)
    for column in cleaned.columns:
        if cleaned[column].dtype != "object":
            continue
        converted = pd.to_numeric(cleaned[column], errors="coerce")
        non_missing = cleaned[column].notna()
        if converted.notna().sum() > 0 and converted.notna().sum() == non_missing.sum():
            cleaned[column] = converted
    return cleaned


@st.cache_data(show_spinner=False)
def load_local_dataset(path: str) -> pd.DataFrame:
    """Citește o bază locală dintr-o cale acceptată."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _coerce_wdi_missing_values(pd.read_csv(file_path))
    if suffix in {".xlsx", ".xls"}:
        return _coerce_wdi_missing_values(pd.read_excel(file_path))
    raise ValueError(f"Format neacceptat: {suffix}")


def get_session_dataset() -> pd.DataFrame | None:
    """Returnează baza curentă sau o citește automat din proiect."""

    dataframe = st.session_state.get("panel_data")
    if dataframe is not None:
        return dataframe

    dataset_path = find_project_dataset()
    if dataset_path is None:
        return None

    try:
        dataframe = load_local_dataset(str(dataset_path))
    except Exception:
        return None

    st.session_state["panel_data"] = dataframe
    try:
        st.session_state["data_source_label"] = str(dataset_path.relative_to(get_project_paths().root))
    except ValueError:
        st.session_state["data_source_label"] = str(dataset_path)
    return dataframe
