"""Constante, căi și utilitare comune pentru experimentul Machine Learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import base64
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils import get_project_paths


RANDOM_STATE = 42
TRAIN_START = 2001
TRAIN_END = 2018
TEST_START = 2019
TEST_END = 2023

NAVY = "#10243E"
TEAL = "#147D7E"
BURGUNDY = "#7C2748"
PLUM = "#67406F"
GOLD = "#D3A13B"
IVORY = "#F7F3EA"
MUTED = "#687083"
WHITE = "#FFFFFF"

CLASS_COLORS = {
    "Participare scăzută": BURGUNDY,
    "Participare medie": GOLD,
    "Participare ridicată": TEAL,
}


@dataclass(frozen=True)
class ModelSpec:
    """Specificația unui model ML folosit în orchestrator."""

    name: str
    family: str
    task: str
    estimator: Any
    param_distributions: dict[str, list[Any]] | None = None
    search_iter: int = 0
    scoring: str = ""
    model_filename: str = ""


@dataclass(frozen=True)
class MLOutputPaths:
    """Căile outputurilor Machine Learning."""

    root: Path
    output_root: Path
    tables: Path
    figures: Path
    models: Path
    shap: Path
    metadata: Path


def ml_paths() -> MLOutputPaths:
    """Returnează căile standard pentru outputurile ML."""

    root = get_project_paths().root
    output_root = root / "outputs" / "machine_learning"
    return MLOutputPaths(
        root=root,
        output_root=output_root,
        tables=output_root / "tables",
        figures=output_root / "figures",
        models=output_root / "models",
        shap=output_root / "shap",
        metadata=output_root / "metadata",
    )


def ensure_output_dirs(paths: MLOutputPaths | None = None) -> MLOutputPaths:
    """Creează directoarele de output necesare."""

    paths = paths or ml_paths()
    for directory in [paths.tables, paths.figures, paths.models, paths.shap, paths.metadata]:
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Înlocuiește valorile infinite cu NA înainte de export."""

    cleaned = dataframe.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    return cleaned


def save_table(dataframe: pd.DataFrame, filename: str, paths: MLOutputPaths | None = None) -> Path:
    """Salvează un tabel CSV în directorul ML."""

    paths = ensure_output_dirs(paths)
    target = paths.tables / filename
    clean_dataframe(dataframe).to_csv(target, index=False, encoding="utf-8")
    return target


def to_json_safe(value: Any) -> Any:
    """Convertește obiecte NumPy/pandas în valori JSON serializabile."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_json_safe(item) for item in value.tolist()]
    if isinstance(value, (pd.Series, pd.Index)):
        return [to_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    if not isinstance(value, (list, dict, tuple, np.ndarray)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def save_metadata(metadata: dict[str, Any], paths: MLOutputPaths | None = None) -> Path:
    """Salvează metadata experimentului ML."""

    paths = ensure_output_dirs(paths)
    target = paths.metadata / "ml_analysis.json"
    target.write_text(json.dumps(to_json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _decode_plotly_typed_array(value: dict[str, Any]) -> list[Any] | dict[str, Any]:
    """Transformă array-urile Plotly base64 în liste JSON clasice."""

    if not {"dtype", "bdata"}.issubset(value):
        return value
    try:
        array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"])).copy()
        if value.get("shape"):
            shape = tuple(int(part.strip()) for part in str(value["shape"]).split(",") if part.strip())
            if shape:
                array = array.reshape(shape)
        return to_json_safe(array)
    except Exception:
        return value


def _plain_plotly_json(value: Any) -> Any:
    """Elimină obiectele NumPy/pandas și typed-array dintr-o figură Plotly."""

    if isinstance(value, dict):
        decoded = _decode_plotly_typed_array(value)
        if decoded is not value:
            return decoded
        return {str(key): _plain_plotly_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_plotly_json(item) for item in value]
    return to_json_safe(value)


def apply_project_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Aplică stilul vizual comun pentru grafice Plotly."""

    if title:
        fig.update_layout(title={"text": title, "x": 0.02, "xanchor": "left"})
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font={"family": "Inter, Source Sans Pro, sans-serif", "color": "#172033", "size": 12},
        title_font={"family": "Georgia, Times New Roman, serif", "size": 21, "color": NAVY},
        margin={"l": 48, "r": 28, "t": 74, "b": 48},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.28, "xanchor": "left", "x": 0, "font": {"color": NAVY}},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E8E2D8", zeroline=False, tickfont={"color": "#172033"}, title_font={"color": NAVY})
    fig.update_yaxes(showgrid=True, gridcolor="#E8E2D8", zeroline=False, tickfont={"color": "#172033"}, title_font={"color": NAVY})
    return fig


def save_figure(fig: go.Figure, filename: str, paths: MLOutputPaths | None = None) -> Path:
    """Salvează un grafic Plotly ca JSON, pentru randare fără recalculare."""

    paths = ensure_output_dirs(paths)
    target = paths.figures / filename
    payload = _plain_plotly_json(fig.to_plotly_json())
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def save_model(model: Any, filename: str, paths: MLOutputPaths | None = None) -> Path:
    """Salvează un pipeline sau model final."""

    paths = ensure_output_dirs(paths)
    target = paths.models / filename
    joblib.dump(model, target)
    return target


def set_matplotlib_cache(paths: MLOutputPaths | None = None) -> None:
    """Direcționează cache-ul matplotlib într-un folder editabil."""

    paths = ensure_output_dirs(paths)
    mpl_cache = paths.output_root / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
