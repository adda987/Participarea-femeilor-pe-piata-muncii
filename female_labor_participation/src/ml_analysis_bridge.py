"""Bridge pentru încărcarea rezultatelor Machine Learning în Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

from src.ml.utils import ml_paths


@dataclass(frozen=True)
class MLResults:
    """Rezultatele persistate ale experimentului Machine Learning."""

    output_root: Path
    tables: dict[str, pd.DataFrame]
    figures: dict[str, Path]
    shap_figures: dict[str, Path]
    models: list[Path]
    metadata: dict[str, Any]


def _input_paths() -> list[Path]:
    """Fișierele care pot invalida outputurile ML."""

    paths = ml_paths()
    ml_sources = sorted((paths.root / "src" / "ml").glob("*.py"))
    return [
        paths.root / "data" / "P_Data_Extract_From_World_Development_Indicators.xlsx",
        paths.root / "config" / "europe_countries.csv",
        *ml_sources,
    ]


def _latest_mtime(paths: list[Path]) -> float:
    """Returnează cea mai recentă dată de modificare pentru fișiere existente."""

    existing = [path.stat().st_mtime for path in paths if path.exists()]
    return max(existing) if existing else 0.0


def _outputs_are_current() -> bool:
    """Verifică dacă outputurile ML există și sunt mai noi decât sursele relevante."""

    paths = ml_paths()
    required = [
        paths.metadata / "ml_analysis.json",
        paths.tables / "regression_leaderboard.csv",
        paths.tables / "classification_leaderboard.csv",
        paths.figures / "train_test_timeline.json",
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False
    return min(path.stat().st_mtime for path in required) >= _latest_mtime(_input_paths())


def ensure_ml_outputs(force: bool = False) -> None:
    """Rulează pipeline-ul ML numai dacă outputurile lipsesc, sunt vechi sau se cere explicit."""

    if not force and _outputs_are_current():
        return
    paths = ml_paths()
    completed = subprocess.run(
        [sys.executable, "-m", "src.ml.model_comparison"],
        cwd=paths.root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "Pipeline-ul Machine Learning s-a încheiat cu eroare."
        raise RuntimeError(message)


def load_ml_results(force: bool = False) -> MLResults:
    """Încarcă tabelele, graficele și metadata pentru pagina Machine Learning."""

    ensure_ml_outputs(force=force)
    paths = ml_paths()
    metadata_path = paths.metadata / "ml_analysis.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tables = {path.stem: pd.read_csv(path) for path in sorted(paths.tables.glob("*.csv"))}
    figures = {path.stem: path for path in sorted(paths.figures.glob("*.json"))}
    shap_figures = {path.stem: path for path in sorted(paths.shap.glob("*.png"))}
    models = sorted(paths.models.glob("*"))
    return MLResults(
        output_root=paths.output_root,
        tables=tables,
        figures=figures,
        shap_figures=shap_figures,
        models=models,
        metadata=metadata,
    )
