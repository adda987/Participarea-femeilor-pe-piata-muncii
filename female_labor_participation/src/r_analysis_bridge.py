"""Bridge minim între Streamlit/Python și analiza transversală rulată în R."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from src.utils import get_project_paths


TABLE_FILES = [
    "simple_coefficients.csv",
    "simple_model_metrics.csv",
    "simple_tests.csv",
    "simple_predictions.csv",
    "multiple_coefficients.csv",
    "multiple_model_metrics.csv",
    "multiple_tests.csv",
    "multiple_vif.csv",
    "multiple_residual_correlations.csv",
    "multiple_influence.csv",
    "robust_hc3_coefficients.csv",
    "dummy_counts.csv",
    "dummy_coefficients.csv",
    "dummy_tests.csv",
    "interaction1_coefficients.csv",
    "interaction2_coefficients.csv",
    "nonlinear_gdp_coefficients.csv",
    "scenario_predictions.csv",
    "ml_metrics.csv",
    "lasso_nonzero_coefficients.csv",
]

FIGURE_FILES = [
    "simple_scatter_regression.png",
    "simple_residuals_fitted.png",
    "simple_qq_plot.png",
    "simple_residual_histogram.png",
    "simple_residual_boxplot.png",
    "simple_residuals_predictor.png",
    "simple_cooks_distance.png",
    "multiple_coefficient_plot.png",
    "multiple_correlation_heatmap.png",
    "multiple_vif_bar.png",
    "multiple_residuals_fitted.png",
    "multiple_qq_plot.png",
    "multiple_residual_histogram.png",
    "multiple_scale_location.png",
    "multiple_cooks_distance.png",
    "multiple_leverage_studentized.png",
    "multiple_observed_fitted.png",
    "interaction1_plot.png",
    "interaction2_plot.png",
    "scenario_predictions.png",
    "ridge_cv_error.png",
    "lasso_cv_error.png",
    "ridge_coefficient_paths.png",
    "lasso_coefficient_paths.png",
    "ols_predicted_actual.png",
    "ridge_predicted_actual.png",
    "lasso_predicted_actual.png",
    "ml_rmse_comparison.png",
    "ml_mae_comparison.png",
    "lasso_nonzero_coefficients.png",
]


@dataclass(frozen=True)
class CrossSectionalResults:
    """Outputurile analizei transversale produse de scriptul R."""

    metadata: dict
    tables: dict[str, pd.DataFrame]
    figures: dict[str, Path]
    output_root: Path


def cross_sectional_paths() -> dict[str, Path]:
    """Returnează căile relevante pentru analiza transversală."""

    root = get_project_paths().root
    output_root = root / "outputs" / "cross_sectional"
    return {
        "root": root,
        "dataset": root / "data" / "P_Data_Extract_From_World_Development_Indicators.xlsx",
        "script": root / "R" / "analiza_transversala.R",
        "country_config": root / "config" / "europe_countries.csv",
        "output_root": output_root,
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "metadata": output_root / "metadata" / "analysis_metadata.json",
    }


def _required_output_paths(paths: dict[str, Path]) -> list[Path]:
    """Listează outputurile obligatorii citite de pagina Streamlit."""

    table_paths = [paths["tables"] / name for name in TABLE_FILES]
    figure_paths = [paths["figures"] / name for name in FIGURE_FILES]
    return [paths["metadata"], *table_paths, *figure_paths]


def _latest_input_mtime(paths: dict[str, Path]) -> float:
    """Returnează timestampul celei mai recente intrări relevante."""

    input_paths = [paths["dataset"], paths["script"], paths["country_config"]]
    existing = [path.stat().st_mtime for path in input_paths if path.exists()]
    return max(existing) if existing else 0.0


def _outputs_are_stale(paths: dict[str, Path]) -> bool:
    """Verifică dacă analiza R trebuie rerulată."""

    outputs = _required_output_paths(paths)
    if any(not path.exists() for path in outputs):
        return True
    oldest_output = min(path.stat().st_mtime for path in outputs)
    return oldest_output < _latest_input_mtime(paths)


def run_cross_sectional_analysis(force: bool = False) -> None:
    """Rulează scriptul R dacă outputurile lipsesc sau sunt mai vechi decât sursele."""

    paths = cross_sectional_paths()
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript nu este disponibil. Instalează R și asigură-te că Rscript este în PATH.")
    if not paths["dataset"].exists():
        raise RuntimeError(f"Fișierul Excel nu a fost găsit: {paths['dataset']}")
    if not paths["script"].exists():
        raise RuntimeError(f"Scriptul R nu a fost găsit: {paths['script']}")

    if not force and not _outputs_are_stale(paths):
        return

    completed = subprocess.run(
        [rscript, str(paths["script"])],
        cwd=paths["root"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "Scriptul R a eșuat fără mesaj suplimentar."
        raise RuntimeError(f"Analiza transversală R a eșuat: {details}")


def _read_table(path: Path) -> pd.DataFrame:
    """Citește un tabel CSV produs de R fără calcule suplimentare."""

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_cross_sectional_results(force: bool = False) -> CrossSectionalResults:
    """Rulează, dacă este necesar, și încarcă outputurile analizei transversale."""

    run_cross_sectional_analysis(force=force)
    paths = cross_sectional_paths()
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    tables = {name: _read_table(paths["tables"] / name) for name in TABLE_FILES}
    figures = {name: paths["figures"] / name for name in FIGURE_FILES if (paths["figures"] / name).exists()}
    return CrossSectionalResults(metadata=metadata, tables=tables, figures=figures, output_root=paths["output_root"])
