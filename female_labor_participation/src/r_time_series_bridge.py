"""Bridge între Streamlit/Python și analiza de serii de timp rulată în R."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from src.utils import get_project_paths


TABLE_FILES = [
    "europe_time_series.csv",
    "series_summary.csv",
    "trend_model.csv",
    "stationarity_level.csv",
    "stationarity_diff1.csv",
    "integration_summary.csv",
    "diff_shocks.csv",
    "arma_candidates.csv",
    "arima_candidates.csv",
    "selected_arima_coefficients.csv",
    "arima_diagnostics.csv",
    "smoothing_models.csv",
    "forecast_holdout.csv",
    "forecast_accuracy.csv",
    "rolling_origin_accuracy.csv",
    "future_forecast.csv",
    "multivariate_series.csv",
    "multivariate_stationarity.csv",
    "granger_results.csv",
    "var_summary.csv",
    "ardl_summary.csv",
]

MANDATORY_FIGURE_FILES = [
    "flfp_europe_evolution.png",
    "flfp_trend_linear.png",
    "acf_level.png",
    "pacf_level.png",
    "flfp_diff1.png",
    "acf_diff1.png",
    "pacf_diff1.png",
    "arima_information_criteria.png",
    "arima_residuals_time.png",
    "arima_residuals_acf.png",
    "arima_residuals_qq.png",
    "arima_residuals_histogram.png",
    "arima_observed_fitted.png",
    "arima_residuals_boxplot.png",
    "smoothing_ses.png",
    "smoothing_holt.png",
    "smoothing_ets.png",
    "smoothing_comparison.png",
    "forecast_arima_holdout.png",
    "forecast_model_comparison.png",
    "forecast_rmse_comparison.png",
    "forecast_mae_comparison.png",
    "forecast_future_2024_2026.png",
    "multivariate_standardized.png",
    "multivariate_ccf_flfp_inflation.png",
    "multivariate_scatter_flfp_inflation.png",
    "multivariate_flfp_inflation_standardized.png",
]

OPTIONAL_FIGURE_FILES = [
    "multivariate_irf_inflation_to_flfp.png",
]

FIGURE_FILES = [*MANDATORY_FIGURE_FILES, *OPTIONAL_FIGURE_FILES]


@dataclass(frozen=True)
class TimeSeriesResults:
    """Outputurile analizei de serii de timp produse de scriptul R."""

    metadata: dict
    tables: dict[str, pd.DataFrame]
    figures: dict[str, Path]
    output_root: Path


def time_series_paths() -> dict[str, Path]:
    """Returnează căile relevante pentru analiza de serii de timp."""

    root = get_project_paths().root
    output_root = root / "outputs" / "time_series"
    return {
        "root": root,
        "dataset": root / "data" / "P_Data_Extract_From_World_Development_Indicators.xlsx",
        "script": root / "R" / "analiza_serii_timp.R",
        "country_config": root / "config" / "europe_countries.csv",
        "output_root": output_root,
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "metadata": output_root / "metadata" / "time_series_analysis.json",
    }


def _required_output_paths(paths: dict[str, Path]) -> list[Path]:
    """Listează outputurile obligatorii pentru pagina Streamlit."""

    table_paths = [paths["tables"] / name for name in TABLE_FILES]
    figure_paths = [paths["figures"] / name for name in MANDATORY_FIGURE_FILES]
    return [paths["metadata"], *table_paths, *figure_paths]


def _latest_input_mtime(paths: dict[str, Path]) -> float:
    """Returnează timestampul celei mai recente intrări relevante."""

    input_paths = [paths["dataset"], paths["script"], paths["country_config"]]
    existing = [path.stat().st_mtime for path in input_paths if path.exists()]
    return max(existing) if existing else 0.0


def _outputs_are_stale(paths: dict[str, Path]) -> bool:
    """Verifică dacă analiza R trebuie rerulată."""

    outputs = _required_output_paths(paths)
    if any(not path.exists() or path.stat().st_size == 0 for path in outputs):
        return True
    oldest_output = min(path.stat().st_mtime for path in outputs)
    return oldest_output < _latest_input_mtime(paths)


def run_time_series_analysis(force: bool = False) -> None:
    """Rulează scriptul R de serii de timp dacă outputurile lipsesc sau sunt vechi."""

    paths = time_series_paths()
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
        raise RuntimeError(f"Analiza de serii de timp R a eșuat: {details}")


def _read_table(path: Path) -> pd.DataFrame:
    """Citește un CSV produs de R."""

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_time_series_results(force: bool = False) -> TimeSeriesResults:
    """Rulează, dacă este necesar, și încarcă outputurile analizei de serii de timp."""

    run_time_series_analysis(force=force)
    paths = time_series_paths()
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    tables = {name: _read_table(paths["tables"] / name) for name in TABLE_FILES}
    figures = {name: paths["figures"] / name for name in FIGURE_FILES if (paths["figures"] / name).exists()}
    return TimeSeriesResults(metadata=metadata, tables=tables, figures=figures, output_root=paths["output_root"])
