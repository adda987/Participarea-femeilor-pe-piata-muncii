"""Bridge între Streamlit/Python și analiza econometrică panel rulată în R."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from src.utils import get_project_paths


TABLE_FILES = [
    "panel_structure.csv",
    "within_between_variation.csv",
    "pooled_coefficients.csv",
    "fe_coefficients.csv",
    "re_coefficients.csv",
    "twoway_coefficients.csv",
    "model_comparison.csv",
    "panel_model_metrics.csv",
    "f_test.csv",
    "lm_test.csv",
    "hausman_test.csv",
    "time_effects_test.csv",
    "selection_tests.csv",
    "selected_model.csv",
    "panel_diagnostics.csv",
    "vif_panel.csv",
    "robust_cluster_coefficients.csv",
    "driscoll_kraay_coefficients.csv",
    "robust_inference_comparison.csv",
    "country_fixed_effects.csv",
    "time_fixed_effects.csv",
    "thematic_models_summary.csv",
    "subperiod_models_summary.csv",
    "panel_final_summary.csv",
    "panel_coefficient_interpretations.csv",
]

FIGURE_FILES = [
    "panel_coverage_heatmap.png",
    "within_between_variation.png",
    "country_mean_ci.png",
    "year_mean_ci.png",
    "female_lfpr_spaghetti.png",
    "representative_years_distribution.png",
    "pooled_fe_re_coefficient_plot.png",
    "observed_fitted_pooled.png",
    "observed_fitted_fe.png",
    "observed_fitted_re.png",
    "time_fixed_effects.png",
    "panel_residuals_fitted.png",
    "panel_qq_plot.png",
    "panel_residual_histogram.png",
    "panel_residual_boxplot.png",
    "panel_residuals_by_country.png",
    "panel_residuals_time.png",
    "panel_residuals_year_mean.png",
    "panel_observed_fitted_final.png",
    "vif_panel_bar.png",
    "country_fixed_effects.png",
    "robust_coefficient_plot.png",
    "coefficient_stability_specs.png",
    "coefficient_stability_subperiods.png",
]


@dataclass(frozen=True)
class PanelResults:
    """Outputurile analizei panel produse de scriptul R."""

    metadata: dict
    tables: dict[str, pd.DataFrame]
    figures: dict[str, Path]
    output_root: Path


def panel_paths() -> dict[str, Path]:
    """Returnează căile relevante pentru analiza panel."""

    root = get_project_paths().root
    output_root = root / "outputs" / "panel"
    return {
        "root": root,
        "dataset": root / "data" / "P_Data_Extract_From_World_Development_Indicators.xlsx",
        "script": root / "R" / "analiza_panel.R",
        "country_config": root / "config" / "europe_countries.csv",
        "output_root": output_root,
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "metadata": output_root / "metadata" / "panel_analysis.json",
    }


def _required_output_paths(paths: dict[str, Path]) -> list[Path]:
    """Listează outputurile obligatorii pentru pagina Streamlit."""

    table_paths = [paths["tables"] / name for name in TABLE_FILES]
    figure_paths = [paths["figures"] / name for name in FIGURE_FILES if name != "coefficient_stability_subperiods.png"]
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


def run_panel_analysis(force: bool = False) -> None:
    """Rulează scriptul R panel dacă outputurile lipsesc sau sunt mai vechi."""

    paths = panel_paths()
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
        raise RuntimeError(f"Analiza panel R a eșuat: {details}")


def _read_table(path: Path) -> pd.DataFrame:
    """Citește un CSV produs de R."""

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_panel_results(force: bool = False) -> PanelResults:
    """Rulează, dacă este necesar, și încarcă outputurile analizei panel."""

    run_panel_analysis(force=force)
    paths = panel_paths()
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    tables = {name: _read_table(paths["tables"] / name) for name in TABLE_FILES}
    figures = {name: paths["figures"] / name for name in FIGURE_FILES if (paths["figures"] / name).exists()}
    return PanelResults(metadata=metadata, tables=tables, figures=figures, output_root=paths["output_root"])
