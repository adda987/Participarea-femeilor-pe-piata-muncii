"""Analiză geografică pentru participarea femeilor pe piața muncii."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.correlation_analysis import (
    ANALYSIS_SLUGS,
    DEPENDENT_SLUG,
    COUNTRY_FIELD,
    ISO3_FIELD,
    YEAR_FIELD,
    prepare_correlation_dataset,
    variable_label,
)
from src.geo_config import (
    EUROPE_GEOJSON_PATH,
    GEOMETRY_ISO_COLUMN,
    GEOMETRY_NAME_COLUMN,
    country_label,
)
from src.utils import PALETTE


MAP_HOVER_SLUGS: list[tuple[str, str, str]] = [
    (DEPENDENT_SLUG, "Participarea femeilor", "%"),
    ("gdp_pc_ppp", "PIB/locuitor PPP", ""),
    ("inflation", "Inflația", "%"),
    ("female_unemployment", "Șomajul feminin", "%"),
    ("fertility", "Fertilitatea", ""),
    ("internet_use", "Utilizarea internetului", "%"),
    ("control_corruption", "Controlul corupției", ""),
]

PARTICIPATION_COLORSCALE = [
    [0.0, PALETTE["burgundy"]],
    [0.42, PALETTE["gold"]],
    [0.58, PALETTE["ivory"]],
    [1.0, PALETTE["teal"]],
]

DELTA_COLORSCALE = [
    [0.0, PALETTE["burgundy"]],
    [0.5, PALETTE["ivory"]],
    [1.0, PALETTE["teal"]],
]


@dataclass(frozen=True)
class GeographicAnalysisResult:
    """Datele agregate necesare hărților europene."""

    geojson: dict[str, Any]
    annual_2001: pd.DataFrame
    annual_2023: pd.DataFrame
    change_2001_2023: pd.DataFrame
    shared_min: float
    shared_max: float
    mapped_country_count: int


def load_europe_geometries(path: Path | None = None) -> gpd.GeoDataFrame:
    """Citește geometriile europene locale și păstrează cheia ISO3."""

    geo_path = path or EUROPE_GEOJSON_PATH
    if not geo_path.exists():
        return gpd.GeoDataFrame(columns=[GEOMETRY_ISO_COLUMN, GEOMETRY_NAME_COLUMN, "geometry"], geometry="geometry", crs="EPSG:4326")

    geometries = gpd.read_file(geo_path)
    if GEOMETRY_ISO_COLUMN not in geometries.columns:
        return gpd.GeoDataFrame(columns=[GEOMETRY_ISO_COLUMN, GEOMETRY_NAME_COLUMN, "geometry"], geometry="geometry", crs="EPSG:4326")

    if geometries.crs is None:
        geometries = geometries.set_crs("EPSG:4326")
    else:
        geometries = geometries.to_crs("EPSG:4326")

    geometries[GEOMETRY_ISO_COLUMN] = geometries[GEOMETRY_ISO_COLUMN].astype(str)
    return geometries[[GEOMETRY_ISO_COLUMN, GEOMETRY_NAME_COLUMN, "geometry"]].copy()


def geometries_to_geojson(geometries: gpd.GeoDataFrame) -> dict[str, Any]:
    """Transformă GeoDataFrame-ul într-un payload GeoJSON pentru Plotly."""

    if geometries.empty:
        return {"type": "FeatureCollection", "features": []}
    return json.loads(geometries.to_json())


def _with_country_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    """Adaugă denumiri românești pe baza codului ISO3."""

    labelled = dataset.copy()
    labelled[COUNTRY_FIELD] = [
        country_label(iso3, fallback)
        for iso3, fallback in zip(labelled[ISO3_FIELD].astype(str), labelled[COUNTRY_FIELD].astype(str))
    ]
    return labelled


def annual_country_indicators(dataset: pd.DataFrame, year: int) -> pd.DataFrame:
    """Agregă indicatorii pentru un an, la nivel de țară."""

    if dataset.empty:
        return pd.DataFrame(columns=[ISO3_FIELD, COUNTRY_FIELD, *ANALYSIS_SLUGS])

    selected = dataset[dataset[YEAR_FIELD].astype("Int64") == year].copy()
    if selected.empty:
        return pd.DataFrame(columns=[ISO3_FIELD, COUNTRY_FIELD, *ANALYSIS_SLUGS])

    annual = (
        selected.groupby([ISO3_FIELD, COUNTRY_FIELD], as_index=False)[ANALYSIS_SLUGS]
        .mean(numeric_only=True)
        .sort_values(COUNTRY_FIELD)
        .reset_index(drop=True)
    )
    return _with_country_labels(annual)


def prepare_geographic_analysis(dataframe: pd.DataFrame | None) -> GeographicAnalysisResult:
    """Pregătește datele geografice, valorile 2001/2023 și schimbarea temporală."""

    geometries = load_europe_geometries()
    geojson = geometries_to_geojson(geometries)
    dataset = prepare_correlation_dataset(dataframe)
    annual_2001 = annual_country_indicators(dataset, 2001)
    annual_2023 = annual_country_indicators(dataset, 2023)

    comparison_values = pd.concat(
        [annual_2001[[DEPENDENT_SLUG]], annual_2023[[DEPENDENT_SLUG]]],
        ignore_index=True,
    )[DEPENDENT_SLUG].dropna()
    if comparison_values.empty:
        shared_min, shared_max = 0.0, 100.0
    else:
        shared_min = float(comparison_values.min())
        shared_max = float(comparison_values.max())
        if shared_min == shared_max:
            shared_min = max(0.0, shared_min - 1.0)
            shared_max = min(100.0, shared_max + 1.0)

    change = annual_2001[[ISO3_FIELD, COUNTRY_FIELD, DEPENDENT_SLUG]].merge(
        annual_2023[[ISO3_FIELD, COUNTRY_FIELD, DEPENDENT_SLUG]],
        on=ISO3_FIELD,
        how="inner",
        suffixes=("_2001", "_2023"),
    )
    change[COUNTRY_FIELD] = [
        country_label(iso3, fallback)
        for iso3, fallback in zip(change[ISO3_FIELD].astype(str), change[f"{COUNTRY_FIELD}_2023"].astype(str))
    ]
    change = change.dropna(subset=[f"{DEPENDENT_SLUG}_2001", f"{DEPENDENT_SLUG}_2023"]).copy()
    change["delta_flfp"] = change[f"{DEPENDENT_SLUG}_2023"] - change[f"{DEPENDENT_SLUG}_2001"]

    mapped_codes = set(geometries[GEOMETRY_ISO_COLUMN].astype(str))
    mapped_country_count = int(annual_2023[annual_2023[ISO3_FIELD].isin(mapped_codes)][DEPENDENT_SLUG].notna().sum())
    return GeographicAnalysisResult(
        geojson=geojson,
        annual_2001=annual_2001,
        annual_2023=annual_2023,
        change_2001_2023=change,
        shared_min=shared_min,
        shared_max=shared_max,
        mapped_country_count=mapped_country_count,
    )


def _format_value(value: object, suffix: str = "", decimals: int = 2) -> str:
    """Formatează o valoare numerică pentru hover, fără a afișa NaN."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not isfinite(numeric):
        return ""
    return f"{numeric:,.{decimals}f}{suffix}"


def _indicator_hover(row: pd.Series) -> str:
    """Construiește textul hover pentru hărțile indicatorilor."""

    lines = [f"<b>{row[COUNTRY_FIELD]}</b>"]
    for slug, label, suffix in MAP_HOVER_SLUGS:
        if slug not in row:
            continue
        formatted = _format_value(row[slug], suffix=suffix)
        if formatted:
            lines.append(f"{label}: {formatted}")
    return "<br>".join(lines)


def _change_hover(row: pd.Series) -> str:
    """Construiește textul hover pentru harta schimbării."""

    lines = [f"<b>{row[COUNTRY_FIELD]}</b>"]
    values = [
        ("FLFP 2001", row.get(f"{DEPENDENT_SLUG}_2001"), "%"),
        ("FLFP 2023", row.get(f"{DEPENDENT_SLUG}_2023"), "%"),
        ("Schimbare", row.get("delta_flfp"), " p.p."),
    ]
    for label, value, suffix in values:
        formatted = _format_value(value, suffix=suffix)
        if formatted:
            lines.append(f"{label}: {formatted}")
    return "<br>".join(lines)


def _theme_geo_figure(fig: go.Figure, title: str, height: int) -> go.Figure:
    """Aplică tema proiectului unei hărți Plotly."""

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(255,255,255,0)",
        showcountries=True,
        countrycolor="#FFFFFF",
        countrywidth=0.6,
        lataxis={"range": [34, 72]},
        lonaxis={"range": [-25, 45]},
    )
    fig.update_layout(
        title={"text": title or ""},
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, Source Sans Pro, Arial, sans-serif", "color": PALETTE["ink"]},
        title_font={"family": "Georgia, Times New Roman, serif", "color": PALETTE["navy"], "size": 21},
        margin={"l": 8, "r": 8, "t": 56 if title else 12, "b": 8},
        hoverlabel={"bgcolor": PALETTE["navy"], "font_color": "#FFFFFF"},
        legend={"font": {"color": PALETTE["ink"]}},
    )
    return fig


def participation_map(
    geojson: dict[str, Any],
    annual_data: pd.DataFrame,
    title: str,
    cmin: float,
    cmax: float,
    height: int = 520,
) -> go.Figure:
    """Construiește o hartă choropleth pentru participarea femeilor."""

    plotting = annual_data.dropna(subset=[DEPENDENT_SLUG]).copy()
    plotting["hover"] = plotting.apply(_indicator_hover, axis=1)
    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            featureidkey=f"properties.{GEOMETRY_ISO_COLUMN}",
            locations=plotting[ISO3_FIELD].astype(str).tolist(),
            z=plotting[DEPENDENT_SLUG].astype(float).tolist(),
            zmin=cmin,
            zmax=cmax,
            colorscale=PARTICIPATION_COLORSCALE,
            marker_line_color="#FFFFFF",
            marker_line_width=0.6,
            colorbar={"title": "%"},
            text=plotting["hover"].tolist(),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    return _theme_geo_figure(fig, title, height)


def change_map(geojson: dict[str, Any], change_data: pd.DataFrame, height: int = 520) -> go.Figure:
    """Construiește harta schimbării 2001-2023."""

    plotting = change_data.dropna(subset=["delta_flfp"]).copy()
    plotting["hover"] = plotting.apply(_change_hover, axis=1)
    max_abs = float(np.nanmax(np.abs(plotting["delta_flfp"]))) if not plotting.empty else 1.0
    if not isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            featureidkey=f"properties.{GEOMETRY_ISO_COLUMN}",
            locations=plotting[ISO3_FIELD].astype(str).tolist(),
            z=plotting["delta_flfp"].astype(float).tolist(),
            zmin=-max_abs,
            zmax=max_abs,
            zmid=0,
            colorscale=DELTA_COLORSCALE,
            marker_line_color="#FFFFFF",
            marker_line_width=0.6,
            colorbar={"title": "p.p."},
            text=plotting["hover"].tolist(),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    return _theme_geo_figure(fig, "Schimbarea participării feminine, 2001–2023", height)


def coverage_summary_text(result: GeographicAnalysisResult) -> str:
    """Returnează un rezumat discret al acoperirii geografice pentru hartă."""

    return f"Harta 2023 utilizează geometrii locale pentru {result.mapped_country_count} țări cu valori reale."
