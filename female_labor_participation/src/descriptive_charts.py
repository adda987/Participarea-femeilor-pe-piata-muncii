"""Grafice reutilizabile pentru analiza descriptivă."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.correlation_analysis import (
    COUNTRY_FIELD as CORR_COUNTRY_FIELD,
    DEPENDENT_SLUG,
    SCATTER_RELATIONSHIPS,
    SLUG_LABELS,
    YEAR_FIELD as CORR_YEAR_FIELD,
    variable_label,
)
from src.descriptive_statistics import (
    COUNTRY_FIELD,
    REPRESENTATIVE_YEARS,
    VALUE_FIELD,
    YEAR_FIELD,
    clean_numeric_series,
    extremes_for_year,
    valid_dependent_observations,
)
from src.utils import PALETTE


YEAR_COLORS = {
    "2001": PALETTE["navy"],
    "2007": PALETTE["teal"],
    "2009": PALETTE["gold"],
    "2019": PALETTE["plum"],
    "2020": PALETTE["burgundy"],
    "2023": PALETTE["violet"],
}


def _theme_figure(fig: go.Figure, title: str, height: int) -> go.Figure:
    """Aplică tema vizuală a proiectului unei figuri Plotly."""

    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, Source Sans Pro, Arial, sans-serif", "color": PALETTE["ink"]},
        title_font={"family": "Georgia, Times New Roman, serif", "color": PALETTE["navy"], "size": 21},
        margin={"l": 34, "r": 22, "t": 64, "b": 42},
        colorway=[PALETTE["navy"], PALETTE["teal"], PALETTE["gold"], PALETTE["plum"], PALETTE["burgundy"]],
        hoverlabel={"bgcolor": PALETTE["navy"], "font_color": "#FFFFFF"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"color": PALETTE["ink"]}},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(7,26,47,0.08)", zeroline=False, tickfont={"color": PALETTE["ink"]}, title_font={"color": PALETTE["navy"]})
    fig.update_yaxes(showgrid=True, gridcolor="rgba(7,26,47,0.08)", zeroline=False, tickfont={"color": PALETTE["ink"]}, title_font={"color": PALETTE["navy"]})
    return fig


def _density_curve(values: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """Calculează o curbă KDE gaussiană fără dependențe suplimentare."""

    series = clean_numeric_series(values)
    if len(series) < 3:
        return None
    std = float(series.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        return None

    data = series.to_numpy(dtype=float)
    bandwidth = 1.06 * std * (len(data) ** (-1 / 5))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return None

    padding = std * 0.4
    x_grid = np.linspace(float(data.min() - padding), float(data.max() + padding), 180)
    scaled = (x_grid[:, None] - data[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    return x_grid, density


def _value_axis_range(values: pd.Series) -> list[float]:
    """Returnează o scară procentuală stabilă pentru variabila analizată."""

    series = clean_numeric_series(values)
    if series.empty:
        return [0, 100]
    minimum = float(series.min())
    maximum = float(series.max())
    padding = max((maximum - minimum) * 0.08, 2.0)
    lower = max(0.0, np.floor((minimum - padding) / 5) * 5)
    upper = min(100.0, np.ceil((maximum + padding) / 5) * 5)
    if lower >= upper:
        return [0, 100]
    return [float(lower), float(upper)]


def _density_axis_range(densities: list[np.ndarray]) -> list[float]:
    """Returnează o scară coerentă pentru densități."""

    maxima = [float(np.nanmax(density)) for density in densities if len(density)]
    if not maxima:
        return [0, 1]
    return [0, max(maxima) * 1.18]


def histogram_with_density(sample: pd.DataFrame, stats: dict[str, object]) -> go.Figure:
    """Construiește histograma cu densitate, medie și mediană."""

    valid = valid_dependent_observations(sample)
    values = valid[VALUE_FIELD]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values.tolist(),
            nbinsx=28,
            histnorm="probability density",
            marker={"color": PALETTE["teal"], "line": {"color": "#FFFFFF", "width": 1}, "opacity": 0.78},
            name="Histogramă",
            hovertemplate="Interval: %{x}<br>Densitate: %{y:.3f}<extra></extra>",
        )
    )

    density = _density_curve(values)
    if density is not None:
        x_grid, y_grid = density
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=y_grid.tolist(),
                mode="lines",
                line={"color": PALETTE["navy"], "width": 3},
                name="Densitate",
                hovertemplate="Valoare: %{x:.2f}<br>Densitate: %{y:.3f}<extra></extra>",
            )
        )

    mean = stats.get("mean")
    median = stats.get("median")
    if isinstance(mean, (int, float)) and np.isfinite(mean):
        fig.add_vline(
            x=float(mean),
            line={"color": PALETTE["gold"], "width": 3, "dash": "dash"},
            annotation_text="Medie",
            annotation_position="top right",
        )
    if isinstance(median, (int, float)) and np.isfinite(median):
        fig.add_vline(
            x=float(median),
            line={"color": PALETTE["plum"], "width": 3, "dash": "dot"},
            annotation_text="Mediană",
            annotation_position="top left",
        )

    fig.update_xaxes(title="Rata participării (%)", range=_value_axis_range(values))
    if density is not None:
        fig.update_yaxes(title="Densitate (1 punct procentual)", range=_density_axis_range([density[1]]))
    else:
        fig.update_yaxes(title="Densitate (1 punct procentual)")
    return _theme_figure(fig, "Distribuția ratei participării femeilor pe piața muncii", 420)


def skewness_chart(sample: pd.DataFrame, stats: dict[str, object]) -> go.Figure:
    """Vizualizează asimetria prin densitate și poziția mediei față de mediană."""

    valid = valid_dependent_observations(sample)
    values = valid[VALUE_FIELD]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values.tolist(),
            nbinsx=24,
            histnorm="probability density",
            marker={"color": PALETTE["burgundy"], "line": {"color": "#FFFFFF", "width": 1}, "opacity": 0.58},
            name="Distribuție",
            hovertemplate="Valoare: %{x}<br>Densitate: %{y:.3f}<extra></extra>",
        )
    )

    density = _density_curve(values)
    if density is not None:
        x_grid, y_grid = density
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=y_grid.tolist(),
                mode="lines",
                line={"color": PALETTE["navy"], "width": 3},
                name="Densitate",
                hovertemplate="Valoare: %{x:.2f}<br>Densitate: %{y:.3f}<extra></extra>",
            )
        )

    mean = stats.get("mean")
    median = stats.get("median")
    if isinstance(mean, (int, float)) and np.isfinite(mean):
        fig.add_vline(
            x=float(mean),
            line={"color": PALETTE["gold"], "width": 3, "dash": "dash"},
            annotation_text="Medie",
            annotation_position="top left",
        )
    if isinstance(median, (int, float)) and np.isfinite(median):
        fig.add_vline(
            x=float(median),
            line={"color": PALETTE["teal"], "width": 3, "dash": "dot"},
            annotation_text="Mediană",
            annotation_position="top right",
        )

    fig.update_xaxes(title="Rata participării (%)", range=_value_axis_range(values))
    if density is not None:
        fig.update_yaxes(title="Densitate (1 punct procentual)", range=_density_axis_range([density[1]]))
    else:
        fig.update_yaxes(title="Densitate (1 punct procentual)")
    return _theme_figure(fig, "Skewness", 340)


def kurtosis_chart(sample: pd.DataFrame, stats: dict[str, object]) -> go.Figure:
    """Vizualizează curtoza prin comparația densității empirice cu o referință normală."""

    valid = valid_dependent_observations(sample)
    values = clean_numeric_series(valid[VALUE_FIELD])
    fig = go.Figure()

    density = _density_curve(values)
    if density is not None:
        x_grid, y_grid = density
        fig.add_trace(
            go.Scatter(
                x=x_grid.tolist(),
                y=y_grid.tolist(),
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(110,36,71,0.16)",
                line={"color": PALETTE["plum"], "width": 3},
                name="Densitate empirică",
                hovertemplate="Valoare: %{x:.2f}<br>Densitate: %{y:.3f}<extra></extra>",
            )
        )

        mean = stats.get("mean")
        std = stats.get("std")
        if isinstance(mean, (int, float)) and isinstance(std, (int, float)) and np.isfinite(mean) and np.isfinite(std) and std > 0:
            normal_density = np.exp(-0.5 * ((x_grid - float(mean)) / float(std)) ** 2) / (float(std) * np.sqrt(2 * np.pi))
            fig.add_trace(
                go.Scatter(
                    x=x_grid.tolist(),
                    y=normal_density.tolist(),
                    mode="lines",
                    line={"color": PALETTE["gold"], "width": 3, "dash": "dash"},
                    name="Referință normală",
                    hovertemplate="Valoare: %{x:.2f}<br>Densitate normală: %{y:.3f}<extra></extra>",
                )
            )

    fig.update_xaxes(title="Rata participării (%)", range=_value_axis_range(values))
    density_ranges = [density[1]] if density is not None else []
    if density is not None and "normal_density" in locals():
        density_ranges.append(normal_density)
    fig.update_yaxes(title="Densitate (1 punct procentual)", range=_density_axis_range(density_ranges))
    return _theme_figure(fig, "KURTOSIS", 340)


def general_box_plot(sample: pd.DataFrame) -> go.Figure:
    """Construiește box plot-ul general pentru variabila dependentă."""

    valid = valid_dependent_observations(sample)
    customdata = valid[[COUNTRY_FIELD, YEAR_FIELD]].astype(str).to_numpy()
    fig = go.Figure(
        go.Box(
            y=valid[VALUE_FIELD].tolist(),
            name="2001–2023",
            boxpoints="outliers",
            marker={"color": PALETTE["burgundy"], "size": 6, "opacity": 0.78},
            line={"color": PALETTE["navy"], "width": 2},
            fillcolor="rgba(201,154,69,0.28)",
            customdata=customdata,
            hovertemplate="Țară: %{customdata[0]}<br>An: %{customdata[1]}<br>Valoare: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_yaxes(title="Rata participării (%)", range=_value_axis_range(valid[VALUE_FIELD]), ticksuffix="%")
    fig.update_xaxes(title="")
    return _theme_figure(fig, "Box plot – participarea femeilor pe piața muncii", 420)


def representative_years_box_plot(sample: pd.DataFrame) -> go.Figure:
    """Compară distribuția variabilei dependente în ani reprezentativi."""

    valid = valid_dependent_observations(sample)
    selected = valid[valid[YEAR_FIELD].isin(REPRESENTATIVE_YEARS)].copy()
    selected[YEAR_FIELD] = selected[YEAR_FIELD].astype(str)
    fig = go.Figure()
    for year in [str(year) for year in REPRESENTATIVE_YEARS]:
        year_data = selected[selected[YEAR_FIELD] == year]
        if year_data.empty:
            continue
        fig.add_trace(
            go.Box(
                x=year_data[YEAR_FIELD].tolist(),
                y=year_data[VALUE_FIELD].tolist(),
                name=year,
                boxpoints="outliers",
                marker={"color": YEAR_COLORS.get(year, PALETTE["teal"]), "size": 5, "opacity": 0.72},
                line={"color": YEAR_COLORS.get(year, PALETTE["teal"]), "width": 2},
                customdata=year_data[[COUNTRY_FIELD]].astype(str).to_numpy(),
                hovertemplate="Țară: %{customdata[0]}<br>An: %{x}<br>Valoare: %{y:.2f}%<extra></extra>",
            )
        )
    fig.update_xaxes(title="An")
    fig.update_yaxes(title="Rata participării (%)", range=_value_axis_range(valid[VALUE_FIELD]), ticksuffix="%")
    return _theme_figure(fig, "Distribuția participării feminine în ani reprezentativi", 460)


def extremes_2023_chart(sample: pd.DataFrame, limit: int = 5) -> go.Figure:
    """Construiește o vizualizare Top/Bottom pentru valorile observate în 2023."""

    extremes = extremes_for_year(sample, year=2023, limit=limit)
    top = extremes["top"].sort_values(VALUE_FIELD, ascending=True)
    bottom = extremes["bottom"].sort_values(VALUE_FIELD, ascending=False)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cele mai ridicate valori", "Cele mai scăzute valori"),
        horizontal_spacing=0.14,
    )
    fig.add_trace(
        go.Bar(
            x=top[VALUE_FIELD],
            y=top[COUNTRY_FIELD],
            orientation="h",
            marker={"color": PALETTE["teal"]},
            name="Top",
            hovertemplate="Țară: %{y}<br>Valoare: %{x:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=bottom[VALUE_FIELD],
            y=bottom[COUNTRY_FIELD],
            orientation="h",
            marker={"color": PALETTE["plum"]},
            name="Bottom",
            hovertemplate="Țară: %{y}<br>Valoare: %{x:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title="Rata participării (%)")
    fig.update_yaxes(title="")
    return _theme_figure(fig, "Extreme europene ale participării feminine – 2023", 430)


def correlation_heatmap(matrix: pd.DataFrame, title: str, height: int = 650) -> go.Figure:
    """Construiește heatmap pentru o matrice de corelații."""

    values = matrix.to_numpy(dtype=float)
    display_values = np.where(np.isfinite(values), values, 0.0)
    text = np.empty(values.shape, dtype=object)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text[row_index, column_index] = f"{value:.2f}" if np.isfinite(value) else ""
    fig = go.Figure(
        go.Heatmap(
            z=display_values.tolist(),
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0.0, PALETTE["burgundy"]],
                [0.22, PALETTE["plum"]],
                [0.5, PALETTE["ivory"]],
                [0.78, PALETTE["teal"]],
                [1.0, PALETTE["navy"]],
            ],
            xgap=1,
            ygap=1,
            text=text.tolist(),
            texttemplate="%{text}",
            textfont={"size": 10, "color": PALETTE["ink"]},
            colorbar={"title": "r"},
            hovertemplate="%{y}<br>%{x}<br>r = %{text}<extra></extra>",
            hoverongaps=False,
        )
    )
    fig.update_xaxes(tickangle=-35, tickfont={"size": 10, "color": PALETTE["ink"]})
    fig.update_yaxes(tickfont={"size": 10, "color": PALETTE["ink"]}, autorange="reversed")
    themed = _theme_figure(fig, title, height)
    themed.update_layout(margin={"l": 170, "r": 28, "t": 64, "b": 128})
    return themed


def correlation_scatter_plot(dataset: pd.DataFrame, x_slug: str, title: str) -> go.Figure:
    """Construiește un scatter plot prestabilit cu linie de trend."""

    valid = dataset[[CORR_COUNTRY_FIELD, CORR_YEAR_FIELD, DEPENDENT_SLUG, x_slug]].dropna().copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=valid[x_slug].tolist(),
            y=valid[DEPENDENT_SLUG].tolist(),
            mode="markers",
            marker={
                "color": PALETTE["teal"],
                "size": 7,
                "opacity": 0.62,
                "line": {"color": "#FFFFFF", "width": 0.6},
            },
            customdata=valid[[CORR_COUNTRY_FIELD, CORR_YEAR_FIELD]].astype(str).to_numpy(),
            name="Observații țară-an",
            hovertemplate=(
                "Țară: %{customdata[0]}<br>"
                "An: %{customdata[1]}<br>"
                f"{variable_label(x_slug)}: %{{x:.2f}}<br>"
                "Participarea femeilor: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    if len(valid) >= 3 and valid[x_slug].nunique() > 1:
        x_values = valid[x_slug].to_numpy(dtype=float)
        y_values = valid[DEPENDENT_SLUG].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(float(np.nanmin(x_values)), float(np.nanmax(x_values)), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line.tolist(),
                y=y_line.tolist(),
                mode="lines",
                line={"color": PALETTE["gold"], "width": 3},
                name="Linie de trend",
                hovertemplate=f"{variable_label(x_slug)}: %{{x:.2f}}<br>Trend: %{{y:.2f}}%<extra></extra>",
            )
        )

    fig.update_xaxes(title=variable_label(x_slug))
    fig.update_yaxes(title="Participarea femeilor (%)", ticksuffix="%")
    return _theme_figure(fig, title, 390)


def missing_by_variable_bar(variable_summary: pd.DataFrame) -> go.Figure:
    """Construiește bar chart pentru valorile lipsă pe variabilă."""

    plotting = variable_summary.sort_values("% lipsă", ascending=True).copy()
    percent = plotting["% lipsă"] * 100
    colors = [
        PALETTE["teal"] if value < 5 else PALETTE["gold"] if value < 15 else PALETTE["burgundy"] if value < 25 else PALETTE["plum"]
        for value in percent
    ]
    fig = go.Figure(
        go.Bar(
            x=percent.tolist(),
            y=plotting["Variabilă"].tolist(),
            orientation="h",
            marker={"color": colors},
            customdata=plotting[["Disponibile", "Lipsă", "Semnal"]].astype(str).to_numpy(),
            hovertemplate=(
                "%{y}<br>% lipsă: %{x:.2f}%<br>"
                "Disponibile: %{customdata[0]}<br>"
                "Lipsă: %{customdata[1]}<br>"
                "%{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="% valori lipsă", range=[0, max(5, float(percent.max()) * 1.12)], ticksuffix="%")
    fig.update_yaxes(title="")
    return _theme_figure(fig, "Valori lipsă pe variabilă", 430)


def missing_by_country_bar(country_summary: pd.DataFrame) -> go.Figure:
    """Construiește bar chart pentru acoperirea datelor pe țări."""

    plotting = country_summary.sort_values("% lipsă", ascending=True).copy()
    percent = plotting["% lipsă"] * 100
    colors = [
        PALETTE["teal"] if value < 5 else PALETTE["gold"] if value < 15 else PALETTE["burgundy"] if value < 25 else PALETTE["plum"]
        for value in percent
    ]
    fig = go.Figure(
        go.Bar(
            x=percent.tolist(),
            y=plotting["Țară"].tolist(),
            orientation="h",
            marker={"color": colors},
            customdata=plotting[["Disponibile", "Lipsă", "Semnal"]].astype(str).to_numpy(),
            hovertemplate=(
                "%{y}<br>% lipsă: %{x:.2f}%<br>"
                "Disponibile: %{customdata[0]}<br>"
                "Lipsă: %{customdata[1]}<br>"
                "%{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="% valori lipsă", range=[0, max(5, float(percent.max()) * 1.12)], ticksuffix="%")
    fig.update_yaxes(title="")
    return _theme_figure(fig, "Acoperirea datelor pe țări", 680)


def country_variable_coverage_heatmap(coverage: pd.DataFrame) -> go.Figure:
    """Construiește heatmap țară x variabilă pentru acoperirea datelor."""

    z_values = (coverage.to_numpy(dtype=float) * 100)
    fig = go.Figure(
        go.Heatmap(
            z=z_values,
            x=coverage.columns.tolist(),
            y=coverage.index.tolist(),
            zmin=0,
            zmax=100,
            colorscale=[
                [0.0, PALETTE["burgundy"]],
                [0.45, PALETTE["gold"]],
                [1.0, PALETTE["teal"]],
            ],
            colorbar={"title": "% disponibil"},
            hovertemplate="Țară: %{y}<br>Variabilă: %{x}<br>Disponibil: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_xaxes(tickangle=-35)
    return _theme_figure(fig, "Heatmap țară × variabilă", 720)


def missing_over_time_line(time_summary: pd.DataFrame) -> go.Figure:
    """Construiește line chart pentru valorile lipsă în timp."""

    percent = time_summary["% valori lipsă"] * 100
    fig = go.Figure(
        go.Scatter(
            x=time_summary["An"].tolist(),
            y=percent.tolist(),
            mode="lines+markers",
            line={"color": PALETTE["navy"], "width": 3},
            marker={"color": PALETTE["gold"], "size": 8},
            customdata=time_summary[["Valori lipsă", "Total"]].astype(str).to_numpy(),
            hovertemplate="An: %{x}<br>% lipsă: %{y:.2f}%<br>Valori lipsă: %{customdata[0]}<br>Total: %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_xaxes(title="An", dtick=2)
    fig.update_yaxes(title="% valori lipsă", range=[0, max(5, float(percent.max()) * 1.18)], ticksuffix="%")
    return _theme_figure(fig, "Evoluția disponibilității datelor în timp", 390)


def dependent_availability_heatmap(availability: pd.DataFrame) -> go.Figure:
    """Construiește heatmap țară x an pentru disponibilitatea variabilei dependente."""

    fig = go.Figure(
        go.Heatmap(
            z=availability.to_numpy(dtype=int),
            x=[str(column) for column in availability.columns],
            y=availability.index.tolist(),
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, PALETTE["burgundy"]],
                [0.49, PALETTE["burgundy"]],
                [0.5, PALETTE["teal"]],
                [1.0, PALETTE["teal"]],
            ],
            showscale=False,
            hovertemplate="Țară: %{y}<br>An: %{x}<br>%{z}<extra></extra>",
        )
    )
    fig.update_traces(
        hovertemplate="Țară: %{y}<br>An: %{x}<br>Status: %{customdata}<extra></extra>",
        customdata=np.where(availability.to_numpy(dtype=int) == 1, "disponibil", "lipsă"),
    )
    fig.update_xaxes(title="An", tickangle=-45)
    fig.update_yaxes(title="")
    return _theme_figure(fig, "Disponibilitatea variabilei dependente", 620)
