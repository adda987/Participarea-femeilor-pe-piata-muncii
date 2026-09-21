"""Componente reutilizabile pentru tabele."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from components.empty_states import render_empty_state
from components.html import render_html


def _format_cell(value: object) -> str:
    """Formatează o celulă pentru tabele HTML statice."""

    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_light_table(
    dataframe: pd.DataFrame | None,
    *,
    empty_title: str = "Tabel fără rânduri",
    empty_message: str = "Tabelul nu conține înregistrări pentru contextul curent.",
    height: int | None = 420,
) -> None:
    """Afișează un tabel luminos, fără containerul negru al dataframe-ului nativ."""

    if dataframe is None or dataframe.empty:
        render_empty_state(empty_title, empty_message)
        return
    display = dataframe.copy()
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in display.columns)
    rows = []
    for _, row in display.iterrows():
        cells = "".join(f"<td>{escape(_format_cell(row[column]))}</td>" for column in display.columns)
        rows.append(f"<tr>{cells}</tr>")
    max_height = f' style="max-height: {int(height)}px;"' if height else ""
    render_html(
        f"""
        <div class="light-table-shell"{max_height}>
            <table class="light-data-table">
                <thead>
                    <tr>{headers}</tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """
    )


def render_dataframe_table(
    dataframe: pd.DataFrame | None,
    *,
    empty_title: str = "Tabel fără rânduri",
    empty_message: str = "Tabelul nu conține înregistrări pentru contextul curent.",
    height: int = 420,
) -> None:
    """Afișează un tabel sau o stare goală coerentă."""

    render_light_table(dataframe, empty_title=empty_title, empty_message=empty_message, height=height)


def render_data_dictionary_editor(template: pd.DataFrame) -> pd.DataFrame:
    """Afișează dicționarul datelor într-o structură editabilă."""

    st.caption("Structură editabilă pentru documentarea variabilelor. Câmpurile marcate ca de completat vor fi actualizate după stabilirea surselor.")
    edited = st.data_editor(
        template,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        height=520,
        column_config={
            "denumirea variabilei": st.column_config.TextColumn("Denumirea variabilei", width="medium"),
            "denumirea scurtă": st.column_config.TextColumn("Denumirea scurtă", width="small"),
            "rolul în model": st.column_config.SelectboxColumn(
                "Rolul în model",
                options=["Variabilă dependentă", "Variabilă explicativă", "Variabilă indicator", "Decalaj temporal", "Interacțiune", "Control", "De completat"],
            ),
            "semnul economic așteptat": st.column_config.SelectboxColumn(
                "Semnul economic așteptat",
                options=["De completat", "pozitiv", "negativ", "ambiguu", "neliniar"],
            ),
        },
    )
    return edited


def render_schema_table(columns: list[str]) -> None:
    """Afișează o listă compactă de câmpuri așteptate."""

    schema = pd.DataFrame({"Câmp": columns, "Stadiu": ["De completat"] * len(columns)})
    render_light_table(schema, height=360)
