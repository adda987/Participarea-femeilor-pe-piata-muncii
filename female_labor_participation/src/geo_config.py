"""Configurație geografică pentru analiza hărților europene."""

from __future__ import annotations

from pathlib import Path

from src.utils import PALETTE, get_project_paths


GEOMETRY_ISO_COLUMN = "ISO3"
GEOMETRY_NAME_COLUMN = "geometry_name"
EUROPE_GEOJSON_PATH: Path = get_project_paths().root / "data" / "geo" / "europe.geojson"

NATURAL_EARTH_ISO_FIXES: dict[str, str] = {
    "France": "FRA",
    "Norway": "NOR",
    "Kosovo": "XKX",
}

COUNTRY_LABELS_RO: dict[str, str] = {
    "ALB": "Albania",
    "AND": "Andorra",
    "AUT": "Austria",
    "BEL": "Belgia",
    "BGR": "Bulgaria",
    "BIH": "Bosnia și Herțegovina",
    "BLR": "Belarus",
    "CHE": "Elveția",
    "CYP": "Cipru",
    "CZE": "Cehia",
    "DEU": "Germania",
    "DNK": "Danemarca",
    "ESP": "Spania",
    "EST": "Estonia",
    "FIN": "Finlanda",
    "FRA": "Franța",
    "GBR": "Regatul Unit",
    "GRC": "Grecia",
    "HRV": "Croația",
    "HUN": "Ungaria",
    "IRL": "Irlanda",
    "ISL": "Islanda",
    "ITA": "Italia",
    "LIE": "Liechtenstein",
    "LTU": "Lituania",
    "LUX": "Luxemburg",
    "LVA": "Letonia",
    "MCO": "Monaco",
    "MDA": "Republica Moldova",
    "MKD": "Macedonia de Nord",
    "MLT": "Malta",
    "MNE": "Muntenegru",
    "NLD": "Țările de Jos",
    "NOR": "Norvegia",
    "POL": "Polonia",
    "PRT": "Portugalia",
    "ROU": "România",
    "RUS": "Federația Rusă",
    "SMR": "San Marino",
    "SRB": "Serbia",
    "SVK": "Slovacia",
    "SVN": "Slovenia",
    "SWE": "Suedia",
    "TUR": "Turcia",
    "UKR": "Ucraina",
    "XKX": "Kosovo",
}

CLUSTER_COLORS: list[str] = [
    PALETTE["navy"],
    PALETTE["teal"],
    PALETTE["burgundy"],
    PALETTE["gold"],
    PALETTE["plum"],
    PALETTE["muted"],
]


def country_label(iso3: str, fallback: str = "") -> str:
    """Returnează denumirea românească a unei țări, când există."""

    return COUNTRY_LABELS_RO.get(str(iso3), fallback or str(iso3))
