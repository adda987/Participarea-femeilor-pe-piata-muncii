"""Configurarea eșantionului european pentru analiza descriptivă."""

from __future__ import annotations


ANALYSIS_START_YEAR = 2001
ANALYSIS_END_YEAR = 2023

COUNTRY_CODE_COLUMN = "Country Code"
COUNTRY_NAME_COLUMN = "Country Name"
YEAR_COLUMN = "Time"

EUROPE_COUNTRY_CODES: frozenset[str] = frozenset(
    {
        "ALB",
        "AND",
        "AUT",
        "BLR",
        "BEL",
        "BIH",
        "BGR",
        "HRV",
        "CYP",
        "CZE",
        "DNK",
        "EST",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HUN",
        "ISL",
        "IRL",
        "ITA",
        "XKX",
        "LVA",
        "LIE",
        "LTU",
        "LUX",
        "MLT",
        "MDA",
        "MCO",
        "MNE",
        "NLD",
        "MKD",
        "NOR",
        "POL",
        "PRT",
        "ROU",
        "RUS",
        "SMR",
        "SRB",
        "SVK",
        "SVN",
        "ESP",
        "SWE",
        "CHE",
        "TUR",
        "UKR",
        "GBR",
    }
)

