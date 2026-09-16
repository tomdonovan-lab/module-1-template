"""Shared helpers for the OWID CO2 notebook."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Readily usable data: derived frame and shared constants
# ---------------------------------------------------------------------------

CONTINENTS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]

# Berkeley color scheme, one color per continent
CONTINENT_COLORS = {
    "Africa":        "#C4820E",   # Medalist
    "Asia":          "#FDB515",   # California Gold
    "Europe":        "#003262",   # Berkeley Blue
    "North America": "#ED4E33",   # Golden Gate
    "Oceania":       "#00A0DF",   # Lap Lane
    "South America": "#46535E",   # Stone Pine
    "Other":         "#DAD7CB",   # Bay Fog
}

# Fuel sources and their Berkeley colors (fixed stack order, bottom -> top)
FUELS = ["Coal", "Oil", "Gas", "Cement"]
FUEL_COLORS = ["#46535E", "#00A0DF", "#FDB515", "#DAD7CB"]  # Stone Pine / Lap Lane / Cal Gold / Bay Fog

# Wide dataset columns -> readable fuel labels
FUEL_LABELS = {
    "coal_co2": "Coal",
    "oil_co2": "Oil",
    "gas_co2": "Gas",
    "cement_co2": "Cement",
}


def countries(frame):
    """Countries only (dropping aggregates: World, continents, income groups).

    The dataset stores a null ``iso_code`` exactly for non-country entities,
    so filtering on it is the reliable way to keep real countries.
    """
    return frame[frame["iso_code"].notna()].copy()


# ---------------------------------------------------------------------------
# Cross-year joins
# ---------------------------------------------------------------------------

def slice_year(dat, year, cols):
    """One year's rows for ``cols`` (which must include 'country')."""
    return (
        dat.loc[dat["year"] == year, cols]
        .rename(columns={c: f"{c}_{year}" for c in cols if c != "country"})
    )


def join_years(dat, cols, year_a, year_b):
    """Inner-join two years on country, keeping only the requested columns."""
    a = slice_year(dat, year_a, cols)
    b = slice_year(dat, year_b, cols)
    return a.merge(b, on="country", how="inner")


def growth_pct(dat, measure, year_a, year_b, dropna=True):
    """Percent growth of ``measure`` between two years, per country.

    Returns a frame with columns ``country``, ``<measure>_<year_a>``,
    ``<measure>_<year_b>`` and ``<measure>_pct_change``. Rows missing either
    endpoint are dropped; non-finite percentages (e.g. a zero baseline) are
    dropped too.
    """
    merged = join_years(dat, ["country", measure], year_a, year_b)
    if dropna:
        merged = merged.dropna(subset=[f"{measure}_{year_a}", f"{measure}_{year_b}"])
    a = f"{measure}_{year_a}"
    b = f"{measure}_{year_b}"
    merged[f"{measure}_pct_change"] = (merged[b] / merged[a] - 1) * 100
    merged = merged[np.isfinite(merged[f"{measure}_pct_change"])]
    return merged.sort_values(f"{measure}_pct_change", ascending=False).reset_index(drop=True)


def pct_by_group(dat, group_col, value_col, year):
    """Mean of ``value_col`` per ``group_col`` for one year, descending."""
    sub = (
        dat[(dat["year"] == year)]
        .dropna(subset=[value_col])
        [[group_col, value_col]]
    )
    return (
        sub.groupby(group_col, as_index=False)[value_col]
        .mean()
        .sort_values(value_col, ascending=False)
        .reset_index(drop=True)
    )