"""Static configuration: dataset schema, model settings, and UI constants."""

from __future__ import annotations

from dataclasses import dataclass

APP_TITLE = "Prediksi Harga Avocado"
GDRIVE_DATA_URL = (
    "https://drive.google.com/drive/folders/1HnuCakpKXkEZBxqHTNphdMHiPObFgrK8?usp=sharing"
)

# --- Dataset schema -------------------------------------------------------------------
# Helper columns that exist in the source CSV but carry no information for the analysis.
DROPPED_COLUMNS: tuple[str, ...] = ("Unnamed: 0", "year")

# Columns the app reads. A CSV without all of them is rejected with a clear message.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "Date",
    "AveragePrice",
    "Total Volume",
    "4046",
    "4225",
    "4770",
    "Small Bags",
    "Large Bags",
    "XLarge Bags",
    "type",
    "region",
)

PLU_COLUMNS: tuple[str, ...] = ("4046", "4225", "4770")
BAG_COLUMNS: tuple[str, ...] = ("Small Bags", "Large Bags", "XLarge Bags")

# --- Time series ----------------------------------------------------------------------
# The source data is weekly; it is resampled to month-end before analysis and forecasting.
# "ME" (month end) replaces the legacy "M" alias, which pandas 3.x no longer accepts.
MONTHLY_FREQ = "ME"

# Pragmatic guard: SARIMA with a 12-month season needs at least two full seasonal cycles.
MIN_MONTHS_FOR_FORECAST = 24

MIN_FORECAST_MONTHS = 1
MAX_FORECAST_MONTHS = 36
DEFAULT_FORECAST_MONTHS = 12


@dataclass(frozen=True)
class SarimaConfig:
    """Fixed SARIMA hyperparameters, as (p, d, q) and (P, D, Q, s)."""

    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12)


# --- Chart colours --------------------------------------------------------------------
HISTORY_COLOR = "cyan"
FORECAST_COLOR = "orange"
SERIES_COLORS: tuple[str, ...] = ("cyan", "orange", "magenta")
