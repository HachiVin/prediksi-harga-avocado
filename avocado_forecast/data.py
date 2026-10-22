"""Data loading, validation, and monthly aggregation.

This module is free of Streamlit imports so it can be unit-tested on its own.
"""

from __future__ import annotations

from typing import IO, Literal

import pandas as pd

from avocado_forecast.config import (
    DROPPED_COLUMNS,
    MONTHLY_FREQ,
    REQUIRED_COLUMNS,
)


class DataValidationError(ValueError):
    """Raised when an uploaded file cannot be used; the message is safe to show to users."""


def load_dataset(source: str | IO[bytes]) -> pd.DataFrame:
    """Read the avocado CSV, validate its schema, and return a date-sorted DataFrame.

    Args:
        source: A file path or a binary file-like object containing the CSV.

    Raises:
        DataValidationError: If the file is unreadable, empty, lacks required columns,
            or its ``Date`` column cannot be parsed.
    """
    try:
        df = pd.read_csv(source)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataValidationError("File tidak dapat dibaca sebagai CSV yang valid.") from exc

    df = df.drop(columns=list(DROPPED_COLUMNS), errors="ignore")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(f"Kolom wajib tidak ditemukan di CSV: {', '.join(missing)}.")
    if df.empty:
        raise DataValidationError("File CSV tidak memiliki baris data.")

    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Kolom 'Date' tidak dapat dibaca sebagai tanggal.") from exc

    return df.sort_values("Date").reset_index(drop=True)


def monthly_aggregate(df: pd.DataFrame, how: Literal["mean", "sum"]) -> pd.DataFrame:
    """Resample weekly records to month-end and aggregate all numeric columns."""
    resampler = df.resample(MONTHLY_FREQ, on="Date")
    return resampler.mean(numeric_only=True) if how == "mean" else resampler.sum(numeric_only=True)


def monthly_price_series(df: pd.DataFrame) -> pd.Series:
    """Return the month-end mean of ``AveragePrice`` as the series used for forecasting."""
    return df.resample(MONTHLY_FREQ, on="Date")["AveragePrice"].mean()
