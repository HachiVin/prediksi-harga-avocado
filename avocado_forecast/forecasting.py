"""SARIMA forecasting of the monthly average price.

This module is free of Streamlit imports so it can be unit-tested on its own.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from avocado_forecast.config import MIN_MONTHS_FOR_FORECAST, SarimaConfig


def forecast_prices(
    series: pd.Series,
    steps: int,
    config: SarimaConfig | None = None,
) -> pd.Series:
    """Fit a SARIMA model on ``series`` and forecast ``steps`` months ahead.

    Args:
        series: Monthly price series with a month-end DatetimeIndex.
        steps: Number of months to forecast (must be >= 1).
        config: SARIMA orders; defaults to the project's fixed configuration.

    Returns:
        A Series named ``"Prediksi Harga"`` indexed by the forecast month-end dates.

    Raises:
        ValueError: If ``steps`` is < 1 or the series is too short for a seasonal model.
    """
    config = config or SarimaConfig()

    if steps < 1:
        raise ValueError("Jumlah bulan prediksi minimal 1.")
    if len(series) < MIN_MONTHS_FOR_FORECAST:
        raise ValueError(
            f"Data bulanan terlalu pendek ({len(series)} bulan). "
            f"Model SARIMA musiman membutuhkan minimal {MIN_MONTHS_FOR_FORECAST} bulan."
        )

    model = SARIMAX(series, order=config.order, seasonal_order=config.seasonal_order)
    fitted = model.fit(disp=False)
    return fitted.forecast(steps=steps).rename("Prediksi Harga")
