import numpy as np
import pandas as pd
import pytest

from avocado_forecast.config import SarimaConfig
from avocado_forecast.forecasting import forecast_prices


def test_forecast_has_requested_length_and_monthly_index(monthly_prices):
    forecast = forecast_prices(monthly_prices, 6)

    assert len(forecast) == 6
    assert forecast.index[0] == monthly_prices.index[-1] + pd.offsets.MonthEnd(1)
    assert forecast.index.freqstr == "ME"
    assert forecast.name == "Prediksi Harga"


def test_forecast_values_are_finite_and_positive(monthly_prices):
    forecast = forecast_prices(monthly_prices, 12)

    assert np.isfinite(forecast).all()
    assert (forecast > 0).all()


def test_forecast_regression_on_sample_data(monthly_prices):
    """Guards against silent behaviour changes (e.g. after a library upgrade)."""
    forecast = forecast_prices(monthly_prices, 3)

    assert forecast.iloc[0] == pytest.approx(1.354, abs=0.02)


def test_default_sarima_config_is_unchanged():
    config = SarimaConfig()
    assert config.order == (1, 1, 1)
    assert config.seasonal_order == (1, 1, 1, 12)


@pytest.mark.parametrize("steps", [0, -3])
def test_non_positive_steps_are_rejected(monthly_prices, steps):
    with pytest.raises(ValueError, match="minimal 1"):
        forecast_prices(monthly_prices, steps)


def test_series_shorter_than_two_seasons_is_rejected(monthly_prices):
    with pytest.raises(ValueError, match="terlalu pendek"):
        forecast_prices(monthly_prices.iloc[:12], 6)
