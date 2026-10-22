import io

import pandas as pd
import pytest

from avocado_forecast.data import (
    DataValidationError,
    load_dataset,
    monthly_aggregate,
)

VALID_HEADER = (
    "Date,AveragePrice,Total Volume,4046,4225,4770,Small Bags,Large Bags,XLarge Bags,type,region"
)
VALID_ROW = "2015-12-27,1.33,64236.62,1036.74,54454.85,48.16,8603.62,93.25,0.0,conventional,Albany"


def _load(text: str) -> pd.DataFrame:
    return load_dataset(io.BytesIO(text.encode()))


def test_load_dataset_drops_helper_columns(sample_df):
    assert "Unnamed: 0" not in sample_df.columns
    assert "year" not in sample_df.columns


def test_load_dataset_parses_and_sorts_dates(sample_df):
    assert pd.api.types.is_datetime64_any_dtype(sample_df["Date"])
    assert sample_df["Date"].is_monotonic_increasing


def test_load_dataset_accepts_minimal_valid_csv():
    df = _load(f"{VALID_HEADER}\n{VALID_ROW}\n")
    assert len(df) == 1


def test_missing_required_column_is_reported_by_name():
    header = VALID_HEADER.replace(",region", "")
    row = VALID_ROW.rsplit(",", 1)[0]
    with pytest.raises(DataValidationError, match="region"):
        _load(f"{header}\n{row}\n")


def test_empty_file_is_rejected():
    with pytest.raises(DataValidationError):
        _load("")


def test_header_only_file_is_rejected():
    with pytest.raises(DataValidationError, match="baris data"):
        _load(f"{VALID_HEADER}\n")


def test_unparseable_dates_are_rejected():
    with pytest.raises(DataValidationError, match="Date"):
        _load(f"{VALID_HEADER}\n{VALID_ROW.replace('2015-12-27', 'bukan-tanggal')}\n")


def test_monthly_price_series_covers_sample_period(monthly_prices):
    assert len(monthly_prices) == 39
    assert monthly_prices.index.min() == pd.Timestamp("2015-01-31")
    assert monthly_prices.index.max() == pd.Timestamp("2018-03-31")
    assert monthly_prices.index.freqstr == "ME"
    assert not monthly_prices.isna().any()


def test_monthly_sum_matches_manual_sum(sample_df):
    january = sample_df[(sample_df["Date"] >= "2015-01-01") & (sample_df["Date"] < "2015-02-01")]
    monthly = monthly_aggregate(sample_df, "sum")
    assert monthly.loc["2015-01-31", "4046"] == pytest.approx(january["4046"].sum())


def test_monthly_mean_matches_manual_mean(sample_df):
    january = sample_df[(sample_df["Date"] >= "2015-01-01") & (sample_df["Date"] < "2015-02-01")]
    monthly = monthly_aggregate(sample_df, "mean")
    assert monthly.loc["2015-01-31", "AveragePrice"] == pytest.approx(
        january["AveragePrice"].mean()
    )
