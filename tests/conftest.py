from pathlib import Path

import pandas as pd
import pytest

from avocado_forecast.data import load_dataset, monthly_price_series

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = ROOT / "data" / "avocado.csv"


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return load_dataset(str(SAMPLE_CSV))


@pytest.fixture(scope="session")
def monthly_prices(sample_df: pd.DataFrame) -> pd.Series:
    return monthly_price_series(sample_df)


@pytest.fixture(scope="session")
def sample_csv_bytes() -> bytes:
    return SAMPLE_CSV.read_bytes()
