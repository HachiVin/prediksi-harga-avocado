"""End-to-end tests that drive the real Streamlit script with simulated user input."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from avocado_forecast.charts import VISUALIZATIONS

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
FORECAST_BUTTON = "Jalankan Prediksi"


def _new_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=120).run()


def _upload(at: AppTest, content: bytes) -> AppTest:
    at.file_uploader[0].set_value(("avocado.csv", content, "text/csv"))
    return at.run()


def _click_forecast(at: AppTest) -> AppTest:
    button = next(b for b in at.button if b.label == FORECAST_BUTTON)
    return button.click().run()


def test_asks_for_upload_before_any_data():
    at = _new_app()

    assert not at.exception
    assert len(at.info) == 1
    assert "unggah" in at.info[0].value.lower()


def test_all_visualizations_render_without_errors(sample_csv_bytes):
    at = _upload(_new_app(), sample_csv_bytes)
    assert not at.exception

    for viz in VISUALIZATIONS:
        at.selectbox[0].select(viz.label).run()
        assert not at.exception, f"Visualization failed: {viz.label}"


def test_forecast_runs_and_shows_table(sample_csv_bytes):
    at = _click_forecast(_upload(_new_app(), sample_csv_bytes))

    assert not at.exception
    assert not at.error
    assert len(at.dataframe) == 1
    assert len(at.dataframe[0].value) == 12  # default horizon


def test_forecast_stays_visible_after_changing_visualization(sample_csv_bytes):
    """Regression: the forecast used to vanish on any widget interaction."""
    at = _click_forecast(_upload(_new_app(), sample_csv_bytes))
    assert len(at.dataframe) == 1

    at.selectbox[0].select(VISUALIZATIONS[-1].label).run()

    assert not at.exception
    assert len(at.dataframe) == 1


@pytest.mark.parametrize("content", [b"foo", b""])
def test_invalid_csv_shows_error_instead_of_crashing(content):
    at = _upload(_new_app(), content)

    assert not at.exception
    assert len(at.error) == 1
