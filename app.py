"""Streamlit entry point: page layout and user interaction only.

Data handling, modelling and charts live in the ``avocado_forecast`` package.
"""

from __future__ import annotations

import io
from typing import NamedTuple

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from avocado_forecast.charts import VISUALIZATIONS, forecast_chart
from avocado_forecast.config import (
    APP_TITLE,
    DEFAULT_FORECAST_MONTHS,
    GDRIVE_DATA_URL,
    MAX_FORECAST_MONTHS,
    MIN_FORECAST_MONTHS,
)
from avocado_forecast.data import DataValidationError, load_dataset, monthly_price_series
from avocado_forecast.forecasting import forecast_prices

FORECAST_STATE_KEY = "forecast_run"

st.set_page_config(page_title="Avocado Price Forecast", page_icon="🥑", layout="wide")


class ForecastRun(NamedTuple):
    """A finished forecast, kept in session state so it survives widget interactions."""

    file_id: str
    steps: int
    history: pd.Series
    forecast: pd.Series


@st.cache_data(show_spinner="Membaca data...")
def read_upload(file_bytes: bytes) -> pd.DataFrame:
    """Parse and validate the uploaded CSV (cached by file content)."""
    return load_dataset(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def run_forecast(prices: pd.Series, steps: int) -> pd.Series:
    """Fit the SARIMA model and forecast (cached by series and horizon)."""
    return forecast_prices(prices, steps)


def render_header() -> None:
    st.title(APP_TITLE)
    st.write(
        "Upload file CSV harga avocado untuk analisis visualisasi dan prediksi "
        "menggunakan **SARIMA**."
    )


def render_upload() -> UploadedFile | None:
    col_upload, col_link = st.columns([3, 1], vertical_alignment="bottom")
    with col_upload:
        uploaded = st.file_uploader("Unggah file CSV", type=["csv"])
    with col_link:
        st.link_button("📂 Download Data dari Google Drive", GDRIVE_DATA_URL, width="stretch")
    return uploaded


def render_visualizations(df: pd.DataFrame) -> None:
    options = {viz.label: viz for viz in VISUALIZATIONS}
    choice = st.selectbox("Pilih jenis visualisasi:", list(options))

    viz = options[choice]
    st.subheader(viz.heading)
    st.plotly_chart(viz.build(df), width="stretch")


def render_forecast(df: pd.DataFrame, file_id: str) -> None:
    steps = int(
        st.number_input(
            "Masukkan jumlah bulan prediksi:",
            min_value=MIN_FORECAST_MONTHS,
            max_value=MAX_FORECAST_MONTHS,
            value=DEFAULT_FORECAST_MONTHS,
        )
    )

    if st.button("Jalankan Prediksi", type="primary"):
        prices = monthly_price_series(df)
        try:
            with st.spinner("Melatih model SARIMA..."):
                forecast = run_forecast(prices, steps)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state[FORECAST_STATE_KEY] = ForecastRun(file_id, steps, prices, forecast)

    # Rendered from session state (not from the button click) so the result stays visible
    # when the user changes another widget, e.g. the visualization dropdown.
    run: ForecastRun | None = st.session_state.get(FORECAST_STATE_KEY)
    if run is None or run.file_id != file_id:
        return

    st.plotly_chart(forecast_chart(run.history, run.forecast, run.steps), width="stretch")

    st.subheader("Tabel Hasil Prediksi")
    table = run.forecast.rename_axis("Tanggal").reset_index()
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Tanggal": st.column_config.DateColumn("Tanggal", format="YYYY-MM-DD"),
            "Prediksi Harga": st.column_config.NumberColumn("Prediksi Harga", format="%.2f"),
        },
    )


def main() -> None:
    render_header()
    uploaded = render_upload()

    if uploaded is None:
        st.info("Silakan unggah file CSV untuk memulai analisis.")
        return

    try:
        df = read_upload(uploaded.getvalue())
    except DataValidationError as exc:
        st.error(str(exc))
        return

    tab_visual, tab_forecast = st.tabs(["📊 Visualisasi Data", "📈 Prediksi Harga"])
    with tab_visual:
        render_visualizations(df)
    with tab_forecast:
        render_forecast(df, uploaded.file_id)


main()
