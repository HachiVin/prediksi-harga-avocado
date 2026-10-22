# 🥑 Avocado Price Forecast

A Streamlit app for exploring weekly U.S. avocado price and volume data and forecasting the monthly average price with a **SARIMA** model.

Built as a hands-on exercise during my Machine Learning Developer internship (Aug–Dec 2025): find an open dataset suitable for forecasting, apply a time-series method I had studied (SARIMA), and practice data visualization.

**Live demo:** [ISI: Streamlit app URL] · **Dataset:** [Avocado Prices on Kaggle](https://www.kaggle.com/datasets/neuromusic/avocado-prices)

## What the app does

1. **Upload** the avocado CSV (a download link to the data is built into the app).
2. **Explore** the data through six interactive Plotly charts:
   - price vs. volume trend
   - sales by PLU (size) category
   - bag-size composition
   - conventional vs. organic price
   - top 10 regions by mean price
   - correlation heatmap
3. **Forecast** the monthly average price 1–36 months ahead and view the result as a chart and a table.

The UI is in Indonesian. Uploaded files are validated, and a missing column or an unreadable date column produces a clear message instead of a stack trace.

## Data

Source: the *Avocado Prices* dataset on Kaggle (by neuromusic), which per its description was compiled from weekly retail scan data published by the Hass Avocado Board and downloaded in May 2018. Credit to the Hass Avocado Board for sharing the data.

The sample file in [`data/avocado.csv`](data/avocado.csv) contains 18,249 weekly records from 2015-01-04 to 2018-03-25, covering 54 regions and two avocado types (`conventional`, `organic`). Columns used by the app: `Date`, `AveragePrice`, `Total Volume`, PLU columns `4046` / `4225` / `4770`, `Small Bags` / `Large Bags` / `XLarge Bags`, `type`, and `region`.

## Method

| Step | Detail |
|---|---|
| Aggregation | Weekly records are resampled to month-end; the target is the mean `AveragePrice` per month (39 monthly points in the sample). |
| Model | `SARIMAX` from statsmodels, order `(1, 1, 1)`, seasonal order `(1, 1, 1, 12)`. |
| Training | The model is fitted on all available history each time a forecast is requested; results are cached per series and horizon. |
| Output | Point forecast for 1–36 months, shown as a line chart and a table. |

## Project structure

```
.
├── app.py                   # Streamlit entry point: layout and interaction only
├── avocado_forecast/
│   ├── config.py            # schema, model settings, constants
│   ├── data.py              # loading, validation, monthly aggregation
│   ├── forecasting.py       # SARIMA fit and forecast
│   └── charts.py            # Plotly figure builders
├── tests/                   # unit tests + end-to-end app tests
├── data/avocado.csv         # sample dataset
├── .streamlit/config.toml   # dark theme and font
├── requirements.txt         # pinned runtime dependencies
└── requirements-dev.txt     # + pytest, ruff
```

`data.py`, `forecasting.py` and `charts.py` do not import Streamlit, so they can be tested and reused independently of the UI.

## Run locally

Tested with Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then upload `data/avocado.csv` in the app.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

The suite covers data validation, monthly aggregation, forecast shape and values, every chart builder, and the full app flow (upload → charts → forecast) using Streamlit's `AppTest`.

## Limitations

- **No hold-out evaluation.** The forecast is not validated against unseen data, so no accuracy metric (MAE, RMSE, etc.) is reported.
- **Short history.** With 39 monthly observations, statsmodels warns that the seasonal starting parameters cannot be estimated; the model converges but has limited evidence for the yearly seasonality.
- **Fixed model order.** The SARIMA orders are not selected by search or information criteria.
- **Unweighted price index.** The target is the plain mean over all rows (all regions and both types), not a volume-weighted national price.
- **Overlapping regions.** The dataset includes aggregate regions (e.g. `TotalUS`) alongside city-level ones. Charts that sum volumes include all rows as provided, so absolute totals are inflated; read those charts for trends rather than levels.

## Tech stack

Python · pandas · statsmodels · Plotly · Streamlit · pytest · ruff
