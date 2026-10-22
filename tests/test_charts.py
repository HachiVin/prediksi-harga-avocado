import plotly.graph_objects as go
import pytest

from avocado_forecast.charts import VISUALIZATIONS, forecast_chart
from avocado_forecast.forecasting import forecast_prices


def test_visualization_labels_are_unique():
    labels = [viz.label for viz in VISUALIZATIONS]
    assert len(labels) == len(set(labels)) == 6


@pytest.mark.parametrize("viz", VISUALIZATIONS, ids=lambda viz: viz.label)
def test_every_visualization_builds_a_figure(viz, sample_df):
    fig = viz.build(sample_df)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_forecast_chart_has_history_and_forecast_traces(monthly_prices):
    forecast = forecast_prices(monthly_prices, 12)
    fig = forecast_chart(monthly_prices, forecast, 12)

    assert [trace.name for trace in fig.data] == ["Data Historis", "Prediksi"]
    assert "12 Bulan" in fig.layout.title.text
