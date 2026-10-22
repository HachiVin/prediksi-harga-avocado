"""Plotly figure builders for the exploratory charts and the forecast chart.

Every builder takes plain pandas objects and returns a ``plotly.graph_objects.Figure``;
nothing here depends on Streamlit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from avocado_forecast.config import (
    BAG_COLUMNS,
    FORECAST_COLOR,
    HISTORY_COLOR,
    PLU_COLUMNS,
    SERIES_COLORS,
)
from avocado_forecast.data import monthly_aggregate

_MARGIN = {"l": 0, "r": 0, "t": 40, "b": 0}
_LEGEND_ON_TOP = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0}


def _style(fig: go.Figure, **layout: object) -> go.Figure:
    """Apply the layout shared by all charts. Colours come from the Streamlit theme."""
    fig.update_layout(margin=_MARGIN, legend=_LEGEND_ON_TOP, hovermode="x unified", **layout)
    return fig


def price_volume_trend(df: pd.DataFrame) -> go.Figure:
    """Monthly mean price (left axis) against monthly mean total volume (right axis)."""
    monthly = monthly_aggregate(df, "mean")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["AveragePrice"],
            mode="lines",
            name="Harga Rata-rata",
            line={"color": HISTORY_COLOR},
            hovertemplate="%{y:.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["Total Volume"],
            mode="lines",
            name="Total Volume",
            line={"color": FORECAST_COLOR},
            yaxis="y2",
            hovertemplate="%{y:,.0f}",
        )
    )
    return _style(
        fig,
        xaxis_title="Tanggal",
        yaxis_title="Harga Rata-rata",
        yaxis2={"title": "Total Volume", "overlaying": "y", "side": "right"},
    )


def size_comparison(df: pd.DataFrame) -> go.Figure:
    """Monthly sales volume for each PLU (size) category."""
    monthly = monthly_aggregate(df, "sum")

    fig = go.Figure()
    for column, color in zip(PLU_COLUMNS, SERIES_COLORS, strict=True):
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly[column],
                mode="lines",
                name=f"PLU {column}",
                line={"color": color},
                hovertemplate="%{y:,.0f}",
            )
        )
    return _style(fig, xaxis_title="Tanggal", yaxis_title="Volume Penjualan")


def bags_composition(df: pd.DataFrame) -> go.Figure:
    """Share of total sales by bag size."""
    totals = df[list(BAG_COLUMNS)].sum()

    fig = px.pie(
        values=totals.values,
        names=totals.index,
        color_discrete_sequence=px.colors.sequential.Viridis,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(margin=_MARGIN, showlegend=False)
    return fig


def price_by_type(df: pd.DataFrame) -> go.Figure:
    """Mean price for conventional versus organic avocados."""
    by_type = df.groupby("type")["AveragePrice"].mean().reset_index()

    fig = px.bar(by_type, x="type", y="AveragePrice", color="type", text_auto=".2f")
    fig.update_layout(
        margin=_MARGIN, showlegend=False, xaxis_title="Tipe", yaxis_title="Harga Rata-rata"
    )
    return fig


def price_by_region(df: pd.DataFrame) -> go.Figure:
    """The ten regions with the highest mean price."""
    top_regions = df.groupby("region")["AveragePrice"].mean().nlargest(10).reset_index()

    fig = px.bar(top_regions, x="region", y="AveragePrice", color="region", text_auto=".2f")
    fig.update_layout(
        margin=_MARGIN, showlegend=False, xaxis_title="Region", yaxis_title="Harga Rata-rata"
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Pearson correlation between all numeric columns."""
    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(margin=_MARGIN, height=560)
    return fig


def forecast_chart(history: pd.Series, forecast: pd.Series, steps: int) -> go.Figure:
    """Historical monthly prices followed by the SARIMA forecast."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history,
            mode="lines",
            name="Data Historis",
            line={"color": HISTORY_COLOR},
            hovertemplate="%{y:.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast,
            mode="lines+markers",
            name="Prediksi",
            line={"color": FORECAST_COLOR},
            marker={"size": 6},
            hovertemplate="%{y:.2f}",
        )
    )
    return _style(
        fig,
        title=f"Prediksi Harga Avocado {steps} Bulan ke Depan",
        xaxis_title="Tanggal",
        yaxis_title="Harga",
    )


@dataclass(frozen=True)
class Visualization:
    """A selectable exploratory chart: dropdown label, section heading, and its builder."""

    label: str
    heading: str
    build: Callable[[pd.DataFrame], go.Figure]


VISUALIZATIONS: tuple[Visualization, ...] = (
    Visualization("Tren Harga & Volume", "Tren Harga & Volume", price_volume_trend),
    Visualization(
        "Perbandingan Penjualan per Ukuran Avocado",
        "Perbandingan Penjualan per Ukuran Avocado",
        size_comparison,
    ),
    Visualization("Komposisi Penjualan Bags", "Komposisi Penjualan Bags", bags_composition),
    Visualization(
        "Perbandingan Harga per Tipe",
        "Perbandingan Harga per Tipe (Conventional vs Organic)",
        price_by_type,
    ),
    Visualization(
        "Rata-rata Harga per Region",
        "Rata-rata Harga per Region (Top 10)",
        price_by_region,
    ),
    Visualization("Korelasi Antar Variabel", "Korelasi Antar Variabel", correlation_heatmap),
)
