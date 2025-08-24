import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Avocado Price Forecast", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background-color: black !important;
            color: white !important;
        }
        .stApp {
            background-color: black !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Prediksi Harga Avocado")
st.write("Upload file CSV harga avocado untuk analisis visualisasi dan prediksi menggunakan **SARIMA**.")

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

gdrive_link = "https://drive.google.com/drive/folders/1HnuCakpKXkEZBxqHTNphdMHiPObFgrK8?usp=sharing"

st.link_button("📂 Download Data dari Google Drive", gdrive_link)


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=["Unnamed: 0", "year"], errors="ignore")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    opsi_visual = [
        "Tren Harga & Volume",
        "Perbandingan Penjualan per Ukuran Avocado",
        "Komposisi Penjualan Bags",
        "Perbandingan Harga per Tipe",
        "Rata-rata Harga per Region",
        "Korelasi Antar Variabel"
    ]

    st.subheader("Visualisasi Data")
    pilih_vis = st.selectbox("Pilih jenis visualisasi:", opsi_visual)

    # VISUALISASI
    if pilih_vis == "Tren Harga & Volume":
        st.subheader("Tren Harga & Volume")
        monthly_df = df.resample("M", on="Date").mean(numeric_only=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_df.index, y=monthly_df["AveragePrice"],
            mode="lines", name="Average Price", line=dict(color="cyan")
        ))
        fig.add_trace(go.Scatter(
            x=monthly_df.index, y=monthly_df["Total Volume"],
            mode="lines", name="Total Volume", line=dict(color="orange"), yaxis="y2"
        ))
        fig.update_layout(
            yaxis2=dict(title="Total Volume", overlaying="y", side="right"),
            xaxis_title="Tanggal", yaxis_title="Harga Rata-rata",
            paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)

    elif pilih_vis == "Perbandingan Penjualan per Ukuran Avocado":
        st.subheader("Perbandingan Penjualan per Ukuran Avocado")
        monthly_df = df.resample("M", on="Date").sum(numeric_only=True)
        fig = go.Figure()
        for col, color in zip(["4046", "4225", "4770"], ["cyan", "orange", "magenta"]):
            fig.add_trace(go.Scatter(
                x=monthly_df.index, y=monthly_df[col],
                mode="lines", name=col, line=dict(color=color)
            ))
        fig.update_layout(
            xaxis_title="Tanggal", yaxis_title="Volume Penjualan",
            paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)

    elif pilih_vis == "Komposisi Penjualan Bags":
        st.subheader("Komposisi Penjualan Bags")
        total_bags = df[["Small Bags", "Large Bags", "XLarge Bags"]].sum()
        fig = px.pie(
            values=total_bags.values,
            names=total_bags.index,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(
            paper_bgcolor="black", font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)

    elif pilih_vis == "Perbandingan Harga per Tipe":
        st.subheader("Perbandingan Harga per Tipe (Conventional vs Organic)")
        tipe_df = df.groupby("type")["AveragePrice"].mean().reset_index()
        fig = px.bar(
            tipe_df, x="type", y="AveragePrice",
            color="type", text_auto=True
        )
        fig.update_layout(
            xaxis_title="Tipe", yaxis_title="Harga Rata-rata",
            paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)

    elif pilih_vis == "Rata-rata Harga per Region":
        st.subheader("Rata-rata Harga per Region (Top 10)")
        region_df = df.groupby("region")["AveragePrice"].mean().nlargest(10).reset_index()
        fig = px.bar(
            region_df, x="region", y="AveragePrice",
            color="region", text_auto=True
        )
        fig.update_layout(
            xaxis_title="Region", yaxis_title="Harga Rata-rata",
            paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)

    elif pilih_vis == "Korelasi Antar Variabel":
        st.subheader("Korelasi Antar Variabel")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # PREDIKSI
    st.subheader("Prediksi Harga Avocado")
    forecast_steps = st.number_input("Masukkan jumlah bulan prediksi:", min_value=1, max_value=36, value=12)

    if "AveragePrice" in df.columns:
        price_df = df[["Date", "AveragePrice"]].resample("M", on="Date").mean()

        if st.button("Jalankan Prediksi"):
            model = SARIMAX(price_df['AveragePrice'], order=(1,1,1), seasonal_order=(1,1,1,12))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=forecast_steps)

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=price_df.index, y=price_df['AveragePrice'],
                mode='lines', name='Data Historis',
                line=dict(color='cyan')
            ))
            fig_forecast.add_trace(go.Scatter(
                x=forecast.index, y=forecast,
                mode='lines+markers', name='Prediksi',
                line=dict(color='orange'), marker=dict(size=6)
            ))
            fig_forecast.update_layout(
                title=f"Prediksi Harga Avocado {forecast_steps} Bulan ke Depan",
                xaxis_title="Tanggal", yaxis_title="Harga",
                paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white")
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            forecast_df = pd.DataFrame({
                "Tanggal": forecast.index,
                "Prediksi Harga": forecast.values
            })
            st.subheader("Tabel Hasil Prediksi")
            st.dataframe(forecast_df.style.format({"Prediksi Harga": "{:.2f}"}))

else:
    st.info("Silakan unggah file CSV untuk memulai analisis.")
