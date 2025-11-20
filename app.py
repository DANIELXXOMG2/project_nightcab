"""
UrbanFlow Dashboard (Streamlit)
==============================
Dashboard inicial para explorar viajes Divvy + clima + métricas ML.

Secciones:
- Resumen (KPIs)
- Demanda Horaria
- Temperatura vs Viajes
- Modelo (métricas)
- Datos (preview)

Para ejecutar:
  streamlit run app.py

Requisitos adicionales (si faltan en requirements.txt):
  streamlit
  plotly
  pyarrow

"""
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# Construcción de rutas robusta independientemente del directorio actual.
BASE_DIR = Path(__file__).resolve().parent          # project_nightcab/
ROOT_DIR = BASE_DIR.parent                          # raíz del repo
DATA_PATH = ROOT_DIR / "data" / "processed" / "master_dataset.parquet"

# ml_metrics.csv puede estar en la raíz (output/) o en project_nightcab/output/
PRIMARY_METRICS_PATH = ROOT_DIR / "output" / "ml_metrics.csv"
ALT_METRICS_PATH = BASE_DIR / "output" / "ml_metrics.csv"

def resolve_metrics_path() -> Path:
    if PRIMARY_METRICS_PATH.exists():
        return PRIMARY_METRICS_PATH
    if ALT_METRICS_PATH.exists():
        return ALT_METRICS_PATH
    return PRIMARY_METRICS_PATH  # default

METRICS_PATH = resolve_metrics_path()

@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"No se encontró el dataset en {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Asegurar timestamps
    if 'started_at' in df.columns:
        df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    return df

@st.cache_data(show_spinner=True)
def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def filter_data(df: pd.DataFrame,
                months: list[int],
                member_types: list[str],
                bike_types: list[str],
                hour_range: tuple[int,int]) -> pd.DataFrame:
    if df.empty:
        return df
    df['month'] = df['started_at'].dt.month
    df['hour'] = df['started_at'].dt.hour
    mask = (
        df['month'].isin(months) &
        df['member_casual'].isin(member_types) &
        df['rideable_type'].isin(bike_types) &
        df['hour'].between(hour_range[0], hour_range[1])
    )
    return df[mask].copy()

def compute_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"Viajes":"0","Duración media (min)":"-","% Miembros":"-","% Casual":"-","% Lluvia":"-"}
    total = len(df)
    avg_dur = df['trip_duration_minutes'].mean()
    member_pct = df['member_casual'].eq('member').mean()*100 if 'member_casual' in df.columns else 0
    casual_pct = 100 - member_pct
    rain_pct = (df.get('precipitation_mm', pd.Series([0]*len(df))) > 0).mean()*100
    return {
        "Viajes": f"{total:,}",
        "Duración media (min)": f"{avg_dur:.2f}",
        "% Miembros": f"{member_pct:.1f}%",
        "% Casual": f"{casual_pct:.1f}%",
        "% Lluvia": f"{rain_pct:.1f}%"
    }

def plot_hourly(df: pd.DataFrame):
    if df.empty:
        return px.scatter(title="Sin datos")
    hourly = df.groupby(df['started_at'].dt.hour)['ride_id'].count().reset_index(name='viajes')
    fig = px.bar(hourly, x='started_at', y='viajes', title='Demanda por hora', labels={'started_at':'Hora','viajes':'Viajes'})
    return fig

def plot_temperature(df: pd.DataFrame):
    if df.empty or 'temperature_2m' not in df.columns:
        return px.scatter(title="Sin datos de temperatura")
    temp = df.groupby(df['temperature_2m'].round())['ride_id'].count().reset_index(name='viajes')
    fig = px.line(temp, x='temperature_2m', y='viajes', title='Viajes vs Temperatura', labels={'temperature_2m':'Temp (°C)','viajes':'Viajes'})
    return fig

def main():
    st.set_page_config(page_title="UrbanFlow Dashboard", layout="wide")
    st.title("UrbanFlow – Divvy + Clima + ML")
    st.caption("Versión inicial: filtros básicos y vistas principales")

    df = load_data(DATA_PATH)
    metrics_df = load_metrics(METRICS_PATH)

    st.sidebar.header("Filtros")
    all_months = sorted(df['started_at'].dt.month.unique().tolist()) if not df.empty else []
    months = st.sidebar.multiselect("Meses", all_months, default=all_months)
    member_types = st.sidebar.multiselect("Tipo usuario", df['member_casual'].unique().tolist() if 'member_casual' in df.columns else [], default=df['member_casual'].unique().tolist() if 'member_casual' in df.columns else [])
    bike_types = st.sidebar.multiselect("Tipo bicicleta", df['rideable_type'].unique().tolist() if 'rideable_type' in df.columns else [], default=df['rideable_type'].unique().tolist() if 'rideable_type' in df.columns else [])
    hour_range = st.sidebar.slider("Rango hora", 0, 23, (0,23))

    fdf = filter_data(df, months, member_types, bike_types, hour_range)
    kpis = compute_kpis(fdf)

    st.subheader("KPIs")
    kpi_cols = st.columns(len(kpis))
    for col,(k,v) in zip(kpi_cols, kpis.items()):
        col.metric(k, v)

    tab1, tab2, tab3, tab4 = st.tabs(["Demanda Horaria","Temperatura","Modelo ML","Datos"])
    with tab1:
        st.plotly_chart(plot_hourly(fdf), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_temperature(fdf), use_container_width=True)
    with tab3:
        st.subheader("Métricas de Modelos")
        if metrics_df.empty:
            st.info("No se encontró ml_metrics.csv")
        else:
            st.dataframe(metrics_df)
        # Opcional: botón para descargar métricas
        if not metrics_df.empty:
            st.download_button("Descargar métricas CSV", data=metrics_df.to_csv(index=False), file_name="ml_metrics.csv", mime="text/csv")
    with tab4:
        st.write(f"Filas filtradas: {len(fdf):,}")
        st.dataframe(fdf.head(100))

    st.markdown("---")
    st.caption("Próximos pasos: mapa de densidad, residuales por hora, importancias de variables.")

if __name__ == "__main__":
    main()
