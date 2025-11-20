"""
Script 06: ML para predicción de duración de viaje
==================================================
Objetivo: Entrenar modelos simples para predecir `trip_duration_minutes` con
features básicos (hora, día, tipo de usuario, tipo de bicicleta y clima).

Incluye:
1. Split temporal dinámico (último mes = test)
2. Modelos: baseline (media), LinearRegression y RandomForest (opcional)
3. HistGradientBoosting como modelo principal
4. Métricas: MAE, RMSE, R²
5. Visualizaciones: scatter Pred vs Real (modelo principal) y histograma de residuales

Se elimina toda visualización extra y features experimentales (cíclicos, hash buckets, múltiples gráficos secundarios) para volver a versión clara y compacta.
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor

import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = "./data/processed/master_dataset.parquet"
OUTPUT_DIR = "./output"
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
METRICS_CSV = os.path.join(OUTPUT_DIR, "ml_metrics.csv")
PRED_ACT_HTML = os.path.join(VIS_DIR, "ml_pred_vs_actual.html")
RESID_HTML = os.path.join(VIS_DIR, "ml_residuals.html")
MAX_TRAIN_ROWS = 120_000


def ensure_dirs():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(VIS_DIR).mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("📥 CARGANDO DATASET MAESTRO PARA ML")
    print("=" * 60)
    df = pd.read_parquet(DATA_PATH)
    print(f"✅ Cargado: {DATA_PATH}")
    print(f"📊 Filas: {len(df):,} | Columnas: {len(df.columns)}")
    return df


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("🛠️  INGENIERÍA DE VARIABLES")
    print("=" * 60)

    # Asegurar dtype datetime
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")

    # Variables temporales
    df["hour_of_day"] = df["started_at"].dt.hour
    df["day_of_week"] = df["started_at"].dt.dayofweek  # 0=Lunes
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["started_at"].dt.month
    df["year"] = df["started_at"].dt.year
    # Flag simple de hora punta opcional (se puede quitar si no aporta)
    df["is_rush_hour"] = df["hour_of_day"].isin([7,8,9,16,17,18]).astype(int)

    # Filtrar outliers de duración y nulos
    df = df[df["trip_duration_minutes"].notnull()].copy()
    df = df[df["trip_duration_minutes"] > 0].copy()
    df["trip_duration_minutes"] = df["trip_duration_minutes"].clip(upper=480)

    # Distancia aproximada (si lat/lng disponibles); si no, se omite.
    if {"start_lat", "start_lng", "end_lat", "end_lng"}.issubset(df.columns):
        df["distance_km"] = _haversine_km(df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"]).astype(float)
    else:
        df["distance_km"] = np.nan

    # Selección de columnas útiles
    cols = [
        "trip_duration_minutes",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "temperature_2m",
        "precipitation_mm",
        "wind_speed_10m",
        "distance_km",
        "is_rush_hour",
        "month",
        "year",
        "member_casual",
        "rideable_type",
    ]
    df = df[cols].dropna().copy()

    print("   ✓ Variables creadas: hour_of_day, day_of_week, is_weekend, month, year")
    print("   ✓ Filas tras limpieza: {:,}".format(len(df)))
    return df


def temporal_split(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("🗓️  SPLIT TEMPORAL: train = todos menos último mes; test = último mes")
    print("=" * 60)
    # Determinar el último año/mes disponible
    ym = (df["year"].astype(int) * 100 + df["month"].astype(int))
    last_ym = int(ym.max())
    last_year, last_month = divmod(last_ym, 100)

    test_mask = (df["year"] == last_year) & (df["month"] == last_month)
    train_mask = ~test_mask

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"   🧪 Mes de test: {last_year}-{last_month:02d}")
    print(f"   📚 Train: {len(train_df):,} | 🧪 Test: {len(test_df):,}")
    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Split temporal inválido: falta variedad de meses para separar train/test.")
    return train_df, test_df


def build_pipelines(numeric_features, categorical_features):
    # Preprocesamiento: escalar numéricas para LinearRegression; RF no lo requiere
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[(
        "onehot",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    linreg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    hgb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_depth=None,
            max_leaf_nodes=31,
            l2_regularization=0.0,
            early_stopping=False,
            random_state=42,
        )),
    ])

    return linreg_pipeline, rf_pipeline, hgb_pipeline


def evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("📈 ENTRENAMIENTO Y EVALUACIÓN")
    print("=" * 60)

    target = "trip_duration_minutes"
    numeric_features = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "temperature_2m",
        "precipitation_mm",
        "wind_speed_10m",
        "distance_km",
        "is_rush_hour",
        "month",
        "year",
    ]
    categorical_features = ["member_casual", "rideable_type"]

    # Submuestreo del set de entrenamiento para acelerar
    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=42)
        print(f"   🔄 Submuestreo de train a {len(train_df):,} filas para acelerar entrenamiento")

    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target]
    X_test = test_df[numeric_features + categorical_features]
    y_test = test_df[target]

    # Baseline: promedio del set de entrenamiento
    baseline_pred = np.full(shape=len(y_test), fill_value=y_train.mean())

    def metrics(y_true, y_pred):
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
        }

    results = []
    results.append({"model": "baseline_mean", **metrics(y_test, baseline_pred)})

    # Pipelines
    linreg_pipeline, rf_pipeline, hgb_pipeline = build_pipelines(numeric_features, categorical_features)

    # Transformación log para reducir skew
    lin_ttr = TransformedTargetRegressor(regressor=linreg_pipeline, func=np.log1p, inverse_func=np.expm1)
    hgb_ttr = TransformedTargetRegressor(regressor=hgb_pipeline, func=np.log1p, inverse_func=np.expm1)
    rf_ttr = TransformedTargetRegressor(regressor=rf_pipeline, func=np.log1p, inverse_func=np.expm1)

    # Linear Regression
    lin_ttr.fit(X_train, y_train)
    lin_pred = lin_ttr.predict(X_test)
    results.append({"model": "linear_regression", **metrics(y_test, lin_pred)})

    # Random Forest (opcional por tiempo)
    # RandomForest opcional (activar si se desea)
    enable_rf = True
    if enable_rf:
        rf_ttr.fit(X_train, y_train)
        rf_pred = rf_ttr.predict(X_test)
        results.append({"model": "random_forest", **metrics(y_test, rf_pred)})

    hgb_ttr.fit(X_train, y_train)
    hgb_pred = hgb_ttr.predict(X_test)
    results.append({"model": "hist_gradient_boosting", **metrics(y_test, hgb_pred)})

    # Guardar métricas
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print("\n📄 Métricas guardadas en:", METRICS_CSV)
    print(metrics_df)

    # Visualizaciones simples (scatter + hist residuales del modelo principal HGB)
    fig_scatter = px.scatter(
        x=y_test,
        y=hgb_pred,
        labels={"x": "Real (min)", "y": "Predicho (min)"},
        title="Predicción vs Real (HistGradientBoosting)",
        opacity=0.5,
    )
    fig_scatter.add_trace(go.Scatter(x=[0, y_test.max()], y=[0, y_test.max()], mode="lines", name="Ideal", line=dict(color="red", dash="dash")))
    fig_scatter.write_html(PRED_ACT_HTML)

    residuals = y_test - hgb_pred
    fig_resid = px.histogram(residuals, nbins=50, title="Histograma de Residuales (HGB)", labels={"value": "Residual (min)"})
    fig_resid.write_html(RESID_HTML)

    print("📊 Visualizaciones guardadas:")
    print("   -", PRED_ACT_HTML)
    print("   -", RESID_HTML)



def main():
    ensure_dirs()
    df = load_data()
    df_fe = feature_engineering(df)
    train_df, test_df = temporal_split(df_fe)
    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Split temporal vacío: verifica meses disponibles en el dataset.")
    evaluate_models(train_df, test_df)
    print("\n✅ ML completado correctamente")


if __name__ == "__main__":
    main()
