"""
Script 03: Procesamiento y Fusión de Datos
===========================================
Pipeline completo de preprocesamiento que:
1. Carga datos de Divvy desde múltiples archivos CSV
2. Carga datos meteorológicos procesados
3. Limpia y transforma ambos DataFrames
4. Fusiona los datos por timestamp horario
5. Guarda el dataset maestro en formato Parquet
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def load_divvy_data(raw_data_path: str = "./data/raw/") -> pd.DataFrame:
    """
    Carga todos los archivos CSV de Divvy en un único DataFrame.
    
    Args:
        raw_data_path: Ruta a la carpeta con archivos CSV de Divvy
        
    Returns:
        DataFrame consolidado con todos los datos de Divvy
    """
    print("=" * 60)
    print("📂 PASO 1: CARGA DE DATOS DE DIVVY")
    print("=" * 60)
    
    # Buscar todos los archivos CSV en la carpeta raw
    csv_pattern = os.path.join(raw_data_path, "*-divvy-tripdata.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {raw_data_path}")
    
    print(f"✅ Encontrados {len(csv_files)} archivos CSV:")
    for file in csv_files:
        print(f"   - {os.path.basename(file)}")
    
    # Cargar y concatenar todos los archivos
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"   ✓ Cargado: {os.path.basename(file)} ({len(df):,} filas)")
    
    # Concatenar todos los DataFrames
    divvy_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\n📊 Total de filas cargadas: {len(divvy_df):,}")
    print(f"📊 Total de columnas: {len(divvy_df.columns)}")
    print(f"📊 Memoria utilizada: {divvy_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return divvy_df


def load_weather_data(weather_path: str = "./data/processed/weather_data.csv") -> pd.DataFrame:
    """
    Carga los datos meteorológicos procesados.
    
    Args:
        weather_path: Ruta al archivo CSV del clima
        
    Returns:
        DataFrame con datos meteorológicos
    """
    print("\n" + "=" * 60)
    print("🌤️  PASO 2: CARGA DE DATOS METEOROLÓGICOS")
    print("=" * 60)
    
    if not os.path.exists(weather_path):
        raise FileNotFoundError(f"No se encontró el archivo: {weather_path}")
    
    weather_df = pd.read_csv(weather_path)
    
    print(f"✅ Archivo cargado: {os.path.basename(weather_path)}")
    print(f"📊 Total de filas: {len(weather_df):,}")
    print(f"📊 Total de columnas: {len(weather_df.columns)}")
    print(f"📊 Columnas disponibles: {', '.join(weather_df.columns.tolist())}")
    
    return weather_df


def clean_divvy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y procesa el DataFrame de Divvy.
    
    Args:
        df: DataFrame de Divvy sin procesar
        
    Returns:
        DataFrame limpio y procesado
    """
    print("\n" + "=" * 60)
    print("🧹 PASO 3: LIMPIEZA Y PROCESAMIENTO DE DATOS DIVVY")
    print("=" * 60)
    
    initial_rows = len(df)
    print(f"📊 Filas iniciales: {initial_rows:,}")
    
    # 1. Convertir columnas de fecha a datetime
    print("\n🔄 Convirtiendo columnas de fecha a datetime...")
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
    print("   ✓ Columnas 'started_at' y 'ended_at' convertidas a datetime")
    
    # 2. Calcular duración del viaje en minutos
    print("\n⏱️  Calculando duración de viajes...")
    df['trip_duration_minutes'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    print("   ✓ Columna 'trip_duration_minutes' creada")
    
    # 3. Eliminar viajes inválidos (duración negativa o cero)
    invalid_duration = df['trip_duration_minutes'] <= 0
    print(f"\n❌ Eliminando {invalid_duration.sum():,} viajes con duración inválida (≤0 minutos)")
    df = df[~invalid_duration].copy()
    
    # 4. Eliminar filas con valores nulos en columnas clave
    print("\n🔍 Eliminando filas con valores nulos en columnas clave...")
    key_columns = ['start_station_name', 'end_station_name', 'started_at', 'ended_at']
    
    nulls_before = df[key_columns].isnull().sum()
    print("   Valores nulos por columna:")
    for col in key_columns:
        print(f"     - {col}: {nulls_before[col]:,}")
    
    df = df.dropna(subset=key_columns).copy()
    rows_removed = initial_rows - len(df)
    print(f"\n   ✓ Total de filas eliminadas: {rows_removed:,}")
    print(f"   ✓ Filas restantes: {len(df):,}")
    
    # 5. Crear columna datetime_hourly (redondear a la hora)
    print("\n🕐 Creando columna 'datetime_hourly' (redondeada a la hora)...")
    df['datetime_hourly'] = df['started_at'].dt.floor('H')
    print("   ✓ Columna 'datetime_hourly' creada")
    print(f"   📅 Ejemplo: {df['started_at'].iloc[0]} → {df['datetime_hourly'].iloc[0]}")
    
    # Resumen de estadísticas de duración
    print("\n📈 Estadísticas de duración de viajes:")
    print(f"   - Duración mínima: {df['trip_duration_minutes'].min():.2f} minutos")
    print(f"   - Duración máxima: {df['trip_duration_minutes'].max():.2f} minutos")
    print(f"   - Duración promedio: {df['trip_duration_minutes'].mean():.2f} minutos")
    print(f"   - Duración mediana: {df['trip_duration_minutes'].median():.2f} minutos")
    
    return df


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y procesa el DataFrame meteorológico.
    
    Args:
        df: DataFrame meteorológico sin procesar
        
    Returns:
        DataFrame limpio y procesado
    """
    print("\n" + "=" * 60)
    print("🧹 PASO 4: LIMPIEZA Y PROCESAMIENTO DE DATOS METEOROLÓGICOS")
    print("=" * 60)
    
    # Convertir columna datetime a formato datetime
    print("\n🔄 Convirtiendo columna 'datetime' a formato datetime...")
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    print("   ✓ Columna 'datetime' convertida correctamente")
    
    # Verificar valores nulos
    null_count = df['datetime'].isnull().sum()
    if null_count > 0:
        print(f"   ⚠️  Encontrados {null_count} valores nulos en 'datetime'")
        df = df.dropna(subset=['datetime']).copy()
        print(f"   ✓ Filas eliminadas: {null_count}")
    else:
        print("   ✓ No se encontraron valores nulos en 'datetime'")
    
    print(f"\n📊 Filas finales: {len(df):,}")
    print(f"📅 Rango de fechas: {df['datetime'].min()} a {df['datetime'].max()}")
    
    return df


def merge_datasets(divvy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusiona los DataFrames de Divvy y clima por timestamp horario.
    
    Args:
        divvy_df: DataFrame de Divvy procesado
        weather_df: DataFrame meteorológico procesado
        
    Returns:
        DataFrame fusionado
    """
    print("\n" + "=" * 60)
    print("🔗 PASO 5: FUSIÓN DE DATASETS")
    print("=" * 60)
    
    print(f"\n📊 Filas antes de fusión:")
    print(f"   - Divvy: {len(divvy_df):,}")
    print(f"   - Clima: {len(weather_df):,}")
    
    # Realizar fusión LEFT JOIN
    print("\n🔄 Realizando fusión (LEFT JOIN) en datetime_hourly = datetime...")
    merged_df = pd.merge(
        divvy_df,
        weather_df,
        left_on='datetime_hourly',
        right_on='datetime',
        how='left',
        suffixes=('', '_weather')
    )
    
    print(f"   ✓ Fusión completada: {len(merged_df):,} filas")
    
    # Verificar registros sin datos meteorológicos
    missing_weather = merged_df['datetime'].isnull().sum()
    if missing_weather > 0:
        print(f"\n   ⚠️  {missing_weather:,} registros sin datos meteorológicos coincidentes")
        print(f"   📊 Porcentaje sin clima: {(missing_weather / len(merged_df) * 100):.2f}%")
    else:
        print("\n   ✅ Todos los registros tienen datos meteorológicos")
    
    # Eliminar columna datetime duplicada del clima
    if 'datetime' in merged_df.columns:
        merged_df = merged_df.drop('datetime', axis=1)
        print("   ✓ Columna 'datetime' duplicada eliminada")
    
    print(f"\n📊 Columnas totales en dataset fusionado: {len(merged_df.columns)}")
    print(f"📊 Memoria utilizada: {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return merged_df


def save_parquet(df: pd.DataFrame, output_path: str = "./data/processed/master_dataset.parquet") -> None:
    """
    Guarda el DataFrame en formato Parquet.
    
    Args:
        df: DataFrame a guardar
        output_path: Ruta del archivo de salida
    """
    print("\n" + "=" * 60)
    print("💾 PASO 6: GUARDADO DEL DATASET MAESTRO")
    print("=" * 60)
    
    # Crear directorio si no existe
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Guardando en: {output_path}")
    
    # Guardar en formato Parquet
    df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
    
    # Obtener tamaño del archivo
    file_size = os.path.getsize(output_path) / 1024**2
    
    print(f"   ✅ Archivo guardado exitosamente")
    print(f"   📊 Tamaño del archivo: {file_size:.2f} MB")
    print(f"   📊 Filas guardadas: {len(df):,}")
    print(f"   📊 Columnas guardadas: {len(df.columns)}")
    
    # Mostrar columnas guardadas
    print("\n📋 Columnas en el dataset maestro:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")


def main():
    """
    Función principal que ejecuta el pipeline completo.
    """
    print("\n" + "=" * 60)
    print("🚀 INICIANDO PIPELINE DE PROCESAMIENTO Y FUSIÓN DE DATOS")
    print("=" * 60)
    
    try:
        # 1. Cargar datos de Divvy
        divvy_df = load_divvy_data()
        
        # 2. Cargar datos meteorológicos
        weather_df = load_weather_data()
        
        # 3. Limpiar datos de Divvy
        divvy_df = clean_divvy_data(divvy_df)
        
        # 4. Limpiar datos meteorológicos
        weather_df = clean_weather_data(weather_df)
        
        # 5. Fusionar datasets
        master_df = merge_datasets(divvy_df, weather_df)
        
        # 6. Guardar dataset maestro
        save_parquet(master_df)
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"\n🎉 Dataset maestro listo en: ./data/processed/master_dataset.parquet")
        print(f"📊 Total de registros: {len(master_df):,}")
        print(f"📊 Total de columnas: {len(master_df.columns)}")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ ERROR EN EL PIPELINE")
        print("=" * 60)
        print(f"\n🚨 {type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    main()
