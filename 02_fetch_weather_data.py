"""
UrbanFlow AI - Weather Data Fetcher
Script para obtener datos climáticos históricos de Open-Meteo para Chicago
"""

import pandas as pd
import requests
from pathlib import Path
from typing import Tuple


def read_divvy_files() -> pd.DataFrame:
    """
    Lee todos los archivos CSV de Divvy desde ./data/raw/
    
    Returns:
        DataFrame concatenado con todos los datos de Divvy
    """
    raw_data_dir = Path("./data/raw")
    
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"El directorio {raw_data_dir} no existe. Ejecuta primero download_divvy_data.py")
    
    # Buscar todos los archivos CSV
    csv_files = list(raw_data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {raw_data_dir}")
    
    print(f"📂 Archivos CSV encontrados: {len(csv_files)}")
    for csv_file in csv_files:
        print(f"   - {csv_file.name}")
    print()
    
    # Leer y concatenar todos los CSV
    dataframes = []
    for csv_file in csv_files:
        print(f"📖 Leyendo {csv_file.name}...")
        df = pd.read_csv(csv_file)
        dataframes.append(df)
    
    # Concatenar todos los DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"✅ Total de registros leídos: {len(combined_df):,}\n")
    
    return combined_df


def get_date_range(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Determina las fechas de inicio y fin del periodo de datos de Divvy.
    
    Args:
        df: DataFrame con datos de Divvy
    
    Returns:
        Tupla con (fecha_inicio, fecha_fin) en formato 'YYYY-MM-DD'
    """
    print("📅 Determinando rango de fechas...")
    
    # Intentar diferentes nombres de columnas para la fecha de inicio
    date_columns = ['started_at', 'start_time', 'starttime', 'started']
    date_column = None
    
    for col in date_columns:
        if col in df.columns:
            date_column = col
            break
    
    if date_column is None:
        # Mostrar las primeras columnas disponibles para ayudar a depurar
        print(f"Columnas disponibles: {df.columns.tolist()[:10]}")
        raise ValueError("No se encontró la columna de fecha de inicio en los datos")
    
    # Convertir a datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Obtener min y max
    start_date = df[date_column].min().strftime('%Y-%m-%d')
    end_date = df[date_column].max().strftime('%Y-%m-%d')
    
    print(f"   Fecha inicio: {start_date}")
    print(f"   Fecha fin: {end_date}")
    print()
    
    return start_date, end_date


def fetch_weather_data(start_date: str, end_date: str, 
                       latitude: float = 41.88, longitude: float = -87.63) -> pd.DataFrame:
    """
    Obtiene datos climáticos históricos de Open-Meteo para Chicago.
    
    Args:
        start_date: Fecha de inicio en formato 'YYYY-MM-DD'
        end_date: Fecha de fin en formato 'YYYY-MM-DD'
        latitude: Latitud de Chicago (default: 41.88)
        longitude: Longitud de Chicago (default: -87.63)
    
    Returns:
        DataFrame con datos climáticos horarios
    """
    print("🌤️  Solicitando datos climáticos de Open-Meteo...")
    print(f"   Coordenadas: ({latitude}, {longitude})")
    print(f"   Periodo: {start_date} a {end_date}")
    
    # Construir la URL de la API
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': [
            'temperature_2m',
            'relative_humidity_2m',
            'precipitation',
            'wind_speed_10m'
        ],
        'timezone': 'America/Chicago'
    }
    
    print(f"   Enviando solicitud a la API...")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"✅ Datos climáticos recibidos exitosamente\n")
        
        # Procesar la respuesta JSON en un DataFrame
        hourly_data = data['hourly']
        
        df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly_data['time']),
            'temperature_2m': hourly_data['temperature_2m'],
            'relative_humidity_2m': hourly_data['relative_humidity_2m'],
            'precipitation_mm': hourly_data['precipitation'],
            'wind_speed_10m': hourly_data['wind_speed_10m']
        })
        
        print(f"📊 Registros climáticos obtenidos: {len(df):,}")
        print(f"   Variables: {', '.join(df.columns[1:])}")
        print()
        
        return df
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ Error HTTP al consultar la API: {e}")
        raise
    except KeyError as e:
        print(f"❌ Error al procesar la respuesta de la API: {e}")
        print(f"   Estructura recibida: {data.keys()}")
        raise
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        raise


def save_weather_data(df: pd.DataFrame) -> None:
    """
    Guarda el DataFrame de clima en ./data/processed/weather_data.csv
    
    Args:
        df: DataFrame con datos climáticos
    """
    # Crear directorio si no existe
    processed_dir = Path("./data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "weather_data.csv"
    
    print(f"💾 Guardando datos climáticos...")
    df.to_csv(output_file, index=False)
    
    file_size = output_file.stat().st_size / (1024 * 1024)  # Tamaño en MB
    print(f"✅ Archivo guardado: {output_file}")
    print(f"   Tamaño: {file_size:.2f} MB")
    print(f"   Registros: {len(df):,}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("🌤️  UrbanFlow AI - Obtención de Datos Climáticos")
    print("=" * 70)
    print()
    
    try:
        # Paso 1: Leer archivos CSV de Divvy
        print("🚴 PASO 1: Lectura de datos de Divvy")
        print("-" * 70)
        divvy_df = read_divvy_files()
        
        # Paso 2: Determinar rango de fechas
        print("📅 PASO 2: Determinación del rango de fechas")
        print("-" * 70)
        start_date, end_date = get_date_range(divvy_df)
        
        # Liberar memoria del DataFrame de Divvy
        del divvy_df
        
        # Paso 3: Obtener datos climáticos
        print("🌤️  PASO 3: Obtención de datos climáticos de Open-Meteo")
        print("-" * 70)
        weather_df = fetch_weather_data(start_date, end_date)
        
        # Paso 4: Guardar datos climáticos
        print("💾 PASO 4: Guardado de datos procesados")
        print("-" * 70)
        save_weather_data(weather_df)
        
        # Resumen final
        print("=" * 70)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print(f"📊 Resumen:")
        print(f"   - Periodo analizado: {start_date} a {end_date}")
        print(f"   - Registros climáticos: {len(weather_df):,}")
        print(f"   - Archivo de salida: ./data/processed/weather_data.csv")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR EN LA EJECUCIÓN")
        print("=" * 70)
        print(f"   {str(e)}")
        print("=" * 70)
        raise
