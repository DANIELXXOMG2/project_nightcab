"""
Análisis Exploratorio de Datos (EDA)
=====================================
Este script realiza un análisis exploratorio básico del dataset maestro
generado en el proceso anterior de fusión de datos de Divvy y clima.

Autor: Data Science Team
Fecha: 2025-10-16
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_master_dataset(file_path: str) -> pd.DataFrame:
    """
    Carga el dataset maestro desde un archivo Parquet.
    
    Args:
        file_path: Ruta al archivo master_dataset.parquet
        
    Returns:
        DataFrame con los datos cargados
    """
    print(f"📂 Cargando dataset maestro desde: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"✅ Dataset cargado exitosamente: {df.shape[0]:,} filas, {df.shape[1]} columnas\n")
    return df


def display_basic_info(df: pd.DataFrame) -> None:
    """
    Muestra información básica del DataFrame.
    
    Args:
        df: DataFrame a analizar
    """
    print("=" * 80)
    print("📊 PRIMERAS 5 FILAS DEL DATASET")
    print("=" * 80)
    print(df.head())
    print("\n")
    
    print("=" * 80)
    print("ℹ️  INFORMACIÓN DEL DATASET")
    print("=" * 80)
    df.info()
    print("\n")
    
    print("=" * 80)
    print("📈 ESTADÍSTICAS DESCRIPTIVAS - COLUMNAS NUMÉRICAS")
    print("=" * 80)
    print(df.describe())
    print("\n")


def analyze_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula y muestra la matriz de correlación entre variables numéricas clave.
    
    Args:
        df: DataFrame con los datos
        
    Returns:
        Matriz de correlación
    """
    print("=" * 80)
    print("🔗 MATRIZ DE CORRELACIÓN - VARIABLES CLAVE")
    print("=" * 80)
    
    # Definir las columnas de interés para la correlación
    key_columns = [
        'trip_duration_minutes',
        'temperature_2m',
        'relative_humidity_2m',
        'precipitation_mm',
        'wind_speed_10m'
    ]
    
    # Verificar qué columnas existen en el DataFrame
    available_columns = [col for col in key_columns if col in df.columns]
    missing_columns = [col for col in key_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️  Columnas no encontradas en el dataset: {missing_columns}")
        print(f"✅ Columnas disponibles para análisis: {available_columns}\n")
    
    if len(available_columns) < 2:
        print("❌ Error: Se necesitan al menos 2 columnas numéricas para calcular correlación")
        return pd.DataFrame()
    
    # Seleccionar solo las columnas disponibles y calcular correlación
    correlation_data = df[available_columns].copy()
    
    # Eliminar filas con valores nulos para el análisis de correlación
    correlation_data_clean = correlation_data.dropna()
    print(f"📊 Filas utilizadas para correlación: {len(correlation_data_clean):,} "
          f"({len(correlation_data_clean)/len(df)*100:.2f}% del total)\n")
    
    # Calcular matriz de correlación
    correlation_matrix = correlation_data_clean.corr()
    
    # Mostrar la matriz con formato mejorado
    print("Matriz de Correlación (Pearson):")
    print("-" * 80)
    
    # Formato personalizado para mejor visualización
    pd.set_option('display.precision', 4)
    pd.set_option('display.width', 120)
    print(correlation_matrix.to_string())
    print("\n")
    
    # Mostrar las correlaciones más fuertes con trip_duration_minutes
    if 'trip_duration_minutes' in correlation_matrix.columns:
        print("🎯 CORRELACIONES CON DURACIÓN DEL VIAJE (trip_duration_minutes):")
        print("-" * 80)
        duration_corr = correlation_matrix['trip_duration_minutes'].sort_values(ascending=False)
        for col, corr_value in duration_corr.items():
            if col != 'trip_duration_minutes':
                strength = get_correlation_strength(corr_value)
                print(f"  {col:30s}: {corr_value:+.4f}  ({strength})")
        print("\n")
    
    return correlation_matrix


def get_correlation_strength(corr_value: float) -> str:
    """
    Clasifica la fuerza de la correlación.
    
    Args:
        corr_value: Valor de correlación
        
    Returns:
        Descripción de la fuerza de correlación
    """
    abs_corr = abs(corr_value)
    if abs_corr >= 0.7:
        return "Fuerte"
    elif abs_corr >= 0.4:
        return "Moderada"
    elif abs_corr >= 0.2:
        return "Débil"
    else:
        return "Muy débil o nula"


def main():
    """
    Función principal que ejecuta el análisis exploratorio.
    """
    print("\n" + "=" * 80)
    print("🔍 ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 80 + "\n")
    
    # Definir rutas
    data_path = Path("./data/processed/master_dataset.parquet")
    output_path = Path("./output/visualizations")
    
    # Verificar que exista el archivo
    if not data_path.exists():
        print(f"❌ Error: No se encontró el archivo {data_path}")
        print("   Asegúrate de haber ejecutado el script 03_process_and_merge_data.py primero.")
        return
    
    # Verificar que exista la carpeta de salida
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Carpeta de salida para visualizaciones: {output_path.absolute()}\n")
    
    # 1. Cargar datos
    df = load_master_dataset(data_path)
    
    # 2. Mostrar información básica
    display_basic_info(df)
    
    # 3. Análisis de correlación
    correlation_matrix = analyze_correlation(df)
    
    # Guardar la matriz de correlación como CSV para referencia
    if not correlation_matrix.empty:
        corr_output_path = Path("./output/correlation_matrix.csv")
        correlation_matrix.to_csv(corr_output_path)
        print(f"💾 Matriz de correlación guardada en: {corr_output_path.absolute()}")
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS EXPLORATORIO COMPLETADO")
    print("=" * 80)
    print("\n📌 Próximos pasos:")
    print("   - Crear visualizaciones detalladas de las variables")
    print("   - Analizar distribuciones y patrones temporales")
    print("   - Identificar outliers y valores atípicos")
    print("   - Realizar análisis de series de tiempo\n")


if __name__ == "__main__":
    main()
