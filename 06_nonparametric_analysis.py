"""
Script 06: Análisis Estadístico No Paramétrico
==============================================
Este script realiza análisis estadísticos no paramétricos sobre los datos
de Divvy para comparar la duración de viajes entre distintos grupos.

Incluye:
1. Carga del dataset maestro en Parquet
2. Limpieza básica (si es necesario)
3. Cálculo de estadísticas descriptivas por grupo
4. Test de Mann-Whitney para comparar duraciones de viajes
5. Reporte de resultados
"""

import pandas as pd
from scipy.stats import mannwhitneyu
from pathlib import Path
import os


def load_master_dataset(path: str = "./data/processed/master_dataset.parquet") -> pd.DataFrame:
    """
    Carga el dataset maestro en formato Parquet.
    """
    print("=" * 60)
    print("📂 CARGANDO DATASET MAESTRO")
    print("=" * 60)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    
    df = pd.read_parquet(path)
    print(f"✅ Dataset cargado: {len(df):,} filas, {len(df.columns)} columnas")
    return df


def descriptive_stats(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas por grupo.
    """
    print("\n📊 ESTADÍSTICAS DESCRIPTIVAS POR GRUPO")
    stats = df.groupby(group_col)[target_col].describe()
    print(stats)
    return stats


def mann_whitney_test(df: pd.DataFrame, group_col: str, target_col: str):
    """
    Realiza el test de Mann-Whitney U entre dos grupos.
    """
    unique_groups = df[group_col].unique()
    if len(unique_groups) != 2:
        raise ValueError("Mann-Whitney requiere exactamente dos grupos")
    
    group1 = df[df[group_col] == unique_groups[0]][target_col]
    group2 = df[df[group_col] == unique_groups[1]][target_col]
    
    print("\n🔬 REALIZANDO TEST DE MANN-WHITNEY")
    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    
    print(f"   - Grupos comparados: {unique_groups[0]} vs {unique_groups[1]}")
    print(f"   - Estadístico U: {u_stat}")
    print(f"   - Valor p: {p_value}")
    
    if p_value < 0.05:
        print("   ✅ Resultado significativo: hay diferencia entre los grupos")
    else:
        print("   ❌ Resultado no significativo: no hay evidencia de diferencia")
    
    # Mostrar medianas para contexto práctico
    median1 = group1.median()
    median2 = group2.median()
    print(f"   - Mediana {unique_groups[0]}: {median1:.2f} min")
    print(f"   - Mediana {unique_groups[1]}: {median2:.2f} min")
    
    return u_stat, p_value, (median1, median2)


def main():
    print("\n" + "=" * 60)
    print("🚀 INICIANDO ANÁLISIS ESTADÍSTICO NO PARAMÉTRICO")
    print("=" * 60)
    
    # 1️⃣ Cargar dataset
    df = load_master_dataset()
    
    # 2️⃣ Definir columnas
    group_col = 'member_casual'        # Comparación: 'casual' vs 'member'
    target_col = 'trip_duration_minutes'
    
    # 3️⃣ Estadísticas descriptivas
    descriptive_stats(df, group_col, target_col)
    
    # 4️⃣ Test de Mann-Whitney
    mann_whitney_test(df, group_col, target_col)
    
    print("\n✅ ANÁLISIS COMPLETADO")


if __name__ == "__main__":
    main()
