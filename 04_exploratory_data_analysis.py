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
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap


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


def visualize_hourly_demand(df: pd.DataFrame, output_path: Path) -> None:
    """
    Visualiza la demanda promedio de viajes por hora del día.
    
    Args:
        df: DataFrame con los datos
        output_path: Ruta donde guardar la visualización
    """
    print("=" * 80)
    print("📊 ANÁLISIS DE DEMANDA POR HORA")
    print("=" * 80)
    
    # Extraer la hora del día
    df['hour_of_day'] = df['started_at'].dt.hour
    
    # Agrupar por hora y calcular el promedio de viajes
    hourly_demand = df.groupby('hour_of_day').size().reset_index(name='num_trips')
    
    # Calcular el promedio de viajes por hora (dividiendo por el número de días únicos)
    num_unique_days = df['started_at'].dt.date.nunique()
    hourly_demand['avg_trips_per_hour'] = hourly_demand['num_trips'] / num_unique_days
    
    print(f"📅 Días únicos en el dataset: {num_unique_days}")
    print(f"🚴 Total de viajes analizados: {len(df):,}")
    print(f"\n📈 Estadísticas de demanda por hora:")
    print(f"   - Hora pico: {hourly_demand.loc[hourly_demand['avg_trips_per_hour'].idxmax(), 'hour_of_day']:.0f}:00 "
          f"con {hourly_demand['avg_trips_per_hour'].max():,.1f} viajes promedio")
    print(f"   - Hora valle: {hourly_demand.loc[hourly_demand['avg_trips_per_hour'].idxmin(), 'hour_of_day']:.0f}:00 "
          f"con {hourly_demand['avg_trips_per_hour'].min():,.1f} viajes promedio")
    print(f"   - Promedio general: {hourly_demand['avg_trips_per_hour'].mean():,.1f} viajes por hora\n")
    
    # Crear el gráfico de barras con Plotly
    fig = px.bar(
        hourly_demand,
        x='hour_of_day',
        y='avg_trips_per_hour',
        title='Demanda Promedio de Viajes por Hora del Día',
        labels={
            'hour_of_day': 'Hora del Día',
            'avg_trips_per_hour': 'Promedio de Viajes por Hora'
        },
        color='avg_trips_per_hour',
        color_continuous_scale='Viridis',
        text='avg_trips_per_hour'
    )
    
    # Personalizar el diseño
    fig.update_traces(
        texttemplate='%{text:.0f}',
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            tickformat=','
        ),
        title=dict(
            font=dict(size=18, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        height=600,
        template='plotly_white'
    )
    
    # Guardar el gráfico
    output_file = output_path / 'hourly_demand.html'
    fig.write_html(str(output_file))
    print(f"💾 Gráfico de demanda horaria guardado en: {output_file.absolute()}\n")
    
    # Limpiar la columna temporal
    df.drop('hour_of_day', axis=1, inplace=True)


def visualize_temperature_impact(df: pd.DataFrame, output_path: Path) -> None:
    """
    Visualiza la relación entre temperatura y número de viajes.
    
    Args:
        df: DataFrame con los datos
        output_path: Ruta donde guardar la visualización
    """
    print("=" * 80)
    print("🌡️  ANÁLISIS DEL IMPACTO DE LA TEMPERATURA")
    print("=" * 80)
    
    # Agrupar por temperatura y hora para evitar overplotting
    df['hour_of_day'] = df['started_at'].dt.hour
    df['date'] = df['started_at'].dt.date
    
    # Agregar datos por fecha, hora y temperatura
    temp_impact = df.groupby(['date', 'hour_of_day', 'temperature_2m']).size().reset_index(name='num_trips')
    
    print(f"📊 Puntos de datos agregados: {len(temp_impact):,}")
    print(f"🌡️  Rango de temperatura: {temp_impact['temperature_2m'].min():.1f}°C - "
          f"{temp_impact['temperature_2m'].max():.1f}°C")
    print(f"🚴 Rango de viajes por hora: {temp_impact['num_trips'].min()} - "
          f"{temp_impact['num_trips'].max():,}\n")
    
    # Calcular estadísticas de correlación
    correlation = temp_impact['temperature_2m'].corr(temp_impact['num_trips'])
    print(f"📈 Correlación temperatura-viajes: {correlation:+.4f} "
          f"({get_correlation_strength(correlation)})\n")
    
    # Crear el gráfico de dispersión con Plotly
    fig = px.scatter(
        temp_impact,
        x='temperature_2m',
        y='num_trips',
        title='Impacto de la Temperatura en el Número de Viajes',
        labels={
            'temperature_2m': 'Temperatura (°C)',
            'num_trips': 'Número de Viajes por Hora'
        },
        color='num_trips',
        color_continuous_scale='RdYlBu_r',
        opacity=0.6,
        hover_data={
            'temperature_2m': ':.1f',
            'num_trips': ':,',
            'date': True,
            'hour_of_day': True
        }
    )
    
    # Añadir línea de tendencia
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        temp_impact['temperature_2m'], 
        temp_impact['num_trips']
    )
    
    # Crear línea de tendencia
    x_range = np.linspace(temp_impact['temperature_2m'].min(), 
                          temp_impact['temperature_2m'].max(), 100)
    y_trend = slope * x_range + intercept
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_trend,
            mode='lines',
            name=f'Tendencia (R² = {r_value**2:.4f})',
            line=dict(color='red', width=3, dash='dash')
        )
    )
    
    # Personalizar el diseño
    fig.update_layout(
        xaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            tickformat=','
        ),
        title=dict(
            font=dict(size=18, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        height=600,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    # Guardar el gráfico
    output_file = output_path / 'temperature_impact.html'
    fig.write_html(str(output_file))
    print(f"💾 Gráfico de impacto de temperatura guardado en: {output_file.absolute()}\n")
    
    # Limpiar las columnas temporales
    df.drop(['hour_of_day', 'date'], axis=1, inplace=True)


def visualize_geographic_heatmap(df: pd.DataFrame, output_path: Path, sample_size: int = 50000) -> None:
    """
    Crea un mapa de calor geográfico de los puntos de inicio de viaje en Chicago.
    
    Args:
        df: DataFrame con los datos
        output_path: Ruta donde guardar la visualización
        sample_size: Número de puntos a muestrear para el mapa (default: 50,000)
    """
    print("=" * 80)
    print("🗺️  VISUALIZACIÓN GEOGRÁFICA - MAPA DE CALOR")
    print("=" * 80)
    
    # Preparación de datos: tomar muestra aleatoria
    total_trips = len(df)
    actual_sample_size = min(sample_size, total_trips)
    
    print(f"📊 Total de viajes en el dataset: {total_trips:,}")
    print(f"🎲 Tomando muestra aleatoria de: {actual_sample_size:,} puntos")
    
    # Muestreo aleatorio con seed para reproducibilidad
    df_sample = df.sample(n=actual_sample_size, random_state=42)
    
    # Extraer coordenadas de inicio
    start_locations = df_sample[['start_lat', 'start_lng']].copy()
    
    # Validar que no haya valores nulos
    null_coords = start_locations.isnull().any(axis=1).sum()
    if null_coords > 0:
        print(f"⚠️  Removiendo {null_coords} puntos con coordenadas nulas")
        start_locations = start_locations.dropna()
    
    print(f"✅ Puntos válidos para visualización: {len(start_locations):,}")
    
    # Calcular el centro del mapa (promedio de coordenadas)
    center_lat = start_locations['start_lat'].mean()
    center_lng = start_locations['start_lng'].mean()
    
    print(f"\n📍 Centro del mapa:")
    print(f"   Latitud:  {center_lat:.6f}")
    print(f"   Longitud: {center_lng:.6f}")
    
    # Estadísticas geográficas
    print(f"\n📏 Rango geográfico:")
    print(f"   Latitud:  {start_locations['start_lat'].min():.6f} - {start_locations['start_lat'].max():.6f}")
    print(f"   Longitud: {start_locations['start_lng'].min():.6f} - {start_locations['start_lng'].max():.6f}")
    
    # Crear el mapa base centrado en Chicago
    print(f"\n🗺️  Creando mapa base de Chicago...")
    chicago_map = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Preparar datos para el HeatMap (lista de [lat, lng])
    heat_data = start_locations[['start_lat', 'start_lng']].values.tolist()
    
    print(f"🔥 Generando capa de mapa de calor con {len(heat_data):,} puntos...")
    
    # Crear el HeatMap
    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=18,
        radius=15,
        blur=20,
        gradient={
            0.0: 'blue',
            0.3: 'cyan',
            0.5: 'lime',
            0.7: 'yellow',
            0.9: 'orange',
            1.0: 'red'
        }
    ).add_to(chicago_map)
    
    # Añadir un título al mapa
    title_html = '''
    <div style="position: fixed; 
                top: 10px; 
                left: 50px; 
                width: 500px; 
                height: 90px; 
                background-color: white; 
                border: 2px solid grey; 
                border-radius: 5px;
                z-index: 9999; 
                font-size: 16px;
                padding: 10px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin: 0; color: #2c3e50;">🚴 Mapa de Calor - Puntos de Inicio de Viajes</h4>
        <p style="margin: 5px 0; font-size: 14px; color: #555;">
            <b>Chicago Divvy Bikes</b> | Muestra: {:,} puntos
        </p>
    </div>
    '''.format(len(heat_data))
    
    chicago_map.get_root().html.add_child(folium.Element(title_html))
    
    # Añadir leyenda
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; 
                left: 50px; 
                width: 200px; 
                background-color: white; 
                border: 2px solid grey; 
                border-radius: 5px;
                z-index: 9999; 
                font-size: 14px;
                padding: 10px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 10px 0; color: #2c3e50;">Intensidad</h4>
        <div style="background: linear-gradient(to right, blue, cyan, lime, yellow, orange, red); 
                    height: 20px; 
                    border-radius: 3px;
                    margin-bottom: 5px;">
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <span>Baja</span>
            <span>Alta</span>
        </div>
    </div>
    '''
    
    chicago_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Guardar el mapa
    output_file = output_path / 'station_heatmap.html'
    chicago_map.save(str(output_file))
    
    print(f"\n💾 Mapa de calor guardado en: {output_file.absolute()}")
    
    # Estadísticas adicionales
    print(f"\n📊 Características del mapa:")
    print(f"   - Tipo: Mapa de calor interactivo")
    print(f"   - Librería: Folium con plugin HeatMap")
    print(f"   - Radio de influencia: 15 píxeles")
    print(f"   - Desenfoque (blur): 20")
    print(f"   - Gradiente: 6 colores (azul → rojo)")
    print(f"   - Nivel de zoom inicial: 12")
    print(f"   - Controles: Zoom, capa base, escala\n")


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
    
    # 4. Visualización de patrones temporales
    print("\n" + "=" * 80)
    print("📊 VISUALIZACIÓN DE PATRONES TEMPORALES")
    print("=" * 80 + "\n")
    
    # 4.1 Demanda por hora
    visualize_hourly_demand(df, output_path)
    
    # 4.2 Impacto de la temperatura
    visualize_temperature_impact(df, output_path)
    
    # 4.3 Visualización geográfica - Mapa de calor
    visualize_geographic_heatmap(df, output_path, sample_size=50000)
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS EXPLORATORIO COMPLETADO")
    print("=" * 80)
    print("\n📌 Visualizaciones generadas:")
    print("   ✓ Demanda horaria promedio (hourly_demand.html)")
    print("   ✓ Impacto de temperatura en viajes (temperature_impact.html)")
    print("   ✓ Mapa de calor geográfico (station_heatmap.html)")
    print("\n📌 Próximos pasos:")
    print("   - Analizar distribuciones y outliers")
    print("   - Realizar análisis de series de tiempo")
    print("   - Explorar patrones por tipo de usuario (member/casual)")
    print("   - Analizar impacto de precipitación y viento\n")


if __name__ == "__main__":
    main()
