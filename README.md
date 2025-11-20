# UrbanFlow: Divvy Trips + Weather + ML

## Descripción General
UrbanFlow es un proyecto de análisis y modelado de la demanda de bicicletas compartidas (Divvy) enriquecido con datos meteorológicos. Integra: ingesta de datos mensuales, fusión con clima horario, limpieza y preparación, exploración, modelado predictivo de duración de viaje y un dashboard interactivo (Streamlit).

## Objetivos
- Entender patrones temporales (hora, día, clima) de la demanda.
- Evaluar impacto de variables meteorológicas sobre cantidad y duración de viajes.
- Construir un modelo base para predecir duración de viaje (minutos).
- Preparar opciones locales de optimización (particionado Parquet, compresión, posible migración futura a formatos columnares más avanzados).
- Presentar resultados y KPIs en un dashboard interactivo.

## Estructura del Repositorio
```
data/
  raw/                 # Archivos originales descargados de Divvy
  processed/           # Conjunto principal procesado (master_dataset.parquet)
output/
  visualizations/      # Gráficos EDA y ML
  ml_metrics.csv       # Métricas de modelos
project_nightcab/
  01_download_divvy_data.py        # Descarga de meses Divvy
  02_fetch_weather_data.py         # Obtención de clima
  03_process_and_merge_data.py     # Limpieza + merge clima
  04_exploratory_data_analysis.py  # Análisis exploratorio
  05_scalable_processing_pyspark.py# Lectura y agregaciones PySpark locales
  07_ml_trip_duration.py           # Pipeline ML (versión simplificada)
  app.py                           # Dashboard Streamlit
  requirements.txt                 # Dependencias
```

## Fuentes de Datos
- Divvy Bike Share: archivos mensuales de viajes (CSV). 
- Clima horario: servicio meteorológico (temperatura, precipitación, viento). 

## Ingeniería de Datos (Resumen)
1. Descarga mensual de viajes (IDs, timestamps, tipos bicicleta, tipo usuario, estaciones).
2. Limpieza: conversión de tipos, cálculo `trip_duration_minutes`, filtrado de outliers básicos.
3. Enriquecimiento: unión con clima por timestamp redondeado/hora.
4. Feature Engineering actual: hora del día, día de semana, fin de semana, rush hour, variables clima.

## Pipeline ML (Duración de Viaje)
- Target: `trip_duration_minutes` (posible transformación log1p utilizada en etapas previas)
- Modelos evaluados: baseline (media), LinearRegression, RandomForestRegressor, HistGradientBoostingRegressor.
- Métricas en `output/ml_metrics.csv` (MAE, RMSE, R²). Mejores resultados actuales ~MAE ≈ 5.78, R² ≈ 0.14 (HGB / RF).
- Próximos pasos: mejorar características (lags, interacción clima, festivos), tuning, validación temporal estricta, explicación de variables.

## Dashboard Streamlit
Características actuales:
- Filtros: meses, tipo usuario, tipo bicicleta, rango horario.
- KPIs: total viajes, duración media, % miembros vs casual, % lluvia.
- Vistas: demanda horaria (bar), viajes vs temperatura (line), métricas ML (tabla), preview datos filtrados.
- Próximos pasos: mapa densidad estaciones, residuales por hora, importancias de modelo, serie temporal diaria.

### Ejecución del Dashboard
```powershell
cd "D:\Nueva carpeta\Desktop\SPARKGT\project_nightcab"
streamlit run app.py
```
Si se ejecuta accidentalmente con `python app.py` aparecerá un warning de `ScriptRunContext`; usar siempre `streamlit run`.

## Instalación del Entorno
Se recomienda entorno virtual.
```powershell
cd "D:\Nueva carpeta\Desktop\SPARKGT\project_nightcab"
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` incluye: pandas, streamlit, plotly, pyarrow, scikit-learn, requests.

## Flujo de Ejecución (Local)
1. (Opcional) Descargar meses adicionales con `01_download_divvy_data.py` (parámetros internos).
2. Obtener clima: `02_fetch_weather_data.py`.
3. Procesar y unir: `03_process_and_merge_data.py` genera `master_dataset.parquet`.
4. EDA: `04_exploratory_data_analysis.py` (genera visualizaciones en `output/visualizations/`).
5. Agregaciones PySpark: `05_scalable_processing_pyspark.py` (lecturas, agrupaciones; limitaciones de escritura en Windows).
6. Modelo ML: `07_ml_trip_duration.py` (re-entrena y actualiza métricas / visualizaciones ML).
7. Dashboard: `streamlit run app.py`.

## Optimización Local
Se priorizará ejecución en entorno local: mejora de performance mediante:
- Lectura eficiente en Parquet con columnas seleccionadas.
- Posible particionado por año/mes dentro de `data/processed`.
- Compresión (snappy) y pruning de columnas no usadas en ML.
- Scripts reutilizables para refrescar datos y re-entrenar sin depender de servicios externos.

## Roadmap Pendiente
- [ ] Particionado físico local (year/month) y cleanup incremental.
- [ ] Mapa estaciones (Folium / PyDeck) en dashboard.
- [ ] Serie temporal diaria y descomposición.
- [ ] Importancias de variables (Permutation / SHAP) y vista explicativa.
- [ ] Tuning modelos + validación temporal.
- [ ] Tests de calidad de datos (esquema, rangos, duplicados). 
- [ ] Sección ética: anonimización y bias (usuarios miembro vs casual).
- [ ] Script de refresco mensual (cron local / tarea programada en OS).

## Buenas Prácticas Consideradas
- Caching de datos en Streamlit para evitar recarga pesada.
- Separación clara entre ingesta, procesamiento, modelado y presentación.
- Uso de Parquet para eficiencia y compresión.
-- Uso de Parquet para eficiencia y posibilidad de particionado local.

## Ejemplos de Comandos Clave
```powershell
# Re-entrenar modelo
python 07_ml_trip_duration.py

# Ejecutar exploración
python 04_exploratory_data_analysis.py

# Ver dashboard
streamlit run app.py
```

## Contribuciones
Pendiente definir guías formales (branching, PRs, lint). Por ahora cambios directos en `master` para exploración rápida.

## Licencia
No se ha definido una licencia. Añadir una (MIT/Apache 2.0) si se planea compartir públicamente.

---
¿Sugerencias o mejoras? Abrir issue o comentar. Próximos pasos inmediatos: ampliación del dashboard y mejoras de modelo.
