"""
Script 05: Procesamiento Escalable con PySpark
==============================================
Descripción
-----------
Este script carga el dataset maestro en formato Parquet ("./data/processed/master_dataset.parquet"),
asegura que las columnas temporales estén en tipo `timestamp` para Spark, realiza agregaciones
escalables (con PySpark) y persiste resultados.

Qué hace exactamente:
- Lee el Parquet maestro generado previamente por el pipeline (script 03).
- Verifica/castea columnas de tiempo: `started_at`, `ended_at`, `datetime_hourly` → `timestamp`.
- Agregación 1: duración promedio, desviación estándar y total de viajes por hora del día.
- Agregación 2: conteo de viajes por temperatura (redondeada a entero).
- Intenta guardar resultados en Parquet con Spark.
    - Si en Windows falla la escritura por dependencias nativas de Hadoop (NativeIO/winutils),
        aplica un fallback para guardar en Parquet usando pandas/pyarrow (manteniendo el artefacto final).


"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, avg, count, stddev, round as spark_round
import pandas as pd

def main():
    print("\n" + "=" * 70)
    print("🚀 INICIANDO PROCESAMIENTO ESCALABLE CON PYSPARK")
    print("=" * 70)

    # 1️⃣ Crear sesión de Spark
    # Configuración orientada a entornos Windows: reduce problemas de escritura Parquet
    # (el procesamiento y las agregaciones se ejecutan con Spark; el fallback sólo aplica a la persistencia).
    spark = (SparkSession.builder
        .appName("Divvy Spark Processing")
        .config("spark.driver.memory", "4g")
        # Configuraciones para minimizar dependencias nativas en Windows
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.hadoop.native.lib", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .getOrCreate())

    print("\n✅ Sesión de Spark creada correctamente")
    print(f"   Versión de Spark: {spark.version}")

    # 2️⃣ Cargar dataset maestro
    # Entrada: Parquet consolidado por el pipeline (script 03), con clima y features base.
    parquet_path = "./data/processed/master_dataset.parquet"
    print(f"\n📂 Cargando dataset desde: {parquet_path}")

    df = spark.read.parquet(parquet_path)

    # 2️⃣1️⃣ Asegurar columnas temporales en tipo timestamp (por seguridad)
    # Las columnas ya fueron convertidas a microsegundos al escribir el Parquet en el script 03.
    # Si alguna llegara como otro tipo, se fuerza el cast.
    timestamp_cols = ["started_at", "ended_at", "datetime_hourly"]
    for col_name in timestamp_cols:
        if dict(df.dtypes).get(col_name) != 'timestamp':
            df = df.withColumn(col_name, col(col_name).cast("timestamp"))

    print(f"\n✅ Dataset cargado con {df.count():,} filas y {len(df.columns)} columnas (timestamps listos)")

    # 3️⃣ Mostrar esquema
    print("\n📋 Esquema del dataset:")
    df.printSchema()

    # 4️⃣ Agregación 1 (Spark): duración promedio por hora
    # Crea `hour_of_day` desde `datetime_hourly` y calcula métricas por franja horaria.
    print("\n📊 Calculando duración promedio por hora del día...")
    df_hourly = (
        df.withColumn("hour_of_day", hour(col("datetime_hourly")))
          .groupBy("hour_of_day")
          .agg(
              spark_round(avg("trip_duration_minutes"), 2).alias("avg_trip_duration"),
              spark_round(stddev("trip_duration_minutes"), 2).alias("std_trip_duration"),
              count("*").alias("total_trips")
          )
          .orderBy("hour_of_day")
    )

    print("\n🕐 Duración promedio por hora:")
    df_hourly.show(24, truncate=False)

    # 5️⃣ Agregación 2 (Spark): relación temperatura-viajes (conteos por temperatura redondeada)
    if "temperature_2m" in df.columns:
        print("\n🌡️ Analizando relación entre temperatura y cantidad de viajes...")
        df_temp = (
            df.groupBy(spark_round(col("temperature_2m"), 0).alias("temperature"))
              .agg(count("*").alias("num_trips"))
              .orderBy("temperature")
        )
        df_temp.show(20, truncate=False)
    else:
        print("\n⚠️ No se encontró columna de temperatura en el dataset")

    # 6️⃣ Persistencia: escribir resultados agregados
    # Se intenta primero con Spark (Parquet). Si falla en Windows por NativeIO/winutils,
    # se recurre a un fallback con pandas/pyarrow para garantizar el artefacto.
    output_dir = "D:/Nueva carpeta/Desktop/SPARKGT/data/processed/spark_aggregations.parquet"
    try:
        df_hourly.coalesce(1).write.mode("overwrite").parquet(output_dir)
        print(f"\n💾 Resultados guardados en (Spark Parquet, carpeta): {output_dir}")
    except Exception as e:
        print(f"\n⚠️  Escritura Parquet con Spark falló: {e}\n   → Aplicando fallback con pandas/pyarrow...")
        # Fallback: guardar como archivo Parquet usando pandas/pyarrow
        fallback_file = "D:/Nueva carpeta/Desktop/SPARKGT/data/processed/spark_aggregations_pyarrow.parquet"
        pdf = df_hourly.toPandas()
        pdf.to_parquet(
            fallback_file,
            index=False,
            engine="pyarrow",
            compression="snappy",
            coerce_timestamps="us",
            allow_truncated_timestamps=True
        )
        print(f"   ✅ Fallback guardado en: {fallback_file}")

    # Finalizar Spark
    spark.stop()
    print("\n✅ Sesión de Spark finalizada correctamente")
    print("=" * 70)

if __name__ == "__main__":
    main()
