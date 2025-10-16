"""
UrbanFlow AI - Divvy Data Downloader
Script para descargar datos históricos de viajes de Divvy
"""

import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import requests


def download_divvy_data(year: int, month: int) -> bool:
    """
    Descarga los datos de viajes de Divvy para un año y mes específicos.
    
    Args:
        year: Año de los datos (ej: 2025)
        month: Mes de los datos (1-12)
    
    Returns:
        True si la descarga fue exitosa, False en caso contrario
    """
    # Crear directorio de datos si no existe
    data_dir = Path("./data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Construir la URL
    url = f"https://divvy-tripdata.s3.amazonaws.com/{year}{month:02d}-divvy-tripdata.zip"
    zip_filename = f"{year}{month:02d}-divvy-tripdata.zip"
    zip_path = data_dir / zip_filename
    
    print(f"📥 Descargando datos de {year}-{month:02d}...")
    print(f"   URL: {url}")
    
    try:
        # Descargar el archivo
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Lanza excepción si hay error HTTP
        
        # Guardar el archivo zip
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r   Progreso: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ Descarga completada: {zip_filename}")
        
        # Descomprimir el archivo
        print(f"📦 Descomprimiendo {zip_filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print(f"✅ Archivos extraídos en {data_dir}")
        
        # Eliminar el archivo zip
        os.remove(zip_path)
        print(f"🗑️  Archivo zip eliminado: {zip_filename}\n")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ Error HTTP: {e}")
        print(f"   Los datos para {year}-{month:02d} podrían no estar disponibles.\n")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}\n")
        return False


def get_last_n_complete_months(n: int = 3) -> list[tuple[int, int]]:
    """
    Obtiene los últimos N meses completos.
    
    Args:
        n: Número de meses a obtener (default: 3)
    
    Returns:
        Lista de tuplas (año, mes)
    """
    today = datetime.now()
    
    # Ir al último día del mes anterior
    first_day_current_month = today.replace(day=1)
    last_complete_month = first_day_current_month - timedelta(days=1)
    
    months = []
    for i in range(n):
        # Calcular el mes i meses atrás
        target_date = last_complete_month.replace(day=1) - timedelta(days=i * 30)
        # Ajustar al primer día del mes correcto
        target_date = target_date.replace(day=1)
        months.append((target_date.year, target_date.month))
    
    # Revertir para tener orden cronológico
    return list(reversed(months))


if __name__ == "__main__":
    print("=" * 60)
    print("🚴 UrbanFlow AI - Descargador de Datos de Divvy")
    print("=" * 60)
    print()
    
    # Obtener los últimos 3 meses completos
    months_to_download = get_last_n_complete_months(3)
    
    print(f"📅 Fecha actual: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"📊 Meses a descargar: {len(months_to_download)}")
    for year, month in months_to_download:
        month_name = datetime(year, month, 1).strftime('%B %Y')
        print(f"   - {month_name}")
    print()
    
    # Descargar los datos
    successful_downloads = 0
    failed_downloads = 0
    
    for year, month in months_to_download:
        success = download_divvy_data(year, month)
        if success:
            successful_downloads += 1
        else:
            failed_downloads += 1
    
    # Resumen final
    print("=" * 60)
    print("📈 Resumen de descargas:")
    print(f"   ✅ Exitosas: {successful_downloads}")
    print(f"   ❌ Fallidas: {failed_downloads}")
    print(f"   📁 Datos guardados en: ./data/raw/")
    print("=" * 60)
