# %% [markdown]
# # Generación de Datos para Competencia
#
# Notebook para generar datos Landsat individuales (COG) para actividad en clases.
# Período: Julio 2023 - Junio 2024, Valle del Maipo, Chile.

# %%
# Instalación e imports
# %pip install rioxarray xarray matplotlib numpy rasterio geopandas planetary-computer pystac-client odc-stac plotly

import os
import shutil
import zipfile
from datetime import datetime

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import plotly.graph_objects as go
import rioxarray as rxr
import xarray as xr
from pyproj import Transformer
from pystac_client import Client

print("✅ Bibliotecas importadas")
rxr

# %%
# Configuración
CONFIG = {
    "bbox": [-70.8, -33.7, -70.6, -33.6],  # Valle del Maipo
    "epsg": "EPSG:32719",  # UTM Zone 19S
    "fecha_inicio": "2022-07-01",
    "fecha_fin": "2024-06-30",
    "sensor": "landsat",
    "resolucion": 30,
    "max_nubes": 20,
}

print(f"🎯 Configuración: {CONFIG['fecha_inicio']} a {CONFIG['fecha_fin']}")
print(f"📍 Área: Valle del Maipo, Resolución: {CONFIG['resolucion']}m")


# %%
# Configuración directa para Landsat
bandas = {"red": "red", "green": "green", "blue": "blue", "nir": "nir08"}
factor_reflectancia = 0.0000275
offset = -0.2
rango_valido = (0, 1)
coleccion = "landsat-c2-l2"
plataforma_filtro = ["landsat-8", "landsat-9"]

print("🔧 Configuración Landsat establecida")

# %%
# Búsqueda de imágenes
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

query_params = {
    "collections": [coleccion],
    "bbox": CONFIG["bbox"],
    "datetime": f"{CONFIG['fecha_inicio']}/{CONFIG['fecha_fin']}",
    "query": {
        "eo:cloud_cover": {"lt": CONFIG["max_nubes"]},
        "platform": {"in": plataforma_filtro},
    },
}

search = catalog.search(**query_params)
items = list(search.items())

print(f"🔍 Imágenes encontradas: {len(items)}")
fechas = [item.datetime.strftime("%Y-%m-%d") for item in items]
print(f"📅 Rango: {min(fechas)} a {max(fechas)}")

# Crear mapeo de fechas a scene_id para nombres originales
scene_mapping = {}
for item in items:
    fecha_key = item.datetime.strftime("%Y-%m-%d")
    scene_mapping[fecha_key] = item.id
print(f"🗂️ Mapeo de escenas creado: {len(scene_mapping)} escenas")

# %%
# Montaje de Google Drive y configuración de directorios
try:
    from google.colab import drive

    drive.mount("/content/drive")

    # Directorio temporal local para generar archivos
    temp_dir = "/tmp/competencia_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Directorio final en Drive para el ZIP
    drive_dir = "/content/drive/MyDrive/Colab Notebooks/data/competencia"
    os.makedirs(drive_dir, exist_ok=True)

    print(f"📁 Drive montado. Directorio temporal: {temp_dir}")
    print(f"📁 Directorio Drive: {drive_dir}")
    IN_COLAB = True
except ImportError:
    # Configuración local
    temp_dir = "./competencia_temp"
    drive_dir = "./competencia"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(drive_dir, exist_ok=True)
    print(f"📁 Directorio temporal local: {temp_dir}")
    print(f"📁 Directorio final local: {drive_dir}")
    IN_COLAB = False

# %%
# Carga de datos completa
bandas_requeridas = [
    bandas["red"],
    bandas["green"],
    bandas["blue"],
    bandas["nir"],
]

load_params = {
    "items": items,
    "bands": bandas_requeridas,
    "bbox": CONFIG["bbox"],
    "crs": CONFIG["epsg"],
    "resolution": CONFIG["resolucion"],
    "groupby": "solar_day",
}

print("⏳ Cargando serie temporal completa...")
ds_temporal = odc.stac.load(**load_params)
print(f"✅ Cargado: {len(ds_temporal.time)} fechas, {list(ds_temporal.data_vars)}")

# %%
# Exportación individual de escenas y bandas como COG (en directorio temporal)
print("💾 Generando escenas individuales como COG en directorio temporal...")

# Configuración para COG
cog_profile = {
    "driver": "COG",
    "compress": "deflate",
    "predictor": 2,
    "blocksize": 512,
    "overview_count": 3,
    "overview_resampling": "average",
}

exported_count = 0
for i, tiempo in enumerate(ds_temporal.time):
    fecha_str = pd.to_datetime(tiempo.values).strftime("%Y-%m-%d")

    # Obtener imagen para esta fecha
    imagen_fecha = ds_temporal.sel(time=tiempo)

    # Obtener scene_id original
    scene_id = scene_mapping.get(fecha_str, f"UNKNOWN_{fecha_str}")
    print(f"📤 Generando fecha {i + 1}/{len(ds_temporal.time)}: {scene_id}")

    # Exportar cada banda por separado al directorio temporal
    for banda_nombre in bandas_requeridas:
        banda_data = imagen_fecha[banda_nombre]

        # Nombre del archivo con scene_id original: scene_id_SR_banda.TIF
        filename = f"{scene_id}_SR_{banda_nombre.upper()}.TIF"
        filepath = os.path.join(temp_dir, filename)

        # Exportar como COG
        banda_data.rio.to_raster(filepath, **cog_profile)
        exported_count += 1

print(f"✅ Generación completada: {exported_count} archivos COG en directorio temporal")

# %%
# Crear archivo ZIP con todos los archivos COG
print("🗜️ Creando archivo ZIP con todos los archivos COG...")

# Nombre del archivo ZIP con timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_filename = f"landsat_competencia_{timestamp}.zip"
zip_path = os.path.join(drive_dir, zip_filename)

# Crear ZIP
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATE, compresslevel=6) as zipf:
    # Agregar todos los archivos TIF del directorio temporal
    for filename in os.listdir(temp_dir):
        if filename.endswith(".TIF"):
            file_path = os.path.join(temp_dir, filename)
            zipf.write(file_path, filename)
            print(f"  ➕ {filename}")

# Obtener tamaño del ZIP
zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"✅ Archivo ZIP creado: {zip_filename}")
print(f"📦 Tamaño: {zip_size_mb:.1f} MB")
print(f"📂 Ubicación: {zip_path}")

# %%
# Limpiar directorio temporal
print("🧹 Limpiando archivos temporales...")
shutil.rmtree(temp_dir)
print("✅ Directorio temporal eliminado")

# %%
# Verificar archivo ZIP creado
print(f"📋 Archivo ZIP final: {zip_filename}")
print(f"📦 Tamaño: {zip_size_mb:.1f} MB")
print(f"📂 Ubicación: {zip_path}")

# Verificar contenido del ZIP
with zipfile.ZipFile(zip_path, "r") as zipf:
    files_in_zip = zipf.namelist()
    print(f"📄 Archivos en ZIP: {len(files_in_zip)}")
    print("Muestra del contenido:")
    for i, f in enumerate(sorted(files_in_zip)[:10]):
        print(f"  {f}")
    if len(files_in_zip) > 10:
        print(f"  ... y {len(files_in_zip) - 10} más")

# %% [markdown]
# ## Análisis con Índices y Series Temporales

# %%
# Conversión a reflectancia y cálculo de índices
print("🔄 Convirtiendo a reflectancia y calculando índices...")

# Conversión directa a reflectancia
ds_refl = ds_temporal * factor_reflectancia + offset
rango_min, rango_max = rango_valido
ds_refl = ds_refl.where((ds_refl >= rango_min) & (ds_refl <= rango_max))

# Cálculo directo de NDVI
nir = ds_refl[bandas["nir"]]
red = ds_refl[bandas["red"]]
denominador = nir + red
ndvi_temporal = xr.where(denominador != 0, (nir - red) / denominador, np.nan)
ndvi_temporal = xr.where(
    (ndvi_temporal >= -1) & (ndvi_temporal <= 1), ndvi_temporal, np.nan
)

# Cálculo directo de EVI
blue = ds_refl[bandas["blue"]]
denominador_evi = nir + 6 * red - 7.5 * blue + 1
evi_temporal = xr.where(
    denominador_evi != 0, 2.5 * (nir - red) / denominador_evi, np.nan
)
evi_temporal = xr.where(
    (evi_temporal >= -1) & (evi_temporal <= 1), evi_temporal, np.nan
)

print("✅ Índices calculados para toda la serie temporal")

# %%
lons = [892718, 894021, 898203, 893878]  # cultivo?, cerro, agua, ciudad
lats = [6265358, 6273453, 6266455, 6270304]

# Transformar a coordenadas UTM
transformer = Transformer.from_crs("EPSG:20048", ds_refl.rio.crs, always_xy=True)
x_utm, y_utm = transformer.transform(lons, lats)

coords_puntos = pd.DataFrame(
    {
        "punto_id": [f"P{i + 1}" for i in range(len(lons))],
        "lon": lons,
        "lat": lats,
        "x": x_utm,
        "y": y_utm,
    }
)

print("📍 Puntos generados:")
for _, row in coords_puntos.iterrows():
    print(f"  {row['punto_id']}: {row['lon']:.0f}, {row['lat']:.0f}")

# %%
# Extracción de series temporales
indices_data = xr.Dataset({"NDVI": ndvi_temporal, "EVI": evi_temporal})

indices_puntos = indices_data.sel(
    x=xr.DataArray(
        coords_puntos["x"],
        dims="punto_id",
        coords={"punto_id": coords_puntos["punto_id"]},
    ),
    y=xr.DataArray(
        coords_puntos["y"],
        dims="punto_id",
        coords={"punto_id": coords_puntos["punto_id"]},
    ),
    method="nearest",
)

# Convertir a DataFrame para plotting
serie_temporal = (
    indices_puntos.to_dataframe()
    .reset_index()
    .melt(
        id_vars=["punto_id", "time"],
        value_vars=["NDVI", "EVI"],
        var_name="indice",
        value_name="valor",
    )
)

print("📊 Series temporales extraídas")

# %%
# Plot de series temporales
# Plot de series temporales
from plotly.subplots import make_subplots

colores = ["red", "blue", "green", "orange"]

# Crear subplots para NDVI y EVI
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["NDVI", "EVI"],
    shared_xaxes=True,
    vertical_spacing=0.1,
)

# Para cada índice
for j, indice in enumerate(["NDVI", "EVI"]):
    # Para cada punto
    for i, punto_id in enumerate(coords_puntos["punto_id"]):
        color = colores[i]

        serie_punto = serie_temporal[
            (serie_temporal["punto_id"] == punto_id)
            & (serie_temporal["indice"] == indice)
        ].dropna(subset=["valor"])

        if len(serie_punto) > 0:
            fig.add_trace(
                go.Scatter(
                    x=serie_punto["time"],
                    y=serie_punto["valor"],
                    mode="lines+markers",
                    name=punto_id,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    showlegend=(j == 0),  # Solo mostrar leyenda en primer subplot
                    legendgroup=punto_id,
                ),
                row=1,
                col=j + 1,
            )

fig.update_layout(
    title="Series Temporales de Índices de Vegetación - 4 Puntos Aleatorios",
    height=600,
    hovermode="x unified",
)

fig.update_yaxes(title_text="NDVI", row=1, col=1)
fig.update_yaxes(title_text="EVI", row=1, col=2)
fig.update_xaxes(title_text="Fecha", row=1, col=1)
fig.update_xaxes(title_text="Fecha", row=1, col=2)

fig.show()

# %%
# Estadísticas finales
print("📈 Estadísticas por punto:")
stats = (
    serie_temporal.groupby(["punto_id", "indice"])["valor"]
    .agg(["mean", "std", "min", "max", "count"])
    .round(3)
)
print(stats)

print("\n✅ Proceso completado:")
print(f"   - {len(files_in_zip)} archivos COG exportados")
print(f"   - {len(ds_temporal.time)} fechas procesadas")
print("   - 4 puntos analizados temporalmente")
print(f"   - Período: {CONFIG['fecha_inicio']} a {CONFIG['fecha_fin']}")
