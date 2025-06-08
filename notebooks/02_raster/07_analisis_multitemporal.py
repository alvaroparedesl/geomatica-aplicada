# %% [markdown]
# # Análisis Multitemporal de Índices de Vegetación
#
# En este cuaderno aprenderemos a realizar análisis multitemporal de índices de vegetación
# utilizando series de tiempo de imágenes satelitales. Trabajaremos con **múltiples fechas**
# para estudiar la evolución temporal de la vegetación.
#
# ## Objetivos de Aprendizaje
#
# Al finalizar este cuaderno serás capaz de:
# * Cargar y procesar series temporales de imágenes satelitales
# * Calcular índices de vegetación para múltiples fechas
# * Extraer y visualizar series temporales de puntos específicos
# * Analizar patrones estacionales y tendencias temporales
# * Identificar cambios en la cobertura vegetal a lo largo del tiempo
# * Interpretar resultados en contexto de monitoreo ambiental

# %% [markdown]
# ## Configuración del entorno

# %%
# @title Instalación de paquetes necesarios
# %pip install rioxarray xarray matplotlib numpy rasterio geopandas planetary-computer pystac-client odc-stac plotly

# %%
# @title Importación de bibliotecas
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import plotly.graph_objects as go
import rioxarray as rxr
import xarray as xr
from plotly.subplots import make_subplots
from pyproj import Transformer
from pystac_client import Client

# Configuración para visualización
rxr
plt.rcParams["figure.figsize"] = (12, 8)
plt.style.use("ggplot")

# Configuración para Colab
try:
    # Verificamos si estamos en Colab
    import google.colab

    IN_COLAB = True
    print("🔧 Configurado para Google Colab")
except ImportError:
    IN_COLAB = False
    print("🔧 Ejecutándose localmente")

print("✅ Bibliotecas importadas exitosamente")

# %% [markdown]
# ## 1. Configuración Modular para Análisis Temporal
#
# **¿Por qué análisis multitemporal?**
#
# El análisis temporal nos permite:
# * **Detectar cambios**: Deforestación, crecimiento urbano, recuperación vegetal
# * **Monitorear ciclos**: Patrones estacionales de cultivos y vegetación natural
# * **Evaluar tendencias**: Degradación o mejora ambiental a largo plazo
# * **Respuesta a eventos**: Sequías, incendios, intervenciones humanas

# %%
# 🎛️ CONFIGURACIÓN PRINCIPAL - Enfoque temporal extendido
CONFIG = {
    # === ÁREA DE ESTUDIO ===
    "bbox": [
        -70.8,
        -33.7,
        -70.6,
        -33.6,
    ],  # Valle del Maipo (formato: [oeste, sur, este, norte])
    "epsg": "EPSG:32719",  # UTM Zone 19S para Chile central
    # === PARÁMETROS TEMPORALES EXTENDIDOS ===
    "fecha_inicio": "2023-06-01",  # Inicio invierno
    "fecha_fin": "2024-05-31",  # Fin otoño (1 año completo)
    # === SENSOR (CONFIGURABLE) ===
    "sensor": "landsat",  # Opciones: 'sentinel-2' o 'landsat'
    # === PARÁMETROS DE CARGA ===
    "resolucion": 30,  # Resolución en metros
    "max_nubes": 30,  # Más tolerante a nubes para tener más imágenes
    "cargar_en_memoria": True,  # Sin chunks - todo en memoria
    # === PARÁMETROS ESPECÍFICOS TEMPORALES ===
    # "frecuencia_muestreo": 15,  # Días entre imágenes objetivo
    "min_imagenes": 12,  # Mínimo de imágenes para análisis temporal
}

# Mostramos la configuración actual
print("🎯 Configuración del análisis temporal:")
print("=" * 50)
for clave, valor in CONFIG.items():
    print(f"  {clave}: {valor}")

print(f"\n📅 Período de análisis: {CONFIG['fecha_inicio']} a {CONFIG['fecha_fin']}")
dias_total = (
    pd.to_datetime(CONFIG["fecha_fin"]) - pd.to_datetime(CONFIG["fecha_inicio"])
).days
print(f"📊 Duración: {dias_total} días ({dias_total / 365.25:.1f} años)")

# %% [markdown]
# ## 2. Funciones Reutilizadas y Especializadas


# %%
# @title Funciones utilitarias (reutilizadas del notebook anterior)
def obtener_configuracion_sensor(sensor):
    """
    Retorna configuración específica para cada sensor satelital.
    """
    if sensor == "sentinel-2":
        return {
            "bandas": {
                "red": "B04",
                "green": "B03",
                "blue": "B02",
                "nir": "B08",
                "swir1": "B11",
                "swir2": "B12",
            },
            "rgb_bandas": ["B04", "B03", "B02"],
            "falso_color": ["B08", "B04", "B03"],
            "factor_reflectancia": 0.0001,
            "offset": 0.0,
            "rango_valido": (0, 1),
            "coleccion": "sentinel-2-l2a",
            "plataforma_filtro": None,
            "descripcion": "Sentinel-2 L2A (ESA) - Series temporales de alta resolución",
        }
    elif sensor == "landsat":
        return {
            "bandas": {
                "red": "red",
                "green": "green",
                "blue": "blue",
                "nir": "nir08",
                "swir1": "swir16",
                "swir2": "swir22",
            },
            "rgb_bandas": ["red", "green", "blue"],
            "falso_color": ["nir08", "red", "green"],
            "factor_reflectancia": 0.0000275,
            "offset": -0.2,
            "rango_valido": (0, 1),
            "coleccion": "landsat-c2-l2",
            "plataforma_filtro": ["landsat-8", "landsat-9"],
            "descripcion": "Landsat 8/9 Collection 2 L2 - Series históricas consistentes",
        }
    else:
        raise ValueError(
            f"Sensor '{sensor}' no soportado. Use 'sentinel-2' o 'landsat'"
        )


def convertir_a_reflectancia(imagen, config_sensor):
    """Convierte números digitales a reflectancia según el sensor."""
    factor = config_sensor["factor_reflectancia"]
    offset = config_sensor["offset"]

    imagen_refl = imagen * factor + offset

    rango_min, rango_max = config_sensor["rango_valido"]
    imagen_refl = imagen_refl.where(
        (imagen_refl >= rango_min) & (imagen_refl <= rango_max)
    )

    return imagen_refl


def calcular_ndvi(imagen, config_sensor):
    """Calcula NDVI con validación de rangos."""
    bandas = config_sensor["bandas"]
    nir = imagen[bandas["nir"]]
    red = imagen[bandas["red"]]

    denominador = nir + red
    ndvi = xr.where(denominador != 0, (nir - red) / denominador, np.nan)
    ndvi = xr.where((ndvi >= -1) & (ndvi <= 1), ndvi, np.nan)

    ndvi.name = "NDVI"
    ndvi.attrs = {
        "long_name": "Normalized Difference Vegetation Index",
        "formula": "(NIR - RED) / (NIR + RED)",
        "range": "-1 to +1",
    }
    return ndvi


def calcular_evi(imagen, config_sensor):
    """Calcula EVI con validación de rangos."""
    bandas = config_sensor["bandas"]
    nir = imagen[bandas["nir"]]
    red = imagen[bandas["red"]]
    blue = imagen[bandas["blue"]]

    denominador = nir + 6 * red - 7.5 * blue + 1
    evi = xr.where(denominador != 0, 2.5 * (nir - red) / denominador, np.nan)
    evi = xr.where((evi >= -1) & (evi <= 1), evi, np.nan)

    evi.name = "EVI"
    evi.attrs = {
        "long_name": "Enhanced Vegetation Index",
        "formula": "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
        "range": "-1 to +1",
    }
    return evi


def calcular_ndwi(imagen, config_sensor):
    """Calcula NDWI con validación de rangos."""
    bandas = config_sensor["bandas"]
    nir = imagen[bandas["nir"]]
    swir1 = imagen[bandas["swir1"]]

    denominador = nir + swir1
    ndwi = xr.where(denominador != 0, (nir - swir1) / denominador, np.nan)
    ndwi = xr.where((ndwi >= -1) & (ndwi <= 1), ndwi, np.nan)

    ndwi.name = "NDWI"
    ndwi.attrs = {
        "long_name": "Normalized Difference Water Index",
        "formula": "(NIR - SWIR1) / (NIR + SWIR1)",
        "range": "-1 to +1",
    }
    return ndwi


def crear_imagen_rgb(dataset, bandas_rgb, factor_brillo=3.0):
    """Crea imagen RGB para visualización."""
    rgb_data = []
    for banda in bandas_rgb:
        rgb_data.append(dataset[banda].values)

    rgb = np.stack(rgb_data, axis=2)
    rgb_enhanced = np.clip(rgb * factor_brillo, 0, 1)
    return rgb_enhanced


print("🔧 Funciones utilitarias creadas exitosamente")

# %% [markdown]
# ## 3. Búsqueda y Carga de Series Temporales

# %%
# Obtenemos la configuración del sensor seleccionado
config_sensor = obtener_configuracion_sensor(CONFIG["sensor"])

print(f"🛰️ Trabajando con: {config_sensor['descripcion']}")
print(f"   Colección: {config_sensor['coleccion']}")
print(f"   Período temporal: {CONFIG['fecha_inicio']} a {CONFIG['fecha_fin']}")

# %%
# Conectamos al catálogo STAC de Planetary Computer
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Configuramos la búsqueda temporal extendida
query_params = {
    "collections": [config_sensor["coleccion"]],
    "bbox": CONFIG["bbox"],
    "datetime": f"{CONFIG['fecha_inicio']}/{CONFIG['fecha_fin']}",
    "query": {"eo:cloud_cover": {"lt": CONFIG["max_nubes"]}},
}

# Filtro adicional para Landsat
if config_sensor["plataforma_filtro"]:
    query_params["query"]["platform"] = {"in": config_sensor["plataforma_filtro"]}

# Realizamos la búsqueda
search = catalog.search(**query_params)
items = list(search.items())

print("🔍 Búsqueda temporal completada:")
print(f"   Imágenes encontradas: {len(items)}")

if len(items) < CONFIG["min_imagenes"]:
    print(f"⚠️ Pocas imágenes encontradas ({len(items)} < {CONFIG['min_imagenes']})")
    print("   Considera:")
    print("   - Aumentar el porcentaje máximo de nubes")
    print("   - Ampliar el período temporal")
    print("   - Cambiar la resolución espacial")
elif len(items) == 0:
    print("❌ No se encontraron imágenes. Ajusta los parámetros de búsqueda.")
else:
    print("✅ Suficientes imágenes para análisis temporal")

    # Información de las imágenes
    fechas_disponibles = [item.datetime.strftime("%Y-%m-%d") for item in items]
    print("\n🕒 Rango de fechas:")
    print(f"   Primera: {min(fechas_disponibles)}")
    print(f"   Última: {max(fechas_disponibles)}")

    # Distribución temporal
    df_fechas = pd.DataFrame({"fecha": pd.to_datetime(fechas_disponibles)})
    df_fechas["mes"] = df_fechas["fecha"].dt.month
    distribucion = df_fechas["mes"].value_counts().sort_index()
    print("\n📊 Distribución por mes:")
    for mes, count in distribucion.items():
        print(f"   Mes {mes}: {count} imágenes")

# %%
# Definimos las bandas necesarias
bandas_requeridas = [
    config_sensor["bandas"]["red"],
    config_sensor["bandas"]["green"],
    config_sensor["bandas"]["blue"],
    config_sensor["bandas"]["nir"],
    config_sensor["bandas"]["swir1"],
]

print(f"🎯 Bandas a cargar: {bandas_requeridas}")

# Configuramos parámetros de carga temporal
load_params = {
    "items": items,
    "bands": bandas_requeridas,
    "bbox": CONFIG["bbox"],
    "crs": CONFIG["epsg"],
    "resolution": CONFIG["resolucion"],
    "groupby": "solar_day",
}

# Carga sin chunks para análisis temporal
if not CONFIG["cargar_en_memoria"]:
    load_params["chunks"] = {"x": 512, "y": 512, "time": 5}
    print("📦 Cargando con chunks temporal")
else:
    print("🧠 Cargando serie temporal completa en memoria")

# Cargamos la serie temporal
print("⏳ Cargando serie temporal... (esto puede tomar varios minutos)")
ds_temporal = odc.stac.load(**load_params)

print("\n📊 Serie temporal cargada:")
print(f"   Dimensiones: {dict(ds_temporal.sizes)}")
print(f"   Variables: {list(ds_temporal.data_vars)}")
print(f"   Fechas disponibles: {len(ds_temporal.time)}")

# Información temporal detallada
fechas_cargadas = pd.to_datetime(ds_temporal.time.values)
print("\n🕒 Información temporal:")
print(f"   Primera imagen: {fechas_cargadas.min()}")
print(f"   Última imagen: {fechas_cargadas.max()}")
print(
    f"   Intervalo promedio: {(fechas_cargadas.max() - fechas_cargadas.min()).days / len(fechas_cargadas):.1f} días"
)
# %%
ds_temporal

# %% [markdown]
# ## 4. Conversión a Reflectancia y Visualización de Muestra

# %%
# Convertimos toda la serie temporal a reflectancia
print("🔄 Convirtiendo serie temporal a reflectancia...")
ds_refl = convertir_a_reflectancia(ds_temporal, config_sensor)

print("✅ Serie temporal convertida a reflectancia")

# Verificamos rangos por banda y fecha
print("\n📈 Verificación de rangos (muestra de primeras 3 fechas):")
print("=" * 70)

for i, fecha in enumerate(ds_refl.time[:3]):
    fecha_str = pd.to_datetime(fecha.values).strftime("%Y-%m-%d")
    print(f"\nFecha: {fecha_str}")

    imagen_fecha = ds_refl.sel(time=fecha)
    for banda in bandas_requeridas:
        datos = imagen_fecha[banda]
        datos_validos = datos.where(~np.isnan(datos))

        if datos_validos.count() > 0:
            min_val = float(datos_validos.min())
            max_val = float(datos_validos.max())
            mean_val = float(datos_validos.mean())
            print(f"  {banda:>8}: {min_val:.4f} - {max_val:.4f} (μ: {mean_val:.4f})")

# %%
# Visualizamos imagen de muestra de la serie temporal
fecha_muestra = ds_refl.time[len(ds_refl.time) // 2]  # Imagen del medio de la serie
imagen_muestra = ds_refl.sel(time=fecha_muestra)
fecha_muestra_str = pd.to_datetime(fecha_muestra.values).strftime("%Y-%m-%d")

# RGB
rgb_muestra = crear_imagen_rgb(imagen_muestra, config_sensor["rgb_bandas"])

# Falso color
falso_color_muestra = crear_imagen_rgb(imagen_muestra, config_sensor["falso_color"])

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(
    f"Serie Temporal - Imagen de Muestra: {fecha_muestra_str}",
    fontsize=16,
    fontweight="bold",
)

# RGB
axes[0].imshow(rgb_muestra)
axes[0].set_title(f"Color Verdadero (RGB)\n{config_sensor['descripcion']}")
axes[0].axis("off")

# Falso color
axes[1].imshow(falso_color_muestra)
axes[1].set_title("Falso Color (NIR-R-G)\nVegetación en tonos rojos")
axes[1].axis("off")

# Información de la serie
plt.figtext(
    0.5,
    0.02,
    f"Serie temporal: {len(ds_refl.time)} imágenes | "
    f"Resolución: {CONFIG['resolucion']}m | "
    f"Período: {pd.to_datetime(ds_refl.time.values).min().strftime('%Y-%m')} - "
    f"{pd.to_datetime(ds_refl.time.values).max().strftime('%Y-%m')}",
    ha="center",
    fontsize=11,
    style="italic",
)

plt.tight_layout()
plt.show()

print(f"🖼️ Imagen de muestra mostrada: {fecha_muestra_str}")
print(f"📊 Posición en serie: {len(ds_refl.time) // 2 + 1}/{len(ds_refl.time)}")

# %% [markdown]
# ## 5. Cálculo de Índices para Serie Temporal

# %%
# Calculamos índices para toda la serie temporal (directamente con xarray)
print("🕒 Calculando índices para serie temporal...")

ndvi_temporal = calcular_ndvi(ds_refl, config_sensor)
evi_temporal = calcular_evi(ds_refl, config_sensor)
ndwi_temporal = calcular_ndwi(ds_refl, config_sensor)

print(f"✅ Índices calculados para {len(ds_refl.time)} fechas")

# Mostramos resumen de los índices calculados
print("\n🌱 Resumen de índices calculados:")
print(f"   Fechas procesadas: {len(ds_refl.time)}")
print("   Índices por fecha: NDVI, EVI, NDWI")

# Verificamos rangos de algunos índices
print("\n📊 Rangos de índices (estadísticas generales):")
print("=" * 60)

for nombre, indice in [
    ("NDVI", ndvi_temporal),
    ("EVI", evi_temporal),
    ("NDWI", ndwi_temporal),
]:
    valores_validos = indice.values[~np.isnan(indice.values)]
    if len(valores_validos) > 0:
        print(
            f"  {nombre:>4}: {np.min(valores_validos):.3f} - {np.max(valores_validos):.3f} "
            f"(μ: {np.mean(valores_validos):.3f}, n: {len(valores_validos)})"
        )
    else:
        print(f"  {nombre:>4}: Sin datos válidos")

# %% [markdown]
# ## 6. Visualización de Índices Temporales

# %%
# Visualizamos índices para fechas específicas (primera, media, última)
indices_fechas = [0, len(ds_refl.time) // 2, -1]

# Creamos figura con espacio adicional para colorbars separadas
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.05], hspace=0.3, wspace=0.1)

fig.suptitle(
    "Evolución Temporal de Índices de Vegetación", fontsize=16, fontweight="bold"
)

# Configuración de índices y sus parámetros
indices_config = [
    {
        "data": ndvi_temporal,
        "name": "NDVI",
        "cmap": "RdYlGn",
        "vmin": -0.2,
        "vmax": 0.8,
    },
    {"data": evi_temporal, "name": "EVI", "cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.6},
    {
        "data": ndwi_temporal,
        "name": "NDWI",
        "cmap": "RdYlBu",
        "vmin": -0.5,
        "vmax": 0.5,
    },
]

# Almacenamos las imágenes para las colorbars
images_for_colorbar = []

for i, indice_config in enumerate(indices_config):  # Filas = índices
    for j, idx_fecha in enumerate(indices_fechas):  # Columnas = fechas
        fecha_str = pd.to_datetime(ds_refl.time[idx_fecha].values).strftime("%Y-%m-%d")

        # Crear subplot en la grilla
        ax = fig.add_subplot(gs[i, j])

        # Mostrar imagen del índice para la fecha específica
        im = ax.imshow(
            indice_config["data"][idx_fecha],
            cmap=indice_config["cmap"],
            vmin=indice_config["vmin"],
            vmax=indice_config["vmax"],
        )

        # Títulos solo en la primera fila (fechas)
        if i == 0:
            ax.set_title(fecha_str, fontsize=12)

        # Etiquetas de índices solo en la primera columna
        if j == 0:
            ax.text(
                -0.15,
                0.5,
                indice_config["name"],
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                verticalalignment="center",
                rotation=90,
            )

        ax.axis("off")

        # Guardamos la primera imagen de cada fila para la colorbar
        if j == 0:
            images_for_colorbar.append(im)

# Añadimos colorbars separadas en la columna dedicada
for i, im in enumerate(images_for_colorbar):
    cbar_ax = fig.add_subplot(gs[i, 3])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)

plt.show()

print("🗓️ Fechas visualizadas:")
for idx in indices_fechas:
    fecha_str = pd.to_datetime(ds_refl.time[idx].values).strftime("%Y-%m-%d")
    print(f"   {fecha_str}")

print("\n📊 Organización del plot:")
print("   Filas: Índices (NDVI, EVI, NDWI)")
print("   Columnas: Fechas (primera, media, última)")

# %% [markdown]
# ## 7. Extracción de Series Temporales por Puntos

# %%
# @title Generamos puntos aleatorios dentro del área de estudio
np.random.seed(42)  # Para reproducibilidad

# Límites del área de estudio en coordenadas geográficas
bbox = CONFIG["bbox"]  # [oeste, sur, este, norte] en grados
lon_min, lat_min, lon_max, lat_max = bbox

# Generamos 6 puntos aleatorios en coordenadas geográficas
n_puntos = 6
lons = np.random.uniform(lon_min, lon_max, n_puntos)
lats = np.random.uniform(lat_min, lat_max, n_puntos)

# Transformamos coordenadas de EPSG:4326 al CRS de la imagen
image_crs = ds_refl.rio.crs
transformer = Transformer.from_crs("EPSG:4326", image_crs, always_xy=True)
x_utm, y_utm = transformer.transform(lons, lats)

# Creamos DataFrame con coordenadas en grados y UTM
coords_puntos = pd.DataFrame(
    {
        "punto_id": [f"Punto_{i + 1}" for i in range(n_puntos)],
        "lon": lons,
        "lat": lats,
        "x": x_utm,
        "y": y_utm,
    }
)

print("📍 Puntos de interés generados:")
print(f"   CRS imagen: {image_crs}")
for i, row in coords_puntos.iterrows():
    print(
        f"   {row['punto_id']:8}: Lon={row['lon']:.4f}°, Lat={row['lat']:.4f}° | X={row['x']:.0f}m, Y={row['y']:.0f}m"
    )

# %%
# @title ¿Dónde están los puntos?

# Creamos un dataset RGB para la imagen de muestra
rgb_combined = xr.concat(
    [
        imagen_muestra[config_sensor["bandas"]["red"]].rename("red"),
        imagen_muestra[config_sensor["bandas"]["green"]].rename("green"),
        imagen_muestra[config_sensor["bandas"]["blue"]].rename("blue"),
    ],
    dim="band",
)

# Configuramos la figura
fig, ax = plt.subplots(figsize=(14, 10))

# Usamos xarray.imshow para mostrar la imagen RGB con coordenadas automáticas
rgb_combined.plot.imshow(ax=ax, rgb="band", robust=True, add_colorbar=False)

# Usamos las coordenadas UTM ya transformadas
x_transformed = coords_puntos["x"].values
y_transformed = coords_puntos["y"].values

colores_puntos = ["red", "blue", "green", "orange", "purple", "cyan"]

# Configuramos el título
ax.set_title(
    f"Imagen RGB con Puntos de Interés | Fecha: {fecha_muestra_str} | Sensor: {config_sensor['descripcion']}",
    fontsize=14,
    fontweight="bold",
)

for i, row in coords_puntos.iterrows():
    color = colores_puntos[i % len(colores_puntos)]

    # Plotear punto en coordenadas transformadas
    ax.scatter(
        x_transformed[i],
        y_transformed[i],
        color=color,
        s=200,
        edgecolor="white",
        linewidth=3,
        alpha=0.9,
        zorder=10,
        label=row["punto_id"],
    )

    # Añadir etiqueta
    ax.annotate(
        row["punto_id"],
        (x_transformed[i], y_transformed[i]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8),
    )

# Añadimos leyenda
# ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=10)

# Ajustamos el diseño
plt.tight_layout()
plt.show()


# %%
# Preparar los datos para el análisis
indexes = xr.Dataset(
    {"NDVI": ndvi_temporal, "EVI": evi_temporal, "NDWI": ndwi_temporal}
)
indexes

# %%
# Extraemos series temporales para todos los puntos de una vez (método eficiente)
print("\n⏳ Extrayendo series temporales para todos los puntos...")
indices_nombres = ["NDVI", "EVI", "NDWI"]
indexes_puntos = indexes.sel(
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
serie_temporal = (
    indexes_puntos.to_dataframe()
    .reset_index()
    .melt(
        id_vars=["punto_id", "time", "y", "x"],
        value_vars=indices_nombres,
        var_name="indice",
        value_name="valor",
    )
)
serie_temporal

# %% [markdown]
# ## 8. Visualización Interactiva de Series Temporales

# %%
# @title Series temporales interactivas por punto

colores = ["green", "red", "blue", "orange", "purple", "cyan"]

# Creamos 3 subplots (uno por índice)
fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=["NDVI", "EVI", "NDWI"],
    shared_xaxes=True,
    vertical_spacing=0.08,
)

# Para cada índice, creamos un subplot
for j, indice in enumerate(indices_nombres):
    # Para cada punto, añadimos una línea en el subplot correspondiente
    for i, punto_id in enumerate(coords_puntos["punto_id"]):
        color = colores[i % len(colores)]

        # Filtramos datos del punto e índice específico
        serie_punto_indice = serie_temporal[
            (serie_temporal["punto_id"] == punto_id)
            & (serie_temporal["indice"] == indice)
        ].copy()

        if len(serie_punto_indice) > 0:
            # Filtramos valores no nulos
            serie_punto_indice = serie_punto_indice.dropna(subset=["valor"])

            if len(serie_punto_indice) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=serie_punto_indice["time"],
                        y=serie_punto_indice["valor"],
                        mode="lines+markers",
                        name=punto_id,
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        showlegend=(
                            j == 0
                        ),  # Solo mostrar leyenda en el primer subplot
                        legendgroup=punto_id,  # Agrupar por punto
                    ),
                    row=j + 1,
                    col=1,
                )

fig.update_layout(
    title="Series Temporales de Índices de Vegetación por Punto",
    height=900,
    hovermode="x unified",
    legend=dict(yanchor="top", y=1, xanchor="left", x=1.02),
)

# Configurar ejes Y
fig.update_yaxes(title_text="NDVI", row=1, col=1)
fig.update_yaxes(title_text="EVI", row=2, col=1)
fig.update_yaxes(title_text="NDWI", row=3, col=1)
fig.update_xaxes(title_text="Fecha", row=3, col=1)

fig.show()

# %%
# @title Estadísticas temporales por índice
# Calculamos estadísticas temporales usando serie_temporal
print("📊 Calculando estadísticas temporales...")

# Estadísticas agrupadas por tiempo (todas las fechas)
stats_por_tiempo = (
    serie_temporal.groupby(["time", "indice"])["valor"]
    .agg(["mean", "std", "min", "max", "count"])
    .round(4)
)

print("\n📅 Estadísticas por fecha e índice (primeras 10 fechas):")
print(stats_por_tiempo.head(30))

# Estadísticas agrupadas por punto
stats_por_punto = (
    serie_temporal.groupby(["punto_id", "indice"])["valor"]
    .agg(["mean", "std", "min", "max", "count"])
    .round(4)
)

print("\n📍 Estadísticas por punto e índice:")
print(stats_por_punto)

# Estadísticas globales por índice
stats_globales = (
    serie_temporal.groupby("indice")["valor"]
    .agg(["mean", "std", "min", "max", "count"])
    .round(4)
)

print("\n🌍 Estadísticas globales por índice:")
print(stats_globales)

# Preparamos datos para visualización temporal
serie_temporal_stats = (
    serie_temporal.groupby(["time", "indice"])["valor"]
    .agg(["mean", "std"])
    .reset_index()
)

# Visualización de estadísticas globales
fig = go.Figure()

# Medias por índice
for indice in ["NDVI", "EVI", "NDWI"]:
    datos_indice = serie_temporal_stats[serie_temporal_stats["indice"] == indice]

    fig.add_trace(
        go.Scatter(
            x=datos_indice["time"],
            y=datos_indice["mean"],
            mode="lines+markers",
            name=f"{indice} (Media Regional)",
            line=dict(width=2),
            error_y=dict(type="data", array=datos_indice["std"], visible=True),
        )
    )

fig.update_layout(
    title="Evolución Temporal de Índices - Estadísticas Regionales",
    xaxis_title="Fecha",
    yaxis_title="Valor del Índice",
    height=500,
    hovermode="x unified",
)

fig.show()

# %%
# @title Análisis estacional
# Agregamos información estacional
serie_temporal_copia = serie_temporal.copy()
serie_temporal_copia["mes"] = pd.to_datetime(serie_temporal_copia["time"]).dt.month
serie_temporal_copia["estacion"] = serie_temporal_copia["mes"].map(
    {
        12: "Verano",
        1: "Verano",
        2: "Verano",
        3: "Otoño",
        4: "Otoño",
        5: "Otoño",
        6: "Invierno",
        7: "Invierno",
        8: "Invierno",
        9: "Primavera",
        10: "Primavera",
        11: "Primavera",
    }
)

# Violin plot agrupado por estación (con box plot interno)
fig = go.Figure()

estaciones = ["Verano", "Otoño", "Invierno", "Primavera"]
colores_indices = {"NDVI": "green", "EVI": "red", "NDWI": "blue"}

for indice in ["NDVI", "EVI", "NDWI"]:
    for estacion in estaciones:
        datos_estacion_indice = serie_temporal_copia[
            (serie_temporal_copia["estacion"] == estacion)
            & (serie_temporal_copia["indice"] == indice)
        ]["valor"].dropna()

        if len(datos_estacion_indice) > 0:
            fig.add_trace(
                go.Violin(
                    y=datos_estacion_indice,
                    name=indice,
                    x=[estacion] * len(datos_estacion_indice),
                    fillcolor=colores_indices[indice],
                    opacity=0.7,
                    legendgroup=indice,
                    showlegend=(
                        estacion == estaciones[0]
                    ),  # Solo mostrar leyenda para la primera estación
                    box_visible=True,  # Mostrar box plot dentro del violin
                    meanline_visible=True,  # Mostrar línea de la media
                    points="outliers",  # Mostrar solo valores atípicos
                )
            )

fig.update_layout(
    title="Variabilidad Estacional de Índices de Vegetación",
    xaxis_title="Estación",
    yaxis_title="Valor del Índice",
    height=600,
    violinmode="group",
    xaxis=dict(categoryorder="array", categoryarray=estaciones),
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
)

fig.show()

# %% [markdown]
# ## 9. Análisis y Interpretación de Resultados

# %%
# Resumen estadístico por punto
print("📊 Resumen estadístico por punto de interés:")
print("=" * 80)

for punto_id in coords_puntos["punto_id"]:
    print(f"\n{punto_id}")
    print("-" * 30)

    for indice in ["NDVI", "EVI", "NDWI"]:
        # Filtramos datos del punto e índice específico
        valores = serie_temporal[
            (serie_temporal["punto_id"] == punto_id)
            & (serie_temporal["indice"] == indice)
        ]["valor"].dropna()

        if len(valores) > 0:
            media = valores.mean()
            std = valores.std()
            minimo = valores.min()
            maximo = valores.max()
            print(
                f"  {indice:4}: μ={media:.3f} ±{std:.3f} [{minimo:.3f}, {maximo:.3f}] (n={len(valores)})"
            )
        else:
            print(f"  {indice:4}: Sin datos válidos")

# %%
# Correlaciones entre índices
print("\n🔗 Análisis de correlaciones entre índices:")
print("=" * 50)

# Creamos tabla pivote para calcular correlaciones
serie_pivot = serie_temporal.pivot_table(
    index=["punto_id", "time"], columns="indice", values="valor"
).reset_index()

correlaciones = serie_pivot[["NDVI", "EVI", "NDWI"]].corr()
print("\nMatriz de correlación (todos los puntos):")
print(correlaciones.round(3))

# Correlaciones por punto (para explorar variabilidad espacial)
print("\nCorrelaciones por punto individual:")

for punto_id in coords_puntos["punto_id"]:
    datos_punto = serie_pivot[serie_pivot["punto_id"] == punto_id]
    if len(datos_punto) > 10:  # Suficientes datos
        corr_punto = datos_punto[["NDVI", "EVI", "NDWI"]].corr()
        print(f"\n{punto_id}:")
        print(f"  NDVI-EVI: {corr_punto.loc['NDVI', 'EVI']:.3f}")
        print(f"  NDVI-NDWI: {corr_punto.loc['NDVI', 'NDWI']:.3f}")
        print(f"  EVI-NDWI: {corr_punto.loc['EVI', 'NDWI']:.3f}")

# %% [markdown]
# ## 10. Interpretación y Conclusiones
#
# **Patrones Temporales Observados:**
#
# * **Ciclos Estacionales**: Los índices muestran variabilidad estacional típica del clima mediterráneo de Chile central
# * **NDVI y EVI**: Correlación alta en áreas vegetadas, diferencias en zonas de alta biomasa
# * **NDWI**: Complementario para detectar estrés hídrico y cuerpos de agua
#
# **Variabilidad Espacial Observada:**
#
# * **Heterogeneidad entre puntos**: Los diferentes puntos muestran patrones temporales únicos
# * **Correlaciones variables**: Las relaciones entre índices varían según la ubicación
# * **Respuesta diferencial**: Cada punto responde de manera distinta a cambios estacionales
#
# **Limitaciones del Análisis Actual:**
#
# * **⚠️ Máscara de Calidad Faltante**: Este análisis no utiliza las bandas de calidad (QA)
#   disponibles en los productos Landsat y Sentinel-2 para enmascarar automáticamente:
#   - Píxeles con nubes y sombras de nubes
#   - Píxeles con cirrus (nubes altas)
#   - Píxeles saturados o con problemas instrumentales
#   - Agua y nieve (cuando no son de interés)
#
# **Próximos Pasos Recomendados:**
#
# 1. **Implementar máscara de calidad** para mejorar la precisión temporal
# 2. **Análisis de tendencias** a largo plazo (>3 años)
# 3. **Detección automática de cambios** usando algoritmos como BFAST
# 4. **Validación con datos de campo** para calibrar interpretaciones
# 5. **Análisis de eventos específicos** (sequías, incendios, intervenciones)
