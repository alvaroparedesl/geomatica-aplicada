# %% [markdown]
# # Análisis Multitemporal de Índices de Vegetación
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/notebooks/02_raster/07_analisis_multitemporal.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/) [![Último commit](https://img.shields.io/github/last-commit/alvaroparedesl/geomatica-aplicada?label=%C3%BAltimo%20commit)](https://github.com/alvaroparedesl/geomatica-aplicada/commits/main)
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
#
# Primero preparamos el entorno: la celda `Preparar entorno` deja disponible la carpeta `utils/` del curso, donde guardamos funciones de apoyo (buscar y cargar imágenes, máscara de nubes, índices, series temporales, etc.).

# %%
# @title Preparar entorno (ejecutar primero)
import subprocess
import sys
from pathlib import Path

if "google.colab" in sys.modules:  # Colab: clonar el repositorio (y actualizarlo si ya estaba)
    RAIZ = Path("/content/geomatica-aplicada")
    if not RAIZ.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/alvaroparedesl/geomatica-aplicada", str(RAIZ)], check=True)
    else:  # una sesión anterior ya lo clonó: traer lo nuevo para no usar código viejo
        subprocess.run(["git", "-C", str(RAIZ), "pull", "-q", "--ff-only"], check=False)
else:  # Local: subir desde la carpeta actual hasta encontrar utils/
    RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "utils").is_dir())
sys.path.insert(0, str(RAIZ))

# %%
# @title Instalación de paquetes necesarios
# %pip install -q rioxarray xarray matplotlib numpy rasterio geopandas planetary-computer pystac-client odc-stac plotly

# %%
# @title Importación de bibliotecas
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import plotly.graph_objects as go
import rioxarray  # noqa: F401  (activa el accesor .rio)

# Funciones de apoyo del curso (carpeta utils/)
from utils.raster import (
    calcular_indices,
    convertir_a_reflectancia,
    extraer_series_puntos,
    lonlat_a_crs,
    nombres_bandas,
    obtener_configuracion_sensor,
    preparar_reflectancia,
)
from utils.stac import (
    agregar_offset,
    conectar_catalogo,
    parametros_busqueda,
    parametros_carga,
    preparar_items,
    renombrar_bandas,
    resumir_imagenes,
)
from utils.viz import (
    crear_imagen_rgb,
    graficar_series_temporales,
    mostrar_paneles,
)

# Configuración para visualización
plt.rcParams["figure.figsize"] = (12, 8)
plt.style.use("ggplot")

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
    "enmascarar_nubes": True,  # Descarta nubes y sombras usando la banda de calidad
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
# ## 2. Funciones de apoyo
#
# En el cuaderno anterior vimos cómo se calculan los índices de vegetación. Aquí reutilizamos esas funciones
# desde la carpeta `utils/`, junto con otras nuevas para trabajar con series de tiempo:
#
# | Paso | Función | Qué hace |
# |------|---------|----------|
# | Búsqueda | `parametros_busqueda`, `preparar_items`, `resumir_imagenes` | Arman los argumentos de `catalogo.search(...)`, limpian los resultados y los resumen en una tabla |
# | Bandas | `nombres_bandas(..., con_calidad=True)` | Traduce nombres genéricos ("nir") a los del sensor y agrega la banda de calidad |
# | Carga | `parametros_carga`, `renombrar_bandas`, `agregar_offset` | Arman los argumentos de `odc.stac.load(...)` (Dataset de xarray con time, y, x) y dejan los nombres de banda y el offset de cada fecha |
# | Calidad | `preparar_reflectancia` | Enmascara nubes y sombras, y convierte a reflectancia |
# | Índices | `calcular_indices` | Calcula NDVI, EVI y NDWI para todas las fechas |
# | Puntos | `lonlat_a_crs`, `extraer_series_puntos` | Transforma coordenadas y extrae la serie de cada punto |
# | Gráficos | `graficar_series_temporales`, `mostrar_paneles` | Series interactivas y paneles de imágenes |
#
# > **Nota:** la conversión a reflectancia es `DN * factor + offset`. En Landsat el factor es 0.0000275 y el
# > offset -0.2; en Sentinel-2 el factor es 0.0001 y el offset es -0.1 en las imágenes con versión de
# > procesamiento (baseline) 04.00 o posterior (introducida el 25 de enero de 2022) y 0 en las anteriores.
# > Depende del catálogo y del baseline de cada imagen, y los helpers lo manejan por imagen.
# >
# > Las llamadas `catalogo.search(...)` y `odc.stac.load(...)` las escribimos nosotros; los helpers de
# > `utils.stac` solo arman sus argumentos y ordenan los datos.

# %% [markdown]
# ## 3. Búsqueda y Carga de Series Temporales

# %%
# Obtenemos la configuración del sensor seleccionado
config_sensor = obtener_configuracion_sensor(CONFIG["sensor"])

print(f"🛰️ Trabajando con: {config_sensor['descripcion']}")
print(f"   Colección: {config_sensor['coleccion']}")
print(f"   Período temporal: {CONFIG['fecha_inicio']} a {CONFIG['fecha_fin']}")

# %%
# Conectamos al catálogo STAC. Landsat solo existe en Planetary Computer; Sentinel-2 usa el catálogo
# por defecto del curso (`CATALOGO_SENTINEL2` en utils/stac.py).
catalog = conectar_catalogo("planetary" if CONFIG["sensor"] == "landsat" else None)

# Buscamos (período extendido): `parametros_busqueda` arma los argumentos de la búsqueda nativa
parametros = parametros_busqueda(
    CONFIG["sensor"],
    CONFIG["bbox"],
    (CONFIG["fecha_inicio"], CONFIG["fecha_fin"]),
    nubes_max=CONFIG["max_nubes"],
    catalogo=catalog,
)
busqueda = catalog.search(**parametros)
items = busqueda.item_collection()

# Quitamos versiones repetidas de una misma toma (Sentinel-2) y ordenamos por fecha
items = preparar_items(items)

print("🔍 Búsqueda temporal completada:")
print(f"   Imágenes encontradas: {len(items)}")

if len(items) == 0:
    print("❌ No se encontraron imágenes. Ajusta los parámetros de búsqueda.")
elif len(items) < CONFIG["min_imagenes"]:
    print(f"⚠️ Pocas imágenes encontradas ({len(items)} < {CONFIG['min_imagenes']})")
    print("   Considera:")
    print("   - Aumentar el porcentaje máximo de nubes")
    print("   - Ampliar el período temporal")
    print("   - Cambiar la resolución espacial")
else:
    print("✅ Suficientes imágenes para análisis temporal")

if len(items) > 0:
    # Tabla resumen: fecha, % de nubes e id de cada imagen
    tabla_imagenes = resumir_imagenes(items)
    fechas_disponibles = pd.to_datetime(tabla_imagenes["fecha"])

    print("\n🕒 Rango de fechas:")
    print(f"   Primera: {fechas_disponibles.min().strftime('%Y-%m-%d')}")
    print(f"   Última: {fechas_disponibles.max().strftime('%Y-%m-%d')}")

    # Distribución temporal
    distribucion = fechas_disponibles.dt.month.value_counts().sort_index()
    print("\n📊 Distribución por mes:")
    for mes, count in distribucion.items():
        print(f"   Mes {mes}: {count} imágenes")

# %% [markdown]
# Además de las bandas espectrales, pedimos la **banda de calidad** (`con_calidad=True`): en Landsat es
# `qa_pixel` y en Sentinel-2 es `SCL`. Cada píxel de esa banda indica si hay nubes, sombras, cirrus, etc.,
# y nos permitirá descartarlos más adelante.

# %%
# Definimos las bandas necesarias (nombres genéricos -> nombres del sensor)
nombres_genericos = ["red", "green", "blue", "nir", "swir1"]
bandas_espectrales = nombres_bandas(CONFIG["sensor"], nombres_genericos)
bandas_requeridas = nombres_bandas(CONFIG["sensor"], nombres_genericos, con_calidad=True)

print(f"🎯 Bandas espectrales: {bandas_espectrales}")
print(f"🎯 Bandas a cargar (con banda de calidad): {bandas_requeridas}")

# Carga sin chunks para análisis temporal
if CONFIG["cargar_en_memoria"]:
    chunks = None
    print("🧠 Cargando serie temporal completa en memoria")
else:
    chunks = {"x": 512, "y": 512, "time": 5}
    print("📦 Cargando con chunks temporal")

# Cargamos la serie temporal
print("⏳ Cargando serie temporal... (esto puede tomar varios minutos)")
parametros_ds = parametros_carga(
    CONFIG["sensor"],
    bandas_requeridas,
    resolucion=CONFIG["resolucion"],
    crs=CONFIG["epsg"],
    catalogo=catalog,
    bbox=CONFIG["bbox"],
    chunks=chunks,
)
ds_temporal = odc.stac.load(items, **parametros_ds)

# Nombres de banda estándar y offset de reflectancia de cada fecha (Landsat queda igual)
ds_temporal = renombrar_bandas(ds_temporal, catalogo=catalog)
ds_temporal = agregar_offset(ds_temporal, items)

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
# ## 4. Máscara de Nubes y Conversión a Reflectancia
#
# Las nubes y sus sombras no representan la superficie, y distorsionan los índices (por ejemplo, una nube
# tiene un NDVI cercano a 0 y haría parecer que la vegetación "desapareció" ese día). Por eso, antes de
# calcular índices, **enmascaramos** esos píxeles: los reemplazamos por `NaN` para que no entren en los cálculos.
#
# * **Landsat**: la banda `qa_pixel` guarda, bit a bit, si el píxel es relleno, nube, nube dilatada, cirrus o sombra.
# * **Sentinel-2**: la banda `SCL` (clasificación de escena) asigna a cada píxel una clase (nube media, nube alta, sombra, cirrus...).
#
# La función `preparar_reflectancia` hace dos cosas en un paso: aplica la máscara de calidad (y elimina la banda
# de calidad del resultado) y convierte los números digitales a reflectancia (`DN * factor + offset`).
# Para ver su efecto, comparamos con la conversión sin máscara.

# %%
# Serie SIN máscara (solo conversión a reflectancia, sin la banda de calidad)
ds_refl_sin_mascara = convertir_a_reflectancia(
    ds_temporal[bandas_espectrales], CONFIG["sensor"]
)

# Serie lista para analizar: máscara de nubes (opcional) + reflectancia
print("🔄 Preparando serie temporal (máscara de calidad + reflectancia)...")
ds_refl = preparar_reflectancia(
    ds_temporal, CONFIG["sensor"], enmascarar=CONFIG["enmascarar_nubes"]
)
ds_refl = ds_refl[bandas_espectrales]  # solo las bandas espectrales (sin la banda de calidad)

print("✅ Serie temporal convertida a reflectancia")
print(f"   Máscara de nubes aplicada: {CONFIG['enmascarar_nubes']}")

# Porcentaje de píxeles válidos por fecha, con y sin máscara (banda roja)
banda_roja = config_sensor["bandas"]["red"]
validos = pd.DataFrame(
    {
        "sin_mascara_%": ds_refl_sin_mascara[banda_roja].notnull().mean(("x", "y")).values * 100,
        "con_mascara_%": ds_refl[banda_roja].notnull().mean(("x", "y")).values * 100,
    },
    index=pd.to_datetime(ds_refl.time.values).strftime("%Y-%m-%d"),
)
print("\n📊 Píxeles válidos por fecha (primeras 10 fechas):")
print(validos.head(10).round(1))

# Verificamos rangos por banda y fecha
print("\n📈 Verificación de rangos (muestra de primeras 3 fechas):")
print("=" * 70)

for i, fecha in enumerate(ds_refl.time[:3]):
    fecha_str = pd.to_datetime(fecha.values).strftime("%Y-%m-%d")
    print(f"\nFecha: {fecha_str}")

    imagen_fecha = ds_refl.sel(time=fecha)
    for banda in bandas_espectrales:
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

mostrar_paneles(
    [
        (f"Color Verdadero (RGB)\n{config_sensor['descripcion']}", rgb_muestra),
        ("Falso Color (NIR-R-G)\nVegetación en tonos rojos", falso_color_muestra),
    ],
    ncols=2,
    titulo=(
        f"Serie Temporal - Imagen de Muestra: {fecha_muestra_str} | "
        f"{len(ds_refl.time)} imágenes | Resolución: {CONFIG['resolucion']}m"
    ),
)

print(f"🖼️ Imagen de muestra mostrada: {fecha_muestra_str}")
print(f"📊 Posición en serie: {len(ds_refl.time) // 2 + 1}/{len(ds_refl.time)}")

# %% [markdown]
# ### Efecto de la máscara de nubes
#
# Comparemos la fecha con más píxeles descartados, sin y con máscara. Los píxeles enmascarados quedan
# en `NaN` y se ven en blanco.

# %%
# Fecha con menos píxeles válidos tras aplicar la máscara
pos_nubosa = int(np.argmin(validos["con_mascara_%"].values))
fecha_nubosa_str = validos.index[pos_nubosa]

rgb_sin_mascara = crear_imagen_rgb(
    ds_refl_sin_mascara.isel(time=pos_nubosa), config_sensor["rgb_bandas"]
)
rgb_con_mascara = crear_imagen_rgb(ds_refl.isel(time=pos_nubosa), config_sensor["rgb_bandas"])

mostrar_paneles(
    [
        ("Sin máscara de nubes", rgb_sin_mascara),
        ("Con máscara de nubes\n(píxeles descartados en blanco)", rgb_con_mascara),
    ],
    ncols=2,
    titulo=f"Efecto de la máscara de calidad - {fecha_nubosa_str}",
)

# %% [markdown]
# ## 5. Cálculo de Índices para Serie Temporal

# %%
# Calculamos NDVI, EVI y NDWI para toda la serie temporal de una sola vez
print("🕒 Calculando índices para serie temporal...")

indexes = calcular_indices(ds_refl, CONFIG["sensor"])  # Dataset con NDVI, EVI y NDWI

ndvi_temporal = indexes["NDVI"]
evi_temporal = indexes["EVI"]
ndwi_temporal = indexes["NDWI"]

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
x_utm, y_utm = lonlat_a_crs(lons, lats, image_crs)

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

# Configuramos la figura
fig, ax = plt.subplots(figsize=(14, 10))

# Mostramos la imagen RGB de muestra, ubicada con sus coordenadas UTM
ax.imshow(
    rgb_muestra,
    extent=[
        float(imagen_muestra.x.min()),
        float(imagen_muestra.x.max()),
        float(imagen_muestra.y.min()),
        float(imagen_muestra.y.max()),
    ],
)

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

# Ajustamos el diseño
plt.tight_layout()
plt.show()


# %%
# Los índices de todas las fechas están en el Dataset `indexes` (NDVI, EVI, NDWI)
indexes

# %%
# Extraemos series temporales para todos los puntos de una vez.
# Para cada punto se toma el píxel más cercano; el resultado queda en formato largo
# (una fila por punto, fecha e índice).
print("\n⏳ Extrayendo series temporales para todos los puntos...")
indices_nombres = ["NDVI", "EVI", "NDWI"]
serie_temporal = extraer_series_puntos(indexes, coords_puntos, columna_id="punto_id")
serie_temporal

# %% [markdown]
# ## 8. Visualización Interactiva de Series Temporales

# %%
# @title Series temporales interactivas por punto

colores = ["green", "red", "blue", "orange", "purple", "cyan"]

# Un gráfico por índice (uno bajo otro), con una línea por punto
graficar_series_temporales(
    serie_temporal,
    columna_id="punto_id",
    indices=indices_nombres,
    colores=colores,
    titulo="Series Temporales de Índices de Vegetación por Punto",
)

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
# **Máscara de Calidad:**
#
# En este análisis usamos la banda de calidad (`qa_pixel` en Landsat, `SCL` en Sentinel-2) para descartar
# automáticamente píxeles con nubes, sombras de nubes, cirrus y píxeles de relleno. Con esto las series
# temporales son mucho más limpias: desaparecen las caídas bruscas del NDVI causadas por nubes.
#
# **Limitaciones del Análisis Actual:**
#
# * **Máscara imperfecta**: La banda de calidad es un algoritmo automático. Puede dejar pasar neblina
#   o nubes muy delgadas, y a veces descarta píxeles brillantes que sí son válidos (por ejemplo, suelo
#   desnudo muy claro).
# * **Agua y nieve**: No se excluyen por defecto. Si no son de interés, se pueden agregar al parámetro
#   `excluir` de `preparar_reflectancia` (por ejemplo, `excluir=["nube", "sombra_nube", "nieve"]` en Landsat).
# * **Menos fechas útiles por píxel**: Al enmascarar, algunos píxeles quedan sin datos en ciertas fechas
#   (se ven como huecos en las series).
#
# **Próximos Pasos Recomendados:**
#
# 1. **Probar otras máscaras**: cambiar `excluir` en `preparar_reflectancia` o comparar con `enmascarar_nubes: False`
# 2. **Análisis de tendencias** a largo plazo (>3 años)
# 3. **Detección automática de cambios** usando algoritmos como BFAST
# 4. **Validación con datos de campo** para calibrar interpretaciones
# 5. **Análisis de eventos específicos** (sequías, incendios, intervenciones)
