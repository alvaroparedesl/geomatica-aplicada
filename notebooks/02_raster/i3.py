# %% [markdown]
# # Trabajo Final: Análisis Multitemporal de Índices de Vegetación
#
# Este notebook guía el análisis de series temporales de índices de vegetación (NDVI y EVI)
# utilizando imágenes Landsat Collection 2 Level 2 para diferentes tipos de cobertura terrestre.
#
# ## Objetivos del Trabajo
#
# 1. Identificar un sector de estudio con características específicas
# 2. Analizar un rango temporal de 2 años (comenzando en junio)
# 3. Procesar imágenes Landsat C2L2 (reflectancia y enmascaramiento de nubes)
# 4. Generar puntos de interés en diferentes usos del suelo
# 5. Extraer y comparar series temporales de NDVI y EVI

# %%
# Instalación de bibliotecas necesarias
# %pip install rioxarray xarray matplotlib numpy rasterio geopandas planetary-computer pystac-client odc-stac plotly

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import plotly.graph_objects as go
import rioxarray as rxr
import xarray as xr
from pyproj import Transformer
from pystac_client import Client
from plotly.subplots import make_subplots

# %% [markdown]
# ## 1. Configuración del Área de Estudio
#
# **IMPORTANTE**: Debes elegir un sector que cumpla con las siguientes características:
#
# - Área de aproximadamente 150-250 km² (similar al ejemplo del Valle del Maipo)
# - Debe contener una mezcla de:
#   - Zonas urbanas
#   - Cultivos agrícolas
#   - Bosque nativo
#   - Plantaciones forestales
#   - Otros usos (agua, suelo desnudo, nieve, etc.)
#
# La zona elegida debe ser aprobada por el profesor antes de continuar.
#
# El bbox se define como [min_lon, min_lat, max_lon, max_lat] en coordenadas geográficas (WGS84).

# %%
# Configuración del área de estudio
# MODIFICA ESTOS VALORES según tu área de estudio aprobada
CONFIG = {
    "bbox": [-70.8, -33.7, -70.6, -33.6],  # Ejemplo: Valle del Maipo, Chile
    "epsg": "EPSG:32719",  # UTM Zone 19S (ajusta según tu zona)
    "fecha_inicio": "2022-06-01",  # Inicio: junio (2 años de datos)
    "fecha_fin": "2024-05-31",  # Fin: mayo del segundo año
    "resolucion": 30,  # Resolución espacial en metros
    "max_nubes": 20,  # Porcentaje máximo de cobertura de nubes permitido
}

# Calcular área aproximada del bbox
lon_diff = CONFIG["bbox"][2] - CONFIG["bbox"][0]
lat_diff = CONFIG["bbox"][3] - CONFIG["bbox"][1]
# Aproximación: 1 grado de latitud ≈ 111 km, 1 grado de longitud ≈ 111 km * cos(latitud)
lat_media = (CONFIG["bbox"][1] + CONFIG["bbox"][3]) / 2
area_km2 = lon_diff * 111 * np.cos(np.radians(lat_media)) * lat_diff * 111

print(f"Área de estudio: {area_km2:.1f} km²")
print(f"Rango temporal: {CONFIG['fecha_inicio']} a {CONFIG['fecha_fin']}")
print(f"Resolución: {CONFIG['resolucion']} m")

# %% [markdown]
# ## 2. Configuración de Landsat Collection 2 Level 2
#
# Landsat C2L2 proporciona datos de reflectancia superficial corregidos atmosféricamente.
# Utilizaremos las bandas necesarias para calcular NDVI y EVI.

# %%
# Configuración de bandas y parámetros de Landsat C2L2
bandas = {
    "red": "red",
    "green": "green",
    "blue": "blue",
    "nir": "nir08",  # NIR para Landsat 8/9
}

# Parámetros de conversión a reflectancia para Landsat C2L2
factor_reflectancia = 0.0000275
offset = -0.2
rango_valido = (0, 1)

# Configuración de la colección
coleccion = "landsat-c2-l2"
plataforma_filtro = ["landsat-8", "landsat-9"]

# Bandas requeridas incluyendo QA_PIXEL para enmascaramiento de nubes
bandas_requeridas = [
    bandas["red"],
    bandas["green"],
    bandas["blue"],
    bandas["nir"],
    "qa_pixel",  # Banda de calidad para enmascaramiento
]

# %% [markdown]
# ## 3. Búsqueda de Imágenes Landsat
#
# Utilizamos Planetary Computer para acceder a las imágenes Landsat C2L2.

# %%
# Conexión al catálogo STAC de Planetary Computer
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Parámetros de búsqueda
query_params = {
    "collections": [coleccion],
    "bbox": CONFIG["bbox"],
    "datetime": f"{CONFIG['fecha_inicio']}/{CONFIG['fecha_fin']}",
    "query": {
        "eo:cloud_cover": {"lt": CONFIG["max_nubes"]},
        "platform": {"in": plataforma_filtro},
    },
}

# Realizar búsqueda
search = catalog.search(**query_params)
items = list(search.items())

print(f"Imágenes encontradas: {len(items)}")
if len(items) > 0:
    fechas = [item.datetime.strftime("%Y-%m-%d") for item in items]
    print(f"Rango de fechas: {min(fechas)} a {max(fechas)}")
else:
    print("ADVERTENCIA: No se encontraron imágenes. Verifica el bbox y las fechas.")

# %% [markdown]
# ## 4. Carga de Datos
#
# Cargamos todas las imágenes de la serie temporal con las bandas necesarias.

# %%
# Parámetros de carga
load_params = {
    "items": items,
    "bands": bandas_requeridas,
    "bbox": CONFIG["bbox"],
    "crs": CONFIG["epsg"],
    "resolution": CONFIG["resolucion"],
    "groupby": "solar_day",  # Agrupar por día solar para evitar duplicados
}

# Cargar serie temporal completa
ds_temporal = odc.stac.load(**load_params)

print(f"Datos cargados: {len(ds_temporal.time)} fechas")
print(f"Bandas disponibles: {list(ds_temporal.data_vars)}")
print(f"Dimensiones: {dict(ds_temporal.dims)}")

# %% [markdown]
# ## 5. Conversión a Reflectancia y Enmascaramiento de Nubes
#
# Los datos Landsat C2L2 vienen en valores de reflectancia escalados. Debemos:
# 1. Convertir a reflectancia real (0-1)
# 2. Aplicar máscara de nubes usando la banda QA_PIXEL
#
# La banda QA_PIXEL contiene información sobre la calidad de cada píxel, incluyendo
# la presencia de nubes. Utilizaremos los bits estándar de Landsat para identificar nubes.

# %%
# Función para enmascarar nubes usando QA_PIXEL
def aplicar_mascara_nubes(qa_pixel):
    """
    Aplica máscara de nubes basada en la banda QA_PIXEL de Landsat C2L2.
    
    Bits relevantes en QA_PIXEL:
    - Bit 3: Dilated Cloud (nube dilatada)
    - Bit 1: Cloud (nube)
    - Bit 2: Cloud Shadow (sombra de nube)
    - Bit 0: Fill (datos faltantes)
    
    Retorna una máscara donde True = píxel válido, False = píxel con nubes/sombra/fill
    """
    # Máscara para píxeles válidos (sin nubes, sombras o fill)
    mascara_valida = (
        (qa_pixel & 0b00000001 == 0)  # No Fill
        & (qa_pixel & 0b00000010 == 0)  # No Cloud Shadow
        & (qa_pixel & 0b00000100 == 0)  # No Cloud
        & (qa_pixel & 0b00001000 == 0)  # No Dilated Cloud
    )
    return mascara_valida


# Conversión a reflectancia
ds_refl = ds_temporal[bandas_requeridas[:-1]] * factor_reflectancia + offset

# Aplicar rango válido de reflectancia
rango_min, rango_max = rango_valido
ds_refl = ds_refl.where((ds_refl >= rango_min) & (ds_refl <= rango_max))

# Obtener banda QA_PIXEL
qa_pixel = ds_temporal["qa_pixel"]

# Aplicar máscara de nubes a todas las bandas
mascara_nubes = aplicar_mascara_nubes(qa_pixel)

# Aplicar máscara a todas las bandas de reflectancia
for banda in ds_refl.data_vars:
    ds_refl[banda] = ds_refl[banda].where(mascara_nubes)

print("Conversión a reflectancia completada")
print("Máscara de nubes aplicada")

# Calcular porcentaje de píxeles válidos por fecha
pixeles_validos = mascara_nubes.sum(dim=["x", "y"]) / (
    mascara_nubes.sizes["x"] * mascara_nubes.sizes["y"]
) * 100
print(f"\nPorcentaje promedio de píxeles válidos (sin nubes): {float(pixeles_validos.mean()):.1f}%")

# %% [markdown]
# ## 6. Cálculo de Índices de Vegetación
#
# Calculamos NDVI (Normalized Difference Vegetation Index) y EVI (Enhanced Vegetation Index)
# para toda la serie temporal.
#
# **NDVI**: (NIR - Red) / (NIR + Red)
# - Rango: -1 a 1
# - Valores altos (>0.5) indican vegetación densa
#
# **EVI**: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
# - Rango: -1 a 1
# - Mejor sensibilidad en áreas de alta biomasa que NDVI

# %%
# Extraer bandas necesarias
nir = ds_refl[bandas["nir"]]
red = ds_refl[bandas["red"]]
blue = ds_refl[bandas["blue"]]

# Cálculo de NDVI
denominador_ndvi = nir + red
ndvi_temporal = xr.where(
    denominador_ndvi != 0, (nir - red) / denominador_ndvi, np.nan
)
# Filtrar valores fuera del rango válido
ndvi_temporal = xr.where(
    (ndvi_temporal >= -1) & (ndvi_temporal <= 1), ndvi_temporal, np.nan
)

# Cálculo de EVI
denominador_evi = nir + 6 * red - 7.5 * blue + 1
evi_temporal = xr.where(
    denominador_evi != 0, 2.5 * (nir - red) / denominador_evi, np.nan
)
# Filtrar valores fuera del rango válido
evi_temporal = xr.where(
    (evi_temporal >= -1) & (evi_temporal <= 1), evi_temporal, np.nan
)

print("Índices NDVI y EVI calculados para toda la serie temporal")

# %% [markdown]
# ## 7. Definición de Puntos de Interés
#
# Define puntos de interés para cada tipo de uso del suelo. Debes identificar
# coordenadas específicas en tu área de estudio para:
#
# - **Urbano**: Zonas urbanas, ciudades, áreas construidas
# - **Agrícola**: Campos de cultivo, áreas agrícolas
# - **Bosque Nativo**: Áreas de bosque nativo
# - **Plantación Forestal**: Plantaciones de especies exóticas (pino, eucalipto)
# - **Otro**: Agua, suelo desnudo, nieve, etc.
#
# Las coordenadas deben estar en el sistema de coordenadas del dataset (UTM).
# Puedes usar herramientas como Google Earth o QGIS para identificar las coordenadas.

# %%
# Definición de puntos de interés por tipo de uso
# MODIFICA estas coordenadas según tu área de estudio
# Las coordenadas deben estar en el sistema de coordenadas del dataset (UTM)

# Ejemplo de estructura (coordenadas en UTM)
# Debes reemplazar estas coordenadas con las de tu área de estudio
puntos_interes = {
    "urbano": {
        "x": [700000],  # Coordenada X en UTM
        "y": [6300000],  # Coordenada Y en UTM
        "nombre": "Zona Urbana",
    },
    "agricola": {
        "x": [702000],
        "y": [6302000],
        "nombre": "Cultivo Agrícola",
    },
    "bosque_nativo": {
        "x": [704000],
        "y": [6304000],
        "nombre": "Bosque Nativo",
    },
    "plantacion": {
        "x": [706000],
        "y": [6306000],
        "nombre": "Plantación Forestal",
    },
    "otro": {
        "x": [708000],
        "y": [6308000],
        "nombre": "Otro (agua/suelo desnudo)",
    },
}

# Crear DataFrame con todos los puntos
lista_puntos = []
for tipo_uso, datos in puntos_interes.items():
    for i, (x, y) in enumerate(zip(datos["x"], datos["y"])):
        lista_puntos.append(
            {
                "tipo_uso": tipo_uso,
                "punto_id": f"{tipo_uso}_{i+1}",
                "nombre": datos["nombre"],
                "x": x,
                "y": y,
            }
        )

coords_puntos = pd.DataFrame(lista_puntos)

print(f"Puntos de interés definidos: {len(coords_puntos)}")
print("\nPuntos por tipo de uso:")
print(coords_puntos.groupby("tipo_uso")["punto_id"].count())

# %% [markdown]
# ## 8. Extracción de Series Temporales
#
# Extraemos los valores de NDVI y EVI para cada punto de interés a lo largo
# de toda la serie temporal.

# %%
# Crear Dataset con ambos índices
indices_data = xr.Dataset({"NDVI": ndvi_temporal, "EVI": evi_temporal})

# Extraer valores en los puntos de interés usando interpolación al vecino más cercano
indices_puntos = indices_data.sel(
    x=xr.DataArray(
        coords_puntos["x"].values,
        dims="punto_id",
        coords={"punto_id": coords_puntos["punto_id"].values},
    ),
    y=xr.DataArray(
        coords_puntos["y"].values,
        dims="punto_id",
        coords={"punto_id": coords_puntos["punto_id"].values},
    ),
    method="nearest",
)

# Convertir a DataFrame para análisis y visualización
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

# Agregar información de tipo de uso
serie_temporal = serie_temporal.merge(
    coords_puntos[["punto_id", "tipo_uso", "nombre"]], on="punto_id", how="left"
)

print(f"Series temporales extraídas: {len(serie_temporal)} observaciones")
print(f"Fechas únicas: {serie_temporal['time'].nunique()}")
print(f"Puntos únicos: {serie_temporal['punto_id'].nunique()}")

# %% [markdown]
# ## 9. Comparación Gráfica de Series Temporales
#
# Visualizamos las series temporales de NDVI y EVI para comparar el comportamiento
# de ambos índices en los diferentes tipos de uso del suelo.

# %%
# Configuración de colores por tipo de uso
colores_uso = {
    "urbano": "red",
    "agricola": "orange",
    "bosque_nativo": "green",
    "plantacion": "darkgreen",
    "otro": "blue",
}

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
    # Para cada tipo de uso
    for tipo_uso in puntos_interes.keys():
        color = colores_uso.get(tipo_uso, "gray")
        
        # Filtrar datos para este tipo de uso e índice
        datos_tipo = serie_temporal[
            (serie_temporal["tipo_uso"] == tipo_uso)
            & (serie_temporal["indice"] == indice)
        ].dropna(subset=["valor"])
        
        if len(datos_tipo) > 0:
            # Agrupar por fecha y calcular promedio si hay múltiples puntos del mismo tipo
            datos_agrupados = (
                datos_tipo.groupby("time")["valor"].mean().reset_index()
            )
            
            fig.add_trace(
                go.Scatter(
                    x=datos_agrupados["time"],
                    y=datos_agrupados["valor"],
                    mode="lines+markers",
                    name=tipo_uso.replace("_", " ").title(),
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    showlegend=(j == 0),  # Solo mostrar leyenda en primer subplot
                    legendgroup=tipo_uso,
                ),
                row=1,
                col=j + 1,
            )

# Configurar layout
fig.update_layout(
    title="Comparación de Series Temporales: NDVI vs EVI por Tipo de Uso",
    height=600,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig.update_yaxes(title_text="NDVI", row=1, col=1, range=[-0.2, 1.0])
fig.update_yaxes(title_text="EVI", row=1, col=2, range=[-0.2, 1.0])
fig.update_xaxes(title_text="Fecha", row=1, col=1)
fig.update_xaxes(title_text="Fecha", row=1, col=2)

fig.show()

# %% [markdown]
# ## 10. Análisis Estadístico
#
# Calculamos estadísticas descriptivas para cada tipo de uso y índice.

# %%
# Estadísticas por tipo de uso e índice
stats = (
    serie_temporal.groupby(["tipo_uso", "indice"])["valor"]
    .agg(["mean", "std", "min", "max", "count"])
    .round(3)
)

print("Estadísticas descriptivas por tipo de uso e índice:")
print(stats)

# %% [markdown]
# ## Resumen del Trabajo
#
# En este notebook has:
#
# 1. Configurado un área de estudio con características específicas
# 2. Buscado y cargado imágenes Landsat C2L2 para un período de 2 años
# 3. Convertido los datos a reflectancia y aplicado enmascaramiento de nubes
# 4. Calculado índices de vegetación (NDVI y EVI)
# 5. Definido puntos de interés para diferentes tipos de uso del suelo
# 6. Extraído series temporales para cada punto
# 7. Comparado gráficamente el comportamiento de NDVI y EVI
#
# **Preguntas para el análisis**:
# - ¿Qué diferencias observas entre NDVI y EVI en cada tipo de uso?
# - ¿Cómo varían los índices a lo largo del año en cada tipo de uso?
# - ¿Qué tipo de uso muestra mayor variabilidad temporal?
# - ¿Hay diferencias estacionales claras en algún tipo de uso?

