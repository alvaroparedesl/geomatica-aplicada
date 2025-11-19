# %% [markdown]
# # Trabajo Final (Interrogación 3)
#
# ## Instrucciones
#
# ### Objetivo:
#
# Analizar series temporales de índices de vegetación (NDVI y EVI) utilizando imágenes
# Landsat Collection 2 Level 2 para diferentes tipos de cobertura terrestre, con el fin
# de comparar el comportamiento temporal de ambos índices y determinar cuál es más
# adecuado para el análisis de coberturas específicas en el área de estudio.
#
# ### Pasos:
#
# 1. Identificar un sector a estudiar que cumpla con las siguientes características:
#    1. No ser superior a 225 km².
#    1. Posea una mezcla de zonas urbanas, cultivos, bosque nativo y plantaciones forestales.
#    1. La zona que elija deberá ser aprobada por el profesor
# 1. Contemplar un rango temporal de 2 años, comenzando en junio
# 1. Buscar imágenes Landsat C2L2 para dicha ubicación
# 1. Tratar las imágenes de manera apropiada, convirtiendo a reflectancia. Enmascare las nubes
#    y cualquier otro tipo de clasificación que estime conveniente para sus análisis.
# 1. Generar una serie de puntos de interés en los siguientes usos:
#    1. Urbano
#    1. Agrícola
#    1. Bosque Nativo
#    1. Plantación Forestal
#    1. Otro (agua, suelo desnudo, nieve, etc)
# 1. Extraer la serie temporal de EVI y NDVI para dichos puntos
# 1. Comparar ambas series gráficamente
#
# ### Entregable
#
# Para la evaluación de este trabajo, es necesario entregar:
#
# 1. Este notebook funcional, con los resultados
# 1. Un informe de máximo 2 páginas (estilo paper), con:
#    1. Introducción (no más de 100 a 200 palabras)
#    1. Métodos:
#        1. ¿Qué imágenes, qué sensor, qué resolución?
#        1. Procesamiento realizado a la imagen, hasta obtener los datos
#           (qué máscaras aplicó, conversiones, índices, etc) y por qué lo realizó.
#    1. Resultados:
#        1. Los gráficos temporales de NDVI y EVI
#        1. Un análisis sobre la temporalidad, explicando brevemente a qué podría deberse
#           el comportamiento en el tiempo de cada cobertura en ambos índices. Si
#           hay anomalías, elucubrar sobre a qué podrían deberse.
#    1. Conclusiones:
#        1. Elegir uno de los 2 índices para su zona de estudio (centrándose en cobertura
#           agrícola, bosque nativo y plantación forestal), justificando la razón técnica
#           de su elección, considerando el que a su juicio desempeñe de mejor manera.
# 1. Una presentación de 5 minutos mostrando sus resultados. Puede ser con algunas
#    diapositivas o los resultados de este mismo notebook.
#
# ### Evaluación:
#
# 1. Notebook funcional sin errores (30%)
# 1. Informe (60%)
# 1. Presentación (10%). Debe tener una nota mínima de 4.0 en este ítem.
#
# ### Fechas
#
# - Presentación: Jueves 27 de Noviembre de 2025, en el horario de clases.
# - Entrega del notebook y del informe: Viernes 28 de Noviembre de 2025.
#
# ### Otros
#
# - Este trabajo es individual.
# - Puede utilizar IA si lo desea para los códigos (este mismo espacio tiene una), pero no
#   para el informe, el cual será sometido a revisión. Ante sospecha, se calificará con la
#   nota mínima (no se arriesgue).
#
# ### Bonus
# - Agregar otro índice a la comparación
# - Utilizar una serie lo más extendida posible (más años)
# - Agregar otra fuente de información (DEM, imagen, etc)

# %%
# @title Instalación de librerías
# %pip install rioxarray xarray matplotlib numpy rasterio geopandas planetary-computer pystac-client odc-stac plotly folium openpyxl

import folium
import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import plotly.graph_objects as go
import xarray as xr
from folium.plugins import MousePosition
from plotly.subplots import make_subplots
from pyproj import Transformer
from pystac_client import Client


# %%
# @title Funciones utilitarias
def aplicar_mascara_qa(qa_pixel, bits_a_enmascarar):
    """
    Aplica máscara basada en la banda QA_PIXEL.

    Args:
        qa_pixel: Banda QA_PIXEL de Landsat C2L2
        bits_a_enmascarar: Lista de bits a enmascarar (ej: [0, 1, 2, 3])

    Returns:
        Máscara donde True = píxel válido, False = píxel a enmascarar
    """
    mascara_valida = np.ones_like(qa_pixel, dtype=bool)

    for bit in bits_a_enmascarar:
        mascara_valida = mascara_valida & (qa_pixel & (1 << bit) == 0)

    return mascara_valida


# %% [markdown]
# ## 1. Configuración del área de estudio

# %%
# Opción 1: Cargar bbox desde un geopackage
# Deja como None si quieres usar bbox manual (Opción 2)
geopackage_path = None  # Ejemplo: "ruta/al/archivo.gpkg"

# Opción 2: Definir bbox manualmente
# MODIFICA estos valores según tu área de estudio aprobada
bbox_manual = [-70.8, -33.7, -70.6, -33.6]
epsg = "EPSG:32719"

# Obtener bbox
if geopackage_path:
    gdf = gpd.read_file(geopackage_path).to_crs(epsg)
    bbox = list(gdf.total_bounds)  # [minx, miny, maxx, maxy]
    # Convertir a formato [min_lon, min_lat, max_lon, max_lat]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
else:
    bbox = bbox_manual


CONFIG = {
    "bbox": bbox,
    "epsg": epsg,
    "fecha_inicio": "2022-06-01",
    "fecha_fin": "2022-10-31",
    "resolucion": 30,
    "max_nubes": 80,
}

# %%
# @title Visualización del área de estudio
min_lon, min_lat, max_lon, max_lat = CONFIG["bbox"]
centro_lat = (min_lat + max_lat) / 2
centro_lon = (min_lon + max_lon) / 2

transformer = Transformer.from_crs("EPSG:4326", CONFIG["epsg"], always_xy=True)

# Initialize the map with specified width and height for a smaller display
mapa = folium.Map(
    location=[centro_lat, centro_lon], zoom_start=10, tiles=None, width=800, height=400
)
MousePosition(position="bottomleft").add_to(mapa)

folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False).add_to(mapa)
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google",
    name="Google Hybrid",
    overlay=False,
).add_to(mapa)

folium.Rectangle(
    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
    color="red",
    fill=False,
    weight=2,
).add_to(mapa)

folium.LayerControl().add_to(mapa)
mapa
# %% [markdown]
# ## 2. Configuración de bandas

# %%
bandas = {
    "red": "red",
    "green": "green",
    "blue": "blue",
    "nir": "nir08",
}

# Parámetros de conversión a reflectancia
factor_reflectancia = 0.0000275
offset = -0.2
rango_valido = (0, 1)

coleccion = "landsat-c2-l2"
plataforma_filtro = ["landsat-8", "landsat-9"]

bandas_requeridas = [
    bandas["red"],
    bandas["green"],
    bandas["blue"],
    bandas["nir"],
    "qa_pixel",
]

# %% [markdown]
# ## 3. Búsqueda de imágenes

# %%
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

print(f"Imágenes encontradas: {len(items)}")
if len(items) > 0:
    fechas = [item.datetime.strftime("%Y-%m-%d") for item in items]
    print(f"Rango de fechas: {min(fechas)} a {max(fechas)}")

# %% [markdown]
# ## 4. Carga de datos

# %%
load_params = {
    "items": items,
    "bands": bandas_requeridas,
    "bbox": CONFIG["bbox"],
    "crs": CONFIG["epsg"],
    "resolution": CONFIG["resolucion"],
    "groupby": "solar_day",
}

ds_temporal = odc.stac.load(**load_params)

# %%
ds_temporal

# %% [markdown]
# ## 5. Conversión a reflectancia y enmascaramiento de nubes


# %%
bits_enmascarar = [0]  # MODIFICAR: agregue o quite según crea necesario.

# Conversión a reflectancia
ds_refl = ds_temporal[bandas_requeridas[:-1]] * factor_reflectancia + offset
rango_min, rango_max = rango_valido
condicion_rango = (ds_refl >= rango_min) & (ds_refl <= rango_max)
for banda in ds_refl.data_vars:
    ds_refl[banda] = xr.where(condicion_rango[banda], ds_refl[banda], np.nan)

# Aplicar máscara
qa_pixel = ds_temporal["qa_pixel"]
mascara = aplicar_mascara_qa(qa_pixel, bits_enmascarar)

for banda in ds_refl.data_vars:
    ds_refl[banda] = xr.where(mascara, ds_refl[banda], np.nan)

# %% [markdown]
# ## 6. Cálculo de índices

# %%
nir = ds_refl[bandas["nir"]]
red = ds_refl[bandas["red"]]
blue = ds_refl[bandas["blue"]]

denominador_ndvi = nir + red
ndvi_temporal = xr.where((nir + red) != 0, (nir - red) / denominador_ndvi, np.nan)
ndvi_temporal = xr.where(
    (ndvi_temporal >= -1) & (ndvi_temporal <= 1), ndvi_temporal, np.nan
)

denominador_evi = nir + 6 * red - 7.5 * blue + 1
evi_temporal = xr.where(
    denominador_evi != 0, 2.5 * (nir - red) / denominador_evi, np.nan
)
evi_temporal = xr.where(
    (evi_temporal >= -1) & (evi_temporal <= 1), evi_temporal, np.nan
)

# %% [markdown]
# ## 7. Definición de puntos de interés

# %%
# MODIFICA estas coordenadas según tu área de estudio (coordenadas en UTM, según el EPSG determinado en CONFIG)
puntos_interes = {
    "urbano": {
        "x": [342122],
        "y": [6279477],
        "nombre": "Zona Urbana",
    },
    "agricola": {
        "x": [339600],
        "y": [6276395],
        "nombre": "Cultivo Agrícola",
    },
    "bosque_nativo": {
        "x": [337146],
        "y": [6280137],
        "nombre": "Bosque Nativo",
    },
    "plantacion": {
        "x": [336553],
        "y": [6278686],
        "nombre": "Plantación Forestal",
    },
    "otro": {
        "x": [341545],
        "y": [6273428],
        "nombre": "Otro",
    },
}

lista_puntos = []
for tipo_uso, datos in puntos_interes.items():
    for i, (x, y) in enumerate(zip(datos["x"], datos["y"])):
        lista_puntos.append(
            {
                "tipo_uso": tipo_uso,
                "nombre": datos["nombre"],
                "x": x,
                "y": y,
            }
        )

coords_puntos = pd.DataFrame(lista_puntos)

# %%
coords_puntos

# %% [markdown]
# ## 8. Extracción de series temporales

# %%
indices_data = xr.Dataset({"NDVI": ndvi_temporal, "EVI": evi_temporal})

indices_puntos = indices_data.sel(
    x=xr.DataArray(
        coords_puntos["x"].values,
        dims="punto",
        coords={"punto": range(len(coords_puntos))},
    ),
    y=xr.DataArray(
        coords_puntos["y"].values,
        dims="punto",
        coords={"punto": range(len(coords_puntos))},
    ),
    method="nearest",
)

serie_temporal = (
    indices_puntos.to_dataframe()
    .reset_index()
    .melt(
        id_vars=["punto", "time"],
        value_vars=["NDVI", "EVI"],
        var_name="indice",
        value_name="valor",
    )
)

serie_temporal = serie_temporal.merge(
    coords_puntos.reset_index().rename(columns={"index": "punto"})[
        ["punto", "tipo_uso", "nombre"]
    ],
    on="punto",
    how="left",
)
# %%
serie_temporal

# %% [markdown]
# ## 9. Comparación gráfica

# %%
colores_uso = {
    "urbano": "red",
    "agricola": "orange",
    "bosque_nativo": "green",
    "plantacion": "yellow",
    "otro": "blue",
}

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["NDVI", "EVI"],
    shared_xaxes=True,
    vertical_spacing=0.1,
)

for j, indice in enumerate(["NDVI", "EVI"]):
    for tipo_uso in puntos_interes.keys():
        color = colores_uso.get(tipo_uso, "gray")

        datos_tipo = serie_temporal[
            (serie_temporal["tipo_uso"] == tipo_uso)
            & (serie_temporal["indice"] == indice)
        ].dropna(subset=["valor"])

        if len(datos_tipo) > 0:
            datos_agrupados = datos_tipo.groupby("time")["valor"].mean().reset_index()

            fig.add_trace(
                go.Scatter(
                    x=datos_agrupados["time"],
                    y=datos_agrupados["valor"],
                    mode="lines+markers",
                    name=tipo_uso.replace("_", " ").title(),
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    showlegend=(j == 0),
                    legendgroup=tipo_uso,
                ),
                row=1,
                col=j + 1,
            )

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
# ## 10. Estadísticas descriptivas

# %%
stats = (
    serie_temporal.groupby(["tipo_uso", "indice"])["valor"]
    .agg(["mean", "std", "min", "max", "count"])
    .round(3)
)

print(stats)

# %% [markdown]
# ## 11. Exportación de datos a Excel

# %%
serie_temporal2 = serie_temporal.copy()

datos_exportar = serie_temporal2.pivot_table(
    index=["time", "tipo_uso"],
    columns="indice",
    values="valor",
    dropna=False,  # Asegura que las columnas NDVI y EVI se mantengan, incluso si todos sus valores son NaN
).reset_index()

# Exportar a Excel
nombre_archivo = "series_temporales_indices_vegetacion.xlsx"
datos_exportar.to_excel(nombre_archivo, index=False, sheet_name="Series Temporales")

print(f"Datos exportados a: {nombre_archivo}")
print(f"Total de registros: {len(datos_exportar)}")
