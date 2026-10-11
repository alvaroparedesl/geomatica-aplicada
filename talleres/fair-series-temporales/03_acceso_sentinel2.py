# %% [markdown]
# # Bloque 3. Acceso y visualización de Sentinel-2
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/03_acceso_sentinel2.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/)
#
# Taller *Datos FAIR, series temporales satelitales y detección de anomalías con Python*. Día 1, tarde (unos 75 minutos).
#
# En el bloque 2 buscamos imágenes con metadatos. Ahora **leemos los píxeles**: cargamos una ventana de una escena Sentinel-2 sobre Concepción y Talcahuano sin descargar la escena completa, la miramos en color verdadero y calculamos índices del agua y de la costa.
#
# **Qué vas a hacer**
#
# | Parte | Tema | Minutos |
# |---|---|---|
# | 1 | Qué es un COG y por qué se puede leer solo una parte | 15 |
# | 2 | Cargar una ventana con el patrón nativo | 10 |
# | 3 | Imagen RGB (color verdadero) | 10 |
# | 4 | Índices: NDWI, NDCI, turbidez y NDVI | 25 |
# | 5 | El valor de un píxel en un punto | 10 |
# | 6 | Resolución espacial: 10, 20 y 60 m | 5 |
#
# Este bloque usa **solo Sentinel-2**.

# %% [markdown]
# ## Preparar el entorno

# %%
# @title Preparar entorno (ejecutar primero)
import subprocess
import sys
from pathlib import Path

if "google.colab" in sys.modules:  # Colab: clonar el repositorio (solo la primera vez)
    RAIZ = Path("/content/geomatica-aplicada")
    if not RAIZ.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/alvaroparedesl/geomatica-aplicada", str(RAIZ)], check=True)
else:  # Local: subir desde la carpeta actual hasta encontrar utils/
    RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "utils").is_dir())
sys.path.insert(0, str(RAIZ))

# %%
# @title Instalar los paquetes de este bloque (solo si faltan)
import importlib.util

# Paquete de requirements.txt -> nombre con el que se importa
NECESARIOS = {"pystac-client": "pystac_client", "planetary-computer": "planetary_computer",
              "odc-stac": "odc.stac", "rioxarray": "rioxarray"}
requisitos = [linea.strip() for linea in (RAIZ / "talleres/fair-series-temporales/requirements.txt").read_text().splitlines()
              if linea.strip() and not linea.startswith("#")]
requisitos = [r for r in requisitos if r.split()[0] in NECESARIOS]

if any(importlib.util.find_spec(modulo) is None for modulo in NECESARIOS.values()):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *requisitos], check=True)
    print("Instalado:", ", ".join(requisitos))
else:
    print("Ya está todo instalado:", ", ".join(NECESARIOS))

# %%
# @title Importar bibliotecas
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

# Funciones de apoyo del curso (carpeta utils/)
from utils.raster import calcular_indices, convertir_a_reflectancia, lonlat_a_crs
from utils.series import bbox_punto, cargar_puntos_csv
from utils.stac import (agregar_offset, conectar_catalogo, parametros_busqueda, parametros_carga, preparar_items,
                        renombrar_bandas, resumir_imagenes)
from utils.viz import crear_imagen_rgb, mostrar_imagen

warnings.filterwarnings("ignore", message="Dataset has no geotransform")
pd.set_option("display.width", 200)

# %% [markdown]
# ## 1. Qué es un COG
#
# Una escena Sentinel-2 completa pesa cientos de megabytes por banda. Si para ver la bahía de Concepción tuviéramos que descargar todo, trabajar con imágenes de la nube sería inviable. Lo hace posible el formato **COG** (*Cloud Optimized GeoTIFF*): un GeoTIFF ordenado de una manera especial.
#
# * **Teselas (*tiles*).** La imagen se guarda en bloques (por ejemplo, de 512 x 512 píxeles) y cada bloque se comprime por separado. Para leer una zona basta con los bloques que la tocan.
# * **Pirámides (*overviews*).** El archivo trae copias reducidas a la mitad, a un cuarto, a un octavo... Para ver la escena entera en miniatura se lee una pirámide pequeña, no los 120 millones de píxeles.
# * **Lecturas parciales por HTTP (*range requests*).** El servidor entrega solo un rango de bytes del archivo. El encabezado dice dónde está cada bloque, así que el programa pide solo esos bytes.
#
# Veámoslo con un archivo real. Buscamos la escena que usaremos todo el bloque: **10 de agosto de 2026**, casi sin nubes, sobre la zona de Concepción, Talcahuano y la desembocadura del Biobío.

# %%
# Ventana de trabajo: Concepción, Talcahuano y la desembocadura del Biobío (grados: oeste, sur, este, norte)
bbox = [-73.22, -36.89, -72.98, -36.66]

catalogo = conectar_catalogo()  # Client.open(...) de pystac_client, con la firma de Planetary Computer
parametros = parametros_busqueda("sentinel-2", bbox, ("2026-08-01", "2026-08-31"), nubes_max=20, catalogo=catalogo)
items = preparar_items(catalogo.search(**parametros).item_collection())
resumir_imagenes(items)

# %%
# La escena casi sin nubes: 10 de agosto de 2026
items = [item for item in items if item.datetime.strftime("%Y-%m-%d") == "2026-08-10"]
item = items[0]
print(item.id, "| nubes:", round(item.properties["eo:cloud_cover"], 3), "%")

# %% [markdown]
# `rasterio.open` abre el archivo de la banda roja (`B04`) por internet. Abrir es leer solo el encabezado: no se descarga la imagen.

# %%
with rasterio.open(item.assets["B04"].href) as archivo:
    print("Tamaño completo:", archivo.width, "x", archivo.height, "píxeles de", archivo.res[0], "m")
    print("Tipo de dato:", archivo.dtypes[0], "| CRS:", archivo.crs.to_string())
    print("Organización:", archivo.tags(ns="IMAGE_STRUCTURE"))
    print("Bloques de lectura:", archivo.block_shapes[0], "(teselas)")
    print("Pirámides (factores de reducción):", archivo.overviews(1))
    print(f"Sin comprimir, la banda ocuparía {archivo.width * archivo.height * 2 / 1e6:.0f} MB")

    # a) Toda la escena en miniatura: rasterio elige la pirámide adecuada
    t0 = time.time()
    miniatura = archivo.read(1, out_shape=(archivo.height // 10, archivo.width // 10))
    print(f"\nEscena completa en miniatura {miniatura.shape}: {time.time() - t0:.1f} s")

    # b) Solo una ventana a resolución completa
    limites = transform_bounds("EPSG:4326", archivo.crs, *[-73.12, -36.84, -73.08, -36.80])
    ventana = from_bounds(*limites, transform=archivo.transform)
    t0 = time.time()
    recorte = archivo.read(1, window=ventana)
    print(f"Ventana de {recorte.shape} píxeles a 10 m: {time.time() - t0:.1f} s "
          f"({recorte.size / (archivo.width * archivo.height) * 100:.2f} % de la escena)")

# %% [markdown]
# La miniatura de toda la escena y la ventana de 4 x 4 km a resolución completa se leen en un par de segundos, y ninguna descargó la banda entera. Eso es un COG. Lo que sigue (`odc.stac`, `rioxarray`) usa esta misma idea por debajo: cuando le pides una ventana, solo lee los bloques que la tocan.
#
# **Tu turno (3 min).** Abre la banda `B08` (infrarrojo cercano) de la misma escena y la `SCL`. ¿Qué tamaño de píxel tiene cada una? (Pista: `archivo.res`.) ¿Y el factor de la primera pirámide?

# %%
# Tu turno
for banda in ["B08", "SCL"]:
    with rasterio.open(item.assets[banda].href) as archivo:
        print(banda, "->", archivo.width, "x", archivo.height, "píxeles de", archivo.res[0], "m | pirámides:", archivo.overviews(1))

# %% [markdown]
# ## 2. Cargar una ventana con el patrón nativo
#
# Ahora cargamos los píxeles de la ventana como un arreglo de `xarray`. Es el patrón nativo del curso, con las llamadas a `odc.stac` escritas por ti:
#
# 1. `parametros_carga(...)` arma el diccionario de argumentos de `odc.stac.load`: nombres de banda, CRS, resolución, firma de los enlaces y un `groupby` que nunca mezcla fechas con offsets distintos. **No carga nada.**
# 2. `odc.stac.load(items, **parametros)` es la llamada nativa que lee.
# 3. `agregar_offset(datos, items)` guarda en la coordenada `offset_reflectancia` el offset de cada fecha (recuerda el bloque 2: depende del catálogo y del baseline).
# 4. `convertir_a_reflectancia(...)` usa esa coordenada para pasar de números digitales (DN) a reflectancia.
#
# Pedimos cinco bandas y la clasificación de la escena (`SCL`):
#
# | Banda | Qué es | Resolución nativa |
# |---|---|---|
# | `B02` | Azul | 10 m |
# | `B03` | Verde | 10 m |
# | `B04` | Rojo | 10 m |
# | `B05` | Borde rojo | 20 m |
# | `B08` | Infrarrojo cercano (NIR) | 10 m |
# | `SCL` | Clasificación de la escena | 20 m |
#
# Las bandas de 20 m se remuestrean a 10 m al cargar (más en la parte 6).

# %%
bandas = ["B02", "B03", "B04", "B05", "B08", "SCL"]
CRS = "EPSG:32718"  # UTM zona 18 sur, la de Concepción

parametros_ds = parametros_carga("sentinel-2", bandas, resolucion=10, crs=CRS, catalogo=catalogo,
                                 bbox=bbox, chunks={"x": 1024, "y": 1024})
print("Argumentos de odc.stac.load:", {k: v for k, v in parametros_ds.items() if k not in ("groupby", "patch_url")})

t0 = time.time()
datos = odc.stac.load(items, **parametros_ds)  # perezosa (dask): aún no lee píxeles
datos = renombrar_bandas(datos, catalogo)      # B02, B04, ... en cualquier catálogo
datos = agregar_offset(datos, items)           # offset de cada fecha
print(datos)

# %% [markdown]
# `datos` describe qué vamos a leer (2.595 x 2.193 píxeles de 10 m) pero todavía no leyó nada. `.compute()` lee los bloques necesarios. Seleccionamos la única fecha con `isel(time=0)` (el offset se conserva al elegir una fecha).

# %%
t0 = time.time()
escena = datos.isel(time=0).compute()
print(f"Lectura: {time.time() - t0:.0f} s | memoria: {sum(v.nbytes for v in escena.data_vars.values()) / 1e6:.0f} MB")
print("Offset de reflectancia de esta fecha:", float(escena.offset_reflectancia))

# %% [markdown]
# Los valores son **DN** (por ejemplo, 1.185), no reflectancias. Convertimos con `convertir_a_reflectancia`, que usa el offset de la fecha. Dejamos fuera `SCL`, que es una clase y no una reflectancia.

# %%
refl = convertir_a_reflectancia(escena[["B02", "B03", "B04", "B05", "B08"]], "sentinel-2")
scl = escena["SCL"]
print({b: f"{float(refl[b].min()):.4f} a {float(refl[b].max()):.2f}" for b in refl.data_vars})

# %% [markdown]
# ## 3. Imagen RGB (color verdadero)
#
# `crear_imagen_rgb` apila tres bandas (rojo, verde y azul) en un arreglo (alto, ancho, 3), multiplica por un factor de brillo y recorta entre 0 y 1. Primero la imagen tal cual la deja la conversión.

# %%
rgb = crear_imagen_rgb(refl, ["B04", "B03", "B02"], factor_brillo=3.5)
mostrar_imagen(rgb, "Concepción y Talcahuano, 10 de agosto de 2026 (color verdadero)", figsize=(9, 10))

# %% [markdown]
# Se ve la bahía de Concepción (arriba), la península de Hualpén, el Biobío bajando hacia el mar (abajo) y su **pluma de sedimentos** (agua más pardusca) frente a la desembocadura. Pero hay **huecos blancos**, sobre todo en los cerros con bosque (al este y suroeste). ¿Qué son?
#
# `convertir_a_reflectancia` pone en `NaN` los valores **fuera de 0 a 1**. En píxeles muy oscuros (sombras, bosque denso, agua limpia), la corrección atmosférica puede dar reflectancias levemente **negativas**, y esos píxeles quedan sin valor. Lo comprobamos:

# %%
for banda in ["B02", "B03", "B04", "B05", "B08"]:
    dn = escena[banda]
    negativos = float(((dn * 0.0001 + float(escena.offset_reflectancia)) < 0).mean()) * 100
    print(f"{banda}: {negativos:.1f} % de los píxeles tiene reflectancia negativa y queda en NaN")

# %% [markdown]
# Entre 1 % y 8 % de los píxeles según la banda (más en azul, NIR y borde rojo). No es un error: es lo que pasa en superficies casi negras. Para **mirar** la imagen los pintamos de negro (`fillna(0)`); para **calcular** índices los dejamos como `NaN`, que es lo honesto: ahí no hay una medida válida.

# %%
rgb = crear_imagen_rgb(refl.fillna(0), ["B04", "B03", "B02"], factor_brillo=3.5)
mostrar_imagen(rgb, "Color verdadero (NaN pintado de negro)", figsize=(9, 10))

# %% [markdown]
# **Tu turno (4 min).** Prueba otras combinaciones:
#
# * **Falso color** (NIR, rojo, verde): `["B08", "B04", "B03"]`. La vegetación se ve roja y el agua casi negra. ¿Qué ves en el humedal Rocuant-Andalién (al noreste de Talcahuano)?
# * Cambia `factor_brillo` (2.5, 4.5). ¿Cuándo se satura la imagen?

# %%
# Tu turno: falso color
falso_color = crear_imagen_rgb(refl.fillna(0), ["B08", "B04", "B03"], factor_brillo=3.0)
mostrar_imagen(falso_color, "Falso color: NIR, rojo, verde", figsize=(9, 10))

# %% [markdown]
# ## 4. Índices del agua y de la costa
#
# Un índice espectral combina bandas para destacar una propiedad. Casi todos son **diferencias normalizadas**, `(A - B) / (A + B)`, con valores entre -1 y 1. En este taller usamos cuatro (las funciones están en `utils/raster.py`):
#
# | Variable | Fórmula | Qué muestra |
# |---|---|---|
# | **NDWI** (McFeeters) | `(B03 - B08) / (B03 + B08)` | Agua abierta: positivo sobre el agua (el verde refleja más que el NIR), negativo en tierra |
# | **NDCI** | `(B05 - B04) / (B05 + B04)` | Clorofila en el agua: sube con el fitoplancton |
# | **Reflectancia roja** (`B04`) | la banda directamente | Turbidez: el agua con sedimentos refleja mucho más en el rojo |
# | **NDVI** | `(B08 - B04) / (B08 + B04)` | Vegetación: alto sobre plantas sanas (humedales, bosques, pastos) |
#
# `calcular_indices` calcula varios a la vez y los entrega en un `Dataset`. Usamos `NDWI_AGUA` (el NDWI de McFeeters; el `NDWI` a secas del curso usa NIR y SWIR y mide humedad de la vegetación).

# %%
indices = calcular_indices(refl, "sentinel-2", ["NDWI_AGUA", "NDCI", "NDVI"])
print({k: f"mediana {float(v.median()):.2f}" for k, v in indices.items()})

# %% [markdown]
# ### NDWI: separar agua de tierra
#
# Con el NDWI positivo definimos una **máscara de agua**. Las otras variables solo tienen sentido en su medio: el NDCI y la turbidez, en el agua; el NDVI, en la tierra.

# %%
agua = indices["NDWI_AGUA"] > 0
print(f"{float(agua.mean()) * 100:.0f} % de la ventana es agua según NDWI > 0")

fig, ejes = plt.subplots(1, 2, figsize=(13, 7))
im = ejes[0].imshow(indices["NDWI_AGUA"], cmap="RdBu", vmin=-1, vmax=1)
ejes[0].set_title("NDWI (McFeeters)")
fig.colorbar(im, ax=ejes[0], fraction=0.046)
ejes[1].imshow(agua, cmap="Blues")
ejes[1].set_title("Máscara de agua (NDWI > 0)")
for eje in ejes:
    eje.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Turbidez: la pluma del Biobío frente a la bahía
#
# La reflectancia roja sube cuando el agua lleva sedimentos. Mostramos `B04` **solo sobre el agua**, con una escala que resalte las diferencias (el agua limpia tiene valores de 0.001 a 0.01).

# %%
rojo_agua = refl["B04"].where(agua)

fig, ax = plt.subplots(figsize=(9, 10))
im = ax.imshow(rojo_agua, cmap="YlOrBr", vmin=0, vmax=0.06)
ax.set_title("Reflectancia roja (B04) sobre el agua: turbidez")
ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.04, label="reflectancia")
plt.tight_layout()
plt.show()

# %% [markdown]
# La **pluma del Biobío** (abajo a la izquierda, frente a la desembocadura) y la desembocadura del Andalién (arriba a la derecha, en la bahía de Concepción) destacan frente al agua oscura de las bahías. Es el contraste que seguiremos en el tiempo en el bloque 4: el punto `biobio_pluma` frente a `bahia_concepcion`.
#
# ### NDCI: clorofila
#
# El NDCI usa el borde rojo (`B05`, 705 nm), donde la clorofila deja una huella. Sube cuando hay más fitoplancton, pero **también puede subir con la turbidez**, así que se interpreta junto a la reflectancia roja. Fíjate en la Laguna Grande de San Pedro (abajo, al centro): su NDCI es mucho mayor que el del mar abierto, lo esperable en aguas someras y ricas en nutrientes.

# %%
ndci_agua = indices["NDCI"].where(agua)

fig, ax = plt.subplots(figsize=(9, 10))
im = ax.imshow(ndci_agua, cmap="YlGn", vmin=-0.2, vmax=0.4)
ax.set_title("NDCI sobre el agua (clorofila)")
ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.04, label="NDCI")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### NDVI: la vegetación de la costa
#
# Ahora al revés: el NDVI **sobre la tierra** (donde el NDWI es negativo). Los humedales, pastizales y bosques destacan.

# %%
ndvi_tierra = indices["NDVI"].where(~agua)

fig, ax = plt.subplots(figsize=(9, 10))
im = ax.imshow(ndvi_tierra, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
ax.set_title("NDVI sobre la tierra")
ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.04, label="NDVI")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Tu turno (8 min).**
#
# 1. Dibuja un histograma del NDWI de toda la ventana. ¿Ves dos grupos (agua y tierra)? ¿Qué valor los separa mejor: 0 u otro?
# 2. Cambia `vmax` de la reflectancia roja a 0.02. ¿Qué parte de la bahía se vuelve visible?
# 3. Calcula el NDVI **mediano** de la tierra y el NDCI **mediano** del agua.

# %%
# Tu turno
valores = indices["NDWI_AGUA"].values.ravel()
valores = valores[~np.isnan(valores)]
plt.figure(figsize=(7, 3.5))
plt.hist(valores, bins=100, color="steelblue")
plt.axvline(0, color="red", ls="--", label="NDWI = 0")
plt.xlabel("NDWI (McFeeters)")
plt.ylabel("Píxeles")
plt.legend()
plt.show()

print(f"NDVI mediano en tierra: {float(ndvi_tierra.median()):.2f} | NDCI mediano en agua: {float(ndci_agua.median()):.2f}")

# %% [markdown]
# ## 5. El valor de un píxel en un punto
#
# El primer paso hacia una serie temporal es **leer el valor en un punto**. Los puntos del taller están en grados (lat, lon) y los datos en metros UTM: `lonlat_a_crs` transforma las coordenadas, y `.sel(x=..., y=..., method="nearest")` elige el píxel más cercano.

# %%
puntos = cargar_puntos_csv().set_index("id")
indices_refl = indices.assign(B04=refl["B04"])

filas = []
for id_punto, punto in puntos.iterrows():
    x, y = lonlat_a_crs(punto.lon, punto.lat, CRS)
    pixel = indices_refl.sel(x=x, y=y, method="nearest")
    filas.append({"punto": id_punto, "tipo": punto.tipo,
                  "B04": float(pixel["B04"]), "NDWI": float(pixel["NDWI_AGUA"]),
                  "NDCI": float(pixel["NDCI"]), "NDVI": float(pixel["NDVI"]),
                  "SCL": int(scl.sel(x=x, y=y, method="nearest"))})
pd.DataFrame(filas).round(3)

# %% [markdown]
# Los valores cuentan la historia del punto: el NDWI es positivo en los puntos de agua y negativo en el humedal, y el NDVI es alto solo en el humedal. La reflectancia roja de `biobio_pluma` es mayor que la de `bahia_concepcion`. La columna `SCL` es la clase de la escena (6 = agua, 4 = vegetación, 5 = suelo desnudo).
#
# Un píxel aislado puede ser **ruidoso** (una ola, un barco, un píxel mixto). Por eso en el bloque 4 no leeremos un píxel sino la **mediana de una ventana** de 150 m alrededor de cada punto.
#
# Comparamos el píxel con la mediana de una ventana circular de 150 m en la pluma del Biobío (agua bastante homogénea, así que casi coinciden). En el siguiente "Tu turno" lo repites en un humedal, donde no coinciden:

# %%
punto = puntos.loc["biobio_pluma"]
x, y = lonlat_a_crs(punto.lon, punto.lat, CRS)
distancia = np.hypot(refl["B04"].x - x, refl["B04"].y - y)
ventana_150 = refl["B04"].where(distancia <= 150)
print(f"Píxel central: {float(refl['B04'].sel(x=x, y=y, method='nearest')):.4f} | "
      f"mediana de la ventana de 150 m: {float(ventana_150.median()):.4f} | "
      f"píxeles en la ventana: {int(ventana_150.notnull().sum())}")

# %% [markdown]
# **Tu turno (3 min).** Elige otro punto (`laguna_grande`, `humedal_rocuant`...) y compara su píxel central con la mediana de su ventana de 150 m. ¿Cuál varía más, el agua o el humedal?

# %%
# Tu turno: cambia el id del punto
id_punto = "humedal_rocuant"
punto = puntos.loc[id_punto]
x, y = lonlat_a_crs(punto.lon, punto.lat, CRS)
distancia = np.hypot(refl["B08"].x - x, refl["B08"].y - y)
for banda in ["B04", "B08"]:
    ventana_punto = refl[banda].where(distancia <= 150)
    print(f"{id_punto} {banda}: píxel {float(refl[banda].sel(x=x, y=y, method='nearest')):.4f} | "
          f"mediana de la ventana {float(ventana_punto.median()):.4f} | desviación {float(ventana_punto.std()):.4f}")

# %% [markdown]
# ## 6. Resolución espacial: 10, 20 y 60 m
#
# Sentinel-2 mide sus bandas con **distinto tamaño de píxel**:
#
# * **10 m:** `B02` (azul), `B03` (verde), `B04` (rojo) y `B08` (NIR).
# * **20 m:** `B05`, `B06`, `B07`, `B8A` (borde rojo) y `B11`, `B12` (infrarrojo de onda corta), además de `SCL`.
# * **60 m:** `B01` (aerosoles costeros), `B09` y `B10`.
#
# Un índice que combina bandas de resoluciones distintas (el NDCI usa `B05` de 20 m y `B04` de 10 m) tiene que remuestrear. `odc.stac.load` lo hace por ti cuando pides una resolución común, y por defecto repite el valor más cercano (*nearest*), lo que da bloques.
#
# Cargamos una ventana de 3 x 3 km alrededor de la Laguna Grande de San Pedro, cada banda a su **resolución nativa**, y la miramos lado a lado.

# %%
punto = puntos.loc["laguna_grande"]  # laguna, orilla del Biobío y zona urbana: hay detalles a distintas escalas
caja = bbox_punto(float(punto.lon), float(punto.lat), 1500)

nativas = {"B02": 10, "B05": 20, "B01": 60}
paneles = {}
for banda, resolucion in nativas.items():
    cargada = odc.stac.load(items, **parametros_carga("sentinel-2", [banda], resolucion=resolucion, crs=CRS,
                                                      catalogo=catalogo, bbox=caja))
    paneles[banda] = (cargada.isel(time=0)[banda].compute(), resolucion)
    print(f"{banda}: píxel de {resolucion} m, ventana de {cargada.sizes['y']} x {cargada.sizes['x']} = {cargada.sizes['y'] * cargada.sizes['x']} píxeles")

# %%
fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
for eje, (banda, (imagen, resolucion)) in zip(ejes, paneles.items()):
    eje.imshow(imagen, cmap="viridis", interpolation="nearest", vmin=np.nanpercentile(imagen, 2), vmax=np.nanpercentile(imagen, 98))
    eje.set_title(f"{banda}: píxeles de {resolucion} m")
    eje.axis("off")
plt.suptitle("La misma zona (3 x 3 km) a 10, 20 y 60 m", y=1.03)
plt.tight_layout()
plt.show()

# %% [markdown]
# A 60 m la laguna pierde su forma, los caminos y las orillas desaparecen y todo se vuelve escalonado: **cada píxel de 60 m cubre 36 píxeles de 10 m**. Eso importa al elegir qué medir:
#
# * Para **agua abierta y bahías** (cientos de metros de ancho), 20 o incluso 60 m pueden bastar.
# * Para **ríos angostos, humedales y bordes de costa**, necesitas 10 m, y aun así hay **píxeles mixtos** (mitad agua, mitad tierra).
# * Una ventana de 150 m de radio alrededor de un punto contiene unos 700 píxeles de 10 m, pero solo unos 180 de 20 m y unos 20 de 60 m.
#
# **Tu turno (3 min).** Cambia `nativas` para comparar `B03` a 10 m con `B03` remuestreada a 20 y a 60 m (pide `resolucion=20` y `resolucion=60` para la misma banda). ¿Cuándo empieza a notarse la pérdida de detalle en la orilla de la laguna?

# %%
# Tu turno: la misma banda remuestreada a distintas resoluciones
fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
for eje, resolucion in zip(ejes, [10, 20, 60]):
    cargada = odc.stac.load(items, **parametros_carga("sentinel-2", ["B03"], resolucion=resolucion, crs=CRS,
                                                      catalogo=catalogo, bbox=caja))
    imagen = cargada.isel(time=0)["B03"].compute()
    eje.imshow(imagen, cmap="gray", interpolation="nearest", vmin=np.nanpercentile(imagen, 2), vmax=np.nanpercentile(imagen, 98))
    eje.set_title(f"B03 a {resolucion} m")
    eje.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Resumen
#
# * Un **COG** guarda la imagen en teselas con pirámides: se puede leer una ventana o una miniatura sin descargar la escena entera.
# * El patrón de carga es nativo y explícito: `parametros_carga` arma los argumentos, `odc.stac.load` lee, `agregar_offset` anota el offset de cada fecha y `convertir_a_reflectancia` lo usa. **Convierte a reflectancia antes de combinar fechas.**
# * Los píxeles muy oscuros pueden tener reflectancia negativa y quedan en `NaN`: se pintan de negro para mirar, pero no se inventan para calcular.
# * **NDWI** separa agua de tierra; **NDCI** apunta a la clorofila; la **reflectancia roja** a la turbidez (la pluma del Biobío frente a la bahía); **NDVI** a la vegetación.
# * Un píxel suelto es ruidoso: en el bloque 4 usaremos la mediana de una ventana alrededor de cada punto.
# * Sentinel-2 tiene píxeles de **10, 20 y 60 m** según la banda: elige la resolución según el tamaño de lo que quieres ver.
#
# Siguiente bloque: **la serie temporal** de esos índices en cada punto durante doce meses (`04_series_temporales`).
