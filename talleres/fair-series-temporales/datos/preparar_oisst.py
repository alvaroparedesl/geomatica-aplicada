"""Prepara los datos OISST del taller desde Microsoft Planetary Computer.

Descarga la temperatura superficial del mar diaria (NOAA OISST v2.1) para la
zona centro sur de Chile (34-40 S, 71-76 O) desde 1981-09-01 hasta el último
día publicado, y guarda:

* oisst_centro_sur.nc: malla de 24 x 20 píxeles de 0,25 grados.
* oisst_concepcion.csv: serie del píxel frente a Concepción (-36.875, -73.375).

Solo se usa Planetary Computer (COG). Los días que el catálogo no tiene NO se
rellenan con otras fuentes: quedan como vacío en la serie a propósito, para
trabajarlos en el taller. Al ejecutarlo se imprime la lista de vacíos.

Uso (tarda unos 10 a 15 minutos):
    python preparar_oisst.py
"""

import concurrent.futures as cf
import time

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
import xarray as xr
from rasterio.windows import from_bounds

COLECCION = "noaa-cdr-sea-surface-temperature-optimum-interpolation"
LIMITES = (-76, -40, -71, -34)  # oeste, sur, este, norte
PUNTO = (-36.875, -73.375)  # latitud, longitud frente a Concepción

inicio = time.time()
catalogo = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)
items = list(
    catalogo.search(collections=[COLECCION], datetime="1981-09-01/2100-01-01").items()
)
print(f"{len(items)} días en el catálogo ({time.time() - inicio:.0f} s)")

# Todas las imágenes comparten la misma malla global, así que la ventana
# de lectura se calcula una sola vez.
with rasterio.open(items[0].assets["sst"].href) as src:
    ventana = from_bounds(*LIMITES, src.transform).round_offsets().round_lengths()
    transformacion = src.window_transform(ventana)
lons = np.array([transformacion.c + transformacion.a * (j + 0.5) for j in range(int(ventana.width))])
lats = np.array([transformacion.f + transformacion.e * (i + 0.5) for i in range(int(ventana.height))])


def leer_dia(item):
    """Lee la ventana de un día. Reintenta y vuelve a firmar si falla."""
    fecha = pd.Timestamp(item.properties["start_datetime"]).tz_localize(None).normalize()
    for intento in range(4):
        try:
            href = item.assets["sst"].href if intento == 0 else pc.sign(item.assets["sst"].href)
            with rasterio.open(href) as src:
                valores = src.read(1, window=ventana)
            # -999 es el valor sin dato (tierra); la escala es 0,01 grados
            return fecha, np.where(valores == -999, np.nan, valores * 0.01).astype("float32")
        except Exception:
            time.sleep(2 * (intento + 1))
    return fecha, None


with cf.ThreadPoolExecutor(32) as ejecutor:
    resultados = list(ejecutor.map(leer_dia, items))

fallidos = [fecha for fecha, valores in resultados if valores is None]
print(f"Lecturas fallidas: {len(fallidos)}")
resultados = sorted((r for r in resultados if r[1] is not None), key=lambda r: r[0])
fechas = pd.DatetimeIndex([fecha for fecha, _ in resultados])

ds = xr.Dataset(
    {"sst": (("time", "lat", "lon"), np.stack([valores for _, valores in resultados]))},
    coords={"time": fechas, "lat": lats, "lon": lons},
).sortby("lat")
ds.sst.attrs.update(units="degC", long_name="Temperatura superficial del mar (OISST v2.1)")
ds.attrs.update(
    source=f"NOAA OISST v2.1, Microsoft Planetary Computer (COG), colección {COLECCION}",
    region="34-40S, 71-76W",
    note="Sin relleno: los días ausentes en Planetary Computer quedan como vacío en el eje temporal",
)
ds.to_netcdf(
    "oisst_centro_sur.nc",
    encoding={"sst": {"zlib": True, "complevel": 9, "dtype": "int16",
                      "scale_factor": 0.01, "_FillValue": -32768}},
)

# Vacíos: días que faltan entre la primera y la última fecha
todos = pd.date_range(fechas[0], fechas[-1])
faltantes = pd.Series(todos.difference(fechas))
print(f"{len(fechas)} días, de {fechas[0].date()} a {fechas[-1].date()}, faltan {len(faltantes)}")
grupos = (faltantes.diff() != pd.Timedelta("1D")).cumsum()
for _, vacio in faltantes.groupby(grupos):
    print(f"  vacío: {vacio.iloc[0].date()} a {vacio.iloc[-1].date()} ({len(vacio)} días)")

serie = ds.sst.sel(lat=PUNTO[0], lon=PUNTO[1], method="nearest").to_series().reindex(todos)
serie.rename_axis("fecha").rename("sst_c").to_csv("oisst_concepcion.csv")
print(f"Listo en {time.time() - inicio:.0f} s")
