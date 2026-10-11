# Datos del taller

Datos preparados antes del taller para que los cuadernos carguen rápido en Colab. Los cuadernos los leen con `utils.datos_taller.abrir_dato(nombre)`: usa esta carpeta si existe y, si no, los descarga de GitHub.

## Temperatura superficial del mar (NOAA OISST v2.1)

| Archivo | Contenido |
|---|---|
| `oisst_centro_sur.nc` | Malla diaria de 24 x 20 píxeles (0,25 grados), 34-40 S y 71-76 O, NetCDF comprimido |
| `oisst_concepcion.csv` | Serie diaria del píxel frente a Concepción (latitud -36.875, longitud -73.375), columnas `fecha` y `sst_c` |
| `preparar_oisst.py` | Script que genera ambos archivos |

- **Fuente:** NOAA Optimum Interpolation Sea Surface Temperature v2.1, leída como COG desde Microsoft Planetary Computer (colección `noaa-cdr-sea-surface-temperature-optimum-interpolation`).
- **Periodo:** 1981-09-01 a 2026-09-24.
- **Unidades:** grados Celsius. Los píxeles de tierra son `NaN`.
- **Licencia:** dominio público (datos de NOAA).
- **Vacíos conocidos:** Planetary Computer no tiene los días 2024-06-20 a 2026-03-05 (624 días) ni 2026-03-07 y 2026-03-08. No se rellenaron con otras fuentes: en el CSV aparecen como filas vacías y en la malla esas fechas no existen. Trabajar con este vacío es parte del taller.

Para actualizar los datos, ejecuta `python preparar_oisst.py` dentro de esta carpeta (tarda unos 10 a 15 minutos).

## Puntos de muestreo

| Archivo | Contenido |
|---|---|
| `puntos_concepcion.csv` | Seis puntos de agua y costa: `id`, `nombre`, `lat`, `lon` (grados decimales, EPSG:4326) y `tipo` (`bahia`, `pluma`, `golfo`, `laguna`, `humedal`) |
| `puntos_concepcion.geojson` | Los mismos puntos en GeoJSON, para abrirlos en QGIS (lo genera `preparar_series_sentinel2.py`) |

- **Fuente:** elegidos para el taller. La cobertura de cada punto se comprobó con ESA WorldCover 2021 (colección `esa-worldcover` de Microsoft Planetary Computer) y con el NDWI y el NDVI de Sentinel-2 de la serie (los puntos de agua tienen NDWI mediano de 0,4 a 0,8; el humedal, NDWI de -0,6 y NDVI de 0,55):

| Punto | Clase WorldCover | Distancia a la orilla (o al borde del humedal) |
|---|---|---|
| `bahia_concepcion` | agua (80) | 2,0 km |
| `san_vicente` | agua (80) | 1,5 km |
| `biobio_pluma` | agua (80) | 1,3 km (pluma frente a la desembocadura; su reflectancia roja mediana es el doble de la de la bahía) |
| `golfo_arauco` | agua (80) | más de 4 km |
| `laguna_grande` | agua (80) | 370 m |
| `humedal_rocuant` | humedal herbáceo (90) | 240 m |

- **Licencia:** CC0 (son coordenadas, sin derechos de autor).

## Series de Sentinel-2

| Archivo | Contenido |
|---|---|
| `series_sentinel2_concepcion.parquet` | Serie por punto y fecha, de 2025-10-01 a 2026-09-30, formato ancho (una fila por punto y fecha) |
| `preparar_series_sentinel2.py` | Script que genera este parquet y el GeoJSON, con el patrón nativo del curso (`pystac_client`, `odc.stac`) |

Columnas de `series_sentinel2_concepcion.parquet`:

| Columna | Descripción |
|---|---|
| `id`, `nombre`, `tipo`, `lat`, `lon` | El punto (de `puntos_concepcion.csv`) |
| `fecha` | Fecha y hora UTC de la toma |
| `ndwi` | NDWI de McFeeters, (B03 - B08) / (B03 + B08). Positivo sobre agua |
| `ndci` | NDCI, (B05 - B04) / (B05 + B04). Sube con la clorofila |
| `b04` | Reflectancia roja (0 a 1). Sube con la turbidez |
| `ndvi` | NDVI, (B08 - B04) / (B08 + B04) |
| `nubes_escena` | Porcentaje de nubes de la escena completa (`eo:cloud_cover` del catálogo), no de la ventana |
| `pixeles_validos` | Fracción (0 a 1) de la ventana que pasó la máscara de calidad SCL |
| `n_pixeles` | Píxeles de la ventana (círculo de 150 m de radio, píxeles de 10 m) |
| `offset_reflectancia` | Offset aplicado a esa fecha (0 o -0,1): lo decide `utils.stac.agregar_offset` según el catálogo y el baseline de procesamiento |

- **Fuente:** ESA Copernicus Sentinel-2 L2A, leído como COG desde Microsoft Planetary Computer (colección `sentinel-2-l2a`).
- **Método:** para cada punto se carga una ventana de unos 300 x 300 m; se enmascaran con la banda SCL las nubes, sombras, cirros, píxeles saturados y sin dato; se convierte a reflectancia con el offset de cada fecha; y se resume con la mediana de los píxeles a 150 m o menos del punto. Si menos de la mitad de la ventana quedó libre de nubes, los índices de esa fecha son `NaN` (la fecha se conserva).
- **Filtro de escena:** se buscó con hasta 60 % de nubes en la escena; la máscara SCL hace el resto del trabajo.
- **Escena descartada:** `S2B_MSIL2A_20260407T143739_R096_T18HXD_20260407T180947` (2026-04-07) está en Planetary Computer con archivos sin sistema de coordenadas: GDAL no puede leerlos y hace fallar toda la carga. El script la excluye (`ITEMS_DEFECTUOSOS`).
- **Metadatos:** el parquet lleva título, fuente, licencia, parámetros y descripción de columnas dentro del archivo: `utils.series.leer_metadatos_series()`.
- **Licencia:** datos Copernicus Sentinel de acceso libre y abierto (Reglamento UE 1159/2013). Cita: "Contiene datos Copernicus Sentinel modificados (2025-2026)".

Para regenerarlo, ejecuta `python preparar_series_sentinel2.py` dentro de esta carpeta (necesita internet y tarda unos minutos).
