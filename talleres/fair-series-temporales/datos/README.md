# Datos del taller

Datos preparados antes del taller para que los cuadernos carguen rápido en Colab. Los cuadernos los leen con `utils.datos_taller.abrir_dato(nombre)`: usa esta carpeta si existe y, si no, los descarga de GitHub.

## Temperatura de superficie y reflectancia de Landsat 8 y 9

| Archivo | Contenido |
|---|---|
| `series_landsat_concepcion.parquet` | Serie por punto y fecha (marzo de 2013 a octubre de 2026), formato ancho: temperatura de superficie en °C, reflectancia, NDWI y NDVI |
| `landsat_malla_concepcion.nc` | Temperatura de superficie (°C) cada 500 m en un cuadrado de 0,8 grados alrededor de la bahía de Concepción y el golfo de Arauco, con todas las fechas con al menos 10 % de píxeles válidos (UTM 18 S, EPSG:32718) |
| `preparar_landsat.py` | Script que genera ambos archivos (`python preparar_landsat.py malla` regenera solo la malla) |

- **Fuente:** USGS Landsat 8 y 9 Collection 2 Level-2, leído como COG desde Microsoft Planetary Computer (colección `landsat-c2-l2`).
- **Calibración:** los archivos del catálogo guardan números enteros (DN). El script los convierte con `temperatura (K) = DN * 0.00341802 + 149.0` (banda `lwir11`, luego se resta 273,15 para °C) y `reflectancia = DN * 0.0000275 - 0.2` (verde, rojo, NIR y SWIR1). DN 0 es "sin dato" y las temperaturas fuera de 230 a 340 K son error. Las funciones son `utils.raster.convertir_a_temperatura` y `convertir_a_reflectancia`.
- **Malla (`landsat_malla_concepcion.nc`):** 495 fechas (28 de marzo de 2013 a 8 de octubre de 2026) de 181 x 146 píxeles de 500 m (UTM 18 S, de 624 a 697 km al este y de 5864 a 5954 km al norte), guardada como entero (centésimas de °C) comprimido: 8,4 MB. Es una **muestra de un píxel cada 500 m** (un píxel de 30 m por cada celda de 500 m, sin promediar), con NaN donde hay nubes. Solo guarda la banda térmica, así que **no tiene el control de calidad con la reflectancia** de la tabla por punto: hay valores imposibles (hasta -43 °C, el piso de la calibración: el tope frío de una nube sin marcar; un 4,6 % está bajo 3 °C), y el cuaderno del bloque 7 los descarta con una regla simple. Una fecha entra solo si al menos el 10 % de sus píxeles quedó libre de nubes.
- **Máscara:** se descartan los píxeles con relleno, nube, nube dilatada, cirrus, sombra de nube (`qa_pixel`) y saturación (`qa_radsat`).
- **Resolución:** la banda térmica tiene 100 m nativos y el USGS la entrega a 30 m. Sobre el agua el producto mide la temperatura de la **piel** de la superficie (no la del agua en profundidad), con errores de alrededor de 1 K o más que crecen con la humedad atmosférica y las nubes delgadas; no es un producto oceanográfico.
- **Licencia:** dominio público (datos del USGS).
- **Escenas descartadas:** `LC08_L2SP_001086_20140305_02_T1` (banda térmica `lwir11`) y `LC08_L2SP_001086_20221106_02_T1` (bandas verde y roja) tienen en Planetary Computer un archivo dañado: el servidor devuelve una página de error de unos 14 kB en vez del GeoTIFF y GDAL falla con "not recognized as being in a supported file format". El script las detecta con `utils.stac.items_legibles` y las excluye. Si Planetary Computer las repara, basta volver a ejecutar el script (tarda unos 45 minutos).

### `series_landsat_concepcion.parquet`

| Columna | Descripción |
|---|---|
| `id`, `nombre`, `tipo`, `lat`, `lon` | El punto (de `puntos_concepcion.csv`) |
| `fecha` | Fecha y hora UTC del paso del satélite |
| `plataforma` | `landsat-8` o `landsat-9` |
| `st_dn` | Mediana de la banda térmica en DN, **antes** de calibrar (para comprobar la calibración) |
| `st_c` | Temperatura de superficie calibrada, °C |
| `verde`, `rojo`, `nir`, `swir1` | Reflectancia calibrada (0 a 1). La roja sube con la turbidez |
| `ndwi` | NDWI de McFeeters, (verde - NIR) / (verde + NIR). Positivo sobre agua |
| `ndvi` | NDVI, (NIR - rojo) / (NIR + rojo) |
| `nubes_escena` | Porcentaje de nubes de la escena completa (`eo:cloud_cover`), no de la ventana |
| `pixeles_validos` | Fracción (0 a 1) de la ventana que pasó la máscara de calidad |
| `n_pixeles` | Píxeles de la ventana (círculo de 200 m de radio, píxeles de 30 m) |

- **Método:** para cada punto se carga una ventana de unos 400 m, se enmascara, se calibra y se resume con la **mediana** de los píxeles a 200 m o menos del punto. Si menos de la mitad de la ventana quedó libre de nubes, las variables de esa fecha son `NaN` (la fecha se conserva: no se rellena ni se elimina).
- **Filtro de escena:** se buscó con menos de 90 % de nubes en la escena; la máscara hace el resto del trabajo.
- **Limitaciones:** unas 10 a 25 observaciones válidas por año por punto (más desde 2022, cuando se suma Landsat 9), con menos en invierno; la máscara de Landsat deja pasar nubes delgadas y neblina costera, que se miden como agua muy fría. `utils.extremos.controlar_calidad_agua` descarta en el agua las observaciones con reflectancia roja mayor que 0,05 o SWIR1 mayor que 0,02 (demasiado brillo para agua abierta).
- **Metadatos:** el parquet lleva título, fuente, licencia, parámetros y descripción de columnas dentro del archivo: `utils.series.leer_metadatos_series()`.
- **Licencia:** dominio público (datos del USGS).

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
