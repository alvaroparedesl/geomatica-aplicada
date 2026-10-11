# Datos del taller

Datos preparados antes del taller para que los cuadernos carguen rápido en Colab.

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
