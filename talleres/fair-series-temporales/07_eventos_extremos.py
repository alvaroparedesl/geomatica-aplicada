# %% [markdown]
# # Detección de eventos extremos: umbrales clásicos y aprendizaje automático
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/07_eventos_extremos.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/)
#
# **Bloque 7 del taller "Datos FAIR, series temporales satelitales y detección de anomalías con Python"** (unos 100 minutos).
#
# Hasta aquí calculaste anomalías: cuánto se aleja un valor de lo normal para esa época del año. Ahora damos el paso siguiente: **decidir cuándo una anomalía es un evento extremo**. Lo haremos con la temperatura superficial del mar frente a Concepción, donde hay una serie de más de 40 años y una pregunta muy concreta: ¿cuántas olas de calor y de frío marinas hubo, cuáles fueron las más importantes y hay alguna ocurriendo hoy?
#
# En este cuaderno vas a:
#
# 1. Conocer los datos (NOAA OISST) y **encontrar su vacío de 2024 a 2026**, que es un problema real con el que tendremos que trabajar.
# 2. Construir **paso a paso, en pandas legible, el método de Hobday** (2016) para olas de calor y de frío marinas, y comprobar que coincide con las funciones de `utils`.
# 3. Interpretar los resultados: conteos, eventos mayores, el evento en curso y las categorías de Hobday (2018).
# 4. **Validar** el resultado contra `xmhw`, una implementación independiente.
# 5. Comparar los umbrales clásicos con un método de **aprendizaje automático** (Isolation Forest) y ver qué detecta cada uno.
# 6. Mapear la **frecuencia de eventos** sobre una malla de 34 a 40 S.
#
# | Parte | Minutos |
# |---|---|
# | Datos y vacío | 15 |
# | Hobday paso a paso | 30 |
# | Resultados y discusión del vacío | 15 |
# | Validación con `xmhw` | 10 |
# | Isolation Forest | 15 |
# | Mapa de frecuencia | 15 |

# %% [markdown]
# ## Configuración del entorno
#
# La celda `Preparar entorno` deja disponible la carpeta `utils/` del curso. Esta vez usamos `utils/extremos.py` (el método de Hobday, la malla y Isolation Forest) y `utils/datos_taller.py` (los datos preparados del taller).

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
# @title Importación de bibliotecas
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.display import display

from utils.extremos import (aislamiento_forest, aplicar_climatologia, cargar_oisst, climatologia_hobday,
                            detectar_eventos, dia_del_anio_366, evento_en_curso, frecuencia_eventos,
                            marcar_dias_evento, resumen_eventos, solapamiento)

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
ROJO, AZUL, GRIS = "#c0392b", "#2471a3", "#7f8c8d"

# %% [markdown]
# ## 1. Los datos: NOAA OISST
#
# **OISST** (*Optimum Interpolation Sea Surface Temperature*, versión 2.1) es un producto de la NOAA de temperatura superficial del mar:
#
# * **Diario**, desde el 1 de septiembre de 1981 hasta hoy.
# * **Malla de 0,25 grados** (unos 25 km): cada píxel resume una zona de unos 25 x 25 km.
# * Combina **satélites** (radiómetros AVHRR), **boyas y barcos**. Un análisis de interpolación óptima mezcla esas fuentes, así que el resultado **no tiene huecos por nubes**: donde el satélite no ve, se estima con lo que miden las demás observaciones y los días vecinos.
# * Ve **el océano abierto y la costa**, pero no **el interior de la bahía**: Concepción y Talcahuano caben dentro de un solo píxel (y de uno mezclado con tierra). Para la bahía misma usaríamos Sentinel-2 o datos locales, como en los bloques anteriores.
#
# Leemos el dato desde Microsoft Planetary Computer (colección `noaa-cdr-sea-surface-temperature-optimum-interpolation`). Para que Colab no dependa de descargas largas, dejamos una copia preparada en `talleres/fair-series-temporales/datos/` y la abrimos con `cargar_oisst`. Elegimos el píxel más cercano a Concepción (latitud -36,875, longitud -73,375).

# %%
sst = cargar_oisst("punto")
print(f"Del {sst.index[0].date()} al {sst.index[-1].date()}: {len(sst)} días en el calendario")
print(f"Días con dato: {sst.notna().sum()}   Días sin dato (NaN): {sst.isna().sum()}")
sst.head()

# %% [markdown]
# Un primer gráfico de la serie completa. Mira bien la parte derecha antes de seguir leyendo.

# %%
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(sst.index, sst, lw=0.5, color=AZUL)
ax.set_ylabel("SST (°C)")
ax.set_title("Temperatura superficial del mar frente a Concepción (OISST, píxel -36,875 / -73,375)")
plt.show()

# %% [markdown]
# ## 2. El vacío de datos: un problema real
#
# La línea se corta entre 2024 y 2026. No es un error del gráfico: **Planetary Computer no tiene OISST para ese periodo**. No lo rellenamos con otra fuente a propósito, porque así aprendemos qué hacer cuando una serie real tiene huecos.
#
# `cargar_oisst` entrega el calendario diario completo con `NaN` en los días sin dato. Primero **encontremos** los vacíos con código, en vez de fiarnos del gráfico: un `NaN` consecutivo forma un bloque, y cada bloque tiene inicio, fin y duración.

# %%
sin_dato = sst.isna()
bloque = (sin_dato != sin_dato.shift()).cumsum()            # número de bloque: cambia cada vez que cambia sin_dato
vacios = (sst[sin_dato].index.to_series()
          .groupby(bloque[sin_dato])
          .agg(inicio="min", fin="max", dias="size")
          .reset_index(drop=True))
vacios

# %% [markdown]
# Hay **dos vacíos**: uno largo de 624 días (del 2024-06-20 al 2026-03-05) y uno de dos días (2026-03-07 y 2026-03-08), separados por un único día con dato. Veamos cuántos días con dato tiene cada año, para dimensionar el problema.

# %%
dias_por_anio = sst.groupby(sst.index.year).apply(lambda s: s.notna().sum())
dias_por_anio.tail(8)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5), gridspec_kw={"width_ratios": [2, 1]})
ax1.plot(sst.loc["2022":].index, sst.loc["2022":], color=AZUL, lw=0.8)
for fila in vacios.itertuples():
    ax1.axvspan(fila.inicio, fila.fin + pd.Timedelta(days=1), color=GRIS, alpha=0.35)
ax1.text(pd.Timestamp("2025-04-01"), sst.loc["2022":].max() - 0.3, "vacío de 624 días", ha="center", color="black")
ax1.set_ylabel("SST (°C)")
ax1.set_title("Desde 2022: el vacío (gris)")
ax2.bar(dias_por_anio.index[-6:], dias_por_anio.iloc[-6:], color=[AZUL if d > 360 else ROJO for d in dias_por_anio.iloc[-6:]])
ax2.set_ylabel("días con dato")
ax2.set_title("Días con dato por año")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### ¿Qué afecta este vacío y qué no?
#
# Antes de decidir qué hacer, razona qué partes del análisis se ven afectadas:
#
# * **La climatología de 1991 a 2020 no se afecta.** El vacío está completamente fuera del periodo base (1991 a 2020), así que los umbrales usan sus 30 años completos.
# * **Los eventos dentro del vacío no se pueden detectar.** No sabemos qué pasó entre junio de 2024 y marzo de 2026: pudo haber olas de calor, de frío o nada. Un conteo que no lo diga está subestimando.
# * **Una racha no puede cruzar el vacío.** Un `NaN` no es "temperatura normal": no excede el umbral, así que corta la racha. Y la regla de unión de brechas (2 días o menos) tampoco puede puentear un vacío, ni siquiera el de un solo día. Si lo hiciera, podría inventar un evento de casi dos años.
# * **Las comparaciones entre décadas se distorsionan.** La década de 2020 solo tiene unos 5 años de dato de los casi 7 transcurridos: menos eventos contados no significa menos eventos ocurridos.
# * **El evento en curso sí es detectable**, porque empieza después del vacío (2026-08-17).
#
# Volveremos a esto con números en la sección 6, cuando ya sepamos contar eventos.

# %% [markdown]
# ## 3. El método de Hobday paso a paso
#
# Hobday et al. (2016) definen una **ola de calor marina** como un periodo de **al menos 5 días seguidos** con temperatura por encima del **percentil 90** de la climatología de ese día del año. Una **ola de frío** es lo mismo hacia abajo (por debajo del percentil 10). Además, dos eventos separados por **2 días o menos** se consideran uno solo.
#
# Los parámetros que usaremos:
#
# | Parámetro | Valor |
# |---|---|
# | Periodo base de la climatología | 1991 a 2020 |
# | Ventana para juntar días | 11 días (5 antes y 5 después) |
# | Suavizado de la climatología | media móvil de 31 días |
# | Umbral | percentil 90 (calor) y 10 (frío) |
# | Duración mínima | 5 días |
# | Unión de brechas | 2 días o menos |
#
# Lo escribiremos con pandas y numpy, sin ocultar ningún paso. Al final lo compararemos con `utils.extremos`.

# %% [markdown]
# ### 3.1 Un calendario de 366 días
#
# La climatología es **una curva por día del año**. El problema: los años bisiestos tienen 366 días y el resto 365, así que el día número 100 no cae en la misma fecha todos los años. La convención de Hobday es usar un calendario de 366 días: el 29 de febrero es siempre el día 60 y, en los años no bisiestos, desde el 1 de marzo se suma 1 (el 1 de marzo es siempre el día 61).

# %%
datos = sst.to_frame("sst")
dia = datos.index.dayofyear.to_numpy()
datos["dia"] = dia + (~datos.index.is_leap_year & (datos.index.month >= 3))   # +1 desde marzo en años no bisiestos
datos.loc[["2019-03-01", "2020-03-01", "2020-02-29"]]

# %% [markdown]
# El 1 de marzo es el día 61 en 2019 y en 2020, y el 29 de febrero de 2020 es el día 60.

# %% [markdown]
# ### 3.2 Climatología: media y percentiles por día del año
#
# Para cada día del año juntamos los valores del periodo base de ese día y de los 5 días anteriores y posteriores (ventana de 11 días) **de todos los años**, es decir, unos 30 años x 11 días = 330 valores por cada día del año. De ahí sacamos la media y los percentiles 90 y 10.
#
# Para hacerlo ordenado armamos una tabla con un año por fila y un día del año por columna.

# %%
base = datos.loc["1991":"2020"]
tabla_anios = base.pivot_table(index=base.index.year, columns="dia", values="sst")   # años x 366 días
print(tabla_anios.shape)
tabla_anios.iloc[:3, 58:63]   # los años no bisiestos no tienen día 60 (29 de febrero): NaN

# %%
mitad = 11 // 2
valores = tabla_anios.to_numpy()
# Ventana circular: el 31 de diciembre (día 366) es vecino del 1 de enero, así que pegamos los bordes
ampliada = np.concatenate([valores[:, -mitad:], valores, valores[:, :mitad]], axis=1)

media, p90, p10 = (np.full(366, np.nan) for _ in range(3))
for i in range(366):
    junto = ampliada[:, i:i + 11].ravel()      # los valores de 11 días de todos los años
    junto = junto[np.isfinite(junto)]          # ignorar los NaN
    media[i], p90[i], p10[i] = junto.mean(), np.percentile(junto, 90), np.percentile(junto, 10)

print(f"El 1 de febrero (día 32): media {media[31]:.2f} °C, p90 {p90[31]:.2f} °C, p10 {p10[31]:.2f} °C")

# %% [markdown]
# ### 3.3 Suavizado
#
# Con solo 30 años las curvas quedan dentadas. Hobday suaviza cada una con una **media móvil de 31 días**, también circular (el final del año enlaza con el principio).

# %%
def suavizar(curva, ancho=31):
    mitad = ancho // 2
    pegada = np.concatenate([curva[-mitad:], curva, curva[:mitad]])      # cerrar el año sobre sí mismo
    return pd.Series(pegada).rolling(ancho, center=True).mean().to_numpy()[mitad:-mitad]


clima_manual = pd.DataFrame({"media": suavizar(media), "p90": suavizar(p90), "p10": suavizar(p10)},
                            index=pd.RangeIndex(1, 367, name="dia"))
clima_manual.iloc[[0, 31, 90, 181, 273]].round(2)

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.fill_between(clima_manual.index, clima_manual["p10"], clima_manual["p90"], color=GRIS, alpha=0.25, label="entre p10 y p90")
ax.plot(clima_manual["media"], color="black", label="media")
ax.plot(clima_manual["p90"], color=ROJO, ls="--", label="p90 (umbral de calor)")
ax.plot(clima_manual["p10"], color=AZUL, ls="--", label="p10 (umbral de frío)")
ax.set_xlabel("día del año (calendario de 366 días)")
ax.set_ylabel("SST (°C)")
ax.set_title("Climatología 1991-2020 frente a Concepción")
ax.legend()
plt.show()

# %% [markdown]
# El ciclo anual es suave (máximo en verano, mínimo en invierno), y los umbrales siguen a la media con una banda de aproximadamente +-1 °C. Ese **margen entre la media y el umbral** también servirá para las categorías.

# %% [markdown]
# ### 3.4 Excedencia: ¿qué días superan el umbral?
#
# Alineamos la climatología con cada fecha de la serie (por día del año) y marcamos con `True` los días en que la temperatura supera el p90 (calor) o queda bajo el p10 (frío). Un `NaN` da `False`: **un día sin dato no excede nada**.

# %%
datos = datos.join(clima_manual, on="dia")                 # media, p90 y p10 de cada fecha según su día del año
datos["anomalia"] = datos["sst"] - datos["media"]
datos["excede_calor"] = datos["sst"] > datos["p90"]
datos["excede_frio"] = datos["sst"] < datos["p10"]
print(f"Días sobre el p90: {datos['excede_calor'].sum()}  ({datos['excede_calor'].sum() / datos['sst'].notna().sum():.1%} de los días con dato)")
print(f"Días bajo el p10:  {datos['excede_frio'].sum()}  ({datos['excede_frio'].sum() / datos['sst'].notna().sum():.1%} de los días con dato)")
datos.loc["1997-07-01":"1997-07-12", ["sst", "p90", "anomalia", "excede_calor"]].round(2)

# %% [markdown]
# Por construcción, cerca del 10 % de los días supera el p90. Pero un día suelto no es una ola de calor: hace falta que persista.

# %% [markdown]
# ### 3.5 Rachas de al menos 5 días
#
# Una **racha** es una secuencia de días consecutivos con `True`. Para encontrarlas numeramos los bloques (igual que hicimos con los vacíos) y nos quedamos con el primer y último día de cada bloque de `True`.

# %%
def rachas(excede):
    """Tabla con inicio, fin y duración de cada racha de True (días consecutivos)."""
    bloque = (excede != excede.shift()).cumsum()
    fechas = excede.index.to_series()[excede]
    tabla = fechas.groupby(bloque[excede]).agg(inicio="min", fin="max").reset_index(drop=True)
    tabla["dias"] = (tabla["fin"] - tabla["inicio"]).dt.days + 1
    return tabla


todas = rachas(datos["excede_calor"])
print(f"Rachas de calor de cualquier largo: {len(todas)}")
print(todas["dias"].value_counts().sort_index().head(8).to_string())

# %% [markdown]
# La mayoría de las rachas son cortas (1 a 4 días): ruido que no queremos llamar evento. Nos quedamos con las de **5 días o más**.

# %%
largas = todas[todas["dias"] >= 5].reset_index(drop=True)
print(f"Rachas de 5 días o más: {len(largas)}")
largas.head()

# %% [markdown]
# ### 3.6 Unir brechas de 2 días o menos (después de filtrar)
#
# Dos rachas largas separadas por una pausa breve (por ejemplo, un día frío en medio de una ola de calor) son, físicamente, el mismo evento. Hobday las une si la **brecha** entre ellas es de **2 días o menos**.
#
# Dos detalles importantes:
#
# 1. **El orden importa: primero se filtra por duración y después se unen las brechas.** Si uniéramos primero, dos rachas de 3 días separadas por 1 día formarían un evento de 7 días; con el orden de Hobday, cada racha de 3 días se descarta antes y no queda nada.
# 2. **No se puede unir a través de un vacío.** Si en la brecha hay un `NaN`, no sabemos qué pasó ahí, así que no unimos aunque la brecha sea corta.

# %%
largas["brecha"] = (largas["inicio"] - largas["fin"].shift()).dt.days - 1        # días entre esta racha y la anterior
largas["hay_vacio"] = [
    bool(sst.loc[fin_ant + pd.Timedelta(days=1): ini - pd.Timedelta(days=1)].isna().any()) if pd.notna(fin_ant) else False
    for fin_ant, ini in zip(largas["fin"].shift(), largas["inicio"])
]
une_con_la_anterior = (largas["brecha"] <= 2) & ~largas["hay_vacio"]            # la primera racha tiene brecha NaN: no se une
largas["evento"] = (~une_con_la_anterior).cumsum()                              # número de evento: sube cuando NO se une
largas[["inicio", "fin", "dias", "brecha", "hay_vacio", "evento"]].head(8)

# %%
eventos_calor_manual = largas.groupby("evento").agg(inicio=("inicio", "min"), fin=("fin", "max")).reset_index(drop=True)
eventos_calor_manual["duracion"] = (eventos_calor_manual["fin"] - eventos_calor_manual["inicio"]).dt.days + 1
eventos_calor_manual["intensidad_max"] = [datos.loc[i:f, "anomalia"].max() for i, f in
                                          zip(eventos_calor_manual["inicio"], eventos_calor_manual["fin"])]
print(f"Olas de calor (versión manual): {len(eventos_calor_manual)}")
eventos_calor_manual.nlargest(5, "duracion")

# %% [markdown]
# Con unas 40 líneas de pandas tenemos el método completo para calor. El de frío es idéntico con `sst < p10`. Ahora lo contrastamos con las funciones de `utils.extremos`.

# %% [markdown]
# ### 3.7 Lo mismo con `utils.extremos`
#
# `climatologia_hobday` calcula la climatología (pasos 3.1 a 3.3) y `detectar_eventos` hace los pasos 3.4 a 3.6, además de calcular duración, intensidad, día del pico y categoría. Comprobemos que dan **exactamente** lo mismo que nuestra versión manual.

# %%
clima = climatologia_hobday(sst)                       # 1991-2020, ventana de 11 días, suavizado de 31
olas_calor = detectar_eventos(sst, clima, "calor")     # p90, 5 días, brechas de 2 días
olas_frio = detectar_eventos(sst, clima, "frio")       # p10

print("Climatología: diferencia máxima con la versión manual:",
      f"{np.abs(clima[['media', 'p90', 'p10']].to_numpy() - clima_manual.to_numpy()).max():.2e} °C")
mismos = (olas_calor["inicio"].to_numpy() == eventos_calor_manual["inicio"].to_numpy()).all() and \
         (olas_calor["fin"].to_numpy() == eventos_calor_manual["fin"].to_numpy()).all()
print(f"Olas de calor: utils {len(olas_calor)} y manual {len(eventos_calor_manual)}; mismas fechas de inicio y fin: {mismos}")
print(f"Olas de frío (utils): {len(olas_frio)}")

# %% [markdown]
# La versión manual y la de `utils` coinciden. Ahora probemos las reglas en **series de juguete**, donde sabemos la respuesta. Armamos una climatología plana (media 0, umbrales +1 y -1) y series con tramos de "calor" (valor 2) y de normalidad (valor 0).

# %%
clima_plana = pd.DataFrame({"media": 0.0, "p90": 1.0, "p10": -1.0}, index=pd.RangeIndex(1, 367, name="dia"))


def juguete(*tramos, vacio=None):
    """Serie diaria: cada tramo es (días, valor). Un 'vacio' opcional es la posición de un NaN."""
    valores = np.concatenate([np.full(n, v, dtype=float) for n, v in tramos])
    if vacio is not None:
        valores[vacio] = np.nan
    return pd.Series(valores, index=pd.date_range("2021-03-01", periods=len(valores), freq="D"))


casos = {
    "3 días calientes, 1 normal, 3 calientes": juguete((3, 2), (1, 0), (3, 2)),
    "5 calientes, 2 normales, 5 calientes": juguete((5, 2), (2, 0), (5, 2)),
    "5 calientes, 3 normales, 5 calientes": juguete((5, 2), (3, 0), (5, 2)),
    "5 calientes, 1 vacío, 5 calientes": juguete((5, 2), (1, 2), (5, 2), vacio=5),
    "4 calientes": juguete((4, 2), (10, 0)),
}
for nombre, serie in casos.items():
    ev = detectar_eventos(serie, clima_plana, "calor")
    print(f"{nombre:42s} -> {len(ev)} evento(s)", list(ev["duracion"]))

# %% [markdown]
# Lee los resultados:
#
# * 3 + 1 + 3: **0 eventos**. Cada racha de 3 días se descarta primero (filtrar antes de unir).
# * 5 + 2 + 5: **1 evento de 12 días**. La brecha de 2 días se une y se cuenta dentro del evento.
# * 5 + 3 + 5: **2 eventos**. Con 3 días de pausa ya no se une.
# * 5 + vacío + 5: **2 eventos**. El día sin dato cuenta como brecha de 1 día, pero no se puentea.
# * 4 días: **0 eventos**. Falta uno para llegar a 5.

# %% [markdown]
# ### Tu turno 1: qué tan sensible es el conteo a los parámetros
#
# Los parámetros de Hobday son convenciones, no leyes de la naturaleza. Cambia los valores de la celda siguiente (por ejemplo, percentil 95, duración mínima de 3 o de 7 días, brecha de 0 o de 5 días) y observa cuánto cambia el número de olas de calor.
#
# Preguntas para pensar:
#
# * ¿Qué parámetro mueve más el conteo?
# * Si quisieras comparar tus resultados con otros estudios, ¿qué convendría dejar fijo?

# %%
percentil = 90          # prueba 95 (más exigente) u 85
duracion_min = 5        # prueba 3 o 7
brecha_max = 2          # prueba 0 o 5

clima_t = climatologia_hobday(sst, percentiles=(percentil, 100 - percentil))
ev_t = detectar_eventos(sst, clima_t, "calor", duracion_min=duracion_min, brecha_max=brecha_max,
                        columna_umbral=f"p{percentil}")
print(f"p{percentil}, mínimo {duracion_min} días, brecha de hasta {brecha_max}: {len(ev_t)} olas de calor "
      f"({ev_t['duracion'].sum()} días en total)")

# %% [markdown]
# ## 4. Resultados: conteos, eventos mayores y evento en curso
#
# Volvemos a la versión de `utils` para sacar las conclusiones.

# %%
print(f"Olas de calor marinas: {len(olas_calor)}")
print(f"Olas de frío marinas:  {len(olas_frio)}")
print(f"Días en ola de calor: {olas_calor['duracion'].sum()}   Días en ola de frío: {olas_frio['duracion'].sum()}")

# %% [markdown]
# ### Los eventos más largos
#
# La tabla de eventos trae, además de las fechas, la **duración** (días), la **intensidad máxima** (la mayor anomalía, en °C), la **intensidad media**, la **intensidad acumulada** (°C-día) y la **categoría** (la veremos en un momento).

# %%
columnas = ["inicio", "fin", "duracion", "dia_pico", "intensidad_max", "intensidad_media", "categoria"]
olas_calor.nlargest(8, "duracion")[columnas].round({"intensidad_max": 2, "intensidad_media": 2}).reset_index(drop=True)

# %% [markdown]
# Los eventos más largos coinciden en el tiempo con años de El Niño:
#
# * **1997** (63 días, de julio a septiembre) y **1998** (52 días, de mayo a julio): El Niño de 1997-98, uno de los más fuertes del registro.
# * **2017** (44 días, de abril a mayo): coincide con el **El Niño costero de 2017**, que fue más intenso frente a Perú y Ecuador. Aquí solo vemos la coincidencia temporal en un píxel; asociar un evento con una causa exige más análisis.
# * **1982** (37 y 35 días, de agosto a octubre): El Niño de 1982-83, el otro gran evento de la serie.
#
# Entre ellos aparece uno **que no es de El Niño: el que empezó el 2026-08-17 y sigue activo**.

# %% [markdown]
# ### El evento en curso
#
# `evento_en_curso` devuelve el evento que sigue activo en el último día de la serie. Un evento "en curso" tiene duración provisoria: puede crecer, o terminar mañana.

# %%
actual = evento_en_curso(olas_calor)
print(f"Última fecha de la serie: {sst.index[-1].date()}")
print(f"Ola de calor en curso desde el {actual['inicio'].date()}: {actual['duracion']} días, "
      f"anomalía máxima {actual['intensidad_max']:.2f} °C (el {actual['dia_pico'].date()}), "
      f"anomalía media {actual['intensidad_media']:.2f} °C, categoría {int(actual['categoria'])}")
print("¿Hay una ola de frío en curso?", evento_en_curso(olas_frio) is not None)

# %% [markdown]
# ### Una mirada a los eventos
#
# En el gráfico de abajo, la línea es la anomalía diaria (respecto de la media climatológica), las franjas rojas son olas de calor y las azules, olas de frío. La franja gris es el vacío.

# %%
anomalia = (sst - aplicar_climatologia(sst, clima)["media"])
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(anomalia.index, anomalia, color="black", lw=0.4)
for ev in olas_calor.itertuples():
    ax.axvspan(ev.inicio, ev.fin + pd.Timedelta(days=1), color=ROJO, alpha=0.5, lw=0)
for ev in olas_frio.itertuples():
    ax.axvspan(ev.inicio, ev.fin + pd.Timedelta(days=1), color=AZUL, alpha=0.5, lw=0)
for fila in vacios.itertuples():
    ax.axvspan(fila.inicio, fila.fin + pd.Timedelta(days=1), color=GRIS, alpha=0.3, lw=0)
ax.set_ylabel("anomalía (°C)")
ax.set_title("Anomalía diaria de SST y eventos de Hobday (rojo: calor, azul: frío, gris: sin dato)")
plt.show()

# %% [markdown]
# Ahora un acercamiento al evento de 1997 para ver qué significa "exceder el umbral": la temperatura (línea azul) sube sobre el p90 (rojo punteado) y se mantiene allí varias semanas.

# %%
tramo = aplicar_climatologia(sst, clima).loc["1997-05-15":"1997-10-15"]
ev97 = olas_calor.loc[olas_calor["duracion"].idxmax()]          # el evento más largo: julio a septiembre de 1997
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(tramo.index, tramo["valor"], color=AZUL, label="SST diaria")
ax.plot(tramo.index, tramo["media"], color="black", label="media climatológica")
ax.plot(tramo.index, tramo["p90"], color=ROJO, ls="--", label="p90")
ax.fill_between(tramo.index, tramo["media"], tramo["valor"],
                where=(tramo.index >= ev97["inicio"]) & (tramo.index <= ev97["fin"]), color=ROJO, alpha=0.3, label="ola de calor")
ax.set_ylabel("SST (°C)")
ax.set_title(f"Ola de calor de {ev97['inicio'].date()} a {ev97['fin'].date()} ({ev97['duracion']} días)")
ax.legend(ncol=2)
plt.show()

# %% [markdown]
# ### Las olas de frío
#
# Misma tabla, hacia abajo. La ola de frío más larga (2007) dura 96 días, más que cualquiera de calor. Las anomalías son negativas.

# %%
olas_frio.nlargest(5, "duracion")[columnas].round({"intensidad_max": 2, "intensidad_media": 2}).reset_index(drop=True)

# %% [markdown]
# ### Categorías de Hobday (2018)
#
# No todos los eventos pesan igual. Hobday et al. (2018) propusieron cuatro categorías según **cuántas veces la anomalía del pico supera la distancia entre la media y el umbral** ese día:
#
# | Categoría | Anomalía del pico respecto de la distancia media-umbral |
# |---|---|
# | I moderada | 1 a 2 veces |
# | II fuerte | 2 a 3 veces |
# | III severa | 3 a 4 veces |
# | IV extrema | 4 veces o más |
#
# Ejemplo con el evento más largo (1997): calculamos el cociente en el día del pico y de ahí sale su categoría.

# %%
pico = ev97["dia_pico"]
margen = clima.loc[dia_del_anio_366(pd.DatetimeIndex([pico]))[0], "p90"] - clima.loc[dia_del_anio_366(pd.DatetimeIndex([pico]))[0], "media"]
print(f"Pico del {pico.date()}: anomalía {ev97['intensidad_max']:.2f} °C, distancia media-p90 {margen:.2f} °C, "
      f"cociente {ev97['intensidad_max'] / margen:.2f} -> categoría {int(ev97['categoria'])}")

# %%
categorias = pd.DataFrame({
    "calor": olas_calor["categoria"].value_counts(),
    "frío": olas_frio["categoria"].value_counts(),
}).fillna(0).astype(int).sort_index()
categorias.index = [f"{c} {n}" for c, n in zip(categorias.index, ["moderada", "fuerte", "severa", "extrema"][:len(categorias)])]
categorias

# %% [markdown]
# La gran mayoría son **moderadas**. Es lo esperado: los eventos extremos de verdad son raros. Las fuertes y severas se reconocen por su impacto en el ecosistema.
#
# ### Resumen por década
#
# `resumen_eventos(eventos, "decada")` cuenta los eventos por década de inicio.

# %%
res_calor = resumen_eventos(olas_calor, "decada")
res_frio = resumen_eventos(olas_frio, "decada")
pd.concat({"calor": res_calor, "frío": res_frio}, axis=1)

# %% [markdown]
# Cuidado con leer la columna de 2020: la década tiene menos eventos, pero **también muchos menos días observados** por el vacío. Comparemos contra los días con dato, y no contra el calendario.

# %%
dias_dato = sst.groupby((sst.index.year // 10 * 10)).apply(lambda s: s.notna().sum())
dias_dato.name = "dias_con_dato"
por_decada = pd.DataFrame({
    "dias_con_dato": dias_dato,
    "calor": res_calor["n_eventos"],
    "frio": res_frio["n_eventos"],
}).fillna(0)
por_decada["calor_por_decada_observada"] = (por_decada["calor"] / (por_decada["dias_con_dato"] / 3652.5)).round(1)
por_decada["frio_por_decada_observada"] = (por_decada["frio"] / (por_decada["dias_con_dato"] / 3652.5)).round(1)
por_decada

# %% [markdown]
# La **tasa por década observada** (eventos / años con dato x 10) corrige parte del problema. La década de 1980 tiene solo 8 años y 4 meses de dato, por lo que su tasa también se normaliza. Con tasas así, la comparación entre décadas es honesta con lo que sabemos.

# %% [markdown]
# ## 5. Validación contra `xmhw`
#
# Una implementación propia es útil para aprender, pero ¿cómo sabemos que está bien? Contrastándola con una **implementación independiente**. [`xmhw`](https://github.com/coecms/xmhw) es una librería de la comunidad que implementa el método de Hobday sobre xarray. No está en PyPI: se instala desde GitHub, fijada a la versión con que probamos este taller (está en `requirements.txt`).
#
# La instalación es opcional: si falla, el resto del cuaderno sigue funcionando.

# %%
# @title Instalar xmhw (opcional)
import importlib.util

if importlib.util.find_spec("xmhw") is None:
    requisito = next(l.strip() for l in (RAIZ / "talleres/fair-series-temporales/requirements.txt").read_text().splitlines()
                     if l.strip().startswith("xmhw"))
    instalacion = subprocess.run([sys.executable, "-m", "pip", "install", "-q", requisito], capture_output=True, text=True)
    if instalacion.returncode != 0:
        print("No se pudo instalar xmhw. Puedes saltarte la validación: el resto del cuaderno no la necesita.")
        print(instalacion.stderr[-400:])
HAY_XMHW = importlib.util.find_spec("xmhw") is not None
print("xmhw disponible:", HAY_XMHW)

# %% [markdown]
# `xmhw` calcula la climatología con `threshold` y detecta con `detect`. Para las olas de frío, `xmhw` invierte la serie, por lo que se pide `coldSpells=True` con `pctile=90` (no 10).

# %%
if HAY_XMHW:
    import xarray as xr
    from xmhw.xmhw import detect, threshold

    serie_xr = xr.DataArray(sst.to_numpy(), dims="time", coords={"time": sst.index.to_numpy()})
    filas = []
    comparados = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tipo, mios, columna, signo in (("calor", olas_calor, "p90", 1), ("frio", olas_frio, "p10", -1)):
            umbral = threshold(serie_xr, climatologyPeriod=[1991, 2020], pctile=90, coldSpells=tipo == "frio")
            ev = detect(serie_xr, umbral["thresh"], umbral["seas"], minDuration=5, joinGaps=True, maxGap=2,
                        coldSpells=tipo == "frio")
            x = pd.DataFrame({"inicio": pd.to_datetime(ev.time_start.values.ravel()),
                              "fin": pd.to_datetime(ev.time_end.values.ravel()),
                              "duracion": ev.duration.values.ravel(),
                              "categoria": ev.category.values.ravel()}).dropna(subset=["inicio"])
            dif_umbral = np.abs(signo * umbral["thresh"].squeeze().to_series().to_numpy() - clima[columna].to_numpy()).max()
            comparados[tipo] = (x, mios)
            filas.append({
                "tipo": tipo, "eventos xmhw": len(x), "eventos utils": len(mios),
                "mismo inicio": len(set(x["inicio"]) & set(mios["inicio"])),
                "mismo fin": len(set(x["fin"]) & set(mios["fin"])),
                "dif. máx. umbral (°C)": round(float(dif_umbral), 4),
            })
    display(pd.DataFrame(filas).set_index("tipo"))
else:
    print("Validación omitida: xmhw no está instalado.")

# %% [markdown]
# El acuerdo es casi total: los **mismos 88 eventos de calor y 111 de frío**, con las mismas fechas de fin. Veamos las pocas diferencias:

# %%
if HAY_XMHW:
    for tipo, (x, mios) in comparados.items():
        solo_x = sorted(set(x["inicio"]) - set(mios["inicio"]))
        solo_m = sorted(set(mios["inicio"]) - set(x["inicio"]))
        print(f"{tipo}: inicios distintos -> xmhw {[d.date() for d in solo_x]} / utils {[d.date() for d in solo_m]}")
    x, mios = comparados["calor"]
    cruce = mios.merge(x, on="fin", suffixes=("", "_xmhw"))
    print("Eventos de calor con distinta duración:",
          cruce.loc[cruce["duracion"] != cruce["duracion_xmhw"], ["inicio", "inicio_xmhw", "fin", "duracion", "duracion_xmhw"]]
          .to_string(index=False))
    print("Eventos de calor con distinta categoría:",
          cruce.loc[cruce["categoria"] != cruce["categoria_xmhw"], ["fin", "categoria", "categoria_xmhw"]]
          .to_string(index=False))

# %% [markdown]
# ### Por qué las diferencias son pequeñas y esperables
#
# * **El primer día de la serie.** `xmhw` no cuenta el 1981-09-01 como inicio de un evento: el primer evento de calor empieza un día después y dura un día menos. En nuestra versión, si la serie parte ya sobre el umbral, el evento parte ese día.
# * **Las curvas de umbral difieren menos de 0,007 °C.** El 29 de febrero y los bordes de la ventana se tratan de forma ligeramente distinta. Un día que cae a menos de eso del umbral puede contarse distinto, y con 88 y 111 eventos basta un día para mover un fin. Aquí no ocurrió en ningún fin de evento.
# * **La categoría se define sobre otro día.** Los dos eventos de calor de categoría distinta (1982-10-25 y 1998-07-16) no son un error de umbral: `xmhw` toma el **mayor cociente a lo largo de todo el evento**, y `utils` el cociente **del día de mayor anomalía**. Como la distancia entre la media y el umbral cambia con la estación, ambos días pueden no ser el mismo. Hobday et al. (2018) hablan de la intensidad del pico, que es lo que implementa `utils`; lo importante es **saber cuál de las dos reglas usó cada herramienta** cuando compares categorías.
# * **Un día antes de un vacío.** En otras series `xmhw` puede contar como último día de un evento el primer día de un vacío, porque trata los `NaN` de otra manera. No ocurre con estos datos, pero es la razón por la que las pruebas de `utils` comparan con tolerancia de un evento y verifican por separado que ningún evento incluya un día sin dato.
#
# La lección general: **dos implementaciones del mismo método casi nunca coinciden al 100 %**, y las diferencias deben poder explicarse. Si no puedes explicar una diferencia, hay un error en alguna de las dos.

# %% [markdown]
# ## 6. Volvamos al vacío: ¿rellenar, ignorar o reportar?
#
# Ya sabemos contar eventos, así que podemos medir qué cambia según lo que hagamos con el vacío de 2024 a 2026. Probamos cuatro opciones:
#
# 1. **Ignorar**: dejar los `NaN` y no detectar nada en ellos (lo que hemos hecho).
# 2. **Rellenar interpolando** linealmente entre el último dato antes del vacío y el primero después.
# 3. **Rellenar con la climatología** (la media del día del año).
# 4. **Recortar** la serie hasta el último dato antes del vacío.

# %%
def contar(serie, nombre):
    c, f = detectar_eventos(serie, clima, "calor"), detectar_eventos(serie, clima, "frio")
    dentro = lambda ev: int(((ev["inicio"] <= vacios.loc[0, "fin"]) & (ev["fin"] >= vacios.loc[0, "inicio"])).sum())
    return {"opción": nombre, "olas de calor": len(c), "olas de frío": len(f),
            "eventos dentro del vacío": dentro(c) + dentro(f),
            "ola en curso detectada": evento_en_curso(c) is not None}


media_del_dia = aplicar_climatologia(sst, clima)["media"]
opciones = pd.DataFrame([
    contar(sst, "1. ignorar (NaN)"),
    contar(sst.interpolate(), "2. interpolar linealmente"),
    contar(sst.fillna(media_del_dia), "3. rellenar con la climatología"),
    contar(sst.loc[:vacios.loc[0, "inicio"] - pd.Timedelta(days=1)], "4. recortar antes del vacío"),
]).set_index("opción")
opciones

# %% [markdown]
# Qué nos dicen los números:
#
# * **Interpolar inventa eventos.** La recta entre dos fechas separadas por casi dos años no tiene ninguna relación con el mar: aparecen eventos de calor **dentro del vacío**, que no pasaron o que no sabemos si pasaron. Veámoslos.

# %%
interpolada = sst.interpolate()
falsos = detectar_eventos(interpolada, clima, "calor")
falsos = falsos[(falsos["inicio"] >= vacios.loc[0, "inicio"]) & (falsos["inicio"] <= vacios.loc[0, "fin"])]
falsos[["inicio", "fin", "duracion", "intensidad_max", "categoria"]].round({"intensidad_max": 2}).reset_index(drop=True)

# %% [markdown]
# Un "evento" de 163 días y categoría III que no existe en ningún dato. Es el peor error posible: **inventar un evento extremo**.
#
# * **Rellenar con la climatología** no cambia los conteos, pero tiene otro problema: convierte 624 días de "no sabemos" en 624 días de "normal" (anomalía cero). Cualquier estadística posterior, como el porcentaje de días en evento, queda diluida y parece más tranquila de lo que fue.
# * **Recortar** pierde los eventos de 2026, incluida la ola de calor en curso, justo lo que más interesa hoy.
# * **Ignorar y reportar** es lo único que no inventa ni diluye nada, pero hay que **decirlo**: el conteo es de un registro incompleto, y las tasas deben normalizarse por los días observados (como hicimos en la sección 4).
#
# ### Tu turno 2: debate (10 minutos)
#
# En grupos de 2 o 3 personas:
#
# 1. Si tuvieras que escribir el resultado en un informe o en un artículo, ¿qué opción elegirías y qué frase pondrías sobre el vacío?
# 2. ¿Existe algún caso en que rellenar sea razonable? Pista: piensa en vacíos de uno o dos días, rodeados de datos, en una variable que cambia lentamente.
# 3. Prueba con código tu idea de la pregunta 2: rellena **solo** los vacíos de 3 días o menos (el de 2 días de marzo de 2026) y deja el largo como `NaN`. Cuidado: `interpolate(limit=3)` rellenaría **también los primeros 3 días del vacío largo**, así que necesitas medir el largo de cada bloque de `NaN` antes de decidir qué rellenar.

# %%
# Punto de partida: largo de cada bloque de NaN en cada fecha (0 si hay dato)
largo_bloque = sin_dato.groupby(bloque).transform("sum").where(sin_dato, 0)
corto = (largo_bloque > 0) & (largo_bloque <= 3)
sst_parcial = sst.copy()
sst_parcial[corto] = sst.interpolate()[corto]          # solo los vacíos cortos
print(f"Días rellenados: {int(corto.sum())}   NaN que quedan: {int(sst_parcial.isna().sum())}")
print(contar(sst_parcial, "5. rellenar solo vacíos cortos"))

# %% [markdown]
# Con vacíos cortos hay una diferencia mínima en el conteo: el relleno de dos días en marzo de 2026 no cambia ningún evento. Para el vacío largo, no hay relleno honesto: o se dice "no hay dato", o se consigue otra fuente (por ejemplo, el archivo de NOAA directamente, que no tiene este hueco) y se **documenta** cuál se usó.

# %% [markdown]
# ## 7. Umbrales clásicos frente a aprendizaje automático
#
# Hobday es un método **basado en reglas**: tú decides el percentil, la duración mínima y la brecha, y el método es transparente (cada evento se explica con una regla). La alternativa es dejar que un algoritmo de **aprendizaje automático** decida qué días son "raros".
#
# **Isolation Forest** (bosque de aislamiento) es un método de detección de atípicos no supervisado. Construye muchos árboles de decisión con cortes aleatorios; los puntos raros quedan aislados con pocos cortes, y los comunes necesitan muchos. No usa umbrales ni conoce el calendario: solo ve números.
#
# `aislamiento_forest` le entrega a cada día tres características: la **anomalía** (respecto de la media climatológica, para quitar la estacionalidad), la anomalía promediada en 7 días y su **cambio** en 7 días. Hay que fijar la **contaminación**: la fracción de días que se espera que sean atípicos (aquí 10 %).

# %%
forest = aislamiento_forest(sst, clima, contaminacion=0.1)
forest[["anomalia", "anomalia_movil", "cambio", "puntaje", "atipico"]].dropna().tail(5).round(2)

# %%
marca_calor = marcar_dias_evento(sst, olas_calor)
marca_frio = marcar_dias_evento(sst, olas_frio)
sol_calor = solapamiento(marca_calor, forest["atipico_calor"])
sol_frio = solapamiento(marca_frio, forest["atipico_frio"])
pd.DataFrame({"calor": sol_calor, "frío": sol_frio}).T.rename(columns={
    "dias_a": "días Hobday", "dias_b": "días Isolation Forest", "comunes": "días en común", "jaccard": "índice de Jaccard"}).round(2)

# %% [markdown]
# El **índice de Jaccard** (días en común / días de la unión) va de 0 (sin ningún día en común) a 1 (idénticos). Con valores de 0,2 a 0,3 los dos métodos **comparten solo entre un cuarto y un tercio** de los días que marcan entre ambos. Son preguntas distintas:
#
# * Hobday pregunta: "¿esta temperatura, sostenida durante 5 días o más, supera lo que ocurre el 10 % de las veces en esta época del año?"
# * Isolation Forest pregunta: "¿esta combinación de anomalía y cambio es poco común en toda la serie?"
#
# Veamos qué ve cada uno en el tiempo.

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(forest.index, forest["anomalia"], color="black", lw=0.3)
for ev in olas_calor.itertuples():
    ax.axvspan(ev.inicio, ev.fin + pd.Timedelta(days=1), ymin=0.5, ymax=1, color=ROJO, alpha=0.5, lw=0)
for ev in olas_frio.itertuples():
    ax.axvspan(ev.inicio, ev.fin + pd.Timedelta(days=1), ymin=0, ymax=0.5, color=AZUL, alpha=0.5, lw=0)
raros = forest[forest["atipico"]]
ax.scatter(raros.index, raros["anomalia"], s=3, color="#f39c12", label="atípico según Isolation Forest", zorder=3)
ax.set_ylabel("anomalía (°C)")
ax.set_title("Hobday (franjas) frente a Isolation Forest (puntos naranjas)")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# ### ¿Por qué Isolation Forest no ve la ola de calor de 2026?
#
# El evento actual tiene una anomalía media de unos +1,3 °C. Para Hobday es una ola de calor porque se **sostiene** por encima del umbral estacional durante semanas. Para Isolation Forest, +1,3 °C **no es raro** comparado con el resto de la serie. Veamos los números.

# %%
en_curso = forest.loc[actual["inicio"]:actual["fin"]]
print(f"Días del evento de 2026: {len(en_curso)}   marcados por Isolation Forest: {int(en_curso['atipico'].sum())}")
print(f"Días de toda la serie con anomalía de +1,3 °C o más: {(forest['anomalia'] >= 1.3).mean():.1%}")
print(f"Anomalía media de los días que Isolation Forest marca como atípicos de calor: "
      f"{forest.loc[forest['atipico_calor'], 'anomalia'].mean():.2f} °C")

# %% [markdown]
# **Ninguno de los 39 días del evento de 2026** queda marcado, y casi el 7 % de **todos** los días de la serie tiene una anomalía igual o mayor. Isolation Forest mira la serie completa (1981 a 2026) y le asigna "rareza" a lo extremo en magnitud y a los cambios bruscos (los días más raros de la serie son pulsos de más de 3 °C en diciembre de 1989 o de 2008), no a lo moderado y persistente. Hobday, en cambio, no ve rareza global: ve **persistencia relativa a la estación**.
#
# ¿Y si le pedimos más? Con más contaminación el bosque marca más días, pero a costa de marcar muchos días corrientes como "atípicos".

# %%
filas = []
for contaminacion in (0.05, 0.1, 0.2, 0.3):
    f = aislamiento_forest(sst, clima, contaminacion=contaminacion)
    filas.append({
        "contaminación": contaminacion,
        "días atípicos": int(f["atipico"].sum()),
        "Jaccard con Hobday (calor)": round(solapamiento(marca_calor, f["atipico_calor"])["jaccard"], 2),
        "días marcados en el evento de 2026": int(f.loc[actual["inicio"]:actual["fin"], "atipico"].sum()),
    })
pd.DataFrame(filas).set_index("contaminación")

# %% [markdown]
# Con 20 % de contaminación el bosque empieza a marcar el evento de 2026, pero ya está llamando "atípico" a uno de cada cinco días: el Jaccard con Hobday mejora poco (de 0,32 a 0,37). La contaminación es un parámetro que **hay que elegir** y que no tiene una respuesta correcta desde los datos; en Hobday, el equivalente (el percentil 90) tiene una definición acordada por la comunidad.
#
# Los días que Isolation Forest considera más raros del registro:

# %%
forest.nsmallest(6, "puntaje")[["anomalia", "anomalia_movil", "cambio", "puntaje"]].round(2)

# %% [markdown]
# ### Umbrales clásicos frente a aprendizaje automático: qué significa cada uno
#
# | | Hobday (umbrales clásicos) | Isolation Forest (aprendizaje automático) |
# |---|---|---|
# | Qué detecta | Periodos de 5 días o más sobre el percentil 90 (o bajo el 10) de cada día del año | Días con combinaciones raras de anomalía, tendencia a 7 días y cambio |
# | Estacionalidad | Integrada (un umbral por día del año) | No la conoce: hay que quitarla antes (anomalías) |
# | Duración | Regla explícita (5 días) | No tiene noción de evento: marca días sueltos |
# | Parámetros | Percentil, duración, brecha: convenciones de la comunidad | Contaminación, árboles, características: decisiones del analista |
# | Comparable entre estudios | Sí, es un estándar | Solo si comparten todas las decisiones |
# | Explicación | "Estuvo 12 días sobre el p90" | "Su puntaje de aislamiento fue bajo" |
# | Cuándo conviene | Cuando existe un estándar y se necesita comunicar | Para explorar, descubrir comportamientos raros o variables sin estándar |
#
# Ninguno es "mejor": responden preguntas distintas. Un buen flujo de trabajo usa Hobday para **definir** olas de calor y Isolation Forest (u otros métodos) para **explorar** lo que ese estándar no captura, como los cambios bruscos de pocos días.

# %% [markdown]
# ### Una nota sobre `marEx`
#
# [`marEx`](https://github.com/wienkers/marEx) es un paquete moderno para identificar y rastrear olas de calor marinas en 3D (espacio y tiempo), pensado para mallas grandes y datos globales. No lo usamos como núcleo de este bloque porque:
#
# * **No tiene modo de punto**: trabaja sobre mallas, no sobre una serie en un píxel.
# * **No aplica la regla de los 5 días** de Hobday.
# * **No detecta olas de frío**.
# * Necesita mucha memoria con mallas medianas y declara soporte solo hasta **Python 3.12**, mientras que Colab usa Python 3.13.
#
# Es una buena herramienta si en el futuro analizas el rastreo de olas de calor a escala de cuenca oceánica. Para este taller, el método de Hobday en pandas es el adecuado: lo entiendes línea a línea y lo puedes validar.

# %% [markdown]
# ### Tu turno 3: otro píxel
#
# Todo lo anterior fue para **un** píxel pegado a la costa. Elige otro de la malla (por ejemplo, uno en mar abierto, a 75,375 O) y repite la detección. Cambia `LAT` y `LON` y compara con Concepción: ¿cuántos eventos tiene? ¿Aparece la ola de calor en curso?

# %%
malla = cargar_oisst("malla")
LAT, LON = -36.875, -75.375            # mar abierto, 2 grados al oeste de Concepción
otro = malla.sel(lat=LAT, lon=LON, method="nearest").to_series()
otro.name = "sst_c"
clima_otro = climatologia_hobday(otro)
calor_otro, frio_otro = detectar_eventos(otro, clima_otro, "calor"), detectar_eventos(otro, clima_otro, "frio")
print(f"Píxel {float(malla.sel(lat=LAT, lon=LON, method='nearest').lat):.3f} / {float(malla.sel(lat=LAT, lon=LON, method='nearest').lon):.3f}:"
      f" {len(calor_otro)} olas de calor y {len(frio_otro)} olas de frío (Concepción: {len(olas_calor)} y {len(olas_frio)})")
print("Ola de calor en curso:", evento_en_curso(calor_otro) is not None)

# %% [markdown]
# ## 8. Mapa de frecuencia de eventos
#
# Hasta ahora miramos un píxel. `frecuencia_eventos` repite el mismo método (climatología 1991 a 2020, p90 o p10, 5 días y brechas de 2) en **cada píxel de mar de la malla** de 34 a 40 S y 71 a 76 O. Tarda alrededor de 25 segundos por tipo de evento.
#
# Ojo con la comparación: como el vacío afecta a todos los píxeles por igual, la métrica `eventos_por_decada` divide por los días con dato, así no penaliza por el vacío.

# %%
frec_calor = frecuencia_eventos(malla, "calor")
frec_frio = frecuencia_eventos(malla, "frio")
print("Píxeles de mar:", int(frec_calor["n_eventos"].notnull().sum()))
print(f"Eventos de calor por década observada: de {float(frec_calor['eventos_por_decada'].min()):.1f} a {float(frec_calor['eventos_por_decada'].max()):.1f}")
print(f"Eventos de frío por década observada:  de {float(frec_frio['eventos_por_decada'].min()):.1f} a {float(frec_frio['eventos_por_decada'].max()):.1f}")

# %%
fig, ejes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, frec, titulo, cmap in ((ejes[0], frec_calor, "Olas de calor", "Reds"), (ejes[1], frec_frio, "Olas de frío", "Blues")):
    imagen = ax.pcolormesh(frec["lon"], frec["lat"], frec["eventos_por_decada"], cmap=cmap, shading="nearest")
    ax.set_facecolor("#d5d8dc")                                   # gris: tierra (sin dato)
    ax.plot(-73.375, -36.875, marker="*", color="black", ms=14, mec="white", label="píxel de Concepción")
    ax.set_title(f"{titulo} (eventos por década)")
    ax.set_xlabel("longitud")
    ax.set_aspect(1 / np.cos(np.deg2rad(37)))
    plt.colorbar(imagen, ax=ax, shrink=0.75)
ejes[0].set_ylabel("latitud")
ejes[0].legend(loc="lower left")
plt.tight_layout()
plt.show()

# %% [markdown]
# El gris es **tierra** (sin dato): se distingue claramente la costa chilena en el borde este de la malla. Como cada píxel usa su propio percentil 90 (o 10), por construcción cada píxel tiene aproximadamente el mismo 10 % de días sobre su umbral. Lo que cambia entre píxeles no es *cuánto* se excede, sino **cuántos eventos distintos** se forman: un píxel donde las anomalías persisten forma pocos eventos largos y otro, donde fluctúan rápido, forma más eventos cortos.
#
# ### Costa frente a mar abierto
#
# Calculemos qué tan lejos de la costa (en columnas de 0,25 grados) está cada píxel de mar, y resumamos la frecuencia por distancia.

# %%
mar = malla.isel(time=0).notnull().to_numpy()                       # True en los píxeles de mar
lejania = np.full(mar.shape, np.nan)
for i in range(mar.shape[0]):
    columnas_mar = np.flatnonzero(mar[i])
    if columnas_mar.size:
        lejania[i, columnas_mar] = (columnas_mar.max() - columnas_mar) * 0.25 * 111 * np.cos(np.deg2rad(37))   # km al píxel de costa

distancia = pd.DataFrame({
    "km_a_la_costa": lejania.ravel(),
    "calor": frec_calor["eventos_por_decada"].to_numpy().ravel(),
    "frio": frec_frio["eventos_por_decada"].to_numpy().ravel(),
    "dias_calor": frec_calor["dias_en_evento"].to_numpy().ravel(),
    "dias_frio": frec_frio["dias_en_evento"].to_numpy().ravel(),
}).dropna()
distancia["franja"] = pd.cut(distancia["km_a_la_costa"], [-1, 1, 50, 110, 1000],
                             labels=["píxel de costa", "hasta 50 km", "50 a 110 km", "más de 110 km"])
resumen_franja = distancia.groupby("franja", observed=True).agg(
    pixeles=("calor", "size"), calor_por_decada=("calor", "mean"), frio_por_decada=("frio", "mean"),
    dias_calor=("dias_calor", "mean"), dias_frio=("dias_frio", "mean")).round(1)
resumen_franja

# %% [markdown]
# **Cómo leerlo.**
#
# * Las **olas de frío** son algo más frecuentes en los píxeles de costa que en el mar abierto: unas 25 por década frente a unas 22 (una diferencia de alrededor de 15 %). Es coherente con el **afloramiento costero** de la zona centro-sur de Chile, donde los vientos del sur empujan el agua superficial mar afuera y sube agua fría desde abajo, con pulsos de pocos días que enfrían la costa.
# * Las **olas de calor** no muestran diferencia entre la costa y el mar abierto (unas 20 por década en todas las franjas).
# * **Cuidado con la sobreinterpretación.** La diferencia costa-mar abierto del frío es del mismo orden que la variación entre píxeles vecinos (en el mapa, de 19 a 30 eventos por década): con 40 años y un vacío, el conteo de eventos es ruidoso. Además OISST tiene píxeles de 25 km, y los de costa mezclan tierra, mar y menos observaciones directas. Lo que vemos es un patrón general que sugiere una hipótesis (el afloramiento), no una demostración, y no la variabilidad de la bahía de Concepción.
#
# ### Tu turno 4: otra mirada al mapa
#
# 1. Cambia el periodo en que cuentas los eventos con `frecuencia_eventos(malla, "calor", periodo=(2010, 2026))`. ¿El patrón cambia en los últimos 15 años?
# 2. Mapea `dias_en_evento` en vez de `eventos_por_decada`: ¿coinciden las zonas con más eventos con las de más días en evento?
# 3. Para comunicar el resultado: ¿cuál de los dos mapas (de calor o de frío) le mostrarías a una persona que trabaja en pesca artesanal de la zona y por qué?

# %%
# Punto de partida: otro periodo (tarda unos 25 segundos por tipo)
frec_reciente = frecuencia_eventos(malla, "calor", periodo=(2010, 2026))
print(f"Eventos de calor por década observada desde 2010: media de la malla {float(frec_reciente['eventos_por_decada'].mean()):.1f} "
      f"(toda la serie: {float(frec_calor['eventos_por_decada'].mean()):.1f})")

# %% [markdown]
# ## 9. Guardar los resultados
#
# Los eventos detectados son un **dato derivado**: si no los guardamos con su documentación, nadie (ni tú en tres meses) sabrá con qué parámetros salieron. Guardamos una tabla en parquet con los metadatos dentro del archivo, igual que hicimos con la serie de Sentinel-2. En los bloques 8 y 9 lo versionaremos y lo publicaremos.

# %%
import tempfile

from utils.series import guardar_series, leer_metadatos_series

SALIDAS = Path("/content/salidas") if "google.colab" in sys.modules else Path(tempfile.gettempdir()) / "salidas_taller"
SALIDAS.mkdir(parents=True, exist_ok=True)

eventos = pd.concat([olas_calor, olas_frio], ignore_index=True)
ruta_eventos = SALIDAS / "eventos_oisst_concepcion.parquet"
guardar_series(eventos, ruta_eventos, metadatos={
    "titulo": "Olas de calor y de frío marinas frente a Concepción (Hobday, OISST 1981-2026)",
    "fuente": "NOAA OISST v2.1 leído desde Microsoft Planetary Computer, píxel -36.875 / -73.375",
    "licencia": "Dominio público (datos de NOAA)",
    "periodo": "1981-09-01 a 2026-09-24",
    "metodo": "Hobday et al. 2016, categorías Hobday et al. 2018; implementación utils/extremos.py, validada con xmhw",
    "parametros": {"base": "1991-2020", "ventana_dias": 11, "suavizado_dias": 31, "percentil_calor": 90,
                   "percentil_frio": 10, "duracion_minima_dias": 5, "brecha_maxima_dias": 2},
    "vacios": "2024-06-20 a 2026-03-05 y 2026-03-07/08 sin dato en Planetary Computer; no se rellenaron ni se puentean",
    "columnas": {"tipo": "calor o frio", "inicio": "primer día del evento", "fin": "último día",
                 "duracion": "días, incluye brechas unidas", "dia_pico": "día de mayor anomalía",
                 "intensidad_max": "°C sobre la media climatológica (negativa en frío)",
                 "intensidad_media": "°C", "intensidad_acumulada": "°C-día",
                 "categoria": "1 moderada a 4 extrema (Hobday 2018)", "en_curso": "True si llega al último día de la serie"},
})
print(f"Guardado: {ruta_eventos}  ({ruta_eventos.stat().st_size / 1024:.0f} KB, {len(eventos)} eventos)")
print(list(leer_metadatos_series(ruta_eventos)))

# %% [markdown]
# ## Resumen
#
# * **OISST** es un producto diario de 0,25 grados, de satélites, boyas y barcos, interpolado y sin huecos por nubes. Ve la costa y el mar abierto, no el interior de la bahía.
# * La serie tiene un **vacío** de 2024-06-20 a 2026-03-05 (y dos días más). Afecta lo que se puede detectar, no la climatología. **Interpolar inventa eventos, rellenar con la media diluye, recortar pierde el presente**: lo honesto es ignorar, reportar y normalizar por días observados.
# * El método de Hobday: climatología 1991-2020 con ventana de 11 días y suavizado de 31, umbrales p90 y p10, rachas de 5 días o más, y unión de brechas de 2 días o menos (**primero filtrar, después unir**; nunca a través de un vacío).
# * En el píxel de Concepción: **88 olas de calor y 111 de frío**; las mayores de calor fueron 1997, 1998, 2017 y 1982, y hay una **ola de calor en curso desde 2026-08-17**.
# * El resultado **coincide con `xmhw`** (salvo el primer día de la serie y dos categorías, por una definición distinta del día del pico).
# * **Isolation Forest** marca días raros en toda la serie, no eventos persistentes: no ve la ola de 2026 porque +1,3 °C no es raro. Umbrales clásicos y aprendizaje automático responden preguntas distintas.
# * En el mapa, la frecuencia de **olas de frío es algo mayor en la costa** (coherente con el afloramiento, pero con ruido), y la de calor no cambia con la distancia a la costa.
#
# ### Para el siguiente bloque
#
# En el bloque 8 refactorizaremos este código en funciones, lo versionaremos con Git y lo subiremos a GitHub. Lo que guardaste en esta sesión (`eventos_oisst_concepcion.parquet`) es un buen candidato para publicar en el bloque 9.
#
# **Referencias**
#
# * Hobday, A. J., et al. (2016). A hierarchical approach to defining marine heatwaves. *Progress in Oceanography*, 141, 227-238.
# * Hobday, A. J., et al. (2018). Categorizing and naming marine heatwaves. *Oceanography*, 31(2), 162-173.
# * Huang, B., et al. (2021). Improvements of the Daily Optimum Interpolation Sea Surface Temperature (DOISST) Version 2.1. *Journal of Climate*, 34, 2923-2939.
# * Liu, F. T., Ting, K. M. y Zhou, Z.-H. (2008). Isolation Forest. *IEEE International Conference on Data Mining*.
