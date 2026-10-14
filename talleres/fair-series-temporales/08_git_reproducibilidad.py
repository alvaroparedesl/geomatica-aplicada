# %% [markdown]
# # Git, reproducibilidad y cierre del flujo de trabajo
#
# [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvaroparedesl/geomatica-aplicada/blob/main/talleres/fair-series-temporales/08_git_reproducibilidad.ipynb) [![Licencia: GPL-3.0](https://img.shields.io/badge/licencia-GPL--3.0-blue.svg)](https://github.com/alvaroparedesl/geomatica-aplicada/blob/main/LICENSE) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://colab.research.google.com/)
#
# **Bloque 8 del taller "Datos FAIR, series temporales satelitales y detección de anomalías con Python"** (unos 65 minutos).
#
# Llevas dos días construyendo un análisis: buscaste imágenes, armaste series, calculaste anomalías y detectaste eventos extremos. Falta lo que hace que ese trabajo **valga más allá de tu sesión de Colab**: que alguien (tú dentro de seis meses, una colega, una revisora) pueda ver **qué hiciste, en qué orden y por qué**, y volver a ejecutarlo y obtener lo mismo. Eso se llama **trazabilidad** y **reproducibilidad**, y la herramienta estándar es **Git**.
#
# En este cuaderno vas a:
#
# 1. Repasar los comandos esenciales de Git (`init`, `add`, `commit`, `push`, `pull`, `branch`, `log`, `diff`).
# 2. Crear un **token de acceso personal de grano fino** en GitHub y guardarlo como Secreto de Colab, **sin que aparezca nunca en una celda, en una salida ni en `.git/config`**.
# 3. **Refactorizar** el código de los cuadernos anteriores en funciones dentro de un módulo `.py`.
# 4. Hacer **al menos tres commits descriptivos**, subirlos a GitHub y **verificar que el proyecto se ejecuta de principio a fin desde cero**.
#
# | Parte | Minutos |
# |---|---|
# | Verificar el token y poder hacer `git push` | 10 |
# | Comandos de Git (práctica local) | 10 |
# | Refactorizar en funciones | 25 |
# | Commits, push y verificación | 20 |
#
# **Importante.** Dedica los primeros diez minutos a la parte del token (sección 2). Si no logras que funcione en ese tiempo, hay un plan de respaldo sin token (sección 8) y sigues con el resto del bloque.

# %% [markdown]
# ## Configuración del entorno

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

# %% [markdown]
# ## 1. Git en cinco ideas
#
# **Git** es un sistema de control de versiones: guarda la **historia** de una carpeta de archivos como una secuencia de fotografías (los *commits*), cada una con autor, fecha y un mensaje que explica el cambio. **GitHub** es un servicio web que guarda una copia remota de esa historia para compartirla.
#
# | Concepto | Qué es |
# |---|---|
# | **Repositorio** | Una carpeta con su historia (el directorio oculto `.git/`) |
# | **Commit** | Una fotografía de los cambios que elegiste guardar, con un mensaje |
# | **Área de preparación** (*staging*) | Dónde eliges qué cambios entran en el próximo commit (`git add`) |
# | **Rama** (*branch*) | Una línea de trabajo paralela, para probar algo sin romper lo que funciona |
# | **Remoto** (*remote*) | Una copia del repositorio en otro lugar (por ejemplo GitHub), con nombre, normalmente `origin` |
#
# | Comando | Para qué sirve |
# |---|---|
# | `git init` | Crear un repositorio en la carpeta actual |
# | `git status` | Ver qué cambió y qué está preparado |
# | `git add archivo` | Preparar un cambio para el próximo commit |
# | `git commit -m "mensaje"` | Guardar los cambios preparados en la historia |
# | `git log` | Ver la historia de commits |
# | `git diff` | Ver las líneas que cambiaron (antes de guardarlas) |
# | `git branch`, `git switch` | Crear o cambiar de rama |
# | `git push` | Subir tus commits a un remoto |
# | `git pull` | Traer los commits nuevos de un remoto |
# | `git clone` | Copiar un repositorio completo (con su historia) |
#
# **¿Por qué importa para la ciencia?** Porque cada commit responde "¿qué cambió, cuándo, quién y por qué?". Si el resultado de un análisis cambió entre marzo y junio, `git log` y `git diff` te dicen en qué línea. Es la **trazabilidad**, y es parte de la R de FAIR: *Reusable* exige que los datos y el código tengan una procedencia clara.

# %% [markdown]
# ## 2. Un token para subir a GitHub, sin exponerlo
#
# Para que Colab pueda hacer `git push` a **tu** repositorio, GitHub necesita saber que eres tú. Se hace con un **token de acceso personal** (PAT, *personal access token*): una contraseña de uso específico que puedes limitar y revocar. Un token es **tan secreto como una contraseña**: quien lo tenga puede escribir en tus repositorios.
#
# ### 2.1 Crear el repositorio y el token (en la web de GitHub)
#
# 1. En GitHub crea un **repositorio nuevo y vacío** (por ejemplo `analisis-taller`). **No marques** "Add a README" ni licencia: queremos que parta sin ningún commit, para que el primer `push` no tenga conflictos.
# 2. Ve a **Settings > Developer settings > Personal access tokens > Fine-grained tokens > Generate new token**.
# 3. Configúralo así:
#     * **Token name:** `taller-colab`
#     * **Expiration:** **7 días** (la más corta que te sirva; el token vence solo)
#     * **Repository access:** **Only select repositories** y elige **solo** el repositorio del taller
#     * **Repository permissions:** **Contents: Read and write** (el resto, sin acceso; *Metadata: Read-only* se agrega solo)
# 4. Pulsa **Generate token** y **copia el valor ahora**: GitHub no lo vuelve a mostrar.
#
# Un token de grano fino limitado a un repositorio y con vencimiento corto es la opción segura: si se filtra, el daño se limita a un repositorio y a una semana.
#
# ### 2.2 Guardarlo como Secreto de Colab
#
# 1. En Colab, abre el panel izquierdo y pulsa el icono de la **llave** (**Secretos**).
# 2. **Agregar secreto nuevo**. **Nombre:** `GITHUB_TOKEN` (exactamente así). **Valor:** pega el token.
# 3. Activa el interruptor **Acceso desde el cuaderno**.
#
# **Reglas que no se negocian:**
#
# * El token **nunca se escribe en una celda de código**, ni siquiera "para probar". Los cuadernos se guardan y se comparten con sus celdas y salidas.
# * **Nunca se imprime** (`print(token)`), ni se muestra en una salida.
# * **Nunca se escribe en `.git/config`**: eso pasa si pones el token dentro de la URL del remoto (`https://TOKEN@github.com/...`), que es lo que muchos tutoriales recomiendan. Más abajo lo mostramos y lo evitamos.
# * Si crees que se filtró, **revócalo de inmediato** en GitHub (Settings > Developer settings) y crea otro.

# %%
# @title Datos de tu repositorio (completa estos campos)
USUARIO_GITHUB = "tu-usuario"   # @param {type:"string"}
REPO_GITHUB = "analisis-taller"  # @param {type:"string"}
NOMBRE = "Participante del taller"   # @param {type:"string"}
CORREO = "participante@example.com"  # @param {type:"string"}
print(f"Repositorio: https://github.com/{USUARIO_GITHUB}/{REPO_GITHUB}")

# %%
# @title Leer el Secreto GITHUB_TOKEN (sin mostrarlo)
import os


def leer_secreto(nombre):
    """El valor de un Secreto de Colab (o de una variable de entorno fuera de Colab). None si no existe."""
    try:
        from google.colab import userdata

        valor = userdata.get(nombre)
        if valor:
            return valor
    except Exception:  # fuera de Colab, o el secreto no existe, o no tiene acceso desde el cuaderno
        pass
    return os.environ.get(nombre) or None


TOKEN = leer_secreto("GITHUB_TOKEN")
CONFIGURADO = TOKEN is not None and USUARIO_GITHUB != "tu-usuario"

if TOKEN is None:
    print("No encuentro el Secreto GITHUB_TOKEN.\n"
          "  1. Abre el panel de Secretos (icono de llave, a la izquierda).\n"
          "  2. Agrega un secreto llamado GITHUB_TOKEN con tu token de GitHub.\n"
          "  3. Activa 'Acceso desde el cuaderno' y vuelve a ejecutar esta celda.\n"
          "Mientras tanto, las partes locales de este cuaderno funcionan igual; las celdas que suben a GitHub se saltarán.")
elif USUARIO_GITHUB == "tu-usuario":
    print("Encontré el token, pero falta tu usuario de GitHub: completa USUARIO_GITHUB en la celda anterior.")
else:
    print(f"Token encontrado ({len(TOKEN)} caracteres). No se muestra.")

# %% [markdown]
# ### 2.3 Verificar el token (sin tocar Git todavía)
#
# Antes de usar el token con Git, comprobamos tres cosas con la API de GitHub: que **funciona**, que tiene permiso de **escritura en tu repositorio** y **cuándo vence**. El token viaja en el encabezado `Authorization` de la petición y no se imprime.

# %%
import requests


def verificar_token(token, usuario, repo):
    """Consulta la API de GitHub: ¿el token sirve, puede escribir en el repositorio y cuándo vence?"""
    respuesta = requests.get(f"https://api.github.com/repos/{usuario}/{repo}",
                             headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, timeout=30)
    vence = respuesta.headers.get("github-authentication-token-expiration", "no informado")
    if respuesta.status_code == 200:
        puede_escribir = respuesta.json().get("permissions", {}).get("push", False)
        print(f"Repositorio encontrado: {respuesta.json()['full_name']} ({'privado' if respuesta.json()['private'] else 'público'})")
        print(f"Permiso de escritura (push): {puede_escribir}")
        print(f"El token vence: {vence}")
        return bool(puede_escribir)
    print(f"GitHub respondió {respuesta.status_code}: {respuesta.json().get('message', '')}")
    if respuesta.status_code in (401, 403):
        print("El token no es válido, venció o no tiene acceso a ese repositorio (revisa 'Repository access').")
    elif respuesta.status_code == 404:
        print("No encuentro ese repositorio con este token: revisa USUARIO_GITHUB y REPO_GITHUB, "
              "y que el token incluya ese repositorio.")
    return False


if CONFIGURADO:
    PUEDE_ESCRIBIR = verificar_token(TOKEN, USUARIO_GITHUB, REPO_GITHUB)
else:
    PUEDE_ESCRIBIR = False
    print("Verificación omitida: falta el token o el usuario (ver las celdas anteriores).")

# %% [markdown]
# Si ves `Permiso de escritura (push): True`, el token está bien y puedes seguir. Si no, es el momento de pedir ayuda: es la parte que más tiempo toma si se deja para el final.

# %% [markdown]
# ## 3. Cómo usar el token sin dejarlo escrito en ninguna parte
#
# Ahora la parte delicada: hacer `git push` **con** el token pero **sin** que quede en `.git/config`. Primero mira por qué el camino fácil es una mala idea. Usaremos un repositorio de práctica y un valor **falso** (no es ningún token real) para ver qué pasa.

# %%
import shutil
import subprocess
import tempfile
from pathlib import Path


def git(*args, cwd, entorno=None, mostrar=True, revisar=True):
    """Ejecuta `git <args>` en `cwd` y devuelve su salida de texto. Imprime el comando y la salida."""
    proceso = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main", *args],
                             cwd=cwd, env=entorno, capture_output=True, text=True)
    salida = (proceso.stdout + proceso.stderr).strip()
    if mostrar:
        print(f"$ git {' '.join(args)}")
        if salida:
            print(salida)
    if revisar and proceso.returncode != 0:
        raise RuntimeError(f"git {args[0]} falló (código {proceso.returncode})")
    return salida


TALLER = Path(tempfile.mkdtemp(prefix="taller_git_"))     # carpeta de trabajo temporal de este cuaderno
demo = TALLER / "demo_filtracion"
demo.mkdir()
git("init", "-q", cwd=demo)
# MAL: el token dentro de la URL del remoto (aquí, un valor falso)
git("remote", "add", "origin", "https://TOKEN_FALSO_DE_EJEMPLO@github.com/usuario/repo.git", cwd=demo, mostrar=False)
print((demo / ".git" / "config").read_text())

# %% [markdown]
# Mira la línea `url = https://TOKEN_FALSO_DE_EJEMPLO@github.com/...`: el secreto quedó **escrito en un archivo de texto plano**. Si ese repositorio se comprime, se sube a Drive o se comparte, el token viaja con él. Además aparece en `git remote -v` y en los mensajes de error.
#
# **La forma segura** es que Git le **pida** las credenciales a un programa auxiliar cada vez que las necesita, y que ese auxiliar las lea de una **variable de entorno que existe solo durante ese comando**. Eso es `GIT_ASKPASS`:
#
# 1. Escribimos un script diminuto que **no contiene el token**: solo responde con el usuario o con el contenido de una variable de entorno.
# 2. Ejecutamos `git push` pasándole esa variable **solo a ese proceso** (no al cuaderno).
# 3. Usamos `-c credential.helper=` para que ningún auxiliar de credenciales guarde nada en disco.
# 4. Al terminar borramos el script y comprobamos `.git/config`.
#
# El remoto queda con la URL **limpia** (`https://github.com/usuario/repo.git`).

# %%
import re


def git_con_token(*args, cwd, token):
    """
    Ejecuta un comando de git que necesita credenciales (push, pull, ls-remote) sin dejar el token en disco.

    - El token viaja en una variable de entorno solo para este proceso, no en la URL ni en los argumentos.
    - GIT_ASKPASS apunta a un script que NO contiene el token: lo lee de esa variable.
    - `credential.helper=` vacío evita que se guarde en ~/.git-credentials u otro almacén.
    - La salida se limpia por si el token apareciera en algún mensaje.
    """
    carpeta = Path(tempfile.mkdtemp(prefix="askpass_"))
    auxiliar = carpeta / "askpass.sh"
    auxiliar.write_text('#!/bin/sh\ncase "$1" in\n  Username*) echo "x-access-token" ;;\n  *) echo "$TOKEN_GIT_UNICO" ;;\nesac\n')
    auxiliar.chmod(0o700)
    entorno = {**os.environ, "GIT_ASKPASS": str(auxiliar), "GIT_TERMINAL_PROMPT": "0", "TOKEN_GIT_UNICO": token}
    try:
        proceso = subprocess.run(["git", "-c", "credential.helper=", *args], cwd=cwd, env=entorno,
                                 capture_output=True, text=True, timeout=120)
    finally:
        shutil.rmtree(carpeta, ignore_errors=True)       # el auxiliar no queda en disco
    salida = (proceso.stdout + proceso.stderr).replace(token, "***").strip()
    print(f"$ git {' '.join(args)}")
    if salida:
        print(salida)
    return proceso.returncode


def verificar_config_limpia(repo, token=None):
    """¿Quedó algún secreto en .git/config o en un almacén de credenciales? Devuelve True si está limpio."""
    texto = (Path(repo) / ".git" / "config").read_text()
    problemas = []
    if token and token in texto:
        problemas.append("el valor del token aparece en .git/config")
    if re.search(r"https?://[^/\s@]+@", texto):
        problemas.append("hay una URL con credenciales (usuario@ o token@) en .git/config")
    if re.search(r"github_pat_|ghp_[A-Za-z0-9]", texto):
        problemas.append("hay algo con forma de token de GitHub en .git/config")
    if re.search(r"credential\.helper|helper\s*=\s*(store|cache)", texto):
        problemas.append("hay un auxiliar de credenciales configurado en el repositorio")
    if (Path.home() / ".git-credentials").exists():
        problemas.append("existe ~/.git-credentials (revisa que no contenga tu token)")
    if problemas:
        print("CON PROBLEMAS: " + "; ".join(problemas))
    else:
        print(".git/config limpio: sin token, sin credenciales en las URL, sin auxiliares de credenciales.")
    return not problemas


print("El repositorio de la demostración (con el valor falso):")
verificar_config_limpia(demo, token="TOKEN_FALSO_DE_EJEMPLO")
shutil.rmtree(demo)    # borramos el repositorio de la demostración

# %% [markdown]
# La comprobación detecta el secreto escrito en la URL. Más abajo la usaremos con el repositorio de verdad, **después** del push.
#
# ¿Cómo sabemos que `git_con_token` funciona? Lo probamos más adelante con tu repositorio. Fíjate en lo que **no** hace: el token nunca aparece en el comando (`git push github main`), que es lo que se ve en pantalla, en los mensajes de error ni en el repositorio.

# %% [markdown]
# ## 4. Práctica local: crear un repositorio y hacer commits
#
# Ahora trabajamos con Git de verdad. Para poder ejecutar el cuaderno de principio a fin en cualquier lugar, el repositorio de práctica vive en una carpeta temporal. En Colab será tu carpeta de trabajo; el paso final lo subirá a GitHub.
#
# Empezamos con el **código de los cuadernos anteriores**, tal como suele quedar: un script plano que va de arriba abajo (cargar la serie de temperatura de Landsat, detectar episodios, calcular anomalías de Sentinel-2). Los parquet se leen con `pd.read_parquet`, como en los bloques anteriores; `abrir_dato` solo busca el archivo en la carpeta `datos/` del curso (o lo descarga de GitHub si no está), porque el script vive en tu proyecto y no junto a los datos.

# %%
PROYECTO = TALLER / "analisis-taller"
PROYECTO.mkdir()

git("init", "-q", "-b", "main", cwd=PROYECTO)
git("config", "user.name", NOMBRE, cwd=PROYECTO, mostrar=False)       # identidad solo para este repositorio
git("config", "user.email", CORREO, cwd=PROYECTO, mostrar=False)
git("status", "--short", "--branch", cwd=PROYECTO)

# %%
SCRIPT_V1 = '''"""Análisis del taller, versión 1: un script plano de arriba hacia abajo."""
import pandas as pd

from utils.datos_taller import abrir_dato
from utils.extremos import climatologia_ventana, controlar_calidad_agua, detectar_episodios
from utils.series import anomalias_zscore

landsat = pd.read_parquet(abrir_dato("series_landsat_concepcion.parquet"))
landsat["fecha"] = landsat["fecha"].dt.normalize()  # solo importa el día, no la hora de la toma
landsat = controlar_calidad_agua(landsat)
st = landsat[landsat["id"] == "bahia_concepcion"].set_index("fecha")["st_c"].astype(float)
clima = climatologia_ventana(st)
calor = detectar_episodios(st, clima, "calor")
frio = detectar_episodios(st, clima, "frio")
print("Episodios de calor:", len(calor), "| Episodios de frío:", len(frio))

tabla = pd.read_parquet(abrir_dato("series_sentinel2_concepcion.parquet"))
serie = tabla[tabla["id"] == "bahia_concepcion"].set_index("fecha")["ndwi"].dropna()
z = anomalias_zscore(serie, referencia="total")
print("Fechas con |z| > 2:", int((z.abs() > 2).sum()), "de", len(z))
'''
(PROYECTO / "analisis.py").write_text(SCRIPT_V1)

git("status", "--short", cwd=PROYECTO)

# %% [markdown]
# `??` significa que Git ve el archivo pero **todavía no lo sigue**. Para guardarlo en la historia son dos pasos: `git add` lo prepara y `git commit` lo guarda con un mensaje.
#
# **Un buen mensaje de commit** responde *qué* cambió y *por qué*, en una línea corta, en el modo imperativo o descriptivo, empezando con un tipo: `feat:` (algo nuevo), `fix:` (corrección), `refactor:` (reorganizar sin cambiar el resultado), `docs:` (documentación). Evita "cambios", "arreglo" o "final2".

# %%
git("add", "analisis.py", cwd=PROYECTO)
git("commit", "-q", "-m", "feat: script plano con episodios de Landsat y anomalías de Sentinel-2", cwd=PROYECTO)
git("log", "--oneline", cwd=PROYECTO)

# %% [markdown]
# ## 5. Refactorizar: del script a funciones
#
# El script funciona, pero tiene tres problemas: no se puede reutilizar una parte sin ejecutar todo, no se puede probar cada paso por separado y los parámetros (el punto, la variable) están enterrados en el código. **Refactorizar** es reorganizar el código **sin cambiar lo que hace**: convertimos los bloques en funciones con nombres que dicen qué hacen y las ponemos en un módulo `.py` que cualquier cuaderno puede importar.
#
# Renombramos con `git mv` (así Git entiende que es el mismo archivo y conserva su historia) y reescribimos.

# %%
git("mv", "analisis.py", "analisis_taller.py", cwd=PROYECTO)

SCRIPT_V2 = '''"""Análisis del taller: episodios de temperatura en Landsat y anomalías de Sentinel-2."""
import pandas as pd

from utils.datos_taller import abrir_dato
from utils.extremos import climatologia_ventana, controlar_calidad_agua, detectar_episodios
from utils.series import anomalias_zscore


def episodios_landsat(punto="bahia_concepcion"):
    """Episodios cálidos y fríos de la temperatura de superficie (Landsat 8 y 9) en un punto, en una sola tabla."""
    landsat = pd.read_parquet(abrir_dato("series_landsat_concepcion.parquet"))
    landsat["fecha"] = landsat["fecha"].dt.normalize()  # solo importa el día, no la hora de la toma
    landsat = controlar_calidad_agua(landsat)
    st = landsat[landsat["id"] == punto].set_index("fecha")["st_c"].astype(float)
    clima = climatologia_ventana(st)
    calor = detectar_episodios(st, clima, "calor")
    frio = detectar_episodios(st, clima, "frio")
    return pd.concat([calor, frio], ignore_index=True)


def anomalias_punto(punto="bahia_concepcion", variable="ndwi"):
    """Anomalía estandarizada (z-score) de una variable de Sentinel-2 en un punto."""
    tabla = pd.read_parquet(abrir_dato("series_sentinel2_concepcion.parquet"))
    serie = tabla[tabla["id"] == punto].set_index("fecha")[variable].dropna()
    return anomalias_zscore(serie, referencia="total")


def main():
    episodios = episodios_landsat()
    print("Episodios de calor:", int((episodios["tipo"] == "calor").sum()),
          "| Episodios de frío:", int((episodios["tipo"] == "frio").sum()))
    z = anomalias_punto()
    print("Fechas con |z| > 2:", int((z.abs() > 2).sum()), "de", len(z))


if __name__ == "__main__":
    main()
'''
(PROYECTO / "analisis_taller.py").write_text(SCRIPT_V2)

# %% [markdown]
# Antes de guardar, mira **qué cambió** con `git diff`. Como renombramos y reescribimos, usamos `--staged` después de `git add`, y `-M` para que Git detecte el renombre.

# %%
git("add", "-A", cwd=PROYECTO)
git("diff", "--staged", "-M", "--stat", cwd=PROYECTO)
print()
git("diff", "--staged", "-M", cwd=PROYECTO)

# %% [markdown]
# Las líneas con `-` se quitaron y las con `+` se agregaron. Nota que la **lógica no cambió**, solo cómo está organizada. Ahora el segundo commit.

# %%
git("commit", "-q", "-m", "refactor: separar el análisis en funciones reutilizables (episodios_landsat, anomalias_punto)", cwd=PROYECTO)
git("log", "--oneline", cwd=PROYECTO)

# %% [markdown]
# ### Tercer commit: guardar los resultados
#
# Un análisis reproducible deja sus **resultados escritos en archivos**, no solo impresos en pantalla. Agregamos a `main()` el guardado de las dos tablas en una carpeta `resultados/`. Esa carpeta **no se sube a Git**: son productos que cualquiera puede regenerar con el código, y Git guarda código, no resultados. Se declara en un archivo `.gitignore`.

# %%
SCRIPT_V3 = SCRIPT_V2.replace('''def main():
    episodios = episodios_landsat()''', '''CARPETA_RESULTADOS = Path(__file__).parent / "resultados"


def main():
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    episodios = episodios_landsat()
    episodios.to_csv(CARPETA_RESULTADOS / "episodios_landsat.csv", index=False)''').replace('''    z = anomalias_punto()
''', '''    z = anomalias_punto()
    z.to_csv(CARPETA_RESULTADOS / "anomalias_ndwi_bahia.csv")
''').replace('import pandas as pd\n', 'from pathlib import Path\n\nimport pandas as pd\n', 1)
(PROYECTO / "analisis_taller.py").write_text(SCRIPT_V3)
(PROYECTO / ".gitignore").write_text("resultados/\n__pycache__/\n*.pyc\n")

git("diff", cwd=PROYECTO)

# %%
git("add", "analisis_taller.py", ".gitignore", cwd=PROYECTO)
git("commit", "-q", "-m", "feat: guardar episodios y anomalías en resultados/ (fuera de Git)", cwd=PROYECTO)

# %% [markdown]
# ### Cuarto commit: una persona que no eres tú tiene que poder ejecutarlo
#
# Un `README.md` dice qué hace el proyecto y cómo ejecutarlo. Sin él, el repositorio no es reutilizable aunque el código esté perfecto.

# %%
README = f'''# Análisis del taller FAIR: eventos extremos y anomalías

Código del taller "Datos FAIR, series temporales satelitales y detección de anomalías con Python".

* `analisis_taller.py`: detecta episodios cálidos y fríos de la temperatura de superficie en la bahía de Concepción
  (Landsat 8 y 9, anomalías respecto de una climatología por ventana de días) y calcula la anomalía del NDWI de
  Sentinel-2 en la misma bahía.

## Cómo ejecutarlo

Necesita la carpeta `utils/` del curso (https://github.com/alvaroparedesl/geomatica-aplicada), pandas y scikit-learn.

    git clone https://github.com/alvaroparedesl/geomatica-aplicada
    PYTHONPATH=geomatica-aplicada python analisis_taller.py

Escribe sus resultados en `resultados/` (no se versiona).

## Autoría

{NOMBRE}. Datos: USGS Landsat 8 y 9 Collection 2 Level-2 (dominio público) y ESA Copernicus Sentinel-2 (acceso libre).
'''
(PROYECTO / "README.md").write_text(README)
git("add", "README.md", cwd=PROYECTO)
git("commit", "-q", "-m", "docs: agregar README con instrucciones para reproducir el análisis", cwd=PROYECTO)
git("log", "--oneline", "--stat", cwd=PROYECTO)

# %% [markdown]
# Ya tienes **cuatro commits**: cada uno con un propósito claro, un mensaje que lo explica y un cambio que se puede revisar por separado. Eso es trazabilidad. `git log -p` muestra la historia completa con todos los cambios.

# %% [markdown]
# ### Tu turno 1: ramas
#
# Una **rama** sirve para probar algo sin tocar lo que ya funciona. Prueba el flujo completo:
#
# 1. Crea una rama (`git switch -c prueba-percentil`).
# 2. Cambia algo (por ejemplo, agrega un parámetro `percentil` a `episodios_landsat`) y haz un commit con un mensaje descriptivo.
# 3. Mira la historia (`git log --oneline --graph --all`).
# 4. Vuelve a `main` (`git switch main`) y verifica que el cambio **no está**.
# 5. Si quedaste conforme, incorpóralo (`git merge prueba-percentil`); si no, borra la rama (`git branch -D prueba-percentil`).
#
# Abajo hay una versión resuelta con un cambio mínimo, para que veas la secuencia. Cámbiala por tu propia idea.

# %%
git("switch", "-c", "prueba-percentil", cwd=PROYECTO)
codigo = (PROYECTO / "analisis_taller.py").read_text()
(PROYECTO / "analisis_taller.py").write_text(
    codigo.replace('"""Episodios cálidos y fríos de la temperatura de superficie (Landsat 8 y 9) en un punto, en una sola tabla."""',
                   '"""Episodios cálidos (p90) y fríos (p10) de la temperatura de superficie (Landsat 8 y 9) en un punto, en una sola tabla."""'))
git("commit", "-q", "-am", "docs: aclarar los percentiles en la descripción de episodios_landsat", cwd=PROYECTO)
git("log", "--oneline", "--graph", "--all", cwd=PROYECTO)
git("switch", "main", cwd=PROYECTO)
git("merge", "-q", "prueba-percentil", cwd=PROYECTO)
git("branch", "-d", "prueba-percentil", cwd=PROYECTO)
git("log", "--oneline", cwd=PROYECTO)

# %% [markdown]
# ## 6. Push y pull: compartir la historia
#
# `git push` **sube** commits a un remoto y `git pull` **trae** los que otra persona (o tú desde otro computador) subió. Para practicarlos sin necesitar internet ni token, creamos un remoto **local**: un repositorio "desnudo" (`--bare`, solo la historia, sin carpeta de trabajo), que es lo que es, en esencia, GitHub.

# %%
REMOTO = TALLER / "remoto.git"
git("init", "-q", "--bare", "-b", "main", str(REMOTO), cwd=TALLER)
git("remote", "add", "origin", str(REMOTO), cwd=PROYECTO)
git("push", "-u", "origin", "main", cwd=PROYECTO)
git("remote", "-v", cwd=PROYECTO)

# %% [markdown]
# Ahora simulamos a **una colaboradora**: clona el repositorio, hace un cambio y lo sube. Tú, desde tu carpeta, lo traes con `pull`.

# %%
colaboradora = TALLER / "colaboradora"
git("clone", "-q", str(REMOTO), str(colaboradora), cwd=TALLER)
git("config", "user.name", "Colaboradora", cwd=colaboradora, mostrar=False)
git("config", "user.email", "colaboradora@example.com", cwd=colaboradora, mostrar=False)
with open(colaboradora / "README.md", "a") as f:
    f.write("\n## Cita\n\nSi usas este código, cita el taller y los datos del USGS (Landsat) y de Copernicus (Sentinel-2).\n")
git("commit", "-q", "-am", "docs: agregar sección de cita al README", cwd=colaboradora)
git("push", "-q", "origin", "main", cwd=colaboradora)

print("--- Desde tu carpeta: ¿hay novedades? ---")
git("pull", "origin", "main", cwd=PROYECTO)
git("log", "--format=%h %an: %s", cwd=PROYECTO)

# %% [markdown]
# En la historia ves el commit de la colaboradora con **su** nombre. Así Git registra quién hizo cada cambio.

# %% [markdown]
# ## 7. Subir a GitHub
#
# Ahora el paso con el remoto real. Esta celda necesita el Secreto `GITHUB_TOKEN`, tu usuario y el repositorio vacío de la sección 2. **Si falta algo, la celda te lo dice y no hace nada.**
#
# Fíjate en el orden:
#
# 1. El remoto `github` se agrega con la URL **limpia**.
# 2. `git push` se ejecuta con `git_con_token`: el token solo existe en el entorno de ese proceso.
# 3. Después comprobamos `.git/config`.

# %%
if not CONFIGURADO:
    print("Salto este paso: no hay token o falta tu usuario de GitHub (ver la sección 2).\n"
          "Puedes seguir con el resto del cuaderno. Si no consigues el token, usa el plan de respaldo de la sección 8.")
elif not PUEDE_ESCRIBIR:
    print("Salto este paso: el token no tiene permiso de escritura en el repositorio (ver la verificación de la sección 2).")
else:
    URL = f"https://github.com/{USUARIO_GITHUB}/{REPO_GITHUB}.git"      # URL limpia: sin token
    if "github" not in git("remote", cwd=PROYECTO, mostrar=False).split():
        git("remote", "add", "github", URL, cwd=PROYECTO)
    codigo_salida = git_con_token("push", "-u", "github", "main", cwd=PROYECTO, token=TOKEN)
    if codigo_salida == 0:
        print(f"\nListo: https://github.com/{USUARIO_GITHUB}/{REPO_GITHUB}/commits/main")
        verificar_config_limpia(PROYECTO, token=TOKEN)
        git("remote", "-v", cwd=PROYECTO)
    else:
        print("\nEl push falló. Causas habituales: el repositorio de GitHub no está vacío (creaste un README), "
              "el token no incluye este repositorio o venció. Revisa el mensaje de arriba.")

# %% [markdown]
# **Qué debes ver:** `git remote -v` con una URL sin token, y `.git/config limpio`. Abre el enlace del repositorio en GitHub: ahí están tus commits con sus mensajes.
#
# Comprobemos también desde GitHub que **llegaron todos los commits** (con la API, igual que verificamos el token):

# %%
if CONFIGURADO and PUEDE_ESCRIBIR:
    respuesta = requests.get(f"https://api.github.com/repos/{USUARIO_GITHUB}/{REPO_GITHUB}/commits",
                             headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30)
    if respuesta.ok:
        commits = [c["commit"]["message"].splitlines()[0] for c in respuesta.json()]
        print(f"{len(commits)} commits en GitHub:")
        for mensaje in commits:
            print(" -", mensaje)
    else:
        print("No pude leer los commits:", respuesta.status_code)
else:
    print("Sin token: esta comprobación se omite. En GitHub, abre la pestaña 'Commits' del repositorio.")

# %% [markdown]
# ### Tu turno 2: los mensajes y la historia
#
# 1. Mira tu historia con `git log --oneline`. Si un colega leyera solo esa lista, ¿entendería qué se hizo y en qué orden?
# 2. Escribe un mensaje de commit para cada cambio de esta lista, con el tipo y una línea descriptiva: (a) corregiste que el percentil del frío estaba en 90 en vez de 10; (b) agregaste un gráfico de los eventos; (c) cambiaste la base de la climatología de 1981-2010 a 1991-2020.
# 3. Ejecuta `git log -p -1` y busca en la salida el momento exacto en que cambió una línea.

# %% [markdown]
# ## 8. Plan de respaldo: "Guardar una copia en GitHub"
#
# Si el token no funcionó en los primeros diez minutos, **no te quedes atrás**. Colab puede subir el cuaderno a GitHub sin token, con su propia integración:
#
# 1. Menú **Archivo > Guardar una copia en GitHub**.
# 2. La primera vez, Colab te pide **autorizar** la aplicación de Colab en tu cuenta de GitHub. Elige a qué repositorios le das acceso (puede ser solo uno).
# 3. Elige el **repositorio**, la **rama** (`main`), la **ruta del archivo** (por ejemplo `cuadernos/analisis.ipynb`) y escribe un **mensaje de commit descriptivo**.
# 4. Marca **"Incluir un enlace de Colab"** si quieres el botón de "Abrir en Colab" en el repositorio.
# 5. Aceptar. Es un commit más en tu historia, con tu mensaje.
#
# Diferencias con el flujo del token: **no hay `.git/config`** ni línea de comandos, sube **un archivo a la vez** (el cuaderno, no el módulo `.py` ni la carpeta completa) y no puedes hacer ramas ni `pull`. Sirve para **guardar y compartir**, no para trabajar con la historia. Para el resto del bloque usa el repositorio local de este cuaderno.

# %% [markdown]
# ## 9. Verificar que el proyecto se ejecuta desde cero
#
# La prueba definitiva de reproducibilidad: **que otra persona, con una copia limpia, obtenga los mismos resultados**. Lo simulamos así:
#
# 1. Ejecutamos el análisis en tu carpeta y guardamos una huella (hash) de los resultados.
# 2. **Clonamos** el repositorio en una carpeta nueva, vacía: solo lo que está en la historia de Git.
# 3. Ejecutamos ahí el análisis y comparamos las huellas.
#
# Si el clon funciona y da lo mismo, tu proyecto no depende de nada que solo exista en tu sesión (un archivo que olvidaste hacer `add`, una variable que quedó en memoria, una ruta absoluta).

# %%
import hashlib
import os


def ejecutar_analisis(carpeta):
    """Ejecuta `python analisis_taller.py` en una carpeta, con la carpeta utils/ del curso disponible."""
    entorno = {**os.environ, "PYTHONPATH": str(RAIZ)}
    proceso = subprocess.run([sys.executable, "analisis_taller.py"], cwd=carpeta, env=entorno, capture_output=True, text=True)
    print(proceso.stdout.strip() or proceso.stderr.strip()[-600:])
    return proceso.returncode


def huella(carpeta):
    resultados = Path(carpeta) / "resultados"
    return {a.name: hashlib.sha256(a.read_bytes()).hexdigest()[:12] for a in sorted(resultados.glob("*.csv"))}


print("--- Tu carpeta ---")
assert ejecutar_analisis(PROYECTO) == 0, "el análisis falló en tu carpeta"
huella_propia = huella(PROYECTO)

limpia = TALLER / "copia_limpia"
git("clone", "-q", str(REMOTO), str(limpia), cwd=TALLER)         # solo lo que está en la historia
print("\n--- Copia limpia (clonada del remoto) ---")
assert ejecutar_analisis(limpia) == 0, "el análisis falló en la copia limpia: falta algo en la historia de Git"
huella_limpia = huella(limpia)

print("\nHuellas tu carpeta  :", huella_propia)
print("Huellas copia limpia:", huella_limpia)
print("¿Mismos resultados?  ", huella_propia == huella_limpia)

# %% [markdown]
# Si las huellas coinciden, el proyecto es **reproducible desde cero**. (En un proyecto real, el clon vendría de GitHub: `git clone https://github.com/tu-usuario/analisis-taller`.)
#
# Una última comprobación, ahora en tu repositorio: el árbol de trabajo debe estar **limpio** (todo lo que importa está guardado) y los resultados no deben estar en Git.

# %%
git("status", "--short", "--branch", cwd=PROYECTO)
print("Archivos que Git sigue:", git("ls-files", cwd=PROYECTO, mostrar=False).split())
print("\nLimpiamos la carpeta temporal de este cuaderno.")
shutil.rmtree(TALLER, ignore_errors=True)

# %% [markdown]
# ### Tu turno 3: llévalo a tu trabajo real
#
# En Colab, lo que hiciste aquí desaparece cuando se cierra la sesión (la carpeta era temporal). Para tu trabajo real:
#
# 1. Crea tu repositorio en GitHub y haz el flujo completo con tu código de los bloques 4 a 7: un módulo `.py` con funciones, el cuaderno que las usa, un `README.md`, un `.gitignore` y un `requirements.txt`.
# 2. Haz **al menos tres commits** con mensajes descriptivos.
# 3. Sube a GitHub con el token (sección 7) o con "Guardar una copia en GitHub" (sección 8).
# 4. Verifica que un clon limpio ejecuta tu cuaderno de principio a fin (`Entorno de ejecución > Reiniciar y ejecutar todo`).
#
# **Este repositorio es tu entregable del día 2**: el cuaderno completo, commits documentados, la serie con sus anomalías y los eventos extremos detectados. Mañana lo presentas (6 minutos) y publicas una versión en Zenodo con metadatos FAIR.

# %% [markdown]
# ## Resumen
#
# * **Git** guarda la historia de tu proyecto como commits con autor, fecha y mensaje: **trazabilidad**. **GitHub** guarda una copia remota para compartir.
# * Un **token de grano fino** limitado a un repositorio y con vencimiento corto, guardado como **Secreto de Colab**, es la forma segura de subir desde Colab.
# * El token **nunca** va en una celda, en una salida ni en `.git/config`. Se pasa con `GIT_ASKPASS` y una variable de entorno que existe solo durante el `push`, y se comprueba que la configuración quedó limpia.
# * **Refactorizar** es reorganizar el código en funciones sin cambiar lo que hace. Un buen commit es pequeño, tiene un propósito y un mensaje que explica el *qué* y el *por qué*.
# * **Reproducible** significa que un clon limpio ejecuta el proyecto y da los mismos resultados.
# * Plan de respaldo sin token: **Archivo > Guardar una copia en GitHub**.
