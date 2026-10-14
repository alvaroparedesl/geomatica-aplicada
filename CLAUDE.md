# Geomática Aplicada: agent instructions

Spanish-language geomatics course (Python, STAC, Sentinel-2, Landsat).
Notebooks live in `notebooks/` (and later `talleres/`), shared helpers in
`utils/`, tests in `utils/tests/`.

## Testing policy: local only

- There is NO CI on purpose. Do not add GitHub Actions or any other CI.
- Checks run locally through pre-commit (`.pre-commit-config.yaml`):
  offline tests, notebook sync check, em dash check, basic hygiene.
  Setup: `pip install -r requirements-dev.txt && pre-commit install`.
  Run by hand: `pre-commit run --all-files` (it only sees tracked files;
  use `--files <paths>` for untracked ones).
- Offline tests: `pytest utils -m "not red"`. They must pass without internet.
- Network tests (marker `red`): `pytest utils -m red`, about 12 minutes. Run
  them before releases, courses and workshops, together with a notebook smoke
  run (`jupyter nbconvert --to notebook --execute` over the `.ipynb` files).

## Notebooks: jupytext workflow

- The `.py` file (jupytext `py:percent`) is the source. Edit only the `.py`.
- The `.ipynb` is generated one way, not paired:
  `jupytext --to notebook path/to/notebook.py`.
- `scripts/verificar_notebooks.py` fails if an `.ipynb` is missing or its cell
  sources differ from the `.py`. Regenerate and commit both files.

## Course text style

- Everything students read is in Spanish: markdown cells, comments,
  docstrings, READMEs.
- Use tuteo (tú tienes, puedes, ejecuta). NEVER use voseo (tenés, podés,
  corré). Check any Spanish verb conjugation before writing it.
- NEVER use em dashes (U+2014); use commas, colons, parentheses or hyphens.
  The pre-commit hook enforces this.
- Match the tone of the existing notebooks: neutral but close, explanatory,
  aimed at third-year natural resources students.

## Commits

- Several small conventional commits (`feat:`, `fix:`, `docs:`, `chore:`,
  `test:`), not one large commit.
- No `Co-Authored-By` trailer.

## Sentinel-2 offset caution

Sentinel-2 L2A reflectance has a radiometric offset (-1000 DN, or -0.1) that
depends on the catalog AND the processing baseline of each scene. NEVER
apply a fixed offset by hand: it is decided per image from its processing
baseline (`s2:processing_baseline`), never as one constant for a whole series.

- `utils/` (`utils/stac.py`, `utils/raster.py`) holds the tested reference:
  `offset_sentinel2`, `agregar_offset`, `renombrar_bandas`,
  `convertir_a_reflectancia`. The course notebooks in `notebooks/` and the
  internals of `utils/series.py` use them. Keep their tests
  (`utils/tests/test_offset_sentinel2.py` and related) passing and extended
  when behavior changes.
- In `talleres/fair-series-temporales/` (notebooks 02 to 04) students see the
  code: notebook 03 applies the offset by hand (`offset = -0.1 if baseline >=
  4.0 else 0.0`, then `DN * 0.0001 + offset`) and notebook 04 defines simple
  `agregar_offset` and `reflectancia_sentinel2` functions at the top. They are
  Planetary Computer only and have no error handling on purpose.
- `utils/tests/test_cuadernos_taller.py` extracts that notebook code and checks
  it gives the same result as the `utils` reference. If you change the rule in
  one place, change the other and keep that test passing.

## STAC catalogs

- Default: Microsoft Planetary Computer.
- AWS alternative: Earth Search collection `sentinel-2-c1-l2a`.
- Earth Search `sentinel-2-l2a` is disabled; do not re-enable it.

## Secrets

Never read or print `credentials.json`, `token.pickle` or `gdrive_config.json`
(Google Drive sync). They are git-ignored.
