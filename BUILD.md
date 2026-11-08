# Construcción y reproducción del benchmark principal LSTM–MLP–GRU y sus estudios complementarios

## Construir los tres libros

El repositorio publica E1 en la raíz, E2 en `/entregable2/` y E3 en `/entregable3/`. Cada carpeta tiene su propio `_config.yml`, `_toc.yml`, introducción y capítulos. La compilación de E2/E3 no carga fuentes ni resultados locales de E1; sus referencias históricas son enlaces externos y metadatos propios.

Desde la raíz, con el entorno de `requirements-book.lock.txt` instalado:

```bash
.venv/bin/jupyter-book build book --all --warningiserror --keep-going
.venv/bin/jupyter-book build book_e2 --all --warningiserror --keep-going
.venv/bin/jupyter-book build book_e3 --all --warningiserror --keep-going
.venv/bin/python scripts/check_book_links.py --book-dir book
.venv/bin/python scripts/check_book_links.py --book-dir book_e2
.venv/bin/python scripts/check_book_links.py --book-dir book_e3
.venv/bin/python scripts/assemble_books_site.py --output-dir /tmp/deeplearning-tres-libros
.venv/bin/python scripts/check_book_links.py --html-dir /tmp/deeplearning-tres-libros
.venv/bin/python -m http.server 8000 --directory /tmp/deeplearning-tres-libros
```

La carpeta de salida del ensamblado debe ser nueva o vacía; usar otra ruta al repetir. Abrir `http://localhost:8000/entregables.html`. La publicación automatizada siempre reconstruye y reúne los tres libros, para conservarlos cuando cambia uno de ellos.

E2 ya cuenta con la guía completa de 65 páginas y capítulos 1–14. La ruta vigente es F2a: series de tiempo, con contrato D7×5 y δ heredados conservados, veinte configuraciones y quince recetas ajustables T12. El commit `599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0` registra el protocolo antes de los ajustes P0/OOF de Bitcoin. Se completaron las 220 trayectorias y 440 ajustes P0/OOF, y se analizaron 305 casos por receta. Ese experimento de desarrollo se distingue de la réplica ETTh1 y del benchmark final, todavía pendiente. E1 distingue ahora el núcleo LSTM–MLP–GRU del estudio anterior complementario con TCN; los registros de E2 conservan sus propias fuentes y cronología.

## Revisión editorial propuesta de E2 (08/11/2026)

La [propuesta de diez modelos y un híbrido](book_e2/sections/15_revision_modelos.md) es documental y no cambia el ejecutor ni el catálogo registrados. Sus cuatro altas y HAR + red residual requieren diseño, implementación y evidencia propios; no se ejecutan al construir los libros. Las anclas históricas E2 GRU–LSTM–TCN se distinguen del núcleo actual E1 MLP–LSTM–GRU.

## Implementación final de E2

La nueva fuente está en `experiments/e2_completion_20261106/` y `scripts/run_e2_final_benchmark.py`, con módulos de arquitectura, estadística, incertidumbre, diagnósticos y perfilado. Sus pruebas usan datos sintéticos; aún no acreditan una tabla final ni su reproducción independiente. La enmienda se registró en `dbb193a72897c80cbd34b452ba3cfe9d82ce6ce0`; el recibo público del 06/11/2026 a las 19:00 UTC acredita doce trayectorias y veinticuatro ajustes cerrados de TSMixer, con HPO incompleto. Es una instantánea, no confirmación de actividad actual o cierre posterior. Mantiene T12/quince asignaciones; conserva once candidatos aleatorios y añade c12 fuente por red. P1 y calibración A usan 53 fechas (08/11–30/12/2024), selección/ajuste hasta el 31/10 y embargo 01–07/11. El primario conserva el corte 24/12 y test conocido de sesenta fechas. Todos los pesos primarios/A/controles negativos y calibradores deben cerrar en ese orden antes del scoring del test.

Los JSON y el código P0 no se modifican para retrofechar estas decisiones. La ejecución final necesita sus propias fuentes fijadas, entorno verificado, checkout limpio y salida privada nueva; se conserva la frontera de publicación de agregados, sin predicciones individuales, pesos ni datos completos de evaluación. El [capítulo8](book_e2/sections/08_reproducibilidad.md) distingue ambos procedimientos.

## Protocolo P0 de E2 y evidencias

La validación documental no abre precios ni ajusta modelos. El export genera un registro prospectivo de candidatos y reglas; sus contadores iniciales no se reescriben para presentarlos como resultados posteriores:

```bash
python scripts/e2_protocol_plan.py validate
python scripts/e2_protocol_plan.py export
python scripts/e2_predictive_uncertainty.py
python scripts/package_e2_remediation.py --check
```

El [capítulo 8](book_e2/sections/08_reproducibilidad.md) detalla el entorno `requirements-e2-remediation.lock.txt`, las fuentes de autores por hash y el flujo explícito `preflight → register → fit → score → analyze`. Un nuevo replay se registra en un checkout limpio y usa una salida privada nueva, fuera del repositorio. El scoring exige el cierre de todos los pesos programados. La compilación editorial consume agregados guardados y no dispara ese flujo.

Las pruebas nuevas son sintéticas y no necesitan datos del mercado ni las fuentes externas de autores:

```bash
python tests/test_e2_protocol_plan.py
python tests/test_e2_temporal_baselines.py
python tests/test_e2_remediation_pilot.py
python tests/test_e2_predictive_uncertainty.py
python tests/test_e2_remediation_statistics.py
python tests/test_e2_literature_screening.py
```

Los tests numéricos necesitan el entorno científico indicado por el lock. Los adaptadores de P0, además, se verifican sobre las fuentes externas fijadas. La reproducción de cálculos desde predicciones privadas requiere esos insumos; el ZIP publica agregados, hashes y software, no checkpoints ni predicciones individuales. La comparación completa de veinte configuraciones desde un entorno limpio no se declara terminada.

Los diseños iniciales de 05/11 y 06/11, su auditor `audit_e2_benchmark_readiness.py` y los paquetes `benchmark_design_download` / `benchmark_analysis_download` se conservan como **archivo de la propuesta v1**, con sus quince filas y bloqueos originales. No son el control de preparación actual ni se cambian sus flags para fingir cumplimiento. Sus módulos numéricos siguen disponibles como código reutilizable. El control actual es `e2_protocol_plan.py` y los recibos de cada fase.

La réplica TimeMixer y el piloto externo ETTh1 permanecen separados en `experiments/e2_paper_guide_20261106/`, `requirements-paper-guide.lock.txt` y el [capítulo 5](book_e2/sections/05_paper_guia.md). Para verificar el suplemento externo sin entrenar:

```bash
python scripts/package_e2_paper_guide.py --check
```

El protocolo de la réplica se publicó antes de los ajustes en `329b0a6c8238d632dd00537c5cefab91360bafd1`; sus discrepancias y enmiendas se conservan. No se mezcla su MSE estandarizada ETTh1 con la RMSE σ₇ de Bitcoin.

El JBook distingue **fase I exploratoria** y **fase II D7 planificada** de GRU–LSTM–TCN. Ambas usan Bitcoin, σ₇ poblacional, horizonte de un día y covariables OHLCV causales. La primera conserva las cuatro entradas y su test enero de 2025–agosto de 2026 ya observado. La segunda registra HPO comparable, piloto, dimensionamiento y evaluación sobre el 01/09–30/10/2026. Sus recetas y registros no se mezclan: GRU/D7 de fase I tiene 32 unidades y 3.777 parámetros; GRU/D7-c03 de fase II tiene 16 unidades y 1.121 parámetros.

## Construcción del libro

Se requiere un clon del repositorio, Python 3.12 y Jupyter Book 1.0.4.post1 para reconstruir las fuentes del libro. El producto 8.1 ya contiene el HTML navegable y no necesita este build. El build consume tablas/figuras guardadas y no entrena modelos, descarga mercado o ejecuta inferencias.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-book.lock.txt
.venv/bin/jupyter-book build book --all --warningiserror --keep-going
.venv/bin/python scripts/check_book_links.py
.venv/bin/python -m http.server 8000 --directory book/_build/html
```

Abrir `http://localhost:8000`. Los comandos de esta página usan Bash en Linux; en Windows, ejecutarlos dentro de WSL2 con Linux. La reproducción experimental requiere `os.sched_getaffinity`, usado al registrar el entorno; el soporte de Windows nativo no está validado. No regenerar resúmenes experimentales para construir HTML.

## Entornos

La librería CV utiliza NumPy menor que 2 y tiene un entorno propio. La preparación EDA usa su lock y los modelos y análisis usan el lock de entrenamiento observado; no sustituir versiones por la última disponible.

```bash
python3.12 -m venv .venv-cv
.venv-cv/bin/python -m pip install -r requirements-cv.txt
python3.12 -m venv .venv-eda
.venv-eda/bin/python -m pip install -r requirements-eda.lock.txt
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements-training.lock.txt
```

El código experimental se organiza así:

| Archivo | Función |
|---|---|
| `scripts/dl_models_gru_lstm_tcn.py` | API exclusiva GRU/LSTM/TCN; LSTM histórica exacta, GRU de 32 unidades y TCN residual de 16 canales |
| `scripts/run_gru_lstm_tcn.py` | Validación de entradas, registro y CV en una salida nueva |
| `scripts/gru_lstm_tcn_final.py` | Cierre, ajuste final y scoring explícito después de congelar pesos |
| `scripts/analyze_gru_lstm_tcn_run.py` | Estadística de nuevas reproducciones, sin dependencia del experimento privado original |
| `scripts/build_gru_lstm_tcn_evidence.py` | Agregación y análisis del trío observado; no ajusta ni predice |
| `scripts/gru_lstm_tcn_diagnostics.py` | BDS, Ljung–Box y Diebold–Mariano exploratorios |
| `scripts/export_gru_lstm_tcn_delivery.py` | Productos y cronología de entrega |
| `scripts/package_gru_lstm_tcn_downloads.py` | Siete ZIP de productos y ZIP conjunto |

Los ayudantes históricos de datos, optimización y la clase LSTM siguen como dependencias compartidas; su presencia no los convierte en otros modelos del libro. Los archivos firmados no se modifican para añadir la nueva API.

## Datos e índices

Leer `data/README.md` y usar `scripts/download_binance_spot.py --help` para adquirir archivos oficiales de Binance Vision con verificación de checksums. Los archivos de un minuto permanecen fuera del checkout. El flujo de EDA conserva los derivados de desarrollo 2023–2024:

- `results/eda_development/daily_market_development.csv`.
- `results/eda_development/daily_volatility_development.csv`.

Un clon del repositorio ya incluye estos dos CSV de desarrollo. Los ZIP públicos omiten los derivados diarios: para reproducir desde las descargas, descomprimir los productos **8.2 (código)** y **8.3 (instrucciones y manifiestos de datos)** en la misma carpeta, conservar sus subdirectorios y crear los entornos `.venv-eda` y `.venv-training` con sus respectivos locks, como se indica arriba. Conservar aparte los `LEEME.md` y `MANIFIESTO.json` de cada ZIP para verificar su integridad individual. Después de adquirir los ZIP oficiales siguiendo `data/README.md`, generar los derivados desde la raíz del proyecto:

```bash
.venv-eda/bin/python scripts/eda_binance_development.py \
  --raw-root /ruta/adquisicion/oficial
```

`--raw-root` señala la carpeta que contiene `BTCUSDT/1m/`; se puede cambiar sin modificar código. El script verifica los hashes del manifiesto versionado, abre únicamente los 24 archivos mensuales de desarrollo 2023–2024 y produce los dos CSV requeridos con calendario UTC, sin comprimir faltantes. Este paso debe terminar antes de `--phase validate`; no abre el test ni ajusta redes.

Los manifiestos de folds D7/D14/D21/D28 provienen de llamadas reales a `split_train_val_forwardChaining(np.arange(731),D,1,1)`, agrupadas explícitamente en doce meses de 2024. La agrupación mensual no es una API nativa mensual ni 366 refits diarios. Se conserva embargo de D días y validación interna de 90 días.

Los derivados diarios completos del **test**, las predicciones individuales y los pesos del experimento observado no forman parte de la exportación pública. Los CSV de **desarrollo 2023–2024** sí están versionados en el repositorio, aunque se omiten de los ZIP de entrega. Una verificación completa de los checkpoints observados requiere la evidencia conservada y sus hashes; un clon público puede verificar los agregados permitidos sin recalcular pesos privados.

## Verificar la publicación sin entrenar

El manifiesto público enumera únicamente archivos agregados y metadatos. En un clon del repositorio:

```bash
.venv-training/bin/python scripts/run_gru_lstm_tcn.py --phase verify \
  --evidence-manifest results/gru_lstm_tcn_20261105/public_allowlist_manifest.json
```

La verificación no registra nuevos ajustes ni abre test. `results/gru_lstm_tcn_20261105/` y su copia estática tienen contenido idéntico por allowlist.

Para verificar las descargas, descomprimir los productos **8.2, 8.3 y 8.5** en una misma raíz de trabajo, conservando los subdirectorios `scripts/`, `data/`, `results/` y `book/_static/`. Guardar aparte el `LEEME.md` y el `MANIFIESTO.json` de cada producto. Si se descargó `entregable1_completo.zip`, primero extraer ese contenedor y después sus ZIP internos. El producto 8.5 contiene los agregados bajo `book/_static/`, por lo que el comando es:

```bash
.venv-training/bin/python scripts/run_gru_lstm_tcn.py --phase verify \
  --evidence-manifest book/_static/gru_lstm_tcn_20261105/public_allowlist_manifest.json
```

Esta verificación de agregados no requiere adquirir mercado ni generar los CSV diarios. El informe HTML del producto 8.1 se abre por separado en `informe/intro.html`.

## Benchmark principal actual D7: nueva ejecución común de 608 fechas

El estudio `e1_608_remediation_20261107` ejecuta de nuevo **GRU32, LSTM32 y MLP16** sobre el contrato común D7×5, σ₇ poblacional y h=1. Su fuente científica inmutable es [`98e0cbcf4bb09dba863289261677dd6e7534d259`](https://github.com/jassan-bit/DeepLearning_202630/commit/98e0cbcf4bb09dba863289261677dd6e7534d259): la publicación se observó mediante GET el **08/11/2026 a las 01:08:27 UTC** y el registro ocurrió a las **01:09:05 UTC**, todavía 07/11 en Colombia, antes del HPO nuevo. El diseño y su README están en `experiments/e1_608_remediation_20261107/`.

**Ejecución cerrada:** HPO, piloto, planificación publicada, ajustes definitivos, scoring y análisis completos. **R requerido=R final=309**, con **927 modelos, 1.854 etapas definitivas y 563.616 pronósticos privados**. **GRU32/c03 es el baseline seleccionado por CV.** El export público y su espejo contienen 34 archivos verificados. Una auditoría numérica independiente comprobó las seis métricas, DM, bootstrap, rangos y diagnósticos residuales. Las tablas de los registros anteriores que siguen conservan su identidad y no se usan como resultados de esta ejecución.

La capacidad permanece fija en 32/32/16 unidades y 3.777/5.025/593 parámetros para GRU/LSTM/MLP. La búsqueda equitativa evalúa **cuatro candidatos por familia**, combinando dos tasas de aprendizaje y dos valores de dropout, con semillas **16101–16102** y diez bloques mensuales enero–octubre de 2024: **24 trayectorias, 240 bloques y 480 etapas de ajuste**. La CV de desarrollo seleccionó **GRU32/c03 como baseline** y congeló las tres recetas antes del piloto. El piloto utiliza **17101–17110**, diez semillas nuevas por familia, **30 trayectorias y 60 etapas**, sobre 01/11–30/12/2024.

La [selección real](https://github.com/jassan-bit/DeepLearning_202630/blob/0f5f5eaeb7131b1413a062cb0cce104a88f3ddf3/experiments/e1_608_remediation_20261107/planning/selection.json) conserva las recetas y sus métricas de CV, distintas de cualquier resultado de evaluación:

| Modelo | Receta elegida por desarrollo | Parámetros | RMSE CV media de dos semillas |
|---|---|---:|---:|
| **GRU32, baseline, c03** | LR 0,003; dropout 0,1 | 3.777 | 0,00499676871694 |
| LSTM32, c02 | LR 0,003; dropout 0 | 5.025 | 0,00639760418885 |
| MLP16, c03 | Una capa ReLU; LR 0,003; dropout 0,1 | 593 | 0,00582374658645 |

El piloto terminó el **08/11/2026 a las 01:15:04 UTC** y el plan se calculó a las **01:15:15 UTC**. La regla `R_final=max(30,R_required)` da **309 semillas por familia**, **18101–18409**, **927 modelos y 1.854 etapas definitivas**. El [plan calculado](https://github.com/jassan-bit/DeepLearning_202630/blob/0f5f5eaeb7131b1413a062cb0cce104a88f3ddf3/experiments/e1_608_remediation_20261107/planning/plan.json), con SHA-256 `6295b408623c811aa707b7c55355ab5fb529306f91049b047bac8dcc4d0fd15f`, se publicó en [`0f5f5eaeb7131b1413a062cb0cce104a88f3ddf3`](https://github.com/jassan-bit/DeepLearning_202630/commit/0f5f5eaeb7131b1413a062cb0cce104a88f3ddf3), observado mediante GET a las **01:17:26 UTC**, antes del primer ajuste definitivo. El piso de treinta no altera ese requerimiento mayor.

La ejecución completa sumó **2.394 etapas**: 480 HPO, 60 piloto y 1.854 definitivas. Todos los pesos y escaladores finales quedaron congelados a las **01:28:08 UTC** del 08/11/2026. La verificación de acceso y la primera lectura de precios ocurrieron a las **01:28:33 UTC**, el scoring terminó a las **01:28:54 UTC** y el análisis a las **01:29:29 UTC**. Cada checkpoint se puntuó una vez sobre las **608 fechas 01/01/2025–31/08/2026**, generando **563.616 pronósticos privados (608×927)**, más una referencia de Persistencia determinista. La [cronología pública](book/_static/e1_608_remediation_20261107/chronology.json) conserva las marcas completas. Este calendario ya era conocido: el alcance sigue siendo **histórico y exploratorio**. El [capítulo 11](book/sections/11_reejecucion_608.md) desarrolla el diseño y sus límites.

El entrypoint actual es `scripts/run_e1_608_remediation.py`. Desde el checkout limpio del commit científico publicado, con Python 3.12.14 y `requirements-e1-curves.lock.txt`, las fases mantienen el orden siguiente y escriben fuera de Git:

```bash
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py register --output-dir /ruta/privada/nueva608 --published-commit 98e0cbcf4bb09dba863289261677dd6e7534d259 --publication-receipt /ruta/recibo_fuente.json
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py hpo --output-dir /ruta/privada/nueva608
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py pilot --output-dir /ruta/privada/nueva608
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py plan --output-dir /ruta/privada/nueva608
```

Una vez publicado el plan real y observado remotamente su commit y SHA-256, se continúa sin cambiar el checkout científico:

```bash
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py register-plan --output-dir /ruta/privada/nueva608 --published-plan-file /ruta/plan608_publicado.json --published-commit SHA40_PLAN --publication-receipt /ruta/recibo_plan608.json
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py fit --output-dir /ruta/privada/nueva608
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py freeze --output-dir /ruta/privada/nueva608
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py score --output-dir /ruta/privada/nueva608 --raw-root /ruta/brutos/binance
PYTHONHASHSEED=0 python scripts/run_e1_608_remediation.py analyze --output-dir /ruta/privada/nueva608
```

`score` comprobó los veinte archivos mensuales originales de enero de 2025 a agosto de 2026 por sus identidades registradas: **42.542.826 bytes originales verificados**, con la adquisición completada. El contexto procede únicamente del CSV de desarrollo 2023–2024 congelado. El cuaderno **`notebooks/descargar_datos_608_e1.ipynb`** se conserva sin ejecutar ni salidas en la distribución: permite adquirir los diecinueve archivos de enero de 2025 a julio de 2026, repartidos en dos ZIP menores de 24 MB, sin volver a descargar agosto, que ya se había recuperado. Su inclusión como ayuda no acredita que ese cuaderno se utilizara para esta adquisición. Una reanudación técnica conserva los artefactos y sus recibos, sin sustituir semillas ni recalcular una puntuación cerrada.

El exportador se ejecutó tras el cierre de todas las fases desde ese mismo checkout hacia una raíz editorial externa. Para reproducir esa verificación sobre un cierre privado propio:

```bash
PYTHONHASHSEED=0 python scripts/export_e1_608_remediation.py --source-root /ruta/checkout_fuente --private-dir /ruta/privada/nueva608 --output-dir /ruta/editorial/results/e1_608_remediation_20261107 --mirror-dir /ruta/editorial/book/_static/e1_608_remediation_20261107
python -m unittest discover -s tests -p 'test_e1_608_export.py'
```

La [publicación cerrada](book/_static/e1_608_remediation_20261107/README.md) contiene **33 archivos públicos y `manifest.json`**, con el contrato cerrado `e1_public_metadata_allowlist_v1`: métricas de las seis medidas por semilla y agregadas, recetas, HPO, piloto, plan, fuentes, cronología, runtime y costes observados de entrenamiento e inferencia. Incluye **33.141 filas observadas de pérdidas finales** internas/de reajuste en CSV, media e IQR con `n_available`, la semilla predefinida 18101, los QQ de las tres diferencias RMSE, Friedman/Wilcoxon y la inferencia primaria/temporal y diagnósticos BDS/Ljung–Box de esta misma cohorte. El reajuste final no tiene pérdidas de validación del test. La potencia planificada corresponde a δ bajo supuestos normales y de transferencia de varianza; la hipótesis primaria es diferencia cero, sin afirmar potencia alcanzada ni temporal.

Las [seis métricas por semilla](book/_static/e1_608_remediation_20261107/run_metrics.csv) y el [resumen agregado](book/_static/e1_608_remediation_20261107/metric_summary.csv) corresponden exclusivamente a esta ejecución nueva:

| Modelo actual, 608 fechas | RMSE, media ± DE entre semillas | R² medio |
|---|---:|---:|
| GRU32, baseline por CV | 0,00521269 ± 0,00112121 | 0,709833 |
| LSTM32 | 0,00625115 ± 0,00079563 | 0,594658 |
| MLP16 | 0,00522813 ± 0,00054111 | 0,717998 |
| **Persistencia determinista** | **0,00408560** | **0,829605** |

**Persistencia tiene menor RMSE que las tres redes.** En la familia primaria Holm-3, GRU–LSTM y LSTM–MLP rechazan igualdad; GRU–MLP no rechaza, con p ajustado **0,822534**. El análisis temporal DM/Holm-6 rechaza GRU–LSTM y los tres pares frente a Persistencia; no rechaza GRU–MLP ni LSTM–MLP. El bootstrap principal de bloques siete conserva intervalos familiares compatibles con mejor RMSE de Persistencia frente a las redes; los CSV separan ese estimando del contraste DM. Ninguna de estas conclusiones acredita utilidad financiera o generalización a otros mercados.

Los [costes por semilla](book/_static/e1_608_remediation_20261107/run_costs.csv) y su [resumen](book/_static/e1_608_remediation_20261107/cost_summary.csv) proceden de los ajustes y del scoring actuales, en CPU, tres trabajadores y un hilo por trabajador:

| Modelo | Parámetros | Entrenamiento, s, media ± DE | Inferencia de 608 fechas, ms, media ± DE | Corridas por medida |
|---|---:|---:|---:|---:|
| GRU32 | 3.777 | 3,832 ± 4,490 | 2,236 ± 0,535 | 309 |
| LSTM32 | 5.025 | 0,989 ± 0,978 | 2,276 ± 0,548 | 309 |
| MLP16 | 593 | 1,022 ± 0,813 | 0,369 ± 0,100 | 309 |

Las [curvas de media/IQR](book/_static/e1_608_remediation_20261107/learning_curves.png), la [semilla predefinida 18101](book/_static/e1_608_remediation_20261107/first_seed_learning_curves.png) y los [QQ de las tres diferencias](book/_static/e1_608_remediation_20261107/qq_rmse_differences.png) acompañan los datos numéricos. El R² original de GRU≈0,854 pertenece únicamente al registro anterior de diez semillas; la nueva media es **0,709833**.

El refresco de descargas exige ese cierre real y rechaza ausencia, mezcla de ejecución/fuentes, R/semillas incompatibles o una cohorte distinta de 608 fechas. **8.2** incorpora el ejecutor, estadística, exportador, pruebas y cuaderno nuevo; **8.5**, los 34 archivos públicos; **8.6/8.7**, el diseño/README fuente y sus registros de configuración y procedencia. No distribuye precios, objetivos, pronósticos diarios, pesos ni valores de escaladores. La edición `20261107_remediation_608` conserva el registro anterior `20261107_primary_608`, el complemento de sesenta fechas y los snapshots históricos; **8.3 permanece byte a byte idéntico**. La edición incorpora los agregados cerrados y sus manifiestos, sin modificar las fuentes científicas congeladas.

## Análisis posterior de errores sin modificar el experimento cerrado

`scripts/build_e1_frozen_error_analysis.py` lee las predicciones cerradas de `e1_608_remediation_20261107`, sus recibos de puntuación/análisis y las 716 etiquetas de desarrollo necesarias para fijar rangos comunes. No carga checkpoints o escaladores, no abre archivos brutos de mercado y no ajusta ni vuelve a pronosticar. Requiere un checkout que conserve los dos espejos públicos originales y las fuentes científicas firmadas, además de acceso al directorio privado de esa ejecución.

En el entorno científico de E1, usando destinos nuevos y vacíos:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/build_e1_frozen_error_analysis.py --private-dir /ruta/privada/608_cerrado --output-dir /tmp/e1-errores-congelados --mirror-dir /tmp/e1-errores-publicos
python -m unittest discover -s tests -p 'test_e1_frozen_error_analysis.py'
```

El suplemento verifica las seis métricas de las 928 corridas y los hashes de 106 archivos antes y después. La ejecución de esta entrega obtuvo diferencia numérica máxima cero y ninguna modificación original. Publica diecisiete archivos y su manifiesto: errores por mes/año/rango de volatilidad, medias mensuales observado/pronosticado, densidad agregada común30×30, comparación por deciles de desarrollo, sesgo y dispersión, descomposición descriptiva de variación y procedencia. Los IC describen inicializaciones condicionadas a cada segmento; las figuras no son un ensemble evaluado ni una validación temporal nueva.

El [capítulo 12](book/sections/12_errores_congelados.md) documenta resultados y alcance. Los agregados están en `results/e1_frozen_error_analysis_20261108/` y su espejo `book/_static/e1_frozen_error_analysis_20261108/`. Los datos privados y los CSV diarios de predicciones no se distribuyen. Un clon público permite revisar código, fuentes y agregados; recalcular el suplemento requiere los pronósticos privados cerrados de la ejecución que se desea comprobar. No se sustituyen por una ejecución nueva para afirmar identidad con ésta.

## Cierre del cribado de los 620 grupos nativos recibidos

`scripts/build_e1_native_620_screening.py` compila decisiones revisadas y fechadas, sin clasificar automáticamente artículos, consultar de nuevo las bases o ajustar modelos. Sus fuentes son el protocolo, las decisiones de cada grupo y las fichas documentales terminales bajo `references/reviews/native_620_screening_20261108/`. Conserva las 834 ocurrencias recibidas y sus 620 grupos provisionales, las siete ocurrencias IEEE no exportadas y la separación entre resúmenes, fuentes primarias, contexto e informes no recuperados.

```bash
python scripts/build_e1_native_620_screening.py
python scripts/build_e1_native_620_screening.py --check
python -m unittest discover -s tests -p 'test_e1_native_620_screening.py'
```

El primer comando reconstruye los agregados en `references/literature_e1/native_620_screening_20261108/` y su espejo `book/_static/literature_e1/native_620_screening_20261108/`, desde las decisiones existentes; el segundo compara sus bytes con esa reconstrucción. No realizan otra lectura documental ni una nueva búsqueda. El manifiesto declara las copias revisadas de los insumos bajo `review_inputs/`; la entrega 8.4 conserva sus alias canónicos para recompilar tras extraer junto al código 8.2 y los agregados 8.5. No incluye los abstracts completos ni los artículos PDF.

`book/references.bib` conserva las setenta entradas de la base. El segundo archivo de `bibtex_bibfiles`, `bibliography_additions.bib` del suplemento, sólo contiene DOI nuevos de fuentes primarias realmente evaluadas; no se cuentan como estudios incluidos todas las referencias citadas. Los informes y sus versiones se concilian por experimento para evitar doble voto. El [capítulo 3, §3.15](book/sections/03_estado_arte.md) presenta el flujo y su relación con el corpus anterior. El núcleo entrenado, los resultados y la limitación del calendario conocido permanecen iguales.

## Registro anterior del 07/11/2026: D7, 608 fechas y diez semillas

El registro anterior `e1_primary_608_20261107` usa las recetas originales LSTM32, MLP16 y GRU32, D7×5, σ₇ poblacional y h=1. El baseline de desarrollo es **GRU32, LR 0,003, dropout 0,1 y 3.777 parámetros**, con R² medio 0,854173 en el test 01/01/2025–31/08/2026. Las tres redes **ya se ejecutaron con diez semillas 4101–4110** sobre las mismas 608 fechas; la MLP16 histórica no requiere un nuevo entrenamiento. LSTM32 conserva LR 0,003, dropout 0,1 y 5.025 parámetros; MLP16, LR 0,001, dropout 0,1 y 593 parámetros. La CV común contiene las 366 fechas de 2024.

| Modelo, test del registro anterior | RMSE media | R² medio |
|---|---:|---:|
| GRU32 | 0,00377511 | 0,854173 |
| LSTM32 | 0,00502979 | 0,741108 |
| MLP16 | 0,00479318 | 0,764012 |
| Persistencia determinista | 0,00408560 | 0,829605 |

La mayor cobertura temporal frente al estudio de sesenta fechas no acredita representatividad global: continúa siendo Bitcoin, un solo calendario histórico conocido. El HPO original fue desigual entre familias y R=10 es el presupuesto de semillas ya ejecutado, sin dimensionamiento prospectivo para esta promoción editorial. La elección anterior de presentar este benchmark como principal ocurrió después de observar sus resultados; no establece una reserva confirmatoria nueva ni independencia temporal entre semillas.

El ensamblador de aquel registro conserva y verifica **métricas ya guardadas**, sin ajustar redes, inferir ni adquirir mercado:

```bash
python scripts/build_e1_primary_608.py
python scripts/build_e1_primary_608.py --check
```

`results/e1_primary_608_20261107/` y su espejo `book/_static/e1_primary_608_20261107/` contienen las métricas por corrida y el resumen común con Persistencia, las recetas originales y la procedencia por hash. El producto **8.5** incorpora toda esta etapa; **8.6/8.7** incluyen `summary.json`, `execution_provenance.json` y `source_manifest.json`, y **8.7** también el manifiesto. Los ejecutores históricos que siguen requieren sus datos y una salida nueva si se desea repetir ajustes; este ensamblador histórico no ejecuta esas rutas.

## Registro complementario cerrado del 07/11/2026: sesenta fechas LSTM–MLP–GRU

La selección bibliográfica actual está en `references/literature_e1/core_selection_20261107/`. Su registro se reconstruye sin entrenar:

```bash
python scripts/build_e1_core_selection.py --check
python -m unittest discover -s tests -p 'test_e1_core_selection.py'
```

Los resultados compatibles anteriores se recopilan separadamente, sin crear nuevas predicciones:

```bash
python scripts/build_e1_historical_core.py --check
python -m unittest discover -s tests -p 'test_e1_historical_core.py'
```

El resultado histórico contiene **240 corridas agregadas**: CV2024 y test histórico 01/01/2025–31/08/2026, cuatro historias D7/14/21/28, tres modelos y diez semillas; el resumen contiene **144 filas de seis métricas**. Sólo D7 lleva seis contrastes pareados posteriores. No se reconstruyen desde esos agregados curvas, pesos o vectores diarios de MLP ausentes, ni se convierte su benchmark exploratorio en una nueva ejecución común.

La fuente del estudio complementario está en `experiments/e1_core_mlp_extension_20261107/`. Mantiene el contrato D7×5, σ₇ y h=1, pero ejecuta **desde cero las tres familias**, con ocho candidatos, dos semillas y diez bloques de validación por modelo. La MLP propia tiene una capa oculta ReLU, entrada aplanada ordenada y salida Softplus; las anchuras candidatas son 32/128, con 1.185/4.737 parámetros. Son adaptaciones del proyecto, distintas de la receta histórica de 593 parámetros. El baseline se congela por CV antes de un piloto nuevo de diez semillas por familia. El plan exige `R_final=max(465,R_required)` y su publicación antes de los ajustes definitivos.

La ejecución original de esta extensión está cerrada en `results/e1_core_mlp_extension_20261107/`, con espejo público en `book/_static/e1_core_mlp_extension_20261107/`: **48 trayectorias HPO/960 etapas, 30 piloto/60 etapas, R requerido=331 y R final=465, 1.395 modelos definitivos/2.790 etapas, 83.700 predicciones privadas y 44.201 filas públicas de pérdidas**. Las fuentes científicas corresponden al commit `91acdeae1efd92f098ff2ce9a32f90c5b0a4b98a`; el plan real se publicó en `593b4ec943dc0f7e257d348f3145e71810f665e4` antes de los ajustes definitivos. El baseline por desarrollo es GRU/c03: 16 unidades, LR 0,003, dropout 0,1; LSTM/c02 usa 16 unidades, LR 0,003 y dropout 0; MLP/c03 usa 32 unidades ReLU, LR 0,003 y dropout 0,1, con 1.185 parámetros.

En las 60 fechas evaluadas, las RMSE medias son GRU 0,00511696, MLP 0,00516664, LSTM 0,00565123 y persistencia 0,00384478. Los CSV públicos conservan mayor precisión e inferencia completa. Persistencia tiene menor error; GRU–MLP no rechaza igualdad tras Holm-3. El calendario de septiembre–octubre de 2026 ya era conocido al diseñar esta extensión: sus resultados son **históricos y exploratorios**, no una nueva reserva confirmatoria. La publicación previa controla el orden de esta ejecución nueva; no convierte sus resultados en desconocidos.

Desde un checkout limpio e inmutable de las fuentes ya publicadas, usando Python 3.12.14 y `requirements-e1-curves.lock.txt`:

```bash
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py register --output-dir /ruta/privada/nueva --published-commit SHA40_FUENTE --publication-receipt /ruta/recibo_fuente.json
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py hpo --output-dir /ruta/privada/nueva
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py pilot --output-dir /ruta/privada/nueva
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py plan --output-dir /ruta/privada/nueva
```

Después se publica el `plan.json` real del piloto y se conserva un recibo de observación remota que identifique su commit y SHA-256. Las fechas de un commit por sí solas no acreditan publicación. Las fases siguientes requieren esos bytes exactos y que todavía no se haya iniciado el ajuste definitivo:

```bash
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py register-plan --output-dir /ruta/privada/nueva --published-plan-file /ruta/plan_publicado.json --published-commit SHA40_PLAN --publication-receipt /ruta/recibo_plan.json
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py fit --output-dir /ruta/privada/nueva
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py freeze --output-dir /ruta/privada/nueva
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py score --output-dir /ruta/privada/nueva --raw-root /ruta/brutos/binance
PYTHONHASHSEED=0 python scripts/run_e1_core_mlp_extension.py analyze --output-dir /ruta/privada/nueva
```

`score` requiere los 32 archivos originales identificados por hash: agosto aporta contexto y del 1 de septiembre al 30 de octubre se evalúan 60 fechas. Sólo los abre después de comprobar todos los pesos/configuraciones nuevos. Sin esos archivos no se presenta una tabla completa de evaluación. No se reutilizan puntuaciones antiguas de GRU/LSTM junto con una MLP nueva. Las predicciones privadas llevan `study_id` y `execution_id` de una sola ejecución; el módulo estadístico rechaza mezclas.

Si el entorno de ejecución no puede acceder a Binance, el cuaderno `notebooks/descargar_datos_originales_e1.ipynb`, incluido sin resultados en 8.2, permite adquirir los 32 archivos originales desde otro entorno y comprobar todos sus SHA-256. También puede [abrirse en Google Colab](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/4033d09005160de8de57b8b2a512cbba7e86c8fd/notebooks/descargar_datos_originales_e1.ipynb). Genera `datos_originales_E1.zip`; antes de usar sus contenidos, el ejecutor vuelve a comprobar los hashes registrados. La disponibilidad del cuaderno no acredita su ejecución ni la adquisición efectiva de los datos.

Una vez cerradas todas las etapas, se ejecuta el exportador **desde el mismo checkout fuente inmutable** hacia directorios públicos nuevos externos a ese checkout:

```bash
PYTHONHASHSEED=0 python scripts/export_e1_core_mlp_extension.py --source-root /ruta/checkout_fuente --private-dir /ruta/privada/nueva --output-dir /ruta/editorial/results/e1_core_mlp_extension_20261107 --mirror-dir /ruta/editorial/book/_static/e1_core_mlp_extension_20261107
python -m unittest discover -s tests -p 'test_e1_core_mlp_export.py'
```

El exportador no ajusta, puntúa ni abre archivos de mercado. Comprueba hashes de todos los artefactos cerrados, las 48 trayectorias HPO/960 etapas, las 30 piloto/60 etapas, el plan real, todas las 3R trayectorias definitivas y sus6R etapas, y los60outcomes de evaluación. Verifica publicaciones anteriores a los ajustes correspondientes, cronología, identidad de ejecución y panel exacto GRU/LSTM/MLP. Publica métricas por semilla, inferencia agregada, configuraciones, procedencia y curvas de pérdidas de desarrollo de las tres familias. El reajuste no recibe una validación inventada de reserva. Los checkpoints, precios, objetivos/pronósticos diarios y valores de escaladores permanecen privados. El manifiesto público limita el producto a esa lista cerrada; una exportación incompleta no se distribuye como resultado final.

## Reproducir una ejecución nueva del estudio complementario de fase I

El adaptador público es una versión portable publicada después de la ejecución local. No se atribuye su hash a un entrenamiento anterior. Para una repetición, elegir una salida vacía distinta de la evidencia cerrada:

```bash
.venv-training/bin/python scripts/run_gru_lstm_tcn.py --phase validate --project-root .
.venv-training/bin/python scripts/run_gru_lstm_tcn.py --phase register \
  --project-root . --output-dir /ruta/nueva/gru-lstm-tcn
.venv-training/bin/python scripts/run_gru_lstm_tcn.py --phase run \
  --project-root . --output-dir /ruta/nueva/gru-lstm-tcn
.venv-training/bin/python scripts/run_gru_lstm_tcn.py --phase verify \
  --project-root . --output-dir /ruta/nueva/gru-lstm-tcn
```

`run` genera un registro si la salida todavía no existe y conserva/verifica los ajustes completados al reanudarse. `--limit-trajectories` es sólo una ejecución técnica parcial; un resultado incompleto no permite cerrar test. El conjunto completo tiene 120 trayectorias, 1.440 segmentos mensuales y 2.880 ajustes.

En la reproducción de fase I los hiperparámetros están fijos: GRU/LSTM ancho 32, LR 0,003; TCN ancho 16, LR 0,001. AdamW, dropout 0,1, weight_decay 0,0001, batch 32, clipping 1, máximo 100 épocas y patience 12 son compartidos. No se ejecutó una nueva búsqueda comparable de hiperparámetros por familia: los resultados no son un óptimo arquitectónico global. Diez semillas son un presupuesto de evaluación y no acreditan potencia 0,80 ni un piloto prospectivo del trío.

## Cierre y test de una reproducción de fase I

El cierre exige CV completo y fija D por media de CV, así como las épocas de diciembre. Leer las opciones exactas de `--help`. Se puede señalar un derivado diario explícito con su auditoría, o un `--raw-root` con los ZIP oficiales ya adquiridos. No se descarga ni se abre contenido del test durante registro/CV/ajuste. El manifiesto de adquisición y la identidad esperada se fijan antes del acceso; los 120 pesos se congelan antes de scoring.

```bash
.venv-training/bin/python scripts/run_gru_lstm_tcn.py --phase close \
  --project-root . --output-dir /ruta/nueva/gru-lstm-tcn \
  --raw-root /ruta/adquisicion/oficial
.venv-training/bin/python scripts/run_gru_lstm_tcn.py --phase fit \
  --project-root . --output-dir /ruta/nueva/gru-lstm-tcn
.venv-training/bin/python scripts/run_gru_lstm_tcn.py --phase score \
  --project-root . --output-dir /ruta/nueva/gru-lstm-tcn \
  --raw-root /ruta/adquisicion/oficial
```

La repetición portable ajusta 120 modelos finales frescos. **El experimento observado publicó 80 ajustes nuevos y 40 LSTM idénticas reutilizadas**, con verificación de receta/época/cohorte/escaladores; no confundir ambos conteos. En ambos casos se conserva la misma partición. Test abarca 2025-01-01–2026-08-31, 608 fechas, y sus veinte meses son intervalos de reporte de pesos fijos. Septiembre–octubre de 2026 no se asigna a la reproducción de fase I. Su uso en fase II depende de un protocolo nuevo y explícito, documentado abajo.

No borrar marcadores de acceso, modificar hiperparámetros después de test ni repetir inferencias para escoger resultados favorables. El test observado del proyecto sigue siendo exploratorio aun cuando se cierre una nueva configuración antes de sus propias inferencias.

## Estadística de una reproducción nueva de fase I

El analizador portable consume únicamente los CSV de la salida nueva y escribe sus propios agregados. No exige archivos del experimento privado original ni impone sus 80 ajustes y 40 reusos. Si sólo se ha completado CV, analizar esa fase:

```bash
.venv-training/bin/python scripts/analyze_gru_lstm_tcn_run.py \
  --project-root . --output-dir /ruta/nueva/gru-lstm-tcn --phase cv
```

Después de completar también el scoring final, `--phase all` analiza CV y test. Requiere ambas fases completas; no ejecuta una fase pendiente:

```bash
.venv-training/bin/python scripts/analyze_gru_lstm_tcn_run.py \
  --project-root . --output-dir /ruta/nueva/gru-lstm-tcn --phase all
```

La salida `reproduction_analysis/` conserva métricas, pares, multiplicidad y diagnósticos con alcance exploratorio. No cambia los pesos, el D seleccionado ni los archivos cerrados de CV/test.

## Estadística y exportación de fase I

El exportador actual consume las predicciones guardadas del experimento local y escribe sólo la allowlist pública. Su CLI permite identificar explícitamente la fuente conservada; no es un comando de entrenamiento ni reconstruye un experimento a partir de resúmenes públicos.

```bash
.venv-training/bin/python scripts/build_gru_lstm_tcn_evidence.py --help
```

Los contrastes D7 de fase I entre GRU/LSTM/TCN son **post-hoc**, con Holm de 3 comparaciones por fase, IC marginales y familiares. Todas las entradas tienen sensibilidad Holm de 12 comparaciones. DM utiliza MSE diario medio entre semillas, HAC y corrección HLN; esa pregunta temporal difiere de la variación de inicialización. BDS y Ljung–Box por mes tienen límites de tamaño y dependencia; rechazar iid no demuestra una no linealidad explotable.

El margen δ=0,00025772842664981106 procede del 5 % de la RMSE CV del GRU/D7 de fase I. Se declaró después de aquel test y no se atribuye como decisión previa de ese experimento. La fase II registra esta convención antes de su reserva y ejecuta su propio piloto y dimensionamiento, como se explica a continuación. Un futuro modelo avanzado de E2 necesita otra planificación; no hereda automáticamente el R de fase II. El margen académico no prueba utilidad financiera.

## Reproducir la fase II D7 planificada

La entrada nueva es `scripts/run_gru_lstm_tcn_planned.py`. Se necesita el diseño canónico `experiments/gru_lstm_tcn_planned_20261105/design.json`, incluido en el producto 8.2, los derivados de desarrollo y los metadatos oficiales del producto 8.3. Los pasos de preparación EDA anteriores son los mismos. Elegir una salida vacía **distinta** de los experimentos cerrados; las siguientes rutas son ejemplos:

```bash
.venv-training/bin/python scripts/run_gru_lstm_tcn_planned.py --phase register \
  --project-root . --output-dir /ruta/nueva/estudio-d7 \
  --raw-root /ruta/adquisicion/oficial --workers 3
.venv-training/bin/python scripts/run_gru_lstm_tcn_planned.py --phase hpo \
  --project-root . --output-dir /ruta/nueva/estudio-d7
.venv-training/bin/python scripts/run_gru_lstm_tcn_planned.py --phase pilot \
  --project-root . --output-dir /ruta/nueva/estudio-d7
.venv-training/bin/python scripts/run_gru_lstm_tcn_planned.py --phase plan \
  --project-root . --output-dir /ruta/nueva/estudio-d7
.venv-training/bin/python scripts/run_gru_lstm_tcn_planned.py --phase fit \
  --project-root . --output-dir /ruta/nueva/estudio-d7
.venv-training/bin/python scripts/run_gru_lstm_tcn_planned.py --phase score \
  --project-root . --output-dir /ruta/nueva/estudio-d7 \
  --raw-root /ruta/adquisicion/oficial
.venv-training/bin/python scripts/run_gru_lstm_tcn_planned.py --phase analyze \
  --project-root . --output-dir /ruta/nueva/estudio-d7
.venv-training/bin/python scripts/run_gru_lstm_tcn_planned.py --phase verify \
  --project-root . --output-dir /ruta/nueva/estudio-d7
```

`register` fija hashes de diseño, fuentes y reserva sin abrir sus payloads de precios. HPO consume solo enero–octubre de 2024 y ejecuta ocho candidatos, dos semillas y diez bloques por familia: 960 ajustes. `pilot` congela las recetas seleccionadas y ejecuta diez semillas nuevas por familia sobre el 01/11–30/12/2024, con 60 ajustes. `plan` calcula R mediante t no central y publica el dimensionamiento local; `fit` exige esa etapa y congela después los 3R checkpoints y escaladores. `score` es explícito y exige el cierre completo antes de construir la nueva cohorte. `analyze` consume sus predicciones guardadas; no reajusta ni elige nuevas recetas.

En la ejecución publicada se obtuvo **R=465**, semillas definitivas 8101–8565, 1.395 modelos y 2.790 etapas de ajuste definitivo. El cálculo conserva el objetivo marginal de potencia 0,80, α*=0,05/3, δ=0,00025772842664981106 y una cota superior simultánea del 80 % de las DE pareadas del piloto. No se sustituye por diez semillas ni se detiene por un p-valor favorable. El resultado de una reproducción puede necesitar revisión de su propia planificación si cambian los datos o recetas; no modificar registros cerrados para forzar coincidencia.

La reserva nueva termina el 30/10/2026: sesenta fechas desde el 01/09. Agosto de 2026 es contexto observable y no se cuenta como resultado de evaluación. La adquisición anterior ya había recorrido los CSV por integridad, timestamps y esquema; la afirmación es que no se habían usado para objetivos o resultados de modelos. Esta evaluación histórica preespecificada no es un pronóstico emitido en vivo. Su planificación por semillas no acredita potencia temporal o conjunta del 80 %.

En los resultados publicados, Shapiro–Wilk rechaza normalidad de las diferencias definitivas. Las t e IC registrados se interpretan como aproximaciones para 465 pares, y se conserva esa limitación sin cambiar después al contraste más favorable. Ninguna comparación DM/Holm-6 rechaza y los intervalos temporales familiares no acreditan una mejora superior al 5 %; la ejecución reproducible no equivale a superioridad demostrada.

Los agregados de fase II están en `book/_static/gru_lstm_tcn_planned_20261105/` dentro del producto 8.5 y también en `results/gru_lstm_tcn_planned_20261105/` en el repositorio. Para verificar únicamente esos archivos públicos, sin registrar ni entrenar, usar el verificador genérico del adaptador anterior:

```bash
.venv-training/bin/python scripts/run_gru_lstm_tcn.py --phase verify \
  --evidence-manifest book/_static/gru_lstm_tcn_planned_20261105/public_allowlist_manifest.json
```

Este comando verifica integridad de la publicación, no acredita que la máquina local haya repetido los entrenamientos. Una reproducción ahora realizada sobre esa reserva ya publicada no vuelve a ser una evaluación desconocida.

## Pruebas

```bash
.venv-training/bin/python -m unittest discover -s tests -p 'test_*gru_lstm_tcn*.py'
.venv-training/bin/python -m unittest discover -s tests -p 'test_planned_study*.py'
```

Las pruebas nuevas comprueban forma, gradientes, causalidad, checkpoints y lectura segura de manifiestos. La equivalencia con el código local entrenado y los resultados numéricos está documentada en los recibos de publicación; no se ejecutan entrenamientos reales al construir el sitio.

## Regenerar descargas

Después de guardar el código y el libro en un commit local, construir HTML y ejecutar el empaquetador en una salida externa nueva. Verifica CRC y hashes por miembro y excluye ZIP recursivos, brutos, escaladores, pesos y vectores tabulares individuales. Las figuras incluyen ilustraciones temporales con semillas fijadas: 4101 en fase I y 8101 en fase II, identificadas explícitamente sin escoger la de menor error. Copiar sus ocho ZIP y catálogos a `book/_static/e1_downloads/` y reconstruir el sitio. Los protocolos de origen quedan preservados en `experiments/gru_lstm_tcn_20261105/original_registration/`; los siete productos públicos conservan la terna GRU–LSTM–TCN y los alcances separados de ambas fases.

## Actualizar la literatura de E1 con exportaciones nativas

La etapa del 6 de noviembre de 2026 añade metadatos de cuatro exportaciones WoS/Scopus y dos fichas de originales completos. El corte anterior de 40 referencias, siete textos y sus frecuencias se conserva; la bibliografía canónica añade dos entradas, con objetivos y papeles separados. Los originales de las exportaciones y sus resúmenes no se redistribuyen. La reproducción de la ingesta requiere los cuatro archivos del autor, identificados por SHA-256 en `references/literature_e1/native_search_20261106/source_manifest.json`.

```bash
python scripts/import_e1_literature.py \
  --scopus '/ruta/export_4a31d458-6ce4-4435-9507-df9ee7cc2881_2026-11-06T220747.614885599.csv' \
  --scopus '/ruta/export_694623e0-7e5c-4597-aca5-c8fc7cb8b431_2026-11-06T220948.394330608.csv' \
  --wos '/ruta/savedrecs (4).txt' \
  --wos '/ruta/savedrecs (3).txt' \
  --search-context references/literature_e1/native_search_context_20261106.json
python scripts/import_e1_literature.py --verify-public references/literature_e1/native_search_20261106
python -m unittest discover -s tests -p 'test_e1_literature_import.py'
```

Estos comandos reproducen la ingesta documental de los archivos recibidos; no ejecutan consultas en WoS, Scopus ni IEEE Xplore. La bitácora editable `provisional_native_log_20261107/` se reconstruye desde la procedencia pública conservada:

```bash
python scripts/build_e1_provisional_native_log.py
```

Su estado histórico es **PROVISIONAL**: las exportaciones son compatibles con cotas **≥8 y ≥59 en WoS**, y **≥19 y ≥158 en Scopus**. Esas cotas no son los totales mostrados por las plataformas. Las fechas referenciadas identifican declaraciones o exportaciones y no acreditan fechas de consulta nativa. Las cadenas propuestas permanecen identificadas como propuestas, sin recuperación de los literales históricos. En esa preparación IEEE Xplore conservaba resultado y fecha desconocidos; desconocido no significa cero. Reproducir esa bitácora no ejecuta búsquedas ni cierra por sí solo el requisito 3.1. Se conservan exactamente las **244 ocurrencias y 182 grupos provisionales** de la ingesta, N7=7 y las frecuencias históricas; las cotas quedan fuera de los totales PRISMA.

La ronda real posterior se conserva en `references/literature_e1/native_execution_20261107/` y su espejo público. Sus exportaciones de WoS y Scopus contienen **271 UT y 538 EID distintos**, respectivamente, coincidentes con los totales nativos observados. El autor confirma la fecha **06/11/2026 en Colombia** para las tres consultas. IEEE aporta **32 resultados nativos y 25 exportados**, con filtro de acceso abierto y años 2018–2026; la fecha 06/11 fue confirmada también para IEEE por el autor, y los siete resultados no exportados se declaran. El expediente separa cada campo observado, comunicado por el autor o desconocido: no deduce una hora de consulta desde el nombre del CSV ni fabrica hashes para capturas presenciadas sólo en la conversación. Scopus incluye 2019–2026 en la consulta; WoS conserva cuatro registros de 2017–2018. Los archivos originales permanecen fuera del repositorio. Los **834 registros y 620 grupos provisionales** de esta nueva ronda no se añaden automáticamente al corpus de textos leídos ni alteran el corte histórico de 244/182.

Para reproducir la ingesta de esta ronda con sus archivos privados y el manifiesto de entrada correspondiente:

```bash
python scripts/ingest_e1_native_execution.py \
  --manifest /ruta/privada/manifiesto_ejecuciones.json \
  --output-dir /ruta/nueva/metadatos_publicos
python -m unittest discover -s tests -p 'test_e1_native_execution.py'
```

El esquema y el README del expediente definen el contrato del manifiesto; estos comandos validan evidencia guardada y no ejecutan las consultas en las plataformas. La fecha de día confirmada por el autor se distingue de una marca temporal automática; el total nativo se distingue del tamaño o rango exportado.

La primera pasada automática no decide elegibilidad final ni modifica frecuencias por menciones de arquitecturas. El expediente posterior `editorial_screening_20261106/` registra decisiones de revisión por título/resumen y distingue exclusión, ambigüedad y necesidad de texto completo; no convierte los candidatos en originales leídos. `primary_reviews_20261106/` contiene extracción editorial, ubicación de evidencia y límites de dos textos completos, más tres candidatos con acceso parcial. Su `manifest.json` valida los derivados, sin inventar hashes de originales no materializados. Las entradas nuevas se verifican contra su fragmento `references.bib`; el catálogo histórico de 40 no se reescribe. `selection_synthesis_20261106/` mantiene una matriz de alternativas y una bitácora que separa consultas acreditadas, información declarada y búsquedas pendientes.

La síntesis homogénea posterior se reconstruye desde sus extracciones editoriales, sin abrir originales privados ni entrenar modelos:

```bash
python scripts/build_e1_homogeneous_synthesis.py
python scripts/build_e1_homogeneous_synthesis.py --check
python -m unittest discover -s tests -p 'test_e1_homogeneous_synthesis.py'
```

`homogeneous_synthesis_20261107/` conserva el corte de 47 identidades, con 26 originales primarios leídos y una transcripción completa. Separa doce estudios de volatilidad realizada histórica, diez con comparadores base extraíbles, y sensibilidades por receta, tipo de publicación y objetivo. Sus votos fueron LSTM10, GRU4, MLP3 y TCN1 dentro del estrato base; el MLP de 593 parámetros implementado históricamente se conserva como alternativa real. La terna registrada GRU–LSTM–TCN expresa un contraste entre celdas recurrentes y filtros causales, no las tres frecuencias máximas ni una solución global de complejidad mínima. Los 14 informes no recuperados al corte son una disposición terminal documentada de aquella ronda, con limitación de cobertura; no requieren compra para poder citar alternativas verificadas ni se eliminan del historial.

La ampliación web del **07/11/2026 en Colombia** se reproduce desde sus metadatos revisados, sin volver a consultar las plataformas ni descargar los originales:

```bash
python scripts/build_e1_free_model_expansion.py --check
python scripts/build_e1_homogeneous_expansion.py --check
python -m unittest discover -s tests -p 'test_e1_free_model_expansion.py'
python -m unittest discover -s tests -p 'test_e1_homogeneous_expansion.py'
```

`free_model_expansion_20261107/` documenta **trece obras: siete seleccionadas para TCN y seis para MLP**. Tres ya estaban en la bibliografía y diez son nuevas; el archivo canónico pasa de 60 a **70 referencias**, sin equiparar referencias metodológicas con estudios incluidos. El catálogo conserva autores, DOI editorial o de preprint, versión de texto gratuito, objetivo, papel de modelo y localizador de evidencia. Ge et al. permanece como preprint; Lensky y Hao se cita como publicación IEEE CAI 2024 con su versión de autor gratuita. La lectura primaria indexada de Vo y su limitación de descarga automática no se transforman en lectura íntegra, y la TCN de la propuesta híbrida de Agal no añade por sí sola una referencia autónoma básica.

`homogeneous_expansion_20261107/` concilia las **57 identidades acumuladas** con la misma definición de comparador y conserva los insumos históricos: 35 originales primarios leídos, una transcripción completa y 21 fuentes en otros niveles de acceso. El constructo cripto pasa a 15 experimentos, 13 con referencias extraíbles; el estrato **elegible por suficiencia de reporte contiene diez**, con LSTM10, MLP4, GRU2 y TCN1. Este N10 elegible no es el N10 descriptivo del corte anterior. Recalcula frecuencias por objetivo, ámbito, independencia, lectura y reporte; no interpreta la cuota dirigida 7/6 como frecuencia de la literatura. Los mercados tradicionales y los objetivos implícitos mantienen sus estratos. Las cifras actuales se obtienen de `summary.json`, `adjudications.csv`, `comparator_quality.csv` y `frequencies.csv`, separados del corte anterior de 47 obras. Esta ampliación no ejecuta modelos, cambia la terna experimental o altera los resultados, fechas y configuraciones del benchmark.

Para refrescar las descargas de esta ampliación, usar el refresco específico después de guardar los cambios de fuentes en un commit y construir E1:

```bash
jupyter-book build book --all --warningiserror --keep-going
python scripts/check_book_links.py --book-dir book
python scripts/refresh_e1_native_downloads.py --output-dir /tmp/e1-native-downloads
```

El destino debe estar vacío y fuera del checkout. El refresco utiliza los ZIP vigentes como base y conserva **8.3 byte a byte**. Mantiene los archivos científicos originales de **8.5 y 8.6**: añade a 8.5 la estadística posterior y las curvas de una reproducción identificada, y a 8.6 su protocolo, registro, runtime y lock. **8.4** recibe las etapas nativa, de originales, de cribado, de síntesis y de recuperación por internet, junto con la bibliografía canónica. **8.7** recibe la comparación documental, el contexto académico del margen y la procedencia de las ampliaciones; **8.2** incorpora el código actual y los locks de los suplementos. Los protocolos y resultados de los experimentos históricos no se reescriben.

La ampliación bibliográfica `open_access_replacement_20261107/` se incorpora por su manifiesto público cerrado: **8.4** incluye sus metadatos y síntesis de raíz, subpartes de revisión e insumos derivados bajo `inputs/`; **8.7** incluye sólo `summary.json`, `source_manifest.json` y `manifest.json` de su raíz cuando estén declarados. El constructor `scripts/build_e1_open_access_replacement.py` entra automáticamente en **8.2** como código Python versionado. Esta ampliación de literatura no requiere repetir entrenamientos ni curvas: se consumen exactamente las evidencias de curvas ya guardadas, manteniendo los archivos científicos de **8.5/8.6** y el binario de **8.3**.

La bitácora `provisional_native_log_20261107/` usa el mismo manifiesto cerrado: **8.4** incorpora todos sus archivos aprobados y **8.7** sólo el resumen y el manifiesto de raíz. Su constructor entra automáticamente en **8.2** como código Python versionado. El refresco conserva el estado histórico de esa preparación; las ejecuciones acreditadas posteriores se añaden por separado en `native_execution_20261107/`, junto a `homogeneous_synthesis_20261107/`, con todos sus miembros aprobados en **8.4** y sus resúmenes/manifiestos en **8.7**. No ejecuta consultas, entrenamientos o curvas ni cambia los archivos científicos de **8.5/8.6** o el binario de **8.3**.

Las etapas `free_model_expansion_20261107/`, `homogeneous_expansion_20261107/` y `core_selection_20261107/` emplean ese mismo contrato cerrado: **8.4** recibe sus fichas, tablas, bibliografía y demás metadatos aprobados; **8.7** recibe únicamente sus resúmenes/manifiestos de raíz; sus constructores y pruebas entran en **8.2**. La edición anterior del paquete se identifica como `20261107_primary_608`, con la fecha real del 07/11/2026 en Colombia. La edición `20261107_remediation_608` conserva también ese registro dentro de 8.7, junto con los registros anteriores `literature_refresh_20261107_core_mlp.json`, `literature_refresh_20261107_free_models.json`, `literature_refresh_20261107.json` y `AMPLIACION_LITERARIA_20261107.md`, además de los del 06/11: un refresco posterior del mismo día no borra la procedencia ya publicada.

El suplemento `results/e1_statistics_supplement_20261106/`, reflejado en el libro, calcula Q–Q, Friedman/Kendall W y sensibilidad Wilcoxon/Holm desde las métricas por semilla guardadas; es **posterior** a la evaluación original y no acredita potencia temporal ni nuevos ajustes. La fundamentación `e1_academic_margin_20261106/` consume sólo desarrollo y mantiene δ y R. Sus manifiestos fijan insumos y productos públicos.

Las curvas de `e1_curve_reproduction_20261106/` corresponden a **ajustes nuevos de desarrollo**, con su [protocolo y registro propios](experiments/e1_curve_reproduction_20261106/README.md). No se presentan como recuperación de los logs originales ni sustituyen métricas o pesos históricos. Para usar este código desde los ZIP, extraer también **8.6**, que contiene `experiments/e1_curve_reproduction_20261106/` y su lock, junto a 8.2/8.3/8.5. La exportación pública contiene pérdidas y metadatos; no incorpora predicciones diarias, observaciones de reserva, pesos ni escaladores. La comparación documental anterior permanece en `e1_dataset_comparison_20261106/`.

El lock de curvas conserva el paquete `torch==2.15.0+cpu` y requiere el índice CPU explícito al instalarlo con Python 3.12.14. Este comando completa la instrucción de instalación del README congelado, sin modificar ese protocolo:

```bash
python -m pip install -r requirements-e1-curves.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Auditoría forense y repetibilidad actual controlada

`results/e1_replay_forensics_20261107/` inspecciona los artefactos de la reproducción completa y documenta la búsqueda de originales sin entrenar redes. Conserva 29 cambios de época de TCN frente a los registros originales; su causa no queda identificada. La falta de un archivo privado original impide comparar sus valores con una tolerancia numérica: un SHA256 no reconstruye pérdidas o tensores. La guía §4.3 exige procedimiento, semillas, dependencias y declarar si se usa determinismo estricto; no exige identidad histórica bit a bit. El procedimiento actual y la identidad histórica tienen alcances distintos.

La comprobación adicional de `experiments/e1_replay_audit_20261107/` se publicó y verificó remotamente antes de su registro y nuevos ajustes. Desde un checkout limpio e inmutable de la fuente publicada, con Python 3.12.14 y el lock anterior en un entorno externo al checkout:

```bash
git worktree add --detach /ruta/checkout_diagnostico 79f8dcc1a4c3bfacfd1fc56c0ecb43d8e0185501
cd /ruta/checkout_diagnostico
PYTHONHASHSEED=0 /ruta/venv/bin/python scripts/audit_e1_replay_forensics.py register \
  --private-dir /ruta/privada/nueva \
  --published-commit 79f8dcc1a4c3bfacfd1fc56c0ecb43d8e0185501
PYTHONHASHSEED=0 /ruta/venv/bin/python scripts/audit_e1_replay_forensics.py fit \
  --private-dir /ruta/privada/nueva --replicate a
PYTHONHASHSEED=0 /ruta/venv/bin/python scripts/audit_e1_replay_forensics.py fit \
  --private-dir /ruta/privada/nueva --replicate b
/ruta/venv/bin/python scripts/audit_e1_replay_forensics.py compare \
  --private-dir /ruta/privada/nueva \
  --prior-private-dir /ruta/privada/reproduccion_completa \
  --output-dir /ruta/nueva/resultado_diagnostico
```

La verificación remota previa debe conservarse aparte; el argumento `--published-commit` por sí solo no prueba publicación. Las rutas privadas y de salida deben ser nuevas. Las réplicas son invocaciones independientes y secuenciales; la CLI impide reutilizar una réplica ya ajustada. No se vuelve a ejecutar `prepare` sobre el protocolo publicado ni se reintenta una semilla para obtener coincidencia. Esta ejecución es optativa para reproducir el diagnóstico y nunca forma parte del build o del empaquetado.

El panel dirigido de seis celdas produjo **12 trayectorias y 24 etapas**, con **18/18 hashes canónicos iguales entre réplicas** y también frente a esas seis celdas de la reproducción completa anterior; la diferencia numérica máxima fue cero. Satisface el criterio secundario fijado de `rtol=1e-6`, `atol=1e-8` en las seis celdas. `results/e1_replay_repeatability_20261107/` publica comparación, registro, entorno, ejecución y prueba de publicación previa, sin pesos, escaladores o predicciones diarias. No se consultó la reserva ni se añadieron muestras independientes al benchmark. El resultado se limita a ese panel: no recupera los archivos originales ni certifica la repetibilidad de todo R465.

El empaquetador admite únicamente las listas explícitas de esos manifiestos, comprueba hashes/bytes, estructura PNG, campos privados, CRC y rutas seguras. Los stages anidados `search_recovery_20261106`, `internet_closure_20261106`, `open_access_replacement_20261107`, `provisional_native_log_20261107`, `homogeneous_synthesis_20261107`, `native_execution_20261107`, `free_model_expansion_20261107`, `homogeneous_expansion_20261107` y `core_selection_20261107` requieren `publication_schema: e1_public_metadata_allowlist_v1`, una lista ordenada sin duplicados `approved_public_files` y `files: {ruta: {sha256, bytes}}` con exactamente los mismos miembros. La auditoría forense y la repetibilidad controlada usan ese mismo contrato en `public_allowlist_manifest.json`: sus archivos aprobados entran en **8.5**; resúmenes y manifiestos en **8.7**; para repetibilidad, registro, runtime, ejecución y verificación de publicación también entran en **8.6/8.7**. El histórico compatible `e1_historical_core_20261107/` y la exportación cerrada nueva `e1_core_mlp_extension_20261107/` usan el mismo contrato en `manifest.json`; van a **8.5** con sus resúmenes/manifiestos en **8.7**. La exportación nueva añade a **8.6/8.7** registro, selección, plan, publicación del plan, cronología, runtime y manifiesto de fuentes. Sus fuentes/diseño y los protocolos de las reproducciones viajan en **8.6/8.7** y sus ejecutores en **8.2**. Sólo se leen esos metadatos admitidos, nunca archivos vecinos sin listar, originales PDF ni payloads experimentales privados. La etapa principal `e1_primary_608_20261107/` se añade por su manifiesto cerrado a **8.5**, con sus registros de configuración y procedencia en **8.6/8.7**. El catálogo histórico de esa etapa usa exclusivamente su resumen y las métricas originales D7: comprueba R=10, 608 fechas de test, 366 de CV, recetas 32/32/16, semillas 4101–4110, Persistencia y hashes de las fuentes guardadas. La exportación cerrada de sesenta fechas también se conserva íntegra como complemento; su metadata previa y su snapshot histórico quedan en `complementary_study_metadata`, sin sobrescribir `historical_study_metadata`. El empaquetador falla ante ausencia o mezcla de cohortes. Renueva catálogo y paquete conjunto; no regenera la ciencia desde el empaquetador histórico. La edición `20261107_remediation_608`, con el nuevo export cerrado, conserva `20261107_primary_608`, `20261107_core_mlp` y las ediciones anteriores del propio 07/11 y del 06/11. Copiar los productos verificados a `book/_static/e1_downloads/` y reconstruir E1 para distribuir las descargas actualizadas. La compilación posterior sólo actualiza enlaces y recursos de distribución; el catálogo identifica el commit de fuentes usado para generar los ZIP, separado del commit posterior que agrega los propios paquetes.

## Ampliación bibliográfica E2 del 08/11/2026

`references/literature_e2/native_20261108/` conserva una ronda independiente de los cortes históricos. La vista del libro se genera desde una lista cerrada de metadatos; nunca desde exportaciones privadas durante el build:

```bash
python scripts/publish_e2_native_literature.py
python scripts/publish_e2_native_literature.py --check
python -m unittest discover -s tests -p 'test_e2_native_literature.py'
jupyter-book build book_e2 --all --warningiserror --keep-going
python scripts/check_book_links.py --book-dir book_e2
```

La validación comprueba 1.634 ocurrencias WoS/Scopus/IEEE, 1.078 grupos, 556 redundancias, 1.054 grupos en ventana, clases y 21 correcciones focalizadas. Por separado comprueba 1.199 apariciones arXiv, 1.004 IDs, pertenencia por consulta, 983 pendientes y Bitcoin como subconjunto de cripto. No suma universos que aún no están conciliados ni transforma etiquetas provisionales en decisiones finales de elegibilidad.

Los scripts de triaje privados distribuidos en `native/` deben copiarse con sus insumos a una carpeta externa a Git: rechazan ejecutarse en un checkout para evitar escribir resúmenes completos en el repositorio. Los originales, abstracts completos y cuerpos no forman parte del manifiesto público. Los JSON de protocolo, registros de ajuste, datos, resultados y descargas históricas no se regeneran ni modifican por esta ampliación.

## Seguimiento agregado de revisión E2 del 08/11/2026

`references/literature_e2/title_abstract_review_20261108/` añade agregados de la lectura individual asistida por IA de 59 prioritarios y de la auditoría de 42/337 exclusiones propuestas. Conserva el ledger de triaje anterior y no exporta resúmenes, citas textuales ni tablas privadas de decisiones. El plan de fases es editorial y no cambia registros experimentales.

```bash
python scripts/build_e2_review_progress.py --check
python -m unittest discover -s tests -p 'test_e2_review_progress.py'
```

Para reconstruir exclusivamente los agregados desde las dos tablas privadas, usar `python scripts/build_e2_review_progress.py --private-dir /ruta/privada/fuera-del-checkout`; los nombres requeridos están en el script. La lista pública está cerrada y su manifiesto fija hashes y bytes. No se confunden 40 candidatos a texto completo con incluidos; 295 exclusiones siguen sin lectura individual. La muestra mixta de azar y títulos de riesgo no permite extrapolar tasa de error global. Las alertas bibliográficas son preguntas de verificación, no duplicados o fechas corregidos automáticamente.

### Seguimiento de lectura y recuperación E2 (08/11/2026)

`python scripts/build_e2_screening_followup.py --check` valida conteos agregados,
campos públicos permitidos, hashes y espejo del segundo corte. El modo opcional
`--private-base /ruta/privada/corpus_e2` contrasta identidad de los 295 registros,
su unión disjunta con los 42 previos y la cohorte de recuperación de 40; esa ruta
no puede estar dentro del repositorio. No publica resúmenes ni originales.
Ejecutar `python -m unittest discover -s tests -p 'test_e2_screening_followup.py'`.
La lectura asistida por IA no representa doble revisión humana ni inclusión final.

Segundo corte de recuperación: `python scripts/build_e2_recovery_second.py --check`
valida el agregado 22/40, las versiones separadas (seis editoriales y dos
preprints), campos permitidos y espejo. Pruebas:
`python -m unittest discover -s tests -p 'test_e2_recovery_second.py'`.
Los cortes previos permanecen históricos; no se reinterpretan como inclusión final.

### Catálogo editorial vigente del 08/11/2026

La presentación E2 usa diez modelos y un candidato HAR residual, sin atribuirle
superioridad ni ejecución integral. `references/e2_current_catalogue_20261108.json`
no es un preregistro experimental. Validar con
`python -m unittest discover -s tests -p 'test_e2_current_catalogue.py'`.
Los protocolos, resultados y recibos históricos permanecen sin cambios; la
lista retirada no se reimprime como selección vigente en capítulos o anexos.

### Sincronización editorial E1/E2/E3

`references/books_alignment_20261108.json` y sus tres espejos públicos documentan
la referencia E1 actual y el candidato CNN-1D + LSTM + atención temporal.
No son un nuevo registro experimental. Validar con
`python -m unittest discover -s tests -p 'test_books_alignment.py'`.
Las huellas verifican que fuentes, plan y métricas E1 no se modificaron.
Construir cada libro por separado y ejecutar `scripts/check_book_links.py`.
