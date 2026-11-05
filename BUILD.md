# Construcción y reproducción de las dos fases GRU–LSTM–TCN

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

Los comandos experimentales del resto de este documento describen **E1**. E2 tiene introducción y sección 1 desarrolladas a partir de las páginas 1–3 recibidas; el resto de su guía, catálogo y protocolo siguen pendientes. E3 conserva estructura inicial. No se han ejecutado experimentos de E2/E3.


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

## Reproducir una ejecución nueva de fase I

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
