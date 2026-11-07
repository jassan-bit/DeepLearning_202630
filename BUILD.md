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

E2 ya cuenta con la guía completa de 65 páginas y capítulos 1–14. La ruta vigente es F2a: series de tiempo, con contrato D7×5 y δ heredados conservados, veinte configuraciones y quince recetas ajustables T12. El commit `599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0` registra el protocolo antes de los ajustes P0/OOF de Bitcoin. Se completaron las 220 trayectorias y 440 ajustes P0/OOF, y se analizaron 305 casos por receta. Ese experimento de desarrollo se distingue de la réplica ETTh1 y del benchmark final, todavía pendiente. Las secciones siguientes sobre GRU–LSTM–TCN describen E1.

## Implementación final de E2

La nueva fuente está en `experiments/e2_completion_20261106/` y `scripts/run_e2_final_benchmark.py`, con módulos de arquitectura, estadística, incertidumbre, diagnósticos y perfilado. Sus pruebas usan datos sintéticos; aún no acreditan una tabla final ni su reproducción independiente. La enmienda debe registrarse antes del primer ajuste real con esta implementación. Mantiene T12/quince asignaciones; conserva once candidatos aleatorios y añade c12 fuente por red. P1 y calibración A usan 53 fechas (08/11–30/12/2024), selección/ajuste hasta el 31/10 y embargo 01–07/11. El primario conserva el corte 24/12 y test conocido de sesenta fechas. Todos los pesos primarios/A/controles negativos y calibradores deben cerrar en ese orden antes del scoring del test.

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

La primera pasada automática no decide elegibilidad final ni modifica frecuencias por menciones de arquitecturas. El expediente posterior `editorial_screening_20261106/` registra decisiones de revisión por título/resumen y distingue exclusión, ambigüedad y necesidad de texto completo; no convierte los candidatos en originales leídos. `primary_reviews_20261106/` contiene extracción editorial, ubicación de evidencia y límites de dos textos completos, más tres candidatos con acceso parcial. Su `manifest.json` valida los derivados, sin inventar hashes de originales no materializados. Las entradas nuevas se verifican contra su fragmento `references.bib`; el catálogo histórico de 40 no se reescribe. `selection_synthesis_20261106/` mantiene una matriz de alternativas y una bitácora que separa consultas acreditadas, información declarada y búsquedas pendientes.

Para refrescar las descargas de esta ampliación, usar el refresco específico después de guardar los cambios de fuentes en un commit y construir E1:

```bash
jupyter-book build book --all --warningiserror --keep-going
python scripts/check_book_links.py --book-dir book
python scripts/refresh_e1_native_downloads.py --output-dir /tmp/e1-native-downloads
```

El destino debe estar vacío y fuera del checkout. El refresco utiliza los ZIP vigentes como base y conserva **8.3 byte a byte**. Mantiene los archivos científicos originales de **8.5 y 8.6**: añade a 8.5 la estadística posterior y las curvas de una reproducción identificada, y a 8.6 su protocolo, registro, runtime y lock. **8.4** recibe las etapas nativa, de originales, de cribado, de síntesis y de recuperación por internet, junto con la bibliografía canónica. **8.7** recibe la comparación documental, el contexto académico del margen y la procedencia de las ampliaciones; **8.2** incorpora el código actual y los locks de los suplementos. Los protocolos y resultados de los experimentos históricos no se reescriben.

El suplemento `results/e1_statistics_supplement_20261106/`, reflejado en el libro, calcula Q–Q, Friedman/Kendall W y sensibilidad Wilcoxon/Holm desde las métricas por semilla guardadas; es **posterior** a la evaluación original y no acredita potencia temporal ni nuevos ajustes. La fundamentación `e1_academic_margin_20261106/` consume sólo desarrollo y mantiene δ y R. Sus manifiestos fijan insumos y productos públicos.

Las curvas de `e1_curve_reproduction_20261106/` corresponden a **ajustes nuevos de desarrollo**, con su [protocolo y registro propios](experiments/e1_curve_reproduction_20261106/README.md). No se presentan como recuperación de los logs originales ni sustituyen métricas o pesos históricos. Para usar este código desde los ZIP, extraer también **8.6**, que contiene `experiments/e1_curve_reproduction_20261106/` y su lock, junto a 8.2/8.3/8.5. La exportación pública contiene pérdidas y metadatos; no incorpora predicciones diarias, observaciones de reserva, pesos ni escaladores. La comparación documental anterior permanece en `e1_dataset_comparison_20261106/`.

El lock de curvas conserva el paquete `torch==2.15.0+cpu` y requiere el índice CPU explícito al instalarlo con Python 3.12.14. Este comando completa la instrucción de instalación del README congelado, sin modificar ese protocolo:

```bash
python -m pip install -r requirements-e1-curves.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

El empaquetador admite únicamente las listas explícitas de esos manifiestos, comprueba hashes/bytes, estructura PNG, campos privados, CRC y rutas seguras. Los stages anidados `search_recovery_20261106` e `internet_closure_20261106` requieren `publication_schema: e1_public_metadata_allowlist_v1`, una lista ordenada `approved_public_files` y `files: {ruta: {sha256, bytes}}` con exactamente los mismos miembros. Sólo se leen esos metadatos admitidos, nunca archivos vecinos sin listar, originales PDF ni payloads experimentales privados. Renueva catálogo y paquete conjunto; no regenera la ciencia desde el empaquetador histórico. Copiar los productos verificados a `book/_static/e1_downloads/` y reconstruir E1 para distribuir las descargas actualizadas. La compilación posterior sólo actualiza enlaces y recursos de distribución; el catálogo identifica el commit de fuentes usado para generar los ZIP, separado del commit posterior que agrega los propios paquetes.
