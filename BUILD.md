# Construcción y reproducción: GRU–LSTM–TCN

El JBook vigente evalúa **GRU, LSTM y TCN** sobre Bitcoin, con entradas7/14/21/28, σ₇ poblacional y horizonte1día. El baseline es **GRU/D7**, seleccionado mediante CV 2024. La implementación usa únicamente OHLCV ya observado; escaladores y épocas se ajustan con entrenamiento permitido. El test histórico 2025–agosto2026 ya estaba observado: la evidencia publicada y los contrastes dentro del trío son exploratorios.

## Construcción del libro

Se requiere Python 3.12 y Jupyter Book 1.0.4.post1. El build consume tablas/figuras guardadas y no entrena modelos, descarga mercado o ejecuta inferencias.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-book.lock.txt
.venv/bin/jupyter-book build book --all --warningiserror --keep-going
.venv/bin/python scripts/check_book_links.py
.venv/bin/python -m http.server 8000 --directory book/_build/html
```

Abrir `http://localhost:8000`. En Windows adaptar la activación del entorno. No regenerar resúmenes experimentales para construir HTML.

## Entornos

La librería CV utiliza NumPy menor que 2 y tiene un entorno propio. Los modelos y análisis usan el lock de entrenamiento observado; no sustituir versiones por la última disponible.

```bash
python3.12 -m venv .venv-cv
.venv-cv/bin/python -m pip install -r requirements-cv.txt
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

El README del dataset documenta la adquisición y los scripts de EDA del repositorio producen los derivados con calendario UTC, sin comprimir faltantes. Los manifiestos de folds D7/D14/D21/D28 provienen de llamadas reales a `split_train_val_forwardChaining(np.arange(731),D,1,1)`, agrupadas explícitamente en doce meses de 2024. La agrupación mensual no es una API nativa mensual ni366 refits diarios. Se conserva embargoD y validación interna de 90 días.

Los derivados diarios y las predicciones individuales del experimento observado no forman parte del repositorio público. Una verificación completa de sus checkpoints requiere la evidencia conservada y sus hashes; un clon público puede verificar los agregados permitidos sin fingir haber recalculado pesos privados.

## Verificar la publicación sin entrenar

El manifiesto público enumera únicamente archivos agregados y metadatos:

```bash
.venv-training/bin/python scripts/run_gru_lstm_tcn.py --phase verify \
  --evidence-manifest results/gru_lstm_tcn_20261105/public_allowlist_manifest.json
```

La verificación no registra nuevos ajustes ni abre test. `results/gru_lstm_tcn_20261105/` y su copia estática tienen contenido idéntico por allowlist.

## Reproducir una ejecución nueva

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

`run` genera un registro si la salida todavía no existe y conserva/verifica los ajustes completados al reanudarse. `--limit-trajectories` es sólo una ejecución técnica parcial; un resultado incompleto no permite cerrar test. El conjunto completo tiene120 trayectorias,1.440 segmentos mensuales y2.880 ajustes.

Los hiperparámetros están fijos: GRU/LSTM ancho 32, LR 0,003; TCN ancho 16, LR 0,001. AdamW, dropout 0,1, weight_decay 0,0001, batch 32, clipping 1, máximo 100 épocas y patience 12 son compartidos. No se ejecutó una nueva búsqueda comparable de hiperparámetros por familia: los resultados no son un óptimo arquitectónico global. Diez semillas son un presupuesto de evaluación y no acreditan potencia 0,80 ni un piloto prospectivo del trío.

## Cierre y test de una reproducción

El cierre exige CV completo y fija D por mediaCV, así como las épocas de diciembre. Leer las opciones exactas de `--help`. Se puede señalar un derivado diario explícito con su auditoría, o un `--raw-root` con los ZIP oficiales ya adquiridos. No se descarga ni se abre contenido del test durante registro/CV/ajuste. El manifiesto de adquisición y la identidad esperada se fijan antes del acceso; los120pesos se congelan antes de scoring.

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

La repetición portable ajusta120 modelos finales frescos. **El experimento observado publicó80 ajustes nuevos y40 LSTM idénticas reutilizadas**, con verificación de receta/época/cohorte/escaladores; no confundir ambos conteos. En ambos casos se conserva la misma partición. Test abarca2025-01-01–2026-08-31,608 fechas, y sus veinte meses son intervalos de reporte de pesos fijos. Septiembre–octubre de 2026 no se asigna automáticamente.

No borrar marcadores de acceso, modificar hiperparámetros después de test ni repetir inferencias para escoger resultados favorables. El test observado del proyecto sigue siendo exploratorio aun cuando se cierre una nueva configuración antes de sus propias inferencias.

## Estadística de una reproducción nueva

Después de CV o del scoring final, el analizador portable consume únicamente los CSV de la salida nueva y escribe sus propios agregados. No exige archivos del experimento privado original ni impone sus80 ajustes+40reusos.

```bash
.venv-training/bin/python scripts/analyze_gru_lstm_tcn_run.py \
  --project-root . --output-dir /ruta/nueva/gru-lstm-tcn --phase all
```

La salida `reproduction_analysis/` conserva métricas, pares, multiplicidad y diagnósticos con alcance exploratorio. No cambia los pesos, el D seleccionado ni los archivos cerrados de CV/test.

## Estadística y evidencias observadas

El exportador actual consume las predicciones guardadas del experimento local y escribe sólo la allowlist pública. Su CLI permite identificar explícitamente la fuente conservada; no es un comando de entrenamiento ni reconstruye un experimento a partir de resúmenes públicos.

```bash
.venv-training/bin/python scripts/build_gru_lstm_tcn_evidence.py --help
```

Los contrastes D7 entre GRU/LSTM/TCN son **post-hoc**, con Holm3 por fase, IC marginales y familiares. Todas las entradas tienen sensibilidad Holm12. DM utiliza MSE diario medio entre semillas, HAC y corrección HLN; esa pregunta temporal difiere de la variación de inicialización. BDS y Ljung–Box por mes tienen límites de tamaño y dependencia; rechazar iid no demuestra una no linealidad explotable.

El margen operativo futuro es5 % del RMSE CV de GRU/D7,0,00025772842664981106. Se declara después del test actual, con sensibilidad2,5/10 %; requiere su propio registro/piloto/evaluación futura. No sustituye el margen histórico del protocolo fuente ni prueba utilidad financiera.

## Pruebas

```bash
.venv-training/bin/python -m unittest discover -s tests -p 'test_*gru_lstm_tcn*.py'
```

Las pruebas nuevas comprueban forma, gradientes, causalidad, checkpoints y lectura segura de manifiestos. La equivalencia con el código local entrenado y los resultados numéricos está documentada en los recibos de publicación; no se ejecutan entrenamientos reales al construir el sitio.

## Regenerar descargas

Después de guardar el código y el libro en un commit local, construir HTML y ejecutar el empaquetador en una salida externa nueva. Verifica CRC y hashes por miembro y excluye ZIP recursivos, brutos, escaladores, pesos y vectores tabulares individuales. Las figuras incluyen una ilustración temporal fija de la semilla 4101, identificada explícitamente. Copiar sus ocho ZIP y catálogos a `book/_static/e1_downloads/` y reconstruir el sitio. Los protocolos de origen quedan preservados en `experiments/gru_lstm_tcn_20261105/original_registration/`; los siete productos públicos corresponden únicamente a la terna vigente.
