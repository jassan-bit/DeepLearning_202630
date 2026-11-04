# Construcción y reproducción del Entregable 1

El libro vigente evalúa **MLP, LSTM y CNN1D con entradas D=7/14/21/28**, objetivo **σ7 poblacional** y horizonte **h=1 día**. Usa la validación forward chaining de `timeseries-cv` 0.1.5, proyecto de DidierRLopes con coautoría de Filipe R. Ramos. Los índices diarios se agrupan explícitamente en doce bloques mensuales de 2024; el embargo es D y cada escalador se ajusta solo con el entrenamiento admitido.

La configuración se cerró antes del test. El ajuste final y la evaluación **ya se completaron**:

| Etapa | Evidencia vigente |
|---|---|
| Historia inicial | 2023 |
| Validación forward | Doce bloques mensuales de 2024; diez semillas por modelo/historia |
| CV principal conjunta | 120 trayectorias, 1.440 segmentos mensuales, 2.880 ajustes y 43.920 predicciones |
| Reutilización D7 | Treinta trayectorias de la representación relativa original |
| Nueva CV D14/21/28 | Noventa trayectorias y 2.160 ajustes |
| Ajuste final | 120 modelos entrenados solo con desarrollo permitido de 2023–2024 |
| Test | 01/01/2025–31/08/2026; 608 fechas comunes y 72.960 predicciones |
| Actualización durante test | Ninguna; sus veinte meses son intervalos de reporte de los mismos pesos |
| Fuera del experimento | Septiembre–octubre de 2026 no se asigna automáticamente al test |

Las configuraciones y las épocas finales se fijaron usando desarrollo. MLP/D7 permanece como selección previa; el ranking observado en test no modifica esa elección. Los registros firmados están en `experiments/filipe_multihistory_20261104/protocol.json` y `freeze.json`.

## Construir el HTML público

Construir el libro usa los resúmenes y figuras guardados. No descarga datos, ajusta pesos ni ejecuta pronósticos. Entorno probado: Python 3.12 y Jupyter Book 1.0.4.post1 (**v1**, no v2).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-book.lock.txt
jupyter-book build book --all --warningiserror --keep-going
python scripts/check_book_links.py
python -m http.server 8000 --directory book/_build/html
```

Abrir <http://localhost:8000>. En Windows, activar con `.venv\Scripts\activate`. El HTML se escribe en `book/_build/html/`. El build trata advertencias como errores y el comprobador verifica enlaces locales y anclas. Las fórmulas usan MathJax desde CDN.

## Entornos experimentales separados

La librería CV requiere NumPy menor que 2; el entrenamiento y los análisis usan su propio entorno fijado. No mezclar los locks:

```bash
python3.12 -m venv .venv-cv
.venv-cv/bin/python -m pip install -r requirements-cv.txt
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements-training.lock.txt
```

Los locks registran las versiones concretas; no se sustituyen por «la última versión» al verificar la ejecución. La adquisición de Binance está documentada en `data/README.md`, con script y manifiestos SHA-256. Los brutos de un minuto están fuera del checkout y **no son necesarios para verificar los resultados guardados**.

Las pruebas de índices se ejecutan en CV; las demás, en entrenamiento:

```bash
.venv-cv/bin/python -m unittest discover -s tests -p 'test_dl_r2_cv.py'
.venv-cv/bin/python -m unittest discover -s tests -p 'test_filipe_multihistory_cv.py'
.venv-training/bin/python -m unittest discover -s tests
```

Los tests de CV se omiten deliberadamente cuando falta esa dependencia en entrenamiento; se ejecutan en el entorno dedicado, como arriba. Los folds firmados ya están versionados. Para reconstruir índices con llamadas reales a la librería, escribir en un directorio **distinto** y comparar los índices:

```bash
.venv-cv/bin/python scripts/prepare_filipe_multihistory.py --histories 14 21 28 --output-dir /tmp/e1-filipe-index-check
```

No sobrescribir los manifiestos firmados: el timestamp documental también forma parte de sus hashes. La llamada efectiva es `split_train_val_forwardChaining(np.arange(731), D, 1, 1)`; la agrupación mensual es una adaptación explícita del proyecto, no una API mensual nativa ni 366 reajustes.

## Evidencia pública y entrega privada

La vista pública `book/_static/filipe_multihistory_20261104/` contiene **resúmenes agregados**, BDS por mes/semilla, código, fechas, calidad y firmas. `public_allowlist_manifest.json` identifica los archivos permitidos. No contiene los nuevos vectores individuales, series diarias, escaladores, checkpoints o archivos TAR completos.

La evidencia completa se conserva localmente. Un evaluador necesita el **paquete privado de evidencia** para recalcular las métricas y verificar los pesos guardados; un clon con solo resúmenes no basta. El paquete se entrega por un canal privado autorizado, acompañado de su manifiesto SHA-256. No es necesario publicar esos archivos para que la evaluación sea reproducible.

Se conservan seis archivos de corridas: el TAR original D7 y cinco de la ampliación (CV D14, D21, D28, ajuste final y predicciones por corrida). El paquete incluye derivados diarios y predicciones ya existentes, no los ZIP brutos de Binance, credenciales o `.git`.

## Crear una entrega privada y verificar una réplica aislada

`scripts/restore_e1_private_artifacts.py` tiene modos `bundle` y `restore`. Rechaza un destino no vacío o dentro del origen y preserva los registros originales. Verifica SHA-256 de los seis TAR, rutas, tipos, duplicados y hashes de cada miembro antes de restaurarlos; no usa `extractall`.

Desde el checkout local **completo**, crear una entrega privada en otro directorio:

```bash
.venv-training/bin/python scripts/restore_e1_private_artifacts.py \
  --source-root . \
  --output-dir /ruta/privada/e1-entrega \
  --mode bundle
```

Se copia solo el código, locks, metadatos e insumos necesarios y los seis TAR verificados. `private_e1_bundle_manifest.json` registra nombres, bytes y SHA-256. Compartir ese directorio privadamente, conservando su estructura; no incorporarlo a `book/_static`, al repositorio público ni a `gh-pages`.

El evaluador crea el entorno con el lock incluido y ejecuta:

```bash
python3.12 -m venv /ruta/privada/e1-entorno
/ruta/privada/e1-entorno/bin/python -m pip install -r /ruta/privada/e1-entrega/requirements-training.lock.txt
/ruta/privada/e1-entorno/bin/python /ruta/privada/e1-entrega/scripts/restore_e1_private_artifacts.py \
  --source-root /ruta/privada/e1-entrega \
  --output-dir /ruta/privada/e1-replica \
  --mode restore --verify
```

La réplica conserva los mismos archivos firmados y su marcador de primer acceso. El auditor independiente se ejecuta **dentro de la copia** y escribe `verification/independent_saved_result_audit.json` y `.md`. `replication_receipt.json` contiene resultado, cantidades y hashes; puede compartirse sin incluir vectores ni pesos.

La comprobación reconstruye σ7 con fechas, escalado de entrenamiento, cohortes y purgas, selección previa, épocas de diciembre, métricas y BDS, y verifica que los 120 modelos finales preceden al test. No abre ZIP de mercado, ajusta redes ni ejecuta nuevos pronósticos. Verificar este backtest guardado no lo convierte en una evaluación nueva desconocida.

Para repetir únicamente la auditoría de una réplica ya restaurada:

```bash
/ruta/privada/e1-entorno/bin/python /ruta/privada/e1-replica/scripts/verify_filipe_multihistory.py \
  --phase test \
  --report /ruta/privada/e1-replica/verification/recheck.json
```

Si faltan insumos privados, un TAR no coincide con su manifiesto o el destino contiene evidencia, la restauración falla. No se deben borrar marcadores ni sobrescribir resultados para forzarla.

## Análisis y exportación desde la evidencia completa

Los analizadores consumen predicciones guardadas; su salida se distingue de ejecutar los modelos:

```bash
.venv-training/bin/python scripts/analyze_filipe_multihistory.py --phase cv
.venv-training/bin/python scripts/analyze_filipe_multihistory.py --phase test
.venv-training/bin/python scripts/verify_filipe_multihistory.py --phase test --report /ruta/privada/auditoria.json
.venv-training/bin/python scripts/export_filipe_multihistory_book.py --phase complete
```

Para conservar byte a byte la instantánea firmada, ejecutar los análisis en una copia restaurada o verificar primero la salida existente; no sobrescribir originales para cambiar decisiones. La exportación pública usa una lista explícita de resúmenes y no publica los archivos privados.

El suplemento inicial `filipe_statistics_20261104` conserva su alcance **D7/2024**, con Holm-12, DM y bootstrap de desarrollo. No se traslada automáticamente a otras historias ni al test. Las ampliaciones estadísticas posteriores al examen de test se identifican como suplementos exploratorios, no como preregistros anteriores al entrenamiento.

## Nuevos entrenamientos

Los ejecutores originales conservan rutas y firmas del experimento cerrado. El ejecutor final bloquea `--phase fit` después del primer acceso al test: es una protección del registro, no un error que deba evitarse borrando `raw_access_started.json`.

Para evaluar cambios de arquitectura, hiperparámetros o selección se necesita una **versión nueva**, con directorio y protocolo propios, piloto y evaluación independiente según corresponda. La réplica de evidencia no reentrena, no abre otra fase experimental ni reasigna septiembre–octubre de 2026.

Las semillas miden variación de entrenamiento condicionada al calendario; no acreditan por sí solas potencia 0,80 ni independencia entre mercados. Siguen documentados los pendientes bibliográficos y de dimensionamiento prospectivo; el ajuste final y el test de las cuatro entradas no son tareas pendientes.
