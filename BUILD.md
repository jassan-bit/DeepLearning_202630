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

Para refrescar los metadatos de la ampliación bibliográfica usando únicamente los archivos públicos verificados:

```bash
python scripts/refresh_e1_literature_delivery.py
python scripts/update_book_item_audit.py
```

El primer comando conserva por hash los 1.440 segmentos de configuración de CV y 120 ajustes finales ya auditados. Actualiza literatura y declaraciones sin necesitar pesos privados, entrenar modelos o ejecutar pronósticos. Los 39 registros bibliográficos separan seis textos BTC de T1 (original financiero de revista), T2 (código de autores) y T3 (identidad editorial con original pendiente).

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

La recuperación documental `references/literature_e1/recovery_20261105/` incorpora el capítulo completo de Miura, Pichl y Kaizoji (2019), con sus comparadores MLP/CNN independientes, la Tabla 1 transcrita y una síntesis de seis estudios. Es una revisión posterior al benchmark: no cambia las configuraciones firmadas ni los resultados. La CNN del artículo no declara dimensionalidad y las recetas del proyecto no se presentan como réplicas exactas. Los paquetes de descarga conservan en `catalogo.json` el commit de las fuentes usadas al construirlos.

La ampliación `references/literature_e1/journal_update_20261105/`, reflejada en `book/_static/literature_e1/journal_update_20261105/`, añade tres fuentes metodológicas financieras al catálogo de **39 referencias**, manteniendo aparte los **seis textos BTC**. Distingue un original editorial completo de Bucci (FNN pequeña), código de autor de HARNet (Conv1D causal) y la identidad editorial de DeepVol corroborada en un CV. Los manuscritos de HARNet y DeepVol no se presentan como leídos, ni se extraen de ellos parámetros o resultados publicados. Las fichas, las métricas verificadas de Bucci, el registro de acceso y las coincidencias secundarias de revistas documentan el alcance. No se redistribuyen aquí esos artículos completos de terceros ni se ejecutan nuevos entrenamientos por esta revisión.

## Piloto nuevo registrado del procedimiento vigente

`experiments/e1_filipe_pilot_20261105/protocol.json` se publicó en el commit `65fc9c49dd6ce3d4ddf13033b1bc54af0f8d0166` antes de iniciar sus entrenamientos. Usa los mismos índices forward de Filipe, representación relativa, recetas y embargo D, con cinco semillas nuevas 6201–6205 por modelo e historia. Sus 60 trayectorias contienen 720 segmentos mensuales, 1.440 ajustes y 21.960 predicciones de validación de 2024. El calendario ya era conocido; no se convierte en evaluación temporal independiente ni se reescribe el benchmark anterior.

El análisis registrado informa media, DE, CV, diferencias pareadas y dimensionamiento de 12 contrastes neuronales: δ=0,00023944847859645605, α=0,05/12 y potencia objetivo marginal de 0,80 bajo diferencias normales. El máximo común es **302 nuevas semillas con DE puntual o 1.680 con la DE superior marginal del 95 %**. El segundo es la recomendación conservadora registrada; ninguna de esas ejecuciones definitivas se inició. Cinco pares no acreditan normalidad y estos cálculos no son potencia alcanzada ni conjunta, ni prueban una mejora avanzada superior al 5 % frente a dos referencias.

Con evidencia privada completa restaurada, verificar los registros y resultados sin reentrenar:

```bash
.venv-training/bin/python scripts/run_filipe_pilot.py --phase validate
.venv-training/bin/python scripts/analyze_filipe_pilot.py --check
.venv-training/bin/python scripts/verify_filipe_pilot.py
```

Las corridas, pesos, escaladores y pronósticos individuales del nuevo piloto se conservan bajo `results/e1_filipe_pilot_20261105/runs/` y no se publican. La web ofrece agregados, 720 configuraciones, hashes y el informe independiente en `book/_static/e1_filipe_pilot_20261105/`. Un clon que solo tenga agregados no puede recalcular las métricas individuales sin esos insumos. No se deben borrar firmas o marcadores para forzar la ejecución sobre un registro cerrado.

Para repetir técnicamente los entrenamientos en una copia aislada, usar el commit de implementación **09d626d**, anterior al registro y sin el estado del piloto. Conservar la investigación original y crear otro checkout limpio de ese commit, instalar el lock de entrenamiento y ejecutar:

```bash
.venv-training/bin/python scripts/run_filipe_pilot.py --phase register
.venv-training/bin/python scripts/run_filipe_pilot.py --phase validate
.venv-training/bin/python scripts/run_filipe_pilot.py --phase run
.venv-training/bin/python scripts/analyze_filipe_pilot.py --phase analyze
.venv-training/bin/python scripts/analyze_filipe_pilot.py --check
```

La copia obtiene un registro con su fecha real; no reutiliza el timestamp histórico. Repetir las mismas semillas no crea nuevas réplicas estadísticas. Esta ruta consume los derivados diarios y manifiestos ya verificados de desarrollo, sin abrir ZIP brutos ni el test. Una propuesta avanzada o una evaluación temporal independiente requiere otro protocolo; septiembre–octubre de 2026 sigue sin asignarse automáticamente.

## Estudio D7 nuevo con orden prospectivo de las aleatorizaciones

El registro `experiments/e1_filipe_confirmatory_d7_20261105/protocol.json` fija tres recetas, los doce folds 2024 y tres comparaciones primarias antes de nuevas corridas. El protocolo fue publicado en 4c1d19e y su recibo en 268f348 antes del primer ajuste. La ampliación usa 20 semillas nuevas por modelo, sin elegir R mediante las cinco anteriores. El R común **79 por modelo** se fijó y publicó antes de las **237 trayectorias definitivas (5688 ajustes)**. Ninguna de las 25 semillas piloto se incorporó a esta fase.

Con evidencia local completa, estas verificaciones son de solo lectura respecto a los entrenamientos:

```bash
.venv-training/bin/python scripts/run_filipe_confirmatory_d7.py --phase validate
.venv-training/bin/python scripts/analyze_filipe_confirmatory_d7.py --phase check
.venv-training/bin/python scripts/verify_filipe_confirmatory_d7.py --phase variance_extension
.venv-training/bin/python scripts/verify_filipe_confirmatory_d7.py --phase definitive
```

Los informes del auditor quedan en `references/audits/revision_filipe_confirmatory_d7_20261105_*.json`. La colección pública está en `book/_static/e1_filipe_confirmatory_d7_20261105/`: incluye protocolo, recibos de publicación, bloqueo de R, catálogos, resúmenes, diagnósticos y hashes. No contiene los checkpoints, valores de escaladores, curvas completas ni predicciones por fecha. El clon público permite revisar los agregados; recalcularlos exige los insumos privados originales. El auditor también necesita un checkout Git con los commits publicados: el ZIP de código no incluye `.git`. Se debe clonar el repositorio y conservar su historia antes de restaurar la evidencia privada; crear un repositorio vacío no acredita la publicación previa.

El flujo original es `register` → publicación real de protocolo/código y recibo → `variance_extension` → análisis `lock` → auditoría de ampliación → publicación real del bloqueo y recibo → `definitive` → análisis `analyze/check` → auditoría definitiva → exportación. Los recibos registran commits y fechas reales; no se deben fabricar ni sustituir por marcas anteriores. No ejecutar `register/lock/analyze` para sobrescribir el estudio cerrado. Una repetición técnica en otra copia debe conservar su fecha y no representa nuevas realizaciones del mercado.

El R usa una cota unilateral 95 % marginal normal de DE y potencia objetivo 0,80 por comparación con α=0,05/3; no es potencia alcanzada ni conjunta. Los t pareados finales, Holm-3, IC del 95 % marginales e IC del 98,333 % individuales Bonferroni (cobertura conjunta al menos 95 %) se fijaron antes. La evidencia nueva estima diferencias por aleatorización del entrenamiento sobre 2024 conocido; no confirma la selección entre historias 7/14/21/28, no convierte el test antiguo en desconocido y no prueba la frontera avanzada de 5 %.
