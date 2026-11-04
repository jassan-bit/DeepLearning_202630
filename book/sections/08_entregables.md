# 8. Entregables: productos y evidencias

Este capítulo reúne los **siete productos del Entregable 1** y permite localizar el informe, el código, los datos, las tablas y las declaraciones utilizadas. Cada apartado desarrolla el producto con evidencia verificable. La disponibilidad de un archivo y el cierre académico de todos sus requisitos se identifican por separado: el respaldo aplicado de MLP/CNN1D y el alcance temporal de los registros conservan sus límites.

| Ítem | Producto desarrollado | Estado de la entrega |
|---|---|---|
| {ref}`8.1 <entrega-8-1>` | Informe en Jupyter Book | Publicado, con capítulos, figuras y archivos de respaldo |
| {ref}`8.2 <entrega-8-2>` | Código EDA, tres modelos y análisis estadístico | Versionado; restauración y verificación privada aislada comprobadas |
| {ref}`8.3 <entrega-8-3>` | Dataset o instrucciones y scripts exactos | Fuente, descargador, calendario y hashes documentados |
| {ref}`8.4 <entrega-8-4>` | Síntesis de literatura y flujo de búsqueda | Tabla y diagrama disponibles; respaldo aplicado del trío todavía parcial |
| {ref}`8.5 <entrega-8-5>` | Comparación, baseline e inferencia | Métricas y suplemento inferencial entregados, con alcance exploratorio |
| {ref}`8.6 <entrega-8-6>` | Dependencias, semillas y configuraciones | Locks, recetas y registros de las corridas identificados |
| {ref}`8.7 <entrega-8-7>` | Declaración de métrica, margen, independencia y partición | Documentada por versión y fecha; no se atribuye al pasado una decisión posterior |

El [manifiesto de los siete productos](../_static/e1_deliverables_20261104/delivery_manifest.json), también disponible [como tabla CSV](../_static/e1_deliverables_20261104/delivery_manifest.csv), identifica sus evidencias. El [inventario de archivos de esta entrega](../_static/e1_deliverables_20261104/public_manifest.json) conserva sus hashes. La [matriz de cierre de la auditoría](../_static/e1_audit_improvements_20261104/cierre_auditoria_e1_20261104.csv) conserva el trabajo que requiere evidencia adicional.

(entrega-8-1)=
## 8.1. Informe en Jupyter Book

El informe es este [Jupyter Book publicado](https://jassan-bit.github.io/DeepLearning_202630/intro.html). Su contenido desarrolla la formulación del problema, EDA, literatura, implementación, benchmark, ruta temporal, limitaciones y productos de entrega.

| Componente del informe | Ubicación |
|---|---|
| Problema, fuente y objetivo σ7 | [Capítulo 1](01_problema_dataset.md) |
| Calidad, distribuciones y estructura temporal | [Capítulo 2](02_eda.md) |
| Revisión y extracción de modelos | [Capítulo 3](03_estado_arte.md) |
| Diseño, implementación y evaluación | [Capítulo 4](04_implementacion_evaluacion.md) |
| Tablas, baseline e inferencia | [Capítulo 5](05_benchmark.md) |
| Procedimiento para regresión temporal | [Capítulo 6](06_procedimiento.md) |
| Limitaciones y criterios de evaluación | [Capítulo 7](07_limitaciones_entregables.md) |
| Decisiones y pendientes verificables | [Matriz de cierre](protocolo_pendiente.md) |

El [manual de construcción](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md) fija Jupyter Book v1, las dependencias y la comprobación de enlaces. Construir el HTML utiliza resultados guardados; no ejecuta otro entrenamiento o pronóstico.

```bash
jupyter-book build book --all --warningiserror --keep-going
python scripts/check_book_links.py
```

(entrega-8-2)=
## 8.2. Código completo, ejecutable y versionado

El [repositorio del proyecto](https://github.com/jassan-bit/DeepLearning_202630) conserva el código de adquisición, EDA, preparación supervisada, **MLP/LSTM/CNN1D**, validación, evaluación y análisis. Las versiones efectivamente utilizadas se identifican por los hashes del [registro de código previo al entrenamiento](../_static/filipe_multihistory_20261104/code_manifest_before_fitting.json) y de la [congelación anterior al test](../_static/filipe_multihistory_20261104/freeze.json).

| Etapa | Código principal | Producto verificable |
|---|---|---|
| Adquisición | [download_binance_spot.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/download_binance_spot.py) | Archivos oficiales y manifiesto SHA-256 |
| EDA y derivados diarios | [eda_binance_development.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/eda_binance_development.py) | Calidad, OHLCV diario y etiquetas históricas |
| Ventanas y transformaciones | [dl_data.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_data.py), [dl_data_v2.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_data_v2.py) | Entradas causales y escaladores de entrenamiento |
| Tres arquitecturas | [dl_models.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/dl_models.py) | MLP, LSTM unidireccional y CNN1D poco profunda |
| CV original D7 | [train_dl_r2_cv.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/train_dl_r2_cv.py) | Corridas D7; se conserva la variante primaria `relative_window` |
| Índices y ampliación de CV | [prepare_filipe_multihistory.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/prepare_filipe_multihistory.py), [train_filipe_multihistory.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/train_filipe_multihistory.py) | Folds y corridas D14/21/28; D7 reutilizado de su registro original |
| Cierre, ajuste final y test | [close_filipe_multihistory.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/close_filipe_multihistory.py), [final_fit_filipe_multihistory.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/final_fit_filipe_multihistory.py) | Decisiones cerradas, 120 modelos finales y evaluación de 608 fechas |
| Métricas y BDS por bloque | [analyze_filipe_multihistory.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/analyze_filipe_multihistory.py) | Tablas por entrada, modelo, semilla y mes |
| Complementos de la auditoría | [analyze_e1_audit_descriptives.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/analyze_e1_audit_descriptives.py), [analyze_e1_final_inference.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/analyze_e1_final_inference.py) | Tablas anuales, residuos, variabilidad e inferencia post-hoc |

La ejecución registrada conserva **120 trayectorias primarias de CV, 1.440 segmentos modelo–mes y 43.920 predicciones**. Tras seleccionar con desarrollo se ajustaron 120 modelos finales y se guardaron **72.960 predicciones de test**. Los comandos y la separación de entornos están desarrollados en [BUILD.md](https://github.com/jassan-bit/DeepLearning_202630/blob/main/BUILD.md).

**Ruta de entrega reproducible.** La evidencia completa se entrega privadamente con un manifiesto de hashes. El [restaurador](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/restore_e1_private_artifacts.py) copia y verifica esa evidencia en un directorio aislado, conservando los registros originales y su marcador de primer acceso al test. La prueba realizada restauró **113 insumos y seis TAR, con 16.890 miembros**, y ejecutó allí una auditoría independiente: [recibo de la réplica](../_static/e1_audit_improvements_20261104/reproducibility/replication_receipt.json) y [resumen de verificación](../_static/e1_audit_improvements_20261104/reproducibility/summary.json).

Esa comprobación recalcula métricas desde resultados guardados y verifica los pesos por sus hashes; no se presenta como repetición de todos los entrenamientos en otra máquina. Un clon público contiene resúmenes y código; para verificar la evidencia individual se necesita el paquete privado indicado en el manual.

(entrega-8-3)=
## 8.3. Dataset e instrucciones exactas de obtención

El dataset procede de **Binance Vision / Binance Spot, par BTCUSDT, velas de un minuto y fechas UTC**. El [manifiesto de adquisición](../_static/data/binance_spot_BTCUSDT_1m_20230101_20261030.json) identifica 75 archivos de 01/01/2023 a 30/10/2026, con URL, checksum, bytes, esquema y auditoría temporal. Registra **2.014.480 filas**, frente a 2.014.560 minutos esperados: hay 80 minutos ausentes y un timestamp de cierre inesperado en marzo de 2023.

Las [instrucciones del dataset](https://github.com/jassan-bit/DeepLearning_202630/blob/main/data/README.md) y el descargador permiten obtener o reanudar la misma adquisición sin claves de API:

```bash
python3 scripts/download_binance_spot.py \
  --start 2023-01-01 --end 2026-10-30 \
  --output /ruta/privada/binance-spot-data --workers 4
```

El script verifica cada ZIP contra el checksum oficial y conserva las discrepancias para revisión. La comprobación local de los archivos adquiridos usa `sha256sum -c SHA256SUMS_2023-01-01_2026-10-30.txt` desde el directorio de datos. La procedencia y condiciones consultadas se documentan en [términos de la fuente](../_static/data_source/terms_snapshot.json); disponibilidad pública no implica redistribución irrestricta.

| Función experimental | Periodo UTC | Uso efectivo |
|---|---|---|
| Historia inicial | 01/01/2023–31/12/2023 | Pasado inicial para ajustar |
| Validación forward | 01/01/2024–31/12/2024 | Doce bloques mensuales, historia creciente y embargo D |
| Ajuste final | Desarrollo permitido de 2023–2024 | Configuración y duración ya cerradas, purga D |
| Test | 01/01/2025–31/08/2026 | 608 fechas comunes, pesos finales fijos |
| Sin asignación al experimento | Septiembre–octubre de 2026 | Adquisición conservada; no incorporada automáticamente al test |

Para modelar se agregan las velas a **OHLCV diario UTC**. Cada entrada contiene D=7/14/21/28 días anteriores; el objetivo sigue siendo la desviación poblacional de **n=7 retornos logarítmicos diarios**, sin anualizar, y h=1. Se conserva el calendario y se excluyen los días incompletos y sus ventanas afectadas, sin imputar ni unir retornos a través de huecos. El [manifiesto de cohorte](../_static/filipe_multihistory_20261104/test/cohort_manifest.json) y la [auditoría de calidad del test](../_static/filipe_multihistory_20261104/test/raw_test_audit.json) registran la aplicación final de esas reglas.

La web publica metadatos y agregados. Los ZIP brutos, derivados diarios del test, predicciones individuales y pesos se conservan fuera de esta vista pública y forman parte de la ruta privada descrita en §8.2.

(entrega-8-4)=
## 8.4. Tabla de síntesis de la literatura y flujo de búsqueda

Se entrega la [tabla de síntesis de seis columnas](../_static/literature_e1/study_synthesis_six_columns_20261103.csv): referencia, problema, dataset, modelos DL, métricas y valores obtenidos por sus autores. La siguiente vista resume los **cinco textos completos incluidos**; el catálogo bibliográfico mantiene 35 referencias y la extracción reúne 81 cifras publicadas.

| Estudio del corpus | Dataset y objetivo descritos | Comparadores DL identificados |
|---|---|---|
| S1 · Akbiyik et al. (2023) | Bitcoin a 15 minutos y Twitter; volatilidad futura agregada de retornos intradiarios | LSTM, GRU y TCN; D-TCN es la propuesta avanzada |
| S2 · Zeri (2026) | BTCUSDT perpetuo/Crypto Lake; log-RV de 30 minutos | GRU, BiLSTM y encoder Transformer |
| S5 · Gao et al. (manuscrito 2023) | Yahoo BTC-USD diario; medida móvil de 30 días desplazada siete días | LSTM de una capa; la densa lineal es contexto |
| S6 · Wang et al. (sin año corroborado) | Bitcoin diario; RV sin definición suficiente para equipararla a σ7 | Transformer OHLCV independiente; EGARCH–Transformer es híbrido |
| S7 · Alakhouch Idriss (2026) | Bitcoin diario; medida futura de 30 días anualizada | LSTM 64→32 con cabeza Dense16→1; no MLP independiente |

Los valores numéricos y las ubicaciones de tablas se conservan en la síntesis descargable y en la [evaluación de calidad](../_static/literature_e1/quality_assessment.csv). Los objetivos, fuentes, periodos y escalas difieren de σ7: sus métricas no se ordenan como si fueran corridas del mismo benchmark.

![Flujo de búsqueda complementaria y recuperación de los cinco textos incluidos](../_static/literature_e1/flow.svg)

El flujo documenta **118 ocurrencias de repositorios, 12 duplicadas y 106 repositorios distintos; 20 inspeccionados; siete textos evaluados, cinco incluidos y dos excluidos**. Es una recuperación complementaria trazable, sin afirmar exhaustividad PRISMA ni búsqueda nativa acreditada en todas las bases.

La recuperación posterior a la auditoría permanece como otra etapa: [13 cadenas, 18 páginas y 92 repositorios únicos](../_static/e1_audit_improvements_20261104/literature/search_log.json), con [candidatos evaluados](../_static/e1_audit_improvements_20261104/literature/candidate_assessment.csv). No aportó manuscritos científicos completos nuevos; sus ocurrencias no se suman al flujo original sin deduplicación documental conjunta.

**Estado del requisito de extracción:** la tabla, el flujo y la trazabilidad están entregados. El [conteo por familia](../_static/literature_e1/family_frequencies.csv) acredita LSTM unidireccional en 3/5 estudios y la familia LSTM en 4/5; **MLP no lineal independiente y CNN1D simple siguen en 0/5**. Faltan fuentes aplicadas completas que sustenten esos dos integrantes del trío. El código candidato y las corridas propias no se cuentan como artículos incluidos ni cierran esa brecha.

(entrega-8-5)=
## 8.5. Tabla comparativa, baseline e inferencia estadística

Se entregan las [métricas de CV](../_static/filipe_multihistory_20261104/analysis_cv/metrics_summary.csv), las [métricas del test para cuatro entradas](../_static/filipe_multihistory_20261104/test/analysis/metrics_summary.csv) y el [desglose anual 2025/2026](../_static/e1_audit_improvements_20261104/descriptive/annual_metrics_summary.csv). Incluyen RMSE, MAE, MSE, MAPE, R² y MASE, cobertura, número de resultados definidos, media, DE e IC entre semillas.

Las tres familias seleccionaron D7 con desarrollo. Para las **608 fechas del test**, la comparación principal queda:

| Modelo D7 | RMSE, media ± DE | R², media ± DE | Papel en la entrega |
|---|---:|---:|---|
| MLP | 0,0047932 ± 0,0003988 | 0,764012 ± 0,042175 | Baseline neuronal seleccionado antes del test |
| LSTM | 0,0050298 ± 0,0002637 | 0,741108 ± 0,026619 | Segundo modelo base |
| CNN1D | 0,0044427 ± 0,0003568 | 0,797351 ± 0,033620 | Tercer modelo base; menor RMSE neuronal puntual en test |
| Persistencia | 0,0040856 | 0,829605 | Referencia determinista contextual |

Las filas neuronales resumen diez semillas; persistencia no tiene repeticiones de inicialización. MLP/D7 conserva la selección fijada con CV, aunque CNN1D/D7 tenga menor RMSE neuronal observada en test. Persistencia obtiene menor RMSE medio que las doce configuraciones neuronales; ese comparador se mantiene visible al plantear una mejora avanzada.

| Producto estadístico entregado | Evidencia |
|---|---|
| Diferencias pareadas, IC, tamaños de efecto y p-valores Holm | [Seis comparaciones principales](../_static/e1_audit_improvements_20261104/inference/paired_comparisons.csv) |
| Sensibilidad de localización por rangos | [Wilcoxon](../_static/e1_audit_improvements_20261104/inference/wilcoxon_sensitivity.csv) |
| Precisión predictiva temporal por h=1 | [Diebold–Mariano con HAC/HLN](../_static/e1_audit_improvements_20261104/inference/dm_tests.csv) |
| Autocorrelación residual | [Ljung–Box](../_static/e1_audit_improvements_20261104/inference/ljung_box.csv) |
| IC pareados de diferencias RMSE/MAE sobre bloques de fechas | [Bootstrap temporal](../_static/e1_audit_improvements_20261104/inference/date_bootstrap.csv) |
| Diferencias absolutas y relativas de error | [RMSE y MAE](../_static/e1_audit_improvements_20261104/inference/relative_differences.csv) |
| Diagnóstico BDS mensual por semilla y entrada | [CV: 2.880 contrastes](../_static/filipe_multihistory_20261104/analysis_cv/bds_by_fold.csv) y [test: 4.800](../_static/filipe_multihistory_20261104/test/analysis/bds_by_fold.csv) |

El complemento final es **post-hoc/exploratorio**: se registró después de inspeccionar el test y reutiliza sus pronósticos guardados. La incertidumbre por semillas, la pérdida temporal y el remuestreo de fechas responden preguntas diferentes; no son confirmaciones independientes. Se conservan las alertas de supuestos, los IC marginales y la multiplicidad. El promedio de p-valores BDS es descriptivo y no compara precisión entre modelos. La discusión detallada y las conclusiones están en [el capítulo 5](05_benchmark.md).

(entrega-8-6)=
## 8.6. Dependencias, semillas y configuración de cada corrida

Los entornos se entregan separados para mantener la compatibilidad de la librería CV y del entrenamiento:

| Entorno | Archivo versionado | Versiones identificadas |
|---|---|---|
| Construcción del libro | [requirements-book.lock.txt](../_static/e1_deliverables_20261104/dependencies/requirements-book.lock.txt) | Python 3.12; Jupyter Book 1.0.4.post1 |
| Generación de índices | [requirements-cv.txt](../_static/e1_deliverables_20261104/dependencies/requirements-cv.txt) | NumPy 1.26; `timeseries-cv` 0.1.5 |
| EDA | [requirements-eda.lock.txt](../_static/e1_deliverables_20261104/dependencies/requirements-eda.lock.txt) | Entorno fijado para preparar los derivados y figuras |
| Entrenamiento y análisis | [requirements-training.lock.txt](../_static/e1_deliverables_20261104/dependencies/requirements-training.lock.txt) | [Runtime registrado](../_static/filipe_multihistory_20261104/runtime.json): PyTorch 2.15.0+cpu, NumPy 2.5.4, pandas 3.1.0 y SciPy 1.18.1 |

Las **diez semillas base son 4101–4110**, comunes entre familias e historias. En CV se usa `semilla_fold = semilla_base + 100000 × ordinal_del_mes`, con enero=0 y diciembre=11; el refit independiente suma 10.000.000. El ajuste final usa `semilla_base + 10000000`. La partición es cronológica y determinista: no tiene semilla de sorteo. El código y las configuraciones registran también el generador del orden de minibatches y el determinismo estricto de PyTorch sobre CPU.

| Familia | Configuración compartida entre historias | Parámetros |
|---|---|---:|
| MLP | Una capa oculta de 16 unidades; LR 0,001 | 80D+33: 593/1.153/1.713/2.273 |
| LSTM | Una capa unidireccional de 32 unidades; LR 0,003 | 5.025 |
| CNN1D | Dos convoluciones causales de 16 canales, kernel 3 y promedio temporal; LR 0,001 | 1.057 |

Se fijaron AdamW, decaimiento de pesos 0,0001, dropout 0,1, batch 32, MSE, máximo de 100 épocas y clipping 1. La parada interna usa MSE de validación, paciencia 12 y mejora mínima 0,00001. Para el ajuste final se reutiliza la época seleccionada en diciembre de 2024 de la misma familia, D y semilla, sin seleccionar duración con test.

Se entregan dos catálogos de configuraciones: [1.440 segmentos modelo–mes de CV](../_static/e1_deliverables_20261104/cv_configuration_catalog.csv) y [120 ajustes finales](../_static/e1_deliverables_20261104/final_configuration_catalog.csv). Los segmentos de CV corresponden a 2.880 entrenamientos internos/refits. Cada fila identifica modelo, D, semilla, ancho, LR, épocas, parámetros y hash de la configuración completa; el catálogo CV añade mes y semillas internas de ajuste/refit. Son metadatos públicos y no contienen los pesos ni valores de los escaladores.

El [protocolo](../_static/filipe_multihistory_20261104/protocol.json), los [registros D7, incluidas sus variantes](../_static/dl_r2_cv_20261104/runs.csv), las [90 trayectorias de la ampliación D14/21/28](../_static/filipe_multihistory_20261104/runs.csv), el [registro de los 120 ajustes finales](../_static/filipe_multihistory_20261104/final_fit_summary.csv) y el [manifiesto de identidades finales](../_static/filipe_multihistory_20261104/final_models_manifest.json) completan la trazabilidad por `D–modelo–semilla`. Los escaladores, curvas, checkpoints y predicciones individuales de cada corrida permanecen en el paquete privado con sus hashes.

(entrega-8-7)=
## 8.7. Declaración de métrica, margen, unidad y partición

La [declaración y cronología versionada](../_static/e1_deliverables_20261104/declaration_and_chronology.json) distingue las decisiones anteriores al primer ajuste o al acceso experimental al test de las añadidas después. Los horarios siguientes corresponden al **4 de noviembre de 2026, Bogotá (UTC−5)**.

| Decisión o registro | Contenido declarado | Momento y alcance |
|---|---|---|
| Registro inicial de desarrollo | RMSE primaria; criterio operativo del 5 % frente a persistencia; semillas, cohorte y planificación del procedimiento inicial | 09:28:17, después del EDA y antes del primer ajuste DL; no anterior a examinar desarrollo |
| Protocolo de cuatro entradas | D=7/14/21/28, n=7, h=1, diez semillas, CV forward y reglas del test | 14:31:02; la ampliación usa desarrollo previamente expuesto |
| Congelación de selección y épocas | MLP/D7 como referencia neuronal, doce combinaciones a reportar, duración y reglas de calidad | 14:42:49, antes del primer acceso experimental de modelado al test, registrado a las 14:44:21 |
| Suplemento estadístico final | Pares principales, familias, incertidumbre temporal y diagnósticos | Posterior a inspeccionar test; explícitamente post-hoc/exploratorio |
| Criterio académico para modelos futuros | Mejora superior al 5 % frente a MLP/D7 **y** persistencia, con incertidumbre y multiplicidad | Definido después del test actual; no se atribuye retroactivamente al protocolo inicial |

**Declaración utilizada para interpretar los resultados:**

| Campo | Definición |
|---|---|
| Métrica primaria | RMSE de σ7, en su escala diaria original y sin anualizar; se calcula por semilla y periodo antes de resumir |
| Métricas secundarias | MAE, MSE, MAPE, R² y MASE; máscaras y cobertura explícitas |
| Margen operativo original | Reducción relativa del 5 % de RMSE frente a persistencia; no margen de equivalencia ni tolerancia financiera |
| Unidad de observación | Día UTC con entrada OHLCV de D días anteriores y etiqueta al cierre del día pronosticado |
| Unidad de aleatoriedad y emparejamiento | Trayectoria/ajuste por semilla, emparejada entre modelos sobre las mismas fechas; incertidumbre condicionada al calendario |
| Alcance de independencia | Una serie y un mercado BTCUSDT; fechas solapadas, meses y semillas no crean mercados independientes |
| Partición | Historia inicial 2023, validación forward 2024, ajuste final con desarrollo permitido y test enero de 2025–agosto de 2026 |
| Regla temporal | Sin barajar la partición; embargo igual al máximo rezago D según §6.4 de la guía; transformaciones ajustadas solo con el pasado admitido |

La [política académica futura](../_static/e1_audit_improvements_20261104/academic_margin_policy.json) define la reducción mediante la razón de medias de RMSE: $1-\mathbb E[m_A]/\mathbb E[m_{\rm ref}]>0,05$, para cada referencia. Se entregan [sensibilidades de 1/2,5/5/10 %](../_static/e1_audit_improvements_20261104/planning/advanced_margin_scenarios.csv) y [27 escenarios frente a la frontera práctica](../_static/e1_audit_improvements_20261104/planning/academic_joint_power_scenarios.csv). Son supuestos de planificación para una comparación nueva, sin afirmación de utilidad financiera o potencia realmente alcanzada.

El criterio y el número de repeticiones de una propuesta futura deben completarse en un registro anterior a su evaluación independiente. El [borrador de piloto](../_static/e1_audit_improvements_20261104/next_study_pilot_DRAFT.json) no ha sido ejecutado. La declaración anterior al acceso experimental al test está documentada; **no se declara un preregistro anterior al EDA, un piloto prospectivo del procedimiento vigente ni un margen avanzado previo al test ya examinado**.
