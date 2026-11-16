(reproducibilidad-de-la-entrega-vigente)=
(e2-reproducibilidad)=
# 8. Reproducibilidad, Ética y Comunicación Científica

**Lectura vigente:** la tabla principal de **16 sistemas sobre 608 fechas** tiene una reproducción registrada de sus **81 predictores en un checkout y un entorno Python nuevos**, con discrepancia máxima absoluta de predicción **0** en calibración y prueba. La guía admite reproducción por otra persona **o en un entorno limpio**: la segunda vía está ejecutada. Se conservan el método, las discrepancias y los límites de [§55](55_reproduccion_ablaciones.md), sin exigir ambos mecanismos ni atribuir una réplica a otro equipo que no está documentada. La [ampliación del 16/11 (§65)](65_cierre_cientifico_real.md) incorpora la auditoría sobre datos reales, controles entrenados, comparación de periodos originales y una IUT CPU completa. La [nueva repetición CPU de las recetas ganadoras](../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/result_v1/README.md) está completada: 78 ajustes neuronales nuevos, tres clásicos, calibración53, congelación81, admisión20ZIP, evaluación608 y análisis primario de 20.000 remuestreos, con sus propios resultados y cargos dentro del límite acumulado existente.

Este capítulo desarrolla los requisitos de **§8 y §8.1 de la guía, pp. 40–41**: entorno, instalación, semillas, evidencia reproducible, uso responsable y comunicación de resultados. La [lista verificable](../_static/e2_repro_prereg_current_v1/guide_checklist.csv), la [trazabilidad](../_static/e2_repro_prereg_current_v1/decision_traceability.csv), el [índice de evidencias](../_static/e2_repro_prereg_current_v1/current_evidence_index.json) y el [manifiesto](../_static/e2_repro_prereg_current_v1/manifest.json) distinguen el alcance acreditado de los puntos parciales. El {download}`índice de esta consolidación <../_static/e2_repro_prereg_current_v1/README.md>` enlaza los insumos; la [tabla A2 original](../_static/e2_benchmark_final_20261113/benchmark_A2.csv) y sus resultados científicos conservan su identidad.

(e2-entorno-reproduccion-minima-vigente)=
## 8.1 Entorno fijado y reproducibilidad mínima

### 8.1.1 Repositorio, instalación y versiones por etapa

El [repositorio del proyecto](https://github.com/jassan-bit/DeepLearning_202630) conserva código, configuraciones, registros y manifiestos. El motor científico ejecutado está fijado en [`1e8745497b484518f766b6da5c19be8362d9add7`](https://github.com/jassan-bit/DeepLearning_202630/tree/1e8745497b484518f766b6da5c19be8362d9add7); el cierre complementario conserva su procedencia en [§55](55_reproduccion_ablaciones.md). Los puntos de entrada actuales son este capítulo, [el protocolo ejecutado](48_protocolo_reducido.md) y [el Apéndice B](61_reproduccion_consolidada.md). Los comandos históricos E1/P0 del README y de otros capítulos conservan su alcance y no describen la ejecución del catálogo actual.

| Etapa | Versiones y archivo de instalación | Alcance comprobado |
|---|---|---|
| **Benchmark Bitcoin y reproducción limpia** | **Python 3.12.12, Torch 2.8.0+cu126, CUDA 12.6, NVIDIA A100-SXM4-80GB**; [requirements GPU](https://github.com/jassan-bit/DeepLearning_202630/blob/35a770a8a53e4a5d6692b7383e2c9a93649f0733/requirements-e2-training-gpu.lock.txt) | El [recibo del entorno limpio](../_static/e2_reproduction_ablations_20261113/historical_clean_runtime.json) conserva perfil de paquetes, hash del lock, instalación en venv nuevo sin paquetes del sistema y checkout del commit científico. |
| **Diagnósticos CPU de pronósticos guardados** | [requirements F2a](https://github.com/jassan-bit/DeepLearning_202630/blob/35a770a8a53e4a5d6692b7383e2c9a93649f0733/requirements-e2-f2a-audit.lock.txt): NumPy 2.3.5, pandas 2.2.3, SciPy 1.16.3, statsmodels 0.14.5, Matplotlib 3.10.8 | Es otro entorno. El [runtime del refinamiento temporal](../_static/e2_temporal_final_audit_20261114/runtime.json) informa Python 3.12.14 y análisis CPU, sin entrenamiento ni inferencia de modelos. |
| **Paper guía, ETTh1** | [requirements de la réplica](https://github.com/jassan-bit/DeepLearning_202630/blob/35a770a8a53e4a5d6692b7383e2c9a93649f0733/requirements-paper-guide.lock.txt); Python 3.12.14, Torch 2.15.0+cpu, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1 | [Runtime observado](../_static/paper_guide_evidence/runtime_versions.json), CPU. No es el entorno histórico de los autores ni el entrenamiento A100 de Bitcoin. |
| **Auditor de entrega** | Python 3.12 o posterior, biblioteca estándar; no necesita instalar NumPy ni Torch | Verifica integridad y reconstruye A2 desde predicciones ya guardadas. El informe registra el intérprete con el que se invoca. |
| **Construcción editorial** | [requirements del libro](https://github.com/jassan-bit/DeepLearning_202630/blob/35a770a8a53e4a5d6692b7383e2c9a93649f0733/requirements-book.lock.txt) y [BUILD.md](https://github.com/jassan-bit/DeepLearning_202630/blob/35a770a8a53e4a5d6692b7383e2c9a93649f0733/BUILD.md) | Renderiza la documentación; compilar el libro no reproduce los entrenamientos ni acredita sus supuestos. |

El archivo GPU fija versiones de las dependencias listadas, incluidas bibliotecas CUDA transitivas. El perfil observado del entorno limpio permite cotejarlas con lo instalado; un lock con otro alcance no se presenta como equivalente. Por ejemplo, [dated-core](https://github.com/jassan-bit/DeepLearning_202630/blob/35a770a8a53e4a5d6692b7383e2c9a93649f0733/requirements-e2-dated-core.lock.txt) declara únicamente dependencias directas, no un cierre transitivo completo. Los locks CPU y del paper guía también conservan sus listas propias.

Se recuperó además el **inventario completo de las 44 distribuciones Python observadas** mediante `importlib.metadata.distributions()`, conservado en el recibo original y cotejado por el productor en el entorno limpio. El {download}`lock observado reconstruido <../_static/e2_repro_prereg_current_v1/environment_observed.lock.txt>` reproduce exactamente el hash `6d02a90c8551905126417f6c92c5248b66d249c8b5930babb96c7f60e6f93716` registrado como `package_lock_sha256`. Los [metadatos de recuperación](../_static/e2_repro_prereg_current_v1/environment_observed.json) explican el orden de nombres, versiones y saltos de línea usado por el generador original. Es recuperación verificable de un inventario instalado, sin nueva instalación; no equivale a fijar el driver o la imagen del sistema operativo.

La plataforma objetivo del lock GPU es **Linux x86_64**. El recibo limpio no fija una versión de distribución/imagen Colab y del sistema operativo con digest reproducible; no se la sustituye por la versión del equipo usado para compilar o auditar el libro. Por tanto, la fijación del entorno a nivel de imagen/SO conserva ese límite, aunque la reproducción observada esté acreditada. **El contenedor es opcional en E2**; la guía lo exige para E3, como requisito de ese futuro entregable, sin convertir su ausencia en un incumplimiento obligatorio de E2.

Para preparar el entorno de paquetes GPU desde la raíz de un checkout versionado, sin iniciar ajustes:

```bash
python3.12 -m venv .venv-e2-gpu
.venv-e2-gpu/bin/python -m pip install -r requirements-e2-training-gpu.lock.txt
```

Esta instalación requiere la plataforma y compatibilidad CUDA indicadas; no ejecuta el benchmark. La reproducción de entrenamiento registrada ya está cerrada y se consulta en §55. Los pesos, escaladores, fuentes y recibos permanecen en **GitHub**, con ubicaciones y hashes en sus manifiestos; `/content` fue almacenamiento temporal. La [rama de resultados](https://github.com/jassan-bit/DeepLearning_202630/tree/results/e2-reduced-delivery-v1) y la [release original](https://github.com/jassan-bit/DeepLearning_202630/releases/tag/e2-reduced-results-20261112-v1) son puntos de entrada; las releases particionadas se resuelven por el manifiesto de cada artefacto.

### 8.1.2 Semillas, configuraciones, particiones y determinismo

| Elemento conservado | Evidencia y alcance |
|---|---|
| Semillas de ajuste | HPO **271201**; finales neuronales **271801–271806**, seis inicializaciones por cada una de trece redes. Los tres clásicos deterministas tienen una trayectoria cada uno. |
| Configuraciones | [Tuplas registradas](../_static/reduced_protocol_20261112/hpo_configurations_registered_bytes_v2.csv), recetas ganadoras y épocas derivadas de validación. SAE conserva AE1/AE2/supervisión; no se descartan etapas auxiliares al contar ajustes o costes. |
| Particiones | [Auditoría de soportes](../_static/e2_eda_current_v1/partition_audit.json): tres folds HPO, **253/435/527 train y 83 validación**; final **655 train, 53 calibración y 608 prueba**; embargo y puente causal explícitos. |
| Predictores y datos | [Congelación de 81 predictores](../_static/e2_benchmark_final_20261113/final_freeze.json), máscaras/firmas, [admisión de datos](../_static/e2_benchmark_final_20261113/data_admission.json) y [manifiesto de insumos públicos](../_static/e2_final_audit_inputs_20261114/manifest.json). |
| Tracking | JSON/JSONL, hashes SHA-256, recibos por intento, checkpoints y diarios aditivos. Se conservan fallos, reutilizaciones y estados terminales, sin atribuir el registro propio a MLflow o W&B. |

El núcleo establece semillas de **Python y Torch, incluida CUDA**, y generadores explícitos para el orden de los lotes. Los análisis NumPy usan generadores y semillas propios registrados: **no se afirma una semilla global NumPy que el núcleo no establece**. El código de [semillas](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/scripts/e2_staged_training.py) y la [política CUDA](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/scripts/e2_cuda_training.py) permiten comprobar la distinción.

La política ejecutada solicita algoritmos deterministas de Torch, **FP32; AMP y TF32 desactivados, cuDNN benchmark desactivado**, y `CUBLAS_WORKSPACE_CONFIG=:4096:8` antes de inicializar CUDA. Es una configuración explícita, no una garantía universal de igualdad entre dispositivos, drivers o versiones. La igualdad numérica comunicada corresponde a los recibos de las ejecuciones observadas. Las semillas controlan aleatoriedad computacional sobre una misma cronología; no constituyen mercados independientes ni resuelven potencia o fuga por sí solas.

### 8.1.3 Un comando para auditar y reconstruir la tabla principal

Desde la raíz del repositorio, el comando actual es:

```bash
python scripts/audit_e2_delivery.py --require-supplements --output work/e2_delivery_audit.json
```

Funciona **sin red, GPU ni paquetes externos** cuando el checkout contiene los insumos publicados. Si se desea aislar también este auditor en un venv nuevo:

```bash
python3.12 -m venv .venv-e2-audit
.venv-e2-audit/bin/python scripts/audit_e2_delivery.py --require-supplements --output work/e2_delivery_audit.json
```

El [código del auditor](https://github.com/jassan-bit/DeepLearning_202630/blob/35a770a8a53e4a5d6692b7383e2c9a93649f0733/scripts/audit_e2_delivery.py) comprueba los archivos enumerados en los manifiestos que admite, hashes, calendario, máscaras y semillas. **Reconstruye 81 RMSE individuales y 16 medias A2 a partir de pronósticos congelados**; el [informe publicado](../_static/e2_delivery_reproduction_20261114/delivery_audit.json) registra discrepancia máxima **2,6020852139652106×10⁻¹⁸**. También coteja informes y vínculos de 81 reproducciones/18 ablaciones, cierre contable y veinte contrastes refinados a 20.000 remuestreos.

El comando reconstruye aritmética y verifica integridad; **no vuelve a entrenar, ejecutar inferencia, repetir sorteos bootstrap o comparar tensores deserializados**. Tampoco se presenta como un único comando que regenere todas las tablas y figuras de todos los capítulos. Los análisis y visualizaciones tienen scripts e insumos identificados en sus páginas: [inferencia temporal](56_inferencia_estabilidad_final.md), [taxonomía](57_errores_techo.md), [galería de curvas](04_implementacion.md), [benchmark](06_benchmark_final.md) y [ruta temporal](07_ruta_temporal.md). La automatización integral de todos esos productos mantiene alcance parcial en B21 del Apéndice B.

La A2 conserva SHA-256 `879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1`. Su estimando es la **media de las RMSE individuales**, no la RMSE del promedio de predicciones. Las comprobaciones de manipulación de hashes, calendario, semillas, estimando y sorteos están documentadas en [§61](61_reproduccion_consolidada.md); esta actualización no vuelve a ejecutarlas.

### 8.1.4 Prueba de entorno limpio y discrepancias

La [reproducción registrada](55_reproduccion_ablaciones.md) repitió **78 ajustes neuronales y reconstruyó tres clásicos**, en un nuevo checkout del motor y venv sin `system-site-packages`. Reutilizar una caché de distribución de paquetes no significa reutilizar el entorno instalado. Los [81 recibos por predictor](../_static/e2_reproduction_ablations_20261113/reproduction.csv) y el [resumen por sistema](../_static/e2_reproduction_ablations_20261113/reproduction_by_model.csv) documentan **81/81 comprobaciones aprobadas**, diferencia máxima de pronóstico **0** en las 53 fechas de calibración y 608 de prueba; los productores también registraron igualdad de los 78 estados neuronales y parámetros exactos de los 3 clásicos.

La discrepancia cero de esta reproducción y el pequeño residuo aritmético del auditor CPU son comprobaciones distintas. La primera fue ejecutada por los workers A100; la segunda recalcula métricas de archivos ya guardados, sin repetir aquellos ajustes o comparaciones tensoriales. No se atribuye la ejecución a un segundo equipo ni a hardware distinto del documentado.

Las **18 ejecuciones de ablación** pertenecen a tres variantes SAE por seis semillas, con sus propias comparaciones; no amplían el número de predictores principales ni sustituyen la prueba de entorno limpio. El [diario final](../_static/e2_reproduction_ablations_20261113/budget_final.json) conserva el cierre de 84 horas autorizadas, **80,2561 h contabilizadas** y cero reservas abiertas. Son horas de campaña, no consumo energético ni GPU activa. La guía no exige 99 ajustes adicionales: ese fue el plan particular de reproducción y ablación ejecutado.

La reproducción numérica conserva recetas, datos y periodo heredado de E1. Acredita trazabilidad y repetición bajo esas condiciones; no demuestra ausencia de sobreajuste, potencia suficiente, cobertura universal o nueva generalización prospectiva. No se relanza aquí un controlador cerrado.

(e2-etica-uso-responsable-vigente)=
## 8.2 Ética, privacidad, sesgos y licencias

### 8.2.1 Datos, subgrupos y consecuencias del error

La fuente es **BTCUSDT de Binance Spot**, con velas de mercado agregadas y sin identificadores personales en los datos utilizados. El uso declarado es académico y no comercial; no se intenta identificar operadores. Esa ausencia de datos personales no elimina los sesgos de un único proveedor, activo, mercado y periodo, ni acredita representatividad de otros mercados.

El [EDA](02_eda.md), los [errores por trimestre](49_diagnosticos_f2a.md), la [taxonomía](57_errores_techo.md) y la [cobertura por grupos](60_calibracion_temporal_condicional.md) desagregan resultados por calendario, rango de pronóstico y régimen disponible al origen. Los grupos definidos con el objetivo ya realizado se señalan como descripciones retrospectivas. Las diferencias de soporte y cobertura no se ocultan: una cobertura global cercana al 90 % puede coexistir con infracobertura en pronósticos altos. **Estos grupos temporales no son una evaluación de equidad demográfica**; no hay atributos personales que permitan demostrarla, ni se afirma que sea innecesaria en otra aplicación.

Subestimar volatilidad podría infravalorar riesgo; sobreestimarla podría restringir una decisión de exposición. Son consecuencias potenciales, no pérdidas o beneficios económicos medidos por este benchmark. Se pronostica σ₇, no rentabilidad, decisión de inversión o VaR. No se han validado utilidad financiera, costes de transacción ni un sistema de negociación; las explicaciones por oclusión tampoco identifican causas económicas. Los resultados adversos, fallos e infracoberturas se conservan junto a los favorables.

### 8.2.2 Qué licencias están respaldadas

| Material | Evidencia y alcance |
|---|---|
| **Datos Binance Vision** | [§1](01_punto_partida.md) conserva **Vision Dataset Terms v1.0**, actualizados 26/08/2026 y revisados en E1, con atribución **CC BY-NC-SA 4.0 salvo designación específica y condiciones adicionales**. [Texto conservado](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/book/_static/data_source/TERMS_AND_CONDITIONS.md) y [procedencia](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/book/_static/data_source/terms_snapshot.json). No se atribuye MIT a los precios. |
| **Datos ETTh1 de la réplica** | La [licencia primaria de ETDataset](../_static/e2_repro_prereg_current_v1/LICENSE.ETDataset.txt), conservada exactamente desde `zhouhaoyi/ETDataset@1d16c8f4f943005d613b5bc962e9eeb06058cf07`, identifica **CC BY-ND 4.0**, con atribución y condición de no distribuir material adaptado. Esta prueba no se interpreta como autorización nueva para redistribuir los CSV transformados. |
| **Código de autores y bibliotecas** | Fuentes fijadas y adaptaciones identificadas en [§44](44_arquitecturas_publicadas.md). Se conserva, por ejemplo, [Apache 2.0 de TiDE](https://github.com/jassan-bit/DeepLearning_202630/blob/35a770a8a53e4a5d6692b7383e2c9a93649f0733/experiments/e2_tide_published_r0_20261112/LICENSE.TiDE). Esa licencia no se extiende automáticamente a todos los ports, dependencias o datos. La comprobación exhaustiva de permisos por componente mantiene alcance parcial. |
| **Pesos y preentrenamiento** | Los predictores E2 se ajustaron con sus propias semillas y datos admisibles; SAE preentrena sus encoders con train. No se declara uso de un checkpoint fundacional externo como si su licencia ya estuviera auditada. La procedencia del código y sus condiciones siguen siendo independientes de quién ajustó los pesos. |
| **Artículos, PDF y figuras de terceros** | DOI, versión, localizadores y fuentes se conservan en [la revisión](03_literatura.md) y [el paper guía](05_paper_guia.md). Acceso o lectura no equivalen a permiso uniforme de redistribución. Los esquemas propios se identifican como tales; no se atribuye una licencia abierta común a los originales. |

El [registro de licencias y alcances](../_static/e2_repro_prereg_current_v1/license_register.csv) y las [fuentes primarias con sus hashes](../_static/e2_repro_prereg_current_v1/license_sources.json) conservan estas comprobaciones por material; no extienden una licencia a los componentes que carecen de adjudicación completa.

**Un repositorio público no demuestra por sí solo permiso de reutilización o redistribución.** La evidencia anterior identifica condiciones verificadas y sus límites; no se declara una auditoría legal completa a partir de citas o hashes. Datos, código, pesos y materiales editoriales conservan condiciones diferenciadas.

(e2-comunicacion-cientifica-vigente)=
## 8.3 Comunicación científica verificable

Las afirmaciones se vinculan a datos, estimandos y pruebas concretos, con bibliografía BibTeX en [references.bib](https://github.com/jassan-bit/DeepLearning_202630/blob/35a770a8a53e4a5d6692b7383e2c9a93649f0733/book_e2/references.bib). Los registros distinguen DOI editorial, DOI de preprint, versión leída y repositorio de autores; un identificador `10.48550/arXiv...` no se presenta como DOI de revista ni como prueba de cuartil. La [revisión sistemática](03_literatura.md) conserva sus límites de acceso y comparabilidad, sin sustituir referencias verificables por afirmaciones generales.

Las tablas y figuras actuales explicitan unidades, agregación, número de ejecuciones y alcance de las bandas:

- La [galería de las trece redes](04_implementacion.md) muestra **78 curvas finales individuales** y **29 curvas HPO disponibles de 39**, con diez ausencias de caché identificadas. Distingue RMSE original, épocas reales y fases SAE; no es sólo TimeMixer ni traslada la MSE estandarizada ETTh1 a Bitcoin.
- El [benchmark](06_benchmark_final.md) identifica barras **±1 DE entre semillas**, IC95 % con su método y clásicos R=1 sin barras ficticias. No se confunden DE, IC del score e intervalos predictivos; los gráficos de barras de comparación mantienen la variación registrada.
- El panel de calibración del Top 3 de [§60](60_calibracion_temporal_condicional.md) utiliza los IC temporales de cobertura ya guardados. Las anchuras son puntos descriptivos sin IC disponibles: se identifican como tales, sin barras de incertidumbre inventadas ni una afirmación de superioridad entre políticas.
- Los Pareto y [rangos de la ruta temporal](07_ruta_temporal.md) explican coste, hardware, FLOPs core y exclusiones; las conexiones de no rechazo Wilcoxon–Holm no demuestran equivalencia ni una distancia crítica única de Nemenyi.

Se reportan p brutos y ajustados, efectos e intervalos según sus familias. El menor RMSE observado no se convierte en superioridad práctica confirmada: los veinte p Holm primarios siguen en 1. El test y δ heredados se aceptan como continuidad E1–E2–E3, con exposición previa declarada; δg sí estaba registrado y [§64](64_correcciones_metodologicas.md) ya ejecutó trece pruebas comparables train–calibración en CPU, con dos tolerancias condicionales y once no establecidas. [§65](65_cierre_cientifico_real.md) añade la comparación formal de los periodos A100 originales y la IUT CPU completa; no demuestra ausencia general de sobreajuste. Los refinamientos numéricos y análisis posteriores conservan sus identidades, sin atribuirles una fecha previa ni seleccionar una conclusión por el contraste más favorable.

(e2-estado-reproduccion-vigente)=
## 8.4 Apéndice B y alcance del cierre

El [Apéndice B vigente](61_reproduccion_consolidada.md) publica los 21 ítems literales, sus estados y fuentes, con descargas [CSV](../_static/e2_point1_review_v1/appendix_B_clarified.csv) y [JSON](../_static/e2_point1_review_v1/appendix_B_clarified.json). **La lista está publicada; eso no significa que todos los ítems estén completos.** No se crea un porcentaje global ni se infiere cumplimiento de una compilación correcta.

La reproducción en entorno limpio está acreditada dentro de sus condiciones. El auditor reconstruye A2 y verifica la entrega, mientras la regeneración integral de todos los productos conserva su alcance propio. Las auditorías reales y los contrastes de δg, periodos y RQ3 ya ejecutados tienen fuentes separadas en [§65](65_cierre_cientifico_real.md); la discrepancia cero no acredita por sí sola sus condiciones metodológicas. Permanecen la potencia limitada declarada, costes históricos incompletos y límites de incertidumbre, así como la fijación de imagen/SO y la revisión no exhaustiva de licencias por componente. El contenedor sigue siendo opcional para E2.

La nueva repetición del entrenamiento CPU ajustó **78 modelos neuronales nuevos, trece redes × seis semillas**, con objetivos reales de train655, recetas y épocas ya fijadas, y reconstruyó los tres clásicos. Calibró con 53 fechas, congeló 81 predictores, admitió los veinte ZIP originales y evaluó las 608 fechas antes de completar su análisis primario de 20.000 remuestreos. La [nueva A2 CPU](../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/result_v1/analysis/benchmark_A2.csv), el [análisis propio](../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/result_v1/analysis/benchmark_analysis_20000.json) y los [pronósticos unidos por fecha](../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/result_v1/predictions_81.npz) quedan separados de A2/RQ1 originales. El [plan anterior a los ajustes](../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/plan_v2/analysis_plan.json), la [admisión presupuestaria](../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/admission_v2/timing_admission.json) y la [recuperación del entorno](../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/runtime_recovery_v2/recovery_decision.json) conservan identidades y fallos. La secuencia está completada dentro de las **84 horas acumuladas ya autorizadas**, sin reiniciar cargos. El [inventario contable](../_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/result_v1/accounting_summary.json) suma **83 workers exitosos y 6.853,632183 s de cargo total del estudio**, incluidos los fallos previos: 5,254805 s de dependencias y 3,327804 s de ARIMA. El intento final reutilizó byte a byte 80 workers exitosos de V3 y ejecutó únicamente ARIMA, scoring y análisis; no volvió a entrenar las redes. La recuperación aplicó la compatibilidad del alias de recibo ARIMA `raw_value`→`raw`. No repite HPO, NAS, ablaciones ni toda la campaña histórica, no acredita paridad A100 y no sustituye la A2 original. **B21 conserva alcance parcial para la campaña completa.**

La [lista de requisitos de esta consolidación](../_static/e2_repro_prereg_current_v1/guide_checklist.csv) mantiene esos alcances junto con los del [preregistro](09_protocolo_registrado.md). Los comandos y recibos originales se preservan con sus fechas; [§65](65_cierre_cientifico_real.md) identifica los experimentos, las nuevas inferencias y los cálculos posteriores, sin atribuirlos a los archivos de ese corte anterior.

````{admonition} Archivo documental: contenido anterior del capítulo 8
:class: dropdown

Este bloque conserva íntegramente el texto anterior, sus encabezados y enlaces. Sus fechas y estados pertenecen a sus cortes originales; la lectura vigente de reproducibilidad, ética y comunicación se presenta arriba.

**Estado documental del 14/11/2026:** la tabla principal compara **16 sistemas sobre 608 fechas**. La reproducción registrada de sus **81 predictores** se completó en un checkout remoto y un entorno Python nuevos, con discrepancia máxima absoluta de predicción **0** en calibración y prueba. También se completaron **18 ablaciones**. Los recibos y sus límites se presentan en [§55](55_reproduccion_ablaciones.md). El Apéndice B vigente está en [§61](61_reproduccion_consolidada.md), con estados completos, parciales, no aplicables e incumplimientos identificados.

## 8.1 Punto de entrada de la entrega

Desde la raíz del repositorio, con Python 3.12 o posterior:

```bash
python scripts/audit_e2_delivery.py --require-supplements --output work/e2_delivery_audit.json
```

Este comando funciona **sin red, GPU ni paquetes externos**. Verifica los manifiestos y archivos publicados, reconstruye las 81 RMSE a partir de los pronósticos congelados y las 16 medias de A2, valida calendario/semillas, comprueba los informes de 81 reproducciones y 18 ablaciones y el refinamiento de los veinte contrastes a 20.000 remuestreos. El informe registra el entorno de esta ejecución y sus discrepancias.

El comando ejecuta una auditoría aritmética y de integridad. **No vuelve a entrenar ni a generar pronósticos, no repite los sorteos bootstrap y no deserializa pesos.** La repetición de los 81 ajustes y las comparaciones tensoriales ya fueron ejecutadas por los workers A100 y están acreditadas separadamente en §55. La auditoría no declara aprobados los requisitos metodológicos pendientes.

## 8.2 Entorno y fuentes científicas

| Etapa | Entorno o fuente identificada |
|---|---|
| Ajuste final y reproducción neuronal | Python **3.12.12**, Torch **2.8.0+cu126**, CUDA **12.6**, NVIDIA **A100-SXM4-80GB** |
| Política registrada | FP32, algoritmos deterministas; AMP y TF32 desactivados; semillas explícitas |
| Motor científico | [`1e8745497b484518f766b6da5c19be8362d9add7`](https://github.com/jassan-bit/DeepLearning_202630/tree/1e8745497b484518f766b6da5c19be8362d9add7) |
| Resultados complementarios cerrados | [`d604ede6bf0978e14a1b1d7bafaae6250fb8ed07`](https://github.com/jassan-bit/DeepLearning_202630/tree/d604ede6bf0978e14a1b1d7bafaae6250fb8ed07) |
| Reproducción registrada | Nuevo checkout y venv sin paquetes del sistema; [recibo del entorno](../_static/e2_reproduction_ablations_20261113/historical_clean_runtime.json) |
| Persistencia | **GitHub**, con inventarios y recibos por intento; sin Google Drive |

No se promete igualdad numérica entre hardware o versiones distintos. Los recibos acreditan la reproducción efectivamente observada, condicionada a esas versiones y a la misma cronología. El [manifiesto de resultados](../_static/e2_reproduction_ablations_20261113/manifest.json) y la [procedencia por ajuste](../_static/e2_reproduction_ablations_20261113/source_evidence.json) enlazan las tablas con los resultados productores.

## 8.3 Reconstrucción y análisis de pronósticos guardados

Los [insumos congelados públicos](../_static/e2_final_audit_inputs_20261114/manifest.json) permiten repetir cálculos sin descargar precios nuevos. [§56](56_inferencia_estabilidad_final.md) incluye el comando completo para recalcular los veinte contrastes con 20.000 réplicas y la fluctuación Giacomini–Rossi. [§57](57_errores_techo.md) incluye el comando de la taxonomía y las referencias de dificultad. Sus resultados son posteriores y no reemplazan los originales ni modifican recetas o calibradores.

La [tabla A2](../_static/e2_benchmark_final_20261113/benchmark_A2.csv) conserva su SHA256 `879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1`. El auditor de entrega comprueba ese digest antes de reconstruir sus métricas. Las métricas son medias de RMSE individuales, no la RMSE de promediar pronósticos.

## 8.4 Reproducción del entrenamiento y presupuesto

La ejecución original y los 81 reajustes de reproducción están cerrados. El límite autorizado fue **84 horas acumuladas**, conservando todos los cargos previos; el diario final contiene **351 eventos**, **80,2561 horas cargadas** y ninguna reserva abierta. Es contabilidad de tiempo de campaña, no GPU activa, energía o dinero. El [balance cerrado](../_static/e2_reproduction_ablations_20261113/budget_final.json) conserva sus hashes.

Las instrucciones antiguas P0/OOF y sus comandos `--stage all` se archivan al final de esta página; corresponden a otros modelos y otro periodo. No se presentan como un comando para reproducir el benchmark actual ni se propone relanzar el controlador cerrado. Un replay futuro del entrenamiento requeriría una salida independiente y autorización presupuestaria propia, conservando los resultados anteriores.

## 8.5 Límites de la reproducción

La guía acepta reproducción por otra persona **o en un entorno limpio**. Los 81 reajustes acreditan la segunda vía; no crean un test nuevo. El periodo 2025–2026 ya se conocía desde E1. La prueba formal respecto al δg heredado, la potencia objetivo y otras condiciones del Apéndice B requieren evidencia propia; la discrepancia numérica cero no las demuestra. La calibración y los análisis estadísticos requieren sus propios supuestos y evidencias.

El programa de auditoría se comprueba con pruebas de manipulación de hashes, calendarios, semillas, estimando y número de remuestreos:

```bash
python -m unittest tests.test_audit_e2_delivery -v
```


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/08_reproducibilidad.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).


````
