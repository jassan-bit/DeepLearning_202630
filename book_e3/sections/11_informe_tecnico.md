# 11 Bloque G Informe técnico en Jupyter Book

El informe debe permitir entender y reconstruir el trabajo con estructura científica: resumen, problema y contribuciones, antecedentes, método, diseño experimental, resultados, discusión, límites y conclusiones. El libro distingue evidencia ejecutada, resultados heredados y tareas pendientes. La CNN-LSTM con atención temporal por contenido y salida aditiva positiva tiene entrenamiento, validación y prueba histórica en diez semillas, además de la comparación estadística posterior. El código y las evidencias del prototipo Dash con Contexto, EDA y Modelos están publicados: catálogo de 17 sistemas y cuatro artefactos ONNX para inferencia CPU. La campaña confirmatoria y el despliegue público en Render siguen pendientes.

Las fuentes de E3 están en **`book_e3/`**. `book/` corresponde a E1 y `book_e2/` a E2; se conservan sus contenidos y direcciones. El Cuadro 7 de la guía, página 58, fija contenidos científicos mínimos; la numeración operativa 1–16 de este libro permite localizarlos sin confundir un capítulo de evaluación o entregables con resultados científicos nuevos.

## 11.1 Estructura mínima de capítulos

La concordancia siguiente conserva los doce apartados del Cuadro 7, numerados 0–11, y los relaciona con las secciones actuales. Un enlace a evidencia heredada mantiene su procedencia y no acredita una ejecución nueva del candidato.

| N.º y capítulo del Cuadro 7 | Contenido mínimo científico | Ubicación en la organización 1–16 | Estado y cierre pendiente |
|---|---|---|---|
| 0 Introducción | Resumen ejecutivo de máximo 250 palabras con cifras, problema, motivación, contribuciones enumeradas, objetivos, mapa y URL del dashboard. | [Portada](../intro.md), [1 Introducción](01_introduccion.md) y [3 Panorama](03_panorama_general.md). | Resumen con cifras y aportes técnicos enumerados disponible, junto al problema, objetivos y mapa; conserva los límites de R1/R2/R3. La URL Render ya está asignada; su funcionamiento público sigue por verificar en el capítulo 10. |
| 1 Trazabilidad de correcciones | Observaciones E2, acciones, evidencia, impacto y ranking antes/después cuando cambie. | [4 Correcciones E2](04_correcciones_e2.md). | Hallazgos internos, auditorías y ranking de seis frente a diez semillas documentados. La ampliación de semillas no se presenta como corrección de datos; no se inventan observaciones docentes. Cierre integral pendiente. |
| 2 Datos: EDA y auditoría | EDA específico temporal, auditoría de fuga, Data Card con fuente/licencia/sesgos e interpretación de figuras. | [1 Introducción](01_introduccion.md), [4 Correcciones](04_correcciones_e2.md), [dashboard](09_dashboard_predictivo.md) y DATA_STATEMENT.md. | EDA interactivo sobre 1316 observaciones con fechas de entradas t−1 y objetivo t; fuente/hash y controles descritos. Resolver controles parciales y ratificar los derechos de cada fuente. |
| 3 Modelos benchmark | Top 10 y baselines, definiciones, hiperparámetros, presupuesto, configuraciones y exportación. | [5 Configuraciones y exportación](05_configuraciones_exportacion.md). | Catálogo completo de E2 y la referencia E3; cuatro ONNX verificados en 608 fechas para la semilla representante 271801. Otros trece sistemas usan resultados precalculados; costes y exportación completa pendientes. |
| 4 Revisión de literatura y novedad | Búsqueda, declaración de novedad, fuente, relación con paper guía, métodos afines y verificación externa. | [6 Modelo original](06_modelo_original.md) y registros bibliográficos enlazados. | Revisión focal y estados de acceso documentados; R1 y revisión humana externa abiertos. |
| 5 Modelo original: diseño | Limitación, hipótesis, transferencia si procede, formulación, pseudocódigo, demostración/verificación, complejidad y bitácora. | [6 Modelo original](06_modelo_original.md) y [7 Diseño riguroso](07_diseno_riguroso.md). | Propiedades y verificaciones artificiales de los diseños históricos con alcance delimitado; pilotos completados. La receta aditiva evaluada se distingue de esos mecanismos. Novedad, conciliación global de Nd y congelamiento confirmatorio pendientes. |
| 6 Modelo original: implementación | Código comentado, pruebas, adaptación, ablación e interpretabilidad. | [6 Modelo original](06_modelo_original.md) y [7 Diseño](07_diseno_riguroso.md). | Implementación, pruebas de software y réplicas de mercado disponibles; ablaciones comparables completas, interpretabilidad y defensa académica pendientes. |
| 7 Protocolo y validación estadística | Estándar, preregistro, equidad, familia, potencia, supuestos, pruebas, ajustes, IC, efectos y adecuación. | [2 Estándar](02_estandar_profundidad.md), [7 Diseño](07_diseno_riguroso.md) y [8 Verificación](08_verificacion_estadistica.md). | Análisis posterior de los 17 sistemas calculado: 48 rechazos tras Holm, IC, efectos y MCS auxiliar. Falta el diseño confirmatorio previo, comparación justa, dimensionamiento de potencia y justificación completa de la batería F2a. |
| 8 Evaluación ampliada y robustez | Tres datasets, contraste entre conjuntos, perturbaciones, escala, distribución, calibración, incertidumbre, explicación, eficiencia y errores. | [8 Verificación estadística](08_verificacion_estadistica.md). | Sensibilidades temporales 7/14/28 y subgrupos de semillas disponibles en BTC. Los tres conjuntos adicionales, calibración del candidato y demás estudios de robustez siguen pendientes. |
| 9 Dashboard y despliegue | Arquitectura, exportaciones, capturas, diseño, ficha, despliegue, humo y URL. | [9 Dashboard](09_dashboard_predictivo.md) y [10 Render](10_despliegue_render.md). | Prototipo multipágina, contrato CSV, ONNX Runtime sin Torch en servicio, avisos por rangos de ajuste y sensibilidad local disponibles. El servicio Render ya fue creado con URL asignada; disponibilidad, humo y estudios completos de explicación/recursos siguen pendientes. |
| 10 Conclusiones y limitaciones | Hallazgos, veredicto R1/R2/R3, amenazas, riesgos, trabajo futuro y uso de IA. | [16 Conclusiones](16_conclusiones.md), [15 Integridad](15_integridad_academica.md) y [anexo final](cierre_reproducibilidad.md). | Conclusiones empíricas retrospectivas y límites disponibles; R1, defensa de R2 y R3 siguen sin acreditarse. La sección 14 de rúbrica no las sustituye. |
| 11 Reproducibilidad y bibliografía | Contenedor, semillas, tabla maestra, lista de verificación y referencias BibTeX. | Esta sección 11 y [anexo final de reproducibilidad y bibliografía](cierre_reproducibilidad.md). | Fuentes, auditorías y bibliografía conectadas al libro; contenedor científico y reproducción independiente de la campaña E3 pendientes. |

Las secciones 2–3 explican el estándar y sus dependencias; la [12 inventaría entregables](12_lista_entregables.md); la [13 organiza la sustentación](13_sustentacion.md); la [14 conserva la evaluación y los topes de la guía](14_evaluacion_calificacion.md). Esas secciones complementan el reporte científico. El [anexo final](cierre_reproducibilidad.md) reúne conclusiones provisionales, checklist, autoría/IA y bibliografía; deberá actualizarse con evidencia definitiva sin convertir pendientes en resultados.

## 11.2 Requisitos técnicos del libro

**Configuración, índice y referencias.** Existen `_config.yml` y `_toc.yml` propios de E3. La organización incorpora `book_e3/references.bib` con los siete registros ya presentes en la bibliografía del candidato, junto con su conexión BibTeX y citas en el cierre. No se alteran los registros originales de `_static/` ni se atribuye lectura completa a cada referencia por el hecho de incluirla. Los dos originales leídos focalmente y los antecedentes heredados o de sólo metadatos mantienen sus estados en la matriz de fuentes. La búsqueda completa de novedad sigue pendiente.

`referencias_metodologicas_v1.bib` incorpora por separado las 37 referencias sugeridas por la guía. Su [índice de aplicabilidad](referencias_sugeridas.md) conserva los números originales y distingue metodología temporal, clasificación y otras tareas. Son metadatos aportados, no 37 nuevas lecturas ni inclusiones en la revisión de novedad. El archivo original de siete referencias y su descarga publicada permanecen intactos.

Las citas y referencias cruzadas deben comprobarse mediante una construcción estricta. El fichero bibliográfico descargable que ya existía como recurso estático se distingue de la bibliografía enlazada por la configuración. Añadir citas funcionales mejora la trazabilidad; no demuestra R1 ni sustituye la evaluación de trabajos cercanos.

**Ejecución y productos precalculados.** La configuración actual conserva `execute_notebooks: off`: construir HTML no ejecuta los experimentos, entrena modelos o regenera automáticamente cada tabla y figura. El informe presenta evidencia guardada y procedimientos documentados. Para cerrar la reproducibilidad científica deberán aportarse celdas o ejecutores comentados que generen cada producto, con entradas, versiones y comandos verificables. Los cálculos costosos podrán conservarse precalculados y versionados si se incluye el código que los produjo y su linaje.

Los comandos `--check` de los auditores actuales comprueban integridad, fuentes y reconstrucción de determinados informes guardados. No reproducen por sí solos una campaña de entrenamiento, toda la A2 con entradas reales o la evaluación ampliada del candidato. Las pruebas sintéticas y los resultados de E2 conservan el alcance concreto de sus recibos.

El cálculo estadístico posterior ya dispone de un ejecutor y exportador en [experiments/e3_estadistica_posterior_20261117](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_estadistica_posterior_20261117), con once insumos inmutables y verificación SHA256. Regenera las métricas, contrastes y figuras desde las predicciones guardadas, sin volver a entrenar ni inferir. Sus **42 pruebas de métodos e insumos** y la verificación de CI se documentan en la [sección 8](08_verificacion_estadistica.md); no equivalen a la reconstrucción humana de toda la campaña.

El bundle `data/e3_dashboard_v1/` deriva el catálogo de la misma tabla ampliada del libro y conserva sus métricas. Las curvas del dashboard muestran la semilla representante 271801, fijada por su identificador, mientras la tabla resume diez inicializaciones neuronales. El historial de 40 épocas de esa semilla E3 y su mejor época 38 se recuperan del checkpoint existente; MSE normalizada durante las actualizaciones y RMSE de validación en σ₇ se muestran con unidades separadas, sin presentarlas como una prueba de brecha train–validación.

**Figuras, tablas y ecuaciones.** Cada figura debe tener título, ejes/unidades, leyenda e interpretación; cada tabla, descripción y referencia en el texto. Las barras de métricas requerirán incertidumbre válida y rotulada, sin inventar barras de error cuando falte su estimación. Los métodos y muestras que soportan esa incertidumbre deben quedar explícitos.

Las ecuaciones usan MyST y símbolos definidos. La configuración activa `numfig`, `math_numfig` y `math_number_all`; las ecuaciones principales tienen etiquetas únicas y referencias funcionales. La revisión técnica de las secciones 1–8 y la formulación preliminar conservó sus fórmulas y comprobó la numeración y las referencias en HTML mediante una construcción estricta. La ecuación de discrepancia de este capítulo sigue el mismo formato y se incorpora a la verificación final del libro. Esa comprobación se registra por versión. La demostración latente disponible no se amplía a OHLCV crudo por añadir numeración o una figura.

**Construcción y publicación.** Desde la raíz del repositorio, los comandos de verificación del libro E3 son:

```bash
python -m pip install -r requirements-book.lock.txt
jupyter-book build book_e3 --all --warningiserror --keep-going
python scripts/check_book_links.py --book-dir book_e3
```

La ruta genérica `book/` de la guía se adapta aquí a `book_e3/` para construir E3 y preservar E1. Deben comprobarse páginas, navegación, descargas, ecuaciones, citas, figuras y enlaces. La publicación en GitHub Pages o una carpeta HTML comprimida debe conservar commit y procedencia; un HTML que compile no acredita ejecución científica ni el servicio Render.

Como referencia técnica comprobada, el [commit `7c17957871f9390e73889c2bcb19e730870fd9bc`](https://github.com/jassan-bit/DeepLearning_202630/commit/7c17957871f9390e73889c2bcb19e730870fd9bc) pasó la [compilación y publicación de los tres libros](https://github.com/jassan-bit/DeepLearning_202630/actions/runs/44213071892), incluida la verificación de sus páginas servidas. E3 conservó 39 páginas HTML y las 118 descargas anteriores; el comprobador local recorrió 3940 enlaces. Ese recibo corresponde a esa versión y las ampliaciones posteriores deben repetir la verificación. `books_manifest.json` vincula los productos publicados con su commit fuente; no es un DOI ni el recibo de reproducción científica por un tercero.

**Entornos y reconstrucción independiente.** El [perfil de servicio de la sección 10](10_despliegue_render.md) dispone de dependencias fijadas y ejecuta cuatro ONNX con CPUExecutionProvider sin importar Torch. La imagen Docker se construyó y su servicio local respondió en las ocho rutas verificadas, con usuario sin privilegios, sistema de archivos de sólo lectura y límite de 512 MiB. Esa comprobación local no acredita el servicio Render. La reproducción científica requiere otro perfil, con framework, versiones, datos y ejecutores adecuados; exportar o repetir una auditoría no equivale a reproducir todo el proyecto.

Una persona ajena al código deberá ejecutar la reconstrucción desde cero en el contenedor científico, obtener la tabla principal y compararla con la versión entregada. Se conservarán identidad/rol del revisor, commit, digest de imagen, datos/versiones, hardware, comandos, resultados, tolerancias y discrepancia máxima observada. El informe distinguirá diferencias por precisión, hardware y no determinismo. Esa reproducción todavía no se ha realizado para la campaña E3; una revisión de otro agente de IA no se registra como la persona independiente.

La disponibilidad de datos autorizados y artefactos es un prerrequisito. La [auditoría vigente](04_correcciones_e2.md) verifica 1316 ventanas, incluidas las 608 de prueba, y conserva **12 controles PASS, 3 PARTIAL y 1 NA**. Esto corrige el antiguo límite de disponibilidad de esas entradas; no cierra por sí solo el Bloque A, los derechos de cada fuente o la reproducción integral. Las diez semillas de E3 sí tienen resultados de mercado guardados; los tres conjuntos adicionales siguen sin elegirse. Un manifiesto o un resultado precalculado no sustituye los demás insumos y procedimientos necesarios para reconstruir la entrega.

**Checklist del último capítulo.** El [anexo final](cierre_reproducibilidad.md) debe conservar la lista de verificación completa y su evidencia, con los estados siguientes actualizados según avance:

| Elemento de reproducibilidad | Evidencia que debe acompañarlo | Estado actual y límite |
|---|---|---|
| Software y dependencias | Versiones exactas, Python, librerías, locks y comandos. | Runtime de servicio sin framework de entrenamiento y entorno de desarrollo separados, con versiones fijadas; perfil científico completo y reproducción integral pendientes. |
| Hardware e imagen | CPU/GPU, memoria, drivers, precisión y digest de contenedor. | Imagen local del servicio construida y comprobada con cuatro ONNX CPU, además de perfiles científicos CUDA/CPU previos. La imagen científica completa aún no fue reconstruida por una persona externa. |
| Semillas y determinismo | Listas por fase, generadores, flags y límites entre plataformas. | E2 y E3 registran 271801–271810 para sus redes; tres clásicos conservan una trayectoria cada uno. La semilla histórica E3 20261116 queda fuera de la media actual. Diez semillas no acreditan potencia suficiente ni paridad entre plataformas. |
| Hiperparámetros y esfuerzo | Rangos, herramienta, intentos, fallos, selección, Nd y presupuesto común. | Recetas E2 y aditiva E3 congeladas para las réplicas; pilotos y consumos conservados. Faltan HPO/política de ajuste comparables, conciliación global de Nd y diseño confirmatorio. |
| Ejecuciones y dispersión | Número de repeticiones, unidades, media/DE, IC y separación temporal/semilla. | E3: RMSE media 0,0029394324, DE muestral 0,0000471254 y MAE media 0,0018928273 en diez semillas y 608 fechas; incertidumbre temporal y de inicialización delimitadas en la sección 8. |
| Datos y código | Fuentes, licencias, versiones, hashes, soporte de ventanas y acceso reproducible. | Bundle local de 1316 observaciones y fuentes verificadas, DATA_STATEMENT.md y ejemplos con siete días pasados; derechos finales y tres conjuntos adicionales pendientes. |
| Pesos y estados completos | Configuración, pesos, escaladores/calibradores y estado del optimizador cuando corresponda. | Tres TorchScript históricos y cuatro ONNX actuales —incluido E3— con metadatos y paridad de semilla 271801 en las 608 fechas; snapshots científicos separados. Faltan política final común y exportaciones de otros trece sistemas. |
| Tabla maestra y productos | Scripts/celdas por tabla y figura, versión, estimando y tolerancia. | El catálogo de 17 sistemas deriva de las métricas del libro y tiene prueba de igualdad; faltan todos los campos comparables de recursos y la calibración E3. Reconstrucción científica integral pendiente. |
| Tiempo, recursos y energía | Costes por fase/modelo, hardware, memoria y energía medida o estimada con método. | Registros cerrados separados: ampliación E2 5130,9707 s y E3 acumulado 948,4396 s, incluidos consumos previos. La etapa estadística no entrena ni infiere. Faltan costes comparables completos y energía con método explícito. |
| Reproducción independiente | Recibo del tercero, contenedor desde cero y máxima discrepancia contra la tabla entregada. | Pendiente de ejecutar y documentar. |
| Archivo persistente y DOI | Versión final archivada y enlace persistente a código, datos permitidos y evidencias. | Pendiente de la entrega final; no se inventa un DOI. |
| Autoría, contribuciones e IA | Aportes individuales y herramientas usadas, revisados por el autor. | Asistencia declarada; validación, firmas y defensa académica pendientes del autor, sin firmas simuladas. |
| Bibliografía y referencias cruzadas | Registros BibTeX, citas, lista bibliográfica y enlaces verificados por build. | Siete antecedentes del candidato y 37 sugerencias metodológicas en archivos separados. El corte técnico de base pasó la construcción estricta; las ampliaciones requieren repetirla, conservando los estados de lectura y la procedencia. |

El cierre mantendrá conclusiones acordes con lo realmente observado. R1, comprensión/defensa de R2 y R3 no se acreditan mediante esta reorganización, la bibliografía o una compilación. Los resultados desfavorables, ausencias y límites de reproducción deben permanecer junto a las afirmaciones que condicionan.

## 11.3 De cada producto a su generador

Una tabla o una figura se considera reconstruible cuando se identifican sus entradas exactas, el generador, el entorno y la comprobación de la salida. Se distinguen tres operaciones: reconstruir agregados a partir de predicciones guardadas, ejecutar los modelos exportados y volver a entrenar una receta. Los ejecutores enumerados aquí permiten verificar las dos primeras y recuperar un historial guardado; sus recibos no acreditan la reconstrucción de toda la campaña de entrenamiento.

| Producto del informe o del dashboard | Generador y entorno | Entradas y comprobación disponible |
|---|---|---|
| Métricas, contrastes, efectos, MCS y figuras del análisis posterior. | `experiments/e3_estadistica_posterior_20261117/run.py`, con las versiones de su `requirements.txt`; `plots.py` genera `ranking_rmse.svg` y `gains_ci.svg`. | Once archivos fijados por commit/SHA256, plan y código registrados. Regenera resultados desde las trayectorias guardadas; conserva la diferencia entre ámbitos temporal y de semillas. |
| Tabla descriptiva ampliada, IC entre semillas y métricas por predictor. | `experiments/e3_revision_bloque_d_20261117/metrics_supplement.py`, con NumPy 2.3.5. | Las 143 trayectorias guardadas y sus cuantiles E2. `--check --mirror` compara los CSV publicados y rechaza cambios de alineación o hash. E3 sigue sin cuantiles predictivos. |
| Catálogo, observaciones y curvas de la aplicación. | `experiments/e3_dashboard_20261117/build_data.py`, en el entorno del dashboard. | Cohorte auditada y tablas del libro admitidas por hash. `--download --check` reconstruye el bundle y exige igualdad de bytes con sus archivos versionados. |
| Predicciones ONNX de la semilla representante 271801. | `experiments/e3_dashboard_20261117/verify_serving.py`, con el lock de inferencia. | Compara los cuatro modelos sobre las mismas 608 ventanas contra predicciones congeladas, sin entrenamiento; declara la tolerancia absoluta 10⁻⁶ en σ₇ y escribe un recibo nuevo. |
| Historial de 40 épocas del candidato y mejor época 38. | `experiments/e3_dashboard_20261117/extract_training_history.py`, en un entorno científico separado con Torch. | Verifica el checkpoint fuente antes de extraer su historial. Recupera valores ya registrados, con escalas MSE de entrenamiento y RMSE de validación distintas; no vuelve a entrenar. |
| Gráficos interactivos y capturas de la aplicación. | `e3_dashboard/plots.py`, callbacks de `pages/` y `tests/test_e3_dashboard_browser.py`. | Los gráficos usan los bundles versionados. Las capturas requieren ejecutar y recorrer la aplicación real; el recibo registra dimensiones, hashes, errores y seis casos de navegador. |

Los comandos siguientes se ejecutan desde la raíz. El cálculo estadístico utiliza su entorno de análisis; la aplicación utiliza el entorno de inferencia. Las salidas de una reproducción se escriben en `work/` para conservar los resultados entregados.

```bash
# Entorno de análisis estadístico fijado en el directorio del ejecutor.
python experiments/e3_estadistica_posterior_20261117/run.py \
  --data work/e3_statistical_inputs --download \
  --output work/e3_reproduccion/estadistica

python experiments/e3_revision_bloque_d_20261117/metrics_supplement.py \
  --data-dir work/e3_statistical_inputs --download --check \
  --mirror book_e3/_static/e3_revision_bloque_d_20261117
```

```bash
# Entorno de inferencia instalado con requirements.txt.
python experiments/e3_dashboard_20261117/build_data.py --download --check

python experiments/e3_dashboard_20261117/verify_serving.py \
  --npz work/e3_dashboard_inputs/current_cohort_arrays.npz --download \
  --output work/e3_reproduccion/serving.json
```

La [ejecución de CI del dashboard correspondiente al corte de base](https://github.com/jassan-bit/DeepLearning_202630/actions/runs/44213072051) aprobó 25 casos unitarios, la reconstrucción del bundle, la comparación ONNX y seis recorridos de navegador. Su job científico aprobó 60 casos y omitió una comparación opcional que requiere SciPy; esa comparación sí formó parte de los 61 casos aprobados en el entorno científico local. Los recibos separan esas coberturas. Ninguno sustituye la evaluación de una URL Render ni el acta de una persona independiente.

## 11.4 Acta de reproducción independiente

El tercero debe trabajar sobre una versión inmutable y un contenedor científico identificado por digest. Antes de ejecutar se fijan las filas que debe producir, las tolerancias por campo y el criterio de aceptación; no se relajan tras observar discrepancias. El acta debe permitir reconstruir estas decisiones:

1. Identificar a la persona y su relación con el proyecto, versión fuente, imagen, hardware, versiones y comandos reales. Confirmar el acceso autorizado a todos los insumos.
2. Registrar si reconstruye agregados, exportaciones o toda la campaña de entrenamiento, y ejecutar desde cero el alcance declarado. Una operación parcial conserva ese nombre en el resultado.
3. Comparar primero los identificadores de dataset, partición, modelo, semilla y métrica, el número de filas y los valores ausentes. Un conjunto de filas diferente no se corrige comparando sólo su intersección.
4. Para cada campo numérico comparable, publicar la discrepancia máxima, las tolerancias absoluta/relativa y los pares que las exceden. Conservar stdout, errores, tiempos y productos; documentar también una reproducción fallida.
5. Adjuntar el concepto firmado por la persona y discutir las diferencias por precisión, hardware, datos, no determinismo o procedimiento. La verificación automatizada ayuda a preparar el acta, pero no firma por el tercero.

Para un campo métrico $k$, sea $\mathcal I_k$ el conjunto completo de filas emparejadas declarado antes de la ejecución. La discrepancia absoluta máxima es

```{math}
:label: eq-e3-reproduccion-discrepancia
D_{\infty,k} = \max_{i\in\mathcal I_k}
\left|z^{\mathrm{tercero}}_{i,k}-z^{\mathrm{referencia}}_{i,k}\right|.
```

La ecuación {eq}`eq-e3-reproduccion-discrepancia` se evalúa **por campo y unidad**, sin mezclar en un máximo RMSE, segundos o memoria. Cada par debe cumplir la tolerancia declarada, absoluta más relativa al valor de referencia. Los campos ausentes se mantienen ausentes y no se reemplazan por cero. La tabla del acta contendrá campo/unidad, número de pares, máxima discrepancia absoluta, tolerancias, cantidad de incumplimientos y dictamen. No se rellena aquí con ceros: la reproducción humana de la campaña E3 sigue pendiente y todavía no existe una discrepancia máxima observada de ese alcance.
