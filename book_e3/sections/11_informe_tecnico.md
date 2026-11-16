# 11 Bloque G Informe técnico en Jupyter Book

El informe debe permitir entender y reconstruir el trabajo con estructura científica: resumen, problema y contribuciones, antecedentes, método, diseño experimental, resultados, discusión, límites y conclusiones. El libro actual organiza las exigencias de la guía y distingue evidencia ejecutada, resultados heredados y tareas pendientes. La campaña científica del candidato, el dashboard y su despliegue todavía no están completados.

Las fuentes de E3 están en **`book_e3/`**. `book/` corresponde a E1 y `book_e2/` a E2; se conservan sus contenidos y direcciones. El Cuadro 7 de la guía, página 58, fija contenidos científicos mínimos; la numeración operativa 1–16 de este libro permite localizarlos sin confundir un capítulo de evaluación o entregables con resultados científicos nuevos.

## 11.1 Estructura mínima de capítulos

La concordancia siguiente conserva los doce apartados del Cuadro 7, numerados 0–11, y los relaciona con las secciones actuales. Un enlace a evidencia heredada mantiene su procedencia y no acredita una ejecución nueva del candidato.

| N.º y capítulo del Cuadro 7 | Contenido mínimo científico | Ubicación en la organización 1–16 | Estado y cierre pendiente |
|---|---|---|---|
| 0 Introducción | Resumen ejecutivo de máximo 250 palabras con cifras, problema, motivación, contribuciones enumeradas, objetivos, mapa y URL del dashboard. | [Portada](../intro.md), [1 Introducción](01_introduccion.md) y [3 Panorama](03_panorama_general.md). | Problema y estados disponibles; resumen final de resultados del candidato y URL Render pendientes. |
| 1 Trazabilidad de correcciones | Observaciones E2, acciones, evidencia, impacto y ranking antes/después cuando cambie. | [4 Correcciones E2](04_correcciones_e2.md). | Hallazgos internos y auditorías documentados; no se inventan observaciones docentes ni cambios de ranking. Cierre integral pendiente. |
| 2 Datos: EDA y auditoría | EDA específico temporal, auditoría de fuga, Data Card con fuente/licencia/sesgos e interpretación de figuras. | [1 Introducción](01_introduccion.md), [4 Correcciones](04_correcciones_e2.md) y fuentes E2 enlazadas. | Auditoría y contratos con alcances explícitos; consolidar el EDA/Data Card científico de E3 y resolver los controles pendientes. |
| 3 Modelos benchmark | Top 10 y baselines, definiciones, hiperparámetros, presupuesto, configuraciones y exportación. | [5 Configuraciones y exportación](05_configuraciones_exportacion.md). | Catálogo de 16 sistemas, selección y tres exportaciones disponibles; completar artefactos, costes y verificación del perfil final. |
| 4 Revisión de literatura y novedad | Búsqueda, declaración de novedad, fuente, relación con paper guía, métodos afines y verificación externa. | [6 Modelo original](06_modelo_original.md) y registros bibliográficos enlazados. | Revisión focal y estados de acceso documentados; R1 y revisión humana externa abiertos. |
| 5 Modelo original: diseño | Limitación, hipótesis, transferencia si procede, formulación, pseudocódigo, demostración/verificación, complejidad y bitácora. | [6 Modelo original](06_modelo_original.md) y [7 Diseño riguroso](07_diseno_riguroso.md). | Propiedad de la cabeza latente y verificación artificial disponibles; validación del mecanismo con mercado y congelamiento pendientes. |
| 6 Modelo original: implementación | Código comentado, pruebas, adaptación, ablación e interpretabilidad. | [6 Modelo original](06_modelo_original.md) y [7 Diseño](07_diseno_riguroso.md). | Implementación y pruebas de software disponibles; ablaciones experimentales e interpretabilidad real pendientes. |
| 7 Protocolo y validación estadística | Estándar, preregistro, equidad, familia, potencia, supuestos, pruebas, ajustes, IC, efectos y adecuación. | [2 Estándar](02_estandar_profundidad.md), [7 Diseño](07_diseno_riguroso.md) y [8 Verificación](08_verificacion_estadistica.md). | Borrador sin congelar; piloto, potencia, tres procedimientos válidos y campaña definitiva no ejecutados. |
| 8 Evaluación ampliada y robustez | Tres datasets, contraste entre conjuntos, perturbaciones, escala, distribución, calibración, incertidumbre, explicación, eficiencia y errores. | [8 Verificación estadística](08_verificacion_estadistica.md). | Estudios especificados; conjuntos adicionales aún sin elegir y resultados E3 pendientes. |
| 9 Dashboard y despliegue | Arquitectura, exportaciones, capturas, diseño, ficha, despliegue, humo y URL. | [9 Dashboard](09_dashboard_predictivo.md) y [10 Render](10_despliegue_render.md). | Diseño documental y artefactos de desarrollo; app, runtime ONNX, URL Render, capturas y humo pendientes. |
| 10 Conclusiones y limitaciones | Hallazgos, veredicto R1/R2/R3, amenazas, riesgos, trabajo futuro y uso de IA. | [16 Conclusiones](16_conclusiones.md), [15 Integridad](15_integridad_academica.md) y [anexo final](cierre_reproducibilidad.md). | Conclusiones provisionales sobre evidencia existente; conclusiones empíricas del candidato pendientes. La sección 14 de rúbrica no las sustituye. |
| 11 Reproducibilidad y bibliografía | Contenedor, semillas, tabla maestra, lista de verificación y referencias BibTeX. | Esta sección 11 y [anexo final de reproducibilidad y bibliografía](cierre_reproducibilidad.md). | Fuentes, auditorías y bibliografía conectadas al libro; contenedor científico y reproducción independiente de la campaña E3 pendientes. |

Las secciones 2–3 explican el estándar y sus dependencias; la [12 inventaría entregables](12_lista_entregables.md); la [13 organiza la sustentación](13_sustentacion.md); la [14 conserva la evaluación y los topes de la guía](14_evaluacion_calificacion.md). Esas secciones complementan el reporte científico. El [anexo final](cierre_reproducibilidad.md) reúne conclusiones provisionales, checklist, autoría/IA y bibliografía; deberá actualizarse con evidencia definitiva sin convertir pendientes en resultados.

## 11.2 Requisitos técnicos del libro

**Configuración, índice y referencias.** Existen `_config.yml` y `_toc.yml` propios de E3. La organización incorpora `book_e3/references.bib` con los siete registros ya presentes en la bibliografía del candidato, junto con su conexión BibTeX y citas en el cierre. No se alteran los registros originales de `_static/` ni se atribuye lectura completa a cada referencia por el hecho de incluirla. Los dos originales leídos focalmente y los antecedentes heredados o de sólo metadatos mantienen sus estados en la matriz de fuentes. La búsqueda completa de novedad sigue pendiente.

`referencias_metodologicas_v1.bib` incorpora por separado las 37 referencias sugeridas por la guía. Su [índice de aplicabilidad](referencias_sugeridas.md) conserva los números originales y distingue metodología temporal, clasificación y otras tareas. Son metadatos aportados, no 37 nuevas lecturas ni inclusiones en la revisión de novedad. El archivo original de siete referencias y su descarga publicada permanecen intactos.

Las citas y referencias cruzadas deben comprobarse mediante una construcción estricta. El fichero bibliográfico descargable que ya existía como recurso estático se distingue de la bibliografía enlazada por la configuración. Añadir citas funcionales mejora la trazabilidad; no demuestra R1 ni sustituye la evaluación de trabajos cercanos.

**Ejecución y productos precalculados.** La configuración actual conserva `execute_notebooks: off`: construir HTML no ejecuta los experimentos, entrena modelos o regenera automáticamente cada tabla y figura. El informe presenta evidencia guardada y procedimientos documentados. Para cerrar la reproducibilidad científica deberán aportarse celdas o ejecutores comentados que generen cada producto, con entradas, versiones y comandos verificables. Los cálculos costosos podrán conservarse precalculados y versionados si se incluye el código que los produjo y su linaje.

Los comandos `--check` de los auditores actuales comprueban integridad, fuentes y reconstrucción de determinados informes guardados. No reproducen por sí solos una campaña de entrenamiento, toda la A2 con entradas reales o la evaluación ampliada del candidato. Las pruebas sintéticas y los resultados de E2 conservan el alcance concreto de sus recibos.

**Figuras, tablas y ecuaciones.** Cada figura debe tener título, ejes/unidades, leyenda e interpretación; cada tabla, descripción y referencia en el texto. Las barras de métricas requerirán incertidumbre válida y rotulada, sin inventar barras de error cuando falte su estimación. Los métodos y muestras que soportan esa incertidumbre deben quedar explícitos.

Las ecuaciones actuales usan MyST y símbolos definidos, pero su numeración y referencias no están completas en todo el libro. El cierre deberá asignar identificadores y números consistentes, citar las ecuaciones pertinentes y verificar el renderizado. La demostración latente disponible no se amplía a OHLCV crudo por añadir numeración o una figura.

**Construcción y publicación.** Desde la raíz del repositorio, los comandos de verificación del libro E3 son:

```bash
python -m pip install -r requirements-book.lock.txt
jupyter-book build book_e3 --all --warningiserror --keep-going
python scripts/check_book_links.py --book-dir book_e3
```

La ruta genérica `book/` de la guía se adapta aquí a `book_e3/` para construir E3 y preservar E1. Deben comprobarse páginas, navegación, descargas, ecuaciones, citas, figuras y enlaces. La publicación en GitHub Pages o una carpeta HTML comprimida debe conservar commit y procedencia; un HTML que compile no acredita ejecución científica ni el servicio Render.

**Entornos y reconstrucción independiente.** El [perfil de producción de la sección 10](10_despliegue_render.md) usa ONNX Runtime sin framework de entrenamiento y sigue pendiente. La reproducción científica requiere un perfil de contenedor separado, con framework, versiones, datos y ejecutores adecuados. No se presenta el venv de las exportaciones como una imagen Docker ni la repetición de una auditoría como una reproducción de todo el proyecto.

Una persona ajena al código deberá ejecutar la reconstrucción desde cero en el contenedor científico, obtener la tabla principal y compararla con la versión entregada. Se conservarán identidad/rol del revisor, commit, digest de imagen, datos/versiones, hardware, comandos, resultados, tolerancias y discrepancia máxima observada. El informe distinguirá diferencias por precisión, hardware y no determinismo. Esa reproducción todavía no se ha realizado para la campaña E3; una revisión de otro agente de IA no se registra como la persona independiente.

La disponibilidad de datos autorizados y artefactos es un prerrequisito: la auditoría actual mantiene límites sobre las entradas de prueba y el candidato carece de campaña de mercado. Un manifiesto, un hash o un resultado precalculado no vuelve disponibles archivos ausentes ni sustituye los insumos necesarios para reconstruirlos.

**Checklist del último capítulo.** El [anexo final](cierre_reproducibilidad.md) debe conservar la lista de verificación completa y su evidencia, con los estados siguientes actualizados según avance:

| Elemento de reproducibilidad | Evidencia que debe acompañarlo | Estado actual y límite |
|---|---|---|
| Software y dependencias | Versiones exactas, Python, librerías, locks y comandos. | Existen locks e informes de software; perfil científico completo y runtime de servicio pendientes de validación. |
| Hardware e imagen | CPU/GPU, memoria, drivers, precisión y digest de contenedor. | Hay entornos documentados de E2 y de pruebas CPU; no hay imagen científica E3 reconstruida por un tercero. |
| Semillas y determinismo | Listas por fase, generadores, flags y límites entre plataformas. | Semillas de pruebas y políticas heredadas disponibles; lista y condiciones de la campaña E3 sin cerrar. |
| Hiperparámetros y esfuerzo | Rangos, herramienta, intentos, fallos, selección, Nd y presupuesto común. | Configuración de desarrollo y bitácora disponibles; HPO y campaña de mercado E3 no ejecutados. |
| Ejecuciones y dispersión | Número de repeticiones, unidades, media/DE, IC y separación temporal/semilla. | E2 y pruebas artificiales conservan sus propios resultados; no aportan la dispersión del candidato en mercado. |
| Datos y código | Fuentes, licencias, versiones, hashes, soporte de ventanas y acceso reproducible. | Fuentes y auditorías parciales disponibles; resolver insumos faltantes y elegir los tres conjuntos adicionales. |
| Pesos y estados completos | Configuración, pesos, escaladores/calibradores y estado del optimizador cuando corresponda. | Tres artefactos heredados de inferencia disponibles; no constituyen el checkpoint completo de la campaña E3. |
| Tabla maestra y productos | Scripts/celdas por tabla y figura, versión, estimando y tolerancia. | Hay auditores de productos guardados; reconstrucción científica integral pendiente. |
| Tiempo, recursos y energía | Costes por fase/modelo, hardware, memoria y energía medida o estimada con método. | Hay medidas parciales y perfiles con alcance propio; no se inventan consumo energético ni costes totales de E3. |
| Reproducción independiente | Recibo del tercero, contenedor desde cero y máxima discrepancia contra la tabla entregada. | Pendiente de ejecutar y documentar. |
| Archivo persistente y DOI | Versión final archivada y enlace persistente a código, datos permitidos y evidencias. | Pendiente de la entrega final; no se inventa un DOI. |
| Autoría, contribuciones e IA | Aportes individuales y herramientas usadas, revisados por el autor. | Asistencia declarada; validación, firmas y defensa académica pendientes del autor, sin firmas simuladas. |
| Bibliografía y referencias cruzadas | Registros BibTeX, citas, lista bibliográfica y enlaces verificados por build. | Siete antecedentes del candidato y 37 sugerencias metodológicas en archivos separados; conservar su procedencia y cerrar la comprobación estricta del conjunto. |

El cierre mantendrá conclusiones acordes con lo realmente observado. R1, comprensión/defensa de R2 y R3 no se acreditan mediante esta reorganización, la bibliografía o una compilación. Los resultados desfavorables, ausencias y límites de reproducción deben permanecer junto a las afirmaciones que condicionan.
