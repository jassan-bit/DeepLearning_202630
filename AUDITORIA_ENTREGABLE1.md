# Auditoría del Jupyter Book — Entregable 1

Fecha: 2 de noviembre de 2026. Alcance: contenido, requisitos, evidencia, numeración, figuras y reproducibilidad. Referencia: `references/course/entregable1_texto.txt` y materiales del curso.

Versiones auditadas: fuente `701dfd559c986aaef44dfa5cb39d44595e3494d6` y rama publicada `1013b4fc357ed1a269aab3f962af8a9f04c6df7e`.

## Dictamen

**El libro tiene estructura y protocolos redactados, y una adquisición de datos comprobada. No tiene un EDA ejecutado ni gráficas de EDA. Tampoco constituye todavía el Entregable 1 completo.**

Se inspeccionó el HTML de `sections/02_eda.html` obtenido directamente de la rama `gh-pages`: contiene **cero imágenes dentro del artículo**. En la fuente tampoco hay figuras, notebooks ni scripts de análisis EDA. Su ausencia no se explica por un enlace de imagen roto o por una gráfica escondida en el sitio.

La consulta HTTPS al sitio público fue bloqueada por el proxy del entorno. Por ello, esta auditoría verifica los archivos fuente y la rama publicada, pero no certifica la versión que devuelve el CDN ni la representación visual en un navegador externo.

## Hallazgos priorizados

### A01 — Alta: faltan las evidencias y gráficas del EDA

Evidencia: `book/sections/02_eda.md:6`, `:15`, `:64`, `:72`, `:87`, `:164` y `:198`. Los apartados están escritos como acciones futuras. El artículo publicado tiene 20 encabezados y ninguna imagen. `scripts/` solo contiene el descargador y el comprobador de enlaces; no hay un pipeline de EDA.

Impacto: los requisitos están enumerados, pero no satisfechos con resultados. Faltan tablas estadísticas, gráficos, pruebas y conclusiones técnicas derivadas de datos.

Acción: implementar y ejecutar el EDA reproducible en desarrollo, guardar resultados y figuras e incorporarlos a cada apartado. Los análisis de precios, retornos y huecos pueden comenzar con los datos existentes. El EDA del objetivo requiere primero cerrar su definición.

### A02 — Alta: el objetivo y el protocolo no están cerrados

Evidencia: `book/sections/01_problema_dataset.md:45`, `:49`, `:116`, `:118`, `:122` y `:134`.

Faltan horizonte físico, frecuencia del retorno objetivo, ventana de entrada, cadencia de pronóstico, rezago de MASE, margen práctico y definición de bloques inferenciales. La longitud de entrada puede justificarse después del EDA temporal de desarrollo; no es necesario inventarla para comenzar la auditoría de calidad y retornos.

Impacto: no puede afirmarse que un histograma de una volatilidad elegida arbitrariamente describa el objetivo definitivo ni que exista un benchmark reproducible.

Acción: fijar y registrar las decisiones antes del entrenamiento y del análisis de test, diferenciando horizonte de agregación, horizonte de predicción y número de salidas.

### A03 — Alta: faltan los tres cierres explícitos en el EDA general

La guía exige que cada análisis concluya con decisión de preprocesamiento, impacto en arquitectura y riesgos identificados (`references/course/entregable1_texto.txt:333`). En los apartados 2.1.1–2.1.5 no aparecen sistemáticamente esos tres cierres; véanse `book/sections/02_eda.md:17`, `:21`, `:40` y `:42`.

El bloque 2.2 sí usa ese formato, aunque todavía como protocolo. Acción: añadir los cierres a todos los análisis y sustentarlos con resultados cuando se ejecuten, en lugar de limitarse a intenciones generales.

### A04 — Alta: la revisión bibliográfica y los tres modelos no se han ejecutado

Evidencia: `book/sections/03_estado_arte.md:43`, `:93`, `:128`, `:140` y `:224`. Las consultas no tienen conteos reales, la síntesis está vacía y las fichas de modelos están pendientes.

Impacto: no hay tres arquitecturas seleccionadas desde la literatura, tablas de frecuencia, diagramas específicos, ecuaciones ni resultados publicados extraídos para sustentar la selección.

Acción: ejecutar la revisión trazable y completar los productos de 3.1–3.3. No convertir las familias candidatas en modelos seleccionados sin evidencia.

### A05 — Alta: faltan productos posteriores del Entregable 1

No existen la sección 3.4 Comparación Crítica ni los capítulos 4 Implementación y Evaluación y 5 Benchmark y Definición del Baseline en `book/_toc.yml` ni en sus archivos. La guía los exige en las líneas 452, 467 y 614 del texto local.

Su ausencia es coherente con un avance parcial, pero impide declarar terminado el entregable. Agregar encabezados no bastará: requieren análisis crítico, implementación, piloto, evaluación estadística y un baseline medido.

### A06 — Media: persisten textos desactualizados

| Ubicación | Problema |
|---|---|
| `book/intro.md:6` y `book/sections/protocolo_pendiente.md:24` | Presentan el avance como solo sección 1, aunque hay capítulos 1–3 |
| `book/sections/01_problema_dataset.md:15`, `:17`, `:21`, `:23` | El resumen dice que faltan fuente, adquisición o hashes, pese a que §1.5 registra la descarga |
| `book/sections/02_eda.md:34` | Dice desconocer atributos de proveedor/cotización ya identificados como Binance y USDT |
| `book/sections/02_eda.md:127`, `:138` | Describe adquisición en futuro pese al estado completado de §2.3.2 |
| `book/sections/02_eda.md:200`, `:201` | No distingue datos brutos ya adquiridos de muestras, folds y test derivado todavía sin congelar |
| `book/sections/referencias.md:20` | Menciona dataset aún no seleccionado, aunque Binance Spot BTCUSDT está confirmado |

Acción: reconciliar todos los resúmenes con un único registro del estado de ejecución. Estas correcciones editoriales no sustituyen los análisis pendientes.

### A07 — Media: integridad de adquisición no equivale a calidad financiera

La evidencia ejecutada corresponde a SHA-256, estructura CSV y timestamps. `scripts/download_binance_spot.py:65` no analiza cierres, coherencia OHLC, volumen ni etiquetas. El libro identifica correctamente esta limitación en §2.3.2.

El manifiesto contiene 75 ZIP, 2.014.480 filas, 80 minutos ausentes y un cierre con timestamp inesperado. Debe investigarse su tratamiento antes de construir retornos y ventanas, conservando los brutos y evitando imputación con futuro.

### A08 — Media: la publicación automática requiere revisión técnica

El workflow construye el HTML y fuerza una rama `gh-pages` usando `github.token` (`.github/workflows/publish-book.yml:33`). Los pushes de `GITHUB_TOKEN` no activan automáticamente la publicación de Pages por rama. Las publicaciones recientes se han realizado manualmente desde esta sesión.

Acción: adoptar un flujo de despliegue de Pages explícito o documentar y verificar el procedimiento de publicación manual. Esto puede explicar actualizaciones que no se vean, pero no explica las gráficas ausentes: tampoco existen en el HTML de la rama publicada.

## Matriz del EDA y sus evidencias

| Apartado | Ítem redactado | Evidencia ejecutada | Producto pendiente |
|---|---|---|---|
| 2.1.1 Objetivo | Sí | No | Histograma/ECDF, cuantiles, asimetría, curtosis y análisis de colas del objetivo definido |
| 2.1.2 Calidad | Sí | Parcial: adquisición y timestamps | Auditoría de valores/OHLC, faltantes por variable y tiempo, ruido y tratamiento documentado |
| 2.1.3 Sesgos y desplazamiento | Sí | No | Comparaciones de desarrollo, cambios de distribución e interpretación temporal |
| 2.1.4 Partición | Calendario propuesto | No hay muestras ni folds materializados | Diagrama temporal, cantidades, proporciones y reglas verificadas |
| 2.1.5 Test congelado | Protocolo | Hashes brutos disponibles; test derivado no congelado | Manifiesto de muestras, reglas y código del test final |
| 2.2.1 Visualización y descomposición | Sí | No | Series, descomposición y diagnóstico de regímenes |
| 2.2.2 Estacionariedad | Sí | No | ADF/KPSS, especificaciones, resultados e interpretación conjunta |
| 2.2.3 Dependencia | Sí | No | ACF/PACF con bandas, espectro, ARCH y evaluación de no linealidad |
| 2.2.4 Frecuencia y huecos | Sí | Parcial: 80 minutos ausentes y un cierre irregular | Localización de huecos, política causal y efecto en ventanas |
| 2.2.5 Anomalías | Sí | No | Criterio, eventos señalados, explicación y sensibilidad del error |
| 2.2.6 Horizonte y validación | Sí | Propuesta | Objetivo cerrado, ventanas/folds implementados y controles contra fuga |

No se exige desbalance de clases a un objetivo continuo ni análisis de imágenes, texto o geografía a esta serie única. La ausencia de esos bloques no es un incumplimiento del alcance actual.

## Gráficas necesarias para comenzar

1. Serie de cierres y retornos de entrenamiento, con unidades y huecos visibles.
2. Cobertura temporal, intervalos ausentes y anomalías de timestamps.
3. Histograma y ECDF de retornos; del objetivo cuando se defina su horizonte.
4. Descomposición y evolución de estadísticas móviles, con periodos justificados.
5. ACF/PACF y periodograma sobre tramos válidos, sin unir extremos de huecos.
6. Comparación de distribuciones por bloques de desarrollo y diagrama de partición.

Las pruebas ADF, KPSS y ARCH necesitan resultados tabulares e interpretación, no solo figuras. Cada análisis debe cerrar con las tres implicaciones exigidas por el entregable.

## Comprobaciones que sí pasaron

- Construcción con advertencias como errores: completada.
- Comprobador local: 9 páginas HTML y 520 enlaces/recursos válidos.
- Numeración de capítulos 1–3 y sus subapartados: jerarquía coherente en el alcance escrito.
- Manifiesto del repositorio y copia servida por el libro: idénticos.
- Fuente directa identificada, adquisición documentada y checksum individual de los 75 ZIP registrado.
- El HTML publicado no contiene referencias al proyecto de Machine Learning anterior; la referencia científica cuyo título incluye “machine learning” es un artículo, no ese proyecto.

Estos controles verifican estructura y adquisición; no acreditan que el EDA, los modelos o las inferencias estén ejecutados.

## Orden recomendado de corrección

1. Corregir estado y textos desactualizados, distinguiendo protocolo de resultados.
2. Ejecutar calidad, cobertura y visualizaciones de precios/retornos en entrenamiento de 2023; contrastar con validación de 2024 sin explorar el test para ajustar decisiones.
3. Cerrar el objetivo y horizonte; usar el EDA de desarrollo para justificar ventana y transformaciones.
4. Completar el EDA general y temporal, con resultados, figuras, código y conclusiones técnicas.
5. Ejecutar la revisión y seleccionar los tres modelos; desarrollar 3.4, implementación, piloto y benchmark.

La descarga hasta octubre de 2026 y el test propuesto hasta agosto de 2026 son distintos por diseño y están explicados en el libro. No debe ampliarse el test automáticamente solo por disponer de septiembre–octubre.

Esta auditoría no modifica el protocolo experimental ni genera resultados de EDA. Los hallazgos describen la instantánea indicada al inicio.
