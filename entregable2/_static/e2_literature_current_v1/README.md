# Evidencias vigentes de la revisión de literatura

Este índice acompaña el capítulo 3 de E2 y su revisión según los apartados 3.1–3.4 de la guía. Consolida evidencias existentes; las búsquedas, lecturas y resultados conservan sus fechas y versiones originales.

## Lectura del expediente

| Producto | Evidencia y alcance |
|---|---|
| Flujo recibido | 11.635 ocurrencias − 1.720 duplicados internos − 356 enlaces entre rondas = **9.559 grupos** con disposición. **42 estudios aplicados únicos**, 120 evaluaciones primarias focales o reutilizadas y **cero pendientes operativos**. |
| Cobertura histórica | Permanecen 374 registros Scopus mostrados pero no recibidos y correspondencia incompleta de totales/rangos WoS. El flujo reconciliado no demuestra cobertura exhaustiva de las bases. |
| Acceso y elegibilidad | 174 informes no recuperados con intentos documentados. No son exclusiones científicas. Los dos criterios DL no identificados tras examinar fuentes conservan incertidumbre científica, sin otra tarea de lectura pendiente en este corte. |
| Extracción y calidad | 42 estudios × doce dimensiones = 504 filas. Para las fuentes arquitectónicas: nueve estudios y diez sistemas; FCN y ResNet comparten un estudio. NR y NV tienen significados distintos. No se afirma doble revisión humana ni se calcula κ. |
| Selección | Diez sistemas y cuatro familias, más seis referencias en el benchmark. Se conservan la justificación individual, los escenarios de cinco componentes y la comparación estratificada. El desempeño global no está calibrado entre tareas y no se identifica un ranking óptimo global. |
| Búsqueda transversal | Tres consultas nativas ejecutadas: totales 106/78/3 y metadatos recuperados 10/10/3. Las 23 ocurrencias no forman una unión deduplicada ni 23 estudios incluidos. Las fuentes originales y los límites de transferencia están documentados. |
| Reproducibilidad externa | Código de autor y fuente pública de datos identificados para nueve de nueve estudios. No equivale a ejecutar todos los artículos ni a verificar todos sus snapshots experimentales. |
| Comparación crítica | Se distinguen tarea, datos, escala, horizonte, lookback, versión, resultado y presupuesto. El gap se formula de manera refutable dentro de las fuentes examinadas; no se afirma inexistencia universal de antecedentes. |

La guía pide un puntaje explícito y reproducible, con desempeño en datasets comparables. No prescribe que el puntaje bibliográfico deba preceder al benchmark. Se declara la cronología real: el análisis documental posterior no se atribuye como causa de la selección anterior.

## Archivos de esta consolidación

- `current_evidence_index.json`: cifras, límites y 24 rutas de evidencia con SHA-256.
- `guide_checklist.json` y `guide_checklist.csv`: cotejo por condición de la guía, con evidencia y alcance.
- `decision_traceability.csv`: hallazgos, decisiones y consecuencias para la revisión y el diseño experimental.
- `03_literatura_before_update.md`: copia exacta del capítulo anterior a esta consolidación; sus pendientes corresponden a aquel texto.
- `manifest.json`: tamaños y hashes del expediente.

Los derivados anteriores mantienen sus bytes. Los campos de ejecución de la extracción bibliográfica antigua no sustituyen las recetas ejecutadas: consultar los capítulos de cuerpos publicados y protocolo reducido. Los costes de los artículos y las mediciones de E2 conservan condiciones y denominadores separados.

## Verificación

Desde un checkout completo del repositorio, con Python 3.12, ejecutar:

```bash
python scripts/verify_e2_prisma_bundle.py references/literature_e2/prisma_received_v1
```

Es una comprobación de lectura: valida hashes, conservación del flujo, identidades, versiones, motivos, acceso y dimensiones de calidad. No recupera artículos, no vuelve a cribar ni entrena modelos. El resultado esperado incluye 9.559 grupos, 42 estudios, 504 filas aplicadas y cero pendientes operativos.

Los hashes de `current_evidence_index.json` permiten comprobar las fuentes enlazadas. El paquete de PRISMA conserva su propio manifiesto. Los originales con resúmenes, cuerpos de artículos y notas extensas no se redistribuyen en este índice; los enlaces públicos usan metadatos, localizadores, hashes y síntesis propias breves.
