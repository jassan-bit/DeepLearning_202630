# Ampliación bibliográfica E2, 08/11/2026

Expediente nuevo, separado del corte 05–06/11 y de todos los protocolos y resultados experimentales. native/ documenta 1.634 ocurrencias WoS/Scopus/IEEE y 1.078 grupos. arxiv/ documenta cuatro consultas API, 1.199 apariciones y 1.004 IDs. Estos universos NO están conciliados entre sí ni con los 7.833 grupos históricos; no sumar sus denominadores.

Los metadatos públicos excluyen exports originales, resúmenes completos, direcciones, afiliaciones y cuerpos de artículos. focal_source_reviews.json registra tres lecturas dirigidas y hashes de sus cuerpos privados, no una revisión exhaustiva. La clasificación es provisional y la elegibilidad final sigue sin determinar.

Desde la raíz del repositorio: `python scripts/publish_e2_native_literature.py --check` valida conteos, pertenencia, privacidad por columnas, hashes y equivalencia de la copia del libro. Sin --check regenera manifiesto y copia pública desde esta lista cerrada. No ejecuta búsquedas ni entrenamientos. Para repetir el triaje con los mismos originales privados, seguir native/LEEME.md fuera de Git; los scripts rechazan ejecución dentro de un checkout.

La hora de recuperación API proviene de su recibo. Los nombres de exportación y campos DA no se convierten en una hora de consulta que no acreditan. La consulta exacta Scopus y la reconciliación de versiones permanecen pendientes. Se conserva el conflicto título/año con DOI diferentes y no se equiparan grupos de referencias con estudios empíricos independientes.
