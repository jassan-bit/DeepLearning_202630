# Cribado delimitado de la revisión bibliográfica

La colección original de Colab se conserva sin cambios: 180 ocurrencias,
166 identidades canónicas y 14 duplicados por DOI/arXiv. No son 166 artículos
únicos verificados. Se recogieron únicamente los primeros diez resultados de
cada una de seis cadenas en Crossref, arXiv y OpenAlex; los conteos nativos no
equivalen a trabajos pertinentes. La sintaxis y el ordenamiento difieren entre
bases, y las cadenas iniciales tienen limitaciones de precisión documentadas.

De 50 registros priorizados por título, el cierre distingue **41 resúmenes
primarios completos observados**, **tres extractos parciales de resumen**,
**una ficha/listado sin resumen** y **cinco resúmenes primarios no obtenidos**.
Se leyeron 32 resúmenes de las respuestas nativas conservadas y nueve mediante
fuentes primarias adicionales. Los 50 tienen una disposición registrada,
incluidos sus límites de acceso; no se afirma haber leído los 50 resúmenes ni
todos sus textos completos. La comparación focal de diez trabajos y las
lecturas de secciones están en `primary_review/`.

`abstract_screening_all_50.csv` conserva las identidades, decisión, paráfrasis,
URL/localizador y alcance de lectura. Las referencias a respuestas antiguas
fijan el commit original y su hash. La fusión usa identidades exactas, no
semejanzas de título. El resumen de estados es
`abstract_screening_summary.json`.

El 20 de noviembre UTC —19 de noviembre en Bogotá— se intentaron seis cadenas
focales bilingües en tres API nativas: **18 intentos**, todos fallidos antes de
una respuesta bibliográfica por el túnel de red. Los conteos de estas consultas
nuevas permanecen desconocidos, no cero. El recibo público de intentos es
`fresh_native_search_receipt.json`; las respuestas válidas de la colección
anterior no se borran ni se convierten en ejecuciones nuevas.

La modalidad de revisión externa por IA declarada ya fue adoptada en 6.13
conforme a la indicación docente comunicada por el autor. Este trabajo no
inventa firmas humanas ni vuelve a exigir una revisión humana de R1.

No se acredita R1 por no encontrar una composición exacta en una búsqueda
delimitada. El examen de antecedentes puede terminar con un dictamen negativo
o no concluyente sobre la contribución actual. No hubo ajustes de modelos,
predicciones de mercado o contrastes estadísticos nuevos.

## Archivos adecuados para publicación

- `abstract_screening_all_50.csv`: paráfrasis y trazabilidad, sin resúmenes
  completos copiados.
- `abstract_screening_summary.json`.
- `fresh_native_search_receipt.json`: consultas y errores propios, sin claves.
- `guide_evidence_status.json`.
- `source_provenance.json` y `verification.json`.
- Este `README.md`.

**No publicar** `saved_abstract_queue.json`: es una caché local de lectura que
contiene resúmenes completos extraídos de los insumos existentes. Tampoco es
necesario volver a publicar los bytes nativos originales, ya conservados en
GitHub. Los scripts auxiliares y `source_inputs.json` contienen rutas locales
de preparación y se conservan sólo como apoyo de ejecución, no como evidencia
de lectura de textos adicionales.
