# Revisión posterior de la búsqueda Colab de R1

Este expediente continúa la revisión de R1 con los resultados reales de Colab del 18 de noviembre de 2026. Fue preparado con asistencia de IA. **No contiene una revisión humana independiente ni acredita R1.**

## Procedencia y conservación

- Código ejecutado: `314f6e7e888b0748c8790cf3c0d9f4fc71bd6a0b`.
- Run: `20261118T202323797581Z`.
- Commit de resultados: `edf6fe312987fd7299f6ca850a20a67e4f1c3cbc`.
- Árbol de resultados: `fe8f43cb97d0cc6d6f462bfc182e9ee9f56101d2`.
- Ruta original: `runs/r1_search/20261118T202323797581Z/`.

Se verificaron los 27 archivos persistidos, 1.626.702 bytes, tamaños, SHA-256 y objetos Git. Los 26 miembros del manifiesto corresponden a 24 respuestas crudas, el recibo y el CSV inicial; el archivo número 27 es el propio manifiesto. Los 10.966 blobs anteriores del repositorio permanecieron idénticos. Los originales, incluido `screening.csv` con sus estados `not_screened`, se conservan sin editar.

`colab_receipts.zip` es una copia empaquetada posterior de esos 27 archivos verificados; no se presenta como un ZIP producido por Colab. `byte_audit.json` y `source_preservation.json` documentan la comprobación. Las decisiones posteriores se guardan en archivos nuevos y no sustituyen las respuestas nativas.

## Lo que demuestran las respuestas

Crossref, arXiv y OpenAlex respondieron correctamente a las seis consultas: 18 respuestas válidas y 180 ocurrencias de metadatos en la primera página. Las treinta solicitudes previstas se conservaron en el recibo. DBLP devolvió seis cuerpos HTML con HTTP 202 no analizables como bibliografía; Semantic Scholar respondió seis HTTP 429. Estos doce intentos mantienen conteos desconocidos y no añaden bases efectivas.

La cobertura de respuestas de tres bases está comprobada. **No equivale a una búsqueda exhaustiva ni a una declaración de ausencia de antecedentes.** El feed de arXiv expande las palabras no entrecomilladas con OR; las consultas españolas y los términos amplios devuelven trabajos de otras áreas. Crossref incluye componentes de figuras. Los conteos entre bases y entre etapas no son comparables como medidas de relevancia.

## Cribado y revisión adicional

`screening/` documenta decisiones sobre los 180 títulos recuperados, mantiene la procedencia de cada consulta y distingue duplicados, componentes, contexto y candidatos para lectura. Cribar títulos no equivale a leer resúmenes o textos completos. Tampoco completa una revisión de todas las páginas de resultados.

`ai_review/` contiene una revisión técnica adicional por IA, iniciada sin acceso a los dictámenes anteriores del proyecto. Su alcance, fuentes realmente consultadas y limitaciones se declaran en sus archivos. Una revisión con contexto inicial separado no convierte a una IA en persona, no prueba independencia académica humana ni sustituye el concepto externo exigido por la guía.

## Cierre pendiente

Debe delimitarse una diferencia sustantiva respecto de los mecanismos próximos, completar el cribado y las lecturas pertinentes, refinar las consultas cuando su sintaxis no represente la pregunta y repetir la búsqueda antes de la entrega. Una persona ajena al diseño debe repetir al menos dos cadenas y emitir su propio concepto identificado y fechado. El [formulario preparado](https://github.com/jassan-bit/DeepLearning_202630/blob/314f6e7e888b0748c8790cf3c0d9f4fc71bd6a0b/experiments/e3_r1_review_20261118/reviewer/REVIEW_FORM.md) conserva su estado sin diligenciar.

El expediente no modifica la arquitectura de 1210 parámetros, los entrenamientos existentes, los pronósticos o los contrastes. No habilita R2, R3 o una campaña real. La aceptación académica de R1 corresponde al evaluador; el resultado de la revisión puede ser favorable, desfavorable o no concluyente.
