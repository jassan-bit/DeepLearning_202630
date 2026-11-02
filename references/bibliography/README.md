# Bibliografía del Entregable 1

Actualización documental del 2 de noviembre de 2026: treinta referencias académicas únicas, con treinta claves citadas en el libro. Comprende siete aplicaciones a Bitcoin, siete fundamentos neuronales y dieciséis fundamentos de volatilidad, diagnóstico y evaluación. No es un registro de experimentos.

- `references_30.json`: metadatos seleccionados, fuentes efectivamente consultadas, ubicación de evidencia, hashes disponibles y límites de acceso por referencia.
- `verification_30.csv`: tabla de consulta y verificación.
- `access_checks_20261102.json`: comprobaciones de acceso, distinguiendo bloqueos de resultados bibliográficos.
- La base BibTeX está en `book/references.bib`; sus citas y bibliografía se generan con Jupyter Book y sphinxcontrib-bibtex.

La verificación de metadatos en bibliografías y documentación académica no equivale a lectura de todos los originales. El texto y las fichas distinguen esos casos. No se acredita exhaustividad de búsqueda en bases bloqueadas, cuartiles no consultados ni selección definitiva de tres modelos a partir de la cantidad de referencias. La referencia de Huang y colaboradores conserva su verificación previa. Los originales descargados para consulta no se redistribuyen en el repositorio.

Comprobación: parsear `book/references.bib`, comprobar treinta claves únicas y citadas, construir el libro con advertencias como errores y ejecutar `scripts/check_book_links.py`.
