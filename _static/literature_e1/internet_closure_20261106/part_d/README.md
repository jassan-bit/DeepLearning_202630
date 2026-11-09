# Ampliación complementaria: dos fuentes externas a la cola de 22

Estas fichas no pertenecen a los 22 originales pendientes y no los sustituyen. No cambian el denominador histórico N7.

- E1X002, Acevedo-Prins y Velásquez (2026), DOI 10.3390/jrfm19090688: acceso parcial al texto primario indexado y revisión estática del repositorio del autor. TCN autónoma confirmada en el código enlazado: cuatro bloques residuales, dos convoluciones causales por bloque y dilataciones 1/2/4/8. Hay LSTM-SSE independiente y no se verificó GRU independiente. El artículo directo devolvió 429 y no se obtuvo texto continuo completo; tampoco se comprobó un release coincidente con sus tablas. Conservar como candidato complementario con verificación parcial, sin usar métricas editoriales no leídas.
- E1X003, Zhou y colaboradores (2025), DOI 10.1186/s40854-025-00768-x: original editorial completo accesible en HTML y PDF52 páginas. Comparación real de volatilidad realizada diaria; verifica LSTM autónoma, pero sus convoluciones temporales son módulos de EMGNN. No aporta TCN ni GRU autónomas. Espacio HPO y Tabla5 fueron inspeccionados directamente en PDF.

El repositorio de E1X002 se inspeccionó a commit 9c8aef3506b1e676f181c5af5b1db7f1834212fb, conservado en /tmp para lectura. No se ejecutó código ajeno ni se copiaron originales o código de terceros a esta carpeta. El HEAD incluye cambios experimentales y metadatos placeholder; la presencia del builder TCN no prueba que ese HEAD reproduzca las cifras publicadas.

reviews.json mantiene el mismo conjunto de campos de part_a. consultations.json registra las consultas y accesos fallidos. read_sources.bib distingue la fuente completa de la lectura parcial. No se guardan textos completos, resúmenes exportados, PDFs o contactos. SHA del original nulo significa que no se materializaron bytes del PDF, aunque el PDF de Springer se leyó con la herramienta web.
