# Entregable 2: doce archivos independientes

Cada ZIP contiene una carpeta D01…D12 con run.py, LEEME.md, sus entradas y un manifiesto. Extraer todo el ZIP antes de ejecutar. No hace falta clonar el repositorio, visitar la página web o descargar otro entregable.

Comando común: `python run.py` (Windows: `py -3.12 run.py`; Linux/macOS: `python3 run.py`). Python 3.12 recomendado. D02 necesita instalar requirements_cpu.txt; los otros once comandos predeterminados usan sólo la biblioteca estándar. Los paquetes incluyen lanzadores ejecutar.cmd/ejecutar.sh.

| ID | Archivo | Salida principal |
| --- | --- | --- |
| D01 | 01_informe_jupyter_book.zip | Extrae el libro HTML offline, fórmulas locales y fuentes para reconstruirlo. |
| D02 | 02_codigo_modelos.zip | Ejecuta las 13 arquitecturas en CPU; incluye fuentes de autores y entrenamiento CUDA local opcional. |
| D03 | 03_datos_contrato.zip | Reconstruye σ₇ y verifica los CSV diarios incluidos; ofrece descarga oficial verificada de datos crudos. |
| D04 | 04_preregistro_auditoria.zip | Genera el informe local con protocolo, congelación, controles ejecutados y pendientes. |
| D05 | 05_ficha_ruta_F2a.zip | Genera la ficha y tablas de pruebas, unidades, p-valores, efectos e IC disponibles. |
| D06 | 06_benchmark_comparaciones.zip | Recalcula métricas de los 81 predictores y reconstruye A2; incluye reanálisis estadístico opcional. |
| D07 | 07_registro_experimentos.zip | Genera tablas y fichas desde configuraciones, semillas, resultados y costes incluidos. |
| D08 | 08_techo_incertidumbre.zip | Reconstruye cobertura original, 6.080 categorías y el proxy de techo con pesos guardados. |
| D09 | 09_arquitecturas_no_aplica.zip | Genera y valida la justificación del catálogo actual con arquitecturas definidas. |
| D10 | 10_paper_guia_replica.zip | Reproduce los agregados de cinco semillas; incluye ETTh1, autores y entrenamiento CPU opcional. |
| D11 | 11_reproducibilidad.zip | Genera la lista de 21 ítems, sus estados, evidencia y recibos. |
| D12 | 12_presentacion_defensa.zip | Genera 14 diapositivas HTML, notas del orador y tablas de apoyo; se pueden imprimir a PDF. |

Las salidas se generan dentro de cada paquete. D01 abre un sitio local completo, con fórmulas y fuentes; puede recompilarse con Jupyter Book. D02 produce un informe de los 13 forwards CPU con fixtures: no reproduce precisión de mercado desde pesos entrenados. D03 genera dataset_audit.html y CSV reconstruidos. Los informes generales se abren en output/informe.html; la presentación está en output/presentacion.html.

Los reentrenamientos opcionales de D02 requieren Linux/WSL2 y CUDA; D10 permite una nueva réplica CPU. Las dependencias, tiempos y comandos están en cada LEEME.md. Estas opciones no se ejecutan por defecto. Los programas originales y sus fuentes están incluidos; los resultados locales se separan de los registros históricos.

D09 documenta NAS no aplicable al catálogo actual. D12 prepara la defensa, sin acreditar que su evaluación oral haya ocurrido. Los estados parciales de la guía permanecen explícitos.

index.json y SHA256SUMS identifican cada descarga. execution_report.json documenta ejecuciones después de extraer los ZIP en carpetas ajenas, con Python aislado y sin dependencias entre paquetes. Los datos y archivos científicos originales conservan sus bytes.
