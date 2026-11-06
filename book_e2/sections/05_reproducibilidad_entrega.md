# Página de trabajo · Reproducibilidad y entrega

Este libro se construye desde `book_e2/`, con `_config.yml` e `_toc.yml` propios. Comparte el repositorio y el entorno editorial con E1/E3, pero su compilación no importa las carpetas de esos libros ni ejecuta sus experimentos.

Esta página es provisional. La guía recibida sitúa reproducibilidad, ética y comunicación en su sección 8; se incorporará su detalle cuando esté disponible.

Desde la raíz del repositorio, después de instalar `requirements-book.lock.txt` en un entorno Python 3.12:

```bash
.venv/bin/jupyter-book build book_e2 --all --warningiserror --keep-going
.venv/bin/python scripts/check_book_links.py --book-dir book_e2
```

El workflow común publica este HTML en `/entregable2/`, junto con los otros libros en sus rutas. La construcción del informe no entrena redes ni abre datos de mercado.

Ya están disponibles los metadatos propios de E2: [contrato heredado](../_static/e1_reference.json), [Data Card](../_static/dataset_card.json), [manifiesto de la fuente](../_static/dataset_manifest_e1.json) y [registro de las secciones 1–5 recibidas](../_static/guia_e2_recibida.json). La bibliografía editorial de E2 se conserva en `references.bib`.

El EDA ampliado se reproduce con `scripts/eda_e2_decisions.py` y `scripts/eda_e2_known_shift.py`, usando el entorno científico de `requirements-eda.lock.txt`. Sus {ref}`instrucciones, tablas y figuras <e2-reproduccion-eda>` conservan hashes de fuentes y separan desarrollo del reanálisis de la reserva ya conocida. El build editorial no ejecuta esos scripts.

Las fuentes primarias y la propuesta bibliográfica están en [sección 3](03_literatura.md). El [paquete de literatura](../_static/literature_delivery/Literatura_E2_codigo_matrices_diagramas.zip) reúne scripts, tablas, fichas y recibos verificables; no redistribuye los PDF ni las exportaciones nativas licenciadas. La compilación de la extracción puede verificarse sin adjuntos privados con `python scripts/compile_e2_fulltext_review.py --check`. El código adaptado, piloto de dimensionamiento y registros del benchmark Bitcoin D7 del Top 10 y sus referencias se incorporarán tras cerrar el catálogo y el protocolo con la guía restante. La existencia de diagnósticos de EDA o métricas extraídas de artículos no significa que ese benchmark esté ejecutado.

La [sección 4](04_implementacion.md) añade un [paquete de diseño y auditoría](../_static/benchmark_design_download/Diseno_E2_propuesta_auditoria.zip). `python scripts/audit_e2_benchmark_readiness.py` verifica metadatos, contratos y hashes; no entrena modelos. Su código cero significa propuesta documental válida, con pendientes explícitos. `--require-ready` devuelve dos mientras falten los prerrequisitos; no es un permiso de entrenamiento ni acredita test desconocido, potencia o ausencia de fuga.


La [sección 5: Paper guía y replicación](05_paper_guia.md) contiene ejecuciones externas ya realizadas: cinco ajustes de TimeMixer en ETTh1 y quince de un piloto sobre TimeMixer, iTransformer y PatchTST. Sus datos, condiciones L96/H96, métrica MSE y criterio de compatibilidad pertenecen a esa réplica; no cambian el contrato Bitcoin D7, su RMSE primaria ni su margen δ. El diseño se registró antes de los ajustes y las discrepancias se conservan aunque no favorezcan la compatibilidad numérica.

El [paquete de réplica y piloto](../_static/paper_guide_download/Paper_guia_E2_replica_piloto.zip) reúne código propio, diseño, lock y agregados verificables, con un [manifiesto SHA-256](../_static/paper_guide_download/MANIFIESTO.json). No contiene el CSV ETTh1, fuentes descargadas de terceros, pesos ni predicciones individuales. Las instrucciones describen su adquisición por versiones y hashes para nuevas reproducciones explícitas. Para verificar que el ZIP corresponde a los materiales publicados, sin volver a entrenar:

```bash
python scripts/package_e2_paper_guide.py --check
```

Los recibos y resultados guardados de ETTh1 se incorporan al HTML como evidencias. La compilación y el ensamblado editorial siguen realizando cero ajustes y cero predicciones: esos conteos se refieren a la operación de publicación, no al total histórico de experimentos E1/E2.
