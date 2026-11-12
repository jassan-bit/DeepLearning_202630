# 42. Auditoría de E2 y trabajo que puede adelantarse

**Actualización de catálogo (12/11/2026):** se adopta el [alcance obligatorio de 16 sistemas de §43](43_catalogo_obligatorio.md). El híbrido adicional y Ridge quedan como opcionales/históricos. Esta auditoría y sus descargas mantienen el corte anterior de 18 sistemas; para preparar el benchmark se utiliza la [nueva plantilla de 16 filas](../_static/required_catalogue_20261112/benchmark_A2_template.csv). Los demás pendientes de la guía continúan abiertos.

**Corte de auditoría: 11/11/2026.** Se compara el JBook del commit 181b9e58032c0011e46dcbe334400568fb82c258 con la guía adjunta de **65 páginas**, SHA256 **93499dcfda6a36d9621524a81821470017ba114ca6b08241a35de5a54f74a654**. Las páginas citadas son las impresas del PDF. Es auditoría documental y de código, sin nuevas corridas ni cambios de Colab; no certifica cumplimiento ni calificación.

## Dictamen

**Hay trabajo importante que podemos completar mientras entrena Colab.** Ya existen contrato D7×5, EDA con trazabilidad A.1, recetas/pruebas, tracking propio, selección HPO/NAS registrada y réplica externa con piloto. Faltan cierres bibliográficos, coherencia documental y un puente de análisis para el catálogo vigente. Terminar corridas no resuelve esas deudas.

| Prioridad | Guía | Hallazgo | Acción que puede adelantarse | Dependencia real |
|---|---|---|---|---|
| **P0** | pp. 2–3, 42–43 | Existe registro de selección desde el 10/11, pero quedan textos de ausencia de HPO/preregistro | Unificar vigencia y cronología; aclaraciones añadidas en esta actualización | El cierre inferencial/R sigue pendiente |
| **P0** | pp. 13, 47 | Los recibos públicos de selección no conservan curvas por época | Preparar exportación, retención y gráficas para futuros ajustes; probar con sintéticos | Antes de los futuros reajustes afectados; no cambiar campaña activa |
| **P0** | pp. 17–19, 44–51, 63 | Drivers de análisis/controles conservan modelos retirados | Adaptadores versionados del catálogo actual y pruebas de artefactos/soportes | Tener un script llamado final no acredita integración |
| **P1** | pp. 9–12 | Puntajes/matrices de otra cohorte omiten cuatro altas actuales | Completar tabla del Top 10, alternativas citadas y razones de exclusión | Verificación bibliográfica, sin GPU |
| **P1** | p. 10 | PRISMA global sin conciliar y consultas transversales sólo locales | Conciliar DOI/versiones/rondas y acreditar tres consultas de mecanismos vigentes | Fuentes, recuperación y decisiones de cribado reales |
| **P1** | pp. 11–12 | Fichas/diagramas existen, pero mezclan fuente, propuesta y receta | Índice único de diez fichas con formas/fases y código fijado | Arquitectura ganadora y costes finales esperan selección |
| **P1** | pp. 3–9, 45, 61 | A.1/EDA disponibles; Data Card conserva otra cohorte/calendario | Consolidar trazabilidad y preparar una nueva capa vigente de contrato | Conservar histórica; no inventar conteos finales |
| **P1** | pp. 2–3, 23, 42–44 | R y familias/reglas inferenciales abiertas | Matriz RQ–comparador–unidad–pruebas–supuestos–multiplicidad y diseño de potencia | Piloto compatible para fijar R; no bastan dos semillas HPO |
| **P1** | pp. 16, 48–50 | Análisis complementarios aún no ejecutados para el catálogo completo | Manifiestos de robustez, calibración, errores, explicabilidad y ablaciones | Recetas/pesos, Top 3, ganador y controles futuros |
| **P1** | pp. 14, 40–41, 47, 63 | Tracking GitHub real; checklist histórico comprimido | CSV de trials/configuraciones, entorno/licencias y 21 filas del Apéndice B | Formulario completo no significa requisitos aprobados |

## Evitar retrabajo

**Curvas:** la guía p. 13 exige train/validación por modelo. El [trainer CUDA](https://github.com/jassan-bit/DeepLearning_202630/blob/181b9e58032c0011e46dcbe334400568fb82c258/scripts/e2_cuda_training.py) genera logs, pero [el exportador de selección](https://github.com/jassan-bit/DeepLearning_202630/blob/181b9e58032c0011e46dcbe334400568fb82c258/scripts/run_e2_cuda_search.py) conserva resúmenes de fases y el worker usa un directorio temporal. Los JSON públicos no permiten reconstruir curvas por época. Debemos preparar la persistencia para próximos ajustes finales; no implica repetir toda la HPO ni modificar el controlador activo.

**Integración, corte del 11/11/2026:** había **14 sistemas neuronales y cuatro comparadores: 18 sistemas**. El [driver final histórico](https://github.com/jassan-bit/DeepLearning_202630/blob/181b9e58032c0011e46dcbe334400568fb82c258/scripts/run_e2_final_benchmark.py) y [su estadística](https://github.com/jassan-bit/DeepLearning_202630/blob/181b9e58032c0011e46dcbe334400568fb82c258/scripts/e2_final_statistics.py) aún codifican 20 identidades/familias anteriores; los [controles de fuga](https://github.com/jassan-bit/DeepLearning_202630/blob/181b9e58032c0011e46dcbe334400568fb82c258/scripts/e2_leakage_controls.py) tampoco equivalen a la auditoría de las 14 redes de ese corte. Podemos preparar ahora adaptadores y pruebas sintéticas de identidad, calendario, máscara común, pares y fallos. Ejecutar el comando histórico completo no cierra este benchmark.

**Top 10:** la tabla de puntajes no incluye FCN-1D, ResNet-1D, Inception-1D ni LSTM-autoencoder actuales. Hay seis fichas con diagramas gráficos y cuatro altas con esquemas ASCII: no faltan todos los diagramas. Deben distinguirse fuente y adaptación: ensemble Inception frente a un miembro, cabezas propuestas lineales frente a Softplus y fases del autoencoder. Toda reconstrucción bibliográfica actual se fecha como tal, sin hacerla pasar por decisión previa al entrenamiento observado.

## Reutilizar lo que ya existe

- **EDA/A.1:** [§2](02_eda.md) ya contiene la tabla y la enmienda F2a. Consolidar el mapa evidencia–decisión no exige repetir STL/ADF/KPSS sin motivo. Los residuos P0 no se atribuyen a modelos nuevos.
- **Paper guía:** [§5](05_paper_guia.md) conserva réplica TimeMixer/ETTh1 y piloto TimeMixer/iTransformer/PatchTST con recibos. Puede redactarse el cierre de su alcance sin repetir ETTh1; no demuestra una limitación Bitcoin automáticamente.
- **Tracking:** la guía admite MLflow, W&B u otra herramienta (p. 14). Documentar el diario GitHub y exportar trials es más útil que migrar herramientas durante la búsqueda.
- **Contrato:** conservar D7×5, RMSE, δ heredado, embargo y once orígenes registrados. Once es decisión del proyecto; la guía exige origen móvil, no ese número literal. La propuesta de tres no es operativa.

## Preparación realizada ahora

1. Notas de vigencia y [cronología de selección (§9)](09_protocolo_registrado.md), separada del cierre inferencial pendiente.
2. Tres fuentes bibliográficas para las cuatro altas (FCN/ResNet comparten fuente). InceptionTime y LSTM-SAE tienen DOI editorial verificado; Wang se identifica como preprint oficial con DOI arXiv, dejando separada la verificación editorial pendiente. No se declara revisión sistemática cerrada.
3. [Apéndice B: 21 ítems auditados](../_static/guide_audit_20261111_181b9e5/appendix_B.csv), con estado/evidencia/acción. Preparar el formulario no aprueba sus casillas.
4. [A.2 histórica: 18 filas del benchmark](../_static/guide_audit_20261111_181b9e5/benchmark_A2_template.csv), [A.3: tabla de contrastes](../_static/guide_audit_20261111_181b9e5/contrasts_A3_template.csv) y [matriz de trabajo de tres RQ](../_static/guide_audit_20261111_181b9e5/RQ_working_matrix.csv). Las métricas finales, p-valores y R quedan vacíos. Son plantillas, no resultados ni preregistro nuevo.

## Orden de trabajo

**Sin nuevos ajustes:** completar fichas/bibliografía/puntajes; conciliar consultas/PRISMA; consolidar EDA/contrato, entorno/licencias/tracking y el alcance de la réplica. Las notas de vigencia y plantillas ya quedan preparadas en esta revisión.

**Código CPU/fixtures en paralelo:** persistencia de curvas, esquemas y adaptadores del catálogo actual; auditoría de soportes/firmas; transformadores de robustez y contratos de calibración/casos. Preparar y probar software no equivale a ejecutar ni aprobar el análisis científico.

**Antes de futuras fases afectadas:** fechar comparadores/familias, δg, unidades/bloques, exclusiones, potencia/R y calibración, declarando resultados ya conocidos. El piloto de potencia requiere al menos cinco semillas en dos o tres modelos representativos (p. 43); no se fijará R por el saldo ni retroactivamente.

**Con evidencia final:** benchmark/IC, inferencia/Pareto, robustez, incertidumbre Top 3, casos, tres ablaciones del mejor modelo y reproducción independiente. El ganador podría no ser el híbrido: sus controles no bastan si gana otro. Las 608 fechas conocidas conservan alcance exploratorio. El resumen final de ≤250 palabras y las conclusiones necesitan esas cifras.

Dejar método, fuentes, contratos y plantillas listos ahora reduce retrabajo; no elimina las actualizaciones que genuinamente dependen de resultados.

## Evidencia y plan verificable

- [19 tareas con prioridad, entregable y criterio de cierre](../_static/guide_audit_20261111_181b9e5/workplan.csv) · [Auditoría JSON](../_static/guide_audit_20261111_181b9e5/audit.json) · [Apéndice B JSON](../_static/guide_audit_20261111_181b9e5/appendix_B.json).
- Detalle del commit auditado, anterior a estas correcciones: [método/datos](../_static/guide_audit_20261111_181b9e5/findings_method.txt), [literatura/fichas](../_static/guide_audit_20261111_181b9e5/findings_literature.txt), [análisis](../_static/guide_audit_20261111_181b9e5/findings_analysis.txt) y [reproducción](../_static/guide_audit_20261111_181b9e5/findings_reproduction.txt).
- [Verificación bibliográfica y límites](../_static/guide_audit_20261111_181b9e5/bibliography_verification.json): revisión puntual de metadatos, no nuevas búsquedas sistemáticas.

Los recibos y descargas históricos se conservan. No se ejecutaron modelos, fijó R, modificaron arquitecturas ni amplió el presupuesto.
