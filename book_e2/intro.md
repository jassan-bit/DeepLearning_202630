# Proyecto Final de Deep Learning: Entregable 2

**Deep Learning · 2026-30 · Jassan Arteta**

**Revisión del 08/11/2026:** la [propuesta de diez modelos y un híbrido candidato](sections/15_revision_modelos.md) incorpora cuatro sustituciones solicitadas y distingue las referencias actuales de E1 de las anclas históricas de E2. Es una propuesta editorial: todavía no modifica el catálogo registrado, no acredita entrenamientos nuevos ni superioridad del modelo 11.

**Ampliación bibliográfica del 08/11/2026:** la [sección 3](sections/03_literatura.md) incorpora una ronda nueva de WoS, Scopus e IEEE (**1.634 ocurrencias → 1.078 grupos**) y cuatro consultas arXiv (**1.199 apariciones → 1.004 IDs**). Los universos se mantienen separados hasta conciliarlos. Hay triaje provisional, tres lecturas focales y fuentes para las nuevas propuestas; no se declara elegibilidad global cerrada ni se modifican resultados experimentales.

**Estado experimental documentado al 6 de noviembre de 2026:** la guía completa de **65 páginas** está recibida y auditada. El libro incorpora las decisiones metodológicas de sus secciones 1–14. La ruta queda clasificada como **F2a: series de tiempo**, conservando las entradas D7×5 OHLCV, el objetivo σ₇, la RMSE, δ y la partición heredados de E1. Se distinguen veinte configuraciones y quince recetas ajustables con presupuesto común. El protocolo P0 se publicó [antes de los ajustes](https://github.com/jassan-bit/DeepLearning_202630/commit/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0).

El [piloto Bitcoin y su evaluación temporal OOF](sections/09_protocolo_registrado.md) se ejecutaron con cuatro recetas fijas: **220 trayectorias y 440 ajustes**. La nueva implementación incorpora los diez adaptadores finales, HPO temporal, dimensionamiento, calibración, análisis estadístico, robustez y ablaciones. La revisión arquitectónica y la prueba de memorización sintética de los diez modelos están disponibles; **el último recibo publicado acredita un HPO iniciado e incompleto al 06/11/2026 19:00 UTC; P1 y los resultados finales Bitcoin no tienen cierre publicado**. La nueva [enmienda de implementación](./_static/completion_protocol/analysis_additions.json) se [registró](./_static/completion_preflight/registration.json) en [`dbb193a`](https://github.com/jassan-bit/DeepLearning_202630/commit/dbb193a72897c80cbd34b452ba3cfe9d82ce6ce0) antes de sus ajustes con datos de mercado. El registro P0 no se reutiliza para retrofechar esa enmienda. La réplica TimeMixer/ETTh1 conserva su alcance externo.

Este Jupyter Book tiene configuración, índice y capítulos propios dentro del repositorio común [DeepLearning_202630](https://github.com/jassan-bit/DeepLearning_202630). E1 y E3 conservan sus direcciones independientes.

El [recibo de avance](./_static/completion_preflight/execution_progress.json) conserva una instantánea fechada de ajustes reales cerrados; no es telemetría en vivo ni una tabla de resultados finales.

## Objetivo y criterio de evaluación

El entregable amplía la revisión de E1 hacia diez mecanismos de Deep Learning, desarrolla un EDA orientado a decisiones e implementa una comparación reproducible y justa, con prevención de fuga. Las conclusiones deben apoyarse en inferencia temporal, relevancia práctica, coste, robustez, calibración, incertidumbre e interpretabilidad. Un resultado significativo entre semillas no equivale a evidencia en nuevos mercados.

La guía completa identifica requisitos que todavía necesitan experimentos o fuentes externas. **Recibir la guía no significa cumplirlos todos.** La [matriz del protocolo vigente](./_static/current_protocol.json) registra qué decisiones están cerradas, qué evidencia existe y qué falta.

## Cómo leer el libro

| Secciones | Contenido y evidencia |
|---|---|
| [1. Problema](sections/01_punto_partida.md) y [2. EDA](sections/02_eda.md) | Contrato E1, RQ, Data Card, análisis ejecutados y decisiones derivadas |
| [3. Literatura](sections/03_literatura.md) | Corte histórico de 7.833 grupos y nueva ronda nativa/arXiv con denominadores separados, triaje provisional y elegibilidad aún abierta |
| [4. Implementación](sections/04_implementacion.md) | Catálogo de veinte configuraciones, requisitos experimentales y comparación justa |
| [5. Paper guía](sections/05_paper_guia.md) | Réplica TimeMixer y piloto externo ETTh1, discrepancias y límites de transferencia |
| [6. Benchmark final](sections/06_benchmark_final.md) | Tablas e inferencia requeridas; resultados finales pendientes |
| [7. Ruta temporal](sections/07_ruta_temporal.md) | Decisión F2a y referencias endógenas persistencia, ingenua semanal y ARIMA |
| [8. Reproducibilidad](sections/08_reproducibilidad.md) y [10. Auditoría](sections/10_auditoria_datos.md) | Entornos, registro y verificación real de soportes/particiones de desarrollo |
| [9. Preregistro y potencia](sections/09_protocolo_registrado.md) | Familias, alternativas, margen de sobreajuste, P0/OOF y reglas de dimensionamiento |
| [11. HPO](sections/11_hpo_equidad.md) y [14. Arquitecturas](sections/14_arquitecturas.md) | Ejes mínimos, presupuesto comparable y aplicación condicional de NAS por modelo |
| [12. Incertidumbre y errores](sections/12_incertidumbre_errores.md) | Calibración disjunta, panel de 305 fechas y auditoría de cien casos únicos por modelo |
| [13. Agregación](sections/13_agregacion_estadistica.md) | Separación de variación entre semillas y dependencia temporal; contrastes e interpretación |

Las páginas de trabajo anteriores se conservan como archivo identificado y enlazan a estas secciones vigentes. No constituyen un protocolo alternativo.

## Matriz de cierre del diseño experimental registrado

| Requisito | Evidencia existente | Para considerarlo terminado |
|---|---|---|
| Ruta metodológica | F2a; persistencia, naive semanal y ARIMA definidos causalmente | Resultados de las veinte configuraciones y auditoría final |
| Diez arquitecturas | [Revisión por modelo](./_static/completion_protocol/architecture_review.json), fuentes fijadas y [sanidad sintética](./_static/completion_preflight/tiny_batch_sanity.json) | Registro cerrado; completar ejecución del pipeline real, sin confundir pruebas con métricas Bitcoin |
| HPO comparable | [180 candidatos](./_static/completion_protocol/candidate_registry.json), quince asignaciones T12 | Completar búsquedas, consumo, fallos y recetas congeladas |
| Preregistro y potencia | [Enmienda](./_static/completion_protocol/analysis_additions.json): P1 de 53 fechas, alternativas y regla de R | Registro previo cerrado; completar P1, simulación y decisión con incertidumbre; diez semillas no acreditan potencia |
| Incertidumbre Top3 | Rama A predefinida para **los veinte modelos**, con 53 fechas de calibración y miembros fijos | Cerrar pesos/calibradores y medir cobertura posterior del mismo predictor |
| Cien errores por modelo | Análisis P0 ya ejecutado; procedimiento final sobre 305 fechas implementado | OOF final, trazabilidad y cien casos únicos auditables por modelo |
| Inferencia y mecanismos | Familias, diagnósticos, coste, robustez, XAI y ablación implementados | Resultados y límites, incluidos desenlaces negativos o no estimables |
| Literatura | Histórico de diez originales/215 páginas; ronda nueva WoS/Scopus/IEEE y arXiv, tres lecturas focales y derivados verificables | Conciliar rondas, resolver cobertura histórica, cribar/adjudicar candidatos y verificar metadatos; no hay cierre global |

La [descarga de la implementación](./_static/completion_download/completion_bundle.zip) permite revisar código, configuración y evidencias. **Implementación, registro y resultado son estados distintos**; esta matriz no certifica la terminación experimental del proyecto.

## Pendientes que requieren evidencia

- **Literatura:** los 374 registros no recibidos corresponden a la búsqueda Scopus histórica de 7.874 resultados. La nueva ronda del 08/11 no se considera su cierre sin conciliación de búsquedas e identificadores. Faltan la conciliación entre rondas, la revisión de exclusiones propuestas, el cribado arXiv pendiente, la adjudicación final de textos completos y los cuartiles por año.
- **Experimentos:** pasaron 235 pruebas integradas y el preflight de veinte configuraciones. La enmienda se registró antes del HPO; el recibo fechado del 06/11 acredita su inicio, no su actividad actual. Faltan cerrar las búsquedas, P1, dimensionamiento y evaluación final. La calibración secundaria se prepara para los veinte modelos; después se podrán informar los tres mejores con su rama válida. No se hereda R=465 de E1 ni se declara suficiente usar diez semillas.
- **Gap transferible:** el diagnóstico registrado en Bitcoin debe informar su resultado aunque sea negativo o inconcluso. Una limitación observada en ETTh1 no demuestra una brecha predictiva corregible en Bitcoin.

## Relación con E1 y E3

Se preservan los tres modelos base, la métrica primaria, δ, la unidad y la partición de E1. Las variantes GRU-E2/LSTM-E2/TCN-E2 recibirán ajuste comparable y se distinguirán de las tres recetas históricas. Las referencias endógenas declaran que utilizan un soporte de historia diferente del D7×5 de las redes.

El test heredado ya es conocido. Registrar ahora hipótesis mejora el procedimiento de las fases posteriores, pero **no recupera su cegamiento**: la interpretación de E2 sobre esa reserva sigue siendo exploratoria. E3 deberá congelar únicamente resultados efectivamente producidos y una motivación respaldada por ellos.

Descargas: [protocolo, código y evidencias actuales](./_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip), [guía recibida](./_static/guia_e2_recibida.json), [contrato E1](./_static/e1_reference.json) y [réplica ETTh1](./_static/paper_guide_download/Paper_guia_E2_replica_piloto.zip).

[Todos los entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html) · [E1](https://jassan-bit.github.io/DeepLearning_202630/intro.html) · [E3](https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html).
