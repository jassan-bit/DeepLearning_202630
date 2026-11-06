# Proyecto Final de Deep Learning: Entregable 2

**Deep Learning · 2026-30 · Jassan Arteta**

**Estado al 6 de noviembre de 2026:** la guía completa de **65 páginas** está recibida y auditada. El libro incorpora las decisiones metodológicas de sus secciones 1–14. La ruta queda clasificada como **F2a: series de tiempo**, conservando las entradas D7×5 OHLCV, el objetivo σ₇, la RMSE, δ y la partición heredados de E1. Se distinguen veinte configuraciones y quince recetas ajustables con presupuesto común. El protocolo de esta etapa se publicó [antes de los ajustes](https://github.com/jassan-bit/DeepLearning_202630/commit/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0).

El [piloto Bitcoin y su evaluación temporal OOF](sections/09_protocolo_registrado.md) ya se ejecutaron como un experimento de desarrollo con cuatro recetas fijas: **220 trayectorias y 440 ajustes**, con sus resultados y límites publicados allí. **El benchmark final de veinte configuraciones, el HPO y la incertidumbre de sus tres mejores modelos siguen pendientes.** El registro del procedimiento no acredita por sí solo su ejecución ni potencia suficiente. La réplica de TimeMixer y el piloto ETTh1 se conservan como evidencia externa, con otro dataset y otra tarea.

Este Jupyter Book tiene configuración, índice y capítulos propios dentro del repositorio común [DeepLearning_202630](https://github.com/jassan-bit/DeepLearning_202630). E1 y E3 conservan sus direcciones independientes.

## Objetivo y criterio de evaluación

El entregable amplía la revisión de E1 hacia diez mecanismos de Deep Learning, desarrolla un EDA orientado a decisiones e implementa una comparación reproducible y justa, con prevención de fuga. Las conclusiones deben apoyarse en inferencia temporal, relevancia práctica, coste, robustez, calibración, incertidumbre e interpretabilidad. Un resultado significativo entre semillas no equivale a evidencia en nuevos mercados.

La guía completa identifica requisitos que todavía necesitan experimentos o fuentes externas. **Recibir la guía no significa cumplirlos todos.** La [matriz del protocolo vigente](./_static/current_protocol.json) registra qué decisiones están cerradas, qué evidencia existe y qué falta.

## Cómo leer el libro

| Secciones | Contenido y evidencia |
|---|---|
| [1. Problema](sections/01_punto_partida.md) y [2. EDA](sections/02_eda.md) | Contrato E1, RQ, Data Card, análisis ejecutados y decisiones derivadas |
| [3. Literatura](sections/03_literatura.md) | Diez originales completos, 215 páginas; cribado por reglas de 7.833 grupos recibidos y elegibilidad aún abierta |
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

## Pendientes que requieren evidencia

- **Literatura:** faltan 374 registros de la búsqueda Scopus de 7.874 resultados. Los reenvíos son idénticos a los lotes anteriores. El cribado por reglas, las consultas transversales locales y las revisiones de resúmenes no sustituyen la adjudicación final de textos completos, los cuartiles verificados ni nuevas búsquedas nativas documentadas.
- **Experimentos:** la verificación fiel de las adaptaciones finales habilitará HPO, P1 y dimensionamiento posterior al HPO; después se podrán ejecutar la comparación final, robustez, ablación y calibración de los tres mejores. No se hereda automáticamente R=465 de E1 ni se declara suficiente usar diez semillas.
- **Gap transferible:** el diagnóstico registrado en Bitcoin debe informar su resultado aunque sea negativo o inconcluso. Una limitación observada en ETTh1 no demuestra una brecha predictiva corregible en Bitcoin.

## Relación con E1 y E3

Se preservan los tres modelos base, la métrica primaria, δ, la unidad y la partición de E1. Las variantes GRU-E2/LSTM-E2/TCN-E2 recibirán ajuste comparable y se distinguirán de las tres recetas históricas. Las referencias endógenas declaran que utilizan un soporte de historia diferente del D7×5 de las redes.

El test heredado ya es conocido. Registrar ahora hipótesis mejora el procedimiento de las fases posteriores, pero **no recupera su cegamiento**: la interpretación de E2 sobre esa reserva sigue siendo exploratoria. E3 deberá congelar únicamente resultados efectivamente producidos y una motivación respaldada por ellos.

Descargas: [protocolo, código y evidencias actuales](./_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip), [guía recibida](./_static/guia_e2_recibida.json), [contrato E1](./_static/e1_reference.json) y [réplica ETTh1](./_static/paper_guide_download/Paper_guia_E2_replica_piloto.zip).

[Todos los entregables](https://jassan-bit.github.io/DeepLearning_202630/entregables.html) · [E1](https://jassan-bit.github.io/DeepLearning_202630/intro.html) · [E3](https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html).
