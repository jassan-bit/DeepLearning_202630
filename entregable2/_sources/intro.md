# Proyecto Final de Deep Learning: Entregable 2

**Decisión vigente: conservar los once orígenes originales y todas las exigencias de E2.** Sólo se optimizan implementación, orden de ejecución y reutilización compatible. La reducción a tres orígenes queda como propuesta no operativa hasta justificar su suficiencia. [Calendario y ejecutor vigentes (§40)](sections/40_calendario_original.md).

**Deep Learning · 2026-30 · Jassan Arteta**

**Avance experimental verificado: 881 ajustes HPO nuevos completados, sin fallos, al 11/11/2026 a las 19:32:18 de Bogotá.** Se suman a 1.802 registros anteriores: **2.683 registros de resultados** entre ambos historiales. Son corridas de selección, no modelos finales ni una entrega terminada. [Detalle y fuentes del corte (§41)](sections/41_avance_optimizacion_11_origenes.md).

**Auditoría contra la guía:** [§42: qué podemos completar mientras entrena Colab](sections/42_auditoria_y_trabajo_paralelo.md). Incluye prioridades, 21 ítems del Apéndice B y plantillas de reporte; distingue preparación de cumplimiento.

## Estado de la entrega y guía de lectura

**Actualización documental del 11/11/2026, cotejada con la guía de 65 páginas aportada por el usuario.** El libro contiene código, revisión bibliográfica, preparación experimental y resultados parciales de selección. **El benchmark final no está terminado y no se declara cumplimiento integral de la guía.**

La [matriz vigente de exigencias, evidencias y pendientes](sections/38_cumplimiento_guia.md) organiza la entrega según la guía y distingue requisitos del PDF, decisiones propias y resultados observados. Cada requisito se cierra con evidencia verificable; describirlo o disponer de código no acredita su ejecución.

| Componente | Estado documental verificable | Dónde revisar |
|---|---|---|
| Problema, datos, EDA y ruta temporal | Contrato definido; auditoría completa del benchmark pendiente | [Problema](sections/01_punto_partida.md), [EDA](sections/02_eda.md), [F2a](sections/07_ruta_temporal.md), [contrato](sections/17_contrato_sincronizado.md) |
| Revisión sistemática y Top 10 | Búsquedas y lecturas parciales; cierre de elegibilidad y puntajes por completar | [Literatura](sections/03_literatura.md), [catálogo](sections/15_revision_modelos.md) |
| Implementación y HPO/NAS | 881 ajustes HPO nuevos completados en el corte vigente; selección completa pendiente | [Recetas](sections/26_registro_recetas.md), [fábricas](sections/28_fabricas_nas.md), [avance vigente](sections/41_avance_optimizacion_11_origenes.md), [corte histórico](sections/36_avance_campana_cuda.md) |
| Réplica del paper guía | Evidencia externa de TimeMixer/ETTh1; no sustituye Bitcoin D7 | [Réplica y limitaciones](sections/05_paper_guia.md) |
| Benchmark e inferencia | Pendientes de recetas seleccionadas, R, reajustes y evaluación común | [Benchmark](sections/06_benchmark_final.md), [inferencia](sections/13_agregacion_estadistica.md) |
| Robustez, incertidumbre e interpretabilidad | Preparación parcial; resultados del catálogo completo pendientes | [Incertidumbre y errores](sections/12_incertidumbre_errores.md), [matriz de cierre](sections/38_cumplimiento_guia.md) |
| Reproducibilidad y entrega | Fuentes y procedimientos disponibles; reproducción integral pendiente | [Reproducibilidad](sections/08_reproducibilidad.md), [lista de entrega](sections/38_cumplimiento_guia.md) |

## Problema, contrato y referencias

Se estudia **OHLCV diario D7×5 → σ₇**, desviación estándar poblacional de siete retornos logarítmicos diarios de Close, **sin anualización y con horizonte de un día**. Se conserva RMSE como métrica primaria heredada de E1 y **δ=0,00025772842664981106**; las recomendaciones de métricas temporales no cambian por silencio ese contrato. Las métricas auxiliares y sus denominadores deben declararse antes de evaluar.

La ruta es **F2a, series de tiempo**: partición cronológica, embargo y transformaciones ajustadas sólo con el pasado. Las **608 fechas históricas conocidas** mantienen carácter exploratorio; una actualización editorial o un nuevo programador no recuperan independencia del test. [Contrato y límites](sections/17_contrato_sincronizado.md).

El catálogo comprende diez modelos base:

1. FCN-1D.
2. ResNet-1D.
3. Inception-1D.
4. LSTM-autoencoder con predictor.
5. TimesNet.
6. TSMixer.
7. TiDE.
8. TimeMixer.
9. PatchTST.
10. iTransformer.

**CNN-1D + LSTM + atención temporal es el candidato adicional 11**, no un ganador demostrado. Los controles **MLP16, LSTM32 y GRU32 de E1** son referencias adicionales, con pesos reinicializados. Persistencia, naive estacional, ARIMA y Ridge completan los cuatro comparadores del diseño de dieciocho sistemas; no se cuentan dentro del Top 10. La taxonomía funcional del Top 10 distingue convolucionales, recurrentes/autoencoder, mezcladores MLP y atención; [§38 explicita la asignación](sections/38_cumplimiento_guia.md). El candidato adicional no se utiliza para completar ese recuento. [Catálogo y adaptaciones](sections/15_revision_modelos.md).

## Diseño vigente, guía y cómputo

El calendario vigente conserva diez selecciones mensuales enero–octubre de 2024 y la selección de enero de 2025: **once orígenes comunes**, con máximo de **85.074 intentos neuronales de selección**. Es una decisión del diseño original, no una cifra literal del PDF. La [propuesta de tres orígenes de §37](sections/37_ajuste_guia.md) queda pendiente de justificación y no debe considerarse definitiva.

Se mantienen las seis variantes NAS y los cinco cuerpos publicados con exención justificada, las 18 configuraciones HPO, k=40 y máximos de 50/200 épocas. La optimización funcional se activa por caso sólo tras medirla en la GPU real. [Alcance y límites (§40)](sections/40_calendario_original.md).

El notebook vigente de once orígenes se enlaza en §40. No cambia por sí solo el Colab activo: hay que esperar el cierre del controlador anterior. Conserva los resultados compatibles y el historial del límite inicial de 48 horas. La ampliación manual a **56 horas acumuladas**, autorizada el 11/11, está registrada antes de reanudar; no hay ampliación automática ni garantía de terminar todas las corridas. [Presupuesto y recibos actuales](sections/41_avance_optimizacion_11_origenes.md).

## Resultados parciales y límites

El [corte vigente de §41](sections/41_avance_optimizacion_11_origenes.md) se fija al commit `38211c3ca8289539ac8697a1b8d406cb0d7480b7`. La primera ola de ocho modelos/referencias con arquitectura fija tiene cobertura HPO cerrada en enero, febrero y marzo de 2024. En abril, FCN-1D, ResNet-1D e Inception-1D completaron 36 ajustes cada uno; TSMixer registra 17 de 36. Los 881 resultados nuevos tienen estado `completed`; el total combinado de 2.683 incluye 1.802 registros históricos cuyos estados no se reauditán aquí. El último evento publicado inicia otra corrida, sin resultado en ese snapshot; no es una comprobación en vivo de Colab. Se añade un CSV de ensayos y configuraciones verificado por huellas; no contiene un ranking final.


El [corte documentado en §36](sections/36_avance_campana_cuda.md), del **11/11/2026 a las 06:55:42 de Bogotá**, contiene **1.748 archivos de resultados**. Registra selección del primer origen para FCN-1D, ResNet-1D, Inception-1D y LSTM-autoencoder y avance NAS-1 de TimesNet. Es un corte histórico, no el contador actual ni evidencia de activación del calendario reducido. Los resultados de E1, los perfiles de recursos y la réplica ETTh1 conservan su propio ámbito.

**No hay ranking final, superioridad demostrada del híbrido ni R definitivo.** La [matriz de cumplimiento](sections/38_cumplimiento_guia.md) indica las evidencias necesarias antes de publicar conclusiones.

## Recorrido recomendado

- **Problema y literatura:** [§1](sections/01_punto_partida.md), [§2](sections/02_eda.md), [§3](sections/03_literatura.md) y [§5: paper guía](sections/05_paper_guia.md).
- **Método y modelos:** [§7: ruta F2a](sections/07_ruta_temporal.md), [§10: auditoría](sections/10_auditoria_datos.md), [§11: equidad](sections/11_hpo_equidad.md), [§14: arquitecturas](sections/14_arquitecturas.md) y [§26: recetas](sections/26_registro_recetas.md).
- **Ejecución y evidencia:** [§29: entrenamiento](sections/29_entrenamiento_temporal.md), [§35: CUDA](sections/35_busqueda_cuda_reanudable.md), [§36: corte histórico](sections/36_avance_campana_cuda.md), [§40: calendario vigente](sections/40_calendario_original.md) y [§41: ejecución y avance](sections/41_avance_optimizacion_11_origenes.md).
- **Entrega y cierre:** [§6: benchmark](sections/06_benchmark_final.md), [§12: incertidumbre](sections/12_incertidumbre_errores.md), [§13: inferencia](sections/13_agregacion_estadistica.md), [§8: reproducción](sections/08_reproducibilidad.md) y [§38: exigencias de la guía](sections/38_cumplimiento_guia.md).

Los capítulos con cortes anteriores conservan sus fechas y recibos. Las notas de vigencia distinguen la preparación histórica del diseño actual; no se trasladan resultados entre protocolos ni se rellenan tablas con cifras de otros experimentos.
