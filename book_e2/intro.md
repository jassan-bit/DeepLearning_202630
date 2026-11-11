# Proyecto Final de Deep Learning: Entregable 2

**Deep Learning · 2026-30 · Jassan Arteta**

## Estado de la entrega y guía de lectura

**Actualización documental del 11/11/2026, cotejada con la guía de 65 páginas aportada por el usuario.** El libro contiene código, revisión bibliográfica, preparación experimental y resultados parciales de selección. **El benchmark final no está terminado y no se declara cumplimiento integral de la guía.**

La [matriz vigente de exigencias, evidencias y pendientes](sections/38_cumplimiento_guia.md) organiza la entrega según la guía y distingue requisitos del PDF, decisiones propias y resultados observados. Cada requisito se cierra con evidencia verificable; describirlo o disponer de código no acredita su ejecución.

| Componente | Estado documental verificable | Dónde revisar |
|---|---|---|
| Problema, datos, EDA y ruta temporal | Contrato definido; auditoría completa del benchmark pendiente | [Problema](sections/01_punto_partida.md), [EDA](sections/02_eda.md), [F2a](sections/07_ruta_temporal.md), [contrato](sections/17_contrato_sincronizado.md) |
| Revisión sistemática y Top 10 | Búsquedas y lecturas parciales; cierre de elegibilidad y puntajes por completar | [Literatura](sections/03_literatura.md), [catálogo](sections/15_revision_modelos.md) |
| Implementación y HPO/NAS | Fábricas y pruebas disponibles; selección CUDA parcial | [Recetas](sections/26_registro_recetas.md), [fábricas](sections/28_fabricas_nas.md), [avance fechado](sections/36_avance_campana_cuda.md) |
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

La [enmienda de §37](sections/37_ajuste_guia.md) programa **tres orígenes de selección**: `2024-01-01`, `2024-10-01` y `2025-01-01`, frente a los once del plan anterior. Conserva el contrato y el origen móvil, pero reduce la cobertura temporal. Tres orígenes son una decisión del proyecto, **no un número impuesto ni aprobado por el PDF**.

El máximo de búsqueda neuronal pasa de **85.074 a 23.202 intentos lógicos** (72,7 % menos). No es una estimación de horas ni el coste total: faltan reajustes, comparadores, robustez, ablaciones, calibración e inferencia. La búsqueda de **1.000 arquitecturas aplica sólo a diseños no definidos por su fuente**. Nuestras seis variantes NAS conservan ese espacio; los cinco cuerpos publicados mantienen su exención justificada. Las 18 configuraciones HPO, k=40 y los máximos de 50/200 épocas son elecciones registradas del proyecto. [Exigencias y decisiones](sections/38_cumplimiento_guia.md).

El [notebook de transición](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable2_calendario_guia.ipynb) conserva los resultados compatibles. **La publicación del libro no cambia el Colab activo:** no se afirma que la enmienda ya esté ejecutándose. Hay que cerrar el controlador anterior y revisar su informe antes de iniciar el nuevo; un intento sin resultado no se repite automáticamente.

## Resultados parciales y límites

El [corte documentado en §36](sections/36_avance_campana_cuda.md), del **11/11/2026 a las 06:55:42 de Bogotá**, contiene **1.748 archivos de resultados**. Registra selección del primer origen para FCN-1D, ResNet-1D, Inception-1D y LSTM-autoencoder y avance NAS-1 de TimesNet. Es un corte histórico, no el contador actual ni evidencia de activación del calendario reducido. Los resultados de E1, los perfiles de recursos y la réplica ETTh1 conservan su propio ámbito.

**No hay ranking final, superioridad demostrada del híbrido ni R definitivo.** La [matriz de cumplimiento](sections/38_cumplimiento_guia.md) indica las evidencias necesarias antes de publicar conclusiones.

## Recorrido recomendado

- **Problema y literatura:** [§1](sections/01_punto_partida.md), [§2](sections/02_eda.md), [§3](sections/03_literatura.md) y [§5: paper guía](sections/05_paper_guia.md).
- **Método y modelos:** [§7: ruta F2a](sections/07_ruta_temporal.md), [§10: auditoría](sections/10_auditoria_datos.md), [§11: equidad](sections/11_hpo_equidad.md), [§14: arquitecturas](sections/14_arquitecturas.md) y [§26: recetas](sections/26_registro_recetas.md).
- **Ejecución y evidencia:** [§29: entrenamiento](sections/29_entrenamiento_temporal.md), [§35: CUDA](sections/35_busqueda_cuda_reanudable.md), [§36: cortes de avance](sections/36_avance_campana_cuda.md) y [§37: enmienda](sections/37_ajuste_guia.md).
- **Entrega y cierre:** [§6: benchmark](sections/06_benchmark_final.md), [§12: incertidumbre](sections/12_incertidumbre_errores.md), [§13: inferencia](sections/13_agregacion_estadistica.md), [§8: reproducción](sections/08_reproducibilidad.md) y [§38: exigencias de la guía](sections/38_cumplimiento_guia.md).

Los capítulos con cortes anteriores conservan sus fechas y recibos. Las notas de vigencia distinguen la preparación histórica del diseño actual; no se trasladan resultados entre protocolos ni se rellenan tablas con cifras de otros experimentos.
