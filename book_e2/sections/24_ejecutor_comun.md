# 24. Ejecutor común y disponibilidad temporal

## Alcance y separación de evidencias

Este corte integra un recorrido común exclusivamente sintético y una auditoría separada de disponibilidad real. **No se ajustaron modelos con datos de mercado, no se ejecutó HPO/NAS ni el experimento final.** El catálogo conserva diez modelos y el candidato CNN-LSTM-atención. No se adjudicaron exenciones NAS.

La ejecución artificial comprueba mecanismos, no pronóstico de Bitcoin ni superioridad. La auditoría real sólo lee derivados diarios de desarrollo para verificar soporte, fechas y calidad; no estima escaladores, modelos, pronósticos o métricas predictivas. Los scripts, contratos y resultados históricos permanecen intactos.

## Dieciocho sistemas, sin ampliar el catálogo

El ejecutor reúne los once candidatos de §§22–23 y siete referencias separadas:

- GRU32, LSTM32 y MLP16, con las arquitecturas E1 comprobadas y parámetros inicializados de nuevo. Se identifican como referencias reajustadas para E2; no se importan pesos ni métricas históricas de E1.
- Persistencia.
- Ridge.
- Naive estacional: un sistema con períodos candidatos 7/14/28, no tres modelos adicionales en el catálogo.
- ARIMA: un sistema con el espacio de órdenes definido en §21.

El smoke fija una sola receta por sistema: Ridge α=1, naive estacional m=7 y ARIMA(1,0,0), además de los cuerpos neurales documentados. No compara alternativas para elegir un ganador. La operación de selección es la identidad de una candidata única y **no implementa la búsqueda completa A/B ni el espacio HPO mínimo de la guía**.

## Calendario sintético que conserva la relación E1

Se generan 320 días artificiales dentro del programa, sin argumentos de archivos, arrays, pesos o etiquetas de mercado. Los índices del fixture permiten comprobar el embargo y la relación entre selección interna y refit:

| Etapa | Origen | Entrenamiento | Evaluación |
|---|---:|---|---|
| Interna | 200 | 7–192: 186 ventanas | 200–282: 83 fechas |
| Refit nuevo | 290 | 7–282: 276 ventanas | 290–297: ocho fechas |

El último objetivo del entrenamiento interno es origen exterior−98; su validación abarca origen exterior−90…−8. El refit sólo usa objetivos hasta origen exterior−8. Cada ARIMA filtra los siete días del puente con parámetros fijos. Estas fechas son artificiales: no constituyen un piloto real ni la evaluación histórica de 608 fechas.

Se preservan D7×5, ventana t−7…t−1, objetivo σ₇ poblacional y h=1. `relative_window`, medias y desviaciones de X y media positiva de y se ajustan exclusivamente al entrenamiento de cada etapa. El refit vuelve a inicializar modelos, optimizadores y escaladores; no se reutiliza el estado ajustado de la etapa interna.

Los dos preentrenamientos SAE y el ajuste supervisado se realizan separadamente en cada etapa, con X/s calculado sólo sobre su entrenamiento. El decoder no pasa al predictor. Ninguna de estas etapas se omite en la contabilidad.

## Pronosticar, guardar, observar y puntuar

La transformación de inferencia recibe únicamente características y no necesita y_t. Las 18 predicciones del día se escriben juntas en un registro JSONL nuevo; el programa hace flush y fsync antes de revelar el objetivo, actualizar el filtro ARIMA o puntuar. Se conserva la predicción cruda y su truncamiento cuando corresponde.

El registro enlaza cada evento por hash y conserva secuencia, planes, etapas de ajuste, predicciones, observaciones y puntuación. Una falla de persistencia detiene el flujo; no revela el objetivo ni borra la fecha para rescatar el resultado. La salida existente se rechaza. La cadena facilita auditoría local, pero no es un servicio remoto inviolable ni demuestra por sí sola que fsync ocurrió: ese orden se verifica además en el código y las pruebas.

Los pesos y escaladores permanecen fijos durante cada evaluación; ARIMA actualiza únicamente su estado. Las métricas se calculan en unidades originales sobre la máscara común prefijada. RMSE sigue siendo la métrica principal; los valores artificiales no deben utilizarse para ordenar los modelos como evidencia científica.

## Coste observado del smoke, no presupuesto experimental

El recorrido realiza 36 etapas de ajuste: 28 pasos supervisados de las 14 redes, cuatro fases auxiliares SAE y cuatro ajustes clásicos (Ridge y ARIMA en ambas etapas). Persistencia y naive estacional no se ajustan. Las 32 etapas neurales utilizan un paso de optimizador cada una.

No son 36 entrenamientos completos ni oportunidades HPO. La igualdad de etapas tampoco implica igualdad de cómputo. Los tiempos registrados corresponden a esta máquina y este fixture; no permiten presupuestar NAS, un piloto o el benchmark de mercado.

## Disponibilidad real: resultados y límite de acceso

La auditoría contrasta los hashes de los derivados diarios de desarrollo con el registro histórico y aplica D7, objetivo válido y todos los rezagos 1/7/14/28 antes de cualquier selección:

| Cohorte | Referencia | Disponible con soporte común |
|---|---:|---:|
| CV enero–octubre de 2024 | 305 | 305 |
| Piloto histórico E1 | 60 | 60 |
| Piloto propuesto E2 | 53 | 53 |
| Desarrollo completo: ventanas D7 y objetivo | 716 | 666 |
| Evaluación final enero de 2025–agosto de 2026 | 608 | No verificado |

Las 50 exclusiones adicionales del desarrollo completo se deben al soporte temporal de los rezagos; el recibo identifica fechas y causas. No se reducen por ello las muestras de entrenamiento neuronal ni se afirma que haya 50 errores de predicción.

El soporte de entrenamiento final conserva 709 muestras hasta 24/12/2024, con 619 internas hasta 25/09 y 83 de validación entre 03/10 y 24/12. Todos los cortes de desarrollo considerados superan las 90 etiquetas mínimas del control ARIMA y disponen de las fechas de sus puentes. **Esto no certifica convergencia ni salida finita de ARIMA.**

Para las 608 fechas finales sólo se consultó la reserva descriptiva: no se abrió el payload ni ningún ZIP protegido. No se suministró ni verificó un derivado diario final en esta auditoría. Por tanto, no se certifica esa máscara común ni se concluye que falten días. Se necesita el derivado autorizado y verificable antes de cerrar esta comprobación, sin eludir la guarda de acceso.

## Pruebas y evidencia reproducible

Se aprobaron **40 pruebas funcionales nuevas**: 29 del ejecutor y 11 de disponibilidad. La integración se ejecutó con las cuatro fuentes externas disponibles; no hubo omisiones. Las pruebas verifican embargo, índices futuros o duplicados rechazados, escalado train-only, inferencia sin objetivo, conservación de arquitecturas E1, refit fresco, conteo de etapas y persistencia antes de revelar y_t. Una falla inyectada al escribir predicciones impide revelar el objetivo y no genera un recibo exitoso. Las pruebas de disponibilidad cubren huecos, rezagos y calidad. El código también verifica hashes de las fuentes registradas; no se ejecutó una prueba de integración que alterara esos archivos.

[Resumen legible](../_static/common_executor_20261109/summary.json) · [Disponibilidad real agregada](../_static/common_executor_20261109/availability.json) · {download}`Recibo sintético completo, gzip <../_static/common_executor_20261109/synthetic_receipt.json.gz>` · {download}`Diario completo de eventos, gzip <../_static/common_executor_20261109/events.jsonl.gz>` · {download}`Reproducción y límites <../_static/common_executor_20261109/README.md>`.

El diario contiene 327 eventos y 91 pares predicción→observación. Se publican completos, comprimidos sin eliminar fases. Los hashes ligan código, contrato y evidencia. La lectura de datos de mercado de esta ejecución fue cero; se consultó únicamente el manifiesto histórico de hashes de código externo, no pesos ni resultados predictivos.

**Límites de integración:** el fixture es completo y finito. El soporte de faltantes de ARIMA no convierte este programa en un ejecutor general de huecos y calidad del mercado. Un rezago ausente provoca fallo explícito; aún debe diseñarse el avance causal de todos los sistemas sobre días no evaluables. MASE usa diferencias adyacentes del entrenamiento artificial contiguo y no debe trasladarse a índices reales comprimidos. La lista de versiones observadas es de dependencias directas, no un lock transitivo ni un contenedor reproducido independientemente.

## Pendientes antes del mercado

1. Resolver por receta la aplicación de NAS y las adaptaciones según la guía; estas decisiones siguen abiertas.
2. Fijar hardware, límites de tiempo/coste y el espacio de búsqueda completo. La receta única de este smoke no satisface por sí sola el HPO exigido.
3. Completar y validar la selección temporal real y la integración de calidad/faltantes sin comprimir fechas ni elegir máscaras por pérdidas.
4. Certificar la disponibilidad final mediante el acceso autorizado, congelar protocolo y batería de inferencia. El test ya conocido sigue siendo exploratorio.
5. Realizar una revisión independiente del entorno y del futuro ejecutor real antes de autorizar cualquier ajuste de mercado. El programa actual conserva la barrera de datos exclusivamente artificiales.
