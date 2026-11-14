# 9. Registro de selección, cierre pendiente y trazabilidad

**Vigencia al 11/11/2026:** hay registro prospectivo de selección neuronal. El cierre de evaluación/inferencia final todavía está pendiente; los párrafos del 08/11 conservados más abajo son antecedentes.

| Fecha real | Evidencia | Alcance |
|---|---|---|
| 10/11/2026, 02:24:57 UTC | [Registro CUDA de selección](https://github.com/jassan-bit/DeepLearning_202630/blob/181b9e58032c0011e46dcbe334400568fb82c258/experiments/e2_cuda_search_20261109/registration.json) | 14 redes, once orígenes y dominios HPO/NAS, registrado antes de sus ajustes CUDA; no incluye comparadores/reajustes finales ni R |
| 11/11/2026 | [Optimización y corrección JSON](https://github.com/jassan-bit/DeepLearning_202630/blob/8e3ab6cfb1b3fc3c804f0c914fa5d92b661058d9/experiments/e2_implementation_optimization_11_origins_20261111/registration_json_metadata_fix.json) | Preserva dominio y saldo; corrige representación del perfil y registra implementación |
| Corte 11/11/2026, 13:31:53 Bogotá | {ref}`Corte histórico conservado en §41 <e2-cuda-cut-20261111-133153>` | Evidencia de selección parcial; no benchmark completo |
| Pendiente | Enmienda de potencia/R, comparadores, soportes y familias finales | Fechar antes de futuras etapas afectadas y declarar resultados ya conocidos; no será preregistro retroactivo de la selección |

Se conserva RMSE, δ heredado, D7×5 y calendario original. Esta aclaración no modifica registros, semillas ni presupuesto. [Auditoría y trabajo paralelo (§42)](42_auditoria_y_trabajo_paralelo.md).

## Antecedentes de diseño al 08/11/2026


**Actualización de planificación:** [§19: auditoría real de desarrollo y plan propuesto](19_plan_auditoria.md). Sin entrenamiento de mercado, HPO ni preregistro nuevo.

**Estado al 08/11/2026: el catálogo nuevo todavía no tiene preregistro experimental propio.** La selección vigente comprende FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer; CNN-1D + LSTM + atención temporal es el candidato adicional 11. Los registros anteriores no autorizan a presentar estas recetas como congeladas ni estos modelos como entrenados.

Esta sección sustituye las reglas prospectivas del catálogo retirado por las decisiones que deben cerrarse para el actual. Los protocolos y recibos históricos se mantienen intactos y se enlazan al final. **No se fijan ni se simulan nuevos presupuestos, candidatos HPO, búsquedas NAS o repeticiones R mediante esta revisión documental.**

## 9.1 Antecedente del 08/11: decisiones que entonces faltaban registrar

**Tabla histórica:** el estado de selección de esta tabla fue sustituido por el registro CUDA y las evidencias enlazadas arriba. Los cierres finales de potencia/R, comparadores e inferencia continúan pendientes en su propio ámbito.

| Decisión | Punto de partida vigente | Cierre necesario antes de entrenamiento de mercado |
|---|---|---|
| Catálogo | Diez modelos neurales y CNN-1D + LSTM + atención temporal candidato | Identidad de recetas, versiones y controles del híbrido |
| Tarea | Bitcoin diario, horizonte uno, σ₇ poblacional de siete retornos | Definición ejecutable de orígenes, etiquetas, disponibilidad y escala |
| Información neural | Siete barras OHLCV, D7×5 | Transformaciones por fold, valores faltantes y verificación de soportes |
| Información del híbrido | Mismo D7×5 relativo que E1, sin canales del objetivo | Causalidad convolucional, orden recurrente y máscara de atención |
| Arquitecturas | Cuatro prototipos compactos; seis implementaciones históricas reutilizables | Fuente, receta adaptada, revisión individual y condición NAS |
| Autoencoder | Reconstrucción y predictor separados | Secuencia de entrenamiento, pérdidas, pesos, congelado y coste de preentrenamiento |
| Híbrido | CNN-1D + LSTM + atención temporal, implementación provisional y smoke sintético | Receta definitiva, entrenamiento de mercado y comparación pendientes |
| Partición | Desarrollo y evaluación históricos ya conocidos | Calendario del nuevo estudio, cortes, embargo y exposición previa declarada |
| Selección | Debe ser temporal y usar sólo el pasado autorizado | Folds internos/externos, elección de épocas, reajustes y separación del test |
| Presupuesto | Comparabilidad y contabilidad completa | Límites por receta, espacios, recursos, incidencias y reglas de parada |
| Métrica | Media de RMSE por trayectoria/semilla en escala original | Cohorte común, métricas secundarias, coberturas y denominadores |
| Hipótesis | No existe superioridad demostrada del candidato 11 | Referencias, efectos prácticos, familias, multiplicidad y criterios de decisión |
| Repeticiones | R no determinado para este catálogo | Piloto posterior a selección, incertidumbre temporal y entre semillas, regla de dimensionamiento |
| Incertidumbre | El score y la predicción son objetos distintos | Remuestreo, supuestos y, si procede, calibración separada |
| Coste | Medición común y por etapas | Hardware, precisión, latencia, memoria y operaciones contabilizadas |
| Publicación | Código, metadatos y resultados agregados trazables | Hashes, recibos, límites de divulgación y conservación de fallos |

El margen histórico δ=0,00025772842664981106 puede documentarse como antecedente. Su uso, interpretación y comparaciones en el nuevo catálogo deberán quedar explícitos en el preregistro; no se ajusta después para favorecer resultados. No se trasladan automáticamente familias de hipótesis, umbrales, semillas o potencia de otra selección.

## 9.2 Calendario, disponibilidad y exposición previa

Los datos de desarrollo de 2023–2024 y las evaluaciones históricas posteriores ya fueron utilizados o examinados en etapas anteriores. En particular, la reserva del **01/09–30/10/2026** es conocida. Mantenerla como evaluación exploratoria no la convierte en una muestra ciega nueva.

El nuevo manifiesto debe registrar, para cada fold, los orígenes autorizados, la última observación disponible, el soporte completo de la etiqueta y la fecha de cada transformación. La referencia E1 exige etiquetas de entrenamiento hasta o−8 para cada origen o y escaladores ajustados sólo al entrenamiento. El antiguo corrector de residuos ya no es el candidato activo; sus guardas no sustituyen las pruebas específicas del nuevo híbrido.

No se comprimen huecos ni se incorporan barras futuras para completar ventanas. El híbrido conserva la ventana D7; las cohortes comunes se verifican contra el contrato E1 de [§17](17_contrato_sincronizado.md). La selección de épocas, los escaladores y el preentrenamiento del autoencoder deben respetar el pasado permitido de cada fold, incluso cuando no utilicen etiquetas.

## 9.3 Orden de las decisiones y ejecuciones

1. **Cerrar diseño:** fuentes y recetas, familias exigidas por la guía, soportes del híbrido, controles y condición NAS. Una familia arquitectónica no equivale a una estructura definida.
2. **Completar e inspeccionar implementación:** ejecutor separado, entorno fijado, convolución causal, recurrencia unidireccional, atención al pasado observado y entrenamiento train-only y pruebas integrales. Los diez tests sintéticos existentes corresponden al prototipo anterior, no al híbrido nuevo.
3. **Registrar antes de ajustar:** versión ejecutable y hashes, calendario y datos ya conocidos, espacios y presupuesto, hipótesis, fallos, selección, dimensionamiento y análisis. Un plan editorial no reemplaza este paso.
4. **Ejecutar selección:** HPO y, sólo si está justificado y registrado, búsqueda estructural; guardar todos los intentos y su coste. Congelar recetas sin mirar la evaluación.
5. **Dimensionar:** usar el piloto y la regla nuevos para distinguir variación entre entrenamientos y dependencia temporal. No adoptar un R anterior ni certificar a modelos no evaluados con un representante sin advertirlo.
6. **Cerrar predictores:** fijar todos los pesos y, si corresponde, calibradores antes del scoring previsto. Ningún reajuste posterior se presenta como el predictor que fue calibrado.
7. **Evaluar y reportar:** comparación sobre cohortes definidas, efectos e incertidumbre, costes, fallos, desviaciones y conclusiones acotadas. Un resultado desfavorable sigue siendo un resultado válido.

Cualquier cambio material posterior exige una enmienda fechada que explique su motivo y qué resultados se conocían. No se reescribe un JSON histórico para aparentar que una decisión existía antes.

## 9.4 Potencia y repeticiones: pendiente de diseño nuevo

El piloto histórico no estima automáticamente la variabilidad de estos once candidatos, ni de sus futuras recetas seleccionadas. El nuevo dimensionamiento debe partir de alternativas prácticas explícitas, un comparador, las familias completas y las dos fuentes de incertidumbre: entrenamiento y calendario.

Se deberán declarar el método de simulación o cálculo, sus supuestos, la incertidumbre de la variabilidad estimada, el máximo de recursos y la regla ante una meta inalcanzable. Aumentar semillas no crea fechas, activos ni regímenes nuevos. Un número pequeño elegido por conveniencia tampoco prueba suficiencia; no se publica aquí un R nuevo sin ese trabajo.

La resolución Monte Carlo no reemplaza la información temporal. Los límites de un piloto breve y la posible transferencia imperfecta de su dispersión se informarán aunque una simulación alcance una meta nominal.

## 9.5 Qué acreditan los registros históricos

| Evidencia preservada | Qué acredita | Qué no acredita |
|---|---|---|
| Registro P0, 06/11/2026 a las 14:52:32 UTC, commit `599a29f` | Decisiones históricas anteriores al inicio de aquel piloto | Preregistro de las arquitecturas, soportes o pérdidas actuales |
| Recibo P0/OOF cerrado | **220 trayectorias y 440 ajustes** de recetas históricas de desarrollo | Entrenamientos del nuevo catálogo, HPO nuevo o potencia del candidato 11 |
| Registro de enmienda, 06/11/2026 a las 18:59:10 UTC, commit `dbb193a` | Fuente y decisiones de una implementación anterior | Congelación del nuevo diseño |
| Recibo HPO, 06/11/2026 a las 19:00 UTC | Doce trayectorias y veinticuatro ajustes cerrados en un inicio parcial histórico | Actividad actual, cierre posterior o ejecución de los once candidatos |
| Diez pruebas sintéticas del 08/11/2026 | Comportamiento limitado de prototipos y guardas con datos artificiales | Entrenamiento Bitcoin, pipeline completo, superioridad o potencia |

Los resultados P0 emplearon recetas fijas y calendarios de desarrollo; sus análisis de régimen y proyecciones de potencia conservan su alcance original. No se incorporan como resultados ni compromisos futuros del benchmark vigente. La réplica externa de [§5](05_paper_guia.md) tampoco añade entrenamientos Bitcoin.

## 9.6 Fuentes originales para auditoría

- [Registro público P0](https://github.com/jassan-bit/DeepLearning_202630/commit/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0), [preregistro original](../_static/remediation_protocol/preregistration.json) y [recibo de ejecución](../_static/remediation_preflight/execution_summary.json).
- [Métricas históricas agregadas](../_static/remediation_pilot_results/aggregate_metrics.csv), [resultados por trayectoria](../_static/remediation_pilot_results/seed_metrics.csv), [análisis de régimen](../_static/remediation_pilot_results/gap.csv) y [proyección histórica de potencia](../_static/remediation_pilot_results/power.json).
- [Resumen histórico](../_static/remediation_pilot_results/summary.json) y [procedencia por hashes](../_static/remediation_pilot_results/provenance.json).
- [Enmienda histórica publicada](https://github.com/jassan-bit/DeepLearning_202630/commit/dbb193a72897c80cbd34b452ba3cfe9d82ce6ce0), [recibo de registro](../_static/completion_preflight/registration.json), [incidencia de arranque](../_static/completion_preflight/execution_incident.json) y [avance HPO observado](../_static/completion_preflight/execution_progress.json).
- [Paquete histórico P0](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip) y [paquete histórico de implementación](../_static/completion_download/completion_bundle.zip).

Estos archivos se conservan sin modificación. Una validación documental o un hash comprueba identidades y reglas; sólo los recibos de la ejecución correspondiente acreditan ajustes y resultados. El catálogo vigente necesita los suyos.
