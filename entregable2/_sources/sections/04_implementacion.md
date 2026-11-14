# 4. Implementación: catálogo obligatorio y antecedentes

**Estado de ejecución al 13/11/2026:** se completaron las trece recetas neuronales, sus 78 ajustes finales y los tres comparadores del catálogo obligatorio. Las implementaciones y cuerpos publicados vigentes se identifican en [§48](48_protocolo_reducido.md), y los resultados medidos están en [§6](06_benchmark_final.md). Los prototipos y estados de integración conservados más abajo corresponden a sus fechas históricas.

**Antecedente de planificación del 12/11/2026:** ese corte fue sustituido por la ejecución documentada en [§48](48_protocolo_reducido.md). Su texto exacto se conserva en el archivo enlazado al final; no describe trabajos finales pendientes.

El híbrido adicional y Ridge se conservan como opcionales/históricos. Este capítulo archiva los prototipos e integraciones anteriores; el inventario vigente se consulta en §§43–44 y el protocolo efectivamente ejecutado en §48.

## Antecedentes de implementación, corte del 08–11/11/2026

**Integración posterior:** [§24 comprueba el recorrido sintético común de 18 sistemas](24_ejecutor_comun.md), con referencias separadas y sin ajustes de mercado.

**Avance posterior:** [§23 implementa los cuerpos de §22 y los controles temporales](23_implementacion_recetas.md), con pruebas exclusivamente sintéticas y preservación de los prototipos anteriores. No es un benchmark de mercado.

**Avance de software:** fábrica de los once candidatos, híbrido y flujo sintético implementados y verificados; recetas provisionales, sin datos de mercado. Véase [§18: integración y pruebas sintéticas](18_integracion_sintetica.md). En el corte sintético de §18 todavía no se había ejecutado la selección posterior de §41; el benchmark de aquel diseño aún no estaba terminado; el catálogo actual se cerró en §6.

**Catálogo histórico al 08/11/2026:** FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer. En ese corte, el modelo 11 propuesto era **CNN-1D + LSTM + atención temporal**, un diseño editorial propuesto, implementado provisionalmente y probado con datos sintéticos, sin entrenamiento de mercado y sin superioridad demostrada. HAR + red residual queda como prototipo histórico inactivo. Esta selección sustituye el catálogo anterior en la presentación de E2; sus registros experimentales se conservan sólo como trazabilidad.

**Estado del prototipo al 08/11/2026:** se conservan cuatro prototipos compactos y el corrector HAR anterior, aislados del ejecutor histórico, con diez pruebas sintéticas documentadas. Esas pruebas no cubren el nuevo híbrido CNN-1D + LSTM + atención temporal. Los otros seis modelos tienen código histórico que necesita integración y validación en el nuevo diseño. **Ese corte no incluía entrenamiento de mercado, HPO, dimensionamiento ni resultados finales de esos prototipos.** La selección parcial posterior permanece separada de este antecedente.

## 4.0 Implementación preliminar y alcance

El código previo se conserva en [redes propuestas](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_proposed_models.py), [guardas temporales](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_proposed_temporal.py) y [pruebas sintéticas](https://github.com/jassan-bit/DeepLearning_202630/blob/main/tests/test_e2_proposed_benchmark.py). No implementa un ejecutor completo de entrenamiento, selección y evaluación ni contiene el híbrido histórico opcional.

Las [fichas bibliográficas](16_fichas_nuevas.md) describen fuentes y adaptaciones posibles. No son la receta de los prototipos compactos ni configuraciones ya seleccionadas:

| Modelo o componente | Implementación disponible | Límite que debe resolverse |
|---|---|---|
| FCN-1D | Tres convoluciones con kernels 3/3/1, ancho predeterminado 16, ReLU y pooling promedio | No replica los anchos, kernels ni normalización de la fuente de clasificación |
| ResNet-1D | Dos bloques de convoluciones 3/3 con proyección de atajo y ancho 16 | No reproduce la profundidad de la arquitectura fuente |
| Inception-1D | Bottleneck, ramas 1/3/5 y max-pooling, una red compacta | No reproduce los módulos ni el ensemble de InceptionTime |
| LSTM-autoencoder con predictor | Encoder LSTM, decoder de latente repetido y cabeza escalar; reconstrucción y predicción separadas | Sin política integrada de preentrenamiento, congelado, selección de épocas o ajuste supervisado |
| HAR + red residual, histórico inactivo | Corrección firmada de D7×5 añadida a un pronóstico HAR externo; truncamiento a cero en inferencia | Se conserva sin activarlo ni transferir sus pruebas al nuevo candidato |
| CNN-1D + LSTM + atención temporal, candidato histórico opcional | Prototipo y pruebas sintéticas disponibles | Construir y verificar el flujo causal, la atención sobre siete estados y la cabeza de σ₇ antes de entrenar |

Las cuatro redes reciben `(batch,7,5)` y producen `(batch,)` mediante Softplus. Son decisiones del prototipo, no valores atribuidos a los artículos. El padding simétrico opera dentro de una ventana ya observada: no añade historia ni constituye por sí mismo una convolución causal intraventana.

## 4.1 Contrato de información

Los diez modelos y el híbrido activo comparten el [contrato sincronizado E1–E2–E3](17_contrato_sincronizado.md): **D7×5 derivado de OHLCV mediante `relative_window` → σ₇, con h=1**. No son niveles OHLCV crudos. σ₇ es la desviación poblacional de siete retornos logarítmicos diarios, `ddof=0`, sin anualización. La evaluación debe usar unidades originales y cualquier transformación aprendida se ajusta exclusivamente en el entrenamiento permitido de cada fold. No se añaden rezagos del objetivo ni historia adicional.

El flujo propuesto es **convoluciones 1D causales que conservan siete posiciones → LSTM unidireccional con `return_sequences=True` → atención temporal sólo sobre los siete estados observados → vector de contexto → cabeza escalar de σ₇**. La convolución utiliza padding sólo a la izquierda cuando sea necesario; no incorpora días anteriores a D7 como datos nuevos. La atención resume la ventana disponible y no consulta la etiqueta futura. Anchos, kernels, profundidad, función de atención, enlace de salida e hiperparámetros definitivos quedan pendientes de definición y registro.

Los controles propuestos son **CNN sola**, **LSTM sola** y **CNN–LSTM sin atención**, con el mismo soporte y calendario. Son ablaciones del híbrido, no nuevas posiciones del catálogo. El soporte HAR 1/7/22 y el cross-fitting residual pertenecen al prototipo anterior y dejan de ser requisitos del candidato histórico opcional.

## 4.2 Seis adaptadores integrados: controles antes de mercado

| Modelo vigente | Verificación que debe conservar el pipeline registrado |
|---|---|
| TimesNet | Frecuencias disponibles con D7, selección por origen, extensión temporal y cabeza de σ₇ |
| TSMixer | Orden de mezclas temporal/canal, normalización, independencia de acompañantes del batch y lectura escalar |
| TiDE | Encoder/decoder y conexiones residuales; ausencia de covariables futuras y escala correcta del objetivo |
| TimeMixer | Escalas no vacías con siete observaciones, descomposición y combinación de salidas |
| PatchTST | Parches y padding explícitos; normalización sólo de lo observado y proyección a σ₇ |
| iTransformer | Tokens de variables, atención, normalización y cabeza que no desnormalice OHLCV como volatilidad |

La [revisión arquitectónica histórica](../_static/completion_protocol/architecture_review.json) permite localizar decisiones anteriores. La integración sintética nueva se verifica por separado en §18; ninguna revisión congela todavía las recetas para mercado. Las verificaciones se repetirán sobre la fuente, el entorno y el pipeline que efectivamente se registren.

## 4.3 Pruebas sintéticas históricas: qué acreditan y qué no

Para los cuatro prototipos compactos y el corrector HAR anterior se documentaron **diez pruebas sintéticas** el 08/11/2026 con Python 3.12.14, torch 2.15.0+cpu, numpy 2.3.5 y un hilo CPU. Cubren formas y finitez, gradientes, invariancia de batch, roundtrip de pesos en memoria, entradas inválidas, reducción de pérdida artificial y restricciones de calendario/alcance. Incluyen iteradores de residuos válidos, inválidos y vacíos.

No se abrieron datos de mercado. Pasar esta suite no demuestra rendimiento, robustez, potencia ni ausencia de toda fuga. Tampoco acredita que los once candidatos estén integrados o listos para entrenar. **Ninguna de estas diez pruebas valida CNN-1D + LSTM + atención temporal**, que dispone de pruebas nuevas separadas en §18.

El corrector HAR histórico inactivo calcula `max(0, HAR + corrección)`. Si la suma es negativa, el truncamiento tiene gradiente cero; la prueba específica confirma esta limitación. Por ello, su `forward` es un enlace provisional de inferencia, **no una pérdida validada para entrenar con datos reales**. La generación de residuos OOF y el enlace pendiente se documentan como límites de ese prototipo retirado, no como próximos pasos del híbrido vigente.

## 4.4 Guardas temporales antes de cualquier ajuste

1. **Ámbito de entrenamiento:** validar orígenes antes de ajustar transformaciones, reconstruir con el autoencoder o entrenar la cabeza supervisada. Preentrenar sin etiquetas sobre todo desarrollo también expondría datos posteriores al fold.
2. **Reinicio por fold:** pesos, estado recurrente y transformaciones nuevos; ningún checkpoint posterior se reutiliza hacia el pasado.
3. **Disponibilidad:** conservar días calendario, huecos y soportes completos de atributos y etiquetas según el contrato sincronizado. La referencia E1 conserva etiquetas hasta o−8 y embargo de siete días por origen o; los cortes deben comprobarse sobre los datos realmente disponibles. Esta separación no se confunde con el soporte extendido del antiguo estimador HAR.
4. **Causalidad del híbrido:** verificar la convolución con padding izquierdo, la dirección de la LSTM y la atención restringida a siete estados observados. No usar ventanas centradas, etiquetas futuras ni estados recurrentes de otro origen.
5. **Auditoría integral:** comprobar construcción de ventanas, `relative_window`, ajustes de transformaciones y selección dentro de cada fold. Las utilidades previas sólo comprueban los metadatos que reciben; no acreditan el futuro pipeline del híbrido.

Las pruebas nuevas deberán cubrir formas, gradientes, independencia de los acompañantes del batch, checkpoint y causalidad, además de duplicados pertinentes y controles negativos. Son verificaciones propuestas, todavía no ejecutadas para el híbrido. Un fallo requiere corrección y nueva verificación; no se resuelve describiéndolo como limitación menor.

## 4.5 Secuencia histórica propuesta para el experimento anterior

- Fijar fuentes, correspondencia con el curso y recetas de los diez modelos y del candidato 11; distinguir adaptación propia de reproducción. Resolver la condición NAS conforme a [§14](14_arquitecturas.md).
- Implementar CNN-1D + LSTM + atención temporal y sus ablaciones conforme al contrato común. Integrar los seis adaptadores en un ejecutor nuevo sin modificar el código ni los protocolos históricos.
- Cerrar calendario, elegibilidad, transformaciones, hipótesis, comparaciones, presupuesto completo y reglas de fallos. El preentrenamiento del autoencoder, el ajuste conjunto del híbrido y todas sus ablaciones cuentan en el coste.
- Validar el pipeline y publicar su propio preregistro antes de ajustes de mercado. Definir selección temporal, congelación de recetas y dimensionamiento sin trasladar un R o presupuesto anterior.
- Cerrar pesos y, cuando correspondan, calibradores antes de evaluar; publicar resultados y costes medidos, también fallos y desenlaces desfavorables.

La secuencia anterior se conserva como antecedente, sin instruir nuevas corridas para el cierre documental actual. El desarrollo y la reserva histórica ya observados no recuperan cegamiento al cambiar los modelos. Una evaluación sobre esa reserva mantendrá alcance exploratorio.

## 4.6 Trazabilidad sin reactivar el catálogo retirado

El [recibo P0/OOF](../_static/remediation_preflight/execution_summary.json) documenta **220 trayectorias y 440 ajustes históricos**. El [recibo HPO del 06/11/2026](../_static/completion_preflight/execution_progress.json) documenta un inicio parcial de una ejecución anterior, no actividad actual ni cierre. Ninguno corresponde al catálogo vigente.

Los protocolos, fuentes, candidatos y resultados originales permanecen intactos: [registro P0](../_static/remediation_protocol/preregistration.json), [registro de la enmienda histórica](../_static/completion_preflight/registration.json) y [paquete histórico de implementación](../_static/completion_download/completion_bundle.zip). Se enlazan para auditar lo que ocurrió, no como obligaciones futuras ni resultados transferibles. El estado del nuevo preregistro está en [§9](09_protocolo_registrado.md), la comparación en [§6](06_benchmark_final.md) y la equidad del ajuste en [§11](11_hpo_equidad.md).


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/04_implementacion.md>` se conserva con su SHA256 en el [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json). Sus estados pendientes describen el corte original.
