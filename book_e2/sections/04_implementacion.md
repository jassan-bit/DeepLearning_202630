# 4. Implementación de los diez modelos y el candidato adicional

**Catálogo vigente al 08/11/2026:** FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer. El modelo 11 es **HAR + red residual**, candidato cuya superioridad no está demostrada. Esta selección sustituye el catálogo anterior en la presentación de E2; sus registros experimentales se conservan sólo como trazabilidad.

**Estado real:** cuatro prototipos compactos y un corrector que recibe una predicción HAR externa, aislados del ejecutor histórico; diez pruebas sintéticas documentadas. Los otros seis modelos tienen código histórico que necesita integración y validación en el nuevo diseño. **No hay entrenamiento de mercado, HPO, dimensionamiento de repeticiones ni resultados finales del catálogo vigente.**

## 4.0 Implementación preliminar y alcance

El código está en [redes propuestas](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_proposed_models.py), [guardas temporales](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_proposed_temporal.py) y [pruebas sintéticas](https://github.com/jassan-bit/DeepLearning_202630/blob/main/tests/test_e2_proposed_benchmark.py). No implementa un ejecutor completo de entrenamiento, selección y evaluación.

Las [fichas bibliográficas](16_fichas_nuevas.md) describen fuentes y adaptaciones posibles. No son la receta de los prototipos compactos ni configuraciones ya seleccionadas:

| Modelo o componente | Implementación disponible | Límite que debe resolverse |
|---|---|---|
| FCN-1D | Tres convoluciones con kernels 3/3/1, ancho predeterminado 16, ReLU y pooling promedio | No replica los anchos, kernels ni normalización de la fuente de clasificación |
| ResNet-1D | Dos bloques de convoluciones 3/3 con proyección de atajo y ancho 16 | No reproduce la profundidad de la arquitectura fuente |
| Inception-1D | Bottleneck, ramas 1/3/5 y max-pooling, una red compacta | No reproduce los módulos ni el ensemble de InceptionTime |
| LSTM-autoencoder con predictor | Encoder LSTM, decoder de latente repetido y cabeza escalar; reconstrucción y predicción separadas | Sin política integrada de preentrenamiento, congelado, selección de épocas o ajuste supervisado |
| HAR + red residual | Corrección firmada de D7×5 añadida a un pronóstico HAR externo; truncamiento a cero en inferencia | Sin estimación HAR/OLS ni generación real de residuos fuera de muestra; enlace de entrenamiento pendiente |

Las cuatro redes reciben `(batch,7,5)` y producen `(batch,)` mediante Softplus. Son decisiones del prototipo, no valores atribuidos a los artículos. El padding simétrico opera dentro de una ventana ya observada: no añade historia ni constituye por sí mismo una convolución causal intraventana.

## 4.1 Contrato de información

Los diez modelos neurales se plantean sobre **D7×5 OHLCV → σ₇**, con horizonte de un día. σ₇ es la desviación poblacional de siete retornos logarítmicos diarios, `ddof=0`, sin anualización. La evaluación debe usar unidades originales y transformaciones aprendidas exclusivamente en el entrenamiento permitido de cada fold. No se añaden rezagos del objetivo como canales neurales sin una modificación explícita del contrato.

**HAR tiene otro soporte.** La utilidad preliminar agrega σ₇ en 1/7/22 días calendario anteriores al origen. Una agregación de 22 días puede necesitar cierres hasta t−29; no recibe sólo las 35 entradas de las redes. Elegir 22 o 30 días para Bitcoin, definir el calentamiento y fijar las fechas elegibles comunes son decisiones pendientes, no hiperparámetros que puedan elegirse observando la evaluación.

Los controles **HAR solo**, **red sola** y controles con soporte comparable son necesarios para interpretar el candidato 11. Son controles del experimento del híbrido, no incorporaciones tácitas al catálogo principal. La ventaja sobre una red D7, si aparece, no puede atribuirse exclusivamente a su arquitectura cuando dispone de historia adicional.

## 4.2 Integración pendiente de los seis modelos con código histórico

| Modelo vigente | Verificación requerida al integrarlo |
|---|---|
| TimesNet | Frecuencias disponibles con D7, selección por origen, extensión temporal y cabeza de σ₇ |
| TSMixer | Orden de mezclas temporal/canal, normalización, independencia de acompañantes del batch y lectura escalar |
| TiDE | Encoder/decoder y conexiones residuales; ausencia de covariables futuras y escala correcta del objetivo |
| TimeMixer | Escalas no vacías con siete observaciones, descomposición y combinación de salidas |
| PatchTST | Parches y padding explícitos; normalización sólo de lo observado y proyección a σ₇ |
| iTransformer | Tokens de variables, atención, normalización y cabeza que no desnormalice OHLCV como volatilidad |

La [revisión arquitectónica histórica](../_static/completion_protocol/architecture_review.json) permite localizar decisiones anteriores. No certifica estas seis integraciones en el nuevo experimento ni congela sus recetas. Las verificaciones se repetirán sobre la fuente, el entorno y el pipeline que efectivamente se registren.

## 4.3 Pruebas sintéticas: qué acreditan y qué no

Se documentaron **diez pruebas sintéticas** el 08/11/2026 con Python 3.12.14, torch 2.15.0+cpu, numpy 2.3.5 y un hilo CPU. Cubren formas y finitez, gradientes, invariancia de batch, roundtrip de pesos en memoria, entradas inválidas, reducción de pérdida artificial y restricciones de calendario/alcance. Incluyen iteradores de residuos válidos, inválidos y vacíos.

No se abrieron datos de mercado. Pasar esta suite no demuestra rendimiento, robustez, potencia ni ausencia de toda fuga. Tampoco acredita que los once candidatos estén integrados o listos para entrenar.

El corrector HAR calcula `max(0, HAR + corrección)`. Si la suma es negativa, el truncamiento tiene gradiente cero; la prueba específica confirma esta limitación. Por ello, su `forward` es un enlace provisional de inferencia, **no una pérdida validada para entrenar con datos reales**. Ajustar la corrección firmada a residuos OOF crudos y truncar sólo al inferir es una alternativa que aún debe diseñarse y registrarse.

## 4.4 Guardas temporales antes de cualquier ajuste

1. **Ámbito de entrenamiento:** validar orígenes antes de ajustar escaladores, reconstruir con el autoencoder o entrenar la cabeza supervisada. Preentrenar sin etiquetas sobre todo desarrollo también expondría datos posteriores al fold.
2. **Reinicio por fold:** pesos, estado y transformaciones nuevos; ningún checkpoint posterior se reutiliza hacia el pasado.
3. **Disponibilidad:** conservar días calendario, huecos y soportes completos de atributos y etiquetas. La guarda preliminar exige etiquetas del estimador base hasta q−8 para un origen q; el nuevo calendario debe justificar e integrar esa separación.
4. **Residuos fuera de muestra:** generar realmente cada pronóstico HAR sin su etiqueta ni información posterior, dentro del train autorizado. El corrector sólo puede aprender residuos cuya verdad ya estaba disponible al ajustarlo.
5. **Auditoría de procedencia:** las utilidades comprueban metadatos recibidos, pero no demuestran que sean verdaderos. Faltan el generador temporal OOF y pruebas integrales que comprueben cortes, escalado y ajuste.

La auditoría debe cubrir también duplicados pertinentes, disponibilidad de atributos y controles negativos. Un fallo requiere corrección y nueva verificación; no se resuelve describiéndolo como limitación menor.

## 4.5 Secuencia necesaria del nuevo experimento

- Fijar fuentes, correspondencia con el curso y recetas de los diez modelos y del candidato 11; distinguir adaptación propia de reproducción. Resolver la condición NAS conforme a [§14](14_arquitecturas.md).
- Implementar HAR, cross-fitting y controles del híbrido; resolver su objetivo, enlace y soporte. Integrar los seis adaptadores en un ejecutor nuevo sin modificar los históricos.
- Cerrar calendario, elegibilidad, transformaciones, hipótesis, comparaciones, presupuesto completo y reglas de fallos. El preentrenamiento y todas las etapas del híbrido cuentan en el coste.
- Validar el pipeline y publicar su propio preregistro antes de ajustes de mercado. Definir selección temporal, congelación de recetas y dimensionamiento sin trasladar un R o presupuesto anterior.
- Cerrar pesos y, cuando correspondan, calibradores antes de evaluar; publicar resultados y costes medidos, también fallos y desenlaces desfavorables.

El desarrollo y la reserva histórica ya observados no recuperan cegamiento al cambiar los modelos. Una evaluación sobre esa reserva mantendrá alcance exploratorio.

## 4.6 Trazabilidad sin reactivar el catálogo retirado

El [recibo P0/OOF](../_static/remediation_preflight/execution_summary.json) documenta **220 trayectorias y 440 ajustes históricos**. El [recibo HPO del 06/11/2026](../_static/completion_preflight/execution_progress.json) documenta un inicio parcial de una ejecución anterior, no actividad actual ni cierre. Ninguno corresponde al catálogo vigente.

Los protocolos, fuentes, candidatos y resultados originales permanecen intactos: [registro P0](../_static/remediation_protocol/preregistration.json), [registro de la enmienda histórica](../_static/completion_preflight/registration.json) y [paquete histórico de implementación](../_static/completion_download/completion_bundle.zip). Se enlazan para auditar lo que ocurrió, no como obligaciones futuras ni resultados transferibles. El estado del nuevo preregistro está en [§9](09_protocolo_registrado.md), la comparación en [§6](06_benchmark_final.md) y la equidad del ajuste en [§11](11_hpo_equidad.md).
