# 23. Implementación de recetas y controles temporales

## Alcance del nuevo corte

Este avance convierte las fichas de §22 en código verificable mediante datos artificiales. No constituye un benchmark de mercado, una selección de hiperparámetros ni una adjudicación NAS. No se han leído datos de mercado en estas pruebas y no hay resultados predictivos nuevos sobre Bitcoin.

Los prototipos y recibos de §18 se conservan sin modificaciones. §22 permanece como registro fechado de las propuestas: su afirmación «sin cambios de implementación» describe aquel corte, no niega el avance documentado aquí. Los resultados históricos de E1 y de los estudios previos no se reasignan a estos cuerpos.

## Qué cambia en la implementación

- **FCN-1D:** cuerpo 128/256/128 con kernels 8/5/3, BN y media temporal. La cabeza escalar Softplus es una adaptación a σ₇, no la salida clasificadora original.
- **ResNet-1D:** tres bloques 64/128/128, nueve convoluciones principales y tres proyecciones de atajo. Se conserva la BN inicial únicamente en la rama principal y la proyección del tercer atajo 128→128.
- **Inception-1D:** seis módulos y dos conexiones residuales, cuatro ramas de 32 canales con kernels 40/20/10. Es un miembro individual, no el ensemble de cinco. El padding de max-pool excluye posiciones ausentes del máximo.
- **LSTM-autoencoder con predictor:** dos preentrenamientos separados, anchos 25 y 26, reconstruyendo ambos la entrada original de cinco canales. El segundo recibe el vector final del primer encoder repetido siete veces. Se transfieren los encoders, se descartan los decoders y se ajusta el predictor con un optimizador nuevo.
- **Otros seis adaptadores y candidato CNN-LSTM-atención:** se reutilizan sin alterar sus cuerpos históricos, con las dimensiones completas de las fichas y las diferencias D7 documentadas. Los seis adaptadores siguen agregando diez momentos de la misma ventana; los cuatro cuerpos nuevos y el híbrido no.

Estos módulos se escribieron como implementaciones propias de las arquitecturas documentadas. No se incorporan pesos publicados ni se afirma igualdad numérica con los entrenamientos de los autores. La traducción de backend, inicialización y cabeza de regresión conserva límites explícitos; el smoke no reemplaza una reproducción independiente del resultado científico.

## Autoencoder: fases, escala y límites

El objetivo auxiliar es X/s, con s=max(1,max|X_train|), estimado exclusivamente sobre entrenamiento. La entrada del encoder no se divide por s. El mismo s se utiliza en ambas fases y no se recalcula para validación. Las observaciones de validación pueden exceder el rango de reconstrucción de los decoders LSTM; se informa sin recortar ni ajustar usando validación.

El primer encoder permanece fijo durante el segundo preentrenamiento. El predictor utiliza ambos encoders con dropout 0,3 y una cabeza Softplus. No hay pérdida conjunta de reconstrucción y pronóstico. Esta escala auxiliar es una adaptación propia declarada, no una operación acreditada por el paper original ni una decisión NAS ya aceptada.

## Comparadores temporales

Las APIs individuales de §21 son un paso de implementación, no un ejecutor completo de la búsqueda clásica ni una certificación de las cohortes reales de 305, 53 o 608 fechas.

- Persistencia y naive estacional consultan σ₇(t−m), con m=1 y m∈{7,14,28}. No sustituyen un rezago ausente por otro valor.
- ARIMA ajusta parámetros y escala hasta origen−8. Después actualiza el estado con observaciones hasta t−1 sin reajustar parámetros y emite el pronóstico antes de incorporar la observación de t.
- El calendario conserva días faltantes. El filtro puede avanzar con una observación ausente, pero no convierte ese dato en una etiqueta evaluable.
- La salida conserva el valor crudo y registra el truncamiento a cero. No convergencia, salida no finita o dependencia incompatible son fallos explícitos, no un pronóstico cero ni una oportunidad de cambiar la receta.
- No se han implementado todavía la selección completa A/B, el registro durable de pronósticos de mercado ni la evaluación común de todos los sistemas.

## Evidencia de ejecución sintética

[Recibo de la ejecución](../_static/recipe_implementation_20261109/synthetic_receipt.json) · [Resumen y hashes](../_static/recipe_implementation_20261109/summary.json) · {download}`Reproducción y límites <../_static/recipe_implementation_20261109/README.md>`.

Se ejecutaron **11/11 modelos, sin omisiones**, sobre 96 días artificiales, 32 ventanas de entrenamiento y ocho de validación. Se realizó un paso supervisado por modelo y uno en cada fase auxiliar SAE: **13 pasos en total**. La validación no intervino en optimización ni selección. No se redujeron anchos o profundidad para el smoke.

El flujo verificó 70 archivos fijados de los cuatro adaptadores externos. Ese alcance no equivale a verificar repositorios completos ni a certificar las fuentes de los cuatro cuerpos propios.

| Modelo | Parámetros del predictor |
|---|---:|
| FCN-1D | 268,929 |
| ResNet-1D | 522,955 |
| Inception-1D | 490,561 |
| LSTM-autoencoder con predictor | 8,739 |
| TimesNet | 1,172,875 |
| TSMixer | 35,044 |
| TiDE | 168,537 |
| TimeMixer | 2,615 |
| PatchTST | 16,664 |
| iTransformer | 795,403 |
| CNN-1D + LSTM + atención temporal | 9,537 |

Los dos autoencoders auxiliares tienen 3.840 y 6.172 parámetros respectivamente; sus decoders no forman parte de los 8.739 parámetros del predictor transferido. Sus etapas sí deben contarse en un presupuesto posterior.

Se aprobaron **39 pruebas nuevas**: 14 de cuerpos, seis de integración del catálogo y 19 de controles temporales. Las seis de catálogo se ejecutaron con fuentes externas verificadas; las 19 de controles, con statsmodels 0.14.5 en entorno separado. Sin esa dependencia, once pruebas de controles se omiten: una ejecución con omisiones no sustituye este resultado.

Las pruebas cubren gradientes, B=1, salida finita, BN poblacional y estadísticas congeladas en evaluación, padding SAME, pooling de valores negativos, fases SAE y transferencia, escala ajustada al entrenamiento, calendario, embargos, invariancia al futuro y filtrado ARIMA sin refit. Un fixture ARIMA con hueco de entrenamiento no converge: su rechazo explícito es parte de la prueba, no un ajuste exitoso. Un fallo del backend detiene terminalmente esa instancia; no puede reintentarse ni seguir filtrando.

**Límites de equivalencia:** la Conv2D cuadrada de la fuente Wang sobre un eje singleton se traduce a Conv1D; cambian pesos inactivos, fan-in e inicialización. La BN usa una convención no fusionada de varianza poblacional, ε=0,001 y retención 0,99, sin prometer igualdad bit a bit con TensorFlow/Keras histórico. Las puertas LSTM se verifican, pero no la serialización CuDNN ni la compatibilidad de checkpoints. Estas pruebas CPU no certifican GPU o precisión mixta.

## Decisiones que siguen abiertas

1. Adjudicar por receta las dudas NAS y las adaptaciones de §22. Implementar un cuerpo no concede una exención académica.
2. Cerrar el espacio HPO mínimo de la guía, hardware, techo de recursos y contabilidad de todas las etapas. Las dos fases SAE no son gratuitas y un tiempo sintético no estima el presupuesto de mercado.
3. Integrar y auditar las cohortes reales comunes de los comparadores, sin escoger máscaras después de ver pérdidas.
4. Congelar protocolo, inferencia y orden de ejecución antes de cualquier entrenamiento de mercado. El test de 608 fechas ya observado continúa siendo exploratorio.
5. Completar las verificaciones de licencias, entorno independiente y límites de reproducción. No se afirma convergencia, potencia, cobertura ni superioridad del híbrido.
