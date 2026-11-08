# 15. Propuesta de revisión: diez modelos y un híbrido candidato

**Revisión editorial del 8 de noviembre de 2026. Estado: propuesta para el siguiente diseño, no catálogo ejecutado ni enmienda experimental registrada.** Esta página recoge el cambio de selección solicitado: combinar arquitecturas trabajadas en el curso con mecanismos de la literatura y estudiar un modelo adicional. No acredita que los once modelos estén implementados, entrenados o que uno sea superior. La correspondencia exacta con las configuraciones del profesor y las fuentes primarias de las nuevas adaptaciones debe documentarse antes de congelar el diseño.

El [catálogo previamente registrado](04_implementacion.md) conserva diez adaptadores, veinte configuraciones y quince recetas ajustables. Sus protocolos, candidatos, registros, resultados P0 y descargas históricas no se modifican. Esta propuesta no autoriza a reutilizar sus recibos como evidencia de las nuevas arquitecturas.

## 15.1 Los diez modelos propuestos

Los cuatro primeros incorporan las familias solicitadas para la revisión del curso; los seis restantes conservan mecanismos del catálogo bibliográfico anterior. La pertenencia a una familia no basta para fijar una arquitectura ni para eximir una búsqueda estructural exigida por la guía.

| N.º | Modelo propuesto | Familia y mecanismo que se compararía | Trabajo necesario antes de entrenar |
|---|---|---|---|
| 1 | **FCN-1D** | Convolución; extracción temporal sin capas recurrentes | Fijar fuente/configuración, kernels válidos en D7, pooling y cabeza escalar de regresión |
| 2 | **ResNet-1D** | Convolución residual; bloques con conexiones de salto | Distinguirla de la TCN histórica; fijar profundidad, dilataciones, padding y cabeza de volatilidad |
| 3 | **Inception-1D** | Convoluciones paralelas con varias escalas | Fijar variante, kernels y bottlenecks compatibles con siete observaciones; no copiar sin revisión una receta de clasificación |
| 4 | **LSTM-autoencoder predictivo** | Representación recurrente con reconstrucción y predicción | Definir encoder, decoder y cabeza supervisada; pérdidas y pesos, uso de reconstrucción y entrenamiento exclusivamente en train |
| 5 | **TimesNet** | Periodicidad y convolución sobre representaciones temporales | Revisar la adaptación existente, selección de frecuencias por origen y compatibilidad con el nuevo diseño |
| 6 | **TSMixer** | Mezcla temporal y de canales mediante MLP | Revisar la receta existente y conservar trazabilidad de cambios |
| 7 | **TiDE** | Encoder/decoder denso con conexiones residuales | Conservar la adaptación sin covariables futuras ni objetivo rezagado, salvo enmienda explícita |
| 8 | **TimeMixer** | Mezcla multiescala y descomposición | Verificar escalas no vacías para D7 y la receta fuente |
| 9 | **PatchTST** | Atención sobre parches temporales | Fijar parches válidos, representación y salida escalar |
| 10 | **iTransformer** | Atención entre tokens de variables | Conservar causalidad por origen, escalado del objetivo y adaptación D7×5 |

**LSTM-autoencoder predictivo no equivale a la LSTM supervisada de E1.** Solo será un comparador distinto si la representación, el objetivo de reconstrucción y la conexión a la predicción quedan definidos y evaluables; cambiar el nombre de una LSTM no crea otro mecanismo.

La lista cubre convolución, recurrencia/autoencoder, MLP/mixing y atención. Sustituye la cobertura explícita de estado selectivo de S-Mamba por recurrencia/autoencoder. Debe cotejarse con las cuatro familias que exige la guía antes de afirmar cumplimiento. Las fichas históricas de los seis modelos conservados son un punto de partida, no validación automática de una nueva receta.

## 15.2 Cambios respecto del catálogo anterior

Se **conservan seis** modelos: TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer. Se propone **retirar cuatro del nuevo Top 10**: Informer, Autoformer, Crossformer y S-Mamba. Se propone **incorporar cuatro**: FCN-1D, ResNet-1D, Inception-1D y LSTM-autoencoder predictivo.

Son cuatro sustituciones del conjunto, no cuatro equivalencias arquitectónicas ni un ranking de calidad. Los modelos retirados permanecen en la revisión y en el registro histórico; no se eliminan sus evidencias. La nueva lista se motiva por cobertura del curso y diversidad de mecanismos, no por resultados Bitcoin inexistentes. No se trasladan a los nuevos modelos las páginas leídas, puntuaciones, licencias o pruebas de los modelos que salen.

La selección bibliográfica previa era provisional. Antes de convertir esta revisión en selección experimental hay que incorporar fuentes primarias y configuraciones verificadas para las cuatro altas, sus criterios de inclusión, adaptación a regresión y comparación con alternativas. Las métricas de clasificación o de otro dataset no demuestran precisión en σ₇.

## 15.3 Modelo 11: HAR + red residual, hipótesis por evaluar

Se propone estudiar un **híbrido HAR + red neuronal de corrección residual**: un componente lineal con agregaciones diarias, semanales y mensuales de volatilidad, más una red que intente aprender el error restante. Es un candidato adicional, **no un modelo superior demostrado** ni una contribución original acreditada. Puede rendir peor que HAR, que una red sola o que persistencia.

La motivación es separar persistencia en distintas escalas y relaciones no lineales. HAR cuenta con un antecedente establecido: Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility*, [DOI 10.1093/jjfinec/nbp001](https://doi.org/10.1093/jjfinec/nbp001). Citar ese antecedente no valida automáticamente su adaptación al objetivo σ₇ calculado con retornos diarios de este proyecto ni la novedad del híbrido.

**El soporte de información cambia.** Un HAR convencional requiere historia del objetivo y una agregación mensual, habitualmente de 22 sesiones. El contrato neural anterior usa únicamente D7×5 OHLCV y prohíbe rezagos de σ₇. Para Bitcoin diario deben fijarse explícitamente días calendario frente a sesiones, longitudes y transformación del objetivo. No puede presentarse HAR como si recibiera las mismas 35 entradas ni construir su componente mensual a partir de siete días. Las fechas evaluables comunes y el periodo de calentamiento deben quedar definidos sin comprimir huecos.

Antes del primer ajuste de esta propuesta se deberán decidir y registrar:

1. La variable del componente HAR (σ₇, su transformación o varianza), la inversión a unidades originales y el tratamiento de salidas negativas. No se asume que volatilidad realizada intradía y σ₇ sean objetivos intercambiables.
2. Los rezagos disponibles en cada origen y la arquitectura exacta de la red residual. La etiqueta futura no puede aparecer entre las entradas.
3. La generación temporal fuera de muestra de los residuos usados para entrenar la corrección, dentro de desarrollo. Usar residuos de test o ajustar la red a errores elegidos después de ver la evaluación introduciría selección indebida.
4. La comparación con **HAR solo**, **red sola** y **HAR + red**, además de las referencias pertinentes. Si cambia el soporte de información, añadir controles con soporte equiparable o declarar que la diferencia no identifica una ventaja exclusivamente arquitectónica.
5. El mismo procedimiento de selección temporal, presupuesto trazable, semillas, costes e inferencia con corrección por multiplicidad. No dar más oportunidades al híbrido hasta obtener una ventaja.
6. Un criterio previo de mejora práctica, calibración y robustez, con posibilidad explícita de resultado nulo o desfavorable. El margen histórico no se cambia después de conocer los resultados.

El test ya conocido de E1/E2 no recupera independencia con esta propuesta. Una comparación sobre él seguirá siendo exploratoria. Una afirmación confirmatoria de superioridad requeriría una evaluación no utilizada para escoger el híbrido, congelada después de cerrar el diseño y con sus límites temporales documentados.

### Fuentes verificadas y límites de las nuevas propuestas

La ampliación bibliográfica del 08/11/2026 distingue el **origen del mecanismo** de su validación para Bitcoin. No transfiere puntuaciones ni resultados de los cuatro modelos retirados a las altas.

| Candidato | Fuente primaria identificada | Evidencia y adaptación pendiente |
|---|---|---|
| FCN-1D | Wang, Yan y Oates, [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/abs/1611.06455) | FCN para clasificación temporal. La fuente no valida regresión de σ₇ en Bitcoin. Fijar kernels compatibles con D7, pooling, canales y cabeza escalar; justificar cambios respecto de clasificación |
| ResNet-1D | Wang, Yan y Oates, [misma fuente primaria](https://arxiv.org/abs/1611.06455) | Red residual para clasificación temporal; no equivale a la TCN histórica. Documentar bloques, padding, profundidad y salida de regresión; una fuente compartida no son dos estudios independientes |
| Inception-1D | Fawaz et al., [InceptionTime: Finding AlexNet for Time Series Classification](https://arxiv.org/abs/1909.04939) | Arquitectura/ensemble de clasificación. Decidir red individual frente a ensemble, kernels y bottlenecks para D7; su precisión clasificatoria no es una RMSE de volatilidad |
| LSTM-autoencoder predictivo | Fuente/configuración exacta **todavía no cerrada** | Los trabajos adyacentes de detección de anomalías o VAE con LSTM no certifican este encoder–decoder supervisado. Deben fijarse fuente primaria, pérdida de reconstrucción, peso de predicción y entrenamiento causal antes del registro |
| HAR + red residual | Lian et al., [Signal Simplification Is Not Predictive Simplification](https://arxiv.org/abs/2610.03019) | Antecedente directo HAR-style + LSTM residual. El preprint informa deterioro frente a HAR en cinco activos estadounidenses; impide alegar novedad genérica o mejora garantizada, sin demostrar que el mismo desenlace ocurra en Bitcoin |

**Evidencia negativa que cambia la justificación.** En el estudio de Lian et al., el MSE medio pasa de 0,3049 para HAR-style a 0,3594 para HAR-style + LSTM; la LSTM sola obtiene 0,2649. Son valores del artículo en su propio objetivo y protocolo, no resultados E2 ni magnitudes comparables con σ₇. La menor varianza residual no basta para justificar una corrección neuronal. E2 debe mantener HAR solo, red sola y la corrección como controles y aceptar resultados desfavorables.

[HARNet (arXiv:2205.07719)](https://arxiv.org/abs/2205.07719) es un antecedente distinto: convoluciones causales dilatadas inicializadas para reproducir HAR, aplicadas a volatilidad realizada de índices bursátiles. No es sinónimo de una LSTM entrenada sobre residuos. Su mecanismo puede informar una alternativa, pero no se añade de forma tácita al catálogo ni convierte el candidato 11 en una réplica.

La [actualización bibliográfica](03_literatura.md) conserva las consultas, la procedencia y los niveles de lectura de esta ronda. La fuente exacta del autoencoder, el código de cada adaptación y la correspondencia con las configuraciones del curso siguen siendo requisitos abiertos; estas fichas no acreditan implementación o entrenamiento.

## 15.4 E1 actual y las anclas históricas de E2

El núcleo actual de [E1](https://jassan-bit.github.io/DeepLearning_202630/intro.html) se organiza alrededor de **MLP, LSTM y GRU**. E2, en cambio, registró su contrato desde la fase II anterior con **GRU/D7-c03, LSTM/D7-c02 y TCN/D7-c02**, como fija [§1](01_punto_partida.md) y su referencia versionada. La TCN sigue siendo una ancla histórica de ese registro; no se transforma retrospectivamente en MLP.

Esta revisión no sustituye las anclas ni reasigna sus métricas, particiones o presupuesto. Si el próximo benchmark incorpora MLP como referencia del E1 actual, deberá identificar la receta y cohorte concretas y decidir si se añade como ancla o como variante reajustada. Esa decisión modifica el recuento experimental y exige trazabilidad propia. **Diez candidatos + un híbrido no significa que el experimento completo tenga once configuraciones**: faltan decidir sus baselines y controles, incluido HAR solo.

## 15.5 Orden de cierre y límites de esta actualización

- Resolver fuentes, configuraciones del curso, cuatro familias exigidas y justificación de las cuatro sustituciones.
- Fijar contrato de entradas, objetivo y soportes del híbrido; concretar baselines, controles y total de configuraciones.
- Diseñar una enmienda nueva con hipótesis, candidatos, presupuesto, validación temporal, potencia, incertidumbre y comparaciones. Conservar sin cambios el registro anterior y declarar qué información ya se observó.
- Implementar los adaptadores nuevos y realizar controles de forma, gradientes, invariancia de batch, causalidad y memorización sintética. Pasar pruebas no equivale a obtener resultados de mercado.
- Solo después, registrar la versión ejecutable, cerrar selección y evaluar según el nuevo plan. Presentar todos los resultados, incluidos los negativos.

**En esta actualización se cambian únicamente documentación y navegación.** No se modifica el ejecutor registrado, no se alteran candidatos ni recibos experimentales, no se entrenan redes y no se publican originales, pesos, predicciones individuales o datos completos de evaluación.
