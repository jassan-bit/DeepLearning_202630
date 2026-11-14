# 15. Catálogo vigente: diez modelos y candidato 11

**Vigencia posterior del catálogo y de su auditoría bibliográfica:** el alcance obligatorio se consulta en [§43](43_catalogo_obligatorio.md); el candidato 11 de este corte histórico queda como antecedente opcional. Los resultados propios se consultan en [§6](06_benchmark_final.md). La [auditoría bibliográfica vigente (§59)](59_auditoria_literatura_final.md) concilia **9.559 grupos recibidos**, conservando aparte las fuentes recuperadas de forma dirigida. El catálogo actual contiene diez sistemas y nueve fuentes primarias; su puntaje de apoyo anterior tiene **36/50 componentes observados**. La nueva sensibilidad contextual de P usa seis fuentes ETTh1/H96 y muestra las diferencias de lookback, split y escala; no convierte esas cifras en resultados Bitcoin ni determina retrospectivamente el catálogo. Persisten cuatro C desconocidos y el límite de revisión global incompleta. **Fechas:** el recibo del corte del 14/11/2026 a las 04:14 UTC corresponde al 13/11/2026 a las 23:14 en Bogotá (UTC−5). Las declaraciones y descargas anteriores se preservan a continuación como evidencia de sus fechas originales.

**Vigencia documental (11/11/2026):** existe selección HPO/NAS registrada y ejecutada para el catálogo actual; las afirmaciones de ausencia de HPO o de falta de registro de selección corresponden a cortes anteriores. El benchmark final, R y su cierre inferencial siguen pendientes. [Registro y alcance actuales (§9)](09_protocolo_registrado.md) · [Calendario (§40)](40_calendario_original.md) · [Corte de resultados (§41)](41_avance_optimizacion_11_origenes.md) · [Auditoría y trabajo paralelo (§42)](42_auditoria_y_trabajo_paralelo.md).

**Avance de software:** fábrica de los once candidatos, híbrido y flujo sintético implementados y verificados; recetas provisionales, sin datos de mercado. Véase [§18: integración y pruebas sintéticas](18_integracion_sintetica.md). No hay HPO ni benchmark vigente ejecutado.

**Selección vigente del 8 de noviembre de 2026. Estado: catálogo editorial aprobado, no benchmark ejecutado ni enmienda experimental registrada.** Esta página recoge el cambio de selección solicitado: combinar arquitecturas trabajadas en el curso con mecanismos de la literatura y estudiar un modelo adicional. La selección editorial por sí sola no acredita implementación; §18 documenta ahora los once candidatos implementados provisionalmente y sus pruebas sintéticas. No hay entrenamiento de mercado ni superioridad demostrada. La correspondencia exacta con las configuraciones del profesor y las fuentes primarias de las nuevas adaptaciones debe documentarse antes de congelar el diseño.

Los diez modelos se mantienen. El candidato 11 activo cambia de HAR + red residual a **CNN-1D + LSTM + atención temporal**. El corrector HAR y sus pruebas quedan como antecedente histórico inactivo. Los recibos, protocolos y código previos permanecen intactos para auditoría, pero no acreditan la implementación o evaluación del nuevo híbrido.

## 15.1 Los diez modelos vigentes

Los diez nombres corresponden a modelos de literatura o adaptaciones a las familias estudiadas; no se presentan como una lista literal de modelos del curso. La pertenencia a una familia no basta para fijar una arquitectura ni para eximir una búsqueda estructural exigida por la guía.

| N.º | Modelo vigente | Familia y mecanismo que se compararía | Trabajo necesario antes de entrenar |
|---|---|---|---|
| 1 | **FCN-1D** | Convolución; extracción temporal sin capas recurrentes | Fijar fuente/configuración, kernels válidos en D7, pooling y cabeza escalar de regresión |
| 2 | **ResNet-1D** | Convolución residual; bloques con conexiones de salto | Distinguirla de la TCN histórica; fijar profundidad, dilataciones, padding y cabeza de volatilidad |
| 3 | **Inception-1D** | Convoluciones paralelas con varias escalas | Fijar variante, kernels y bottlenecks compatibles con siete observaciones; no copiar sin revisión una receta de clasificación |
| 4 | **LSTM-autoencoder con predictor** | Representación recurrente con reconstrucción y predicción | Definir encoder, decoder y cabeza supervisada; pérdidas y pesos, uso de reconstrucción y entrenamiento exclusivamente en train |
| 5 | **TimesNet** | Periodicidad y convolución sobre representaciones temporales | Revisar la adaptación existente, selección de frecuencias por origen y compatibilidad con el nuevo diseño |
| 6 | **TSMixer** | Mezcla temporal y de canales mediante MLP | Revisar la receta existente y conservar trazabilidad de cambios |
| 7 | **TiDE** | Encoder/decoder denso con conexiones residuales | Conservar la adaptación sin covariables futuras ni objetivo rezagado, salvo enmienda explícita |
| 8 | **TimeMixer** | Mezcla multiescala y descomposición | Verificar escalas no vacías para D7 y la receta fuente |
| 9 | **PatchTST** | Atención sobre parches temporales | Fijar parches válidos, representación y salida escalar |
| 10 | **iTransformer** | Atención entre tokens de variables | Conservar causalidad por origen, escalado del objetivo y adaptación D7×5 |

**LSTM-autoencoder con predictor no equivale a la LSTM supervisada de E1.** Solo será un comparador distinto si la representación, el objetivo de reconstrucción y la conexión a la predicción quedan definidos y evaluables; cambiar el nombre de una LSTM no crea otro mecanismo.

La lista cubre convolución, recurrencia/autoencoder, MLP/mixing y atención. Debe cotejarse con las cuatro familias que exige la guía antes de afirmar cumplimiento. Las fichas históricas de los seis modelos conservados son un punto de partida, no validación automática de una nueva receta.

## 15.2 Alcance único del catálogo

Sólo los diez modelos anteriores y el candidato 11 forman la selección vigente. Los modelos de E1 —MLP, LSTM y GRU— son referencias separadas, no posiciones adicionales ni sustitutos de esta lista. Cualquier control diagnóstico del híbrido debe declararse como control, sin ampliar tácitamente el catálogo.

La selección se motiva por diversidad de mecanismos y cobertura de las familias del curso, no por resultados Bitcoin inexistentes. No se trasladan páginas leídas, puntuaciones, licencias, pruebas o presupuestos de otros modelos. Las métricas de clasificación o de otros datasets no demuestran precisión en σ₇.

## 15.3 Modelo 11: CNN-1D + LSTM + atención temporal, hipótesis por evaluar

Se propone estudiar **convoluciones 1D causales → LSTM unidireccional con `return_sequences=True` → atención temporal sobre siete estados observados → vector de contexto → cabeza escalar de σ₇**. La CNN extraería patrones locales, la LSTM representaría su secuencia y la atención los agregaría. Esa motivación no demuestra que combinar los tres mecanismos sea mejor que usar uno o dos.

El híbrido mantiene el [contrato sincronizado E1–E2–E3](17_contrato_sincronizado.md): **D7×5 derivado de OHLCV con `relative_window`, σ₇ poblacional y h=1**, sin anualización, nuevos canales, rezagos del objetivo o historia adicional. Las convoluciones conservan las siete posiciones con padding izquierdo cuando sea necesario; la atención sólo pondera estados de la ventana observada. No requiere la historia mensual ni los residuos fuera de muestra del prototipo HAR anterior.

**Estado: implementación provisional con pruebas sintéticas, sin entrenamiento de mercado ni superioridad demostrada.** No se afirma que esta combinación exacta reproduzca una receta del curso, un artículo identificado o una contribución original. Anchos, kernels, profundidad, función de atención, enlace e hiperparámetros definitivos siguen pendientes. Las cuatro redes compactas y sus diez pruebas compartidas con el corrector HAR son evidencia histórica separada; las pruebas nuevas del candidato se identifican en §18.

Antes del primer ajuste se deberán decidir y registrar:

1. La receta ejecutable de cada bloque, la cabeza de σ₇ y la pérdida, respetando el escalado del objetivo del contrato común y su inversión a unidades originales para evaluar.
2. Las pruebas de causalidad, direccionalidad recurrente, siete estados observados, ausencia de fuga y reinicio por fold; se verificará también la construcción de `relative_window`.
3. Las ablaciones **CNN sola**, **LSTM sola** y **CNN–LSTM sin atención**, con calendario e información comunes. Sus lecturas y diferencias de capacidad se fijarán explícitamente. Son controles, no posiciones nuevas del catálogo.
4. Un procedimiento de selección temporal y presupuesto trazable que contabilice el híbrido y sus ablaciones. No se darán oportunidades extra hasta obtener una ventaja.
5. Comparadores, mejora mínima relevante, incertidumbre, multiplicidad y coste, con posibilidad explícita de resultados nulos o desfavorables. No se cambiarán reglas tras conocer los resultados.

El test ya conocido de E1/E2 no recupera independencia con esta propuesta. Una comparación sobre él seguirá siendo exploratoria. Una afirmación confirmatoria de superioridad requeriría una evaluación no utilizada para escoger el híbrido, congelada después de cerrar el diseño y con sus límites temporales documentados.

### Fuentes verificadas y límites de las nuevas propuestas

Las [cuatro fichas bibliográficas ampliadas](16_fichas_nuevas.md) reúnen diagramas, bloques matemáticos, resultados publicados y costes analíticos, separados de cualquier adaptación E2. La fuente LSTM-SAE verificada acredita preentrenamiento seguido de ajuste supervisado, no pérdida conjunta.

La ampliación bibliográfica del 08/11/2026 distingue el **origen del mecanismo** de su validación para Bitcoin. No transfiere puntuaciones ni resultados de los cuatro modelos retirados a las altas.

| Candidato | Fuente primaria identificada | Evidencia y adaptación pendiente |
|---|---|---|
| FCN-1D | Wang, Yan y Oates, [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/abs/1611.06455) | FCN para clasificación temporal. La fuente no valida regresión de σ₇ en Bitcoin. Fijar kernels compatibles con D7, pooling, canales y cabeza escalar; justificar cambios respecto de clasificación |
| ResNet-1D | Wang, Yan y Oates, [misma fuente primaria](https://arxiv.org/abs/1611.06455) | Red residual para clasificación temporal; no equivale a la TCN histórica. Documentar bloques, padding, profundidad y salida de regresión; una fuente compartida no son dos estudios independientes |
| Inception-1D | Fawaz et al., [InceptionTime: Finding AlexNet for Time Series Classification](https://arxiv.org/abs/1909.04939) | Arquitectura/ensemble de clasificación. Decidir red individual frente a ensemble, kernels y bottlenecks para D7; su precisión clasificatoria no es una RMSE de volatilidad |
| LSTM-autoencoder con predictor | Sagheer y Kotb (2019), [LSTM-SAE](https://doi.org/10.1038/s41598-019-55320-6) | Fuente de preentrenamiento autoencoder y ajuste supervisado para forecasting, no Bitcoin. La variante exacta E2 sigue abierta; una pérdida conjunta de reconstrucción/predicción sería adaptación propia, no reproducción de la fuente |
| CNN-1D + LSTM + atención temporal, candidato activo | Fuente exacta de esta combinación y adaptación aún no verificada | Diseño propuesto, no réplica acreditada del curso o de la literatura. Requiere justificación por bloque, implementación y pruebas propias; no hay resultados E2 |
| HAR + red residual, antecedente inactivo | Lian et al., [Signal Simplification Is Not Predictive Simplification](https://arxiv.org/abs/2610.03019) | Antecedente del candidato anterior, no del nuevo flujo CNN/LSTM/atención. El preprint informa deterioro frente a HAR en cinco activos estadounidenses; se conserva como contexto histórico |

**Evidencia negativa del antecedente HAR, conservada como contexto.** En el estudio de Lian et al., el MSE medio pasa de 0,3049 para HAR-style a 0,3594 para HAR-style + LSTM; la LSTM sola obtiene 0,2649. Son valores del artículo en su propio objetivo y protocolo, no resultados E2 ni magnitudes comparables con σ₇. La menor varianza residual no basta para justificar una corrección neuronal. Esta observación no valida ni refuta el híbrido CNN/LSTM/atención. Los controles HAR solo/red sola/corrección pertenecían al diseño anterior y ya no son una obligación del candidato activo.

[HARNet (arXiv:2205.07719)](https://arxiv.org/abs/2205.07719) es un antecedente distinto: convoluciones causales dilatadas inicializadas para reproducir HAR, aplicadas a volatilidad realizada de índices bursátiles. No es sinónimo de una LSTM entrenada sobre residuos. Se conserva como referencia histórica: no es la fuente exacta del nuevo candidato, no se activa como alternativa ni se añade de forma tácita al catálogo.

La [actualización bibliográfica](03_literatura.md) conserva las consultas, la procedencia y los niveles de lectura de esta ronda. La fuente exacta del autoencoder, el código de cada adaptación y la correspondencia con las configuraciones del curso siguen siendo requisitos abiertos; estas fichas no acreditan implementación o entrenamiento.

## 15.4 Referencias de E1 y recibos históricos

MLP, LSTM y GRU de [E1](https://jassan-bit.github.io/DeepLearning_202630/intro.html) se identifican como referencias separadas. Los recibos previos de E2 conservan los nombres y recetas efectivamente usados; no se renombran ni se convierten en nuevas posiciones del catálogo. [Trazabilidad histórica](09_protocolo_registrado.md).

## 15.5 Orden de cierre y límites de esta actualización

- Resolver fuentes, configuraciones del curso, cuatro familias exigidas y justificación de las cuatro sustituciones.
- Aplicar el contrato sincronizado D7×5 `relative_window` → σ₇, h=1; concretar el flujo CNN/LSTM/atención, sus ablaciones y el total de configuraciones sin ampliar el catálogo.
- Diseñar una enmienda nueva con hipótesis, candidatos, presupuesto, validación temporal, potencia, incertidumbre y comparaciones. Conservar sin cambios el registro anterior y declarar qué información ya se observó.
- Implementar los adaptadores nuevos y realizar controles de forma, gradientes, invariancia de batch, causalidad y memorización sintética. Pasar pruebas no equivale a obtener resultados de mercado.
- Solo después, registrar la versión ejecutable, cerrar selección y evaluar según el nuevo plan. Presentar todos los resultados, incluidos los negativos.

**La revisión original de esta sección cambió únicamente documentación y navegación.** No se modifica el ejecutor registrado, no se alteran candidatos ni recibos experimentales, no se entrenan redes y no se publican originales, pesos, predicciones individuales o datos completos de evaluación.

El avance previo de [§4.0](04_implementacion.md) conserva cuatro prototipos compactos y el corrector HAR, con sus diez pruebas sintéticas, sin modificar el catálogo registrado ni acreditar entrenamiento de mercado. La sustitución inicial fue documental; posteriormente §18 añadió implementación y pruebas sintéticas del nuevo híbrido, sin entrenamiento de mercado. La tabla de diferencias allí evita confundirlos con las recetas bibliográficas de [§16](16_fichas_nuevas.md).
