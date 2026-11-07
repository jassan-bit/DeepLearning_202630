# Mecanismos, complejidad y selección de E1 · 07/11/2026

Este expediente documenta una revisión posterior de la terna **GRU–LSTM–TCN**. Lee código y metadatos existentes; no construye redes, no ajusta modelos y no cambia el libro, los resultados ni la identificación histórica de los experimentos. `mechanisms.json` contiene las ecuaciones, el desglose de parámetros, las recetas originales, las limitaciones y los hashes de los insumos. Integra los conteos homogéneos de [part_corpus](../part_corpus/README.md); los siete estudios históricos no se presentan como universo final.

**Recomendación:** conservar la terna como registro de un contraste específico entre mecanismos recurrentes y filtros temporales. Sus diferencias estructurales se pueden justificar; el cumplimiento conjunto de §3.2 permanece condicionado. La evidencia actual no permite presentarla como las tres familias más frecuentes ni como la terna de mínima complejidad. MLP es una alternativa referencial que debe conservarse en la evaluación.

## Frecuencia homogénea y sensibilidad

El estrato principal para comparar referencias contiene **10 experimentos con al menos un comparador neuronal autónomo extraíble**. El universo de constructo continuo realizado/histórico contiene 12, incluidas dos fuentes sin papel de referencia simple confirmado. La frecuencia cuenta una familia una vez por experimento independiente, sin añadir cabezas internas, propuestas o la reutilización de los pronósticos Dudek 2024 por Dudek 2025.

| Familia | Referencias extraíbles N=10 | Todo el constructo N=12 | Sensibilidad de reporte N=8 |
|---|---:|---:|---:|
| LSTM | 10/10 | 10/12 | 8/8 |
| GRU | 4/10 | 4/12 | 2/8 |
| MLP | 3/10 | 3/12 | 2/8 |
| TCN | 1/10 | 1/12 | 1/8 |
| CNN genérica | 1/10 | 1/12 | 0/8 |
| Encoder Transformer | 1/10 | 1/12 | 1/8 |

La sensibilidad N=8 excluye **juntas** S8 y PSO-GRU por insuficiencia de reporte: no penaliza sólo las incorporaciones nuevas. En publicaciones identificadas de constructo realizado/histórico, N=7, los numeradores son LSTM 7, GRU 3, MLP 3, CNN 1 y TCN 1; las grises, N=5, aportan LSTM 3, GRU 1 y encoder 1. La sensibilidad inclusiva de objetivos no reconciliados, N=16, produce LSTM 12, GRU 6, MLP 3, encoder 3, TCN 1 y CNN 1, además de las familias de CryptoMamba. Estos estratos no se intercambian ni son estimaciones de prevalencia global.

Ampliar el problema para incluir volatilidad implícita cripto añade CVI: N=13, con LSTM 11, GRU 4, MLP 3 y TCN 2. La variante TCN de CVI tiene cabeza/parámetros compartidos; no se afirma que sea una TCN pura o mínima ni que CVI equivalga a sigma7. MLP sigue teniendo más respaldo referencial que TCN por este conteo.

El estrato de desviación estándar/log-desviación tiene N=6, LSTM 5, GRU 2, TCN 1 y encoder 1; el de varianza también tiene N=6, LSTM 5, GRU 2, MLP 3 y CNN 1. La escala del objetivo es una sensibilidad pertinente, pero **no se adopta ahora como filtro para eliminar MLP**: ninguno de esos estratos coincide exactamente con sigma7 diario de E1. La selección principal por frecuencia sería **LSTM–GRU–MLP**, con las limitaciones de receta y calidad de cada fuente todavía visibles.

## Qué exige la no redundancia

La guía, §3.2, paso 6, página física 14, dice: «Verificar que los tres modelos no sean redundantes entre sí (deben diferir en el sesgo inductivo que incorporan)». **No exige literalmente tres familias amplias distintas**, ni define un umbral cuantitativo de diversidad. Sus familias orientativas incluyen LSTM y GRU, y §3.3 las enumera como tipos de arquitectura. Eso permite fundamentar una distinción entre mecanismos de estado dentro de la recurrencia; no convierte cualquier variación de ancho, semilla o número de compuertas en una justificación suficiente.

GRU y LSTM comparten un sesgo importante: procesan la ventana mediante una transición recurrente con pesos compartidos entre posiciones y compuertas dependientes de la entrada. Difieren en su parametrización funcional:

- **GRU:** el reset modula la contribución recurrente al candidato; la actualización interpola candidato y estado anterior. No hay una celda separada ni una compuerta independiente de salida.
- **LSTM:** la celda aditiva separada combina retención y escritura mediante olvido y entrada; otra compuerta controla su emisión al estado oculto. La retención interna y la salida pueden regularse separadamente.
- **TCN:** compone filtros temporales causales de pesos compartidos, con dilataciones y conexiones residuales. No actualiza un estado recurrente entre instantes ni entre ventanas.

Por tanto, existe una **justificación mecanística acotada** de diferencias por pares: dos mecanismos recurrentes y uno convolucional. Son **dos familias amplias**, y una terna con MLP, una sola recurrente y TCN cubriría tres familias amplias. La guía no impone esa segunda interpretación como requisito literal. Tampoco proporciona una métrica que permita demostrar que el grado de diferencia GRU–LSTM es suficiente con independencia de los demás criterios. No se demuestra inclusión estricta entre clases funcionales, independencia de errores, complementariedad predictiva o un óptimo único de diversidad.

## El contrato observado es D7 para las tres

Los `forward` de [GRU](../../../../scripts/dl_models_gru_lstm_tcn.py) y [LSTM](../../../../scripts/dl_models.py) llaman al encoder sin pasar un estado inicial. PyTorch usa ceros, y el código no guarda ni transporta estado entre ventanas. **Las tres redes reciben siete observaciones reales de las mismas cinco covariables OHLCV.** La memoria de celda LSTM opera dentro de esa ventana; no aporta información previa a ella.

La TCN tiene campo receptivo teórico

\[
R=1+2(k-1)(1+2+4)=29,
\]

pero con D7 las posiciones anteriores a la ventana son padding izquierdo. No recibe 29 días reales. El contraste entre recurrencia y convolución se sostiene por sus reglas de representación, no por una diferencia de información disponible.

El paralelismo es una propiedad computacional: las convoluciones pueden procesar posiciones en paralelo y las transiciones recurrentes dependen del estado precedente. Eso no demuestra por sí solo diversidad de sesgo, menor tiempo medido o menor error. Igualmente, el número de parámetros, la profundidad, el campo receptivo y la coincidencia entre la tarea de una fuente y E1 son atributos distintos; no se suman como si fueran pruebas intercambiables de no redundancia.

## Capacidad exacta de las recetas propias de fase II

El catálogo `results/gru_lstm_tcn_planned_20261105/final_configurations.csv` contiene 465 filas finales por familia y un único ancho/conteo en cada una. Se contrastaron esas filas con la definición de capas y con sumas algebraicas, sin importar el framework ni ejecutar redes.

| Receta final | Estructura propia | Parámetros entrenables |
|---|---|---:|
| GRU c03 | Una capa unidireccional, H=16; dropout 0,1; cabeza escalar | 1.121 |
| LSTM c02 | Una capa unidireccional, H=16; dropout 0; cabeza escalar | 1.489 |
| TCN c02 | Tres bloques, seis convoluciones k=3, F=8, dilataciones 1/2/4; proyección inicial 5→8; dropout 0 | 1.185 |

Con cinco atributos y dos vectores de sesgo por proyección recurrente en PyTorch:

\[
P_{GRU}=3H(p+H+2)+(H+1)=1.121,
\]
\[
P_{LSTM}=4H(p+H+2)+(H+1)=1.489.
\]

Las seis convoluciones, la proyección residual inicial y la cabeza de la TCN suman:

\[
P_{TCN}=5kF^2+(kp+p+8)F+1=1.185,
\]

para tres bloques de dos convoluciones, sesgos activados, p=5≠F=8 y sin normalización entrenable. Su orden exacto por bloque es:

```text
a = dropout(ReLU(conv1_causal(z)))
b = dropout(ReLU(conv2_causal(a)))
salida = ReLU(b + proyección(z))
```

La proyección es identidad salvo en el primer bloque. Dropout, ReLU y Softplus no añaden parámetros. El desglose está en el JSON. Una capa recurrente y seis convoluciones no son profundidades directamente equivalentes: las recurrentes se despliegan sobre D posiciones. Tener tamaños cercanos tampoco prueba igual capacidad representacional.

Estos son los tamaños de **fase II**. Las recetas de fase I siguen siendo H=32, H=32 y F=16, con 3.777, 5.025 y 4.289 parámetros. Ninguna cifra se atribuye retrospectivamente a otra fase.

## Qué está publicado y qué se adaptó

| Fuente | Receta recuperada | Qué permite afirmar |
|---|---|---|
| S1, Akbiyik et al. (2023), Apéndice C, tablas 5–6 | LSTM H=261; GRU H=43; TCN F=287, k=5 y seis niveles de dilatación, base 4. Ganadores y presupuesto Optuna publicados. | Son comparadores autónomos del mismo estudio. La compactación propia 16/16/8 no es su réplica ni la recuperación de datos omitidos por el artículo. |
| S8, Miura et al. (2019), pp. 168–171 | MLP «4 capas», dos variantes; CNN «2 capas + Dense»; LSTM «2 capas + Dense»; GRU «1 capa + 2 Dense». | Confirma esas referencias autónomas. Faltan anchos y otros detalles; no hay totales exactos. No se sabe si las cuatro capas MLP incluyen salida ni se acredita dimensión/causalidad de CNN. |
| S2, Zeri (2026), texto y notebook fijados | Transformer de dos encoders, d=48, cuatro cabezas, FF=96, 96 posiciones aprendidas, pooling último+media y cabeza 96→48→1. | Conteo analítico de esa receta: 49.249, no cifra de parámetros reportada por el autor. La condición de manuscrito gris se conserva. |
| S6, Wang et al. (s. f.), §3.4 y tablas 1–2 | Transformer de dos encoders, d=32, cuatro cabezas y dropout 0,1. | Faltan FFN, lectura, posiciones y orden de normalización: total desconocido. No se rellenan con defaults. Su objetivo operativo no está reconciliado; permanece en sensibilidad del corpus. |
| E1OA-R02, Dudek et al. (2024), §2.6 p. 5 | MLP con tres entradas, una capa oculta tanh y siete salidas lineales; Levenberg–Marquardt con regularización bayesiana. | Apoya una referencia densa de una capa oculta. El ancho seleccionado sigue ausente; no hay total exacto. |
| E1X001, Brauneis y Sahiner (2026; online 2024), cuerpo completo | MLP con una capa oculta sigmoid y salida lineal; optimización bayesiana de recetas, TensorFlow 2. | Otro comparador autónomo simple; ancho final ausente. Sus resultados no prueban superioridad universal ni equivalencia con sigma7. |

El JSON registra URL, ubicación, hiperparámetros y supuestos, con hashes del original cuando fue materializado y de la ficha cuando sólo se conserva la revisión primaria. Los datos Dudek/Brauneis proceden de esa revisión guardada; no se declara una nueva inspección remota de sus originales. Para S1, los totales **derivados** son 299.724 para LSTM bajo un sesgo Keras; GRU 10.158 con dos sesgos o 10.029 con uno; TCN 4.563.396 bajo supuestos explícitos de la versión de `keras-tcn` y sus bloques. Estos últimos conteos no se convierten en hechos de implementación cuya versión no está fijada. S1 usa p=1, q=96 y D=96; E1 usa p=5, q=1 y D=7. Comparar sin más sus totales con los propios no compara igual tarea ni igual receta.

El total 47.521 usado anteriormente para Transformer es un **escenario algebraico** que cambia p=41 por p=5 y mantiene d, profundidad y capacidad posicional de S2. No es el total original, una receta mínima o un modelo ejecutado. Un encoder más pequeño sigue siendo posible; la atención cuadrática no basta para declararlo inviable con D7.

Existe además una MLP propia histórica, anterior a las fases actuales, con una capa oculta ReLU H=16, D7 y cinco atributos: `H(Dp+2)+1=593` parámetros. Consta en `results/dl_benchmark_20261104/selected_configuration.json` y en código. Su plantilla y grid de anchos 16/32, tasas 0,001/0,003 y semillas 101/202 están en `experiments/dl_benchmark_20261104/protocol_initial.json`, registrado a las 14:28:17 UTC del 04/11/2026 y descrito como previo a sus ajustes. La configuración seleccionada consta a las 14:37:30. Son metadatos y firma internos, no acreditación externa de ese registro.

Por tanto, los 593 parámetros corresponden a una **receta histórica existente con búsqueda declarada antes de sus ajustes**, no sólo a una posibilidad conceptual. No se añade a la terna ni se cambia su etiqueta, y no se usan sus resultados para elegir una sustitución. Su compatibilidad exacta de preprocesamiento, fitting y presupuesto con las fases actuales se verifica aparte; no se presupone.

Un filtro «receta publicada suficiente **o** HPO propio registrado antes de ajustar» no permite eliminar MLP por falta de receta: TCN tiene S1 y HPO propio, y MLP tiene la plantilla/grid propio anterior. Exigir adicionalmente haber sido ajustada en fase II convertiría la disponibilidad posterior de resultados en un filtro selectivo. La guía pide capacidad de compartir pipeline y partición, no pertenecer ya a los tres modelos ajustados. Tampoco se acredita aquí el cumplimiento íntegro de la búsqueda E2 por esos procedimientos históricos; la regla debe comprobarse simétricamente. Una nueva adaptación incompletamente especificada requiere declarar su espacio de arquitectura antes de nuevos ajustes.

En el empate TCN–encoder de 1/10, la complejidad tampoco proporciona una victoria clara a TCN. Comparar sus 1.185 pesos propios con los 49.249 del encoder original sería asimétrico. Las recetas originales arrojan TCN ≈4,56 millones, condicional al bloque/dependencia, frente a encoder 49.249; cambian p/q y la tarea. El escenario algebraico p=5, q=1, manteniendo los anchos/profundidades originales, da TCN 4.542.924 y encoder 47.521. No son compactaciones mínimas ni modelos ejecutados. Estos datos no justifican que toda TCN sea más simple que todo encoder, y no se inventa una receta de atención para hacerlo perder.

## Regla de decisión sin pesos hechos para la terna

La guía pide considerar conjuntamente recurrencia, complejidad y sesgo; no fija pesos ni un desempate universal. Se propone una comparación parcial y transparente:

1. Aplicar las mismas reglas de inclusión, objetivo operativo y papel comparativo a todos los originales. Contar una familia una vez por experimento independiente; conservar componentes y propuestas aparte. Publicar los denominadores y las sensibilidades.
2. Comparar frecuencia dentro del mismo estrato, junto con parámetros, profundidad, receta recuperada y mecanismo. Un valor desconocido sigue siendo desconocido: no es cero, ausencia ni prueba de mayor complejidad.
3. Mantener separados receta original, escenario analítico de entrada común y receta propia. La posibilidad de compactar se reconoce para todas las familias; el número de pesos de una compactación no demuestra simplicidad universal.
4. Examinar diferencias por pares y cobertura de familias de cada terna. Declarar una dominancia sólo si todos los atributos relevantes son conocidos y comparables, la alternativa es al menos igual en todos y mejor en alguno. En otro caso, indicar intercambio o relación indeterminada.
5. Si persisten alternativas, explicitar qué prioridad conduce a cada elección. Una prioridad adoptada hoy es posterior al experimento. No debe presentarse como prerregistro ni seleccionarse usando la reserva observada.

La diversidad tampoco tiene una escala ordinal definida. Hasta declarar una relación cualitativa entre ternas, ese atributo vuelve indeterminada una afirmación de dominancia. Con parámetros publicados faltantes y tareas de distinta dimensión no se ha calculado una frontera Pareto completa. Hay alternativas plausibles que no se pueden declarar dominadas:

| Terna ilustrativa | Beneficio | Coste o límite |
|---|---|---|
| GRU–LSTM–TCN actual | Contraste entre dos reglas de estado recurrente y una representación convolucional; recetas propias explícitas. | Dos puestos comparten recurrencia y el respaldo referencial de TCN es limitado. No prueba mayor frecuencia ni mínima complejidad global. |
| LSTM–GRU–MLP | Las tres mayores frecuencias homogéneas: 10/10, 4/10 y 3/10; añade mezcla densa específica por posición. | No incluye convolución; anchos publicados MLP incompletos. Existe receta/HPO histórico propio, sin igualdad presupuesta con fase II. |
| MLP–LSTM–TCN | Cubre tres familias amplias; GRU podría ocupar el puesto recurrente bajo otra prioridad explícita. | Sacrifica apoyo de una recurrente y no garantiza menor complejidad o error. |
| LSTM–GRU–Transformer | Aporta atención; era el orden por frecuencia de la tabla histórica al incluir S6. | En el corpus principal, encoder 1/10 empata TCN y queda por debajo de MLP 3/10. S2 no es receta mínima y S6 es parcial. |

La terna actual conserva un sentido experimental claro. Eso **no certifica por sí solo el cumplimiento conjunto de §3.2**. Las diferencias de mecanismos sostienen una interpretación acotada de no redundancia; la matriz homogénea reconoce MLP y no demuestra que TCN deba ocupar su lugar por frecuencia o mínima complejidad. Una prioridad específica por contrastar celdas recurrentes y filtros causales puede explicar conservar el estudio, con ese alcance explícito. No convierte la terna en top3 ni reconstruye una selección prospectiva. La corrección documental es reconocer ese intercambio y conservar las alternativas; una sustitución requiere verificar compatibilidad y procedencia de un benchmark ya existente o diseñar una comparación futura equivalente, sin nuevos ajustes en este expediente.

La equivalencia con regresión logística o SVR lineal que pide la guía es de **propósito comparativo**: referencias reproducibles de complejidad controlada para medir la contribución de un avanzado. No es una equivalencia formal de capacidad, supuestos o clases funcionales. Los modelos clásicos conservan su papel de contexto.
