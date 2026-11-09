# NAS: contraste independiente por arquitectura

Revisión de fuentes y código del árbol publicado en el commit 314ee0f. No se entrenaron ni ejecutaron modelos y no se modificaron resultados. Los dictámenes siguientes son propuestas para cerrar el diseño, no exenciones académicas concedidas ni preregistro.

## Conclusión que afecta al plan

La guía **no exige automáticamente 1.000 arquitecturas a los once modelos**, pero tampoco permite eximirlos por llamarse como un artículo. Su condición exige distinguir la arquitectura definida en la fuente de la receta efectivamente implementada. Los cuatro prototipos compactos actuales cambian cuerpos completos. Los otros seis parten de diseños publicados, con diferentes grados de adaptación a D7. El híbrido es una composición propia provisional, sin una fuente exacta verificada para su receta completa.

Los diez modelos seleccionados se conservan. Las alternativas metodológicas son restaurar y documentar un cuerpo publicado, justificar y adjudicar su adaptación, o aplicar la búsqueda de §14 a la arquitectura propia que corresponda. La guía no especifica el límite entre una adaptación legítima de tarea y un diseño nuevo. Esa incertidumbre debe quedar visible; no puede resolverse solamente mediante nombres, defaults de código o restricciones de cómputo.

## 1. Regla primaria de la guía

Fuente: PDF adjunto `Entregable2DeepLearningProject(3).pdf`, 65 páginas; las páginas siguientes son las impresas en el PDF, que coinciden con su numeración de páginas.

- **§14, p. 52:** «Cuando el paper de referencia no especifica la arquitectura de un modelo (o el modelo se define solo por su familia, por ejemplo “un MLP”, “una LSTM” o “una CNN”), la arquitectura no se elige por criterio personal». En la misma caja: «Si el paper sí define la arquitectura, se usa tal como está definida y esta sección no aplica a ese modelo».
- **§14.2, p. 53:** excluye de la búsqueda modelos con arquitectura definida, y da expresamente el ejemplo de «PatchTST con su configuración publicada». Conservan el presupuesto de ajuste de los demás hiperparámetros de §11.2. No significa que cualquier implementación denominada PatchTST esté exenta.
- **§14.1, p. 52:** tres bloques ocultos, anchos independientes {16,32,64,128,256}, ocho patrones de dropout de tasa fija 0,2: 125 × 8 = 1.000 arquitecturas por modelo afectado. Fase 1 exhaustiva, sin poda previa.
- **§14.2, p. 53:** si una familia no admite tres bloques naturalmente, se declara una correspondencia aproximada y se mantiene la cardinalidad. La regla para una base preentrenada fija y búsqueda solo en el cabezal no sirve para estos adaptadores entrenados desde cero: utilizar código de los autores no equivale a utilizar pesos preentrenados.
- **§14.3–14.4, p. 54:** HPO previo en referencia (64,64,64), patrón (0,1,0), usando validación; fijar los demás factores. Fase 1: todas las arquitecturas, una semilla común, presupuesto reducido declarado. Fase 2: k ≥ 10 más la referencia, al menos cinco semillas distintas de la inicial, presupuesto completo. Selección solo por validación. Confirmación posterior según protocolo final y evaluación una sola vez en test.
- **§14.4, p. 55:** mismo espacio, k y repeticiones por modelo sujeto a búsqueda; reportar y justificar desviaciones si faltan recursos. La falta de GPU o tiempo no constituye por sí sola una exención.
- **§14.5, p. 56:** regla de selección registrada y preferencia por parsimonia en empate técnico; no reemplazarla por un desempate lexicográfico de identificadores. Una ausencia de significación por sí sola no prueba equivalencia.

**Separación necesaria:** quedar fuera de NAS no elimina el HPO de §11.2, p. 47. La guía exige ajustar y reportar factores como learning rate, weight decay, dropout, batch size y factores estructurales pertinentes. Debe registrarse qué dimensiones son fijas por diseño publicado y cuáles se ajustan. La coexistencia de «usar tal como está definida» y el HPO estructural requiere una explicación de la receta, no una afirmación de réplica literal mientras se cambian bloques.

## 2. Matriz para integrar en el libro

Los estados siguientes son **dictámenes provisionales**, no concesiones académicas de exención. «Diseño explícito» se refiere a la evidencia de la fuente, no a cumplimiento del código actual.

| Modelo | Diseño en fuente primaria | Diferencia relevante del código D7 actual | Decisión defendible antes de entrenar |
|---|---|---|---|
| FCN-1D | Sí: Wang et al., §II.B, pp. 2–3 y Fig. 1b | Tres convoluciones de ancho 16 y kernels 3/3/1, sin BatchNorm; no el cuerpo 128/256/128, 8/5/3 de la fuente | Restaurar cuerpo publicado y justificar cabeza escalar, o tratar el compacto como diseño propio sujeto a §14; no eximir el compacto por su nombre |
| ResNet-1D | Sí: Wang et al., §II.C, p. 3 y Fig. 1c | Dos bloques de dos convoluciones, ancho 16, sin BatchNorm; fuente: tres bloques y nueve convoluciones principales | Misma bifurcación: cuerpo publicado documentado frente a diseño compacto propio; adjudicación pendiente |
| Inception-1D | Sí: Fawaz et al., §3.1–3.2, pp. 5–6 y código de autor | Un módulo, kernels 1/3/5 y ancho 16; sin las seis unidades, normalización y conexiones residuales del cuerpo fuente | Resolver red individual frente a ensemble y restaurar diseño o aplicar §14 al propio; la red actual no es InceptionTime publicado literal |
| LSTM-autoencoder con predictor | Código primario del autor sí contiene diseños explícitos; texto/tablas del paper no revalidados en esta revisión | Un encoder y un decoder de ancho 16 con proyección, frente a preentrenamiento apilado y preset de dos encoders 25/26 | Verificar el diseño del paper y elegir receta explícita, o reconocer diseño propio y aplicar §14; ni ausencia de diseño ni exención por paper demostradas todavía |
| TimesNet | Sí: §3 y Table 7, p. 14 del PDF | D7 se extiende a 8; se usan las cuatro frecuencias no DC en lugar del top-k original; cuerpo y dimensiones mezclan decisiones del paper/código | No basta decir «cabeza cambiada». Justificar/adjudicar modificación estructural; restaurar configuración compatible o incluir diseño adaptado en §14 si se considera nuevo |
| TSMixer | Sí: Fig. 1, p. 2; Appendix E, Table 8, p. 21; implementación de autor | Conserva seis bloques y feature MLP 512; cambia longitud, salida escalar, momentos y dropout | Candidato fuerte a cuerpo publicado más puente de tarea, sujeto a registro explícito; no es réplica exacta ni exención automática de toda la receta |
| TiDE | Sí: Table 8, p. 21; código primario de encoder/decoder | Covariables propias, sentinel futuro cero y decoder temporal de salida 8 en vez de 1, además de cabeza | Cambio estructural real que exige adjudicación; no presentarlo como mero reemplazo de readout |
| TimeMixer | Sí: §3; Table 7, p. 14; script ETTh1 de autor | Reduce tres downsamplings publicados a dos, dando longitudes 7→3→1; elimina proyección de variables y calendario | Justificar/adjudicar el cambio de escalas; fuente explícita no prueba que esta receta esté exenta |
| PatchTST | Sí: §4.1, p. 6; Appendix A.1.4, p. 14; es ejemplo explícito de la guía | Mantiene cuerpo ETTh pequeño, pero rellena D7 hasta 16 y crea dos patches; cabeza conjunta y momentos | Documentar geometría/padding y puente; exención condicional si se admite configuración publicada adaptada, no por nombre |
| iTransformer | Sí: Appendix A.2, Algorithm 1, p. 13; script ETTh1 de autor | Conserva encoder L=2, D=256, FF=256, H=8; entrada 7 y cabeza conjunta sustituyen forecast por variable | Candidato fuerte a cuerpo publicado más puente; registrar adaptaciones y adjudicar, sin llamarlo réplica literal |
| CNN-1D + LSTM + atención temporal | No se verificó una fuente primaria exacta para esta receta completa; esto no afirma que no exista literatura semejante | Composición propia CNN32→LSTM32→atención16, kernel3, una LSTM, cabeza escalar | Si se evalúa como arquitectura propia de E2, planificar §14 salvo justificación académica expresa. La guía E2 no resuelve por sí sola el tratamiento de una propuesta separada E3 |

Las referencias MLP/LSTM/GRU heredadas de E1 no se cuentan dentro de estos diez. Conservar sus resultados históricos no convierte automáticamente sus recetas heredadas en nuevos modelos sin diseño ni las incorpora silenciosamente a m. El alcance de cualquier búsqueda nueva sobre esas referencias debe definirse por separado.

## 3. Evidencia por modelo y cautelas de fuente

### FCN-1D

- Fuente primaria: Wang, Yan y Oates, [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/pdf/1611.06455), §II.B, PDF pp. 2–3, Fig. 1b. Define tres convoluciones, filtros 128/256/128, kernels 8/5/3, BatchNorm y ReLU, global average pooling y salida de clasificación.
- Código de autor: [FCN.py](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/FCN.py), construcción de red aproximadamente líneas 50–65. La URL `master` es mutable: fijar commit antes de reproducir.
- Actual: `scripts/e2_proposed_models.py`, `ConvBlock` y `ConvRegressor('FCN1D')`. No usa BatchNorm y cambia todos los anchos/kernels, además de Softplus de regresión. La diferencia no se reduce al cabezal. Un kernel 8 con padding SAME es implementable en D7; no hace falta reducirlo a 3 por una imposibilidad geométrica.

### ResNet-1D

- Mismo paper de Wang et al., §II.C, PDF p. 3 y Fig. 1c. Tres bloques residuales con nueve convoluciones principales, anchos 64/128/128. Código [ResNet.py](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/ResNet.py), `build_resnet`, aproximadamente líneas 13–89, especifica kernels 8/5/3, normalización y atajos.
- Actual: dos `ResidualBlock`, cada uno con dos convoluciones 3/3 y ancho común 16; sin normalización. Restaurar el cuerpo exige auditar también las proyecciones de atajo y fijar una versión de código, no copiar solo la etiqueta «ResNet».

### Inception-1D / InceptionTime

- Fuente primaria: Fawaz et al., [InceptionTime: Finding AlexNet for Time Series Classification](https://arxiv.org/pdf/1909.04939), §3.1–3.2, PDF pp. 5–6. Distingue la red con seis módulos y el ensemble de cinco redes. Código de autor [classifiers/inception.py](https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py), constructor y `_inception_module`, aproximadamente líneas 12–85: 32 filtros por rama, bottleneck 32, kernels 40/20/10, pooling, BatchNorm/ReLU y conexiones residuales cada tres módulos.
- Actual: un módulo, bottleneck 16, ramas 1/3/5 y rama pooling, sin BatchNorm ni conexiones residuales. Una red individual con el cuerpo publicado es una adaptación de alcance explícita, pero no equivale al ensemble completo. El coste del ensemble no puede desaparecer del presupuesto si se lo adopta.

### LSTM-autoencoder con predictor

- Fuente declarada: Sagheer y Kotb, [Unsupervised Pre-training of a Deep LSTM-based Stacked Autoencoder for Multivariate Time Series Forecasting Problems](https://doi.org/10.1038/s41598-019-55320-6). El cuerpo del artículo no quedó recuperado verificablemente en esta revisión: no se certifican sus tablas o páginas a partir de fichas secundarias.
- Fuente primaria accesible: [repositorio de los autores](https://github.com/2M-kotb/LSTM-based-Stacked-Autoencoder). [2layer_evaluate.py](https://github.com/2M-kotb/LSTM-based-Stacked-Autoencoder/blob/master/2layer_evaluate.py), aproximadamente líneas 124–140 y 187–242: preset de unidades 25/26, preentrenamiento por capas, transferencia de encoders, dropout, repetición temporal y predictor. [2layer_selection.py](https://github.com/2M-kotb/LSTM-based-Stacked-Autoencoder/blob/master/2layer_selection.py) busca otros hiperparámetros con anchos fijados. Hay variantes de una, dos y tres capas; no un único default universal.
- Actual: `PredictiveLSTMAE`, un encoder LSTM 5→16, decoder 16→16 más proyección a cinco variables y predictor 16→1. Cambia profundidad, anchos, decoder y regularización. La política de preentrenamiento/congelado no queda resuelta por que exista un forward.
- Pendiente preciso: validar directamente el diseño en el artículo, fijar commit del repositorio, escoger variante y mapearla a D7. La falta de lectura del paper no prueba que este carezca de diseño; el código explícito tampoco basta por sí solo para afirmar que cumple literalmente la condición «paper» de la guía.

### TimesNet

- Fuente primaria: [TimesNet](https://arxiv.org/pdf/2210.02186), v3, Table 7, PDF p. 14, y detalle dimensional p. 12. Publica k=5, dos capas y regla de d_model acotada entre 32 y 512 para forecasting. El código de autor está fijado en `thuml/Time-Series-Library@4e938a1767106324dd753b2a44832bf870a0252e`; [script ETTh1](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh) usa d_model=16 y d_ff=32. Paper y script no son una única fila idéntica.
- Actual: `_CanonicalTimesBlock` y `TimesNetAdapter` en `e2_final_models_mlp.py`, aproximadamente líneas 324–386. d_model=32, d_ff=32, dos bloques, seis kernels; extensión aprendida D7→8. En ocho puntos hay cuatro frecuencias no DC: se incluyen todas, con periodos 8/4/2/2 y pesos por amplitud individual. Se sustituye la selección top-k original. Mantener seis convoluciones mediante una fusión algebraicamente equivalente no es el cambio problemático; la selección de frecuencias y el puente de tarea sí requieren explicación.

### TSMixer

- Fuente primaria: [TSMixer](https://arxiv.org/pdf/2303.06053), v5, Fig. 1, PDF p. 2; Appendix E, Table 8, p. 21. La fila ETTh1/H96 publica seis bloques, feature hidden=512, ReLU y dropout=0,9. La misma tabla imprime un espacio de hidden widths que no incluye 512: conservar la discrepancia y el preset concreto, no corregir la fuente tácitamente.
- Código de autor: [tsmixer.py](https://github.com/google-research/google-research/blob/d3a4bd8f0eceddda553e47085033b537935b283f/tsmixer/tsmixer_basic/models/tsmixer.py), `res_block` y `build_model`, aproximadamente líneas 20–69. Define mezcla temporal y de canales, normalización prebloque, seis repeticiones mediante parámetro y proyección temporal final.
- Actual: `_MixerBlock`/`TSMixerAdapter`, aproximadamente líneas 167–191. Conserva el cuerpo de seis bloques y MLP de canales 5→512→5; tiempo 7→7, normalización compatible. Cambia readout por Linear(45,1), diez momentos de la misma ventana y Softplus. Dropout actual 0/0,1 no es 0,9 publicado. Los cambios de salida y regularización deben declararse aunque no se considere una arquitectura nueva.

### TiDE

- Fuente primaria: [TiDE](https://arxiv.org/pdf/2304.08424), v5, Table 7, PDF p. 20, y Table 8, p. 21. Fuente de implementación: [models.py](https://github.com/google-research/google-research/blob/d3a4bd8f0eceddda553e47085033b537935b283f/tide/models.py), `MLPResidual`, `_make_dnn_residual` y construcción del modelo; [train.py](https://github.com/google-research/google-research/blob/d3a4bd8f0eceddda553e47085033b537935b283f/tide/train.py), configuración `hidden_dims`; [ETTh1](https://github.com/google-research/google-research/blob/d3a4bd8f0eceddda553e47085033b537935b283f/tide/scripts/etth1.sh).
- La combinación hidden=256, decoder output=8 y temporal hidden=128 aparece en Table 8 y en las ejecuciones H336/H720 del script; no atribuirla a H96, cuyo script usa 512/32/16. `num_layers=2` genera `hidden_dims=[256,256]`: en este código son dos transformaciones Dense dentro de un bloque residual, no dos bloques residuales independientes.
- Actual: `TiDEAdapter`, aproximadamente líneas 208–249. Encoder/decoder 256, feature encoder 5→64→4, embedding de serie 16; covariables observadas derivadas de D7 y sentinel futuro cero, sin calendario ni valores futuros observados. El decoder temporal produce latent8, no salida1 de la fuente; evita el colapso de LayerNorm sobre un escalar. Sigue readout Linear(50,1) con momentos. Este ajuste cambia un módulo, no solo la cabeza.

### TimeMixer

- Fuente primaria: [TimeMixer](https://arxiv.org/pdf/2405.14616), v1, §3, PDF pp. 3–5, Table 7, p. 14. Publica para ETTh1 M=3, dos capas y d_model=16. [Script ETTh1 fijado](https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh) especifica d_ff=32, factor de pooling 2 y tres downsamplings. El artículo también usa M=1 para otro régimen de tareas; no hay un M universal.
- Actual: `TimeMixerAdapter`, aproximadamente líneas 252–280. Conserva PDM/FMM y los valores L=2, D=16, FF=32, pero fija dos downsamplings: 7→3→1. Un tercero produciría longitud cero con este pooling. Elimina la proyección por variable, componentes temporales y una LayerNorm upstream registrada pero no utilizada; incorpora cabeza escalar con momentos. La justificación geométrica es real, pero no equivale automáticamente a aprobación académica de la exención.

### PatchTST

- Fuente primaria: [PatchTST](https://arxiv.org/pdf/2211.14730), v2, §4.1, PDF p. 6; Appendix A.1.4, p. 14. Define patch16, stride8 y contextos largos; para ETTh/ILI usa configuración pequeña de tres capas, cuatro cabezas, d_model16, FF128. El paper indica dropout0,2; [script ETTh1 de autor](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised/scripts/PatchTST/etth1.sh) usa dropout0,3 y contexto336. Mantener diferenciadas ambas fuentes.
- Actual: `PatchTSTAdapter`, aproximadamente líneas 389–413. Conserva el encoder pequeño, RevIN, BatchNorm y atención residual. Normaliza los siete valores reales, replica nueve a la izquierda hasta16 y ocho a la derecha, generando dos patches de16. El padding no agrega información; sí cambia la geometría respecto al experimento fuente. Sustituye el forecast por variable por cabeza conjunta Linear(170,1), con momentos. El ejemplo de p. 53 de la guía solo prueba que una configuración publicada puede quedar exenta, no valida esta adaptación por sí mismo.

### iTransformer

- Fuente primaria: [iTransformer](https://arxiv.org/pdf/2310.06625), v4, Appendix A.2, Algorithm 1, PDF p. 13, y configuración p. 14. Define tokens de variable y encoder invertido; permite varias capas y dimensiones. [Script ETTh1 fijado](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh), preset H96: dos capas, D256 y FF256; ocho cabezas en `run.py` del mismo commit. No atribuir ese preset a todos los horizontes: los largos usan otras dimensiones.
- Actual: `e2_final_models_attention.py`, construcción del encoder aproximadamente líneas 284–295. L=2, D256, FF256, H8; cambia la longitud de embedding a7 y elimina la proyección/denormalización por variable, usando cabeza conjunta Linear(1290,1), con diez momentos. Conserva el cuerpo del encoder con un puente de tarea explícito. Eso respalda una vía de adaptación de fuente, no una afirmación de réplica integral.

### Híbrido CNN-1D + LSTM + atención temporal

- Fuente exacta para esta receta: **pendiente**. La composición es habitual en la literatura, pero citar fuentes independientes de CNN, LSTM y atención no define por sí solo esta arquitectura completa ni sus anchos.
- Actual: `e2_current_models.py`, `CausalCNNLSTMAttention`, aproximadamente líneas 175–231. CNN causal kernel3 y ancho32 → LSTM unidireccional de una capa y ancho32 → proyección aditiva de atención16 → suma ponderada de siete estados → dropout0,1 → Linear y Softplus. El smoke usa anchos8 y dropout0; ninguno selecciona una receta de mercado. La atención ve solo los siete días ya observados; eso no acredita por sí solo la causalidad del preprocesamiento completo.
- Para aplicar §14 podría declararse una correspondencia aproximada de tres bloques CNN/LSTM/atención y variar (w1,w2,w3), con ubicaciones de dropout inequívocas. **Es una propuesta por adjudicar**, no una traducción que la guía ya apruebe: hay que definir si ancho del tercer bloque controla la proyección de atención, dónde se inserta dropout y cómo se mantienen mil arquitecturas distintas. Las ablaciones deben mantener comparabilidad y contarse en presupuesto aparte.
- No retirar ni reducir el candidato por esta incertidumbre. Si continúa como candidato E2, registrar su clasificación antes de buscar; si se estudia separadamente en E3, consultar la guía E3 todavía pendiente, sin importar una exención desde ella.

## 4. Trazabilidad técnica revisada

Se contrastaron las fuentes de cuatro repositorios de autor disponibles localmente con los hashes registrados en `book_e2/_static/current_implementation_20261108/recipes.json`: TimesNet 34/34 archivos, TimeMixer 22/22, PatchTST 6/6 e iTransformer 8/8 coinciden. Esto valida los subconjuntos declarados, no la totalidad de esos repositorios ni la equivalencia de las adaptaciones. TSMixer/TiDE se contrastaron con archivos primarios fijados de Google Research. Los repositorios de los cuatro clásicos todavía requieren fijar commit para una futura reproducción.

Los seis adaptadores añaden diez momentos de la misma ventana (cinco medias y cinco desviaciones estándar) a su cabeza. Son transformaciones de D7, no nuevos días ni observaciones futuras, pero modifican el modelo y deben figurar en la receta. Ninguno usa pesos de autor preentrenados. Los cuatro compactos y el híbrido no llevan esos diez momentos. No atribuir a los once una arquitectura o cabeza idéntica por compartir contrato de entrada y salida.

## 5. Escenarios de coste, no compromiso de NAS

Definir m después de adjudicar cada receta. La fórmula de la guía, p. 55, es `m × [1000 × t_corta + (k+1) × R2 × t_completa]`. Para evitar duplicar la referencia si ya está entre las k mejores, puede escribirse `k+c`, con c=0 en ese caso y c=1 si se añade. Registrar los IDs únicos. R2≥5 corresponde a la confirmación de NAS y no sustituye al número R del protocolo final.

Si la búsqueda completa se repite en F folds/orígenes, el coste uniforme es `m × F × [1000 × t_corta + (k+c) × R2 × t_completa]`. Si cada modelo cuesta distinto, usar la suma sobre modelos y repeticiones, no multiplicar por un tiempo supuestamente común. No multiplicar por folds que ya estén incluidos dentro de la trayectoria cronometrada.

| Escenario hipotético | Qué tendría que haberse decidido | Cortas con F=1 | Completas con k=10, c=1, R2=5 |
|---|---|---:|---:|
| m=0 | Los diez usan diseños fuente aceptados; híbrido con diseño aceptado o fuera de esta búsqueda por una decisión expresa | 0 | 0 |
| m=1 | Diez recetas de fuente/adaptaciones aceptadas y solo el híbrido entra en §14 | 1000 | 55 |
| m=4 | Solo los cuatro compactos entran; el tratamiento del híbrido y los otros seis queda resuelto expresamente | 4000 | 220 |
| m=5 | Cuatro compactos más híbrido entran; otros seis aceptados como diseños de fuente adaptados | 5000 | 275 |
| m=11 | Sensibilidad máxima: las once recetas se tratan como diseños propios sujetos a §14 | 11000 | 605 |

No se ha seleccionado ninguno de esos escenarios. m=11 no es una obligación demostrada de la guía. m=0 no significa coste cero de investigación: seguirían HPO, pilotos, entrenamiento final, controles, ablaciones y análisis. Podrían surgir valores intermedios según la decisión sobre TimesNet, TiDE, TimeMixer o PatchTST.

Una trayectoria debe abarcar el pipeline completo: preentrenamiento y fine-tuning del autoencoder, miembros de un ensemble si lo define la receta, validación interna y refits que correspondan. HPO previo, pilotos, confirmación final, controles y ablaciones no están incluidos en los números anteriores. Medir t_corta/t_completa en hardware identificado antes de publicar horas, con política de early stopping y paralelismo declarada. Este informe no inventa ni extrapola horas a partir de un smoke.

La calculadora aritmética `scripts/e2_nas_cost_scenarios.py`, incluida en el repositorio, expresa esos conteos sin cargar torch ni datos. Para k=10 y la referencia ya seleccionada, c=0 reduce a 50 trayectorias completas por modelo. La diferencia es contabilidad de diseños únicos, no poda del espacio.

## 6. Decisiones que deben quedar abiertas hasta su cierre

1. Fijar una receta fuente o adaptada por modelo, con commit, dimensiones, cuerpo y cabeza claramente separados.
2. Clasificar cada cambio como de tarea/interfaz, regularización/HPO o cambio estructural; pedir adjudicación cuando la guía no resuelva su admisibilidad. No convertir una inferencia del equipo en una exención concedida.
3. Registrar por modelo si §14 aplica y por qué; si aplica, definir la traducción a tres bloques y comprobar la cardinalidad de mil diseños únicos antes de entrenar.
4. Fijar m, k, R2, fases, referencia, folds/orígenes, presupuesto y selección exclusivamente con validación; mantener test y resultados históricos intactos.
5. Si falta presupuesto, reportar la desviación y sus consecuencias. Un grid de cuatro learning rates/weight decays no sustituye las mil arquitecturas ni completa el HPO mínimo.
6. No anunciar que se han entrenado los diez más híbrido, ni que la NAS está realizada: la evidencia vigente es implementación provisional y verificación sintética.
