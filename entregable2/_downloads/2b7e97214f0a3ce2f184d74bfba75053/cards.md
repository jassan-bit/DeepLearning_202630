# 22. Once fichas de recetas propuestas

**Estado: propuestas concretas para adjudicar el diseño, no cambios de implementación ni entrenamiento.** Este corte no modifica los scripts de modelos, no ejecuta forwards o ajustes y no concede exenciones NAS. Las pruebas sintéticas de §18 pertenecen a aquellas versiones, no certifican los nuevos cuerpos propuestos aquí. El catálogo de diez modelos y el candidato CNN-LSTM-atención se conserva.

[Once fichas estructuradas](../_static/recipe_cards_20261109/cards.json) · {download}`Documento de especificaciones <../_static/recipe_cards_20261109/cards.md>`.

## Contrato común de las fichas

- Entrada X=[B,7,5], siete barras OHLCV t−7…t−1, con `relative_window` y escaladores del entrenamiento de cada fold según §17. La transformación se fija antes de validación; medias y DE sobre N×7 posiciones, sin datos posteriores.
- Objetivo σ₇ poblacional, h=1, sin anualización. Entrenar sobre y/a, donde a es la media positiva de las etiquetas de entrenamiento; la salida Softplus es escalar y se multiplica por a para evaluar en unidades originales. No centrar y ni inferir una transformación diferente de la fuente.
- Las fuentes de clasificación no validan esta tarea. D7, cinco canales y cabeza escalar se declaran adaptaciones. Padding en una ventana cerrada conserva causalidad respecto del día pronosticado; no garantiza causalidad de cada prefijo interno ni añade días observados.
- Las tres redes convolucionales propuestas y el SAE no agregan momentos; los seis adaptadores mantienen diez momentos de la D7 escalada en la cabeza. El híbrido no los agrega. Esta asimetría requiere control y no acredita comparación de backbones puros.
- Las dimensiones y componentes que siguen fijan una **propuesta de arquitectura**, no una receta ganadora. El HPO mínimo, épocas, optimizador, presupuesto, semillas finales y condiciones NAS deben resolverse y registrarse conforme a §§20–21. Dropout inicial es un punto de partida explícito, no óptimo validado.
- No se descargan pesos preentrenados. Cuando se indica preentrenamiento SAE, se propone reconstrucción exclusivamente sobre entrenamiento propio del fold. Las licencias y los archivos exactos deben comprobarse antes de reutilización de código.

**Cómo leer el estado NAS:** “adjudicación pendiente” significa que se identificó exactamente la duda entre diseño publicado y adaptación. No es un permiso concedido ni obliga automáticamente a 1.000 búsquedas por conservar el nombre. Restaurar un cuerpo publicado abre una vía defendible de diseño definido; todavía deben aceptarse la adaptación D7 y la cabeza.

## Convenciones específicas de las cuatro recetas reconstruidas

- Entrada externa: `(B,7,5)`, siete días cerrados y cinco canales del contrato `relative_window` vigente; sin rezagos adicionales de la etiqueta, calendario ni datos futuros. Usar las transformaciones comunes ajustadas exclusivamente con el entrenamiento de cada fold. No añadir los diez momentos utilizados por otros adaptadores.
- Salida: Dense con un escalar y Softplus para predecir y/a, donde a es la media positiva de las etiquetas de entrenamiento del fold, sin centrar. Pronóstico en unidades originales: a×Softplus(z). Si a no es válida y positiva, registrar fallo del contrato; no sustituirla usando validación/test. No utilizar la denormalización de OHLCV para σ₇. Cuerpo, adaptación de tarea y ajustes de entrenamiento se registran separadamente.
- Las convoluciones usan stride 1 y dilatación 1. Para kernel temporal k, padding SAME de ceros: izquierda `floor((k−1)/2)` y derecha `ceil((k−1)/2)`. Así, kernel 8 usa 3/4, kernel 5 usa 2/2 y kernel 3 usa 1/1. No se añaden días observados. La convolución no es causal posición a posición, pero toda la ventana está disponible antes del pronóstico; no describirla como convolución causal.
- BatchNorm por canal, con estadísticas del entrenamiento; validación y prueba en modo evaluación sin actualizaciones. Propuesta explícita: epsilon 0,001, gamma inicial1, beta0 y actualización de estadísticas móviles `0,99×anterior+0,01×batch`. Inicialización Glorot uniforme para convoluciones y Dense, biases cero. Fijar versión y convención de varianza al implementar; no asumir equivalencia bit a bit entre Keras y PyTorch por compartir un nombre.
- Se propone conservar cero dropout en los tres cuerpos CNN fuente. El dropout 0,3 del SAE se coloca como se especifica abajo. El HPO de §11.2 sigue pendiente: estos valores son una receta de referencia, no una búsqueda completada ni permiso para omitir factores requeridos.
- Los SHA enlazados son snapshots verificados del código de autor, no certificaciones del código exacto utilizado en la publicación. La traducción Conv2D con dimensión singleton a Conv1D se declara; no se promete cargar checkpoints originales como si sus tensores fueran idénticos.

## 22.1 FCN-1D con cuerpo publicado y cabeza de regresión

**ID de receta:** `fcn1d_wang_d7_v1_proposed`.

**Decisión propuesta:** conservar la FCN de tres bloques de Wang et al.; reemplazar solo la interfaz de entrada, su traducción 1D y la cabeza de clasificación.

**Fuente:** Wang, Yan y Oates, [Time Series Classification from Scratch with Deep Neural Networks](https://arxiv.org/pdf/1611.06455), §II.B, pp. 2–3 y Fig. 1b. Código [FCN.py, construcción de red](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/3ecc8971fa768bd01ada7f3a613688255e6256c2/FCN.py#L50-L83), SHA `3ecc8971fa768bd01ada7f3a613688255e6256c2`.

**Receta E2 propuesta:**

1. Conv1D 5→128, kernel 8, bias; BatchNorm128; ReLU. Forma `(B,7,128)`.
2. Conv1D 128→256, kernel 5, bias; BatchNorm256; ReLU. Forma `(B,7,256)`.
3. Conv1D 256→128, kernel 3, bias; BatchNorm128; ReLU. Forma `(B,7,128)`.
4. Media sobre las siete posiciones → `(B,128)`; Dense128→1 con bias → Softplus → `(B,)`.

Sin pooling intermedio, capas adicionales, canales estadísticos, ensemble ni pesos preentrenados. Los filtros, kernels, normalización y activación proceden del cuerpo publicado; el objetivo σ₇, entrada multivariada D7 y enlace de salida son adaptación propia. El prototipo actual width16 y kernels3/3/1 no se convierte en esta red por actualizar la documentación.

**Entrenamiento y recursos pendientes:** no copiar las 2.000 épocas ni la entropía cruzada del clasificador. Fijar pérdida de regresión, HPO, parada, semillas y presupuesto compartidos antes de ejecutar. Medir memoria, parámetros y tiempo del pipeline completo; no extrapolar desde el compacto.

**Pruebas requeridas:** forma y alineación con kernel par8; conservación de longitud 7; BN sin actualización en evaluación; gradiente finito hacia las tres convoluciones; no lectura de fechas exteriores; equivalencia funcional de operaciones frente a una referencia controlada y tolerancia declarada.

**NAS:** hay diseño explícito en paper y código. Esta propuesta ofrece una vía de cuerpo publicado con adaptación de tarea, pendiente de adjudicar. No afirmar exención de la receta sin documentar la traducción y el nuevo cabezal. Si se conserva el compacto actual, su clasificación se resuelve por separado conforme a §14, pp. 52–56.

## 22.2 ResNet-1D con tres bloques publicados

**ID de receta:** `resnet1d_wang_d7_v1_proposed`.

**Decisión propuesta:** conservar tres bloques con anchos64/128/128 y tres convoluciones por bloque. Para evitar una mezcla tácita de implementaciones, adoptar explícitamente la convención del snapshot de Wang: normalización inicial de la rama principal y proyección en los tres atajos.

**Fuente:** mismo artículo, §II.C, p. 3 y Fig. 1c. Código [ResNet.py, build_resnet](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/3ecc8971fa768bd01ada7f3a613688255e6256c2/ResNet.py#L15-L94), mismo SHA.

**Receta E2 propuesta:** en cada bloque, rama principal Conv(k8)→BN→ReLU→Conv(k5)→BN→ReLU→Conv(k3)→BN; sumar la rama de atajo y aplicar ReLU. Todas las convoluciones tienen bias.

- Bloque1: entrada5, salida64; BN5 adicional únicamente antes de la primera convolución principal. Atajo desde la entrada sin esa BN: Conv1D5→64, kernel 1, y BN64.
- Bloque2: entrada64, salida128; atajo Conv1D64→128, kernel 1, y BN128.
- Bloque3: entrada128, salida128; atajo **Conv1D128→128, kernel 1, y BN128**, no identidad. Esa elección reproduce la convención explícita del archivo, cuyos condicionales comparan con los canales de entrada originales.
- Media temporal sobre siete posiciones →128; Dense128→1 →Softplus.

Son nueve convoluciones principales más tres proyecciones de atajo. No confundir nueve con el total de capas convolucionales. La BN inicial no normaliza el atajo del primer bloque. No usar preactivación, bottleneck ni reducción temporal distintos de lo descrito.

**Entrenamiento y recursos pendientes:** contar el cuerpo completo y las proyecciones; HPO, pérdida, épocas y recursos permanecen sin ejecutar. Las 1.500 épocas del código de clasificación no son una asignación autorizada para E2.

**Pruebas requeridas:** dimensiones antes de cada suma; grafo con las tres proyecciones; alineación SAME8; prueba que detecte reemplazo accidental del tercer atajo por identidad; BN train/eval; gradientes por rama y sin datos fuera de D7.

**NAS:** cuerpo definido, pero la adopción multivariada 1D y cabeza de regresión se declaran y adjudican. No exención por el nombre ResNet. El compacto actual de dos bloques y cuatro convoluciones principales permanece como implementación distinta hasta que se autorice y verifique una sustitución.

## 22.3 Inception-1D como red individual de seis módulos

**ID de receta:** `inception1d_single_d7_v1_proposed`.

**Decisión propuesta:** una red individual con el cuerpo InceptionTime, no el ensemble de cinco miembros. Se mantiene el nombre de catálogo Inception-1D y se describe el alcance individual en cada resultado futuro.

**Fuente:** Fawaz et al., [InceptionTime](https://arxiv.org/pdf/1909.04939), §3.1–3.2, pp. 5–6. Código [inception.py](https://github.com/hfawaz/InceptionTime/blob/952d115e83fb3a66d75c5858c702f8dd5eeb18b7/classifiers/inception.py#L13-L105), SHA `952d115e83fb3a66d75c5858c702f8dd5eeb18b7`; [nne.py](https://github.com/hfawaz/InceptionTime/blob/952d115e83fb3a66d75c5858c702f8dd5eeb18b7/classifiers/nne.py) define separadamente el ensemble de cinco inicializaciones.

**Receta E2 propuesta:** seis módulos, cada uno con cuatro ramas de 32 canales y salida128. Entrada de módulo1: cinco canales; módulos2–6:128.

- Rama compartida para tres convoluciones: bottleneck Conv1D(C,32,k1), activación lineal; después tres Conv1D(32,32) con kernels40/20/10.
- Cuarta rama, desde la entrada original del módulo: max-pool3, stride 1, SAME; Conv1D(C,32,k1).
- Concatenar las cuatro ramas →BN128→ReLU. Todas esas convoluciones carecen de bias, siguiendo el código.
- Después del módulo3: sumar proyección Conv1D5→128,k1 +BN de la entrada de la red; ReLU. Después del módulo6: sumar proyección Conv1D128→128,k1 +BN del estado posterior al primer residual; ReLU. También las proyecciones carecen de bias.
- Media temporal →128; Dense128→1 con bias →Softplus.

Mantener kernels40/20/10 con padding19/20,9/10 y4/5 respectivamente. Muchos términos de los kernels recaerán en padding sobre D7: esto no impide la operación ni acredita utilidad predictiva. Para max-pool, las posiciones ausentes se excluyen del máximo, equivalente a menos infinito; **no rellenarlas con cero** cuando las entradas puedan ser negativas. No hay ReLU independiente antes de concatenar las ramas.

**Entrenamiento y recursos pendientes:** una trayectoria comprende un solo miembro. R semillas de evaluación no forman automáticamente un ensemble: se reportan como repeticiones, salvo que se registre explícitamente otra regla. HPO, pérdida, batch, parada y tiempo se fijan/medirán antes de entrenar.

**Pruebas requeridas:** seis módulos, cuatro ramas y 128 canales; ramas lineales; dos proyecciones residuales; max-pool con entradas negativas; alineación de kernels pares sobre D7; gradientes y BN congelada en evaluación; no promedio oculto de cinco modelos.

**NAS:** el cuerpo individual está definido en fuente, aunque el método InceptionTime completo incluye ensemble. Someter la adaptación individual, D7 y cabeza a adjudicación; no afirmar réplica del ensemble ni exención de la variante compacta actual de un único módulo.

## 22.4 LSTM-autoencoder con predictor de dos encoders

**ID de receta:** `lstm_sae_25_26_d7_v1_proposed`.

**Decisión propuesta:** adaptar la variante concreta del autor `2layer_evaluate.py`, con anchos25/26, no los anchos16 del compacto ni5/47 de otra ficha. Se elige por trazabilidad y por su predictor escalar, sin evidencia de superioridad en BTC.

**Fuentes y límite:** Sagheer y Kotb, [artículo LSTM-SAE](https://doi.org/10.1038/s41598-019-55320-6); el cuerpo del paper no se recuperó verificablemente en esta revisión. El diseño siguiente procede del [código de autor fijado](https://github.com/2M-kotb/LSTM-based-Stacked-Autoencoder/blob/8668132ef6f8855dbab289c5a2b2478f871c4c4b/2layer_evaluate.py), SHA `8668132ef6f8855dbab289c5a2b2478f871c4c4b`: configuración, bucle de preentrenamiento y construcción del predictor. Los localizadores por símbolo (`hidden_layers`, `AE.fit`, `trained_encoder`, `model.add`) son inequívocos; no trasladar números de líneas del extracto web a un archivo con otra representación.

**Estructura y fases propuestas para D7:**

1. AE1: X `(B,7,5)` →LSTM25, estado final `(B,25)` →Repeat7 →LSTM5 con salida secuencial `(B,7,5)`.
2. Calcular encoder1 en modo evaluación sobre entrenamiento y fijar esas representaciones. AE2: repetir su vector final siete veces `(B,7,25)` →LSTM26 final →Repeat7 →LSTM5 secuencial. **Ambos decoders reconstruyen la entrada original de cinco canales, con escala auxiliar X/s definida abajo**, no la representación de25 canales. Encoder1 no se reajusta durante AE2.
3. Transferir ambos encoders y descartar los decoders. Predictor: LSTM25 final →Dropout 0,3 →Repeat7 →LSTM26 final →Dropout 0,3 →Dense26→1 →Softplus. Ajustar los dos encoders supervisadamente, sin congelación inicial y sin pérdida conjunta de reconstrucción.

Las fases AE no llevan dropout añadido. Para el port se proponen puertas sigmoid, activación de estado tanh, kernel de entrada Glorot uniforme, kernel recurrente ortogonal y bias efectivo de olvido1, demás biases0; comprobar su correspondencia con CuDNNLSTM antes de transferir pesos. Estados recurrentes se reinician por ventana; no arrastrarlos entre orígenes. Esta secuencia **no** equivale a dos LSTM ordinarias apiladas con `return_sequences=True`: el segundo encoder recibe repeticiones del último estado del primero. Longitud25→7, variables8→5, enlace Softplus y transformaciones E2 son adaptaciones propias; no copiar el preprocesamiento de contaminación atmosférica del ejemplo.

**Propuesta auxiliar explícita para la reconstrucción:** los estados de salida de un decoder LSTM estándar están acotados. El D7 transformado no garantiza valores en [-1,1]. Para mantener los decoders LSTM5 y no agregar una Dense inventada, se propone usar como objetivo auxiliar `X/s`, con `s=max(1,max(abs(X_train)))`, un escalar calculado solo sobre el entrenamiento de la fase/fold y compartido por AE1/AE2. La entrada de ambos encoders y el predictor sigue el contrato D7 sin esta división adicional. Conservar s y su procedencia; no recalcularlo con validación/test ni recortar esos datos. La validación puede quedar fuera del rango reconstruible: cuantificar este riesgo y su efecto sobre la selección sin modificar s después de observarla. Esta normalización del objetivo auxiliar es una **adaptación propia por adjudicar**, no parte del código fuente. Si no se acepta, bloquear el preentrenamiento hasta definir un objetivo reconstruible; no ocultar el problema entrenando contra valores inalcanzables.

**Entrenamiento propuesto y pendientes:** MSE de reconstrucción en las fases AE; pérdida supervisada y escala de σ₇ según protocolo común. Reiniciar optimizador entre fases; transferir solo los pesos de encoders. Definir presupuestos separados para AE1, AE2 y ajuste final, selección con validación interna y semillas. Las cifras fuente291/439/463 épocas y batch146 se registran como antecedentes, no como presupuesto de E2. No hay ejecución ni permiso para comenzar.

**Pruebas requeridas:** ambas reconstrucciones `(B,7,5)`; AE2 contra X/s, no contra Repeat(encoder1); conservación exacta de pesos encoder1 durante AE2; transferencia verificable; ambos encoders con gradiente en ajuste final; decoders ausentes de inferencia; s train-only; representación intermedia recalculada dentro de cada fold y nunca heredada de otro corte. Verificar equivalencia de puertas, biases y activaciones al traducir CuDNNLSTM antiguo a una implementación actual; no asumir compatibilidad de pesos.

**Recursos:** una trayectoria incluye AE1+generación de representaciones+AE2+ajuste supervisado. Registrar consumo por fase y total, incluidos fallos y refits. Un forward del predictor no estima ese coste y el preentrenamiento no es gratuito ni externo.

**NAS:** existe un diseño concreto en código de autor, pero la condición de §14 habla del paper y este no quedó certificado; además se proponen adaptaciones. Mantener `adjudicación pendiente`, sin eximir por el nombre LSTM-AE ni concluir ausencia de diseño. Si se exige §14, registrar una correspondencia completa de tres bloques y mil arquitecturas: no llamar NAS al preset25/26 ni inventar silenciosamente una tercera capa.

## 22.5 TimesNet: adaptador D7 explícito

**Propuesta TN-D7-r1.** Mantener el adaptador actual como receta concreta, no atribuirle una reproducción literal. Entrada [B,7,5]; normalización por ventana con media y varianza poblacional, ε=10⁻⁵; embedding a 32; proyección temporal aprendida 7→8; dos TimesBlocks, d_model32, d_ff32, seis kernels Inception2D y GELU, con normalización de capa. En cada bloque usar las cuatro frecuencias no DC de longitud 8, periodos 8/4/2/2, pesos por amplitud de cada ejemplo y agregación residual. Leer el último token latente 32, concatenar diez momentos de la D7 escalada y aplicar Linear(42,1)+Softplus. Dropout inicial propuesto0,1, pendiente de HPO.

**Fuente y diferencia.** [TimesNet v3](https://arxiv.org/pdf/2210.02186v3), §3 y tabla 7, p14; [script ETTh1 fijado](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh). La fuente publica top-k5; el código D7 usa todas las cuatro frecuencias admisibles y elimina selección dependiente del batch. D32 deriva de la regla del paper, FF32 del script: no es una fila publicada única. Proyectar 7→8 genera una representación latente, no una octava observación real ni acceso a t.

**NAS / profesor.** Proponer adjudicación explícita del cambio de selección de frecuencias y del puente D7: ¿se admite como adaptación de tarea o requiere §14? Hasta resolverlo, NAS=null; no iniciar1.000 variantes ni conceder exención.

**Verificación y recursos pendientes.** Revalidar forma, cuatro frecuencias, periodos y pesos, equivalencia algebraica de las convoluciones fusionadas, gradientes y batch-invariancia en evaluación; detectar dependencia artificial de padding. Perfilar memoria, FFT, convoluciones 2D y latencia en hardware común, incluyendo entrenamiento futuro. El cuerpo coincide con software provisional existente; esta ficha no ejecuta ni modifica ese software ni valida desempeño de mercado.

## 22.6 TSMixer: cuerpo fuente y cabeza escalar

**Propuesta TSM-D7-r1.** Seis bloques residuales de mezcla temporal y de canales. En cada bloque: BatchNorm sobre el batch para cada posición temporal/canal (momentum 0,99 en convención Keras, no el parámetro homónimo de PyTorch; ε0,001), Linear temporal7→7, ReLU y dropout; segunda normalización, MLP5→512→5 con ReLU y dropout. Aplanar35 activaciones; concatenar cinco medias y cinco DE de la ventana escalada; Linear(45,1)+Softplus. Dropout inicial 0,1. No variables calendario ni covariables futuras.

**Fuente y diferencia.** [TSMixer v5](https://arxiv.org/pdf/2303.06053v5), Fig.1 p2, apéndiceE/tabla 8 p21; [implementación Google fijada](https://github.com/google-research/google-research/blob/d3a4bd8f0eceddda553e47085033b537935b283f/tsmixer/tsmixer_basic/models/tsmixer.py). Se conserva el cuerpo seis bloques/H512 de la fila ETTh1. La fuente usa dropout0,9 en esa fila;0,1 es propuesta propia de HPO, no reproducción exacta. La tabla imprime un rango que no contiene512: registrar la discrepancia, no corregirla silenciosamente. D7, salidaσ y momentos son adaptaciones.

**NAS / profesor.** Candidato a cuerpo definido más puente de tarea. Pregunta concreta: ¿cabeza45→1 y longitud 7 conservan la condición de diseño definido, y qué ejes estructurales de HPO deben permanecer fijos? No conceder exención automática.

**Verificación y recursos pendientes.** Comprobar equivalencia del port PyTorch con la normalización Keras, estadísticas sólo del entrenamiento y evaluación invariante al acompañamiento de batch. Batch1 no prueba entrenamiento útil: puede anular la variación normalizada. Medir memoria y latencia de seis bloques/MLP 512; comparar variante sin momentos como control separado pendiente. Coincide con el cuerpo del adaptador provisional, no con entrenamiento ya ejecutado.

## 22.7 TiDE: variante latente8 declarada

**Propuesta TIDE-D7-r1.** Normalización por ventana y transformación afín aprendida por canal. Embedding de identidad de variable16; feature encoder residual5→64→4. Concatenar historia de siete pasos, covariables observadas transformadas y embedding; encoder residual55→256→256, decoder residual256→256→8 y decoder temporal residual(8+4)→128→8. La covariable futura es un sentinel cero transformado, no OHLCV futuro. Añadir proyección residual7→1 por variable, difundida sobre sus ocho coordenadas latentes; concatenar los40 valores latentes (5×8) y diez momentos; Linear(50,1)+Softplus. Dropout 0,1, LayerNorm ε0,001.

**Fuente y diferencia.** [TiDE v5](https://arxiv.org/pdf/2304.08424v5), tablas 7–8 pp20–21; [models.py](https://github.com/google-research/google-research/blob/d3a4bd8f0eceddda553e47085033b537935b283f/tide/models.py), [preset ETTh1](https://github.com/google-research/google-research/blob/d3a4bd8f0eceddda553e47085033b537935b283f/tide/scripts/etth1.sh). El preset256/8/128 corresponde a H336/H720, no al H96 del script. `num_layers=2` implica dos Dense dentro de un bloque residual de este código, no dos bloques independientes. El decoder temporal original emite1; aquí emite8 para evitar LayerNorm(1) degenerada. Cambian covariables y salida, no sólo el último readout.

**NAS / profesor.** Es una adaptación estructural explícita. Adjudicar si el decoder latente8 y las covariables sustitutas conservan la identidad de TiDE admisible o deben estudiarse como diseño propio sujeto a §14. No simplificarlo a MLP genérico ni presentarlo como exento por el nombre.

**Verificación y recursos pendientes.** Pruebas de ausencia de colapso escalar, contribución de encoder/decoder/residual, sentinel sin valores futuros y cinco identidades de variables; equivalencia de semántica `num_layers`. Perfilar memoria de capas256 y operaciones por ventana. La receta coincide con el adaptador provisional; no se ejecuta una variante nueva en esta ficha.

## 22.8 TimeMixer: tres escalas para siete días

**Propuesta TM-D7-r1.** Canal-independiente; pooling promedio factor2 con dos descensos para longitudes7→3→1; normalización por escala, embedding a16; dos bloques PDM de mezcla pasado estacional/tendencia, d_ff32; descomposición por media móvil25 con padding de bordes. FMM con una proyección temporal a horizonte1 por escala; sumar representaciones resultantes [B,5,16]. Concatenar80 latentes y diez momentos; Linear(90,1)+Softplus. Dropout 0,1. Sin covariables calendario ni proyección final a OHLCV.

**Fuente y diferencia.** [TimeMixer v1](https://arxiv.org/pdf/2405.14616v1), §3 y tabla 7 p14; [script ETTh1](https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh). El preset fuente usa tres descensos; D7 con ese pooling produciría una escala vacía. La propuesta reduce a dos, conserva PDM/FMM y cambia la cabeza. Padding del filtro25 no añade observaciones y su comportamiento en escala1 merece evaluación específica.

**NAS / profesor.** Adjudicar la reducción de escalas y readout como adaptación necesaria o diseño nuevo; ni la necesidad geométrica concede exención ni el nombre obliga automáticamente a NAS. Mantener pendiente la decisión.

**Verificación y recursos pendientes.** Comprobar longitudes7/3/1, información y gradientes por escala, degeneración de normalización en longitud1, orden de mezcla y ausencia de calendario futuro. Perfilar escalas, operaciones y memoria. El código actual proporciona un punto de partida mecánico, no evidencia de que tres escalas sean útiles en BTC.

## 22.9 PatchTST: cuerpo pequeño con geometría D7

**Propuesta PT-D7-r1.** RevIN sobre los siete días reales, con afín y sin restar sólo el último. Replicar nueve posiciones a la izquierda para longitud16 y ocho al final; patch16, stride8: dos parches por cada una de cinco variables. Encoder pequeño: tres capas, cuatro cabezas, d_model16, FF128, GELU, normalización BatchNorm y atención residual. Aplanar5×2×16=160 latentes y agregar diez momentos; Linear(170,1)+Softplus. Dropout inicial 0,1 en el cuerpo; la cabeza conjunta Linear(170,1) no incluye dropout. El head del autor y su dropout se retiran. Sin descomposición adicional.

**Fuente y diferencia.** [PatchTST v2](https://arxiv.org/pdf/2211.14730v2), §4.1 p6 y A.1.4 p14; [preset ETTh1](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised/scripts/PatchTST/etth1.sh). Se conserva el cuerpo pequeño. La fuente usa contextos largos y dropout0,2; el script usa contexto 336 y dropout 0,3. Ninguno equivale a esta receta. Los dos parches contienen repeticiones de siete observaciones, no32 observaciones distintas. La cabeza conjunta cambia el forecast por variable a σ₇.

**NAS / profesor.** PatchTST publicado es ejemplo exento en la guía, pero debe adjudicarse este padding/puente concreto. Pregunta: ¿se acepta el cuerpo publicado con geometría corta explícita o exige una ruta distinta de adaptación/NAS? No conceder exención por nombre.

**Verificación y recursos pendientes.** Normalizar antes del padding; verificar dos parches, posiciones replicadas, independencia entre canales del cuerpo y mezcla sólo en readout. Probar variación de entradas y gradientes para detectar degeneración de parches casi redundantes. Medir memoria, latencia y coste de tres capas; no heredar rendimiento del contexto 336.

## 22.10 iTransformer: cinco tokens de variables

**Propuesta IT-D7-r1.** Normalización por ventana (varianza poblacional+10⁻⁵); embedding de cada serie de siete valores a256, creando cinco tokens OHLCV. Dos capas encoder de atención invertida, ocho cabezas, d_model256, FF256, GELU, normalizaciones de la fuente y dropout 0,1. Aplanar5×256=1.280, añadir diez momentos y aplicar Linear(1290,1)+Softplus. Se elimina el projector y la desnormalización que predecían cada variable original.

**Fuente y diferencia.** [iTransformer v4](https://arxiv.org/pdf/2310.06625v4), algoritmo1 p13 y configuración p14; [preset ETTh1 fijado](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh). Se conserva el cuerpo L2/D256/FF256 del presetH96 y ocho cabezas del código, pero se adapta longitud a7 y salida a volatilidad. No atribuir ese preset a todos los horizontes publicados.

**NAS / profesor.** Candidato a cuerpo definido con puente de tarea; registrar si embedding7 y cabeza conjunta con momentos son admisibles sin NAS. Su definición primaria no acredita automáticamente la receta completa.

**Verificación y recursos pendientes.** Confirmar atención entre cinco variables, no siete días; orden estable de canales y shapes, consistencia del port con el autor, efecto de normalización y gradientes. Perfilar dimensión 256, parámetros/memoria y latencia bajo hardware común; comparar cabeza sin momentos por separado. No se han ejecutado comparaciones de mercado.

## 22.11 CNN-1D + LSTM + atención temporal

**Propuesta HIB-D7-r1.** Entrada [B,7,5]; Conv1D causal5→32, kernel 3, stride 1, padding cero sólo a izquierda2 y ReLU. LSTM unidireccional32→32 de una capa, con estados iniciales cero en cada ventana y salida de los siete estados. Atención aditiva: Linear(32,16)+tanh, vector puntuador16→1 sin sesgo, softmax sobre el eje temporal; suma ponderada de siete estados. Dropout 0,1 sobre contexto32; Linear(32,1)+Softplus. No concatenar momentos ni arrastrar estado entre ventanas. No pooling previo a atención.

**Fuente y diferencia.** No se verificó un artículo que defina esta receta completa. Es una composición propia documentada en `scripts/e2_current_models.py`, no una contribución original demostrada por juntar bloques conocidos. Coincide con los anchos del preset provisional, distinto del smoke8/8/8. La causalidad de prefijos se refiere al tensor transformado: `relative_window` utiliza C[t−1] en todos sus pasos.

**NAS / profesor.** Si se trata como diseño propio E2, proponer §14 y acordar correspondencia de los tres bloques: w₁ controla canales CNN, w₂ unidades LSTM, w₃ proyección de atención. Deben precisarse tres ubicaciones de dropout0,2 y probar 1.000 arquitecturas únicas; esta correspondencia aún no está implementada ni aprobada. Si su papel final pertenece a E3, aclararlo con la guía específica, sin inventar una exención.

**Verificación y recursos pendientes.** Siete pesos no negativos que suman1, gradientes a los tres bloques, estados reiniciados, prefijos transformados causales y salida positiva. Las ablaciones que aíslan bloques (LSTM+atención, CNN+atención, CNN+LSTM+pooling uniforme) todavía deben implementarse y presupuestarse; los controles antiguos no las sustituyen. Estimar memoria/tiempo de NAS tras definir su alcance, no multiplicar el smoke por 1.000 como coste real.

## 22.12 Decisiones y costes todavía pendientes

Se proponen cuerpos publicados para FCN/ResNet y la red Inception individual; SAE adopta una variante de código del autor con adaptaciones explícitas. Ninguno sustituye todavía a su prototipo en el repositorio. Las seis recetas de literatura y el híbrido se especifican a partir de los adaptadores existentes, sin atribuirles reproducción experimental ni entrenamiento de mercado.

**Preguntas a resolver antes de congelar:** si D7/cabeza se acepta como adaptación de tarea de los cuerpos publicados; cómo tratar las alteraciones estructurales de TimesNet/TiDE/TimeMixer y la geometría de PatchTST; qué evidencia del paper se exige para el SAE; y cómo encuadrar el híbrido E2/E3 y su correspondencia con tres bloques. No inferir respuestas por el nombre de una red.

La receta SAE propuesta tiene dos fases de reconstrucción y una supervisada. Los subtotales anteriores que contaban una sola reconstrucción por trayectoria no describen esta propuesta. Como ejemplo meramente aritmético, si se conservara el escenario reducido de §19 (cuatro configuraciones, dos semillas, diez folds y dos etapas inner/refit), serían 320 ajustes auxiliares AE en lugar de 160;2.240 supervisados+320=2.560 etapas, y piloto 280+40=320. **No es el presupuesto final conforme a la guía ni trabajo autorizado o realizado**: HPO ampliado, NAS, controles y recursos siguen pendientes. Cada una de las tres fases debe contabilizarse en tiempo/memoria y en sus refits.

No se dan horas ni parámetros medidos de las nuevas recetas sin implementación y perfilado. El siguiente paso es adjudicar las recetas, luego implementar los cuerpos aceptados y verificar equivalencia/causalidad sin trasladar resultados históricos. Cualquier entrenamiento requiere su plan y autorización correspondientes.
