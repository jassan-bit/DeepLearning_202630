# Comparación formal del mecanismo realmente evaluado

Fecha de revisión: 18 de noviembre de 2026. Fuente fijada: [src/e3_innovation_model.py](https://github.com/jassan-bit/DeepLearning_202630/blob/76b2397d99027457b922225cec8a4f0aab1cc04e/src/e3_innovation_model.py), SHA256 `7f362d38ca83ed302dc9c822b4e4e19a0d446e4c01477f42b0e9a8da823ae7b1`. Se verificó igualdad de bytes con el archivo local consultado. Análisis asistido por IA; revisión humana independiente pendiente.

## Identidad y procedencia de los componentes

La receta de 1210 parámetros emplea CNN causal, LSTM unidireccional y pooling temporal softmax por contenido. La lectura produce un escalar `z`; con `a = y_scale > 0`, `epsilon = 10^-4` y un piso observable `b = sqrt(A)`, devuelve

\[
\widehat y=b+a[\epsilon+\operatorname{softplus}(z)].
\]

`load_innovation_data` construye A con los seis retornos conocidos de siete cierres anteriores al objetivo. `_components_from_score` aplica exactamente esta cabeza. La receta usa MSE normalizada; no incorpora NLL, momentos distribucionales ni la penalización o saturación del primer módulo. El análisis no atribuye al autor una transferencia histórica de fuentes que sólo se identificaron después.

## Derivación del soporte: una identidad algebraica

Sean `r_1,...,r_6` los retornos conocidos, `mu` su media y `u` el retorno futuro. La media de los siete retornos es `mu+(u-mu)/7`. Al centrar la suma de cuadrados alrededor de esa media, el término cruzado de los seis conocidos se anula, porque `sum(r_i-mu)=0`. De ello resulta

\[
\sum_{i=1}^{6}(r_i-\bar r_7)^2+(u-\bar r_7)^2
=\sum_{i=1}^{6}(r_i-\mu)^2+\frac67(u-\mu)^2.
\]

Dividir por siete —el objetivo usa `ddof=0`— da

\[
A=\frac17\sum_{i=1}^{6}(r_i-\mu)^2,
\qquad y^2=A+\frac6{49}(u-\mu)^2,
\qquad y\geq \sqrt A.
\]

El piso es alcanzable con `u=mu`. Es una descomposición elemental de suma de cuadrados; su demostración y uso en esta tarea no se presentan como descubrimiento matemático. El objetivo es la desviación estándar móvil realizada y centrada de siete retornos, sin anualización. No equivale automáticamente a varianza integrada latente, varianza realizada intradía o una suma no centrada de cuadrados.

Sólo la auditoría de etiquetas puede usar `u` realizado. El cálculo del piso y la predicción no lo necesitan ni lo reciben. La información del piso está contenida en los cierres observables: hacerlo explícito impone estructura, pero no añade información futura.

## Relación precisa con una guarda conocida

[Brosowsky, Keck, Dünkel y Zöllner, AAAI2021](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf), p.6816, propone una guarda escalar sobre un umbral por muestra mediante `exp(v)+b`. Esto es un antecedente directo del patrón de soporte dependiente de la muestra. La versión preprint2020 tiene otra lista de autores y se mantiene separada.

Para cada `a>0` fijo, defínanse

\[
b'=b+a\epsilon,\qquad g(z)=\log[a\operatorname{softplus}(z)].
\]

En aritmética real y para todo `z` finito,

\[
b'+\exp(g(z))=b+a[\epsilon+\operatorname{softplus}(z)].
\]

`g` es una biyección creciente de R a R. Ambas parametrizaciones cubren la misma semirrecta abierta `(b',infinito)` al permitir variar libremente su escalar interno. Así, la mera propiedad «salida sobre un piso por muestra» y sustituir un enlace positivo por otro no bastan para acreditar una guarda original. El desplazamiento `a*epsilon` también es explícito: la salida de E3 no cubre todo `[b,infinito)` ni alcanza su frontera física.

**Alcance exacto de esta equivalencia:** es una identidad escalar después de componer con `g`, y una igualdad de rango admisible. No demuestra igualdad de clases neuronales con una misma lectura afín, porque `g(z_theta(X))` puede requerir una composición que esa lectura no representa. Tampoco demuestra igualdad de gradientes respecto de los parámetros, geometría del riesgo, trayectorias del optimizador, estabilidad de punto flotante o resultados con capacidad finita. En el mismo valor de `z`, las derivadas de los enlaces originales son distintas: `a*sigmoid(z)` frente a `exp(z)`. Un efecto de entrenamiento tendría que formularse y medirse con controles comparables; no se infiere de la identidad anterior.

Un caso límite delimita otra pretensión: si todos los retornos conocidos y el futuro coinciden, `A=y=0`, mientras la cabeza con `a>0` produce al menos el desplazamiento positivo en el límite. Respetar un soporte interior no garantiza insesgamiento, calibración o error mínimo en cada historia.

## Atención, estructura económica y teoría

[Dasoulas et al.2021](https://proceedings.mlr.press/v139/dasoulas21a/dasoulas21a.pdf), §3.3 y §5, es un antecedente del pooling con consulta aprendida y normalización de atención. E3 no implementa LipschitzNorm. [Kim et al.2021](https://proceedings.mlr.press/v139/kim21i/kim21i.pdf), §6, y el seguimiento [Castin et al.2024](https://arxiv.org/html/2312.14820v2), §1.2 y §3, delimitan la regularidad en dominios acotados. La cota particular del proyecto aporta verificación técnica, sin establecer por ello novedad de atención o robustez global OHLCV.

[HARNet](https://arxiv.org/html/2205.07719v1), §4, muestra un antecedente de incorporación de estructura de volatilidad en una red. Su arquitectura, inicialización HAR y objetivo de varianza realizada no son los de E3. Ser diferente de HARNet o de [TimeMixer](https://arxiv.org/html/2405.14616v1), §3, tampoco demuestra novedad frente a la familia de guardas usada.

## Utilidad observada y atribución pendiente

El [control posterior de soporte](https://github.com/jassan-bit/DeepLearning_202630/blob/76b2397d99027457b922225cec8a4f0aab1cc04e/experiments/e3_remediacion_20261118/constraint_audit/README.md) ya conservado cuantifica una limitación concreta: 23,01 % de las salidas históricas predictor-fecha incumplen el piso. Aplicar a todos la proyección `max(prediccion,b)` reduce la ventaja RMSE descriptiva de E3 de 24,10 % a 10,33 %. El control sólo piso mejora su MAE en 3,70 %. Este expediente no recalcula esas cifras ni ejecuta nuevos contrastes.

La proyección reduce o mantiene por observación el error cuadrático y absoluto cuando el objetivo pertenece al semieje admisible. Esa propiedad pertenece a una regla conocida de proyección; no demuestra causalmente el beneficio de la cabeza neuronal. Tras proyectar siguen vigentes el calendario conocido y las diferencias históricas de ajuste E2/E3.

## Afirmación defendible y decisión de R1

La afirmación actualmente respaldada es: **implementación y evaluación retrospectiva de una adaptación con soporte explícito para sigma7, junto con una auditoría de comparadores que muestra la importancia de tratar ese soporte de forma común**. No se presenta como una invención de atención, softplus, la identidad de varianza, una guarda de semieje o una prueba de superioridad confirmatoria.

El residuo por examinar es si existe una modificación sustantiva de mecanismo, pérdida o entrenamiento más allá de esa parametrización de tarea. La revisión actual no la establece. Completar una búsqueda sin coincidencia literal no basta: deben examinarse equivalencias funcionales y trabajos próximos. Si el expediente y la revisión humana no justifican una adaptación sustantiva, la ruta correcta será reconocer el límite o formular otro mecanismo con su propia evaluación futura. No se cambia automáticamente el candidato para perseguir la nota.
