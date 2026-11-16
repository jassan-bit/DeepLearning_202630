# Ficha de diseño: CNN-LSTM-atención v2

La [ficha completa](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/README.md)
conserva la arquitectura **CNN causal → LSTM → atención temporal por contenido**
de M00. La propuesta modifica la cabeza de salida y la pérdida. Está en fase de
diseño: la red v2 y su coordinador no están implementados ni entrenados.

## Base común y modificación propuesta

Se mantienen D7×5, CNN de ocho canales con kernel 3, LSTM de dimensión 12,
consulta de atención aprendida y lectura escalar: **1210 parámetros entrenables**
por nueva variante. M00 conserva la atención y retira las dos modificaciones
del primer full: penalización de discordancia y saturación radial.

La lectura positiva pasa a representar la escala de una magnitud de innovación.
Para los seis retornos conocidos, con media $m_t$, se calcula

$$
A_t=\frac17\sum_{i=t-6}^{t-1}(r_i-m_t)^2,\qquad
c=\frac6{49},\qquad
y_t^2=A_t+cU_t^2,\qquad U_t=|r_t-m_t|.
$$

$A_t$ procede de la entrada. $U_t=\sqrt{(y_t^2-A_t)/c}$ es una etiqueta
transformada para aprender la parte desconocida. No se usa como predictor ni
identifica el signo del próximo retorno.

## Densidad y predicción en unidades de σ7

Se propone modelar $U_t$ mediante una half-Student-t de cinco grados de libertad,
con escala positiva $s_\theta(X_t)$ producida por la misma red. Su densidad es
$q(u\mid X)=2t_5(u/s)/s$ para $u\ge0$. En el interior $y>\sqrt A$,

$$
p_\theta(y\mid X)=q_\theta(u\mid X)\frac{y}{cu}.
$$

La salida puntual busca la media de σ, aproximada mediante una regla positiva
de 128 nodos:

$$
\widehat y=g_{128}(A,s)=\sum_{j=1}^{128}a_j\sqrt{A+cs^2z_j^2},
\qquad a_j>0,\quad\sum_j a_j=1.
$$

Esta salida respeta el piso $\sqrt A$ en aritmética exacta. También es homogénea
en escala y Lipschitz respecto de $s$ para $A$ fijo. La
[demostración](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/theory.md)
delimita los casos de borde y no extiende esa garantía a todo el mapa OHLCV.
En la verificación sintética, N=128 observó un error relativo máximo de
aproximadamente $2.48\times10^{-5}$ en la grilla declarada. Esto no es una cota
uniforme certificada del promedio continuo.

La pérdida propuesta combina MSE normalizada sobre σ y NLL de la magnitud U
estandarizada, con peso auxiliar 0.1. Las escalas se calculan sólo en ajuste.
El jacobiano omitido al optimizar U es constante en θ en el interior; la NLL de U
no se presenta como el logscore completo de σ. La familia Student-t y el peso
auxiliar son hipótesis de diseño, sin garantía de mejor RMSE o calibración.

## Procedencia y estado de novedad

La [procedencia de seis fuentes](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/literature/sources.json)
distingue las ideas conocidas y la adaptación propuesta. SurVAE aporta
antecedentes de transporte de densidades y plegado de signos; la folded-t,
las densidades neuronales y las restricciones por muestra también tienen
antecedentes. La combinación específica de esta ficha se propone en esta
colaboración con asistencia de IA: **no se atribuye como modelo publicado por
esos autores ni se certifica R1**. Se conserva la CNN-LSTM-atención, sin adoptar
el codificador de otro trabajo.

## Hipótesis, controles y piloto previsto

El [plan estructurado](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/pilot_plan.json)
declara tres pruebas de desarrollo:

1. Verificar soporte, propiedades del operador y gradientes antes de entrenar.
2. Exigir al candidato al menos 2 % menos RMSE que el mejor de cinco controles
   coherentes para superar una heurística de continuación. Ese 2 % no es δ
   confirmatorio ni una prueba de significancia.
3. Comparar la misma cabeza con y sin NLL auxiliar para aislar su contribución.

Los controles son el piso solo, piso más incremento medio de ajuste, M00 v1
proyectado con pesos conservados, una cabeza positiva aditiva sobre la misma red
y la cabeza probabilística entrenada sólo con MSE. Así, una ventaja sobre una
salida que incumple el piso no basta para atribuir una mejora al mecanismo nuevo.

Se prevé la misma partición 532/7/116, una semilla, máximo 40 épocas por red y
paciencia 8. Las 116 fechas continúan siendo desarrollo expuesto. El tope
adicional conjunto propuesto es **600 segundos**, con un máximo combinado de
969.48 de los 1200 segundos originales. El presupuesto previo se conserva.
Si una receta queda interrumpida por el tope, se declara piloto incompleto.
La ficha no inicia esa ejecución ni modifica el notebook v1.

## Verificación reproducible

El verificador implementa únicamente el operador matemático y usa entradas
sintéticas; no carga datos de mercado o pesos entrenados:

```bash
python -m pip install -r requirements-e3-design-cpu.lock.txt
python scripts/e3_verify_innovation_design.py --check
python -m pytest tests/test_innovation_design.py -q
```

Los nodos, pesos, hash y resumen están en
[verification/](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification).
Las lecturas focales y verificaciones internas no sustituyen la revisión
sistemática de novedad ni la verificación humana externa.
