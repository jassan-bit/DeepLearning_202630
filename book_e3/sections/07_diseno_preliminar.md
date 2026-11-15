# Diseño preliminar del modelo original

La especificación inicial se desarrolla ahora en el [Bloque C](09_modelo_original.md), con demostración e implementación verificadas sobre datos artificiales. Este antecedente conserva la motivación y distingue las decisiones iniciales de resultados de mercado todavía inexistentes.

## Problema que podría abordar

Se mantiene la continuidad CNN causal → LSTM unidireccional → atención temporal para BTCUSDT, D7×5 → σ₇ con h=1. La contribución candidata debe residir en una modificación algorítmica que resuelva una limitación pertinente; apilar estos bloques no acredita R1.

El [paper guía TimeMixer](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/05_paper_guia.html) tiene una limitación operativa de alineación de calendario medida en ETTh1 que no se traslada al adaptador BTC sin marcadores de calendario. La limitación predictiva por régimen quedó inconclusa tras Holm. Por ello, ninguna se afirma aquí como defecto predictivo de Bitcoin demostrado.

Una motivación alternativa proviene del [estrés de entradas de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/54_robustez_interpretabilidad.html): con 10 % de celdas desplazadas ±5 DE, la RMSE aumentó 585,40 % en SAE, 513,98 % en GRU32 y 1.552,52 % en TimeMixer. Es un resultado retrospectivo bajo corrupción sintética de tensores; puede romper coherencia OHLCV y no caracteriza necesariamente errores de mercado plausibles. Sirve para formular una hipótesis de desarrollo, no para garantizar una mejora.

## Hipótesis provisional

Penalizar discordancia entre escalas en la atención y acotar sus valores podría reducir sensibilidad a corrupción de entradas frente al mismo híbrido con atención convencional, sin deterioro limpio mayor que el margen heredado. La hipótesis fallaría si la mejora desaparece bajo perturbaciones plausibles, degrada extremos reales o no se mantiene con las ablaciones. Mejorar robustez no sustituye la superioridad primaria exigida frente a todos los rivales.

## Formulación candidata

Para siete estados latentes $H\in\mathbb{R}^{7\times d}$, una matriz de promedio causal fija $P$, independiente de $H$, y el estado $h_j$:

$$
d_j(H)=\sqrt{\|h_j-(PH)_j\|_2^2+\varepsilon^2},\qquad
s_j=q^\top h_j-\lambda d_j(H),\qquad
a=\operatorname{softmax}(s/\tau).
$$

$$
v_j=\frac{B h_j}{\sqrt{B^2+\|h_j\|_2^2}},\qquad
c=\sum_{j=1}^{7}a_jv_j,\qquad
f(H)=\operatorname{softplus}(w^\top c+b).
$$

$B>0$ es un radio de saturación escalar, no una matriz; $\varepsilon>0$, $\lambda\ge0$ y $\tau>0$. La salida recupera la escala original con la media positiva del objetivo de entrenamiento. La regla candidata combina penalización de discordancia y saturación radial; su procedencia y su novedad deben contrastarse con trabajos cercanos antes de adoptarla.

## Propiedad que se investigará

Para pesos de inferencia fijos, $\|q\|_2\le Q$, $0\le\lambda\le\Lambda$, $\tau\ge\tau_{\min}>0$ y $\|w\|_2\le W$, se propone demostrar la siguiente cota del módulo:

$$
|f(H)-f(H')|\le
W\left[1+\frac{B\sqrt7}{2\tau_{\min}}
\left(Q+\Lambda\|I-P\|_2\right)\right]\|H-H'\|_F.
$$

La ruta de demostración usa que la saturación radial y Softplus son 1-Lipschitz, cada valor tiene norma acotada por $B$, la discordancia tiene constante $\|I-P\|_2$ y el jacobiano de softmax tiene norma espectral a lo sumo $1/2$. La demostración completa y la verificación numérica de la cabeza latente se presentan en el Bloque C; no se extienden automáticamente al predictor desde OHLCV crudo.

La cota se refiere a estados latentes. No demuestra por sí sola estabilidad del predictor respecto a OHLCV crudo. Extenderla exige controlar CNN, LSTM, preprocesamiento y reescalado de la salida; la constante en escala original se multiplica por la media del objetivo usada al entrenar.

## Experimentos de desarrollo necesarios

1. Buscar antecedentes de atención robusta, discordancia multiescala, saturación e influencia acotada en volatilidad financiera y en otras aplicaciones. Registrar fuentes, fechas y diferencias; descartar la propuesta como original si la evidencia no sostiene R1.
2. Comparar con el mismo híbrido convencional bajo preprocesamiento y capacidad comparables. Registrar cada variante evaluada y el esfuerzo de diseño, usando sólo desarrollo.
3. Verificar cota, gradientes, causalidad y exportación. Probar perturbaciones controladas, restricciones cercanas a sus límites y extremos genuinos de volatilidad.
4. Diseñar al menos tres ablaciones aisladas y sustantivas conforme a la guía. Se implementaron tres ablaciones funcionales: sin penalización, sin saturación y sin contenido. La tercera examina un componente convencional, no una tercera innovación; su suficiencia académica y sus efectos en datos reales siguen por evaluar.

La penalización podría atenuar un salto de volatilidad legítimo al considerarlo discordante. Deben examinarse especialmente errores de cola y cobertura condicional. No se han fijado anchos, operador $P$, radios, temperaturas, semillas, datasets adicionales o tag de protocolo, ni se ha comprobado la novedad de esta combinación.

## Revisión de novedad prevista

La búsqueda requiere al menos tres bases y cinco cadenas, consulta hacia atrás/adelante desde tres a cinco trabajos cercanos, búsqueda de implementaciones y revisión externa de al menos dos cadenas. Los términos iniciales incluirán robust attention, bounded influence, temporal attention, multiscale disagreement, realized volatility, Bitcoin y sus equivalentes en español. Son términos de planificación, no consultas ejecutadas ni una declaración de ausencia de antecedentes.
