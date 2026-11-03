## 3.3. Fichas documentales de las tres familias candidatas

Las tres familias siguientes representan memoria recurrente, convolución causal y atención. Su candidatura es **documental y acotada al corpus recuperado**; todavía no acredita la selección definitiva por recurrencia y mínima complejidad exigida por la guía. Se distinguen: arquitectura reportada por la fuente, cálculo analítico derivado de su descripción y adaptación que necesitaría el proyecto. La elaboración de estas fichas no ejecutó código de Deep Learning ni generó predicciones o resultados nuevos con datos de mercado; las referencias Ridge del EDA se documentan por separado en el capítulo 2.

Tras la recuperación dirigida, el corpus exploratorio contiene **cinco textos completos**: S1, S2, S5, S6 y S7. La **familia LSTM aparece en 4/5**, incluida la BiLSTM de S2; sus variantes unidireccionales están en S1/S5/S7, **3/5**. GRU aparece en **2/5**, TCN en **1/5** y encoder Transformer en **2/5**, S2/S6. S6 y S7 se incorporan como literatura gris con riesgos explícitos; una mayor cantidad de textos no acredita revisión por pares, mínima complejidad o comparabilidad de sus resultados. Las fichas concretas se mantienen apoyadas en LSTM de S5, TCN de S1 y SELF-ATT de S2. S6/S7 amplían evidencia de las familias, no definen réplicas idénticas de esas configuraciones. Una referencia general de arquitectura no cuenta como estudio aplicado adicional.

El contrato del proyecto conserva cinco covariables OHLCV diarias, entradas de D=7/14/21/28 días, volatilidad histórica de n=7 retornos diarios y horizonte h=1. Al inicio de t se conoce información hasta t−1 y se pronostica Y_t=σ_(t+1), que incorpora r_t. Este objetivo no es volatilidad realizada de siete días enteramente futuros. Ninguno de los cinco textos utiliza exactamente este contrato.

### 3.3.1. LSTM unidireccional

#### Referencias y papel comparativo

**S1.** Akbiyik, M. E., Erkul, M., Kämpf, K., Vasiliauskaite, V. y Antulov-Fantulin, N. (2023). *Ask “Who”, Not “What”: Bitcoin Volatility Forecasting with Twitter Data*. WSDM 2023. DOI: [10.1145/3539597.3570387](https://doi.org/10.1145/3539597.3570387). El artículo compara LSTM, GRU y TCN antes de proponer D-TCN. La LSTM es un comparador y no el modelo novedoso del trabajo. Evidencia: PDF pp. 4–6, §§5.2–5.4, Fig. 1 y Tabla 2; código `scripts/hpo.py`, líneas 151–235.

**S5.** Gao, M., Li, Q. y Tao, Z. *Comparing the Predictive Power of Five Models on Bitcoin Volatility*. Manuscrito de 2023 y código del primer autor; referencia editorial declarada a Springer, *Learning and Analytics in Intelligent Systems*, vol. 32, pp. 473–486, DOI declarado [10.1007/978-3-031-38074-7_41](https://doi.org/10.1007/978-3-031-38074-7_41). Año editorial externo pendiente de corroboración; no se infiere de la fecha del DOCX. El texto compara media, random walk, GARCH, una red densa lineal y LSTM. La arquitectura sencilla de una capa y veinte unidades está descrita en Tabla 4 y en la celda 111 del [notebook fijado](https://github.com/ga0m1/bitcoin-volatility-models/blob/f19551ca4cca768354eb7e2b7c1dd5c912bd0d41/BTC_USD.ipynb).

#### Arquitecturas efectivamente descritas

S5 utiliza catorce observaciones de volatilidad como entrada, una LSTM de veinte unidades y una salida Dense(1). No incorpora una segunda capa recurrente. El notebook compila con MSE y Adam, batch 64, máximo 200 épocas, paciencia 20 y restauración de pesos elegidos según `val_rmspe`. El callback de checkpoint monitoriza `loss`, un criterio distinto: esta diferencia debe conservarse al interpretar el procedimiento y no se sustituye por un protocolo supuesto. No se reportan varias semillas ni dispersión entre corridas.

S1 utiliza un día de 96 retornos de quince minutos como entrada y produce 96 retornos del día siguiente mediante una cabeza densa. Su búsqueda varía unidades recurrentes entre 32 y 512, dropout entre 0 y 0,5, learning rate entre 10⁻⁷ y 10⁻², weight decay entre 10⁻⁹ y 10⁻² y ε entre 10⁻² y 10⁻¹. Usa AdamW y pérdida cuadrática ε-insensible, 250 configuraciones por red, treinta épocas y veinte corridas para la comparación. El apéndice C, Tabla 5, sí reporta la configuración ganadora de S1: H=261, dropout 0,0237, ε=0,0364, learning rate 0,00521 y weight decay 9,25×10⁻⁷. La revisión anterior omitía esta tabla; se corrige esa extracción. Estos valores de autores no son una configuración del proyecto.

#### Apoyo adicional de la familia LSTM: S7

**S7.** Alakhouch Idriss (2026). *Bitcoin Volatility Forecasting: Econometric and Machine Learning Approaches Applied to Daily Price Data (July 2010–February 2024)*. Informe fechado 02/06/2026, sin DOI ni revisión por pares acreditados. [Informe y versión de autor](https://github.com/idrissalk/Bitcoin-Volatility-Forecasting/blob/a98099e4125e1f80a094a46889d97bc31abfe30b/Repport/Asset-volatility-Forecasting.pdf), §3.5.3 y Tabla 4.5, p. 34 impresa; [código](https://github.com/idrissalk/Bitcoin-Volatility-Forecasting/blob/a98099e4125e1f80a094a46889d97bc31abfe30b/Python/Python.py), líneas 366–415.

El informe compara una LSTM independiente con GARCH, Random Forest y XGBoost, y presenta una propuesta EGARCH+XGBoost. Declara ventana de 60 días, dos capas LSTM de 64/32 unidades, dropout 0,2, Dense(16, ReLU) y salida lineal escalar; usa Adam 10⁻³, batch 32, máximo 200 épocas y paciencia 15. Es una variante más profunda que la ficha concreta de S5. El PDF declara doce atributos, mientras el código enumera ocho, incluida varianza condicional EGARCH; no se trata de las cinco OHLCV del proyecto. El objetivo del código es desviación móvil futura de treinta retornos, anualizada, no σ7 diario.

Tabla 4.5 reporta LSTM RMSE 0,0358, MAE 0,0254 y MAPE 13,5 %, sin dispersión entre semillas; son valores del autor, no verificados mediante reproducción. El código ajusta MinMaxScaler de entradas y objetivo sobre todo `df_garch` antes del corte temporal, riesgo aparente de fuga de información. También monitoriza `val_loss`, frente a MAE descrita en el informe, y los recuentos declarados 3.849+1.006=4.855 no reconcilian el total 4.955. Estas discrepancias impiden usar la fuente para acreditar superioridad o una réplica exacta y no se corrigen inventando resultados o dimensiones. Se cuenta una sola vez como apoyo exploratorio de familia.

#### Ecuaciones, flujo y sesgo inductivo

Para x_j∈R^p, h_j,c_j∈R^H y activación logística s:

$$
i_j=s(W_ix_j+U_ih_{j-1}+b_i),\quad
f_j=s(W_fx_j+U_fh_{j-1}+b_f),\quad
o_j=s(W_ox_j+U_oh_{j-1}+b_o),
$$

$$
g_j=\tanh(W_gx_j+U_gh_{j-1}+b_g),\quad
c_j=f_j\odot c_{j-1}+i_j\odot g_j,\quad
h_j=o_j\odot\tanh(c_j),\quad
\widehat y=W_yh_D+b_y.
$$

```text
Ventana histórica D × p
       ↓ lectura secuencial con parámetros compartidos
LSTM: compuertas i/f/o + memoria c + estado h
       ↓ último estado H
Cabeza densa H → q
```

Las compuertas controlan el almacenamiento y olvido del estado; su memoria ofrece un mecanismo diferente al de filtros locales o atención directa. Esto no garantiza aprender dependencias largas en una muestra pequeña ni acredita superioridad predictiva. El orden de los datos y la disponibilidad de las entradas siguen siendo parte del contrato causal.

#### Parámetros y costo analítico

Una LSTM con un vector de sesgo por compuerta y cabeza de q salidas tiene:

$$
P_{\mathrm{LSTM}}=4H(p+H+1)+Hq+q.
$$

En una implementación con dos vectores de sesgo por compuerta, como la parametrización de `torch.nn.LSTM`, la expresión es:

$$
P_{\mathrm{LSTM,2bias}}=4H(p+H+2)+Hq+q.
$$

Para la configuración de S5, p=1,H=20,q=1, la primera expresión da **1.781 parámetros**, cálculo analítico de las capas descritas y no un total reportado por el artículo. Una adaptación con p=5,H=20,q=1 daría **2.101** con un sesgo o **2.181** con dos. Estos últimos conteos son hipótesis de adaptación, no tamaños de un modelo del proyecto ya implementado. Cambiar D altera la longitud del cómputo, pero no el número de pesos de una LSTM con salida tomada del último estado. Con p=1, H=261 y q=96 de S1, la fórmula de un sesgo produce **299.724 parámetros**; es un cálculo derivado de la configuración ganadora de Tabla 5, no un total textual ni una medición ejecutada. La diferencia frente a S5 refleja configuraciones y tareas distintas.

Por ventana, el costo dominante de inferencia es O(D·H(p+H)+Hq). Un minibatch de B ventanas multiplica ese costo por B; la retropropagación mantiene el mismo orden, con mayor constante y almacenamiento de estados O(BDH). La recurrencia impone dependencia secuencial entre pasos. No se infiere tiempo real ni consumo de GPU a partir de esta expresión.

#### Resultados publicados y límites de comparabilidad

| Fuente y evaluación | Resultado publicado | Dispersión y escala |
|---|---|---|
| S1, Tabla 2, test de 48 días | LSTM RMSE 2,37; MAE 1,70; MAPE 0,30; MSLE 0,09 | DE 0,19 / 0,26 / 0,08 / 0,01, respectivamente, entre veinte corridas; RV = 100·sqrt(sum(r_15min²)); MAPE como fracción |
| S5, Tabla 4, comparación de validación | LSTM(20) RMSE 0,074071; RMSPE 0,406084 | Sin dispersión entre semillas; comparación sobre valores escalados y objetivo de treinta días desplazado siete días |

No se ordenan estos RMSE como si midieran el mismo problema. S1 predice retornos intradiarios y después agrega volatilidad; S5 emplea volatilidad pasada como entrada y no centra los retornos al construir su medida. El contrato del proyecto conserva OHLCV diario y desviación poblacional centrada de siete retornos. La pérdida MSE fijada para el diseño del proyecto coincide nominalmente con S5, pero no con la pérdida ε-insensible de S1; que una pérdida comparta nombre tampoco iguala escalas y objetivos.

**Fortalezas:** comparador recurrente documentado en tres fuentes unidireccionales, con variantes distintas, arquitectura pequeña completamente identificable en S5 y número de parámetros independiente de D. **Límites:** S5 contiene fechas inconsistentes y carece de repeticiones e inferencia; S1 dispone de veinte corridas, pero solo 48 días de test. La variante, transformación, activación de salida y configuración propia deben distinguirse de la réplica de cada fuente. No se atribuye a las fuentes una salida `softplus`: sus cabezas descritas son lineales.

### 3.3.2. TCN: convolución temporal causal y dilatada

#### Referencias y papel comparativo

S1 documenta una **TCN única** como comparador de LSTM, GRU, AR-RV y GARCH. Luego propone una **D-TCN de dos ramas**, una para retornos y otra para señales de Twitter. La ficha selecciona la TCN comparativa, no la D-TCN novedosa. La base conceptual de convoluciones causales y dilatadas está descrita por Bai, Kolter y Koltun (2018), referencia de fundamentos del libro; ese trabajo genérico no se cuenta como otro estudio de Bitcoin.

Fuentes: S1, Fig. 1, §§5.2–5.4 y Tabla 2; [hpo.py por commit](https://github.com/meakbiyik/ask-who-not-what/blob/bcc0e4e89213da2ca3ed968574aa2d08cee39c44/scripts/hpo.py), líneas 151–235. `dtcn.py` corresponde a la extensión propuesta y no debe utilizarse como si describiera la base de una sola rama.

#### Arquitectura y entrenamiento publicados

El código invoca `TCN` de `keras-tcn` con seis niveles de dilatación `(a⁰,a¹,a²,a³,a⁴,a⁵)`, a∈{2,3,4}, kernel k∈{2,…,6}, canales F∈{32,…,512}, dropout en [0,0,5], skip connections activadas o no y alternativas de normalización batch/weight/layer/ninguna. Después se aplica Dense(96) y reshape a 96×1 retornos futuros. Entrada fuente: 96×1 retornos observados. El artículo compara modelos con información financiera de precios antes de introducir las ramas de Twitter.

La selección utiliza 250 configuraciones, treinta épocas por configuración, AdamW y pérdida cuadrática ε-insensible sobre retornos. Los rangos de learning rate, weight decay y ε son los mismos de la ficha LSTM. Los veinte reinicios de la comparación final se evalúan después de agregar los 96 retornos predichos a RV diaria. El apéndice C, Tabla 6 del artículo, sí publica los hiperparámetros ganadores: F=287 filtros, kernel k=5, base de dilatación a=4, dropout 0,217, ε=0,0913, learning rate 6,49×10⁻⁵, weight decay 5,93×10⁻⁶, skip connections activadas y ninguna normalización. Se corrige la omisión de esa tabla en la extracción anterior. Sigue sin estar fijada la versión de `keras-tcn`, por lo que los detalles internos de sus bloques deben verificarse antes de una réplica.

#### Ecuaciones y diagrama

Para una convolución causal con dilatación d_b:

$$
u_{j,f}=b_f+\sum_{i=0}^{k-1}\sum_{c=1}^{C_{in}}W_{i,c,f}\,x_{j-d_b i,c}.
$$

Solo intervienen índices anteriores o iguales a j; el padding es a la izquierda. En un bloque residual representativo de dos convoluciones:

$$
H^{(b)}=\phi\!\left(F_b(H^{(b-1)})+S_b(H^{(b-1)})\right),
$$

donde F_b representa convoluciones, activaciones, dropout y la normalización que se declare, y S_b es identidad o proyección 1×1 para igualar canales. La representación temporal final alimenta la cabeza densa.

```text
Ventana pasada D × p
       ↓ convoluciones 1D causales
Bloques residuales con dilataciones crecientes
       ↓ representación temporal F
Cabeza densa F → q
```

El sesgo inductivo es localidad temporal con pesos compartidos y contexto ampliado mediante dilataciones. Se diferencia de almacenar una memoria recurrente y de relacionar todas las posiciones mediante atención. Atender una ventana pasada no elimina el requisito de que las variables originales y sus transformaciones fueran observables en cada origen.

#### Parámetros, profundidad y complejidad: hipótesis explícitas

La evidencia directa fija **seis niveles de dilatación**. El número de convoluciones por bloque depende de la implementación residual de la biblioteca. El esquema habitual de dos convoluciones por bloque daría doce convoluciones, pero este conteo **no aparece como total textual del artículo ni sustituye la versión ausente de la dependencia**.

Una convolución con sesgo aporta k·C_in·C_out+C_out parámetros. Para J bloques de dos convoluciones, ancho constante F, p≠F, una sola proyección residual inicial p→F, sin normalización entrenable y una cabeza F→q, se deriva:

$$
P_{\mathrm{TCN}}=(k+1)pF+(2J-1)kF^2+(2J+1)F+Fq+q.
$$

El campo receptivo de ese esquema es:

$$
R=1+2(k-1)\sum_{b=0}^{J-1}d_b.
$$

Con los hiperparámetros ganadores F=287, k=5, base a=4, J=6, p=1 y q=96, la fórmula bajo esas hipótesis da **4.563.396 parámetros** y un campo receptivo formal **R=10.921**. Son cálculos condicionados a dos convoluciones por bloque y un stack, no totales publicados ni mediciones de una implementación. R excede los 96 retornos observados: no crea historia disponible adicional.

Ejemplo **analítico dentro del espacio descrito**, no configuración ganadora: J=6,F=32,k=2,p=1,q=96, sin normalización entrenable, da 26.208 parámetros y R=127 para dilataciones 1/2/4/8/16/32. Una adaptación p=5,q=1 bajo esas mismas hipótesis daría 23.457 parámetros. La normalización, otros stacks o una estructura residual distinta cambian el conteo. No se presentan estos valores como observados en un entrenamiento.

El costo dominante por ventana es O(D·Σ_l k_l C_(l−1)C_l+Fq), aproximadamente O(DJkF²) en la parte de ancho constante. Para minibatch B, se multiplica por B. El entrenamiento añade almacenamiento de activaciones y retropropagación del mismo orden. Las operaciones pueden paralelizarse dentro de la ventana; esto no demuestra menor latencia real que LSTM. Si D es de 7–28 días, un campo receptivo formal mayor que D no crea historia observada adicional: parte del contexto correspondería a padding.

#### Resultados publicados, fortalezas y limitaciones

S1, Tabla 2, informa **TCN RMSE 2,58±0,07; MAE 1,63±0,06; MAPE 0,23±0,01; MSLE 0,11±0,01**, veinte corridas, test de 48 días y la misma escala RV de la ficha LSTM. En ese estudio, TCN tiene mejor MAPE/MAE que LSTM, mientras LSTM tiene menor RMSE/MSLE. El orden depende de la métrica; no se comunica un ganador universal. Los contrastes del artículo son unilaterales respecto a AR-RV y no implican automáticamente un contraste directo entre las redes ni control de toda la multiplicidad.

**Fortalezas:** comparador separado de la propuesta híbrida, filtros compartidos y contexto multiescala con cálculo paralelo. **Límites:** frecuencia 1/5 en el corpus exploratorio, detalles internos dependientes de una biblioteca sin versión fijada y test corto. La fuente respalda TCN y no acredita una CNN1D de una sola capa como réplica exacta. «Base comparativa» no significa mínima profundidad absoluta.

Para el proyecto, pasar de retornos intradiarios a cinco OHLCV diarios y de 96 salidas a una volatilidad escalar sería una adaptación. Su profundidad, dilataciones, normalización y salida deben registrarse antes de implementar en E2. Si falta detalle de arquitectura, corresponde consultar el protocolo de E2 §14 citado por la guía; esta ficha no inventa sus reglas.

### 3.3.3. Transformer encoder compacto, comparador SELF-ATT

#### Referencia, estado y papel experimental

**S2.** Zeri, T. (octubre de 2026). *Forecasting High-Frequency Bitcoin Volatility: Neural Networks, Market Microstructure and On-Chain Data*. Manuscrito público de investigador independiente, sin DOI ni revisión por pares acreditados. [Fuente TeX fijada](https://github.com/tommasozeri/Forecasting-High-Frequency-Bitcoin-Volatility-Neural-Networks/blob/d69f951f2ced7273bdc4c3e5a8a4cb220c2ee5ce/paper_en/manuscript.tex), §§Models/Training, líneas 179–189, tabla de resultados y apéndice, líneas 395–409. Arquitectura leída en celda 21 del notebook `modelli_btc_hf.ipynb`, mismo commit.

SELF-ATT es uno de los tres comparadores neuronales genéricos —junto a GRU y BiLSTM— del estudio; la contribución se centra en pipeline, comparación y ablación de atributos, no en declarar una nueva arquitectura de atención. La ficha concreta describe S2. El encoder aparece además en S6, con una configuración distinta: **2/5 textos del corpus exploratorio**, ambos sin revisión por pares acreditada; esto no prueba recurrencia en toda la literatura. Es compacto frente a grandes Transformers, pero tiene más parámetros que la LSTM simple de S5; no se le atribuye mínima complejidad absoluta.

#### Arquitectura y protocolo documentados

Run B utiliza 48 barras de cinco minutos y 41 atributos de Binance Futures BTCUSDT perpetuo mediante Crypto Lake. El objetivo es log-RV de los siguientes treinta minutos, seis barras, a partir del precio medio del libro de órdenes; no es desviación móvil diaria de siete retornos. Hay 31.524 predicciones de test, correspondientes al 13/09–31/12/2025. La tabla publica métricas de predicciones promediadas entre tres semillas, no media y desviación de tres métricas.

La arquitectura contiene proyección lineal F→d, posiciones aprendidas de longitud máxima M, dos bloques encoder pre-LayerNorm y cuatro cabezas. Cada bloque incluye atención multihead, dos conexiones residuales y feedforward de ancho 2d. El código no especifica la activación del encoder, por lo que usa ReLU por defecto; la cabeza sí usa GELU. La cabeza concatena último token y media de tokens y aplica dropout→Dense(2d,d)→GELU→Dense(d,1).

La configuración elegida reportada para SELF-ATT en Run B es d=48, dropout 0,21, learning rate 3·10⁻⁴, weight decay 1,4·10⁻³, ruido de entrada 0,12 y ventana L=48. El código conserva M=96 posiciones, aunque la ventana elegida sea menor. La pérdida es MSE sobre el objetivo log-RV estandarizado, según la línea 119 del manuscrito y `loss_fn` en la celda 23. Se declara AdamW, batch 512, máximo cuarenta épocas, paciencia seis, clipping 1 y semillas 42/43/44; cada red recibe veinte trials de búsqueda. Las predicciones finales se promedian entre semillas antes del cálculo de métricas.

#### Apoyo adicional de encoder Transformer: S6

**S6.** Wang, Y., Yu, Q., Peng, J., Yang, B. y Chen, Z. (s.f.). *A comparative analysis of volatility predictions via EGARCH, Transformer and hybrid EGARCH-Transformer models: changes in forecasting ability after introducing sentiment factors*. [Borrador de autores](https://github.com/cathyw111/bitcoin-volatility-forecasting/blob/61feeec81d695e19eac9c819be41d704e6279547/research-draft.pdf), §3.4 y Tablas 1–2, pp. 6–7, y Tabla 4, p. 10. La versión del repositorio está disponible en 2026; la fecha de commit no se presenta como año editorial. No se acreditan DOI ni revisión por pares.

Su comparador `model1` es un encoder Transformer sobre OHLCV diario de Bitcoin, con historia de 20 días, d=32, cuatro cabezas, dos capas, dropout 0,1 y salida Softplus. Declara MSE, Adam 0,001, batch 32, máximo 100 épocas, paciencia 10 y semilla 42. El periodo es 25/03/2021–30/12/2024 y el objetivo declarado RV1D, cuya construcción exacta no quedó suficientemente especificada. Faltan ancho feedforward, pooling/lectura, orden de normalización y versión de implementación; por ello no se inventa un total de parámetros ni se traslada el conteo 49.249 de S2 a S6.

Tabla 4 informa para `model1` RMSE 1,4738, R² −0,7347, QLIKE 0,4369 y MAE 0,4369, sin dispersión. Se conservan literalmente, sin tratarlos como cifras validadas: el texto llama mejor al MAE 0,7399 de `model9` aunque 0,4369 es menor; la combinación RMSE/R² no permite reconciliar una cohorte común con la definición convencional, y el corte 70/30 de Transformer no está reconciliado con los 203 días de test EGARCH. No se ejecutó DM por costo computacional, según §6.2. Sin datos o código de entrenamiento no puede verificarse disponibilidad, escalado ni el origen de esas diferencias. La fuente respalda la presencia de un comparador de atención con información OHLCV, pero no acredita una mejora reproducida ni cierra su especificación.

#### Ecuaciones, flujo y causalidad

$$
H^{(0)}=XW_{in}+b_{in}+P,\qquad
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Con d=48 y cuatro cabezas, d_k=12 por cabeza. Esta notación desagrega el espacio de atención; el manuscrito usa un símbolo d genérico. Los bloques son pre-LN con suma residual tras atención y tras feedforward. La salida usa:

$$
z=\left[H^{(2)}_L;\,\frac1L\sum_{j=1}^{L}H^{(2)}_j\right],\qquad
\widehat y=w_2^\top\operatorname{GELU}\!\left(W_1\operatorname{Dropout}(z)+b_1\right)+b_2.
$$

```text
Ventana histórica L × F
       ↓ proyección a d + posiciones aprendidas
Encoder pre-LN ×2: atención 4 cabezas → residual → FF(2d) → residual
       ↓ concatenar último token y media
Dropout → Dense(2d,d) → GELU → Dense(d,1)
```

El sesgo inductivo permite relacionar posiciones distantes mediante atención dependiente del contenido, sin recurrencia ni una vecindad local fija. El encoder se llama sin máscara causal: es compatible con pronóstico si toda su ventana está disponible antes del origen. Atender posiciones posteriores dentro de una ventana enteramente pasada no implica por sí solo utilizar el futuro. La construcción causal de cada atributo debe verificarse por separado.

#### Complejidad y número de parámetros derivado

Para feedforward m=2d, F atributos y M posiciones aprendidas, cada bloque encoder aporta:

$$
P_{encoder}=4d^2+2dm+9d+m=8d^2+11d.
$$

La proyección aporta (F+1)d; posiciones, Md; y cabeza, 2d²+2d+1. Para dos bloques:

$$
P_{\mathrm{SELF-ATT}}=18d^2+(F+M+25)d+1.
$$

F=41,M=96,d=48 da **49.249 parámetros**: 2.016 de proyección, 4.608 de posiciones, 37.920 de encoder y 4.705 de cabeza. Es un cálculo analítico de capas leídas, **no un total reportado por el manuscrito ni una medición por instanciación**. Cambiar F a cinco o modificar el máximo de posiciones cambia el conteo; no se presenta 49.249 como el tamaño del futuro modelo OHLCV del proyecto.

El costo por bloque y minibatch B es O(B(L²d+Ld²)), más proyección O(BLFd) y cabeza O(Bd²). El almacenamiento de atención es cuadrático en L por cabeza. La retropropagación conserva el orden con mayor constante. Estas expresiones no prueban eficiencia ni escalabilidad medida frente a otras redes.

#### Valores publicados y crítica

| Modelo de S2, Run B | R² respecto a media train | R² respecto a media test | QLIKE | Spearman |
|---|---:|---:|---:|---:|
| SELF-ATT | 0,600 | 0,566 | 0,562 | 0,749 |
| BiLSTM | 0,586 | 0,550 | 0,582 | 0,742 |
| GRU | 0,582 | 0,546 | 0,564 | 0,743 |
| HAR | 0,508 | 0,466 | 0,773 | 0,676 |

El manuscrito reporta DM de SELF-ATT frente a HAR de 19,7, p unilateral <10⁻¹⁵ y HAC de cinco rezagos por solapamiento. No es un contraste directo SELF-ATT contra BiLSTM ni una prueba ejecutada por este proyecto. R² respecto a media train no equivale al R² convencional respecto a media test. No hay en esa tabla desviación ni IC entre las tres semillas: se evalúa su promedio de predicciones.

**Fortalezas:** mecanismo distinto, encoder/cabeza descritos y configuración acotada; relaciones directas entre posiciones de la ventana. **Límites:** los dos apoyos aplicados de encoder son literatura gris; S6 tiene especificación parcial y métricas inconsistentes. La ficha concreta de S2 utiliza datos licenciados, una ventana de evaluación y costo cuadrático, con resultados no reproducidos aquí. Además, las redes utilizan 24–48 barras mientras los referentes lineal y LightGBM usan solo la última: la comparación no aísla completamente arquitectura de historia informativa. El desglose de atributos de libro tiene una discrepancia de enumeración; se conserva F=41 como dimensión declarada, con esa salvedad.

La fuente tiene salida lineal para log-RV estandarizado. Adaptarla a OHLCV diario y volatilidad no negativa requiere un contrato propio; añadir `softplus` o reducir capas no sería una descripción de la arquitectura publicada. No se fijan esas modificaciones a partir de cifras de otro objetivo.

### 3.3.4. Comparación documental y cobertura de las fichas

| Dimensión | LSTM | TCN | Transformer encoder |
|---|---|---|---|
| Mecanismo | Estado recurrente y memoria con compuertas | Filtros causales compartidos y dilatados | Atención por contenido entre tokens de la ventana |
| Fuente de arquitectura | S5 concreta; S1 comparativa y ganadora en Tabla 5; S7 apoyo distinto | S1 comparativa y ganadora en Tabla 6 | S2 concreta; S6 apoyo parcial distinto, ambos grises |
| Frecuencia del corpus exploratorio | Familia 4/5; unidireccional 3/5 | 1/5 | 2/5, ambas fuentes grises |
| Profundidad identificada | S5 una capa, H20; S1 H261 | Seis niveles, F287, k5 y base4; convoluciones por bloque dependen de biblioteca no fijada | Dos encoder, cuatro cabezas |
| Parámetros fuente | 1.781 derivados para S5; 299.724 para S1 con un sesgo | 4.563.396 condicionados al bloque residual de dos convoluciones y un stack | 49.249 derivados para Run B |
| Datos/objetivo idénticos a E1 | No | No | No |
| Configuración OHLCV propia congelada | No; adaptación distinguida | No; falta variante y protocolo aplicable | No; cambio de frecuencia, dimensión y salida distinguido |

Las fichas aportan evidencia, ecuaciones, diagramas, resultados publicados y análisis de complejidad, con los límites y detalles no verificados identificados. No completan una búsqueda de hiperparámetros propia ni convierten una fuente en treinta estudios. La diversidad de mecanismos no acredita por sí sola mayor recurrencia y mínima complejidad; la selección definitiva sigue pendiente. Si el criterio académico exige exclusivamente estudios revisados por pares, S2/S6 no acreditan todavía el respaldo del encoder y S7 tampoco cumple ese filtro. La extracción definitiva de tres familias permanece abierta.
