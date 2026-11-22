# Batería temporal prospectiva, versión 3

**Especificación vigente: `temporal-fixed-seeds-mse-v3`. R3 no acreditada.**
La reparación v3 conserva los fallos v1/v2 y cambia sustantivamente la
calibración; su plan y todos los hashes se fijaron antes de su simulación.
Dictamen independiente: `qa/statistical_validation/v3/report.json`.

## Especificación operativa v3

Se conserva el estimando prospectivo de media temporal del MSE medio sobre
semillas fijas; su RMSE agregada es una transformación monótona, y se conserva
por separado la media histórica de RMSE individuales. No se transforma ningún
error de ensamble en métrica individual. El cambio respecto al endpoint E1
requiere alineación explícita; la media histórica de RMSE favorable es un
resguardo adicional, no un p-valor de esa métrica. Los modelos y semillas están
condicionados, sin convertir R×T en observaciones iid.

La API anterior sigue disponible, con parámetros adicionales opcionales:

```python
result = analyze(
    predictions, y, systems,  # [J,R,T], [T], candidata primero
    bootstrap_reps=8191, block_length=32, hac_lags=48,
    instruments=z,           # [T,>=2]: 1 + volatilidad conocida al origen
    margin=.02, seed=2026112103, horizon=1,
    reference_reps=49999, reference_seed=2026112103,
    bootstrap_backend="auto",  # C++ opcional o numpy equivalente
)
```

Estos valores corresponden al **plan sintético**, no a un dimensionamiento de
mercado inventado. Familia completa: 20 rivales, tres métodos, EPA y margen,
120 hipótesis con Holm conjunto. Se conservan p crudos y ajustados,
estadísticos, direcciones e inferioridad. `d_t=mean_s(q_j−q_c)` y
`w_t=.98²*mean_s(q_j)−mean_s(q_c)` mantienen las definiciones explicadas abajo.

| Procedimiento v3 | Hipótesis y calibración vigente |
|---|---|
| DM–HLN/HAC | `E[d]=0` o `E[w]=0`; estadístico media/SE HAC y factor HLN; referencia **fixed-b escalar simulada** con el mismo T, K, kernel y factor. Sustituye la t(T−1) de v2. |
| Bootstrap estacionario studentizado | Mismos nulos de media; cada réplica usa `t*=(mean*−mean)/sqrt(HAC*/T)`, recalculando HAC en esa réplica. Sustituye el bootstrap centrado sin studentizar. |
| GW condicional/HAC | `E[h_t d_t]=0` o `E[h_t w_t]=0`; Wald `T*gbar' Ω⁻¹gbar`, referencia **fixed-b vectorial q-dimensional**. Sustituye χ²; no son q copias de DM. |

GW conserva constante más instrumento predecible no constante; puede rechazar
por previsibilidad aun con media cero y no demuestra dirección por sí mismo.
La superioridad exige además medias positivas y DM/bootstrap favorables. No
se fuerza Clark–West sin anidamiento ni se fabrica un MCS direccional. Los
tres métodos son familias distintas de F2; sus nulos no se presentan como
idénticos ni como resultados independientes.
El campo de alerta heredado `significant_inferiority` se conserva en la fuente
congelada y puede incluir un rechazo GW con media descriptiva negativa. La
decisión direccional del evaluador debe exigir EPA de **DM o bootstrap**
significativo y media negativa; GW aislado es evidencia condicional distinta,
no prueba de inferioridad media. Este resguardo se aplica fuera del kernel y
no altera ningún estadístico, p-valor o intervalo de la auditoría congelada.

### Fundamento de la referencia fixed-b

La fuente primaria verificada es Kiefer y Vogelsang (2005), documento CAE
05-08, Teorema 3 y §3.3, p. 12. La ubicación y URL consultadas constan en
`v3/primary_source.json`. No se inventó un hash del PDF cuando su descarga
directa no estuvo disponible.

La aplicación concreta es el modelo de ubicación de los momentos: bajo H0,
si `T^(-1/2) sum_{t<=Tr} g_t ⇒ Σ^(1/2) B_q(r)`, los parciales centrados
convergen al puente browniano transformado. Con bandwidth proporcional,
`HAC ⇒ Σ^(1/2) Q_q(b) Σ^(1/2)'`. El Wald converge entonces a
`B_q(1)' Q_q(b)^(-1) B_q(1)`: la matriz desconocida se cancela por una
transformación invertible. El caso q=1 da el pivot t escalar. Para GW se aplica
a `g_t=h_t d_t` o `h_t w_t`; requiere FCLT **vectorial** y LRV no singular.

Se simulan 49999 realizaciones gaussianas iid de dimensión q, calculando el
mismo estadístico en la malla T. Ésta es una aproximación discreta del pivot,
**no la distribución finita exacta bajo dependencia arbitraria**. La
referencia escalar incluye el factor HLN tanto observado como simulado; no se
promete una mejora adicional independiente por aplicar esa misma escala a
ambos. La caché depende de T, K, q, h, número de realizaciones y semilla fija,
nunca de pérdidas observadas. Se registran hashes de las referencias. Para
Bartlett se usa `b=(K+1)/T`, coherente con el peso `1−lag/(K+1)`.

Los tres métodos reportan `mc_tail_count`, `mc_reps` y
`mc_p_rule="absolute_tail_plus_one"`; el p es `(count+1)/(N+1)`, contando
estadísticos absolutos al menos tan extremos. GW es no negativo. La
incertidumbre Monte Carlo de cola debe considerarse en **los tres métodos**,
incluidas las referencias DM/GW. El mínimo posible ahora es `1/(N+1)`;
para B una resolución aritmética suficiente al primer paso de Holm120 exige
B≥2399. Esto no demuestra precisión de colas ni potencia.
Los CP de la auditoría se refieren a réplicas temporales independientes
condicionadas a las referencias gaussianas ya congeladas; no integran la
incertidumbre de volver a simular otra tabla de referencia. Los conteos y la
semilla fija permiten al planificador tratar esa incertidumbre por separado.

### Intervalos y regla más fuerte

El bootstrap-t vuelve a estimar HAC en cada réplica y usa el intervalo
pivotal `mean−quantile(t*)*SE_observado`. Para la ganancia se studentiza
`log(Qc/Qj)` con score delta `qc_t/Qc−qj_t/Qj`, usando medias y HAC nuevas en
cada réplica, y se transforma a `1−exp(logratio/2)`.

Se reportan **CI95 reales y CI99 reales**, sin renombrar los segundos. El gate
v3 exige los seis rechazos Holm, medias MSE y práctica positivas, media RMSE
histórica favorable, efecto temporal `mean(d)/sd(d)≥0.2`, límite inferior de
CI95 y CI99 de diferencia >0, y límite inferior de **CI99 de ganancia >2 %**.
La auditoría contrasta cobertura efectiva ≥0.95 del procedimiento operativo
CI99 y reporta por separado cualquier infracobertura detectada de CI95. Es un
intervalo nuevo más conservador, no una relajación retroactiva de v1/v2.

Las fechas permanecen dependientes. Son necesarios estacionariedad,
dependencia débil y los FCLT relevantes, instrumentos realmente conocidos al
origen, momentos suficientes y covarianza no singular. h=1 no elimina el
solapamiento: K y longitud de bloque deben fijarse en desarrollo y respetar el
soporte de las etiquetas. Los resúmenes diagnósticos no certifican esos
supuestos. Ningún resultado admite por sí solo incertidumbre de selección,
partición o nuevas semillas.

### Implementación exacta y aceleración opcional

La HAC móvil usa `Ω=sum_t S_t S_t' / [T*(K+1)]`, donde S son sumas de ventanas
de la serie centrada con ceros en ambos extremos. Cada pareja de lag k
aparece K+1−k veces, reproduciendo exactamente Bartlett con divisor T. El
algoritmo O(T) se contrasta en pruebas contra la matriz de la fórmula por lags.

El kernel C++ local `v3/bootstrap_kernel.cpp` acelera sólo esa aritmética;
no genera RNG, p-valores ni decisiones. Se compila con argv fijo, `-O3`, sin
`-ffast-math`, en directorio temporal privado y hash de fuente+flags. Si no hay
compilador, `auto` usa numpy; `native` explícito informa indisponibilidad. La
salida indica el backend. Pruebas comparan ambas rutas en medias, t, CIs,
conteos y p, con escalas de pérdida 10⁻¹² y 10¹². Los índices se generan en
slabs de hasta 64 MiB; la aritmética usa lotes pequeños, sin B×T×J permanente.

Las 19 pruebas deterministas pasan: incluyen los casos anteriores, HAC móvil
matricial exacta, varianza por réplica, pivot vectorial invariante, metadatos
de colas, intervalos CI99 más anchos y equivalencia numpy/nativa. Certifican
estos casos de software, no una garantía finita de mercado.

### Evidencia v3 y reproducción

`v3/plan.json` y `v3/freeze.json` se fijaron antes de observar la nueva MC:
192×5 réplicas, T768, diez semillas fijas, 20 rivales, B8191, HAC48, bloques32,
referencias49999 y techo900 s CPU. La decisión se toma con CP95 exactos:
FWER sólo demostrado si su límite superior ≤0.05; cobertura operativa sólo si
el inferior ≥0.95; potencia por rival sólo si el inferior ≥0.90. Los demás
casos quedan inconclusos o fallidos según los criterios fijados. No se ocultan
CI95 adversos ni se llama validación general a un resultado acotado.

`v3/replicates.jsonl` conserva cada réplica y todos los p crudos/ajustados,
estadísticos, direcciones y conteos MC en orden declarado. `v3/report.json`
contiene dictamen, intervalos, hashes de referencias y cómputo. Su campo
`general_validation_established` permanece falso y
`protocol_closure_permitted` falso: faltan dimensionamiento real, admisión y
evaluación confirmatoria de paneles y requisitos restantes de R3. Un eventual
`admitted=true` sólo se refiere al alcance explícito de los cinco DGPs.

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p test_statistics.py -v
OPENBLAS_NUM_THREADS=1 python qa/statistical_validation/v3/run_validation.py --output-dir /tmp/temporal-v3-reproduction
```

El runner verifica todos los hashes congelados y exige un directorio nuevo.
Referencia primaria: [Kiefer–Vogelsang, CAE 05-08, 2005](https://cae.economics.cornell.edu/05-08.pdf).

## Registro histórico v1/v2: procedimiento anterior y fallos preservados

**Lo que sigue documenta v1/v2; las referencias t/χ², bootstrap no
studentizado e intervalos básicos de estas versiones no son la especificación
operativa v3 anterior.**

Estado: **código implementado; R3 no acreditada**. La primera validación sintética
falló. Su plan, código y resultados se conservan. La segunda auditoría tiene un
plan distinto, fijado antes de su ejecución, y no rehabilita el régimen pequeño
que falló. El estado preciso de esa auditoría está en
`qa/statistical_validation/report_v2.json`; no se sustituye por la potencia de un
caso favorable.

## Cambio de estimando y alcance de R3

La guía original, pp. 23–28, exige métrica media favorable frente a cada rival,
tres pruebas pertinentes de su Tabla 4/F2, Holm, margen práctico, efecto e
intervalo, consistencia adicional, adecuación y robustez. La ruta de t,
bootstrap y Wilcoxon **sobre semillas** de un calendario fijo no satisface la
batería temporal solicitada y no se reutiliza aquí con otro nombre.

La versión `temporal-fixed-seeds-mse-v2` registra prospectivamente, antes de
resultados confirmatorios nuevos, la métrica primaria

\[
 Q_j = E_t\{R^{-1}\sum_{s=1}^R (\hat y_{jst}-y_t)^2\}.
\]

Las ejecuciones y modelos ajustados están **condicionados y congelados**. La
expectativa se refiere a la secuencia temporal del proceso, bajo los supuestos
descritos abajo. La estimación es la media de pérdidas cuadráticas de las
semillas fijas por fecha y luego la media sobre fechas. La raíz `sqrt(Q_j)` es
la RMSE del MSE agregado, una transformación monótona de esta métrica. Se elige
porque produce una pérdida aditiva por origen y permite usar una batería F2
coherente sin convertir R×T observaciones correlacionadas en réplicas iid.

Se conservan por separado en `metrics`:

1. `mse` y `rmse_pooled = sqrt(mean_s,t(error²))`;
2. `mean_seed_rmse = mean_s(sqrt(mean_t(error²)))`, su dispersión y cada RMSE
   individual: ésta es la agregación histórica;
3. el error de un ensamble **no se calcula ni sustituye** a las anteriores:
   `sqrt(mean_t((mean_s(pred)-y)²))` corresponde a otro predictor.

Las dos primeras agregaciones pueden incluso ordenar los sistemas de manera
opuesta. Por ejemplo, dos semillas de errores constantes 0 y 2 producen una
media de RMSE de 1 y una raíz del MSE agregado de √2; un rival con errores 1.3
en ambas semillas queda entre ambas cifras. El ensamble del primer sistema
tendría RMSE 1 y constituye otro objeto. Hay una prueba determinista de este
caso.

El cambio de endpoint requiere el cierre explícito del protocolo prospectivo y
su alineación con el endpoint fijado en E1; **no se declara automáticamente
compatible con R3 histórica**. Como resguardo adicional, el gate estadístico
también exige que la media histórica de RMSE sea favorable. Este resguardo no
convierte los p-valores de MSE en p-valores para la media de RMSE individuales.
La versión 2 corrige tolerancias numéricas dependientes de las unidades; no
selecciona otra arquitectura ni usa resultados de mercado.

## API y datos requeridos

```python
from engine.statistics import analyze
result = analyze(
    predictions, y, systems,  # [J,R,T], [T], nombres; candidata primero
    bootstrap_reps=B, block_length=L, hac_lags=K,
    instruments=z,           # [T,>=2]: 1 + volatilidad disponible al origen
    margin=.02, seed=20261121, horizon=1,
)
```

Todas las fechas deben estar ordenadas y alineadas. El evaluador externo debe
verificar el orden, los timestamps, las 21 identidades y la familia completa;
la API admite familias pequeñas para fixtures, pero no las acredita como la
familia confirmatoria. Debe guardar exactamente las mismas predicciones que
entrega a esta función. No se ajusta ningún modelo en `analyze`.

Los instrumentos deben proceder de información demostrablemente disponible al
origen: una constante y volatilidad observada con retraso suficiente son una
elección concreta. Con etiquetas solapadas, `y[t-1]` puede todavía no ser
observable; no se fabrica automáticamente ese instrumento. La constante sola,
dos columnas constantes o instrumentos colineales se rechazan. Si no se
entregan instrumentos, GW se marca no disponible con p=1 y el gate no pasa.
El centrado/escalado numérico de columnas es una transformación lineal
invertible de los instrumentos originales (incluida la constante); el Wald es
invariante a ella, y no modifica la información de los pronósticos.

La semilla del bootstrap sólo controla el generador de índices y no selecciona
semillas de entrenamiento. Un rival determinista puede repetir su traza en el
eje de semillas para alinear el tensor; `unique_prediction_seed_traces` deja
visible esa repetición y no se interpreta como ajustes independientes.

## Contrastes, dirección y margen

Para cada rival j y candidata c se construyen

\[
 d_t=\overline q_{jt}-\overline q_{ct},\qquad
 w_t=(1-.02)^2\overline q_{jt}-\overline q_{ct}.
\]

EPA usa el nulo de media cero de `d`. El nulo de frontera práctica usa `w`.
Una media positiva de `w` equivale exactamente a una reducción mayor al 2 % en
la raíz del MSE agregado: `1-sqrt(Qc/Qj) > .02`. No equivale a un margen del
2 % en MSE ni a un 2 % en la media de RMSE individuales.

Los p-valores son bilaterales; dirección y estadístico se conservan por
separado. Un p pequeño con `d<0` conserva evidencia de inferioridad. Un `w<0`
puede indicar sólo que falta el margen, por lo que **no se interpreta como
inferioridad en EPA**. El gate de superioridad exige medias de `d` y `w`
positivas y rechazo de ambas fronteras por todos los métodos. Es una condición
conservadora para las hipótesis unilaterales `E[d]<=0` y `E[w]<=0`.

| Procedimiento F2 | Hipótesis y estadístico | Dependencia y limitación |
|---|---|---|
| DM–HLN con HAC | `E[d]=0` o `E[w]=0`. Media / error estándar HAC, factor HLN por horizonte h y referencia t(T−1). | Bartlett/Newey–West con K positivo fijado en desarrollo; T fechas, no R×T. HLN corrige horizonte y no elimina el solapamiento de la etiqueta. Es aproximado en muestras finitas. |
| Bootstrap estacionario | Los mismos nulos de media, con distribución centrada de la media temporal y colas bilaterales con corrección +1. | Índices circulares con reinicios geométricos de longitud media L; **un mismo índice para todos los sistemas y semillas**. No se remuestrean semillas. Requiere bloques adecuados y consistencia bajo dependencia débil; las colas extremas pueden estar mal aproximadas. |
| GW condicional con HAC | Nulo de momentos `E[h_t d_t]=0` o `E[h_t w_t]=0`; Wald `T*gbar' Ω⁻¹ gbar`, χ² con K_h grados de libertad. | Constante más instrumento predecible variable, covarianza HAC de momentos y rango completo. Es una restricción instrumental de igualdad condicional, no una prueba de toda la función condicional ni de superioridad uniforme en todos los estados. |

GW no es DM con constante solamente. Puede rechazar por previsibilidad de la
diferencia aun si su media es cero: una prueba determinista verifica ese caso
y que el gate de superioridad permanece cerrado. La dirección de GW es la
dirección descriptiva de la media, **no** el signo de su Wald, que es siempre
no negativo. Se necesitan además media favorable y DM/bootstrap favorables.

Los tres procedimientos pertenecen a filas explícitas de F2: comparación
incondicional con referencia analítica, inferencia temporal no paramétrica y
restricciones condicionales con instrumentos. Sus nulos no son idénticos y no
se presentan como tres réplicas independientes. No se fuerza Clark–West sin
anidamiento, no se cuenta un cambio de cola/bandwidth como nueva prueba y no
se transforma un MCS global en p-valores direccionales por rival.

## Independencia, supuestos y diagnóstico

La unidad del proceso es la fecha/origen. Las fechas **no son independientes**:
se exige estacionariedad de pérdidas y momentos, dependencia débil compatible
con CLT temporal/HAC y bootstrap estacionario, momentos suficientes y
covarianzas no singulares. Las realizaciones aleatorias independientes son
los bloques de innovación del proceso bajo esas hipótesis; no las semillas
repetidas sobre un mismo calendario. La inferencia no integra incertidumbre
de selección, partición ni reentrenamiento. Una recomendación de diez semillas
no demuestra por sí misma potencia temporal o generalización a futuras
semillas.

K y L deben cerrarse con planificación sobre desarrollo, con crecimiento
compatible con T y con el soporte real de las etiquetas y entradas; nunca se
seleccionan sobre el test. `h=1` no autoriza HAC de lag 0 cuando la volatilidad
observada se solapa. La API rechaza K=0 y registra K, L, h, B, semilla y número
de instrumentos. El pipeline debe verificar causalidad y purgas aparte.

Se reportan autocorrelación de lag 1, medias de ambas mitades, varianza de largo
plazo y rango/condición de los momentos GW. `stationarity_verified=False`
evita convertir esos resúmenes en una certificación de estacionariedad. Esos
diagnósticos pueden revelar problemas, pero no justifican retocar la regla
después de ver el test. Se conservan las series/predicciones para diagnósticos
complementarios preregistrados.

Una varianza estimada nula, bootstrap constante o covarianza singular produce
estado explícito, estadístico nulo y p=1; no crea certeza infinita ni implica
equivalencia. Las tolerancias son relativas a la escala de pérdidas. Pruebas
con pronósticos y etiquetas reescalados por 10⁻⁶, 10⁻³, 10³ y 10⁶ comprueban
la invariancia de p-valores, direcciones y efecto.

## Familia, efectos y relación con R3

En un panel con 20 rivales se aplican **120 hipótesis**: 20×3 métodos×2
fronteras (EPA y margen). `holm` ordena todos los p y aplica el máximo
acumulado `(m-i+1)*p(i)`, preservando p crudo, ajustado, estadístico, dirección
y estado. Cada panel/ablación/sensibilidad requiere una familia declarada; la
regla entre paneles puede ser más restrictiva, pero no se relaja con un panel
favorable. Holm controla FWER si los p individuales son válidos; **no repara
una mala calibración finita**, como muestra la auditoría v1.

El p mínimo bilateral del bootstrap es `2/(B+1)`. Para que una familia de 120
pueda siquiera superar su primer umbral de Holm, B debe ser al menos 4799.
Es un límite de resolución aritmético, no evidencia de potencia, precisión de
colas ni suficiencia de B. El planificador debe usar la familia real y reportar
su incertidumbre Monte Carlo; no se impone un número universal ficticio de
eventos de cola.

Se reportan diferencia media de MSE, ganancia relativa de RMSE agregada,
efecto pareado temporal `mean(d)/sd(d)` y un intervalo marginal de 95 % para la
diferencia mediante bootstrap básico y para la ganancia mediante percentiles
temporales. El efecto es descriptivo de la distribución por fechas, no un
Cohen calculado sobre semillas supuestamente iid. Los intervalos no son
simultáneos para 20 rivales; la familia de pruebas sí usa Holm. No se compara
ese tamaño del efecto con una desviación estándar de medias que inflaría el
efecto según T.

El `provisional_statistical_gate` por rival exige los seis rechazos Holm,
direcciones favorables, margen, media histórica de RMSE favorable, efecto
temporal ≥0.2 y límite inferior del intervalo de diferencia >0. El agregado
requiere todos los rivales. `significant_inferiority` usa cualquier EPA
ajustado significativo con media negativa: no se pierde un adverso escogiendo
la prueba más favorable.

El código mantiene siempre `r3_accredited=False`: faltan resultados
confirmatorios admitidos, alineación del endpoint E1, potencia pertinente,
evaluación y consistencia de tres paneles adicionales, adecuación (residuos,
calibración, brechas) y robustez. Nadeau–Bengio corresponde a remuestreo de
particiones, que aquí no se realiza; aplicarlo a fechas de una reserva fija
sería una corrección sin objeto. La sensibilidad pertinente debe repetir la
inferencia temporal con reglas fijadas de antemano, conservando también
resultados no robustos. Si la guía exige literalmente el caso con/sin
Nadeau–Bengio, se documenta su falta de pertinencia a este diseño y se resuelve
esa alineación; no se inventa su ejecución.

Las 14 pruebas deterministas pasan. Incluyen la fórmula HLN/HAC, Holm con
empates, distinción de agregaciones, frontera práctica exacta, rechazo GW sin
ventaja media, mantenimiento de inferioridad, repetición de semillas sin
aumentar T, referencia determinista, casos degenerados e invariancia de
unidades. Esto certifica los casos de software comprobados, no calibración
estadística general.

## Evidencia y reproducción

`qa/statistical_validation/plan_v1.json` se congeló antes de ejecutar 400
réplicas (80 por escenario): T=192, diez semillas fijas, 20 rivales, B=4999,
HAC12 y bloque12. El hash y los hashes del código y runner están en
`freeze_v1.json`. La fuente original y runner están preservados como
`statistics_v1.py` y `runner_v1.py`. Se probaron nulos completos y parciales,
dependencia AR(0.5), solapamiento de cinco pasos, frontera del margen y una
alternativa fuerte, con instrumentos exógenos predecibles.

La v1 **falló**: en el nulo AR(0.5), 16/80 familias rechazaron falsamente,
FWER=0.20, CP95 [0.119,0.304], y la cobertura de la diferencia fue 68/80=0.85,
CP95 [0.753,0.920]. En el nulo parcial hubo 15/80, en el solapamiento 21/80 y
en la frontera práctica 13/80 familias con falso rechazo. Todos los métodos
contribuyeron; el resultado no se atribuye sólo al tercero. El caso fuerte
dio potencia conjunta 80/80, CP95 [0.955,1], sin reparar la calibración nula.
Costó 23.85 s de CPU. La batería no queda admitida en ese régimen.

Después de conservar ese fallo, se fijó `plan_v2.json`: corrección de unidades,
T=1536, HAC48, bloque64, nueva semilla y máximo 240 s de CPU. Es un alcance
de muestra distinto, con bandwidth creciente, no un cambio de tolerancias.
Los criterios **se mantienen**: demostrar FWER sólo si el límite superior
CP95 es ≤0.05, cobertura nominal sólo si el inferior es ≥0.95 y potencia 0.90
sólo si su inferior lo garantiza. Cuando esos límites no resuelven la
afirmación, se informa inconclusa; un fallo no se convierte en aprobación
porque el estimador puntual parezca aceptable. No habrá una tercera búsqueda
de configuraciones sintéticas en esta entrega.

La v2 terminó las 400 réplicas en 144.61 s de CPU (168.46 s sumando ambas
versiones) y **también falló** los criterios fijados:

| Escenario v2 | FWER, CP95 | Cobertura MSE, CP95 | Resultado |
|---|---|---|---|
| Nulo AR(0.5) | 4/80=0.050 [0.014,0.123] | 77/80=0.9625 [0.894,0.992] | Inconcluso para demostrar ambos objetivos |
| Nulo parcial | 3/80=0.0375 [0.008,0.106] | 70/80=0.875 [0.782,0.938] | Fallo de cobertura; FWER inconcluso |
| Nulo con solapamiento 5 | 8/80=0.100 [0.044,0.188] | 72/80=0.900 [0.812,0.956] | FWER/MSE inconclusos; cobertura de ganancia RMSE falla: 71/80, límite superior 0.947 |
| Frontera práctica | 11/80=0.1375 [0.071,0.233] | 74/80=0.925 [0.844,0.972] | Fallo de FWER |
| Alternativa fuerte | No hay nulos verdaderos | 77/80=0.9625 [0.894,0.992] | Potencia conjunta 80/80 [0.955,1], limitada a esta alternativa grande |

El dictamen legible por máquina es `verdict_v2.json`:
`FAILED_FINITE_SAMPLE_VALIDATION`, `admitted=false`. El fallo de FWER es el
evento conservador de cualquier falso rechazo dentro de las 120 pruebas;
no es una estimación de que la regla de intersección "supera a todos" rechace
falsamente con esa misma frecuencia. Esa distinción tampoco autoriza a
afirmar control familiar de los contrastes individuales solicitado por la
guía. Ni el código correcto de Holm ni la potencia favorable sustituyen ese
requisito. No se retocaron tolerancias ni se ejecutó una tercera búsqueda.

Queda pendiente una revisión metodológica y validación **en una nueva
versión**, con una calibración de colas y varianzas temporales que satisfaga
la regla completa; aumentar T o B sin demostrarlo no basta. El cierre y la
apertura de la reserva deben permanecer bloqueados por este dictamen. No se
atribuye este bloqueo a una autorización institucional inexistente: es una
limitación estadística observada en una tarea que autorizó estas pruebas.

Los JSONL conservan cada réplica y sus intervalos; los informes contienen CP
exactos, hashes, cómputo y escenarios incompletos si se agota el presupuesto.
`general_validation_established=False` y `protocol_closure_permitted=False`
impiden convertir evidencia acotada en admisión del test de mercado. Esta
auditoría no estima potencia a la alternativa pequeña del plan de mercado.

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -p test_statistics.py -v
OPENBLAS_NUM_THREADS=1 python qa/statistical_validation/run_validation.py --output-dir /tmp/temporal-v2-reproduction
```

El segundo comando exige un directorio nuevo y verifica los hashes congelados;
no sobrescribe evidencias. Para reconstruir v1 se copian `statistics_v1.py` a
`engine/statistics.py` y `runner_v1.py` a
`qa/statistical_validation/run_validation.py` **en una copia temporal** del
paquete, conservando plan/freeze v1; no se restaura encima del código vigente.

## Referencias primarias

- Diebold y Mariano (1995), *Comparing Predictive Accuracy*, JBES 13(3),
  253–263, doi:10.1080/07350015.1995.10524599.
- Harvey, Leybourne y Newbold (1997), *Testing the equality of prediction mean
  squared errors*, International Journal of Forecasting 13, 281–291,
  doi:10.1016/S0169-2070(96)00719-4.
- Newey y West (1987), *A Simple, Positive Semi-definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica 55, 703–708,
  doi:10.2307/1913610.
- Politis y Romano (1994), *The Stationary Bootstrap*, JASA 89, 1303–1313,
  doi:10.1080/01621459.1994.10476870.
- Giacomini y White (2006), *Tests of Conditional Predictive Ability*,
  Econometrica 74, 1545–1578, doi:10.1111/j.1468-0262.2006.00718.x.
- Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*,
  Scandinavian Journal of Statistics 6, 65–70.

Estas referencias motivan la construcción y sus supuestos; no garantizan
automáticamente el error finito de cualquier T, bandwidth, proceso no
estacionario o cola extrema del bootstrap.
