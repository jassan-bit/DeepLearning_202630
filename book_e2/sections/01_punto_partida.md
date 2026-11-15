# 1. Definición del Problema, Preguntas de Investigación y Dataset

El [benchmark reducido](06_benchmark_final.md) está completado para dieciséis sistemas y 608 fechas, con seis semillas por red. [§48](48_protocolo_reducido.md) documenta las recetas y el protocolo ejecutados; [§49](49_diagnosticos_f2a.md) contiene los diagnósticos posteriores. La arquitectura vigente de TiDE es **TIDE-D7-PUBLISHED-R0-v2**, acreditada antes de sus ajustes; las variantes NAS anteriores mantienen otra identidad.

**Antecedente de planificación del 12/11/2026:** ese corte fue sustituido por la ejecución documentada en [§48](48_protocolo_reducido.md). Su texto exacto se conserva en el archivo enlazado al final; no describe trabajos finales pendientes.

**Catálogo obligatorio vigente: dieciséis sistemas.** Diez modelos de literatura y adaptaciones, las referencias MLP16, LSTM32 y GRU32 de E1, y Persistence, SeasonalNaive y ARIMA. Las referencias E1 se cuentan aparte del Top 10 y forman parte de los dieciséis. El híbrido CNN-1D + LSTM + atención temporal y Ridge quedan como antecedentes opcionales. La [enmienda del 12/11/2026](45_enmienda_protocolo.md) conserva el antecedente de transición; el cuerpo TiDE publicado, el calendario reducido y la ejecución efectiva están documentados en §48.

Esta sección desarrolla las páginas 2–3 de la guía completa de 65 páginas. Se conservan la formulación y la ficha de datos; la ruta es **F2a, series de tiempo**, como explica la {ref}`sección 7 <e2-ruta-temporal>`. El [registro histórico `599a29f`](https://github.com/jassan-bit/DeepLearning_202630/tree/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0) y el [piloto P0 ejecutado](09_protocolo_registrado.md) se conservan para trazabilidad, sin trasladar sus candidatos, presupuesto o resultados al catálogo vigente. Los periodos heredados ya se conocían: su reutilización tiene alcance exploratorio. Los diagnósticos del EDA se presentan en la [sección 2](02_eda.md).

## 1.1 Definición formal del problema

(e2-contrato-heredado)=
### 1.1.1 Contrato vigente y procedencia histórica

**La evaluación vigente conserva las 608 fechas del E1 principal, del 01/01/2025 al 31/08/2026.** Véase [§17](17_contrato_sincronizado.md): representación relativa, escalado de entrenamiento, referencias MLP/LSTM/GRU y límites de evaluación conocida. Las tablas siguientes describen el contrato y el calendario efectivamente ejecutados en E2.

El [Entregable 1, commit `94a4b03`](https://github.com/jassan-bit/DeepLearning_202630/tree/94a4b0324e93361e8928d708fdfeb50110e3686f) y el [contrato descargable histórico](../_static/e1_reference.json) conservan la procedencia D7, incluidas sus fases y la reserva complementaria de 60 días ya evaluada en E1. Esa reserva no se abrió para puntuar el benchmark vigente. Las recetas y resultados históricos conservan su identidad; la selección y los reajustes actuales son los documentados en [§48](48_protocolo_reducido.md).

| Elemento heredado | Definición conservada en E2 |
|---|---|
| Problema | Volatilidad histórica de Bitcoin, BTCUSDT en Binance Spot; formulación heredada inicialmente denominada regresión temporal, con ruta E2 resuelta como F2a, series de tiempo |
| Entrada | D=7 días de cinco canales OHLCV; horizonte h=1 día |
| Objetivo | σ₇ poblacional de siete retornos logarítmicos diarios de Close, sin anualización |
| Referencias de E1 | MLP16, LSTM32 y GRU32, fuera del Top 10 e incluidas en los dieciséis sistemas; selección propia y reajuste común ejecutados en §48 |
| Métrica primaria | Media entre semillas de la RMSE de cada trayectoria sobre las fechas elegibles comunes, en escala original |
| Margen absoluto | δ = **0,00025772842664981106** unidades de volatilidad diaria |
| Unidad de independencia | Una serie BTCUSDT de un proveedor; las fechas y etiquetas solapadas tienen dependencia temporal |
| Partición | Desarrollo en 2023–2024 y evaluación heredada de 608 fechas en 2025–agosto de 2026; sin incorporar etiquetas de evaluación al ajuste |

Las recetas del experimento de origen se consultan en el [diseño versionado de fase II](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/experiments/gru_lstm_tcn_planned_20261105/design.json). Se conserva su trazabilidad sin reproducir aquí una segunda selección de modelos.

El calendario de [§48](48_protocolo_reducido.md) concreta la selección y las subparticiones de desarrollo. La [enmienda registrada de §45](45_enmienda_protocolo.md) y el contrato histórico preservan los diseños anteriores sin presentarlos como el procedimiento ejecutado.

| Función | Periodo UTC y regla ejecutada |
|---|---|
| Desarrollo | 01/01/2023–31/12/2024, con usos restringidos por cada etapa |
| Optimización | Tres folds internos con orígenes 01/01, 01/07 y 01/10/2024; 83 fechas de validación por fold, 249 concatenadas; seis configuraciones y una semilla HPO por red; embargo de siete días |
| Ajuste final | Pesos y transformaciones ajustados con etiquetas admisibles hasta el **31/10/2024**, para los dieciséis sistemas |
| Puente de embargo | 01–07/11/2024 |
| Calibración y piloto | **08/11–30/12/2024, 53 fechas**; recetas congeladas y sin nuevo HPO |
| Puente causal | 31/12/2024, sólo entradas históricas y estado admisible de comparadores; sin reajuste ni recalibración |
| Evaluación heredada | **01/01/2025–31/08/2026, 608 fechas comunes**; pesos, transformaciones y calibradores congelados antes del scoring de esta campaña |

El embargo excluye etiquetas de entrenamiento demasiado próximas al origen según la regla D=7. Los metadatos conservan fechas, máscaras y exclusiones; no se asigna una semilla a la partición cronológica determinista. Las fechas de evaluación ya eran conocidas desde E1: la congelación de esta campaña acredita trazabilidad y separación de etapas, sin convertir el periodo heredado en una confirmación prospectiva nueva.

### 1.1.2 Entradas, salida y disponibilidad causal

Sea $P_t>0$ el cierre del día UTC $t$. Se define

$$
r_t=\log(P_t/P_{t-1}),\qquad
\bar r_t=\frac{1}{7}\sum_{j=0}^{6}r_{t-j},\qquad
Y_t=\sigma_7(t)=\sqrt{\frac{1}{7}\sum_{j=0}^{6}(r_{t-j}-\bar r_t)^2}.
$$

Se necesitan **ocho cierres consecutivos para siete retornos**. El denominador es 7 (`ddof=0`). No se anualiza ni se sustituye esta variable por volatilidad intradía o por el precio.

Sea $Z_t=(O_t,H_t,L_t,C_t,V_t)$ el OHLCV agregado del día. El pronóstico se emite al comienzo del día $t$, con

$$
X_t=(Z_{t-7},\ldots,Z_{t-1})\in\mathcal X\subseteq\mathbb R^{7\times5},
\qquad Y_t\in\mathcal Y=\mathbb R_{\ge0}.
$$

Así, $r_t$ y $Y_t$ solo se conocen al terminar el día pronosticado. Esta convención equivale a escribir «entrada hasta $t$, objetivo $\sigma_7(t+1)$» desplazando el índice; no añade información del futuro. La volatilidad pasada puede usarse para diagnósticos por régimen, pero no se añade aquí como sexto canal del modelo.

El preprocesamiento heredado usa logaritmos de OHLC relativos al último cierre conocido de la ventana y `log1p` del volumen; los escaladores se ajustan exclusivamente en entrenamiento permitido. El objetivo se divide por su media de entrenamiento y las métricas se calculan después de invertir esa transformación. Se conserva el calendario y se excluyen ventanas que dependan de días inválidos, sin unir artificialmente fechas separadas por huecos.

### 1.1.3 Espacio de hipótesis, pérdida y riesgo

El Top 10 vigente contiene **FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer**. Son modelos de la literatura o adaptaciones de familias; no se presentan como los nombres literales del curso. **CNN-1D + LSTM + atención temporal** conserva una propuesta histórica opcional, sin superioridad demostrada y sin obligación de completar nuevas corridas para el catálogo requerido.

Para esos modelos, el espacio de hipótesis es $\mathcal H=\bigcup_{m\in\mathcal M}\mathcal H_m$. Las diez redes y las tres referencias E1 parten del contrato $\mathcal X\subseteq\mathbb R^{7\times5}$. No incorporan historia mensual ni etiquetas rezagadas como entradas adicionales. La variable objetivo es no negativa; el enlace de salida y cualquier posprocesamiento se fijan por receta antes de puntuar. MLP16, LSTM32 y GRU32 mantienen sus cuerpos E1, se reinicializan y conservan identidad experimental propia dentro del catálogo obligatorio.

Se conserva la pérdida cuadrática:

$$
\mathcal L(f(X),Y)=(Y-f(X))^2,\qquad
R_P(f)=\mathbb E_{(X,Y)\sim P}[\mathcal L(f(X),Y)].
$$

Sobre una cohorte elegible común $\mathcal T$, su versión empírica es

$$
\widehat R_{\mathcal T}(f)=\frac{1}{|\mathcal T|}\sum_{t\in\mathcal T}(Y_t-f(X_t))^2,
\qquad
\widehat\mu_m=\frac{1}{R}\sum_{s=1}^{R}\sqrt{\widehat R_{\mathcal T}(f_{m,s})}.
$$

$\widehat\mu_m$ es la métrica primaria. No se reemplaza por la RMSE de un ensamble ni por la raíz de la media de todos los errores mezclando semillas. La dependencia y el cambio de distribución impiden tratar el riesgo empírico histórico como una estimación iid de cualquier mercado futuro. La guía exige heredar la partición de E1: se conservan esas fechas y su exposición previa, junto con la [congelación anterior al scoring actual](../_static/e2_benchmark_final_20261113/final_freeze.json) y la [máscara común](../_static/e2_benchmark_final_20261113/data_admission.json). El alcance de la comparación es retrospectivo.

### 1.1.4 Tipo de tarea y relevancia

La clasificación metodológica de E2 es **F2a, series de tiempo**, resuelta tras recibir la guía completa. El {ref}`diagnóstico residual de §2.2.3 <e2-diagnostico-forward>` aporta autocorrelación fuerte de la referencia histórica Ridge; RT1 (PDF p32) exige reclasificar, además del diagnóstico del EDA C (p6). La {ref}`sección 7 <e2-ruta-temporal>` conserva esa resolución y los controles causales. Se conserva la tarea predictiva, D7×5, σ₇, h=1 y δ; la reclasificación no añade Y rezagada como canal de las redes. El calendario y el soporte de los comparadores obligatorios se fijan en §45; retirar Ridge del alcance obligatorio no elimina su diagnóstico histórico.

Tres trabajos respaldan la relevancia científica del pronóstico de volatilidad en criptomonedas: Miura, Pichl y Kaizoji estudian redes neuronales para volatilidad realizada {cite:p}`miura2019`; Guo, Bifet y Antulov-Fantulin estudian la información de órdenes de compra y venta {cite:p}`guo2018`; Akbiyik y colaboradores incorporan información de Twitter al pronóstico de volatilidad de Bitcoin {cite:p}`akbiyik2023`.

Estos antecedentes motivan investigar memoria temporal, covariables y generalización. **No se afirma que utilicen nuestro snapshot, idéntico objetivo σ₇ ni exactamente las entradas aquí disponibles.** Tampoco constituyen por sí solos la revisión sistemática de E2 ni prueban uso del dataset exacto en una revista Q1. El interés aplicado es medir capacidad predictiva y coste bajo un benchmark académico; no se concluye rentabilidad, reducción de riesgo financiero o utilidad de trading.

### 1.1.5 Métricas, margen y unidad de independencia

La primaria es RMSE; las secundarias heredadas son MAE, MSE, R², MAPE y MASE. MAPE se define solo donde el objetivo sea positivo y se reportará su cobertura; no se añadirá un épsilon arbitrario. MASE usa el denominador histórico ingenuo de lag 1 sobre pares de días consecutivos elegibles de entrenamiento; no se calcula comprimiendo huecos ni usando test para escalar.

La [A2 ejecutada](../_static/e2_benchmark_final_20261113/benchmark_A2.csv) publica MAE, **sMAPE** y R² junto con RMSE. sMAPE es una métrica simétrica distinta de MAPE: esa columna no acredita el cálculo de MAPE, MSE o MASE en A2 ni cambia la métrica primaria heredada.

El margen absoluto heredado procede de $0,05\times0,005154568532996221$, RMSE CV de la GRU/D7 de fase I. Es una tolerancia académica operacional equivalente al 5 % de ese error de referencia, sin utilidad financiera demostrada. **No se recalcula como el 5 % de una nueva RMSE.** Se conserva aparte la regla relativa académica de E1 frente a GRU y persistencia; no se confunde una reducción porcentual con δ absoluto. Su origen y fecha están documentados: heredar el margen no lo vuelve una decisión previa a los resultados históricos iniciales.

Hay **una única serie de mercado** ($N_{\text{series}}=1$). Las semillas describen variabilidad del entrenamiento condicionada a los datos; no son nuevas series independientes. Las etiquetas σ₇ comparten retornos y las ventanas comparten observaciones. La comparación debe emparejar fechas y distinguir variación entre semillas de incertidumbre temporal. El remuestreo por bloques evita suponer independencia diaria, pero su diseño y sensibilidad requieren justificación en desarrollo. Ningún número de semillas de E1 o del P0 histórico acredita potencia para las nuevas comparaciones. El procedimiento vigente de [§48](48_protocolo_reducido.md) completó el piloto y fijó R=6 por recursos; su potencia condicional estimada fue 0,123, inferior al objetivo de 0,80.

## 1.2 Preguntas de investigación e hipótesis

Las tres preguntas conservan la formulación científica original. La [enmienda de §45](45_enmienda_protocolo.md) concreta RQ1 en **veinte contrastes primarios del Top 10 frente a GRU32 y Persistence**, con ajuste Holm; RQ2 y RQ3 se mantienen como análisis secundarios exploratorios y no reciben automáticamente la potencia ni el control familiar de RQ1. El [registro anterior](https://github.com/jassan-bit/DeepLearning_202630/blob/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0/experiments/e2_remediation_20261106/preregistration.json) conserva su alcance histórico. Las hipótesis del híbrido opcional quedan fuera de la familia primaria vigente.

Para cada candidato $m$, sea $g_m=\mu_{\mathrm{ref}}-\mu_m$: valores positivos indican menor RMSE que la referencia predefinida. Las diferencias deben usar las mismas fechas elegibles y un tratamiento explícito de las semillas. MLP16, LSTM32 y GRU32 aportan referencias separadas; su receta, cohorte y papel inferencial no se escogen después de mirar los resultados. Ningún candidato recibe la etiqueta de ganador por construcción.

### 1.2.1 RQ1 · Ganancia predictiva relevante

**Pregunta:** ¿algún candidato reduce la RMSE respecto a la referencia predefinida en más que el margen heredado δ?

$$
H_{0,m}:g_m\le\delta,\qquad H_{1,m}:g_m>\delta.
$$

El efecto mínimo es δ en unidades originales de volatilidad. §45 registra un contraste unilateral pareado respecto a δ mediante bootstrap temporal por bloques, con las mismas fechas remuestreadas para ambos modelos y la variabilidad entre trayectorias tratada separadamente. La distribución nula se centra en la frontera $g_m=\delta$; no basta probar una diferencia contra cero. La alternativa registrada para planificar potencia es $2\delta$; no se promete potencia de detección en la frontera de la hipótesis nula ni potencia ya demostrada.

La decisión registrada usa el contraste unilateral respecto a δ y Holm sobre veinte comparaciones, con nivel familiar 0,05; efecto e incertidumbre se interpretan respecto al mismo margen. [§56](56_inferencia_estabilidad_final.md) conserva el cálculo original y su refinamiento posterior a 20.000 remuestreos, sin modificar la familia. Una ganancia puntual superior a δ, por sí sola, no prueba mejora práctica con incertidumbre controlada.

### 1.2.2 RQ2 · Deterioro de la ventaja en alta volatilidad

**Pregunta:** ¿la ventaja de un candidato frente a la referencia disminuye en más de δ durante días de alta volatilidad conocida al emitir el pronóstico?

El antecedente metodológico define el estrato causal $A_t=\mathbf1\{\sigma_7(t-1)>q_{0,75}^{2023}\}$, con $q_{0,75}^{2023}=0,024247163097617024$, obtenido únicamente de las 350 etiquetas válidas σ₇ de 2023. Su [perfil de entrenamiento está versionado en E1](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/book/_static/eda_general_quality/target_profiles.csv). El estrato usa información pasada; no es un nuevo canal de entrada ni un umbral optimizado en test.

Con RMSE calculada por trayectoria dentro de cada estrato, sean $g_{m,N}$ y $g_{m,A}$ las ganancias en régimen normal y alto, y $d_m=g_{m,N}-g_{m,A}$:

$$
H_{0,m}:d_m\le\delta,\qquad H_{1,m}:d_m>\delta.
$$

El efecto mínimo es una erosión absoluta δ de la ventaja. El contraste es un bootstrap temporal pareado de $d_m$, conservando las etiquetas causales de régimen y recalculando la RMSE de cada trayectoria dentro de cada estrato antes de promediar entre semillas. La referencia secundaria se fija en GRU32. La distribución nula se centra en $d_m=\delta$; no se prueba sólo una diferencia contra cero. Se reportan fechas y bloques útiles por estrato; si el soporte no permite estimar la incertidumbre, se declara no estimable o inconcluso. Rechazar indica deterioro relevante; **no rechazar no demuestra robustez ni equivalencia**.

**Aclaración documental posterior:** el nivel nominal es 0,05 para un contraste individual identificado de antemano. Si se comunican los diez candidatos, se aplica Holm a sus diez p-valores de RQ2 y se decide con p ajustado ≤0,05. Esta precisión y la especificación descargable de §1.2.4 no se atribuyen a la ejecución pasada. La [ampliación del 15/11/2026](64_correcciones_metodologicas.md) sí ejecuta esa especificación sobre pronósticos guardados: siete contrastes rechazan tras Holm10. Es evidencia exploratoria posterior, sin potencia específica demostrada; el no rechazo de SAE, Inception y TimeMixer no demuestra robustez.

### 1.2.3 RQ3 · Mejora conjunta de error y coste

**Pregunta:** ¿algún candidato mejora la RMSE en más de δ y reduce el coste de inferencia frente a la referencia predefinida?

Sea $c_m$ la diferencia de latencia media por ventana, candidato menos referencia, bajo el mismo hardware, precisión y batch size:

$$
H_{0,m}:(g_m\le\delta)\ \text{o}\ (c_m\ge0),\qquad
H_{1,m}:(g_m>\delta)\ \text{y}\ (c_m<0).
$$

La afirmación conjunta requiere una prueba de intersección-unión frente a GRU32: $p_{\mathrm{IUT}}=\max(p_{\mathrm{gain}},p_{\mathrm{cost}})$. El primer componente contrasta $g_m>\delta$ mediante bootstrap temporal pareado; el segundo contrasta $c_m<0$ mediante bootstrap de diferencias de medias de latencia por bloques pareados, centrado bajo la frontera $c_m=0$. El margen de error sigue siendo δ; cero es el umbral direccional de coste, sin tolerancia económica demostrada. Se reportan magnitud e intervalo del ahorro, además del p-valor conjunto.

La latencia requiere observaciones crudas comparables con entradas de desarrollo o sintéticas, mismo hardware, precisión y batch, calentamiento, sincronización, bloques de sesión pareados y orden intercalado documentados. Se conserva el emparejamiento entre candidato y GRU32 y la agregación de sus seis predictores; las repeticiones del cronómetro no crean mercados independientes. Los perfiles agregados de coste disponibles no acreditan ese contraste ni la IUT.

**Aclaración documental posterior:** el nivel nominal de la IUT individual es 0,05 y exige que ambos componentes sean válidos y ≤0,05. Si se comunican los diez candidatos, se aplica Holm a sus diez p-valores IUT y se decide con p ajustado ≤0,05. Esta regla no se atribuye al pasado. Falta la ejecución inferencial conjunta con costes crudos comparables; no se declara resuelta RQ3 ni se le transfiere la potencia de RQ1.

### 1.2.4 Multiplicidad y criterio de decisión común

La familia primaria vigente contiene veinte comparaciones del Top 10 frente a GRU32 y Persistence y utiliza Holm con nivel familiar 0,05, según §45. **No se heredan los recuentos ni las asignaciones estadísticas del catálogo retirado.** Las referencias E1 y los comparadores sí forman parte de los dieciséis sistemas obligatorios; el híbrido y Ridge no forman parte de esa familia primaria.

RQ2 y RQ3 conservan alcance exploratorio. Si se comunica el contraste conjunto de RQ3, se obtiene primero el p-valor de intersección-unión y se declara su multiplicidad y sus límites; no se incorpora silenciosamente al ajuste primario. Los IC marginales del 95 % se distinguen de los límites simultáneos. Las pruebas de equivalencia, calibración, robustez y diagnóstico necesitan su propio alcance; no quedan cubiertas automáticamente por el ajuste de las preguntas principales.

| Pregunta y alcance | Prueba, efecto y criterio de decisión |
|---|---|
| RQ1 · análisis primario registrado; evaluación retrospectiva | Bootstrap temporal unilateral de $g_m>\delta$, veinte contrastes frente a GRU32 y Persistence; Holm, α familiar 0,05. Reglas y resultados en §48/§56, sin cambios por esta aclaración. |
| RQ2 · exploratoria; aclaración documental posterior | Bootstrap temporal unilateral de $d_m>\delta$ frente a GRU32. α nominal 0,05 por contraste; para los diez candidatos, Holm dentro de RQ2 y p ajustado ≤0,05. La falta de estimación o de rechazo no prueba robustez. |
| RQ3 · exploratoria; aclaración documental posterior | IUT de $g_m>\delta$ y $c_m<0$ frente a GRU32; máximo de los dos p-valores válidos. α nominal 0,05 por IUT; para los diez candidatos, Holm dentro de RQ3 y p ajustado ≤0,05. Sin costes crudos comparables no se emite decisión conjunta. |

La [especificación focal en JSON](../_static/e2_point1_review_v1/rq_specification.json) fija los detalles de remuestreo y requisitos de datos de estas aclaraciones y conserva su fecha documental original. La [ejecución posterior de RQ2](64_correcciones_metodologicas.md) tiene plan, recibo y resultados propios; no se reescribe ese JSON histórico para atribuirle cálculos anteriores. La potencia específica de ambas preguntas no está demostrada.

El margen absoluto δ conserva su valor y finalidad. **δg sí existía en el registro del 6 de noviembre**, commit [`599a29f`](https://github.com/jassan-bit/DeepLearning_202630/blob/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0/experiments/e2_remediation_20261106/preregistration.json): el campo `generalization_gap` fijó δg=δ para la brecha RMSE de validación menos entrenamiento en modo inferencia, con unidades y justificación. Reconocer ese linaje no demuestra ausencia de sobreajuste de los diez candidatos ni convierte las sensibilidades posteriores de [§56](56_inferencia_estabilidad_final.md) en aquella prueba ejecutada. El [P0 histórico de §9](09_protocolo_registrado.md) completó 220 trayectorias y 440 ajustes de desarrollo, pero no selecciona ni dimensiona el catálogo vigente. El registro y el control de multiplicidad no recuperan el cegamiento histórico.

## 1.3 Justificación del dataset

### 1.3.1 Selección, tamaño y calidad

Se conserva **Binance Spot BTCUSDT**, obtenido de [Binance Public Data](https://github.com/binance/binance-public-data) y [Binance Vision](https://data.binance.vision/). Es pertinente por ofrecer una serie continua de un mercado de criptomonedas con OHLCV y trazabilidad de archivos; permite reproducir el benchmark de E1. El tamaño bruto no se interpreta como millones de muestras supervisadas independientes.

El snapshot inventariado contiene **75 ZIP, 97.442.605 bytes comprimidos y 2.014.480 filas de doce campos**, entre el 01/01/2023 y el 30/10/2026. Los 1.399 días del calendario corresponden a 2.014.560 minutos esperados. Se detectaron **80 minutos ausentes** y una anomalía adicional de `close_time` el 24/03/2023. Por ello, ese día no es elegible para OHLCV diario completo; no se imputa ni se elimina el hueco del calendario.

El desarrollo 2023–2024 tiene 731 días de calendario y 730 días OHLCV completos. En 2023 hay 350 etiquetas σ₇ válidas; en 2024, 366. Las pérdidas adicionales de ventanas dependen de D, el objetivo y el embargo: los conteos de filas brutas, días y ejemplos supervisados son distintos. El ajuste final D7 de E1 utilizó 709 ejemplos elegibles; ese recuento no se reasigna al ajuste E2 hasta octubre de 2024. La evaluación actual admite veinte ZIP, 875.520 minutos y **608 fechas supervisadas comunes**, sin días excluidos en esa cohorte.

Como respaldo de la fuente, Goutte y colaboradores utilizaron datos horarios de Bitcoin de Binance y describieron el acceso mediante API {cite:p}`goutte2023_binance_context`. *Finance Research Letters* fue Q1 en Finance según SCImago/SJR de 2023, con SJR 1,903. Este antecedente acredita uso académico de Binance; su tarea de negociación y periodo 2017–2022 no equivalen al objetivo σ₇ ni al snapshot de E2.

La [evidencia focal de fuente y cuartil](../_static/e2_point1_review_v1/q1_context_evidence.json) conserva DOI, localizadores editoriales y fila histórica del índice; no afirma cuartil JCR ni lectura íntegra del PDF. La comprobación por DOI y título normalizado no encontró coincidencias en los 9.559 grupos recibidos. Se incorpora como **contexto externo de la fuente**, sin añadirlo a los 42 estudios aplicados ni modificar PRISMA. El requisito de uso previo de la fuente en Q1 no exige identidad exacta de fechas, etiquetas o hashes.

### 1.3.2 Procedencia, licencia, ética y privacidad

Los archivos originales y sus checksums se descargan del proveedor; el [manifiesto descargable](../_static/dataset_manifest_e1.json) identifica cada objeto y su integridad. Las [instrucciones y scripts de adquisición de E1](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/data/README.md) permiten reconstruir la fuente sin incorporar precios individuales a este capítulo.

La copia de términos revisada en E1 el 03/11/2026 corresponde a **Vision Dataset Terms v1.0, actualizados el 26/08/2026**: establece CC BY-NC-SA 4.0 salvo designación específica y condiciones adicionales. Véanse el [texto conservado](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/book/_static/data_source/TERMS_AND_CONDITIONS.md) y su [registro de procedencia](https://github.com/jassan-bit/DeepLearning_202630/blob/94a4b0324e93361e8928d708fdfeb50110e3686f/book/_static/data_source/terms_snapshot.json). La ficha vigente registra una nueva consulta del texto oficial con SHA-256 idéntico al conservado. No se presenta como una licencia MIT de los datos ni se retrofecha la revisión.

El uso es académico y no comercial. Las velas agregadas no incluyen identificadores personales; no se intentará identificar operadores. La ausencia de datos personales no elimina sesgos del mercado ni convierte los resultados en asesoría financiera. Se mantienen la atribución y la separación entre licencia de datos y código.

### 1.3.3 Representatividad, sesgos y partición pública

La población observada es un único par, exchange y periodo. BTCUSDT no representa otras criptomonedas, otros proveedores ni cotizaciones en moneda fiduciaria; USDT añade exposición a una stablecoin. La selección de un activo superviviente, los cambios de régimen, las interrupciones del proveedor y la cobertura histórica limitan la generalización. El volumen refleja negociación de ese mercado, sin medir todo el mercado mundial.

No se ha identificado una **partición oficial de machine learning para este snapshot** en la fuente utilizada. Los archivos mensuales y diarios del proveedor son una organización de descarga, no folds de evaluación. Se conserva la evaluación temporal de E1 y la selección E2 usa tres folds internos anteriores a ella para respetar disponibilidad causal y comparabilidad. Los periodos conocidos se documentan como tales; no se barajan fechas ni se crean nuevos grupos independientes artificiales.

## 1.4 Descripción técnica · Data Card

Descargar la [Data Card vigente en JSON](../_static/e2_point1_review_v1/dataset_card_current.json), el [contrato heredado](../_static/e1_reference.json) y el [manifiesto de archivos](../_static/dataset_manifest_e1.json). La [Data Card histórica](../_static/dataset_card.json) conserva sus bytes y estados anteriores; no describe el cierre de ejecución actual.

| Campo | Descripción verificable |
|---|---|
| Identificador | Snapshot E1 de Binance Spot BTCUSDT, velas de un minuto, 01/01/2023–30/10/2026 |
| Fuente y recolección | Objetos históricos mensuales/diarios de Binance Vision; adquisición E1 entre 02/11/2026 14:47:23 y 14:47:45 UTC; veinte ZIP del periodo heredado admitidos para el scoring actual, sin añadir fechas nuevas |
| Versión | Ficha `e2-btcusdt-d7-current-v1`; fuente E1 `94a4b0324e93361e8928d708fdfeb50110e3686f` y manifiesto SHA-256 `af723bcca2caffcbe44ae5243e2863bbcdcd39ae8cd42df491af0a2857926bb2` |
| Estructura bruta | 75 ZIP; 2.014.480 filas; doce campos de kline |
| Esquema | Timestamps de apertura/cierre; OHLC, volumen, volumen cotizado, número de operaciones, volúmenes taker base/cotizado y campo ignore |
| Tipos | Timestamps normalizados a UTC, conteo de operaciones entero y magnitudes numéricas; la unidad temporal original se verifica por archivo antes de convertir |
| Entrada del modelo | Siete días consecutivos × cinco variables OHLCV; 35 valores antes de transformaciones |
| Objetivo y horizonte | σ₇ poblacional de retornos diarios; día siguiente al último día de entrada; ocho cierres necesarios |
| Etiquetado | Cálculo determinista desde cierres diarios completos, sin anotadores humanos; acuerdo entre anotadores no aplica |
| Ruido del objetivo | σ₇ es un estimador histórico dependiente del cierre y ventana elegidos; no es observación directa de volatilidad latente ni etiqueta sin error |
| Cobertura | Mercado BTC/USDT en Binance Spot, operación 24/7, calendario diario UTC; no hay unidades geográficas ni población de personas |
| Filtros | Validación de timestamps y velas, agregación de días completos, retornos sobre días consecutivos, exclusión de ventanas afectadas por huecos y embargo heredado; sin imputación |
| Calidad conocida | 80 minutos ausentes y una anomalía adicional de cierre el 24/03/2023; un día diario incompleto |
| Desarrollo | 2023–2024: 731 días de calendario, 730 OHLCV completos; etiquetas σ₇ válidas 350 en 2023 y 366 en 2024 |
| Selección, ajuste y calibración | Tres folds HPO, 249 fechas de validación concatenadas; ajuste final con etiquetas hasta 31/10/2024 y calibración 08/11–30/12/2024, 53 fechas |
| Evaluación heredada | **01/01/2025–31/08/2026, 608 fechas comunes** y 81 predictores congelados; periodo conocido desde E1. La reserva complementaria de 60 fechas es un antecedente fuera de este scoring |
| Licencia | Términos Vision conservados en E1, CC BY-NC-SA 4.0 con condiciones y excepciones indicadas por la fuente |
| Límites de uso | Benchmark académico de una sola serie; sin afirmación de utilidad financiera ni generalización a otras criptomonedas |

El hash anterior identifica **el archivo manifiesto**; no es un único checksum de todos los precios concatenados. Cada ZIP tiene su comprobación individual en ese archivo. La Data Card y el contrato documentan la transferencia; el [benchmark final Bitcoin D7](06_benchmark_final.md) ya compara dieciséis sistemas sobre 608 fechas; se distingue del P0 histórico de desarrollo archivado en §9. El [recibo de las veinte corridas externas de ETTh1](../_static/paper_guide_evidence/execution_receipt.json) tiene otro alcance.

### 1.4.1 Estado de cierre de esta sección

| Requisito visible | Estado |
|---|---|
| Formulación matemática, tarea, métricas, δ y unidad | Documentados con la herencia de E1 |
| Relevancia con al menos tres referencias | Tres referencias de contexto citadas; no sustituyen la revisión sistemática |
| Tres RQ, hipótesis, efecto y reglas | RQ1 ejecutada con veinte contrastes; refinamiento B=20.000 de §56. RQ2 ejecutada posteriormente en §64 con siete rechazos Holm10; RQ3 conserva sus requisitos específicos de latencia e IUT. Ambas son exploratorias, con potencia específica no demostrada; el piloto de RQ1 sigue insuficiente y declarado. |
| Justificación y Data Card | Ficha vigente con procedencia, versión, hashes, calendario ejecutado, calidad y límites; ficha histórica conservada |
| Uso previo de la fuente/dataset en Q1 | Goutte et al. (2023), Binance identificado y SCImago/SJR Finance 2023 Q1; contexto externo, sin cambiar PRISMA 9.559/42 |
| Catálogo y protocolo | Dieciséis sistemas evaluados; cuerpos actuales adjudicados, incluida TiDE publicada r=0; protocolo ejecutado en §48 |
| Partición y alcance de la prueba | Se conservan las 608 fechas exigidas por la herencia E1, su exposición previa y la congelación de esta campaña; alcance retrospectivo |

La [lista literal de requisitos del punto 1](../_static/e2_point1_review_v1/point1_checklist.json) enlaza cada campo con su evidencia y distingue definición documental de ejecución inferencial. La [revisión bibliográfica de §3](03_literatura.md) conserva el corpus aplicado y sus denominadores. El respaldo contextual de Binance no añade una inclusión a ese corpus ni acredita la eficacia de una arquitectura en este dataset.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/01_punto_partida.md>` se conserva con su SHA256 en el [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json). Sus estados pendientes describen el corte original.
