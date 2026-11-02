# 1. Definición del Problema y Dataset

**Correspondencia con los requisitos de las páginas 2–3**

La siguiente tabla permite localizar las respuestas solicitadas en la guía. Las fórmulas y elecciones metodológicas son propuestas de trabajo; los campos empíricos pendientes no se presentan como hechos comprobados.

| Requisito del entregable | Respuesta y ubicación |
|---|---|
| Espacios de entrada y salida, función y pérdida | §1.3: ventana OHLCV diaria, salida no negativa, predictor supervisado y MSE |
| Tipo de problema admitido | §1.2: series de tiempo; se usa el pasado de la propia serie |
| Pregunta de investigación e hipótesis operativa | §1.4: comparar el error de pronóstico con persistencia |
| Métricas primaria y secundarias | §1.4: RMSE, MAE, MAPE y MSE; MASE adicional con condiciones explícitas |
| Margen práctico e independencia | §1.4: margen pendiente de fijar y bloques temporales pendientes de dimensionar; las ventanas no son independientes |
| Relevancia científica o aplicada | §1.1: pronóstico de incertidumbre de Bitcoin y antecedente científico |
| Disponibilidad, licencia y procedencia | §1.5: Binance Spot BTCUSDT confirmado, términos documentados y adquisición directa verificada; falta completar la descripción de generación de velas |
| Uso previo del dataset en literatura | §1.5: no verificado; el artículo citado es antecedente del problema |
| Tamaño, dimensionalidad y almacenamiento | §1.5: 75 ZIP, 2.014.480 filas CSV de 12 columnas y esquema de entrada propuesto; ventanas supervisadas pendientes |
| Tipo de datos y unidad de observación | §1.2 y §1.5: precios temporales y origen de pronóstico |
| Etiquetado y ruido esperado | §1.3 y §1.5: cálculo automático de dispersión futura y fuentes de ruido |
| Dependencia entre observaciones | §1.4–§1.6: solapamiento temporal, partición cronológica y prevención de fuga |
| Versión y hash del dataset | §1.5: instantánea del 02/11/2026, manifiesto y SHA-256 de los 75 ZIP verificados; etiquetas y derivados supervisados pendientes |

**Límite de este avance:** la fuente Binance Spot BTCUSDT está seleccionada y su adquisición, términos, tamaño y hashes están documentados. El EDA de desarrollo está ejecutado y documentado en la sección 2 y aún deben cerrarse la definición del objetivo, las muestras supervisadas y el protocolo experimental. No se afirma que el Entregable 1 completo esté terminado.

## 1.1. Motivación y alcance

Se propone evaluar si el historial de rendimientos logarítmicos derivados del cierre de Bitcoin contiene información útil para pronosticar su volatilidad futura. El interés aplicado es estimar incertidumbre del activo, no predecir directamente su precio ni recomendar operaciones financieras.

El estudio de Huang, Sangiorgi y Urquhart (2024) examina pronósticos de volatilidad de Bitcoin con datos de alta frecuencia y redes neuronales; respalda la pertinencia de investigar este problema. Su definición de volatilidad, datos y horizontes no son automáticamente equivalentes a los de este proyecto: sus resultados no se trasladan como resultados propios. Véase la [referencia primaria](referencias.md).

**Estado:** BTCUSDT, modelado diario a partir de velas de un minuto, entradas de 7/14/21/28 días y horizonte principal de un día están confirmados. Se exploran ventanas de volatilidad de 7/14/21/28 días; falta cerrar las configuraciones de evaluación y las demás decisiones experimentales.

## 1.2. Tipo de problema y reloj de información

**Tipo: series de tiempo**, con salida continua y aprendizaje supervisado. Se pronostica volatilidad futura de Bitcoin usando exclusivamente información histórica disponible al emitir el pronóstico; no se barajan observaciones entre periodos.

Se distingue **adquisición a un minuto** de **modelado diario**. Los archivos oficiales BTCUSDT se agregan en días UTC completos. Cada observación diaria contiene apertura, máximo, mínimo, cierre y volumen (OHLCV); un día incompleto se marca no elegible en los derivados, sin modificar los ZIP originales.

Sea $P_d>0$ el cierre del día UTC $d$. El retorno de modelado es **diario**:

$$
r_d=\log\left(\frac{P_d}{P_{d-1}}\right).
$$

El pronóstico del día $t$ se emite **después del cierre del día $t-1$ y antes de observar el cierre del día $t$**. En ese momento solo están disponibles OHLCV y retornos de días con índice menor o igual a $t-1$. Un OHLCV diario no está disponible al comenzar ese mismo día.

La configuración aportada establece modelado diario, cierre UTC y horizonte principal de un día. Las noticias solo se incorporarían con una fuente verificable y fechas de publicación anteriores al pronóstico; no se han adquirido ni forman parte de la entrada implementada de este avance.

## 1.3. Formulación matemática y configuración de la guía

### 1.3.1. Entradas: 7, 14, 21 y 28 días de historia

El usuario confirmó que los tamaños representan **días**, no siete o veintiocho minutos. Se consideran $D\in\{7,14,21,28\}$ observaciones diarias consecutivas:

| Historia $D$ | Velas de un minuto que originan esos días | Observaciones de entrada al modelo diario |
|---|---:|---:|
| 7 días | 10.080 | 7 |
| 14 días | 20.160 | 14 |
| 21 días | 30.240 | 21 |
| 28 días | 40.320 | 28 |

No se confunde la cantidad de velas brutas con el número de pasos diarios que recibe la red. El **Close** es la variable central para los retornos y las etiquetas. Para la configuración OHLCV diaria aportada, sea $z_d=(O_d,H_d,L_d,P_d,V_d)$:

$$
\mathcal X_D\subset\mathbb R^{D\times5},\qquad
X_t=(z_{t-D},\ldots,z_{t-1})^\top.
$$

Cualquier transformación o variable derivada debe calcularse causalmente. El EDA no ha seleccionado todavía un subconjunto final de covariables ni su escalado; las comparaciones deberán mantener el mismo pipeline entre modelos.

### 1.3.2. Volatilidad histórica y objetivo de un día adelante

Se respeta la fórmula de la guía, con divisor poblacional $n$:

$$
\bar r_t=\frac1n\sum_{i=1}^{n}r_{t-i},\qquad
\sigma_t=\sqrt{\frac1n\sum_{i=1}^{n}(r_{t-i}-\bar r_t)^2}.
$$

Aquí $n\ge2$ es la cantidad de **retornos diarios** usada para calcular la volatilidad histórica. Es distinta de los $D$ días de entrada: confirmar $D=7,14,21,28$ no fija automáticamente $n$. **El EDA estudia como candidatas $n=7,14,21,28$ días; no se seleccionó una definición ganadora ni se entrenaron modelos.**

Con la convención de emisión al inicio del día $t$, $\sigma_t$ ya es conocida. El horizonte principal de un día consiste en pronosticar:

$$
Y_t=\sigma_{t+1},\qquad
\mathcal Y=\mathbb R_{\ge0},\qquad
f_\theta:\mathcal X_D\longrightarrow\mathcal Y,\qquad
\widehat Y_t=f_\theta(X_t).
$$

$\sigma_{t+1}$ incorpora $r_t$, que todavía no se conoce al emitir el pronóstico y se observa al cerrar ese día. Esta indexación evita presentar como pronóstico una cantidad ya calculable con los datos disponibles. Es una convención operacional explícita de la fórmula del curso; no se anualiza ni multiplica por 100 en el cálculo. Las gráficas pueden mostrar porcentaje con la conversión indicada.

La volatilidad es un proxy de dispersión de retornos observados, con incertidumbre de muestreo y posible ruido del proveedor. No se confunde con volatilidad latente observada sin error. Las etiquetas necesitan $n$ retornos válidos consecutivos y, por tanto, $n+1$ cierres diarios; no se forman atravesando días incompletos.

### 1.3.3. Pérdida y referencia contextual

Se propone minimizar MSE sobre los orígenes elegibles de entrenamiento:

$$
\mathcal L(\theta)=\frac1{|\mathcal T_{\mathrm{train}}|}
\sum_{t\in\mathcal T_{\mathrm{train}}}(Y_t-f_\theta(X_t))^2.
$$

La referencia de persistencia para el mismo horizonte será $\widehat Y_t^{\mathrm{pers}}=\sigma_t$, calculada únicamente con el pasado. No ocupa uno de los tres puestos de modelos de Deep Learning. Se comparará en los mismos orígenes elegibles para cada tamaño de entrada.

Escalado, imputación permitida y cualquier transformación se ajustarán solo con el entrenamiento de cada fold. Las métricas finales se expresarán en la escala original del objetivo y la salida no negativa se garantizará al implementar las redes.

## 1.4. Pregunta, hipótesis y evaluación a priori

**Pregunta propuesta:** ¿pueden tres modelos simples de Deep Learning, usando 7, 14, 21 o 28 días de información OHLCV de Bitcoin, pronosticar la volatilidad histórica del siguiente día con menor RMSE que la persistencia bajo evaluación cronológica?

Para cada modelo $m$ se define la mejora:

$$
D_m=\mathrm{RMSE}_{\mathrm{pers}}-\mathrm{RMSE}_{m}.
$$

La hipótesis operativa es que alguno de los tres modelos, que aún deben seleccionarse con literatura, alcance $D_m>\delta$ fuera de muestra, para un margen práctico $\delta>0$ **por fijar antes del experimento definitivo**. La comparación estadística de mejora positiva y la relevancia práctica superior a $\delta$ se distinguirán; no se declarará éxito solo por un valor $p$ pequeño. Las comparaciones múltiples requerirán corrección. No se afirma superioridad antes de experimentar.

| Elemento | Propuesta / estado |
|---|---|
| Métrica primaria | RMSE, pendiente de aprobar |
| Métricas reportadas | MAPE, MAE, RMSE y MSE por tamaño de entrada y horizonte; MASE adicional según §6.3 del Entregable 1 |
| Margen práctico $\delta$ | Pendiente; fijar valor y justificación en unidades del objetivo |
| Nivel de significación | $\alpha=0.05$ requerido por la guía; método y multiplicidad por registrar |
| Partición | Calendario proporcionado: historia 2023, validación 2024, test 2025–agosto de 2026; ventanas y elegibilidad pendientes |
| Unidad observacional | Origen temporal $t$ con entrada histórica y etiqueta futura |
| Unidad de independencia inferencial | Bloques temporales; definición, longitud y supuestos pendientes |
| Referencia contextual | Persistencia de la volatilidad histórica disponible al emitir el pronóstico |
| Selección de tres modelos | Pendiente de revisión de literatura; no hay arquitectura congelada |

Para $N$ predicciones evaluables, se propone:

$$
\mathrm{RMSE}=\sqrt{\frac1N\sum_t(Y_t-\widehat Y_t)^2},
\qquad
\mathrm{MAE}=\frac1N\sum_t|Y_t-\widehat Y_t|.
$$

Para MASE, se propone dividir MAE por la media de diferencias absolutas $|Y_t-Y_{t-s}|$ calculada **solo sobre etiquetas de entrenamiento completamente observadas**, para un rezago de referencia $s$ que debe fijarse. El denominador se compartirá entre modelos. Si vale cero, MASE no estará definida y se reportará como tal; no se sustituirá por un valor arbitrario. No se congelará MASE sin definir $s$ y la cadencia de emisión.

Las ventanas de entrada y las etiquetas futuras pueden solaparse. Ni las filas, ni distintas semillas sobre el mismo test, son réplicas independientes de mercados. Se deberá justificar la inferencia temporal mediante bloques y analizar sensibilidad a su longitud. Tener un solo activo limita la generalización a otros mercados.

## 1.5. Dataset: pertinencia y ficha técnica

**Fuente seleccionada:** Binance Spot, par BTCUSDT, velas de un minuto adquiridas directamente de los archivos públicos oficiales. La adquisición, verificación y auditoría se describen en §2.3 del [EDA](02_eda.md). Se verificaron 75 ZIP oficiales para el periodo 01/01/2023–30/10/2026 inclusive UTC: 2.014.480 filas. El manifiesto contiene los SHA-256 y la auditoría de timestamps. El EDA de calidad, precios y retornos de desarrollo está ejecutado y documentado en la sección 2; el análisis del objetivo y la construcción de ventanas supervisadas dependen de cerrar su definición.

Un conjunto de cierres de Bitcoin con marcas de tiempo y resolución suficiente permitiría construir entradas y etiquetas de manera trazable. Su idoneidad dependerá de cubrir suficientes regímenes y de contar con continuidad, procedencia y condiciones de uso verificables. Acceso público no equivale a licencia abierta ni a permiso de redistribución.

| Campo exigido | Estado y criterio de cierre |
|---|---|
| Nombre y proveedor | Binance Spot; archivos públicos oficiales de velas de un minuto |
| Activo y par | Bitcoin, cotizado en USDT; par BTCUSDT |
| Disponibilidad pública | Archivos oficiales accesibles y descargados desde data.binance.vision; la API devuelve HTTP 451 por ubicación y no se utiliza |
| Licencia y redistribución | Binance Vision Dataset Terms v1.0, actualizados el 26/08/2026: CC BY-NC-SA 4.0 salvo indicación específica, con condiciones adicionales; ver §2.3.4 |
| Procedencia | Pendiente: documentar generación de velas, cierre y agregación por proveedor |
| Uso previo del mismo dataset | No verificado; el antecedente temático no demuestra igualdad de dataset |
| Periodo temporal y zona horaria | Descarga: 01/01/2023–30/10/2026 inclusive UTC; timestamps de apertura en milisegundos antes de 2025 y microsegundos desde 2025. Calendario experimental en §1.6 |
| Frecuencia | Brutos de un minuto; observaciones de modelado y retornos del objetivo diarios, cierre UTC |
| Tamaño | 75 ZIP, 97.442.605 bytes comprimidos y 2.014.480 filas brutas; cierres válidos y ventanas finales pendientes |
| Estructura de modelado | OHLCV diario; entradas $(N,D,5)$ para $D=7,14,21,28$; salida escalar de volatilidad de un día adelante |
| Formato de almacenamiento | CSV oficiales de 12 columnas dentro de ZIP; timestamps normalizados solo para auditar cobertura, brutos intactos |
| Etiquetas | Cálculo automático de $Y_t$ según fórmula aprobada; sin anotación manual |
| Dependencia | Temporal, ventanas solapadas y cambios de régimen; se verificará en EDA |
| Versión / fecha de extracción | Instantánea verificada el 02/11/2026; versión identificada por el manifiesto y SHA-256 individuales |
| Hash de datos | SHA-256 de los 75 ZIP contrastados con checksums oficiales; manifiesto enlazado en §2.3.2. Derivados y etiquetas de desarrollo documentados en §2; configuraciones y manifiestos supervisados finales pendientes |

Con $T$ días OHLCV consecutivos válidos y sin pérdidas por partición, construir entradas de $D$ días y una etiqueta con $n$ retornos diarios permite como máximo $T-\max(D,n+1)$ orígenes de un día adelante, con historia suficiente para la referencia contextual. Es una relación teórica, **no el tamaño real**; huecos, filtros y separación entre particiones pueden reducirlo.

### 1.5.1. Etiquetado, ruido y controles previstos

El objetivo se derivará de precios observados, por lo que puede reflejar errores del proveedor, precios anómalos, cambios de liquidez y ruido de microestructura. El tamaño finito de la ventana y la definición elegida introducen incertidumbre como proxy. No se ha medido todavía la magnitud de estos efectos.

Antes de construir etiquetas se comprobarán orden temporal, duplicados, cierres no positivos, huecos, frecuencia real y significado del timestamp. No se rellenarán huecos ni se eliminarán extremos sin justificarlo; un retorno que cruza un hueco no debe confundirse con un retorno regular de un minuto.

### 1.5.2. Separación temporal y prevención de fuga

- Fijar entrenamiento, validación y test por tiempo; no mezclar aleatoriamente ventanas.
- Excluir del entrenamiento todo origen cuya etiqueta alcance validación o test.
- Ajustar escalamiento, imputación, selección de ventanas y otras decisiones solo con datos de desarrollo permitidos.
- Una ventana de validación/test puede usar historia previa ya observable; nunca retornos posteriores al origen.
- Implementar validación forward chaining con los objetivos de 2024 y separación temporal coherentes con $D$, $n$ y la disponibilidad de etiquetas.
- Congelar y verificar el test por hash antes de evaluación definitiva; no usarlo para elegir arquitectura, hiperparámetros o preprocesamiento.

## 1.6. Calendario proporcionado

La captura aclaratoria suministrada por el usuario especifica este calendario, que reemplaza la referencia histórica menos precisa «selección 2023–2024; evaluación 2025–2026»:

| Etapa | Periodo | Uso |
|---|---|---|
| Historia inicial | 01/01/2023–31/12/2023 | Construir las primeras muestras elegibles de entrenamiento |
| Validación forward chaining | Objetivos del 01/01/2024 al 31/12/2024 | Entrenar con el pasado y acumular errores de validación para comparar configuraciones |
| Ajuste final | Datos elegibles de 2023–2024 | Reajustar la configuración seleccionada |
| Test externo | 01/01/2025–31/08/2026 | Evaluar sin seleccionar configuraciones a partir del test |

Se reportarán **2025 y 2026 por separado**; el segundo año solo cubre hasta el 31 de agosto, no el año completo.

**Regla operativa propuesta, pendiente de materializar con $n$ y timestamps diarios:** asignar una etiqueta a una etapa solo si todo su intervalo futuro está contenido en ella. Una etiqueta que cruce del 31 de diciembre de 2024 a 2025 no puede usarse para ajuste final. La historia de entrada anterior al inicio de una etapa sí puede utilizarse si ya era observable. El entrenamiento en cada origen de validación solo incluirá etiquetas totalmente conocidas en ese instante. La política de actualización durante test todavía debe preregistrarse: no se presupone reentrenamiento con sus etiquetas.

Los días indicados son límites de calendario, no timestamps exactos ni garantía de disponibilidad del proveedor. La zona horaria es UTC y la cadencia principal es diaria; los límites de disponibilidad y las reglas de elegibilidad se materializarán en el manifiesto de muestras antes de entrenar. Se contaron filas y timestamps brutos para auditar adquisición; no se han construido muestras supervisadas ni explorado precios o resultados de test. La descarga adicional de septiembre–octubre de 2026 no cambia automáticamente el calendario experimental.

## 1.7. Estado de cierre

La formulación es revisable y la ficha de datos está deliberadamente incompleta. La sección solo podrá declararse cerrada después de fijar $n$, completar la auditoría de desarrollo y congelar métricas, $\delta$, partición y unidad inferencial. La [lista de decisiones pendientes](protocolo_pendiente.md) evita confundir texto redactado con evidencia experimental.


### 1.7.1. Reporte exigido por cada tamaño de entrada

Se entregarán resultados de test para 7, 14, 21 y 28 días de entrada, idealmente por fold y con media y desviación estándar: **MAPE, MAE, RMSE y MSE**, por horizonte y agregados cuando corresponda. Se añadirá MASE según la guía general de series de tiempo. La configuración principal tiene un horizonte de un día; no se inventan salidas adicionales.

El **BDS** se aplicará a los residuos del primer horizonte, con parámetros, tamaño de muestra, p-value y supuestos documentados por fold. No se puede calcular todavía: no existen predicciones entrenadas ni residuos de test. BDS sobre retornos crudos no sustituiría este requisito.

MAPE requiere definir el tratamiento de etiquetas de volatilidad cero: no se añadirá un epsilon arbitrario ni se ocultarán casos excluidos. Hasta fijar esa política y ejecutar los modelos, las tablas de desempeño permanecen pendientes.
