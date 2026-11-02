# 1. Definición del Problema y Dataset

## 1.1. Motivación y alcance

Se propone evaluar si el historial de rendimientos logarítmicos derivados del cierre de Bitcoin contiene información útil para pronosticar su volatilidad futura. El interés aplicado es estimar incertidumbre del activo, no predecir directamente su precio ni recomendar operaciones financieras.

El estudio de Huang, Sangiorgi y Urquhart (2024) examina pronósticos de volatilidad de Bitcoin con datos de alta frecuencia y redes neuronales; respalda la pertinencia de investigar este problema. Su definición de volatilidad, datos y horizontes no son automáticamente equivalentes a los de este proyecto: sus resultados no se trasladan como resultados propios. Véase la [referencia primaria](referencias.md).

**Estado:** Bitcoin y trabajo sobre cierre/volatilidad están confirmados. La formulación de pronóstico siguiente es una **propuesta pendiente de aprobación**, en particular su horizonte y proxy objetivo.

## 1.2. Tipo de problema y reloj de información

**Tipo propuesto: series de tiempo**, con salida continua y aprendizaje supervisado. El historial de la misma serie financiera constituye la información predictora; no se tratarán sus ventanas como filas i.i.d. ni se realizará partición aleatoria. No es una tarea de clasificación.

Sea $P_t>0$ el precio de cierre de Bitcoin de una barra finalizada en el instante $t$, con separación temporal regular $\Delta$. Se define:

$$
r_t=\log\left(\frac{P_t}{P_{t-1}}\right).
$$

Cada pronóstico se emite **después de observar el cierre $P_t$**. En ese instante están disponibles únicamente datos con índice menor o igual a $t$.

El fragmento suministrado indica frecuencia de **1 minuto**, pero su leyenda utiliza «día» y «retorno diario». Por eso $\Delta$ se mantiene simbólico: no se presupone que esas dos convenciones sean compatibles. Si se confirma $\Delta=1$ minuto, todos los índices siguientes representarán minutos, no días.

## 1.3. Formulación matemática propuesta

Sean $L\geq1$ la longitud de entrada y $h\geq2$ el número de retornos futuros usados en el objetivo. El horizonte físico es $H=h\Delta$. Ambos valores están pendientes.

### Entrada, salida y función objetivo

$$
\mathcal X=\mathbb R^{L\times1},\qquad
X_t=(r_{t-L+1},\ldots,r_t)^\top.
$$

Para construir $L$ retornos se necesitan $L+1$ cierres, desde $P_{t-L}$ hasta $P_t$. La entrada tiene una característica por paso temporal; no se incorporan volumen, noticias ni otras criptomonedas.

$$
\bar r^{+}_{t,h}=\frac1h\sum_{j=1}^{h}r_{t+j},\qquad
Y_t=\sigma^{+}_{t,h}
=\sqrt{\frac1h\sum_{j=1}^{h}(r_{t+j}-\bar r^{+}_{t,h})^2}.
$$

$$
\mathcal Y=\mathbb R_{\geq0},\qquad
f_\theta:\mathcal X\longrightarrow\mathcal Y,\qquad
\widehat Y_t=f_\theta(X_t).
$$

$Y_t$ es la desviación estándar poblacional de los retornos del intervalo futuro $(t,t+h]$: un **proxy de dispersión**, no volatilidad latente observada sin error. Se propone el divisor $h$ para ser coherentes con el divisor $n$ de la guía. No se anualiza, no se convierte a porcentaje y no equivale a $\sqrt{\sum r^2}$; cambiar de definición exige revisar el protocolo. $h=1$ produciría un objetivo siempre cero con esta definición y queda excluido.

La salida no negativa deberá garantizarse en la implementación futura. Las etiquetas solo se conocen al llegar a $t+h$, nunca en el momento del pronóstico.

### Pérdida propuesta

Para el conjunto de orígenes de entrenamiento $\mathcal T_{\mathrm{train}}$:

$$
\mathcal L(\theta)=\frac{1}{|\mathcal T_{\mathrm{train}}|}
\sum_{t\in\mathcal T_{\mathrm{train}}}
\left(Y_t-f_\theta(X_t)\right)^2.
$$

Se propone minimizar **MSE**. Las transformaciones que se decidan se ajustarán exclusivamente con entrenamiento. Las métricas se reportarán en la escala original del objetivo, invirtiendo cualquier transformación.

### Relación con la fórmula recibida

La imagen del curso presenta $\sigma_t$ con retornos $r_{t-1},\ldots,r_{t-n}$, que excluyen $r_t$. Esa fórmula describe una ventana histórica. La definición propuesta de $Y_t$ usa explícitamente los retornos futuros $r_{t+1},\ldots,r_{t+h}$ para distinguir **estimar volatilidad pasada** de **pronosticar volatilidad futura**. No se afirma que la imagen ya haya fijado este objetivo.

Como referencia contextual, se propone persistencia con una ventana pasada de la misma duración:

$$
\bar r^{-}_{t,h}=\frac1h\sum_{j=0}^{h-1}r_{t-j},\qquad
\widehat Y_t^{\mathrm{pers}}
=\sqrt{\frac1h\sum_{j=0}^{h-1}(r_{t-j}-\bar r^{-}_{t,h})^2}.
$$

Este baseline usa solo información conocida en $t$. Requiere al menos $h$ retornos históricos y se evaluará en los mismos orígenes que todos los modelos. No cuenta como uno de los tres modelos de Deep Learning.

## 1.4. Pregunta, hipótesis y evaluación a priori

**Pregunta propuesta:** ¿pueden modelos simples de Deep Learning, usando únicamente una ventana histórica de retornos de Bitcoin, pronosticar la dispersión de los retornos del siguiente intervalo $H$ con menor error que la persistencia, bajo evaluación estrictamente cronológica?

Para cada modelo $m$ se define la mejora:

$$
D_m=\mathrm{RMSE}_{\mathrm{pers}}-\mathrm{RMSE}_{m}.
$$

La hipótesis operativa es que alguno de los tres modelos, que aún deben seleccionarse con literatura, alcance $D_m>\delta$ fuera de muestra, para un margen práctico $\delta>0$ **por fijar antes del experimento definitivo**. La comparación estadística de mejora positiva y la relevancia práctica superior a $\delta$ se distinguirán; no se declarará éxito solo por un valor $p$ pequeño. Las comparaciones múltiples requerirán corrección. No se afirma superioridad antes de experimentar.

| Elemento | Propuesta / estado |
|---|---|
| Métrica primaria | RMSE, pendiente de aprobar |
| Métricas secundarias | MAE y MASE por horizonte, pendientes de aprobar |
| Margen práctico $\delta$ | Pendiente; fijar valor y justificación en unidades del objetivo |
| Nivel de significación | $\alpha=0.05$ requerido por la guía; método y multiplicidad por registrar |
| Partición | Calendario proporcionado: historia 2023, validación 2024, test 2025–agosto de 2026; ventanas y elegibilidad pendientes |
| Unidad observacional | Origen temporal $t$ con entrada histórica y etiqueta futura |
| Unidad de independencia inferencial | Bloques temporales; definición, longitud y supuestos pendientes |
| Referencia contextual | Persistencia de igual duración, propuesta |
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

**No se ha seleccionado ni descargado un dataset.** Por ello, esta sección especifica qué debe verificarse y no atribuye licencia, volumen, calidad ni procedencia a datos inexistentes.

Un conjunto de cierres de Bitcoin con marcas de tiempo y resolución suficiente permitiría construir entradas y etiquetas de manera trazable. Su idoneidad dependerá de cubrir suficientes regímenes y de contar con continuidad, procedencia y condiciones de uso verificables. Acceso público no equivale a licencia abierta ni a permiso de redistribución.

| Campo exigido | Estado y criterio de cierre |
|---|---|
| Nombre y proveedor | Pendiente: elegir fuente exacta y mercado/exchange |
| Activo y par | Bitcoin confirmado; moneda de cotización y par pendientes |
| Disponibilidad pública | Pendiente: verificar URL/API y condiciones de acceso |
| Licencia y redistribución | Pendiente: registrar términos/versiones y permiso para conservar/publicar datos |
| Procedencia | Pendiente: documentar generación de velas, cierre y agregación por proveedor |
| Uso previo del mismo dataset | No verificado; el antecedente temático no demuestra igualdad de dataset |
| Periodo temporal y zona horaria | Calendario confirmado en §1.6; zona horaria y convención de apertura/cierre pendientes |
| Frecuencia | 1 minuto sugerido por imagen; confirmar coherencia con leyenda diaria |
| Tamaño | Pendiente: filas brutas, cierres válidos y ventanas finales; no se estima como medido |
| Estructura mínima propuesta | Marca temporal única y cierre positivo; $X$ con forma $(N,L,1)$, $Y$ con forma $(N,)$ |
| Formato de almacenamiento | Pendiente; CSV o Parquet son opciones, no archivos adquiridos |
| Etiquetas | Cálculo automático de $Y_t$ según fórmula aprobada; sin anotación manual |
| Dependencia | Temporal, ventanas solapadas y cambios de régimen; se verificará en EDA |
| Versión / fecha de extracción | Pendiente de adquisición |
| Hash de datos | Pendiente: SHA-256 de cada archivo bruto y manifiesto de transformaciones |

Con $T$ cierres consecutivos válidos y sin pérdidas por partición, usar simultáneamente una entrada de $L$ retornos y el baseline de $h$ retornos permite como máximo $T-\max(L,h)-h$ orígenes. Es una relación teórica, **no el tamaño real**; huecos, filtros y separación entre particiones pueden reducirlo.

### Etiquetado, ruido y controles previstos

El objetivo se derivará de precios observados, por lo que puede reflejar errores del proveedor, precios anómalos, cambios de liquidez y ruido de microestructura. El tamaño finito de la ventana y la definición elegida introducen incertidumbre como proxy. No se ha medido todavía la magnitud de estos efectos.

Antes de construir etiquetas se comprobarán orden temporal, duplicados, cierres no positivos, huecos, frecuencia real y significado del timestamp. No se rellenarán huecos ni se eliminarán extremos sin justificarlo; un retorno que cruza un hueco no debe confundirse con un retorno regular de un minuto.

### Separación temporal y prevención de fuga

- Fijar entrenamiento, validación y test por tiempo; no mezclar aleatoriamente ventanas.
- Excluir del entrenamiento todo origen cuya etiqueta alcance validación o test.
- Ajustar escalamiento, imputación, selección de ventanas y otras decisiones solo con datos de desarrollo permitidos.
- Una ventana de validación/test puede usar historia previa ya observable; nunca retornos posteriores al origen.
- Implementar validación forward chaining con los objetivos de 2024 y separación temporal coherentes con $L$, $h$ y la disponibilidad de etiquetas.
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

**Regla operativa propuesta, pendiente de concretar con $h$ y timestamps:** asignar una etiqueta a una etapa solo si todo su intervalo futuro está contenido en ella. Una etiqueta que cruce del 31 de diciembre de 2024 a 2025 no puede usarse para ajuste final. La historia de entrada anterior al inicio de una etapa sí puede utilizarse si ya era observable. El entrenamiento en cada origen de validación solo incluirá etiquetas totalmente conocidas en ese instante. La política de actualización durante test todavía debe preregistrarse: no se presupone reentrenamiento con sus etiquetas.

Los días indicados son límites de calendario, no timestamps exactos ni garantía de disponibilidad del proveedor. La zona horaria, inclusividad de límites, cadencia y datos elegibles se fijarán antes de construir ventanas. No se han contado muestras ni accedido al test.

## 1.7. Estado de cierre

La formulación es revisable y la ficha de datos está deliberadamente incompleta. La sección solo podrá declararse cerrada después de aprobar objetivo/horizonte, seleccionar y auditar el dataset y congelar métricas, $\delta$, partición y unidad inferencial. La [lista de decisiones pendientes](protocolo_pendiente.md) evita confundir texto redactado con evidencia experimental.
