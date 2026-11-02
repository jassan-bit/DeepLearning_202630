# 7. Limitaciones Inferenciales de Trabajar con Tres Modelos

Un benchmark con tres modelos y pocas ejecuciones puede tener **potencia muy baja**. El Entregable 1 exige reconocerlo y cuantificarlo, en lugar de interpretar los p-values como si cualquier diseño pudiera detectar diferencias pequeñas.

```{important}
Las limitaciones siguientes son propiedades y riesgos del diseño; no resultados observados de los tres modelos. Sus implementaciones, piloto y comparaciones aún están pendientes. Un resultado inconcluso bien sustentado es aceptable; no se buscará significación a cualquier costo.
```

## 7.1. Potencia, tamaño de muestra y multiplicidad

### 7.1.1. Friedman con tres modelos y cinco bloques

Con tres modelos y cinco bloques, Friedman puede tener poca potencia y el post-hoc de Nemenyi ser conservador. La ausencia de significación no demuestra equivalencia ni descarta diferencias de importancia práctica.

Los bloques deben representar unidades defendibles del experimento. Las 366 fechas de validación, las semillas y las configuraciones no se mezclarán como si fueran réplicas independientes de la misma naturaleza. Se conservarán el emparejamiento, el alcance de la inferencia y la dependencia temporal descritos en §5.4.

### 7.1.2. Límite del Wilcoxon pareado

Para $R$ pares sin ceros ni empates y un contraste bilateral exacto, el menor p-value posible, cuando todas las diferencias tienen la misma dirección, es:

$$
p_{\min}=\frac{2}{2^R}.
$$

Con **cinco pares**, $p_{\min}=0{,}0625>0{,}05$: ese Wilcoxon bilateral no puede rechazar al nivel indicado ni en el caso más extremo. Con **seis**, el mínimo es $0{,}03125$. El tratamiento de ceros, empates y aproximaciones del software debe declararse; el tamaño efectivo puede diferir del nominal.

La guía establece cinco ejecuciones como mínimo general y seis para pruebas por rangos, pero el número definitivo deberá responder al piloto y la potencia. No se elegirá una prueba paramétrica solo para conseguir significación: sus supuestos deben justificarse.

### 7.1.3. Corrección de Holm con tres pares

Tres modelos producen tres comparaciones pareadas dentro de una configuración. En Holm, el menor p-value se compara primero con $\alpha/3\approx0{,}0167$ cuando $\alpha=0{,}05$.

Por tanto, seis pares tampoco garantizan superar ese primer umbral con Wilcoxon bilateral exacto: su mínimo de $0{,}03125$ sigue siendo mayor. Incluso cuando el número de pares permita un p-value suficientemente pequeño, eso no garantiza potencia adecuada para el efecto real que interesa detectar.

Si se realizan afirmaciones sobre varias ventanas, objetivos u horizontes, la familia de comparaciones debe reflejarlo. No se ocultará esa ampliación tratando cada tabla como si fuera la única explorada.

## 7.2. Cómo abordar estas limitaciones

### 7.2.1. Repeticiones y validación respetando el tiempo

La guía propone considerar validación cruzada repetida con correcciones de varianza, por ejemplo Nadeau–Bengio, cuando el diseño y sus supuestos lo permiten. Esa recomendación **no autoriza aplicar K-fold aleatorio a BTCUSDT** ni convertir un esquema genérico 5 × 5 en una regla automática para series de tiempo.

Este proyecto mantiene validación de origen móvil y controles de disponibilidad temporal. Si se combinan semillas y distintos tramos u orígenes, se documentará su dependencia y se justificará la inferencia resultante. No se aplicará una corrección derivada para otro diseño sin comprobar su adecuación al entrenamiento creciente y al solapamiento de etiquetas.

Cuando corresponda se utilizará remuestreo temporal por bloques, con definición y sensibilidad documentadas. Aumentar el número de filas de un archivo de resultados sin aumentar información independiente no resuelve la baja potencia.

### 7.2.2. Diferencia mínima detectable y tamaños de efecto

Se reportará la **MDE con el número de ejecuciones disponible**, su relación con $\delta$ y los supuestos del dimensionamiento de §4.1. Si el estudio no puede distinguir una diferencia de interés práctico, se ampliará el diseño o se explicitará esa limitación.

Las comparaciones incluirán tamaños de efecto e intervalos de confianza, conservando el emparejamiento. Los intervalos temporales deberán respetar bloques cuando proceda; no se remuestrearán días dependientes como si fueran i.i.d.

No se interpretará un intervalo amplio como equivalencia ni una diferencia media pequeña como ausencia de efecto. La precisión de la estimación forma parte del resultado y debe figurar junto a su valor puntual.

### 7.2.3. Probabilidad de superioridad

Puede complementarse la comparación con una probabilidad de superioridad definida explícitamente. Para métricas pareadas en las que menor es mejor, una estimación descriptiva entre las ejecuciones observadas es:

$$
\widehat p_{A<B}=\frac1R\sum_{s=1}^{R}
\left[\mathbb I(m_{A,s}<m_{B,s})+
\frac12\mathbb I(m_{A,s}=m_{B,s})\right].
$$

Se reportarán unidad, tratamiento de empates, tamaño e incertidumbre. Esta proporción describe las repeticiones consideradas y no se presentará como probabilidad universal de ganar en cualquier mercado o periodo. Si se estima sobre fechas o bloques, se cambiará y justificará la unidad, sin mezclarla con semillas.

### 7.2.4. Equivalencia requiere un análisis separado

Para sostener equivalencia se utilizará un procedimiento predefinido, como **TOST** con margen aplicado o un análisis bayesiano con **ROPE**, con sus supuestos y reglas respectivos. El margen no se elegirá después de ver los resultados y la ausencia de significación no lo sustituye.

El análisis distinguirá tres situaciones: evidencia de diferencia, evidencia de equivalencia bajo un margen definido y resultado inconcluso. No se forzará a todos los resultados a encajar en “gana” o “empata”.

## 7.3. Objetivo y criterio de honestidad inferencial

El propósito del entregable es **establecer el baseline y su variabilidad con evidencia**, no obtener $p<0{,}05$ a toda costa. Un resultado no concluyente deberá explicar el efecto estimado, su intervalo, la sensibilidad del diseño y las limitaciones.

No se cambiarán retrospectivamente la métrica, los pares, las semillas, los periodos, las reglas de exclusión o las pruebas para conseguir un resultado favorable. Los análisis añadidos después se identificarán como exploratorios y se distinguirán de los confirmatorios.

Todavía no se ha medido la potencia ni la superioridad entre los modelos del proyecto. Las advertencias teóricas de este capítulo no constituyen pruebas ejecutadas sobre sus resultados.

## 7.4. Entregables y estado real

| Producto exigido | Evidencia disponible | Trabajo pendiente |
|---|---|---|
| Informe en Jupyter Book | Capítulos y navegación publicados; EDA con figuras | Completar productos experimentales y conclusiones sustentadas |
| Código completo de extremo a extremo | Scripts de adquisición, EDA y partición | Tres modelos, entrenamiento, evaluación y análisis estadístico ejecutables |
| Dataset o instrucciones exactas de adquisición | Archivos oficiales, URLs, checksums, manifiesto y descargador | Conservar trazabilidad de derivados finales y test supervisado |
| Síntesis de literatura y flujo de búsqueda | Estrategia y plantillas de la sección 3 | Búsquedas, conteos, cribado, matriz real y diagrama de flujo |
| Tabla comparativa, baseline e inferencia | Protocolo y tablas previstas en la sección 5 | Predicciones, métricas, intervalos, efectos, contrastes y baseline medido |
| Dependencias, semillas y configuración de cada corrida | Entornos de EDA/partición y borrador de preregistro | Entorno de entrenamiento, registros y artefactos de las corridas |
| Declaraciones a priori de métrica, margen, unidad y partición | RMSE propuesta, calendario, unidades discutidas y preregistro fechado como borrador | Fijar $\delta$, plan inferencial y configuración final antes de ejecutar |

La disponibilidad del libro no acredita que todos los productos estén completos. Los resultados inexistentes no se reemplazarán por ejemplos, tablas vacías presentadas como mediciones o conclusiones de literatura atribuidas al experimento propio.

## 7.5. Criterios de evaluación del entregable

Los pesos de la rúbrica suministrada son:

| Criterio | Peso |
|---|---:|
| EDA | 30 % |
| Revisión de literatura y extracción de los tres modelos base | 30 % |
| Implementación de los tres modelos base | 20 % |
| Benchmark, definición del baseline y análisis estadístico | 20 % |
| **Total** | **100 %** |

Estos valores son **pesos de evaluación**, no porcentajes de avance ni una calificación obtenida. Haber redactado una sección o superado la construcción HTML no equivale a satisfacer el criterio científico correspondiente.

## 7.6. Buenas prácticas y cierre

El cierre del Entregable 1 deberá comprobar que:

1. Se realizaron múltiples ejecuciones, con el mínimo aplicable y el número que exija la potencia.
2. Se reportaron media, desviación muestral, IC del 95 % y tamaños de efecto con unidades identificadas.
3. Se justificaron las pruebas, sus supuestos, correcciones y limitaciones de independencia.
4. Se mantuvo consistencia experimental entre los tres modelos: datos, pipeline, particiones, semillas y esfuerzo de selección.
5. El código, las dependencias, los datos o sus instrucciones, y las configuraciones permiten reproducir el procedimiento completo.
6. La métrica principal, $\delta$, la unidad inferencial y el protocolo quedaron fijados antes de la evaluación pertinente.
7. El baseline y el margen exigible a modelos avanzados se concluyeron desde resultados verificables, distinguiendo significación y relevancia práctica.

El análisis esperado es exhaustivo y reconoce la potencia limitada de comparar solo tres modelos. En el estado actual, este cierre es una lista de comprobación pendiente de completar con la revisión, implementaciones, piloto y benchmark; no una declaración de cumplimiento total.
