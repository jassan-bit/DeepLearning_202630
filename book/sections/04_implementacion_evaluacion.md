# 4. Implementación y Evaluación de los Tres Modelos Base

Los tres modelos extraídos de la literatura deberán implementarse, compararse bajo el **mismo entorno experimental** y evaluarse con un protocolo común. Esta sección documenta el diseño y los requisitos del piloto; no presenta entrenamientos ni resultados que aún no existen.

```{important}
Los datos y el EDA de desarrollo están disponibles y los folds de validación se materializaron con `timeseries-cv`. Todavía faltan la selección bibliográfica de los tres modelos, el margen práctico y el cierre de configuraciones. El piloto, el dimensionamiento y el benchmark no se han ejecutado.
```

## 4.1. Diseño Experimental

### 4.1.1. Variables de entrada y representación

La unidad de modelado es un **día UTC de BTCUSDT**, construido con las velas oficiales de un minuto. Las covariables de partida son OHLCV diario; el cierre es la variable central para calcular retornos y volatilidad. No se incorporan noticias sin una fuente y disponibilidad temporal verificadas.

Para una historia de $D\in\{7,14,21,28\}$ días, la representación inicial es $X\in\mathbb R^{N\times D\times5}$. Una MLP puede utilizar una vista aplanada de $5D$ componentes; una red secuencial conserva los $D$ pasos. Cambiar la disposición del tensor no permite dar a un modelo información adicional. La representación y cualquier selección de atributos se fijarán antes de comparar los modelos.

Las ventanas no se forman atravesando días incompletos. Los ZIP originales permanecen intactos y las exclusiones se reproducen mediante el código y los manifiestos de las secciones 1 y 2.

### 4.1.2. Variable objetivo y horizonte

Se exploran cuatro definiciones de volatilidad histórica poblacional de **retornos diarios**, con $n=7,14,21,28$ y divisor $n$, sin anualizar. Esta ventana es distinta de $D$, la longitud de entrada.

Se mantiene el horizonte principal de **un día adelante** y la convención de §1.3: el pronóstico se emite al inicio del día $t$, usando información hasta $t-1$, para una volatilidad que incorpora el retorno todavía desconocido del día $t$. La salida es escalar y no negativa. Los análisis de EDA no han seleccionado una definición ganadora ni autorizado horizontes adicionales.

La configuración inicial propone **MSE como pérdida** y **RMSE como métrica primaria**. El reporte incluirá MAPE, MAE, RMSE y MSE, con MASE adicional según la guía de series de tiempo. Antes del entrenamiento deben cerrarse la política de ceros de MAPE y el rezago del denominador MASE; no se usarán resultados de test para elegirlos.

### 4.1.3. Partición, proporciones y validación

El calendario experimental conserva entrenamiento inicial de 2023, validación creciente durante 2024 y test previsto de 2025–agosto de 2026. Se diferencian proporciones de **calendario** de proporciones de muestras supervisadas:

| Etapa | Días de calendario UTC | Proporción del calendario experimental | Estado de elegibilidad |
|---|---:|---:|---|
| Historia inicial de entrenamiento, 2023 | 365 | 27,26 % | 364 días OHLCV completos; muestras según ventana |
| Validación, 2024 | 366 | 27,33 % | 366 días completos y un fold diario por configuración |
| Test previsto, 01/01/2025–31/08/2026 | 608 | 45,41 % | Muestras supervisadas y congelación final pendientes |

Las proporciones suman el calendario de 1.339 días; no son un reparto aleatorio ni los pesos efectivos de entrenamiento. Los días de septiembre–octubre de 2026 descargados quedan fuera de ese calendario, sin asignación automática al test.

Se ejecutó `split_train_val_forwardChaining` sobre índices diarios. El manifiesto contiene **366 folds por configuración**, para 16 combinaciones de $D$ y $n$. Las etiquetas de entrenamiento se comprueban contra su disponibilidad en cada origen. Al avanzar la validación, pueden incorporarse etiquetas anteriores de 2024 ya observadas, según los límites que produce la función; nunca etiquetas futuras.

La cronología es determinista y no necesita semilla para barajar. Las semillas sí se registrarán para inicialización, optimización y otras operaciones estocásticas. La selección de hiperparámetros utilizará exclusivamente desarrollo y el mismo calendario, presupuesto y criterio entre modelos; no se ha ejecutado aún esa búsqueda. Si se informa desempeño de 2024 después de usarlo para seleccionar configuraciones, se identificará como desempeño de selección, no como estimación de test independiente.

### 4.1.4. Pipeline común y controles de fuga

1. **Construcción de datos:** comprobar hashes y disponibilidad; agregar días completos; formar entradas y etiquetas alineadas; aplicar las mismas exclusiones para los tres modelos dentro de cada configuración.
2. **Preprocesamiento:** ajustar escalado, imputación permitida y selección de variables solo con el entrenamiento admitido de cada fold. Los parámetros exploratorios de PCA no se transfieren automáticamente al experimento.
3. **Entrenamiento:** inicialización nueva por semilla; arquitectura, optimizador, pérdida, batch size, épocas máximas y regla de parada registrados antes de comparar resultados.
4. **Early stopping:** utilizar datos pasados de ajuste interno, sin consultar el objetivo futuro del fold evaluado. El test no puede intervenir en la parada ni en decisiones de entrenamiento.
5. **Validación:** conservar predicciones, objetivos, fechas, configuración, semilla y duración por fold. Calcular las métricas con la agregación predefinida.
6. **Evaluación definitiva:** congelar selección y manifiestos antes de test; aplicar el procedimiento una vez fijado y reportar 2025 y 2026 por separado.

Los folds tienen una salida diaria. Por ello, para cada semilla y configuración se concatenarán las predicciones fuera de entrenamiento y se calculará:

$$
\operatorname{RMSE}_{m,s}=\sqrt{\frac1{N_{\mathrm{val}}}
\sum_{t\in\mathcal V}(Y_t-\widehat Y_{m,s,t})^2}.
$$

**No se promediarán las RMSE de folds unitarios como si fueran la RMSE global:** ese promedio equivale a MAE. El reporte por fold se conservará por trazabilidad, separado de la métrica agregada de cada ejecución.

### 4.1.5. Estudio piloto: al menos cinco ejecuciones por modelo

Antes del experimento definitivo se ejecutarán **al menos cinco semillas distintas de cada uno de los tres modelos**, sobre el mismo protocolo de validación. Se prevén las semillas comunes `11, 23, 37, 53, 71`; no representan ejecuciones ya realizadas. Cada repetición parte de una inicialización nueva y produce la métrica primaria agregada de su configuración.

Para valores $m_1,\ldots,m_R$ de RMSE por ejecución se reportarán:

$$
\bar m=\frac1R\sum_{s=1}^R m_s,\qquad
s_m=\sqrt{\frac1{R-1}\sum_{s=1}^R(m_s-\bar m)^2},\qquad
\mathrm{CV}=\frac{s_m}{\bar m}.
$$

CV puede mostrarse como porcentaje multiplicando por 100; si la media es cero, no está definido. Se conservarán ejecuciones fallidas y su diagnóstico; no se descartarán semillas por obtener resultados desfavorables.

La selección de configuraciones y folds incluidos en el piloto se cerrará antes de ejecutarlo. Cubrir las 16 configuraciones con tres modelos, cinco semillas y 366 folds supondría **87.840 ajustes de modelo**; es un conteo de diseño, no trabajo ejecutado ni estimación de tiempo. El alcance y los recursos deben fijarse antes de lanzar esa carga, sin reducirlos retrospectivamente para favorecer un resultado.

Las semillas describen variabilidad del entrenamiento sobre un mercado observado. **Las 366 fechas y las 5.856 filas configuración–fold no son $R$ repeticiones independientes**, y distintas semillas tampoco constituyen mercados independientes.

### 4.1.6. Diferencia mínima de interés y diferencias pareadas

El margen práctico $\delta>0$ debe fijarse antes del piloto, sin elegirlo para acomodar sus resultados, con justificación en las unidades de la RMSE y coherencia con §1.4. Sigue pendiente; no se inventa un valor para producir un tamaño muestral.

Para un contraste predefinido entre modelos $A$ y $B$, sobre las mismas fechas y semillas:

$$
d_s=m_{A,s}-m_{B,s},\qquad
\widehat\sigma_d=
\sqrt{\frac1{R-1}\sum_{s=1}^R(d_s-\bar d)^2},\qquad
\text{efecto estandarizado}=\frac{\delta}{\widehat\sigma_d}.
$$

La dispersión relevante es la de las **diferencias pareadas**, no la desviación de un modelo aislado ni el error entre días tratado como independiente. Se fijarán los contrastes principales y el tratamiento de multiplicidad antes del dimensionamiento.

### 4.1.7. Dimensionamiento de ejecuciones y potencia

La guía requiere potencia $1-\beta\ge0{,}80$ y nivel $\alpha=0{,}05$. Bajo la aproximación normal bilateral para las diferencias pareadas:

$$
R_{\mathrm{req}}\approx
\left\lceil
\frac{\left(z_{1-\alpha/2}+z_{1-\beta}\right)^2\widehat\sigma_d^2}{\delta^2}
\right\rceil.
$$

El factor de cuantiles es aproximadamente **7,85** para esos valores. El número definitivo respetará el mínimo aplicable y el mayor requerimiento de los contrastes preespecificados. Si se utilizan pruebas por rangos, se atenderá además el mínimo de **seis ejecuciones** indicado en §4.3 de la guía; cinco es el mínimo del piloto, no un tamaño definitivo garantizado.

Se documentarán los supuestos y la incertidumbre de estimar dispersión con pocas repeticiones. La aproximación no resuelve por sí sola dependencia temporal o multiplicidad; si se planifican varias afirmaciones, se registrará el ajuste y se revisará la potencia correspondiente. Si la normalidad no es defendible, se justificará una alternativa o simulación de potencia que respete el diseño, sin escogerla según los resultados de test.

Una dispersión piloto estimada como cero requiere investigar redondeo, determinismo o piloto insuficiente; no justifica concluir que no hacen falta ejecuciones. **No hay un valor calculado de $R$ todavía**, porque faltan $\delta$ y las diferencias observadas del piloto.

### 4.1.8. Diferencia mínima detectable

Para las ejecuciones efectivamente disponibles se reportará la aproximación:

$$
\mathrm{MDE}(R)\approx
\frac{\left(z_{1-\alpha/2}+z_{1-\beta}\right)\widehat\sigma_d}{\sqrt R}.
$$

Si MDE es mayor que $\delta$, se ampliará el diseño o se declarará esa limitación explícitamente. No se afirmará potencia suficiente por haber completado cinco semillas. El valor $\widehat\sigma_d$ servirá como punto de partida para el dimensionamiento del Entregable 2, sin considerarlo definitivo.

### 4.1.9. Preregistro fechado y trazabilidad

Se creó el [borrador de preregistro](../_static/experiments/preregistro_piloto.json), conservado en `experiments/preregistro_piloto.json`, fechado el **2 de noviembre de 2026**, con configuración conocida, semillas previstas y campos por cerrar. Su estado es **borrador no congelado**, no un preregistro experimental completo. La [plantilla CSV de resultados](../_static/experiments/piloto_resultados_plantilla.csv) contiene solo encabezados y no representa ejecuciones realizadas.

Antes del piloto se cerrará el protocolo inicial. Después se incorporarán sus resultados y el dimensionamiento en una versión fechada y congelada **antes de la ejecución definitiva**, conservando el historial de cambios. El registro debe identificar los modelos, datos y hashes, código, folds, objetivos, comparaciones, transformaciones, reglas de parada, $\delta$, métrica, supuestos y número final de ejecuciones.

### 4.1.10. Estado del diseño y del piloto

| Producto | Estado |
|---|---|
| Datos y EDA de desarrollo | Ejecutados y documentados |
| Folds de validación de 2024 | Materializados con `timeseries-cv` |
| Selección bibliográfica de tres modelos | Pendiente |
| Configuraciones, contrastes, margen práctico y política de métricas | Pendientes de cierre |
| Preregistro | Borrador fechado; no congelado |
| Cinco ejecuciones piloto por modelo | No ejecutadas |
| Media, desviación, CV y diferencias pareadas | Sin resultados todavía |
| $R$ requerido y MDE | Pendientes de piloto y $\delta$ |
| Entrenamiento definitivo y test | No ejecutados |

La presencia de esta sección no sustituye las implementaciones y evaluaciones posteriores del Entregable 1.
