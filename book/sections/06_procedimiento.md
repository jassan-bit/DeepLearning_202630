# 6. Guía de Procedimiento por Tipo de Problema

Este proyecto sigue la ruta de **Series de Tiempo del apartado 6.3 del Entregable 1**. Se pronostica volatilidad futura de BTCUSDT mediante información histórica del mismo proceso financiero; no se trata como una tabla de observaciones independientes.

La numeración de este capítulo es continua dentro del libro. La referencia al apartado 6.3 identifica su ubicación en la guía original; no obliga a reproducir las rutas de clasificación, regresión espacial, imágenes o texto, que no corresponden al alcance actual.

```{important}
Este capítulo aplica la ruta metodológica de la guía a las decisiones y evidencias del proyecto. Adquisición, EDA de desarrollo y folds están ejecutados. La revisión bibliográfica, las implementaciones, el piloto y el benchmark siguen pendientes; describir su procedimiento no acredita haberlos realizado.
```

## 6.1. Ruta aplicable: series de tiempo

### 6.1.1. Partición cronológica y origen móvil

Se respetará la relación temporal **entrenamiento < validación < prueba**, interpretada por disponibilidad de información en cada pronóstico. La historia inicial es 2023; los orígenes de validación están en 2024 y el test previsto comprende 2025–agosto de 2026. Las etiquetas futuras no entran en entrenamiento y las transformaciones no se ajustan con información posterior al origen evaluado.

Los folds se materializaron con `split_train_val_forwardChaining`, con entradas de 7/14/21/28 días, horizonte principal de un día y avance de un día. Se conservan los manifiestos y sus controles de disponibilidad en la [sección 2](02_eda.md). La función trabaja con índices del calendario y el código mapea entradas OHLCV y etiquetas de volatilidad por separado.

La validación creciente puede incorporar etiquetas anteriores de 2024 cuando ya se conocen. Eso no permite acceder al día futuro evaluado ni reutilizar la etiqueta externa para early stopping. La reutilización de historia observable en una entrada de validación no equivale a compartir información futura.

### 6.1.2. Ventanas y preprocesamiento causal

Los archivos originales son velas de un minuto; la observación de modelado es **diaria UTC**, con OHLCV de días completos. Se distinguen longitud de entrada $D$, ventana de volatilidad $n$ y horizonte de pronóstico. Las entradas candidatas confirmadas son $D=7,14,21,28$ días; se exploran por separado las cuatro ventanas de volatilidad diaria.

La guía pide justificar los rezagos y la longitud de entrada con **ACF/PACF** y evidencia temporal. El EDA ya contiene esos diagnósticos a frecuencia diaria y complementos de minuto y hora. Sus resultados orientan la discusión y el contraste de las ventanas candidatas, pero no seleccionan una ganadora por sí solos. La elección deberá sustentarse con validación y literatura, sin examinar test para ajustar la ventana.

Diferenciación, normalización o transformaciones se calcularán con el pasado permitido. No se aplicarán automáticamente por el resultado de una sola prueba. Los scalers de implementación se ajustarán de nuevo dentro de cada fold; no se transfieren sin más los parámetros exploratorios de PCA del EDA.

Se conserva la cuadrícula temporal. El día incompleto de desarrollo no se interpola en la implementación actual: los derivados marcan su falta y se excluyen ventanas afectadas. Si se adoptara otra política de imputación, tendría que ser causal, justificada y registrada antes de evaluar.

### 6.1.3. Familias candidatas y selección desde la literatura

La guía enumera como referencias típicas para series de tiempo:

- MLP sobre una ventana de rezagos.
- RNN simple.
- LSTM o GRU de una capa.
- CNN 1D poco profunda.

Son **familias orientativas**, no los tres modelos ya seleccionados. Se elegirán exactamente tres a partir de su uso como baseline en estudios pertinentes, complejidad y diversidad de sesgo inductivo, siguiendo la [sección 3](03_estado_arte.md).

Una MLP puede aplanar las mismas observaciones diarias que una red recurrente o convolucional conserva como secuencia. Esta diferencia de representación no autoriza cambiar las covariables disponibles, las etiquetas o las muestras para favorecer a una arquitectura.

### 6.1.4. Pérdida y estrategia de pronóstico

La guía admite error cuadrático o absoluto, coherente con la métrica de evaluación. El diseño actual propone **MSE como pérdida y RMSE como métrica primaria**, pendiente de cierre del protocolo antes de entrenar. El error absoluto y otras decisiones se contrastarán de forma predefinida si se incorporan, sin seleccionarlos con test.

La salida principal es una volatilidad escalar no negativa de **un día adelante**. Es una estrategia directa a un paso: el modelo utiliza el pasado permitido y emite ese objetivo. El ejemplo sintético 4/3/2 de la figura de `timeseries-cv` ilustra tres salidas, pero no cambia el horizonte principal del proyecto.

Si se amplía a varios horizontes, se declarará antes de la evaluación una estrategia recursiva, directa por horizonte o multi-salida, incluyendo dimensiones, objetivos y disponibilidad. No se combinarán esas estrategias como si fueran el mismo experimento ni se atribuirán siete salidas por analogía con otro estudio.

### 6.1.5. Métricas por horizonte y referencia ingenua

Para cada configuración y horizonte se reportarán **RMSE, MAE y MASE**, como exige la ruta temporal. Se añaden MSE, MAPE y R² según el diseño y los materiales específicos recibidos, con las políticas de ceros y casos no definidos de §4.4.

Se comparará con persistencia y con un ingenuo estacional si se justifica su periodo y se registra antes. Las referencias utilizarán las mismas fechas y escala que las redes y no sustituirán ninguno de los tres modelos de Deep Learning.

La métrica por ejecución se calculará agrupando las predicciones fuera de entrenamiento. En los folds diarios de un solo objetivo, promediar RMSE por fold equivaldría a MAE y R² por fold no estaría definido. Los valores agregados y por periodo se reportarán sin ocultar diferencias de cobertura.

### 6.1.6. Inferencia y diagnóstico de residuos

La ruta de la guía requiere:

| Herramienta | Pregunta y aplicación |
|---|---|
| Diebold–Mariano por horizonte | Diferencia de precisión mediante pérdidas pareadas por fecha, con varianza de largo plazo y corrección de muestra pequeña justificadas |
| Ljung–Box | Autocorrelación de residuos con rezagos y grados de libertad documentados |
| BDS | Diagnóstico de hipótesis i.i.d. de residuos, sin interpretar automáticamente un rechazo como prueba exclusiva de no linealidad |
| Friedman con post-hoc | Comparación conjunta de modelos con bloques y correcciones adecuados, reconociendo sus límites de potencia |

Estos procedimientos se desarrollan en [§5.4](05_benchmark.md). Se aplicarán a predicciones y residuos reales de los modelos, no a estadísticas descriptivas del EDA presentadas como si fueran errores predictivos.

Las volatilidades móviles comparten retornos y pueden producir dependencia incluso con horizonte de un día. No se impondrá independencia diaria por la longitud del horizonte ni se contarán todas las semillas y fechas como observaciones i.i.d. Se reportarán diferencias, IC, efectos, multiplicidad y potencia o MDE.

## 6.2. Correspondencia entre procedimiento y evidencia

| Requisito de la ruta de series de tiempo | Evidencia o ubicación | Estado |
|---|---|---|
| Fuente, frecuencia y reloj de información | Secciones 1.2–1.3 y 2.3 | Documentados |
| Calidad y continuidad | EDA y derivados diarios de 2023–2024 | Ejecutados |
| Partición temporal y origen móvil | Código y manifiestos de `timeseries-cv`, §2.1.4 | Folds ejecutados; test supervisado final sin congelar |
| ACF/PACF y justificación de ventanas | EDA diario, §2.2.3 | Diagnósticos ejecutados; selección final pendiente |
| Preprocesamiento solo con train | Diseño §4.1–§4.3 | Regla documentada; integración con modelos pendiente |
| Tres arquitecturas extraídas de literatura | Sección 3 | Pendiente |
| Pérdida y estrategia | Secciones 1 y 4 | Propuesta de MSE y horizonte principal de un día; cierre experimental pendiente |
| Métricas y comparación ingenua | §4.4 | Protocolo; sin predicciones de modelos |
| Inferencia temporal y Friedman/post-hoc | §5.4 | Protocolo; sin pruebas de benchmark ejecutadas |

## 6.3. Condiciones para continuar con el benchmark

Se deberán completar la revisión bibliográfica y la selección de tres modelos; cerrar las configuraciones, transformaciones y políticas de métricas; fijar $\delta$ y el plan de comparaciones; ejecutar el piloto y dimensionar repeticiones. El test permanecerá fuera de esas decisiones.

El riesgo principal de esta ruta es **data leakage temporal**. Se comprobarán los límites y la disponibilidad de cada entrada, transformación y etiqueta antes de entrenar, y se conservará evidencia de esos controles en cada corrida. La concordancia con esta guía es un requisito del diseño, no un sustituto de la ejecución ni de sus resultados.
