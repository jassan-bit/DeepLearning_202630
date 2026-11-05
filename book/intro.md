# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Avance:** Entregable 1 · CV de Filipe, cuatro historias y evaluación reservada · 4 de noviembre de 2026

Este libro desarrolla el **Entregable 1 de Deep Learning** para el pronóstico de volatilidad de **Bitcoin**, con datos oficiales de **Binance Spot BTCUSDT**. El problema corresponde al **tipo 4: regresión temporal**: cinco covariables OHLCV diarias, objetivo poblacional de siete retornos logarítmicos y horizonte de un día. Las cuatro longitudes **D=7,14,21,28** cambian la historia de entrada, manteniendo **n=7 y h=1**.

La selección usa **`split_train_val_forwardChaining` de `timeseries-cv`**, proyecto de DidierRLopes con coautoría de **Filipe R. Ramos**. Sus orígenes diarios se agrupan en doce bloques mensuales de 2024. Cada ajuste y escalador usa únicamente el pasado permitido, con embargo de D días y validación interna cronológica. Las redes principales usan precios relativos al último cierre observable de su ventana, diez semillas comunes y las mismas 366 fechas.

El capítulo 5 presenta las **tablas completas de CV y de test para cada entrada**, con MAPE, MAE, RMSE, MSE, R² y MASE; media, DE e IC entre semillas; y BDS por bloque mensual de residuos de horizonte h=1. La mejora posterior a la auditoría añade resultados **separados de 2025 y enero–agosto de 2026**, variabilidad cuantitativa, diagnóstico agregado de errores y un **suplemento inferencial post-hoc** de las tres configuraciones D7 seleccionadas. Los promedios de p-valores BDS son descriptivos; los nuevos contrastes no reabren la selección ni se presentan como confirmación prospectiva.

## Qué contiene este avance

- [Problema y dataset](sections/01_problema_dataset.md): formulación, justificación, fuente y descripción técnica.
- [Análisis exploratorio de datos](sections/02_eda.md): calidad, objetivo y estructura temporal; ejemplo oficial y aplicación de la función de Filipe.
- [Estado del arte](sections/03_estado_arte.md): 36 referencias y seis textos completos; respaldo primario de las familias MLP/LSTM/CNN y límites explícitos de su adaptación.
- [Implementación y evaluación](sections/04_implementacion_evaluacion.md): cuatro historias, configuraciones, semillas, purgas, congelación y artefactos reproducibles.
- [Benchmark y baseline](sections/05_benchmark.md): CV 2024, tablas finales del test 2025–agosto de 2026 y diagnósticos por bloque.
- [Procedimiento temporal](sections/06_procedimiento.md): ruta 6.4, selección con Filipe y evaluación final sin reajustes durante test.
- [Limitaciones y entregables](sections/07_limitaciones_entregables.md): dependencia, potencia, equivalencia y estado de los productos.
- [Entregables del proyecto](sections/08_entregables.md): los siete productos desarrollados, sus tablas, declaraciones, configuraciones y evidencias de entrega.
- [Decisiones y pendientes](sections/protocolo_pendiente.md): acuerdos, evidencia verificable y trabajo aún abierto.

## Alcance del proyecto

El trío implementado es **MLP sobre covariables rezagadas, LSTM unidireccional y CNN1D poco profunda**. Las cuatro historias se comparan dentro de un mismo objetivo. La CV principal reúne **120 trayectorias, 1.440 segmentos modelo–mes, 2.880 ajustes efectivos y 43.920 predicciones**. Incluye las treinta trayectorias relativas D7 ya registradas y noventa nuevas D14/21/28; el control absoluto pertenece únicamente al ciclo inicial D7 y se conserva como comparación de preprocesamiento.

Se mantiene **2023 como historia inicial y 2024 como validación forward chaining**. Tras cerrar la configuración con desarrollo se ajustaron **120 modelos finales**, con escaladores y épocas fijados antes de abrir el test. La prueba cubre **01/01/2025–31/08/2026**, con **608 fechas comunes evaluables y 72.960 predicciones**. Sus veinte meses son bloques de reporte de pesos finales fijos: no son veinte reajustes ni folds de selección. Septiembre–octubre de 2026 permanece fuera del experimento.

La elección previa al test y sus hashes están en [la congelación](_static/filipe_multihistory_20261104/freeze.json); el [método y la ejecución](_static/filipe_multihistory_20261104/method.json) y los manifiestos permiten comprobar el orden de las fases. Se reportan las doce combinaciones sin elegir una nueva configuración por sus resultados de test.

Las decisiones de desarrollo usaron 2024 ya expuesto: la CV es exploratoria y no validación anidada independiente. Diez semillas no acreditan potencia 0,80. Los contrastes Holm-12, DM y bootstrap del ciclo inicial D7 conservan su alcance; no se trasladan automáticamente a las cuatro historias o al test. El BDS mensual tiene una referencia asintótica aproximada ante bloques cortos y etiquetas solapadas.

La [matriz de cierre](sections/protocolo_pendiente.md) distingue requisitos todavía abiertos y mejoras ejecutadas a partir de la auditoría del PDF. Miura et al. (2019) aporta respaldo aplicado independiente a MLP y CNN, una vez en seis estudios; su dimensión CNN no está reportada y las recetas son adaptaciones. La revisión posterior no reescribe la selección como a priori. Un {ref}`piloto nuevo registrado antes de sus corridas <piloto-filipe-registrado>` añade cinco semillas por modelo e historia para dimensionar futuras repeticiones condicionadas a 2024; no vuelve prospectivo el benchmark histórico ni acredita potencia alcanzada. El criterio avanzado de superar el 5 % frente a MLP/D7 y persistencia requiere su propio piloto y una evaluación nueva independiente. TOST/ROPE solo será exigible si se afirma equivalencia. Solo se considera Bitcoin; no se inventan requisitos de documentos completos de E2/E3 que no se han recibido.

## Reproducibilidad y privacidad

`BUILD.md` explica la construcción del libro. Los scripts de preparación, entrenamiento, análisis, congelación y evaluación conservan localmente índices, escaladores, pesos, curvas y predicciones. El libro publica tablas agregadas, código y manifiestos de fechas, calidad y hashes. {download}`El alcance de los resúmenes públicos <_static/filipe_multihistory_20261104/README.md>` distingue fuentes, reutilización D7, nuevos ajustes y evaluación final. Los vectores individuales, derivados diarios y paquetes completos permanecen en el espacio local; construir HTML no vuelve a entrenar ni abre archivos brutos.

El EDA y la selección de desarrollo no abrieron los ZIP de test para modelado. El ciclo final los abrió **después de congelar los 120 modelos**, exclusivamente para la evaluación autorizada. El repositorio y el sitio son públicos; los documentos originales completos del curso no se incorporan al HTML.

[Consultar el Jupyter Book publicado](https://jassan-bit.github.io/DeepLearning_202630/).
