# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Avance:** Entregable 1 · Validación forward chaining de Filipe · 4 de noviembre de 2026

Este libro desarrolla el **Entregable 1 de Deep Learning** para el pronóstico de volatilidad de **Bitcoin**, con datos oficiales de **Binance Spot BTCUSDT**. El problema corresponde al **tipo 4: regresión temporal**, con cinco covariables OHLCV diarias, objetivo de volatilidad de siete retornos y horizonte de un día, sin rezagos explícitos de la volatilidad como entrada.

La evaluación utiliza **`split_train_val_forwardChaining` de `timeseries-cv`**, proyecto de DidierRLopes con coautoría de **Filipe R. Ramos**. Sus orígenes diarios se agrupan en **12 bloques mensuales de 2024**. En cada corte se ajustan de nuevo MLP, LSTM y CNN1D con el pasado permitido; las predicciones se emiten diariamente con pesos fijos dentro del mes y embargo de siete días.

La representación principal expresa OHLC respecto al último cierre observable de cada ventana. Se compara con una representación de precios absolutos bajo los mismos folds, configuraciones y diez semillas. En **todo 2024**, los R² medios de la representación relativa son **0,6157 para MLP, 0,5789 para LSTM y 0,5921 para CNN1D**. **Persistencia alcanza 0,7596 y conserva el menor error anual.** Los [resultados del capítulo 5](sections/05_benchmark.md) separan el año completo del resumen de julio–diciembre y reportan la incertidumbre entre semillas.

## Qué contiene este avance

- [Problema y dataset](sections/01_problema_dataset.md): formulación, justificación y descripción técnica, con sus incisos numerados.
- [Análisis exploratorio de datos](sections/02_eda.md): calidad, objetivo, desplazamientos y bloques específicos de EDA; ejemplo oficial y aplicación de la función de Filipe.
- [Estado del arte](sections/03_estado_arte.md): 35 referencias, búsqueda pública y revisión de cinco textos completos, con evidencia y lagunas explícitas.
- [Implementación y evaluación](sections/04_implementacion_evaluacion.md): arquitecturas, preprocesamiento, folds realmente usados, configuraciones, semillas y artefactos reproducibles.
- [Benchmark y baseline](sections/05_benchmark.md): comparación de las tres redes y persistencia, métricas anuales y semestrales, costos, intervalos y contrastes ejecutados.
- [Procedimiento temporal](sections/06_procedimiento.md): ruta 6.4 y entrenamiento creciente con información disponible en cada origen.
- [Limitaciones y entregables](sections/07_limitaciones_entregables.md): dependencia, potencia, equivalencia y estado de los productos exigidos.
- [Decisiones y pendientes](sections/protocolo_pendiente.md): acuerdos, evidencia verificable y trabajo pendiente.
- [Referencias y materiales](sections/referencias.md): bibliografía y fuentes documentales.

## Alcance del proyecto

El trío implementado sigue la ruta 6.4: **MLP sobre covariables rezagadas, LSTM sobre secuencias de covariables y CNN1D poco profunda**. GRU es una alternativa para el puesto recurrente. Las tres redes reciben la misma información OHLCV y usan las mismas fechas y reglas de ajuste. Las dos representaciones de entrada se compararon con **diez semillas por modelo y variante**: **60 trayectorias, 720 segmentos modelo–fold, 1.440 ajustes y 21.960 predicciones**. Cada segmento comprende selección interna de duración y reajuste con todo el entrenamiento permitido del mes.

Se conserva el calendario: **2023 como entrenamiento inicial, 2024 para validación forward chaining, ajuste final con desarrollo tras cerrar las configuraciones y test previsto del 01/01/2025 al 31/08/2026**. Septiembre–octubre de 2026 queda sin asignar al test. Se mantienen la exclusión del día incompleto 24/03/2023, el calendario y las purgas. El ajuste final y la evaluación de test aún no se han ejecutado.

Los hiperparámetros se fijaron con datos de desarrollo ya examinados. Esta CV es exploratoria y no constituye validación anidada de toda la selección. Los intervalos entre semillas condicionan al mismo calendario; diez semillas no acreditan potencia 0,80. Se ejecutaron doce contrastes por semillas con Holm, seis comparaciones temporales DM, quince diagnósticos residuales y bootstrap de fechas. Las diferencias entre las tres redes no quedan distinguidas con Holm-12; el candidato MLP se basa en su media observada. Las pruebas temporales y los IC marginales responden preguntas distintas. La sensibilidad de diseño se calculó retrospectivamente; el piloto previo y el dimensionamiento prospectivo siguen pendientes. El margen operativo del 5 % frente a persistencia no se ha demostrado. La [revisión por ítem](sections/protocolo_pendiente.md) distingue los análisis ejecutados de los pendientes reales. El [inventario de 390 incisos](_static/e1/revision_items_libro_20261104.csv) registra la comprobación de toda la numeración.

La revisión mantiene **35 referencias, cinco textos completos y 81 valores de sus autores**. El respaldo aplicado de MLP no lineal independiente y CNN1D simple sigue incompleto en ese corpus. Las tres fichas técnicas, con 27 campos numerados, corresponden a MLP, LSTM y CNN1D ejecutados; los perfiles de otras arquitecturas se conservan como antecedentes documentales en el repositorio. Ejecutar modelos no completa por sí solo la guía ni acredita generalización fuera de 2024.

Solo se considera Bitcoin. Los documentos completos de los entregables 2 y 3 no se han recibido; no se inventan sus requisitos.

## Reproducibilidad y privacidad

Las instrucciones de construcción y reproducción están en `BUILD.md` en el repositorio. El libro usa Jupyter Book 1 con dependencias fijadas y se construye sin ejecutar notebooks. Los índices, las predicciones y los artefactos de la CV permiten comprobar sus resultados; el ajuste utiliza un entorno PyTorch separado del entorno de `timeseries-cv`.

El ciclo de entrenamiento y análisis no abrió archivos de test. La adquisición previa verificó timestamps, esquema y conteos de los brutos, sin utilizarlos para seleccionar modelos. El repositorio fuente y el sitio son públicos; los materiales originales del curso no se incorporan al HTML.

[Consultar el Jupyter Book publicado](https://jassan-bit.github.io/DeepLearning_202630/).
