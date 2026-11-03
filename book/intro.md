# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Avance:** Entregable 1 · Secciones 1–7 · Revisión del 3 de noviembre de 2026

Este libro desarrolla el **Entregable 1 de Deep Learning** para el pronóstico de volatilidad de **Bitcoin**, utilizando datos obtenidos directamente de **Binance Spot (BTCUSDT)**. El problema corresponde al **tipo 4: regresión temporal**, con covariables OHLCV diarias y sin rezagos explícitos de la volatilidad objetivo; el precio de cierre permite calcular los retornos y las etiquetas. La revisión se contrasta con la guía del Entregable 1. El alcance autorizado aquí comprende documentación, EDA y preparación de particiones; el usuario reservó la ejecución de modelos para E2. Esa instrucción de trabajo se distingue de la guía escrita, que sí incorpora ejecución en sus secciones 4–5.

## Qué contiene este avance

- [Problema y dataset](sections/01_problema_dataset.md): formulación matemática, hipótesis propuesta, métricas y ficha técnica.
- [Análisis exploratorio de datos](sections/02_eda.md): adquisición directa desde Binance Spot BTCUSDT y análisis de calidad, precios y retornos de desarrollo.
- [Estado del arte](sections/03_estado_arte.md): tabla de las 33 referencias citadas, búsqueda y cribado trazables, matriz, frecuencias, tres fichas y comparación crítica; candidatos de trabajo y extracción definitiva pendiente.
- [Diseño de la etapa E2](sections/04_implementacion_evaluacion.md): criterios de implementación, validación y piloto previstos, sin ejecución propia en esta revisión.
- [Criterios del benchmark para E2](sections/05_benchmark.md): métricas, tablas futuras, inferencia y reglas de interpretación.
- [Procedimiento temporal](sections/06_procedimiento.md): ruta 6.4 de regresión temporal y disponibilidad causal de covariables.
- [Limitaciones y entregables](sections/07_limitaciones_entregables.md): potencia, límites inferenciales, rúbrica y estado verificable de los productos exigidos.
- [Decisiones, revisión y pendientes](sections/protocolo_pendiente.md): matriz de requisitos, correcciones ejecutadas, evidencia parcial y proyección a E2.
- [Referencias y materiales](sections/referencias.md): catálogo bibliográfico inicial y fuentes adicionales de las fichas.

## Alcance del proyecto

El PDF solicita EDA, literatura, tres referencias DL e implementación experimental; sus requisitos empíricos permanecen identificados. Por instrucción del usuario, esta intervención corrige y documenta los avances y reserva entrenamiento, tuning, piloto y benchmark propios para E2. No se atribuye al docente una modificación que no se ha aportado ni se declara cumplimiento total del PDF original.

La elección documental de familias prioriza diversidad entre memoria, convolución y atención. No equivale al ranking de frecuencia pura: GRU es más frecuente que TCN y Transformer en el pequeño corpus. La guía exige además recurrencia y mínima complejidad; la extracción definitiva de tres modelos básicos sigue parcial. Las configuraciones concretas y la suficiencia académica de la fuente gris de atención deben revisarse antes de implementar. Las treinta referencias iniciales se conservan y se añaden tres fuentes para las fichas, dando 33 entradas citadas.

La revisión corrigió los índices de validación con un embargo explícito de D días, conservó la exclusión del 24/03/2023 y documentó una reserva bruta de test mediante metadatos. Estos productos preparan la evaluación posterior; no son resultados de modelos. El [estado por requisito](sections/protocolo_pendiente.md) identifica lo documentado, lo parcial y lo pendiente.

Solo se considera Bitcoin. Otras criptomonedas son una posible ampliación futura. Los documentos completos de los entregables 2 y 3 aún no se han recibido; no se inventan sus requisitos.

## Reproducibilidad y privacidad

Las instrucciones están en `BUILD.md` en la raíz del repositorio. El libro usa Jupyter Book 1 con dependencias fijadas; se genera HTML localmente sin ejecutar notebooks. El repositorio fuente es público. El sitio publica únicamente el HTML generado en `book/_build/html`, desde la rama `gh-pages`. Los documentos originales del curso no se incorporan al HTML, aunque están disponibles en el repositorio fuente.

[Consultar el Jupyter Book publicado](https://jassan-bit.github.io/DeepLearning_202630/).
