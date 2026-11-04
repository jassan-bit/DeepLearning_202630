# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Avance:** Entregable 1 · Secciones 1–7 · Primera ejecución de desarrollo: 4 de noviembre de 2026

Este libro desarrolla el **Entregable 1 de Deep Learning** para el pronóstico de volatilidad de **Bitcoin**, utilizando datos obtenidos directamente de **Binance Spot (BTCUSDT)**. El problema corresponde al **tipo 4: regresión temporal**, con covariables OHLCV diarias y sin rezagos explícitos de la volatilidad objetivo; el precio de cierre permite calcular los retornos y las etiquetas. La revisión se contrasta con la guía del Entregable 1. El 4 de noviembre el usuario autorizó correr los tres modelos y actualizar este libro, ampliando el alcance anterior de documentación y EDA. Ya se ejecutaron **MLP, LSTM y CNN1D poco profunda**, con búsqueda, piloto y comparación de desarrollo. Este ciclo no abrió archivos de test de 2025–2026 ni evaluó sus objetivos. La adquisición previa sí verificó timestamps, esquema y conteos de los brutos; esos controles no se utilizaron para seleccionar modelos.

En **184 fechas de julio–diciembre de 2024**, con **30 semillas nuevas por red** y una ventana común de **7 días**, LSTM obtuvo el menor RMSE medio entre las redes: **0,008903**. MLP obtuvo **0,015990** y CNN1D **0,017587**. **Persistencia obtuvo 0,004822 y superó a las tres.** Son resultados de desarrollo y calibración ya expuestos al piloto, sin confirmación en test. La búsqueda previa, sobre enero–junio, había seleccionado MLP como referencia operacional; esa identidad se conserva y se distingue del líder descriptivo posterior.

## Qué contiene este avance

- [Problema y dataset](sections/01_problema_dataset.md): definición del problema, justificación del dataset y descripción de la técnica, con sus ítems numerados.
- [Análisis exploratorio de datos](sections/02_eda.md): adquisición Binance Spot BTCUSDT, perfiles y colas del objetivo, calidad, reconstrucción de etiquetas y desplazamiento marginal; diez bloques específicos con todos sus incisos y aplicabilidad justificada, incluido el bloque C con los residuos de cuatro referencias lineales.
- [Estado del arte](sections/03_estado_arte.md): tabla de las 35 referencias citadas, búsqueda pública ejecutada y verificación posterior de procedencia e indexación, cribado trazable, matriz, frecuencias y tres fichas; extracción definitiva pendiente.
- [Implementación y evaluación](sections/04_implementacion_evaluacion.md): arquitecturas implementadas, búsqueda equitativa, piloto y 201 corridas terminadas; configuraciones, semillas y registros reproducibles.
- [Benchmark de los Modelos Base y Definición del Baseline](sections/05_benchmark.md): métricas, costos, diferencias, intervalos, diagnósticos y gráficos de desarrollo; límites del dimensionamiento y evaluación final pendiente.
- [Procedimiento temporal](sections/06_procedimiento.md): aplicabilidad de las once rutas, desarrollo de la 6.4 y secuencia de cierre coherente con los protocolos actuales.
- [Limitaciones y entregables](sections/07_limitaciones_entregables.md): límites exactos de Wilcoxon/Holm, potencia, equivalencia y estado de los siete productos; rúbrica sin autocalificación.
- [Decisiones, revisión y pendientes](sections/protocolo_pendiente.md): matriz de requisitos, correcciones ejecutadas, evidencia parcial y proyección a E2.
- [Referencias y materiales](sections/referencias.md): catálogo bibliográfico inicial y fuentes adicionales de las fichas.

## Alcance del proyecto

El PDF solicita EDA, literatura, tres referencias DL e implementación experimental. La primera ejecución comprende **96 corridas de búsqueda**, **15 de piloto** y **90 de comparación**, cada una con selección interna de épocas y un reajuste nuevo sobre 2023: **402 ajustes efectivos**. Los cuatro Ridge con $\alpha=1$ fijo del bloque C se conservan como diagnóstico de EDA separado. La implementación propia no cierra las lagunas bibliográficas ni acredita cumplimiento total de la guía.

El trío implementado sigue la ruta 6.4: **MLP sobre covariables rezagadas, LSTM sobre secuencias de covariables y CNN 1D poco profunda**. GRU es una alternativa para el puesto recurrente. Las tres familias reciben la misma información OHLCV, partición, objetivo y presupuesto de búsqueda. El [protocolo inicial](_static/dl_benchmark_20261104/protocol_initial.json) se registró antes del primer ajuste y la [selección de desarrollo](_static/dl_benchmark_20261104/selected_configuration.json) precedió al piloto y a las nuevas semillas. El EDA ya existía: este registro no antecede a toda observación histórica de los datos.

Las fichas anteriores de LSTM, TCN y encoder Transformer se conservan como perfiles de fuentes recuperadas. Los cinco textos incluidos no acreditan una MLP no lineal independiente ni una CNN1D simple independiente; hace falta completar ese respaldo aplicado. La TCN ganadora de S1 no equivale a la CNN poco profunda implementada y el encoder no ocupa uno de los tres puestos. Se mantienen las 35 referencias y los conteos documentales, sin presentarlos como prueba de frecuencia global.

Se conservan el embargo de D días, la exclusión estricta del 24/03/2023 y el calendario. Las redes se ajustaron exclusivamente sobre muestras admisibles de **2023**, sin reajustes diarios durante 2024. El piloto estimó hasta **1.212 repeticiones** para detectar el efecto pequeño planificado; se ejecutó el límite registrado de **30**, por lo que no se afirma potencia 0,80 para esa pregunta. Los intervalos entre semillas condicionan al calendario; la incertidumbre temporal se estudia aparte con bloques. El [estado por requisito](sections/protocolo_pendiente.md) conserva estas limitaciones y los pendientes de test, literatura y márgenes para modelos avanzados.

Solo se considera Bitcoin. Otras criptomonedas son una posible ampliación futura. Los documentos completos de los entregables 2 y 3 aún no se han recibido; no se inventan sus requisitos.

## Reproducibilidad y privacidad

Las instrucciones están en `BUILD.md` en la raíz del repositorio. El libro usa Jupyter Book 1 con dependencias fijadas; se genera HTML localmente sin ejecutar notebooks. El repositorio fuente es público. El sitio publica únicamente el HTML generado en `book/_build/html`, desde la rama `gh-pages`. Los documentos originales del curso no se incorporan al HTML, aunque están disponibles en el repositorio fuente.

[Consultar el Jupyter Book publicado](https://jassan-bit.github.io/DeepLearning_202630/).
