# Bitcoin: pronóstico de volatilidad

**Proyecto de Deep Learning · 2026-30**  
**Profesor:** Dr. Lihki Rubio  
**Autor:** Jassan Arteta  
**Avance:** Entregable 1 · Secciones 1–7 · Validación forward chaining ejecutada: 4 de noviembre de 2026

Este libro desarrolla el **Entregable 1 de Deep Learning** para el pronóstico de volatilidad de **Bitcoin**, utilizando datos obtenidos directamente de **Binance Spot (BTCUSDT)**. El problema corresponde al **tipo 4: regresión temporal**, con covariables OHLCV diarias y sin rezagos explícitos de la volatilidad objetivo; el precio de cierre permite calcular los retornos y las etiquetas. La revisión se contrasta con la guía del Entregable 1. El 4 de noviembre el usuario autorizó correr los tres modelos y actualizar este libro, ampliando el alcance anterior de documentación y EDA. Ya se ejecutaron **MLP, LSTM y CNN1D poco profunda**, con búsqueda, piloto y comparación de desarrollo. Este ciclo no abrió archivos de test de 2025–2026 ni evaluó sus objetivos. La adquisición previa sí verificó timestamps, esquema y conteos de los brutos; esos controles no se utilizaron para seleccionar modelos.

La revisión de los R² negativos llevó a ejecutar **validación forward chaining con `timeseries-cv`**, proyecto de DidierRLopes con coautoría de Filipe R. Ramos. Sus índices alimentan **12 bloques mensuales de 2024**: cada modelo se vuelve a ajustar con el pasado permitido y predice diariamente con horizonte de un día. Se compararon el escalado original y precios relativos al último cierre de cada ventana, usando diez semillas comunes y las mismas arquitecturas.

En las **mismas 184 fechas de julio–diciembre**, los R² medios con precios relativos son **0,6578 en MLP, 0,6093 en LSTM y 0,6898 en CNN1D**, frente a −2,7400, −0,1281 y −3,7586 en el primer ciclo estático. Esta comparación con el ciclo anterior es descriptiva: cambian la política de ajuste y las semillas (30 frente a 10). Dentro de la nueva CV sí se comparan ambos preprocesamientos con las mismas diez semillas. Los resultados originales se conservan.

**Persistencia obtiene R²=0,6700 en ese semestre y sigue siendo la mejor referencia en todo 2024**, con R²=0,7596. CNN1D reduce el RMSE semestral aproximadamente un 3,10 % frente a persistencia; no alcanza el margen operativo del 5 %. Son datos de desarrollo ya expuestos, sin confirmación en el test reservado. El [capítulo 5](sections/05_benchmark.md) separa año completo, periodo común e incertidumbre.

## Qué contiene este avance

- [Problema y dataset](sections/01_problema_dataset.md): definición del problema, justificación del dataset y descripción de la técnica, con sus ítems numerados.
- [Análisis exploratorio de datos](sections/02_eda.md): adquisición Binance Spot BTCUSDT, perfiles y colas del objetivo, calidad, reconstrucción de etiquetas y desplazamiento marginal; diez bloques específicos con todos sus incisos y aplicabilidad justificada, incluido el bloque C con los residuos de cuatro referencias lineales.
- [Estado del arte](sections/03_estado_arte.md): tabla de las 35 referencias citadas, búsqueda pública ejecutada y verificación posterior de procedencia e indexación, cribado trazable, matriz, frecuencias y tres fichas; extracción definitiva pendiente.
- [Implementación y evaluación](sections/04_implementacion_evaluacion.md): arquitecturas implementadas, búsqueda y piloto iniciales; nuevo ciclo con 720 segmentos modelo–fold y 1.440 ajustes, con índices y registros reproducibles.
- [Benchmark de los Modelos Base y Definición del Baseline](sections/05_benchmark.md): métricas y gráficos de ambos ciclos; R²/RMSE de CV anual y comparación del mismo semestre, con pruebas y límites separados.
- [Procedimiento temporal](sections/06_procedimiento.md): aplicabilidad de las once rutas, desarrollo de la 6.4 y secuencia de cierre coherente con los protocolos actuales.
- [Limitaciones y entregables](sections/07_limitaciones_entregables.md): límites exactos de Wilcoxon/Holm, potencia, equivalencia y estado de los siete productos; rúbrica sin autocalificación.
- [Decisiones, revisión y pendientes](sections/protocolo_pendiente.md): matriz de requisitos, correcciones ejecutadas, evidencia parcial y proyección a E2.
- [Referencias y materiales](sections/referencias.md): catálogo bibliográfico inicial y fuentes adicionales de las fichas.

## Alcance del proyecto

El PDF solicita EDA, literatura, tres referencias DL e implementación experimental. La primera ejecución estática comprendió **96 corridas de búsqueda**, **15 de piloto** y **90 de comparación**, cada una con selección interna de épocas y un reajuste nuevo sobre 2023: **402 ajustes efectivos**. La nueva CV comprende **60 trayectorias**, dos preprocesamientos, tres modelos, diez semillas y doce cortes: **720 segmentos modelo–fold y 1.440 ajustes efectivos**, con 21.960 predicciones. Los cuatro Ridge con $\alpha=1$ fijo del bloque C se conservan como diagnóstico de EDA separado. La implementación propia no cierra las lagunas bibliográficas ni acredita cumplimiento total de la guía.

El trío implementado sigue la ruta 6.4: **MLP sobre covariables rezagadas, LSTM sobre secuencias de covariables y CNN 1D poco profunda**. GRU es una alternativa para el puesto recurrente. Las tres familias reciben la misma información OHLCV, partición, objetivo y presupuesto de búsqueda. El [protocolo inicial](_static/dl_benchmark_20261104/protocol_initial.json) se registró antes del primer ajuste y la [selección de desarrollo](_static/dl_benchmark_20261104/selected_configuration.json) precedió al piloto y a las nuevas semillas. El [protocolo de CV](_static/dl_r2_cv_20261104/protocol.json) y sus [folds reales](_static/dl_r2_cv_20261104/folds.json) registran el segundo ciclo. Ambos registros son posteriores al EDA; el nuevo diseño también es posterior a observar los resultados negativos del primer ciclo.

Las fichas anteriores de LSTM, TCN y encoder Transformer se conservan como perfiles de fuentes recuperadas. Los cinco textos incluidos no acreditan una MLP no lineal independiente ni una CNN1D simple independiente; hace falta completar ese respaldo aplicado. La TCN ganadora de S1 no equivale a la CNN poco profunda implementada y el encoder no ocupa uno de los tres puestos. Se mantienen las 35 referencias y los conteos documentales, sin presentarlos como prueba de frecuencia global.

Se conservan el embargo de D días, la exclusión estricta del 24/03/2023 y el calendario: **2023 inicial, 2024 forward chaining, ajuste final con desarrollo una vez cerradas las configuraciones y test previsto del 01/01/2025 al 31/08/2026**. Septiembre–octubre de 2026 sigue fuera del test. El primer ciclo mantuvo pesos fijos de 2023; el segundo incorpora pasado permitido de 2024 en cada corte mensual. No se ejecutaron aún el ajuste final de desarrollo ni la evaluación de test.

El piloto del primer ciclo estimó hasta **1.212 repeticiones** y se aplicó un máximo de **30**; no se afirmó potencia 0,80. Las **diez semillas de la nueva CV** son un presupuesto exploratorio y no heredan ese dimensionamiento. Los IC por semillas condicionan al calendario; las pruebas temporales y diagnósticos del primer ciclo no se atribuyen a las nuevas predicciones. El [estado por requisito](sections/protocolo_pendiente.md) conserva estos límites y los pendientes de literatura, test y márgenes para modelos avanzados.

Solo se considera Bitcoin. Otras criptomonedas son una posible ampliación futura. Los documentos completos de los entregables 2 y 3 aún no se han recibido; no se inventan sus requisitos.

## Reproducibilidad y privacidad

Las instrucciones están en `BUILD.md` en la raíz del repositorio. El libro usa Jupyter Book 1 con dependencias fijadas; se genera HTML localmente sin ejecutar notebooks. El repositorio fuente es público. El sitio publica únicamente el HTML generado en `book/_build/html`, desde la rama `gh-pages`. Los documentos originales del curso no se incorporan al HTML, aunque están disponibles en el repositorio fuente.

[Consultar el Jupyter Book publicado](https://jassan-bit.github.io/DeepLearning_202630/).
