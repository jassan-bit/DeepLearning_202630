# 1 Introducción

Este tercer entregable da continuidad al proyecto de pronóstico de volatilidad de Bitcoin. La guía exige reunir **correcciones de E2, un producto predictivo, una contribución neuronal defendible y un informe científico reproducible**, sus cuatro componentes obligatorios. El avance actual incluye una CNN-LSTM con atención entrenada en diez semillas y su evaluación estadística retrospectiva; todavía no equivale al cierre de los cuatro productos.

El problema es BTCUSDT de Binance Spot. Al inicio del día $t$ se utilizan las siete barras OHLCV anteriores para estimar la desviación poblacional de siete retornos logarítmicos, sin anualizar y con horizonte de un día:

$$
r_t=\log(C_t/C_{t-1}),\qquad
\bar r_t=\frac17\sum_{j=0}^{6}r_{t-j},\qquad
\sigma_7(t)=\sqrt{\frac17\sum_{j=0}^{6}(r_{t-j}-\bar r_t)^2}.
$$

La etiqueta requiere ocho cierres y contiene el retorno del día objetivo, aún desconocido al emitir el pronóstico. Las entradas son las siete barras de $[t-7,t)$, ordenadas de antigua a reciente, con cinco canales. El [contrato de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html) define `relative_window`, escaladores de entrenamiento, disponibilidad y máscaras. Se conserva la ruta metodológica temporal **F2a**; no se importan pruebas de clasificación o de conjuntos independientes sin justificar su aplicación.

## 1.1 Antecedentes y motivación

E1 estableció las referencias neuronales. El corte histórico de E2 comparó diez candidatos, tres referencias neuronales y tres comparadores clásicos: **16 sistemas sobre 608 fechas conocidas**, con seis semillas por red. En ese corte, LSTM-autoencoder obtuvo la menor RMSE media, **0,00380343**. Sus veinte contrastes primarios y su conclusión histórica conservan identidad; no se sustituyen por el análisis posterior de E3.

La extensión posterior conservó los **81 predictores originales** y añadió cuatro semillas por cada una de las trece redes: ahora hay **133 predictores E2**, diez por red y una trayectoria por clásico. No se repitió HPO. Con diez semillas, el menor RMSE medio de E2 sigue siendo el de LSTM-autoencoder, **0,00387289**. El [capítulo 4](04_correcciones_e2.md) distingue esta ampliación de una corrección de datos y conserva el ranking antes/después.

La receta E3 evaluada es **CNN causal → LSTM unidireccional → atención temporal por contenido, con salida aditiva positiva y 1210 parámetros**. Sus diez semillas obtuvieron RMSE medio **0,0029394324**, una reducción observada del **24,10 %** frente al mejor rival por RMSE de la extensión E2. El [capítulo 8](08_verificacion_estadistica.md) conserva los 48 contrastes posteriores favorables tras Holm, sus supuestos, intervalos y sensibilidades. El MCS auxiliar conserva diez sistemas, incluido E3, y no establece un ganador único.

La combinación CNN-LSTM-atención y una mejor métrica **no acreditan por sí solas originalidad**. Los pilotos de penalización/saturación, cabezas de salida y momentos se conservan como historia de desarrollo en los [capítulos 6](06_modelo_original.md) y [7](07_diseno_riguroso.md). El piloto por momentos ya completó sus tres recetas y no superó su heurística; no sigue pendiente ni se renombra como el modelo de la comparación actual.

## 1.2 Cuatro componentes obligatorios y objetivos

| Componente de la guía | Producto que debe entregarse | Estado comprobable |
|---|---|---|
| Correcciones de E2 | Observaciones aplicables resueltas, tabla de trazabilidad, auditoría e impacto consistente en resultados | Auditorías y correcciones internas documentadas en [4](04_correcciones_e2.md); cierre integral del Bloque A pendiente. No se inventan observaciones del docente. |
| Dashboard predictivo | Aplicación **Dash Plotly**, desplegada en **Render**, con **Contexto, EDA y Modelos**; Top 10 con su configuración seleccionada y artefactos ligeros, o resultados precalculados con justificación de recursos | Diseño en [9](09_dashboard_predictivo.md) y [10](10_despliegue_render.md); tres referencias exportadas con alcance limitado. Aplicación integrada, perfil final de inferencia, URL y prueba de humo pendientes. |
| Modelo neuronal original o transferido | Mecanismo, capa, pérdida o entrenamiento con diferencia sustantiva, fuentes atribuidas, implementación algorítmica, propiedad formal pertinente y comparación justa, reproducible y con protocolo previo | Receta CNN-LSTM evaluada y desarrollo documentado en [6](06_modelo_original.md), [7](07_diseno_riguroso.md) y [8](08_verificacion_estadistica.md). La novedad, la defensa humana y los requisitos completos de R3 siguen sin acreditarse. |
| Informe científico | Jupyter Book que documente implementación, método, experimentos, resultados, discusión y paquete de reproducción | Este libro y sus fuentes están publicados; [11](11_informe_tecnico.md) y [12](12_lista_entregables.md) distinguen lo disponible de contenedor, reproducción independiente y cierre pendientes. |

Los objetivos operativos son resolver las correcciones aplicables, conservar la selección por validación y los hashes, justificar la contribución frente a los antecedentes, evaluarla bajo la ruta F2a y llevar los resultados a un servicio reproducible. Un buen resultado estadístico no cierra automáticamente los otros componentes.

## 1.3 Avances y límites actuales

El inventario inicial de **81 predictores y 167 metadatos** corresponde a su corte histórico, no al tamaño actual de la extensión E2. Se conserva porque documenta escaladores, calibradores y artefactos ya verificados. La comparación posterior incorpora **133 predictores E2 y diez de E3**, con fechas y objetivos idénticos y sin borrar los resultados anteriores.

El proceso E3 utilizó **532 ejemplos de ajuste y 116 de validación**, mientras que E2 utilizó **655 de ajuste y 53 de calibración**. Compartir las 608 fechas de prueba y diez semillas por red no iguala esos tratamientos. Las fechas de prueba ya eran conocidas y no se presentan como una confirmación ciega. Los análisis retrospectivos, la variación entre semillas, las propiedades formales históricas y las pruebas de software conservan alcances diferentes.

El [capítulo 8](08_verificacion_estadistica.md) ya incluye métricas, efectos, p-valores, Holm, sensibilidades y MCS. En cambio, sMAPE y R² aún deben incorporarse a la tabla resumida vigente; la cobertura y anchura de intervalos del candidato requieren una etapa de calibración separada. Tampoco están cerrados la evaluación en tres conjuntos adicionales, el dimensionamiento de potencia, la adecuación completa y los demás estudios de robustez.

**R1, la defensa académica de R2 y R3 siguen abiertos.** La demostración del primer módulo de atención modificada no se transfiere automáticamente a la receta aditiva actual. La implementación, el resultado favorable y la publicación del libro no acreditan por sí solos novedad, dominio humano del código o entrega completa.

## 1.4 Cómo leer el informe

El cuerpo principal sigue exclusivamente los **capítulos 1 a 16 de la guía**. La [sección 2](02_estandar_profundidad.md) mantiene el estándar único de profundidad; la [3](03_panorama_general.md) organiza los bloques A–G y sus dependencias. Las [secciones 4](04_correcciones_e2.md), [5](05_configuraciones_exportacion.md) y [6](06_modelo_original.md) reúnen correcciones, artefactos y contribución. La [7](07_diseno_riguroso.md) desarrolla el diseño; la [8](08_verificacion_estadistica.md), la evaluación y sus límites. Las [secciones 14](14_evaluacion_calificacion.md) y [15](15_integridad_academica.md) conservan las condiciones de evaluación e integridad, sin asignar una nota al avance.

Los [anexos y la concordancia](concordancia_secciones.md) conservan las plantillas, el detalle de apoyo y las direcciones publicadas. Cuando la guía precisa o restringe un requisito en una sección posterior, se mantiene esa condición; el libro no rebaja el estándar después de observar resultados.

El dashboard de Render aún no tiene URL pública verificada. Este Jupyter Book no se presenta como ese dashboard. La asistencia de IA en código, teoría, revisión focal y documentación está declarada; la persona autora deberá comprender y defender la entrega.
