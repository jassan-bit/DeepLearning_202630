# Borrador: controles temporales F2a para E2

Propuesta documental basada en el contrato E2 publicado en el commit 314ee0f. No es preregistro, implementación ni resultado. Se consultaron §§17–20, código de preprocesamiento y fuentes técnicas públicas. No se abrieron series de evaluación 2025–2026 ni se ajustaron modelos.

## 1. Decisión y contrato preservado

Añadir **naïve estacional y ARIMA no estacional**, separados de los once candidatos neuronales. Con las tres bases E1, Ridge y persistencia se mantienen al menos 18 sistemas; las ablaciones no integran ese conteo. La obligación ARIMA **o** ETS procede de la guía p30 según el cotejo de §20; no se afirma haber releído aquí el PDF original.

D7×5 relativo, columnas OHLCV, escaladores neuronales train-only, σ₇ poblacional sin anualización y h=1 no cambian. Los controles usan historia univariada propia de y_t=σ₇(t); no añaden canales a las redes ni pretenden igualdad de información histórica. La comparación será entre sistemas con sus representaciones declaradas, no un experimento que aísle exclusivamente arquitectura.

Recomiendo ARIMA por disponer de una implementación en espacio de estados con distinción explícita entre estimación de parámetros y filtrado de nuevas observaciones. Permite un control lineal con/sin integración sin presuponer estacionalidad. Es una decisión de diseño, no evidencia de que ARIMA supere a ETS o a las redes. ETS sería alternativa legítima, pero no se añade como otra búsqueda oculta. La API oficial especifica órdenes, componentes estacionales y tendencia; aquí se restringe deliberadamente la familia [S1]. La documentación de actualizaciones distingue filtrado de refit [S2–S3], y la de espacio de estados muestra tratamiento de observaciones faltantes [S4].

## 2. Reloj, soportes y separación de permisos

Sean o el origen del bloque y t el día que se pronostica al comienzo UTC. Hay tres conjuntos distintos:

1. **Etiquetas para estimar parámetros:** fechas s≤o−8; para selección interna con origen v=o−90, s≤v−8=o−98. Esto incluye escalas, inicializaciones estimadas y cualquier criterio de ajuste.
2. **Observaciones para condicionar estado:** y_s ya realizada, s≤t−1. Los siete días del embargo y los días previos del bloque pueden alimentar el filtro una vez observados, con parámetros y transformación congelados. No contribuyen a una nueva optimización ni a reestimar estado inicial retrospectivamente.
3. **Etiquetas para puntuar:** y_t sólo después de emitir y guardar el pronóstico. No se usan antes para estado, selección ni corrección de esa predicción.

La regla train≤o−8 restringe **estimación**, no prohíbe observar pasado disponible. Si se quisiera prohibir también la actualización de estado, sería otro experimento de pronóstico multipaso desde un estado congelado, no esta evaluación rodante h=1.

Soportes causales mínimos:

- Etiqueta y_t: ocho cierres consecutivos válidos C[t−7:t], incluidos extremos.
- Red: barras completas [t−7,t−1], además de soporte válido de y_t para puntuar; conserva máscara original.
- Persistencia: y[t−1], es decir C[t−8:t−1]. No sustituir por último y válido de fecha distante.
- Naïve estacional de periodo m: y[t−m], es decir C[t−m−7:t−m]. Con m=7 son t−14…t−7; m=14 requiere t−21…t−14; m=28 requiere t−35…t−28. No necesita rellenar todos los días intermedios para emitir este valor concreto.
- ARIMA: secuencia diaria de y desde el inicio permitido de desarrollo hasta el corte de estimación; después, filtrado diario hasta t−1. Requiere historia mayor que D7 y no se le atribuye el mismo número de muestras de entrenamiento que a una red.

Se conserva la bandera de calidad diaria de E1: un cierre procedente de un día incompleto no se admite por el mero hecho de ser finito. No comprimir el calendario ni imputar el 24/03/2023. Construir máscara de y a partir de sus ocho cierres y calidad, no copiar ciegamente elegibilidad D7. Se admiten y=0; se rechazan negativos, infinitos y datos no observados.

En ARIMA, los y inválidos interiores se representan como faltantes sobre frecuencia diaria; se propaga el estado sin actualización de medición. Eso no crea una etiqueta imputada ni habilita puntuar un objetivo inválido. Un adaptador debe probar este comportamiento en datos sintéticos antes de autorización real; no usar dropna sobre el eje temporal. Se propone exigir al menos 90 y válidos para estimación, umbral de ingeniería declarado, no regla teórica de suficiencia ni garantía de identificación.

### Máscara común

Preconstruir la intersección de elegibilidad neuronal, etiqueta verdadera, persistencia y soportes de los tres periodos estacionales, antes de comparar pérdidas. En selección se usa **una misma máscara para todos los periodos y órdenes**, no una máscara ganadora dependiente de errores. ARIMA debe entregar predicción finita para cada fecha planificada; un fallo de convergencia no autoriza borrar esa fecha del benchmark.

Publicar fechas excluidas, motivos, hashes, tamaños por fold y diferencia respecto del soporte histórico. Las 305 fechas CV, 53 del piloto propuesto y 608 históricas son referencias, no coberturas verificadas para los controles nuevos. No reducir entrenamiento neuronal sólo porque un control no tenga soporte de evaluación. En evaluación futura, congelar la regla de máscara por disponibilidad, no seleccionar fechas por resultados; la verificación efectiva del periodo 2025–2026 sigue fuera de este encargo.

## 3. Naïve estacional

Definición h=1: ŷ_t=y[t−m]. Es el último valor del mismo periodo bajo la convención periódica elegida, conforme a la definición de los autores de Forecasting: Principles and Practice [S5]. No hay parámetros continuos ni ajuste numérico.

Proponer M={7,14,28} días UTC como hipótesis de repetición semanal, quincenal y cada cuatro semanas; **28 no significa mes calendario**, y ninguno se declara temporada demostrada. El tamaño 7 del objetivo móvil no demuestra un ciclo semanal: el solapamiento de etiquetas produce dependencia por construcción.

Seleccionar m por RMSE concatenada de los diez folds de desarrollo con máscara común. Mostrar también los tres resultados y persistencia m=1; no eliminar el naïve estacional si pierde contra persistencia. Congelar m antes del piloto, sin volver a escogerlo según piloto o test. Empate exacto: menor m, regla determinista previamente declarada. Un eventual análisis de estacionalidad de desarrollo será descriptivo, no un permiso para ampliar M a posteriori sin registrar una nueva búsqueda y su coste.

No sustituir y[t−m] inválido por interpolación, por y[t−1] o por otro ciclo. La receta no tiene actualización aprendida: cambia su valor de entrada cuando el pasado se observa. No repetirla por semilla.

## 4. ARIMA: espacio acotado y receta propuesta

Candidatos: (p,d,q), p,q∈{0,1,2}, d∈{0,1}: **18 especificaciones**. En d=0, tendencia constante; en d=1, sin tendencia ni drift. Componente estacional (0,0,0,0), sin exógenas, sin búsqueda automática fuera del espacio. ARIMA(0,1,0) sin drift reproduce la lógica de paseo aleatorio y constituye un control de consistencia con persistencia cuando no hay faltantes y el estado está condicionado correctamente; registrar la redundancia, no venderla como evidencia independiente.

Escala propuesta para estabilidad numérica: z=y/a, a=media positiva de y válidos en el entrenamiento propio del control, sin centrar ni logaritmos. a queda congelada por ajuste; no tomar el escalador neuronal salvo igualdad demostrada de soportes. Si a no es positiva, el candidato falla explícitamente. No estimar Box–Cox ni otro enlace. Ajustar por máxima verosimilitud en espacio de estados, con restricciones de estacionariedad/invertibilidad activadas. La integración d=1 se modela dentro del estado; no eliminar días ni diferenciar manualmente un vector comprimido.

Propuesta técnica inicial para implementar y probar: statsmodels.tsa.arima.model.ARIMA, fit(method='statespace'), optimizador L-BFGS, maxiter=200, una inicialización determinista propia de la versión fijada, sin reinicios aleatorios. Fijar versión exacta y opciones tolerancias/inicialización en el manifiesto antes de ejecutar; la documentación estable consultada no acredita una versión local fijada ni compatibilidad ya probada. Estos límites son propuestas, no resultados de convergencia ni costes medidos.

Emitir pronóstico condicional a un paso, invertir a y y aplicar max(0,ŷ_raw), decisión ex ante coherente con objetivo no negativo. Guardar raw y truncado y tasa de truncamiento; la primaria usa truncado. Esto es ARIMA más una regla explícita de salida, no distribución predictiva lognormal ni promesa de intervalos calibrados. No seleccionar posterior entre raw y truncado.

Registrar convergencia, parámetros, log-verosimilitud, alertas, finitud, inicialización y duración. Un candidato con fallo en algún fold no entra en selección completa y conserva su recibo de fallo; no sustituir silenciosamente otra receta. No reemplazar salidas fallidas por ceros: el truncamiento a cero sólo aplica a un pronóstico negativo finito de un ajuste válido. No ampliar rangos al ver desempeño. Cero reintentos previstos en el presupuesto base. Si todos fallan, declarar bloqueo. Si falla el ajuste final del ganador, detener ese control; una alternativa o reintento requiere regla predefinida/enmienda, no ranking de test.

## 5. Selección, congelación y actualización

### Escenario A recomendado: selección global de desarrollo, compatible con la naturaleza de §19

- En cada uno de los diez orígenes mensuales enero–octubre de 2024, estimar cada candidato una vez con historia válida s≤o−8. ARIMA no selecciona épocas, por lo que no necesita duplicar ajuste interno/refit de las redes en este escenario.
- Con esos parámetros filtrar o−7…o−1. Para cada día del bloque: pronosticar; guardar; al cierre incorporar su y observado para el pronóstico del día siguiente. No refit dentro del bloque.
- Cada candidato genera una trayectoria concatenada, hasta 305 fechas antes de comprobar máscara. Seleccionar por RMSE en unidades originales, no promedio no ponderado de RMSE mensual. Los controles tienen una trayectoria determinista, no una media ficticia de semillas.
- Para órdenes usar regla propuesta de parsimonia: conjunto con RMSE≤RMSE mínima+δ histórico (0,00025772842664981106); elegir menor número de parámetros estimados, luego menor d, luego menor p+q, luego RMSE y finalmente orden numérico (p,d,q). Reportar todo el conjunto. Ésta es una regla de tolerancia práctica propuesta, no prueba de equivalencia ni IC. Debe aceptarse antes de ejecutar; no altera δ ni obliga a extrapolarla a otras familias.
- Congelar orden, m y todas las reglas antes del piloto. Este escenario usa CV para seleccionar y **no** constituye OOF insesgado del procedimiento de selección. No atribuir esa propiedad a la mera existencia de ventanas internas en las redes.
- Piloto propuesto de §19: origen 08/11/2024, parámetros y escala hasta 31/10; filtrar 01–07/11 con parámetros fijos, pronosticar 08/11–30/12. No escoger orden/m otra vez con el piloto.
- Ajuste final propuesto: origen 01/01/2025, estimar sólo hasta 24/12/2024, filtrar 25–31/12 con parámetros fijos. Durante evaluación se admite sólo actualización del estado con observaciones ya realizadas. Ningún fit, modificación de escala, selección o calibración usa y de 2025–2026. Esta descripción no autoriza abrir ese periodo ahora.

La actualización se implementaría mediante extend diario o append(refit=False) tras verificar equivalencia causal en la versión fijada [S2–S3]. No usar estado suavizado con observaciones posteriores al origen, ni reconstruir pronósticos antiguos con un resultado que ya contiene el bloque completo. Las predicciones se persisten antes de revelar y_t.

### Escenario B alternativo, no acumulativo: selección interna por origen

Si se pide evaluar un procedimiento anidado, cada origen o necesita 18 ajustes con etiquetas hasta o−98, evaluación interna de o−90…o−8 con estado causal y selección del orden/m dentro de ese bloque; después un ajuste del ganador hasta o−8. Antes de emitir la primera predicción interna, filtrar el puente o−97…o−91 con parámetros ajustados hasta o−98. La ventana interna nominal contiene 83 fechas, sujetas a soporte. El orden podría variar entre folds; se evalúa el procedimiento, no un orden fijo elegido sobre toda CV. Antes de piloto/final se aplica la misma regla interna, no una mezcla oportunista con A. No llamar confirmatoria esta alternativa sobre fechas históricamente conocidas.

## 6. Contabilidad de ajustes, no estimación de horas

Supuestos: F=10 folds, C=18 candidatos, un solo optimizador/inicialización por ajuste, sin fallos/reintentos, sin intervalos paramétricos por bootstrap, sin ETS adicional ni variantes estacionales ARIMA. Un fit significa estimación de parámetros, no llamada forecast ni avance de filtro.

| Escenario | CV | Piloto | Final | Total si se autorizan todas las fases |
|---|---:|---:|---:|---:|
| A: selección global, una vez por orden/fold | C×F=180 | 1 | 1 | 182 |
| B: selección interna y refit ganador por origen | F×(C+1)=190 | C+1=19 | C+1=19 | 228 |
| Sólo orden fijado externamente antes de CV, sin buscar | F=10 | 1 | 1 | 12 |
| Naïve estacional y persistencia, cualquier escenario | 0 | 0 | 0 | 0 |

El tercer escenario sólo aplica si el orden se fija de antemano sin usar esa CV para elegirlo; no representa el coste de A reducido después de conocer el ganador. A y B son alternativas, no 410 ajustes obligatorios. Ajustes adicionales = reintentos realmente intentados + variantes incorporadas por enmienda; conservar éxitos y fallos en el consumo.

El naïve estacional sí tiene coste de evaluación/selección: en A, hasta 3×305=915 pronósticos de desarrollo, condicionados a soporte. ARIMA A: hasta 18×305=5.490 pronósticos CV y sus avances diarios de filtro; nominalmente añade 53 de piloto y 608 de evaluación sólo si esos soportes se verifican y autorizan. No se multiplican estos controles por R ni por las diez semillas del piloto neuronal. En B, el coste interno de predicciones aumenta además de los fits, y se debe registrar por fechas reales.

Estos ajustes se suman a las partidas neuronales/Ridge cuando el plan elegido se integre; no vuelven completo el subtotal reducido de §19. NAS, HPO ampliado neuronal, robustez, calibración y ablaciones siguen pendientes. Las 18 especificaciones ARIMA no acreditan equidad con la cuadrícula neuronal reducida de cuatro puntos, ya insuficiente según §20; el presupuesto común y la cobertura HPO por familia siguen por cerrar. No hay horas, memoria pico ni FLOPs estimados: se necesitan perfilado autorizado, hardware, versión y datos de desarrollo para medirlos. Los tiempos del ejemplo de statsmodels no se transfieren a BTCUSDT.

## 7. Pruebas y salida exigidas antes de entrenamiento real

1. Perturbar todo futuro ≥t y comprobar que el pronóstico emitido para t no cambia; perturbar datos posteriores al corte de ajuste y comprobar parámetros y escala invariantes.
2. Separar prueba de estado: cambiar y[t−1] válido puede cambiar ŷ_t, sin cambiar parámetros. Cambiar y_t no puede cambiar ŷ_t.
3. Verificar las fechas o−8/o−7 y o−98/o−97; distinguir embargo de estimación y observación permitida.
4. Calendario diario, y con ocho cierres, cero válidos, día incompleto y propagación de faltantes sin compresión ni relleno; probar límites de m=7/14/28.
5. Probar persistencia frente a ARIMA(0,1,0) en caso sintético sin huecos, estado serializado/recuperado y equivalencia entre procesamiento incremental y filtrado causal de prefijos.
6. Prueba de fallo sin reemplazo ni borrado de filas; selección invariante al orden de evaluación de candidatos; cero accesos al test desde runner de selección.
7. Recibo por fit: commit, versiones, configuración, inicio/corte de historia, máscara/hash, conteo válido, escala, periodo/orden, parámetros, estado al corte, observaciones consumidas, método filtrado, pronósticos raw/truncados, warnings, convergencia y duración. Registrar selección separada de estimación.

Métricas: RMSE primaria en soporte común, MAE/MSE/R² y MAPE con cobertura. Para MASE comparativa conservar **un denominador común train-only** definido por E1 sobre cambios de etiquetas consecutivas elegibles; no cambiarlo por periodo ganador ni usar un denominador propio por control sin etiquetar un análisis separado. Nunca enlazar fechas separadas por huecos como si fueran consecutivas. Inferencia temporal, DM-HLN/HAC, multiplicidad e intervalos siguen en especificación estadística separada: los controles no generan réplicas independientes por semilla ni aportan p-valores por existir.

## 8. Fuentes verificadas y alcance

Fuentes técnicas consultadas en línea el 08/11/2026. La elección de espacio, umbral de 90 observaciones, transformación, truncamiento y reglas de selección son propuestas propias; las fuentes no las validan para BTC.

- [S1] statsmodels, ARIMA: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html . API de órdenes, tendencia, estacionalidad y restricciones.
- [S2] statsmodels, ARIMAResults.extend: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.extend.html . Continuación de observaciones y resultados de filtrado.
- [S3] statsmodels, Forecasting in statsmodels: https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html . Ejemplo rodante, distinción append/refit y extend. No se heredan sus tiempos ni resultados.
- [S4] statsmodels, SARIMAX: Model selection, missing data: https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_internet.html . Tratamiento de faltantes en espacio de estados; no sustituye la prueba del adaptador ARIMA elegido.
- [S5] Hyndman y Athanasopoulos, Forecasting: Principles and Practice, §5.2: https://otexts.com/fpp3/toolbox.html#seasonal-na%C3%AFve-method . La URL consultada https://otexts.com/fpp3/simple-methods.html redirige al capítulo; definición de naïve estacional.
- [E2] Archivos leídos: book_e2/sections/17_contrato_sincronizado.md, 18_integracion_sintetica.md, 19_plan_auditoria.md y 20_contraste_guia.md en el commit de referencia indicado. Enlace fuente del contrato: https://github.com/jassan-bit/DeepLearning_202630/blob/main/book_e2/sections/17_contrato_sincronizado.md . Los archivos locales fueron la evidencia leída; no se afirma verificar aquí el contenido remoto de ese enlace.
- [Código] scripts/dl_data_v2.py leído para escalado relativo y cortes temporales; no se ejecutó ni se alteró.
