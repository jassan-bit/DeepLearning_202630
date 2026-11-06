# 4. Implementación de Modelos Benchmark

**Estado: P0/OOF registrado y ejecutado; nueva implementación final registrada y HPO en ejecución.** La guía completa de 65 páginas ya está recibida. Se cerraron las reglas de ejecución antes de iniciar P0 y OOF; el benchmark final de veinte configuraciones, su HPO todavía no está cerrado y su dimensionamiento P1 no se ha ejecutado. P0/OOF ya está ejecutado y analizado en [§9](09_protocolo_registrado.md), como diagnóstico de desarrollo separado del benchmark final.

La [sección 9](09_protocolo_registrado.md) identifica el registro y los recibos. La réplica y el piloto **ETTh1** de [§5](05_paper_guia.md) conservan su alcance externo; no cuentan como entrenamientos ni potencia del benchmark Bitcoin. La propuesta v1 se mantiene como archivo histórico[^v1].

## 4.1 Contrato experimental y veinte configuraciones

Se conserva el contrato de [§1](01_punto_partida.md): siete días de OHLCV, cinco canales, predicción a un día de σ₇ —desviación poblacional de siete retornos, `ddof=0`—, escalado causal y métrica primaria igual a la **media de las RMSE por semilla en escala original**. No se incorporan rezagos de σ₇ a las redes.

| Grupo | Configuraciones | Tratamiento |
|---|---|---|
| Top 10 | TSMixer, TiDE, TimeMixer, TimesNet, PatchTST, Informer, Autoformer, iTransformer, Crossformer, S-Mamba | Diez adaptadores implementados y revisados por modelo, con pruebas sintéticas; resultados Bitcoin pendientes |
| Bases ajustadas en E2 | GRU-E2, LSTM-E2, TCN-E2 | Mismo presupuesto de ajuste que el Top 10; no sustituyen las anclas |
| Anclas heredadas E1 | GRU/D7-c03, LSTM/D7-c02, TCN/D7-c02 | Recetas históricas fijas, contexto y referencia; no se les atribuye HPO nuevo |
| Referencias de la ruta temporal | Ridge, ARIMA, persistencia y naive estacional | Ridge y ARIMA ajustables; las dos referencias ingenuas son deterministas |

Son **20 configuraciones, 15 ajustables y 5 referencias fijas**. GRU/D7-c03 conserva su papel de baseline heredado. La comparación adicional contra GRU-E2 permite examinar la ganancia frente a una base retunada; ambas familias quedaron declaradas antes del piloto.

Ridge usa las mismas 35 entradas ordenadas. ARIMA y las referencias ingenuas emplean historia causal del objetivo como excepción explícita de la ruta temporal: no reciben la misma representación que las redes. Se conserva la predicción cruda de los estimadores clásicos y se aplica la regla fija `max(0, predicción)` para su salida de volatilidad; no se eliminan fechas por un pronóstico negativo. Los detalles causales están en [§7](07_ruta_temporal.md).

### 4.1.1 Calendario heredado y test conocido

| Periodo UTC | Uso |
|---|---|
| 2023 | Historia inicial; ajuste de transformaciones y umbrales permitidos |
| Enero–octubre de 2024 | Diez orígenes mensuales; OOF y desarrollo para selección temporal anidada |
| 1 de noviembre–30 de diciembre de 2024 | Sesenta fechas históricas de P0, conservadas con su alcance original |
| 8 de noviembre–30 de diciembre de 2024 | **53 fechas** para P1 posterior al HPO y calibración secundaria A; ajuste/selección hasta 31/10 y embargo 01–07/11 |
| Desarrollo 2023–2024 | Ajuste final nominal al 1 de enero de 2025; etiquetas hasta el 24 de diciembre de 2024 |
| 1 de septiembre–30 de octubre de 2026 | Reserva de sesenta fechas **ya observada en E1**; futuras comparaciones exploratorias |
| Enero de 2025–agosto de 2026 | Archivo histórico conocido; no se reasigna a entrenamiento o a un test desconocido |

En un origen q se admiten etiquetas de ajuste hasta q−8: horizonte uno y embargo siete. Los índices conservan días inválidos y los soportes de entradas/etiquetas; no basta comparar nombres de archivos. La prueba histórica no recupera cegamiento por registrar ahora un protocolo. La fase futura de scoring E2 debe ocurrir con pesos, recetas, cohortes y análisis fijados y mantiene alcance exploratorio.

### 4.1.2 HPO común y selección temporal

La [enmienda HPO preparada](../_static/completion_protocol/hpo.json) conserva **T=12 configuraciones a cada una de las quince recetas ajustables**, con semilla de generación `202611061`: 180 candidatos, no 180 entrenamientos. Folds, semillas, selección de épocas y reajustes multiplican el trabajo.

| Control | Regla registrada |
|---|---|
| Presupuesto | 172.800 CPU-s asignados por familia ajustable; mismo hardware y trazabilidad del consumo real |
| Redes | Once candidatos aleatorios originales: LR 10⁻⁴–0,003, weight decay 10⁻⁶–0,01, dropout 0–0,3 y batch {16,32,64}; candidato c12 respaldado por fuente, con excepciones exactas declaradas |
| Estructura | Profundidad, ancho, cabezas, parches, escalas y kernels sujetos al caso fuente de cada arquitectura; no se retocan silenciosamente |
| Ridge | Doce valores de regularización en escala logarítmica 10⁻⁶–10³; representación fija de 35 entradas |
| ARIMA | Doce órdenes: p∈{0,1,2}, d∈{0,1}, q∈{0,1}; tendencia constante si d=0 y sin tendencia si d=1 |
| Optimización neural | AdamW, MSE con escala de Y aprendida en train, sin scheduler, FP32, clipping de norma 1 |
| Parada | Máximo 100 épocas, paciencia 12 y mejora mínima 0; límite de 900 s por ajuste de optimizador |
| Semillas HPO | `2026110611` y `2026110612`; los estimadores deterministas no repiten ajustes idénticos para fingir dos realizaciones |

La igualdad de asignación no demuestra que todos completen el mismo esfuerzo. Se conservan intentos inválidos, fallos y timeouts, sin reposición selectiva o extensiones para una familia. Una búsqueda incompleta se declara como tal; no permite atribuir la diferencia exclusivamente al backbone.

Dentro de cada origen externo mensual se usan tres bloques internos de treinta días, anteriores al corte permitido. La selección de épocas usa el intervalo nominal q−90 a q−8: **83 fechas potenciales**, antes de elegibilidad; su entrenamiento temprano termina en q−98. Después se reajusta desde una inicialización registrada y con transformaciones aprendidas por separado.

Se concatenan fechas internas por trayectoria, se calcula su RMSE y se promedia entre semillas. Se minimiza esa métrica; no la media de RMSE mensuales. Los meses externos evalúan el procedimiento de selección. La búsqueda forward adicional de receta final usa desarrollo, se distingue de aquella evaluación y debe cerrarse antes de P1. Los desempates siguen error, latencia de desarrollo, parámetros e identificador registrados.

## 4.2 Arquitecturas finales, adaptadores y condición NAS

La [revisión final](../_static/completion_protocol/architecture_review.json), junto con la [matriz original conservada](../_static/remediation_protocol/architecture_decisions.json), separa **arquitectura publicada, configuración fuente, receta pequeña P0 y adaptación final D7**. Por ejemplo, PatchTST fuente conserva tres bloques y FFN 128 en su configuración pequeña publicada; el piloto de dos bloques y FFN 32 no se convierte por ello en su configuración final.

La [revisión independiente de las diez adaptaciones](../_static/completion_protocol/architecture_review.json) coteja fuentes, presets, mecanismos, gradientes, invariancia de batch y ausencia de valores futuros. Las estructuras finales están definidas individualmente; no requieren NAS en esos diez casos. La prueba sintética de memorización pasó para los diez modelos. Estos controles habilitan la verificación integrada y el registro; **no acreditan todavía HPO, ausencia de toda fuga ni rendimiento Bitcoin**. Los cambios D7 y la procedencia de cada valor se detallan en §14.

Para el piloto limitado, TimeMixer, iTransformer y PatchTST usan latentes `[B,5,16]`. Su cabeza común concatena 80 latentes y diez estadísticas —media y desviación poblacional por canal de **la misma** entrada escalada D7—, seguidas de `Linear(90,1)` y Softplus. No son diez covariables externas. TimeMixer conserva `moving_avg=25` y dos escalas 7/3. Esta adaptación no se presenta como réplica literal ni exonera las configuraciones finales.

Si una arquitectura queda definida únicamente por familia, se activa §14: 1.000 combinaciones de tres anchos {16,32,64,128,256} y ocho patrones de dropout 0,2. El protocolo condicional exige piloto de ruido, HPO previo del control 64/64/64 con patrón 010, cribado completo, k finalistas más control y al menos cinco semillas nuevas en fase 2. No hay NAS ejecutado ni obligación de aplicar 1.000 búsquedas a los diez modelos por defecto. La revisión debe justificar el caso y preservar el mismo esfuerzo entre los modelos afectados.

Cada adaptador real debe pasar forma, gradientes/checkpoint, escala del objetivo y causalidad: sin verdad futura en decoders, sin desnormalizar una columna de OHLCV como σ₇ y sin selección de periodos entre orígenes del mismo batch. TiDE conserva sus bloques de proyección/decoder con las modificaciones de tarea declaradas; Crossformer debe conservar HED y la combinación de escalas, no sustituirse por un encoder aislado.

## 4.3 P0 ejecutado y P1 de dimensionamiento

El registro público [599a29f](https://github.com/jassan-bit/DeepLearning_202630/commit/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0) precede al inicio de P0/OOF. El checkout de ejecución se registró a las **14:52:32 UTC del 6 de noviembre de 2026** y superó sus controles previos específicos. Se usan dos trabajadores y un thread por entrenamiento; estos controles no certifican los veinte pipelines finales.

| Rama registrada P0 | Modelos y semillas | Alcance y trabajo planificado |
|---|---|---|
| Noviembre–diciembre | TimeMixer, iTransformer, PatchTST y GRU/D7-c03; cinco semillas nuevas `660101`–`660105` | 20 trayectorias / 40 ajustes; factibilidad y variabilidad condicional; gap descriptivo |
| OOF enero–octubre | Los mismos cuatro modelos, cinco semillas y recetas; diez orígenes | 200 trayectorias / 400 ajustes; 305 fechas potenciales; familia principal del diagnóstico P0 |

Se completaron **220 trayectorias y 440 ajustes**, verificadas mediante recibos cerrados; se puntuaron y analizaron en §9. Cada trayectoria separa selección de épocas y reajuste. Estos resultados P0 no se presentan como ejecución de HPO o del benchmark final.

El diagnóstico principal P0 compara RMSE de alta volatilidad frente a régimen normal en los tres modelos Top 10, con Holm sobre tres hipótesis y α=0,01. El régimen usa σ₇(t−1) y el cuantil 0,75 de entrenamiento. Su conjunto principal quedó fijado en OOF antes de resultados; noviembre–diciembre no aporta una segunda familia entre la que elegir un resultado favorable. Es un estudio de desarrollo separado de las familias del benchmark final.

**P0 no dimensiona recetas posteriores al HPO.** P1 se corrige a **08/11–30/12/2024, 53 fechas**, con ajuste/selección hasta 31/10 y embargo 01–07/11. El corte reconoce que las recetas heredadas ya se seleccionaron con octubre; no se declara un corte ficticio 24/10. Usará recetas finales congeladas, cinco modelos registrados —tres Top 10 y ambas referencias GRU— y diez semillas nuevas `660201`–`660210`. Dimensionará R con meta 0,80, alternativas de ganancia/erosión 2δ y ahorro de latencia del 20 %, tratando aleatoriedad de entrenamiento y calendario por separado. El máximo es R=200; si no alcanza la meta se informa potencia insuficiente y diferencia mínima detectable. Aumentar R no crea más fechas o mercados.

## 4.4 Auditoría, costes y registro de incidencias

La auditoría formal de §10 se exige a cada pipeline real: soportes cronológicos, duplicados/casi duplicados pertinentes, ajuste de transformaciones, disponibilidad de atributos, sanidad con etiquetas permutadas y desplazamientos explicados. Los checks no aplicables se motivan. Un fallo debe corregirse y repetirse en la rama afectada; una nota al pie no valida un resultado contaminado.

Se registran configuraciones, fases, origen, semilla, hashes de datos/código/índices/checkpoints, curvas, épocas, fallos y consumo. Solo cabe reanudar con la misma configuración ante interrupción de infraestructura documentada; no descartar una semilla por mal desempeño. Ni un hash ni una prueba sintética acreditan por sí solos ausencia de fuga, entrenamiento justo o potencia.

Entrenamiento y HPO se miden por separado. Latencia futura: mismo CPU, batch 1, un thread, calentamiento de 256 llamadas y 61 bloques intercalados de 256 llamadas; se declaran precisión y coste del preprocesamiento. Parámetros, FLOPs/MACs, memoria y throughput se identifican con su método de medición. Energía no medida no se informa como cero.

## 4.5 Evaluación, incertidumbre y análisis complementarios

La [sección 6](06_benchmark_final.md) fija las tablas, familias y diagnósticos finales. Se conservan MAE, MSE, R², MAPE y MASE junto a RMSE. MAPE publica cobertura sobre objetivos positivos; R² y MASE quedan indefinidos si su denominador es cero. La escala MASE usa exclusivamente pares consecutivos elegibles del entrenamiento, sin comprimir huecos.

El margen heredado es **δ=0,00025772842664981106**. El nuevo margen operacional de sobreajuste es δg=δ, en las mismas unidades de RMSE; no se le atribuye una decisión previa a E1. Una brecha train–validación requiere predicción en modo evaluación y reglas comparables; no se confunde con el gap P0 entre regímenes.

Los IC temporales remuestrean bloques comunes de fechas y recalculan la RMSE por trayectoria antes de promediar. Los IC entre semillas describen entrenamiento condicionado al mismo calendario. Ninguno proporciona intervalos predictivos por sí solo. La rama A prepara los veinte predictores antes de conocer el ranking final, con cinco miembros fijos por red y 53 fechas de calibración. El proxy de predictibilidad, γ y la cobertura requieren ejecución y conservan sus supuestos; véanse §§6/12.

## 4.6 Robustez, interpretabilidad y ablación previstas

Estas ramas están implementadas y preespecificadas en la [enmienda](../_static/completion_protocol/analysis_additions.json); sus resultados reales siguen pendientes:

- **Robustez:** tres severidades de ruido (0,01/0,03/0,05), ausencia de barras (0,05/0,10/0,20) y outliers (0,10/0,25/0,50), además del control limpio. Los shocks comunes por fecha conservan la coherencia de ventanas solapadas; OHLC comparten shock y volumen tiene shock propio. La imputación usa medias crudas de train. Las referencias endógenas reciben fallos causales de su historia, con coeficientes congelados, en lugar de atribuirles inmunidad.
- **Interpretabilidad:** tres modelos elegidos en desarrollo entre los veinte, con oclusión de catorce grupos —siete barras OHLC y siete volúmenes— y baseline de 35 medias de celdas aprendido solo en train. Integrated Gradients es secundario y se informa únicamente si supera completitud; fidelidad por borrado/inserción y estabilidad usan las primeras cinco semillas finales fijadas. Las intervenciones en entradas correlacionadas no prueban causalidad.
- **Ablación:** tres componentes definidos para cada uno de los diez backbones antes del ajuste. La mejor red Top10 según desarrollo se reajusta desde cero en control y tres variantes, con el mismo R/semillas finales, corte 31/10 y **DEV53 (08/11–30/12/2024)**. No usa test ni reutiliza el control primario entrenado hasta diciembre. Tres contrastes prácticos bilaterales usan Holm α=0,05. Las cuatro recetas/semillas quedan predefinidas; cada checkpoint de desarrollo puede podarse tras cerrar su scoring, conservando la cadena de hashes y cuatro ejemplos de la primera semilla. Antes del primer scoring de esta rama deben estar cerrados todos los pesos primarios, A y de controles negativos. Si gana un clásico globalmente, se declara esa diferencia.

La taxonomía de errores debe ser excluyente y tener al menos cien casos únicos por modelo principal. Los 305 orígenes potenciales OOF permiten planificarla; sesenta fechas de reserva no se multiplican por semillas para inventar cien casos. P0 no satisface automáticamente el reporte de errores, la calibración o la robustez del benchmark final.

## 4.7 Evidencia vigente y archivo

- [Enmienda preparada](../_static/completion_protocol/analysis_additions.json), [HPO T12](../_static/completion_protocol/hpo.json), [revisión arquitectónica](../_static/completion_protocol/architecture_review.json) y [180 candidatos](../_static/completion_protocol/candidate_registry.json).
- [Registro P0 y sus 19 campos](09_protocolo_registrado.md) y [preregistro original conservado](../_static/remediation_protocol/preregistration.json).
- [Validación documental](../_static/remediation_protocol/validation.json) y [180 candidatos predefinidos](../_static/remediation_protocol/candidate_registry.json): describen reglas y pruebas, sin resultados de HPO.
- [Descargar protocolo, código y evidencias](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip).

Los guardas separan registro Git, preflight P0/OOF, revisión arquitectónica final, HPO, P1, R, pesos finales y scoring. El registro público ya existe para P0; la preparación del benchmark completo requiere sus recibos adicionales. No se cambia un `ready=false` a «cumplido» porque el documento o un fichero exista.

[^v1]: La [propuesta v1 fechada el 5 de noviembre](../_static/benchmark_design/proposal.json), su [informe](../_static/benchmark_design/report.json) y su [ZIP](../_static/benchmark_design_download/Diseno_E2_propuesta_auditoria.zip) se conservan como antecedentes: catálogo 15, grid 8 y cómputo propuesto antiguo. No son el diseño vigente ni habilitan sus entrenamientos. Sus comprobaciones históricas conservan su alcance original.
