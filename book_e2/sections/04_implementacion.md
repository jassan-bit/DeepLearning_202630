# 4. Implementación de Modelos Benchmark

**Versión vigente: protocolo registrado el 6 de noviembre de 2026, commit `599a29f`.** La guía completa de 65 páginas ya está recibida. Se cerraron las reglas de ejecución antes de iniciar P0 y OOF; el benchmark final de veinte configuraciones, su HPO y su dimensionamiento P1 todavía no están ejecutados. P0/OOF ya está ejecutado y analizado en [§9](09_protocolo_registrado.md), como diagnóstico de desarrollo separado del benchmark final.

La [sección 9](09_protocolo_registrado.md) identifica el registro y los recibos. La réplica y el piloto **ETTh1** de [§5](05_paper_guia.md) conservan su alcance externo; no cuentan como entrenamientos ni potencia del benchmark Bitcoin. La propuesta v1 se mantiene como archivo histórico[^v1].

## 4.1 Contrato experimental y veinte configuraciones

Se conserva el contrato de [§1](01_punto_partida.md): siete días de OHLCV, cinco canales, predicción a un día de σ₇ —desviación poblacional de siete retornos, `ddof=0`—, escalado causal y métrica primaria igual a la **media de las RMSE por semilla en escala original**. No se incorporan rezagos de σ₇ a las redes.

| Grupo | Configuraciones | Tratamiento |
|---|---|---|
| Top 10 | TSMixer, TiDE, TimeMixer, TimesNet, PatchTST, Informer, Autoformer, iTransformer, Crossformer, S-Mamba | Diez backbones con arquitectura fuente identificada; adaptadores finales pendientes de revisión |
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
| 1 de noviembre–30 de diciembre de 2024 | Sesenta fechas de desarrollo para P0; después, P1 con recetas finales congeladas |
| Desarrollo 2023–2024 | Ajuste final nominal al 1 de enero de 2025; etiquetas hasta el 24 de diciembre de 2024 |
| 1 de septiembre–30 de octubre de 2026 | Reserva de sesenta fechas **ya observada en E1**; futuras comparaciones exploratorias |
| Enero de 2025–agosto de 2026 | Archivo histórico conocido; no se reasigna a entrenamiento o a un test desconocido |

En un origen q se admiten etiquetas de ajuste hasta q−8: horizonte uno y embargo siete. Los índices conservan días inválidos y los soportes de entradas/etiquetas; no basta comparar nombres de archivos. La prueba histórica no recupera cegamiento por registrar ahora un protocolo. La fase futura de scoring E2 debe ocurrir con pesos, recetas, cohortes y análisis fijados y mantiene alcance exploratorio.

### 4.1.2 HPO común y selección temporal

El [diseño HPO registrado](../_static/remediation_protocol/hpo.json) asigna **T=12 configuraciones a cada una de las quince recetas ajustables**, con semilla de generación `202611061`: 180 candidatos, no 180 entrenamientos. Folds, semillas, selección de épocas y reajustes multiplican el trabajo.

| Control | Regla registrada |
|---|---|
| Presupuesto | 172.800 CPU-s asignados por familia ajustable; mismo hardware y trazabilidad del consumo real |
| Redes | Learning rate loguniforme 10⁻⁴–0,003; weight decay loguniforme 10⁻⁶–0,01; dropout uniforme 0–0,3; batch ∈ {16,32,64} |
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

La [matriz de decisiones](../_static/remediation_protocol/architecture_decisions.json) separa **arquitectura publicada, configuración fuente, receta pequeña P0 y adaptación final D7**. Por ejemplo, PatchTST fuente conserva tres bloques y FFN 128 en su configuración pequeña publicada; el piloto de dos bloques y FFN 32 no se convierte por ello en su configuración final.

Las diez excepciones finales de arquitectura siguen sin verificación operativa. Antes del HPO real deben cotejarse código, manifiesto, parámetros fuente y modificaciones indispensables de contexto/salida. Los valores no reportados permanecen identificados; no se completan con preferencias personales o con resultados de P0.

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

**P0 no dimensiona recetas posteriores al HPO.** P1 usará recetas finales congeladas, cinco modelos registrados —tres Top 10 y ambas referencias GRU— y diez semillas nuevas `660201`–`660210`. Dimensionará R con meta 0,80, alternativas de ganancia/erosión 2δ y ahorro de latencia del 20 %, tratando aleatoriedad de entrenamiento y calendario por separado. El máximo es R=200; si no alcanza la meta se informa potencia insuficiente y diferencia mínima detectable. Aumentar R no crea más fechas o mercados.

## 4.4 Auditoría, costes y registro de incidencias

La auditoría formal de §10 se exige a cada pipeline real: soportes cronológicos, duplicados/casi duplicados pertinentes, ajuste de transformaciones, disponibilidad de atributos, sanidad con etiquetas permutadas y desplazamientos explicados. Los checks no aplicables se motivan. Un fallo debe corregirse y repetirse en la rama afectada; una nota al pie no valida un resultado contaminado.

Se registran configuraciones, fases, origen, semilla, hashes de datos/código/índices/checkpoints, curvas, épocas, fallos y consumo. Solo cabe reanudar con la misma configuración ante interrupción de infraestructura documentada; no descartar una semilla por mal desempeño. Ni un hash ni una prueba sintética acreditan por sí solos ausencia de fuga, entrenamiento justo o potencia.

Entrenamiento y HPO se miden por separado. Latencia futura: mismo CPU, batch 1, un thread, calentamiento de 256 llamadas y 61 bloques intercalados de 256 llamadas; se declaran precisión y coste del preprocesamiento. Parámetros, FLOPs/MACs, memoria y throughput se identifican con su método de medición. Energía no medida no se informa como cero.

## 4.5 Evaluación, incertidumbre y análisis complementarios

La [sección 6](06_benchmark_final.md) fija las tablas, familias y diagnósticos finales. Se conservan MAE, MSE, R², MAPE y MASE junto a RMSE. MAPE publica cobertura sobre objetivos positivos; R² y MASE quedan indefinidos si su denominador es cero. La escala MASE usa exclusivamente pares consecutivos elegibles del entrenamiento, sin comprimir huecos.

El margen heredado es **δ=0,00025772842664981106**. El nuevo margen operacional de sobreajuste es δg=δ, en las mismas unidades de RMSE; no se le atribuye una decisión previa a E1. Una brecha train–validación requiere predicción en modo evaluación y reglas comparables; no se confunde con el gap P0 entre regímenes.

Los IC temporales remuestrean bloques comunes de fechas y recalculan la RMSE por trayectoria antes de promediar. Los IC entre semillas describen entrenamiento condicionado al mismo calendario. Ninguno proporciona por sí solo intervalos predictivos; techo de desempeño, γ, calibración y cobertura de tres modelos siguen exigiendo ejecución y evidencia propias.

## 4.6 Robustez, interpretabilidad y ablación previstas

Estas ramas finales siguen pendientes. Se fijan en desarrollo y antes de su scoring, sin escoger condiciones por resultados de test:

- **Robustez:** ruido causal en OHLC con intensidades 0,05/0,10/0,20 de la escala de retornos train y ausencia sintética de volumen 0,10/0,20/0,30, además del control limpio. Se mantienen etiquetas/cohorte y realizaciones comunes por timestamp; se reportan curvas, error absoluto y área normalizada, sin fingir nuevas muestras independientes.
- **Interpretabilidad:** tres modelos seleccionados en desarrollo, Integrated Gradients u oclusión registrada si el mecanismo no admite gradientes adecuados, con controles de completitud, borrado/inserción de días completos, fidelidad y estabilidad entre semillas. Una atención visual no prueba causalidad.
- **Ablación:** al menos tres componentes reales de la mejor red profunda elegida en desarrollo; mismas semillas, fechas y presupuesto comparable. Se publican diferencias pareadas, incertidumbre, multiplicidad y cambios de capacidad/coste. Si gana una referencia clásica, no se renombra a la mejor red como ganadora global.

La taxonomía de errores debe ser excluyente y tener al menos cien casos únicos por modelo principal. Los 305 orígenes potenciales OOF permiten planificarla; sesenta fechas de reserva no se multiplican por semillas para inventar cien casos. P0 no satisface automáticamente el reporte de errores, la calibración o la robustez del benchmark final.

## 4.7 Evidencia vigente y archivo

- [Protocolo registrado y sus 19 campos](09_protocolo_registrado.md), [HPO](../_static/remediation_protocol/hpo.json), [arquitecturas](../_static/remediation_protocol/architecture_decisions.json) y [preregistro](../_static/remediation_protocol/preregistration.json).
- [Validación documental](../_static/remediation_protocol/validation.json) y [180 candidatos predefinidos](../_static/remediation_protocol/candidate_registry.json): describen reglas y pruebas, sin resultados de HPO.
- [Descargar protocolo, código y evidencias](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip).

Los guardas separan registro Git, preflight P0/OOF, revisión arquitectónica final, HPO, P1, R, pesos finales y scoring. El registro público ya existe para P0; la preparación del benchmark completo requiere sus recibos adicionales. No se cambia un `ready=false` a «cumplido» porque el documento o un fichero exista.

[^v1]: La [propuesta v1 fechada el 5 de noviembre](../_static/benchmark_design/proposal.json), su [informe](../_static/benchmark_design/report.json) y su [ZIP](../_static/benchmark_design_download/Diseno_E2_propuesta_auditoria.zip) se conservan como antecedentes: catálogo 15, grid 8 y cómputo propuesto antiguo. No son el diseño vigente ni habilitan sus entrenamientos. Sus comprobaciones históricas conservan su alcance original.
