# 45. Enmienda del protocolo: calendario común, presupuesto y prueba final

**Registro anterior conservado:** para el trabajo nuevo afectado, la [enmienda reducida de §48](48_protocolo_reducido.md) establece seis plazas HPO, una semilla y selección global sobre tres folds internos para la evaluación heredada posterior. Este capítulo y sus archivos mantienen intactas sus reglas originales y su procedencia; la nueva enmienda no acredita ejecución ni modifica automáticamente Colab.

**Enmienda prospectiva del 12/11/2026:** se mantiene el [catálogo obligatorio de dieciséis sistemas](43_catalogo_obligatorio.md) y la [adjudicación de nueve cuerpos definidos y TiDE NAS](44_arquitecturas_publicadas.md). Se registran **tres anclas HPO comunes, una sola NAS inicial para TiDE y diez reajustes mensuales de evaluación**. El refit histórico de enero de 2025 utiliza una receta ya seleccionada, sin otra búsqueda.

Esta enmienda rige el trabajo nuevo afectado. Los resultados y las fechas de acceso anteriores conservan su procedencia; el registro actual no se presenta como un preregistro anterior a todas las ejecuciones de E2. La publicación tampoco modifica el proceso ni las variables de una sesión privada de Colab.

## Por qué se cambia el calendario

La ruta F2a exige origen móvil, embargo, disponibilidad causal y el mismo esquema para todos los sistemas (guía, pp. 30–31). **No fija once búsquedas completas.** La repetición de NAS en cada fold externo aparece en §14.5, p. 56, como una extensión que se realiza **«cuando el costo lo permita»**. Por eso se distingue seleccionar una receta de reajustar sus pesos con más pasado disponible.

Las tres anclas cubren el inicio del estudio, una actualización semestral y una actualización anterior al bloque reservado para potencia. Sus validaciones tienen 83 días y no se solapan. La elección conserva una evaluación mensual más amplia y reduce búsquedas repetidas; **no demuestra por sí sola que tres anclas sean suficientes para todos los regímenes**. Se reportarán métricas por origen, cambios de receta y estabilidad temporal. Una cobertura insuficiente se declarará como limitación; no se añadirán anclas después de observar la prueba final.

La propuesta histórica de §37 y el [plan sellado de §40](40_calendario_original.md) conservan sus fechas. Esta enmienda no activa sin cambios aquella propuesta: fija otro calendario y mantiene los reajustes mensuales.

## Calendario común de selección y evaluación

| Ancla HPO | Entrenamiento interno hasta | Validación interna, ambos extremos incluidos | Primer uso de la receta |
|---|---|---|---|
| 01/01/2024 | 25/09/2023 | 03/10–24/12/2023 | Enero–junio de 2024 |
| 01/07/2024 | 25/03/2024 | 02/04–23/06/2024 | Julio–septiembre de 2024 |
| 01/10/2024 | 25/06/2024 | 03/07–23/09/2024 | Octubre de 2024 y refit histórico de enero de 2025 |

En cada ancla, los trece sistemas neuronales reciben las mismas oportunidades HPO y los comparadores ajustables usan la misma validación temporal. La selección es **local al ancla**; no se agrega información de anclas posteriores para elegir una receta que se aplique a meses anteriores.

La evaluación de desarrollo conserva los **diez bloques mensuales disjuntos de enero a octubre de 2024**, con un origen al inicio de cada mes. En cada origen se reajustan pesos y transformaciones desde cero con el pasado admisible, utilizando la receta más reciente que ya estaba disponible. **Reajuste mensual no significa nueva HPO ni nueva NAS.** El calendario tiene 305 días objetivo antes de aplicar la máscara de calidad común; el número elegible se acredita con los datos, no con ese conteo de calendario.

El refit histórico del **01/01/2025** usa la receta seleccionada en octubre y etiquetas de entrenamiento hasta el **24/12/2024**. Sus pesos y escaladores quedan fijos para la evaluación histórica principal de 608 fechas. No se ejecuta HPO sobre octubre–diciembre para este refit, porque se preserva la separación del piloto de potencia.

Este procedimiento es un origen móvil con actualización de receta por anclas previamente fijadas. No se denomina validación anidada con búsqueda completa en cada fold, ni se aplica retrospectivamente un ganador de octubre o enero de 2025 a meses anteriores.

### Embargo y disponibilidad

Para un origen de pronóstico `o`, la última etiqueta de entrenamiento es **`o−8 días`**. Se excluyen de entrenamiento las siete etiquetas **`o−7,…,o−1`**. La validación interna se extiende de `o−90` a `o−8`, con entrenamiento interno hasta `o−98`.

La ventana predictora contiene las siete barras OHLCV observadas anteriores al día objetivo. Las observaciones del puente pueden alimentar esa ventana y el estado causal de los comparadores, sin actualizar pesos, escaladores o hiperparámetros. La etiqueta σ₇ requiere ocho cierres y comparte retornos con etiquetas próximas; por eso se conserva este embargo incluso con horizonte de un día. No se comprimen huecos del calendario ni se ajustan transformaciones con evaluación.

## HPO y una sola NAS TiDE

Se conservan **18 configuraciones HPO por red y ancla**, con dos semillas: **271201 y 271202**. Los puntos de tasa de aprendizaje, weight decay y batch mantienen sus oportunidades registradas. Las restauraciones de §44 tienen los knobs estructurales fijados al cuerpo fuente y recetas nuevas; los sistemas conservados mantienen sus dominios aplicables.

El límite de cada ajuste completo sigue en **200 épocas**, paciencia **20**, mejora estricta y primer checkpoint en caso de igualdad. Se conservan AdamW, FP32, clipping y scheduler registrados. Los fallos consumen su intento y se conservan; no se sustituyen silenciosamente configuraciones o semillas.

Para TiDE se aplica esta secuencia:

1. HPO de enero de 2024 sobre la arquitectura de referencia registrada.
2. Congelación de los hiperparámetros ganadores y evaluación de **las 1.000 arquitecturas NAS-1**, con semilla 271301 y máximo de 50 épocas.
3. Confirmación de los finalistas y la referencia, sin duplicarla, con las cinco semillas **271401–271405**, distintas de NAS-1, y máximo de 200 épocas.
4. Congelación de la arquitectura ganadora antes del primer bloque externo. Las HPO de julio y octubre ajustan hiperparámetros sobre **esa arquitectura ya congelada**; no vuelven a ejecutar NAS.

La arquitectura TiDE queda elegida bajo las condiciones iniciales de desarrollo, sin afirmar que sea óptima para todos los regímenes posteriores. La clave de una HPO posterior debe incluir la arquitectura congelada: un resultado obtenido sobre la referencia no equivale a uno sobre la ganadora.

El piloto de ruido conserva doce arquitecturas y cinco semillas, anteriores a la primera evaluación de 2024. Fija **k20 o k40** mediante la regla registrada de correlación de rangos; correlación mediana inferior a 0,5 o no estimable conduce a k40. Se congela k antes de NAS-1. La cota presupuestaria reserva k40; no se usa k10 sólo por urgencia.

### Criterios de selección

- **HPO local:** menor media de las RMSE individuales de las dos semillas, sobre la validación común del ancla; empate exacto por ID estable.
- **Épocas de refit:** mediana redondeada hacia arriba de las mejores épocas de las semillas ganadoras, por fase. Para TiDE en enero se usan las cinco semillas de NAS-2; en julio y octubre se usan las dos semillas HPO sobre la arquitectura ganadora de enero. Los otros cuerpos definidos usan sus dos semillas HPO. SAE conserva AE1, AE2 y supervisado como tres etapas reales.
- **NAS:** menor RMSE medio de confirmación; el empate técnico por δ o solapamiento de IC nominales permite la regla de parsimonia por parámetros, seguida de métrica e ID estable.

**Solapar IC no demuestra equivalencia.** El informe NAS todavía debe acreditar el conjunto indistinguible mediante el análisis pertinente TOST/ROPE/MCS, la correlación entre fases y el sesgo del ganador. Esos análisis no se dan por cerrados con la regla de selección.

ARIMA compara **18 órdenes** `(p=0..2, d=0..1, q=0..2)` en cada ancla. SeasonalNaive selecciona entre periodos **7/14/28** y Persistence conserva lag 1. Los sistemas deterministas no adquieren réplicas independientes por repetirlos con etiquetas de semilla distintas.

## Presupuesto y contabilidad

| Selección neuronal programada | Cota de ajustes lógicos |
|---|---:|
| HPO: 13 redes × 18 configuraciones × 2 semillas × 3 anclas | 1.404 |
| NAS-1 TiDE: 1.000 arquitecturas × una semilla | 1.000 |
| NAS-2 TiDE: hasta 41 candidatos distintos × cinco semillas, con k40 | 205 |
| **Total máximo de selección neuronal** | **2.609** |

La cota es anterior a descontar reutilización compatible. **No es el número de modelos finales, un mínimo literal del PDF ni una estimación de horas.** SAE añade dos etapas auxiliares por ajuste HPO: hasta 216 etapas adicionales y **2.825 fases de optimización** para esta selección. ARIMA añade **54 ajustes**. Los pilotos, perfiles, reajustes mensuales y finales, controles, ablaciones, reproducción y análisis complementarios quedan fuera de esas cotas de selección y deben contabilizarse por separado.

El límite autorizado conserva su historia **48 → 56 → 64 horas acumuladas**. Migrar de plan no concede otras 64 horas. Deben conciliarse los diarios de `campaigns/e2-cuda-selection-v1` y `campaigns/e2-cuda-crop-11-v2`, incluidos sesiones abiertas, reservas y trabajo de desenlace incierto. El saldo de unas 4 h 9 min de §41 pertenece a un corte histórico; **el saldo actual queda sin fijar hasta la conciliación**. No se amplía automáticamente el presupuesto ni se garantiza terminar el plan dentro del saldo.

La reutilización requiere coincidencia de receta, arquitectura, datos/partición, configuración, semilla y política de ejecución. Las cuatro recetas restauradas de §44 no heredan los resultados de sus variantes NAS anteriores. Los ajustes compatibles reutilizados conservan sus recibos y no se cuentan como trabajo nuevo.

## Métricas y unidad de análisis

La primaria se conserva: **media entre semillas de las RMSE individuales, en unidades originales de σ₇**, con **δ = 0,00025772842664981106**. No es la RMSE de un ensemble ni la raíz de una MSE agregada entre semillas.

Para resumir el origen móvil, cada semilla concatena sus fechas elegibles de los diez bloques mensuales y calcula una RMSE sobre esa trayectoria; después se resume entre semillas. Se reporta también cada origen. Los diez meses de una misma serie no se tratan como diez datasets independientes ni se convierten en réplicas iid por repetir semillas.

Las secundarias registradas son **MAE, MSE, R², MAPE, MASE, RMSSE y razón de RMSE frente a Persistence**. MAPE usa sólo objetivos positivos e informa cobertura. MASE y RMSSE utilizan diferencias entre etiquetas de días consecutivos elegibles del entrenamiento permitido, sin saltar huecos del calendario. Un denominador cero o ausente produce «no disponible» y su motivo; no se modifica la primaria ni se eliminan sus objetivos cero para favorecer una métrica secundaria.

### Familia primaria y criterios inferenciales

La familia primaria consta de **veinte contrastes**: cada uno de los diez modelos del Top 10 frente a **GRU32** y frente a **Persistence**. Se controla **FWER = 0,05 mediante Holm**. Para cada comparación se define la ganancia como `RMSE_baseline − RMSE_modelo`: la hipótesis nula es **ganancia ≤ δ** y la alternativa es **ganancia > δ**. ARIMA, SeasonalNaive y los demás contrastes conservan su papel de comparación o análisis secundario; no se introducen como otra familia primaria elegida después de mirar resultados.

La familia se aplica a la cohorte prospectiva declarada. Los mismos contrastes históricos se reportan por separado como exploratorios, con su máscara y linaje de pesos; no sustituyen la confirmación. Persistence aporta una trayectoria determinista y no adquiere réplicas independientes por copiarla entre semillas.

Los IC, efectos y contrastes deben respetar la dependencia temporal. Se registra **bootstrap temporal por bloques de catorce días**, con emparejamiento de semillas y fechas. Los bloques de **7 y 28 días** se usan sólo como sensibilidad diagnóstica, no para elegir el p-valor más favorable. La batería de tres pruebas admisibles requiere verificar sus supuestos y completar su implementación/evaluación; no se declara aprobada por registrar esta política. También siguen pendientes el conjunto indistinguible TOST/ROPE/MCS y los análisis de sesgo del ganador. El solapamiento de IC usado para seleccionar no los reemplaza.

## Potencia y repeticiones pendientes

El piloto queda en **08/11–30/12/2024**, con recetas congeladas en octubre y entrenamiento hasta **31/10/2024**. Se utilizan **TimeMixer, PatchTST y GRU32**, con al menos las cinco semillas pareadas **271701–271705**. Su bloque queda después de la última validación HPO, que termina el 23 de septiembre. Sus etiquetas sirven para dimensionar, no para volver a elegir arquitectura o hiperparámetros.

**R sigue sin fijar.** Se seleccionará un R común dentro del rango operativo registrado **6–30** mediante el piloto pareado, bootstrap temporal de catorce días y la familia primaria de veinte contrastes con Holm. La planificación utiliza una **alternativa de ganancia 2δ**: su separación respecto al margen nulo δ es δ. No se promete potencia 0,80 para una ganancia situada exactamente en la frontera nula δ.

El criterio de planificación exige que el **límite inferior de Wilson de la potencia simulada sea al menos 0,80**, no sólo su estimación puntual. El mínimo de seis ejecuciones no demuestra potencia. El piloto de 2024 y sus tres representantes tampoco garantizan el comportamiento de todos los modelos ni el régimen de una prueba en 2026–2027. Si el R necesario excede el techo operativo o los recursos, se reportan potencia alcanzable, MDE y la limitación; no se afirma haber alcanzado 0,80 por utilizar treinta semillas.

## Datos conocidos y prueba independiente

| Periodo o etapa | Estado de exposición y uso registrado |
|---|---|
| 2023–2024 | Desarrollo conocido: selección, evaluación temporal y piloto con roles separados |
| 01/01/2025–31/08/2026, 608 fechas | Ya evaluado en E1; evaluación histórica exploratoria del nuevo ajuste, fuera de la selección actual |
| 01/09/2026–30/10/2026, 60 fechas | También evaluado previamente; reserva histórica conocida, no test virgen |
| Prueba independiente nueva | Prospectiva; comienza después de la congelación global y todavía no tiene fechas materiales certificadas |

El manifiesto disponible termina el 30/10/2026. El [registro de exposición](../_static/protocol_amendment_20261112/exposure_registry.json) conserva las evidencias de accesos y evaluaciones. Que una función reserve metadatos sin abrir precios sólo describe esa función: no borra la exposición histórica del proyecto. Cambiar semillas, plan o nombre del conjunto no restablece independencia.

### Linaje histórico y linaje prospectivo

El benchmark histórico utiliza pesos reajustados al 01/01/2025 con entrenamiento hasta el 24/12/2024 y receta de octubre. La confirmación prospectiva tendrá **otros pesos**: se permite utilizar el pasado conocido hasta **24/08/2026** para su ajuste; se excluyen las etiquetas del **25–31/08** y se reserva **01/09–30/10/2026** para calibración. Los hiperparámetros y la arquitectura ya deben estar congelados antes de ese reajuste y esa calibración.

Esos pesos prospectivos no pueden presentarse como entrenados sólo en 2024 ni utilizarse para reevaluar las 608 fechas como si quedaran fuera de su entrenamiento. Los dos linajes se mantienen separados en configuraciones, hashes y tablas.

### Regla de la cohorte prospectiva

El primer día objetivo **T es exactamente el primer día UTC posterior a la congelación global**; no es una fecha mínima a partir de la cual se pueda elegir después otro comienzo. La congelación debe ser posterior al registro y acreditar los dieciséis sistemas, todos los pesos y escaladores, R y semillas comunes, recetas, calibradores y plan de análisis. **El recibo y la vinculación de fechas deben quedar registrados antes de T.** Un recibo antiguo no permite reservar retrospectivamente una cohorte cuyo primer día ya comenzó. Las fechas permanecen sin materializar hasta que exista esa vinculación válida.

Las observaciones entre el corte conocido **30/10/2026** y **T−1** constituyen un puente de información aún no adquirido en este registro. Sólo podrán aportar OHLCV pasado disponible en cada origen y el estado causal de ARIMA; no se usan para reajustar pesos, escaladores, calibradores o hiperparámetros. Su adquisición y disponibilidad deben acreditarse, conservando el linaje de ajuste hasta el 24/08 y calibración de septiembre–octubre.

Se fija antes de observar resultados una duración de **100 días calendario consecutivos**, de T a T+99. Sólo equivale a cien casos elegibles si no se pierden días por calidad o soporte común; no garantiza potencia. No se detiene ni prolonga el periodo según sus métricas. Si quedan menos de cien casos elegibles, se informa la insuficiencia y el análisis de cien casos por modelo principal permanece pendiente; no se extiende automáticamente la cohorte para cambiar el resultado.

Durante la prueba, cada predicción usa sólo información disponible en su origen y se persiste antes de observar su etiqueta. Se mantienen fijos pesos, escaladores y calibradores; las actualizaciones causales de estado de los comparadores conservan su registro. La adquisición futura, la máscara común y la separación de acceso todavía deben ejecutarse y verificarse. Registrar esta regla no acredita que ya exista una prueba independiente concluida.

## Registro, archivos y criterios de cierre

La hora de registro real es la del primer commit inmutable, identificada por el [recibo Git](../_static/protocol_amendment_20261112/git_registration.json). `prepared_at_utc` sólo fecha la preparación del documento. El ejecutor deberá usar la hora del recibo Git al vincular la congelación; los hashes de una declaración no sustituyen la comprobación de los archivos de pesos y calibradores.

El selector HPO registrado exige las dieciocho oportunidades identificadas, completadas o fallidas con motivo. Una pausa con un registro parcial no produce un ganador definitivo; los subconjuntos se admiten únicamente como fixtures sintéticas explícitas. Las comprobaciones de calendario y criterios no acreditan por sí solas todos los recibos de ejecución.

- [Protocolo estructurado](../_static/protocol_amendment_20261112/protocol.json) y [registro de la enmienda](../_static/protocol_amendment_20261112/registration.json).
- [Calendario común descargable](../_static/protocol_amendment_20261112/calendar.csv).
- [Exposición y roles de datos](../_static/protocol_amendment_20261112/exposure_registry.json).
- [Política de selección y presupuesto](../_static/protocol_amendment_20261112/selection_budget_policy.json).
- [Verificación del protocolo y sus límites](../_static/protocol_amendment_20261112/verification.json).
- Código: [calendario](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_amended_calendar.py), [separación de datos](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_amended_data_guard.py), [métricas](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_amended_metrics.py) y [registro](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_protocol_amendment.py).

| Pendiente | Evidencia para darlo por cerrado |
|---|---|
| Migrar el controlador GPU | Plan versionado aplicado, identidad de recetas/arquitecturas y máscara común verificadas, sin dos controladores activos |
| Conciliar presupuesto y caché | Ambos diarios y reservas conciliados; saldo derivado del techo acumulado, resultados compatibles identificados e intentos inciertos preservados |
| Fijar k y completar NAS TiDE | Piloto de ruido, finalistas/referencia congelados, ambas fases completas y análisis del conjunto indistinguible/sesgo |
| Completar HPO y reajustes | Dieciocho oportunidades por red/ancla, comparadores comunes, fallos y costes íntegros, uso causal de cada receta |
| Fijar R y demostrar su alcance | Piloto pareado, bloques de 14 días y veinte contrastes Holm; alternativa 2δ, límite Wilson y potencia/MDE con sus límites reportados |
| Congelar y materializar la prueba prospectiva | Recibo global y vinculación antes de T, comienzo exacto al día siguiente, artefactos/hash, puente adquirido y acceso causal auditados |
| Completar evaluación y guía | Tablas con métricas/IC/costes, inferencia admisible, robustez, incertidumbre, explicabilidad, ablaciones, errores y reproducción con evidencia |

Los pendientes se revisarán hasta completar sus criterios; no se sustituyen por casillas marcadas, contador de corridas o registro documental. Los capítulos y descargas anteriores conservan sus bytes. Esta enmienda no inicia entrenamiento de mercado, no modifica la sesión de Colab y no acredita por sí sola cumplimiento integral de E2.
