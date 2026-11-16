# Protocolo de evaluación del Bloque D — borrador de trabajo

Fecha de elaboración: 16 de noviembre de 2026. Versión documental: borrador 0.1.

**Estado: NO CONGELADO. No es un preregistro ni autoriza una campaña de cómputo.** No existe tag confirmatorio de este documento. Las casillas abiertas deben resolverse y revisarse antes de registrar `protocolo.md`, su fecha, su tag y las huellas de los insumos. No se han ejecutado nuevos ajustes, análisis de potencia ni contrastes del candidato sobre mercado al elaborar este borrador.

## 1. Compromiso y procedencia

Se conserva el estándar teórico de la sección 2 del E3: propiedad formal y verificación numérica; evaluación amplia; familia completa; al menos tres pruebas válidas por contraste; potencia por contraste de al menos 0,90; potencia conjunta reportada; al menos diez semillas o el número mayor requerido por el dimensionamiento. Falta formalizar `experiments/estandar.md` con fecha y tag anteriores al cierre del protocolo. La lista de requisitos no certifica su cumplimiento.

El problema heredado es BTCUSDT, ventana D7×5 OHLCV, objetivo σ7 poblacional de siete retornos logarítmicos, sin anualización, horizonte h=1, ruta F2a. Las 608 fechas de enero de 2025 a agosto de 2026 y la reserva histórica de septiembre–octubre de 2026 tienen exposición previa. No se las vuelve a denominar prueba ciega. Falta fijar una separación efectiva entre desarrollo y confirmación y documentar qué afirmaciones permiten los datos disponibles.

Fuentes de lectura: [ruta temporal de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/07_ruta_temporal.html), [auditorías posteriores de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/64_correcciones_metodologicas.html), [diseño riguroso de E3](https://jassan-bit.github.io/DeepLearning_202630/entregable3/sections/07_diseno_riguroso.html). La A2 de E2 conserva SHA-256 `879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1`.

El candidato conserva CNN causal → LSTM unidireccional y añade una regla de atención con penalización de discordancia respecto a una referencia causal fija y saturación radial de valores. Sus antecedentes, la identidad pseudo-Huber de la saturación y las diferencias todavía por acreditar están documentados en la [sección 6](https://jassan-bit.github.io/DeepLearning_202630/entregable3/sections/06_modelo_original.html). No se acredita un origen enteramente nuevo ni una transferencia validada sólo por escribir el módulo. La revisión de fuentes y la justificación de cada adaptación deben cerrarse antes del protocolo.

La bitácora actual declara Nd,mercado=0 y cuatro variantes funcionales comprobadas con datos artificiales. El Nd total y el esfuerzo acumulado, incluidos intentos y fallos, aún deben conciliarse; la prueba diminuta de 350 pasos sobre ocho ejemplos artificiales no se renombra como validación de mercado. Fuente: `experiments/e3_bloque_c_20261115/design_log.md`.

## 2. Métricas, hipótesis y márgenes

La métrica primaria heredada es la **media de las RMSE individuales de las ejecuciones**, en unidades originales de σ7. No se sustituye por la RMSE de un ensemble o por la raíz del MSE agregado. Para un rival j, la orientación de la ganancia es Δj = media(RMSEj,r − RMSEoriginal,r); positivo favorece al candidato. La hipótesis confirmatoria propuesta es H0j: Δj ≤ δ frente a H1j: Δj > δ, con **δ = 0,00025772842664981106** para el problema BTC heredado.

El margen de brecha δg heredado tiene el mismo valor numérico, pero otro significado: g = RMSEvalidación − RMSEtrain, con riesgos comparables. La dirección es H0: θg ≥ δg frente a H1: θg < δg; no una prueba de igualdad g=0. Debe fijarse si θg es la media o la mediana y su unidad de incertidumbre; el resultado de una no acredita automáticamente la otra. La equivalencia requiere TOST o ROPE y su margen propio previamente declarado, no un p-valor no significativo.

Métricas secundarias propuestas: MAE, R², calibración puntual, cobertura y ancho de intervalos, y costes de entrenamiento/inferencia. Su catálogo, niveles, estimandos y familias todavía deben cerrarse. Se incluirá una regla probabilística adecuada, por ejemplo pinball para los cuantiles realmente disponibles. CRPS y PIT sólo se reportarán si se obtiene una distribución predictiva adecuada; no pueden reconstruirse de un único intervalo del 90 %.

La hipótesis de mecanismo y las predicciones secundarias de la sección 7 son propuestas de desarrollo. Deben fijarse las perturbaciones, severidades, subconjuntos, umbrales de cola y criterios de éxito/fracaso antes de la evaluación correspondiente. Los análisis no previstos y las desviaciones posteriores se registrarán como exploratorios y no sustentarán por sí solos R3.

## 3. Familia completa de comparación

El inventario actual conserva **16 rivales**, m=16, sin reducirlo según el desempeño observado:

| Grupo heredado | Rivales |
|---|---|
| Top 10 de E2 | FCN-1D; ResNet-1D; Inception-1D; LSTM-autoencoder con predictor; TimesNet; TSMixer; TiDE; TimeMixer; PatchTST; iTransformer. |
| Referencias neuronales de E1 | MLP16; LSTM32; GRU32. |
| Comparadores clásicos | Persistence; SeasonalNaive; ARIMA. |

El roster procede de `experiments/e3_bloques_ab_20261115/master_configurations.csv`. Falta cerrar el Bloque A y vincular las recetas y artefactos confirmatorios a hashes verificables. Las recetas se congelan; completar las repeticiones previstas antes del protocolo no permite retunear los rivales después de observar la comparación final. Las seis semillas históricas no completan el mínimo de diez de E3.

## 4. Particiones, unidad de análisis y semillas

Pendientes: archivos admisibles de datos y versiones; fechas de desarrollo/confirmación y de cada conjunto adicional; reglas de corte y purga por el soporte de ventanas y objetivos; disponibilidad temporal de todas las covariables; preprocesamiento ajustado sólo en entrenamiento; política única de reentrenamiento, calibración y selección; auditoría de fuga aplicable y resultados.

Se usarán las mismas particiones, preprocesamiento y semillas pareadas entre modelos cuando exista aleatoriedad. Se declarará una lista común de al menos diez semillas independientes o más según potencia, los fallos y su tratamiento antes de confirmar. **La lista definitiva no está fijada.** Los clásicos deterministas no generan independencia por copiar diez veces la misma trayectoria. La variación del candidato contra un clásico y la incertidumbre del calendario deben representarse sin inventar ajustes clásicos.

Los días con objetivos solapados no son casos iid; las semillas sobre el mismo calendario no son mercados independientes. El diseño distinguirá bloques temporales sincronizados, trayectorias e inicializaciones. El remuestreo conservará el emparejamiento entre modelos y la dependencia temporal; longitudes de bloque, réplicas, generadores y análisis de sensibilidad deben quedar definidos antes del cálculo final.

Si hay medidas repetidas por unidad se justificará un modelo de efectos mixtos con efectos de modelo, unidad y semilla, o se documentará su inaplicabilidad. Un intercepto aleatorio no elimina por sí solo la autocorrelación: su estructura residual y la inferencia deben respetarla. La sensibilidad de la conclusión incluirá pruebas paramétrica/no paramétrica pertinentes, otra métrica secundaria registrada y un subconjunto razonable de semillas fijado antes de ver resultados. Se justificará si Nadeau–Bengio aplica y, en ese caso, se compararán las decisiones con y sin la corrección; si no aplica, se documentará el motivo sin ejecutarlo como trámite.

## 5. Tres pruebas, multiplicidad y decisión R3

Falta cerrar una batería de **al menos tres pruebas distintas admisibles de la Tabla 4 y justificadas para F2a por rival**, con hipótesis, estadísticos, supuestos, estimandos y ámbito de incertidumbre explícitos. El bootstrap temporal pareado de la ganancia de RMSE es candidato natural. La t pareada y la aleatorización de signos condicionadas al calendario necesitan justificar tanto su admisibilidad para la ruta como independencia/normalidad o simetría de las ganancias, respectivamente; no se cuentan como tres confirmaciones temporales homogéneas por su mera presencia.

DM–HLN, Giacomini–White y MCS son opciones F2a sujetas a sus supuestos. Una prueba de igualdad del MSE diario no contrasta automáticamente una ganancia de RMSE mayor que δ. No se traslada ese margen directamente a pérdidas cuadráticas. Clark–West sólo procede con anidamiento pertinente; Nadeau–Bengio corresponde a diseños de particiones solapadas que lo justifiquen, no se aplica genéricamente a días o semillas. Cambiar la longitud del bloque, B o la semilla del remuestreo no crea otra prueba.

La propuesta de control es Holm a α=0,05 sobre los 16 rivales para cada procedimiento previamente definido. Para cada rival se toma el **máximo de los p-valores ajustados** de las pruebas exigidas: todas deben rechazar y tener orientación favorable. La afirmación «supera a todos» requiere que todos los rivales satisfagan la regla, una unión-intersección; ningún fallo o resultado no concluyente se elimina de la familia. No se usa el menor p-valor como voto ganador. Deben fijarse las reglas correspondientes si un procedimiento global se integra en la batería. MCS tendrá nivel previamente declarado y no aporta por sí solo dirección de superioridad ni relevancia δ.

Familias secundarias separadas y previamente enumeradas: conjuntos adicionales; ablaciones; subgrupos/regímenes; sensibilidades/robustez; adecuación/calibración. Se reportan tamaños, hipótesis, p crudos y ajustados de todas ellas. No se mezclan ni se fraccionan después de observar resultados para relajar la decisión principal.

R3 exige simultáneamente: (1) mejor media que cada rival; (2) batería válida con control familiar y ganancia mayor que δ; (3) efecto no despreciable según el estimador aplicable, con umbral fijado —la guía cita |δCliff| ≥ 0,147 o |dCohen| ≥ 0,2 cuando corresponda— e IC95 % de ganancia con límite inferior positivo, idealmente mayor que δ; (4) mejor media en al menos dos de los tres conjuntos adicionales y ausencia de inferioridad significativa frente a cualquier rival en cualquiera; (5) adecuación del propio modelo —baseline trivial, brecha δg, calibración, residuos y coherencia validación/prueba—; (6) conclusión estable con la prueba alternativa y sensibilidades predefinidas. No rechazar inferioridad no demuestra equivalencia ni inocuidad. La elección del tamaño de efecto respetará el emparejamiento y la unidad independiente.

## 6. Potencia y presupuesto

Se requiere piloto de desarrollo con al menos cinco semillas por par, todavía no ejecutado. Se dimensionarán R y el calendario teniendo en cuenta el rival más exigente, la batería completa, el ajuste de multiplicidad y la dependencia. El mínimo R≥10 no garantiza potencia. El objetivo es al menos 0,90 por contraste con la regla completa; la potencia conjunta de superar a toda la familia se calculará y reportará bajo dependencia, no como producto de potencias marginales salvo independencia justificada. Se informarán supuestos, MDE e incertidumbre del piloto.

La fórmula normal con α'=0,05/m sirve sólo de orientación inicial, no certifica la potencia de una batería con Holm, bootstrap y varias pruebas. Para H0:Δ≤δ la potencia se dimensiona a una alternativa Δalt>δ; la distancia relevante es Δalt−δ. Deben fijarse esa alternativa y el método de simulación/estimación. No se confunde detectar una diferencia δ frente a cero con rechazar un margen δ.

Pendientes antes de congelar: número de configuraciones, espacio relativo, máximo de épocas, parada temprana, semillas HPO, reglas de fallo, límite de tiempo, hardware y coste real por modelo. Se declara si el presupuesto se mide en intentos, tiempo o ambos; si el candidato cuesta más por intento, se iguala tiempo total o se reportan ambos análisis. El esfuerzo de diseño se reporta separado de HPO. El tamaño de parámetros/FLOPs y el control de capacidad se documentan; cada ablación que cambie capacidad exige su variante de tamaño igualado conforme a la sección 7.4.

No se asignan horas nuevas ni se reinicia el presupuesto acumulado de E2 con este documento. Cualquier ejecución científica futura debe tener un alcance y una contabilidad concretos antes de comenzar.

## 7. Evaluación ampliada

Se elegirán al menos tres conjuntos públicos de la misma tarea de pronóstico de volatilidad y con la misma métrica primaria. **Ninguno está seleccionado todavía.** Cada uno debe diferir del principal en al menos dos ejes documentados —tamaño, dominio, ruido, desbalance o dimensionalidad—, con fuente, licencia, versión, hash y criterio de selección registrados antes de ver el desempeño.

Todos los 16 rivales y el candidato se evalúan en cada conjunto sin excepciones, con presupuesto comparable y auditoría de fuga. Se fijarán las unidades y la justificación de los márgenes prácticos para las escalas adicionales sin recalibrarlos según resultados. Se reportan resultados por conjunto; no se promedian RMSE de escalas distintas. La consolidación prevista usa rangos, Friedman con Iman–Davenport y pares con Holm, o una alternativa bayesiana con ROPE previamente elegida; no se elige la más favorable después. Tres conjuntos ofrecen potencia global limitada, y series del mismo mercado pueden compartir dependencia.

Por conjunto se informarán la diferencia, su IC y tamaño de efecto, y la probabilidad de superioridad con orientación P(RMSEoriginal < RMSErival), definida sobre las unidades de prueba declaradas y acompañada de su incertidumbre. No es la probabilidad posterior de que una hipótesis sea verdadera ni permite convertir días o semillas correlacionados en nuevas réplicas independientes.

## 8. Robustez, adecuación e informes

La matriz mínima de estudios incluye sensibilidad a hiperparámetros propios, estabilidad entre semillas y particiones, perturbaciones a tres severidades, curva de tamaño de muestra, escalabilidad, cambio de distribución fuera del diseño, calibración/reglas de puntuación, cobertura por subgrupos, eficiencia/frontera de Pareto y análisis de errores compartidos. Se aplicará el mismo procedimiento a toda la familia. Las perturbaciones plausibles preservarán coherencia OHLCV y disponibilidad temporal; el estrés artificial retrospectivo se identifica por separado.

La estabilidad incluirá distribuciones, IC y coeficiente de variación cuando sea interpretable —es inestable si la media se aproxima a cero—. ANOVA o Kruskal–Wallis requieren unidades y supuestos válidos; no se aplican a semillas×días como si fueran observaciones independientes. Las curvas de tamaño de muestra emplearán fracciones temporales admisibles, al menos 10 %, 25 %, 50 % y 100 % cuando el soporte de la tarea lo permita, con varias semillas y costes registrados; cualquier imposibilidad se declara sin contar un estudio no ejecutado.

Para F2a, los diagnósticos candidatos incluyen Mincer–Zarnowitz con dependencia, Ljung–Box y ARCH-LM, BDS/Jarque–Bera cuando proceda, brechas comparables y estabilidad, y cobertura/ancho con IC temporales. Las pruebas de clasificación o bloques espaciales no se importan por defecto. La aceptación, alternativas y límites de cada diagnóstico deben fijarse antes de la evaluación. No rechazar un diagnóstico no demuestra su nula.

Se producirán tablas de métricas media±DE e IC temporales, pruebas/hipótesis/supuestos/estadísticos/p crudos/p ajustados/efectos/IC/conclusión, MCS con candidato, resultados por conjunto y consolidación, Tablas A y B del candidato, potencia/MDE/conjunta, cómputo y todos los estudios de robustez. La DE entre semillas y el IC temporal tendrán etiquetas diferentes. Hardware, precisión, hilos, lotes y backend de latencia/memoria se mantendrán comparables. Se reportarán resultados adversos y no concluyentes; ninguna casilla pendiente se representa con ceros o p-valores ficticios.

## 9. Condiciones para convertir este borrador en protocolo registrado

- Cerrar A y las recetas confirmatorias; auditar datos y vincular artefactos/hashes.
- Cerrar mecanismo, fuente y adaptación, hipótesis D1–D5, variantes y esfuerzo Nd total.
- Documentar separación desarrollo/confirmación y exposición histórica; elegir los tres conjuntos con sus criterios antes de evaluarlos.
- Fijar métricas, márgenes por escala, unidad, familia, tres pruebas válidas, supuestos, intervalos y decisión completa R3.
- Completar piloto, potencia por contraste/conjunta y número final de semillas; fijar sus identificadores y fallos admisibles.
- Fijar presupuestos, equidad, determinismo, ambiente, reentrenamiento, calibración y reproducibilidad.
- Registrar `experiments/estandar.md` y después `protocolo.md`, con fecha y tags Git reales previos a los experimentos definitivos; conservar este borrador y el historial.

Cada desviación posterior se declara con motivo y efecto sobre la interpretación. Los cambios de arquitectura, pérdida o espacio de búsqueda después del cierre generan una nueva versión del diseño y del protocolo, sin retrofecharlos ni borrar intentos. Al corte de este documento R1, la defensa académica de R2 y R3 siguen sin acreditarse.
