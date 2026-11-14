# 13. Agregación estadística e interpretación

**Estado vigente al 13/11/2026:** el [benchmark de §6](06_benchmark_final.md) cerró R=6 para trece redes, tres comparadores deterministas y veinte contrastes primarios. La [auditoría F2a de §49](49_diagnosticos_f2a.md) añade una familia exploratoria separada de 826 hipótesis; no altera la familia primaria ni convierte las semillas en mercados independientes. [§50](50_comparaciones_f2a.md) aporta DM–HLN, MCS y rangos condicionados, con hipótesis distintas de las primarias y aplicabilidad limitada al calendario observado. Los estados históricos conservados debajo no describen el cierre vigente.

**Vigencia documental (11/11/2026):** existe selección HPO/NAS registrada y ejecutada para el catálogo actual; las afirmaciones de ausencia de HPO o de falta de registro de selección corresponden a cortes anteriores. El benchmark final, R y su cierre inferencial siguen pendientes. [Registro y alcance actuales (§9)](09_protocolo_registrado.md) · [Calendario (§40)](40_calendario_original.md) · [Corte de resultados (§41)](41_avance_optimizacion_11_origenes.md) · [Auditoría y trabajo paralelo (§42)](42_auditoria_y_trabajo_paralelo.md).

**Estado:** la nueva enmienda estadística para los diez modelos y CNN-1D + LSTM + atención temporal como candidato 11 está pendiente. MLP, LSTM y GRU de E1 son referencias separadas. El [diagnóstico P0 ejecutado](09_protocolo_registrado.md) conserva su identidad histórica; sus reglas y resultados no se convierten en evidencia del catálogo vigente ni en superioridad del híbrido.

## 13.1 Qué se estima y qué se remuestrea

Para cada modelo $m$ y semilla $s$ se calcula $\operatorname{RMSE}_{ms}=\sqrt{N^{-1}\sum_t(y_t-\widehat y_{mst})^2}$. La métrica primaria es la **media de esas RMSE** entre semillas. La RMSE de un ensamble de pronósticos es otro estimando y se identifica por separado. Tampoco se promedian sin ponderación RMSE mensuales para sustituir la pérdida sobre todas las fechas elegibles.

| Unidad | Función | Límite |
|---|---|---|
| Semilla emparejada | Variación de inicialización/optimización con calendario fijo | No es otra trayectoria de mercado |
| Origen diario | Error predictivo sobre σ₇ | Etiquetas, covariables y regímenes dependen temporalmente |
| Fold mensual forward | Predictor congelado durante el mes, entrenado solo con historia admisible | Los folds comparten entrenamiento; no son datasets independientes |
| Serie BTCUSDT | Dominio de aplicación de este experimento | No permite estimar varianza entre activos, bolsas o mercados |

La nueva enmienda debe identificar la selección temporal de cada receta. Si una referencia histórica utilizó resultados posteriores a alguno de sus orígenes OOF, esa exposición debe declararse; no se le atribuye selección anidada causal por cambiar el nombre o ejecutar semillas nuevas. La identidad de las referencias E1, sus cohortes y sus métricas se conserva por separado.

Los modelos se comparan sobre fechas comunes y con tratamiento explícito de las semillas. Si un control es determinista, repetir aritméticamente su trayectoria no genera observaciones independientes ni una DE de entrenamiento estimada. Los faltantes mantienen su posición en el calendario y se excluyen de la métrica únicamente mediante una máscara común auditada. El híbrido usa las mismas entradas D7×5; cualquier cambio de información exige declararlo antes de evaluar.

## 13.2 Inferencia del diagnóstico P0 histórico

Este procedimiento pertenece al registro anterior; no es la inferencia registrada para el catálogo vigente. El análisis de P0 usa un panel modelo×semilla×fecha y bootstrap circular sobre **bloques del calendario**, con índices comunes entre modelos. Se conserva el emparejamiento y la pertenencia al régimen se filtra después de remuestrear; no se concatenan días de alta volatilidad separados como si fueran contiguos. El bloque principal es de siete días; catorce y veintiocho días son sensibilidades registradas, con **4.999** réplicas y semilla Monte Carlo 20261106. La elección de bloques para el benchmark definitivo deberá justificarse en desarrollo antes de su scoring.

El gap registrado contrasta, para TimeMixer, iTransformer y PatchTST, $H_0:G_m\leq\delta$ frente a $H_1:G_m>\delta$, donde $G_m$ es la media entre semillas de RMSE en alta volatilidad menos RMSE en el resto. Se utiliza exclusivamente OOF enero–octubre de 2024, Holm sobre tres hipótesis y **α=0,01**. Noviembre–diciembre aporta descripción y variación de entrenamiento; no se abre otra oportunidad para rechazar el mismo gap. El estadístico bajo la nula se centra en su frontera δ; no se transforma un contraste contra cero en una afirmación de mejora práctica.

La clasificación usa σ₇(t−1) y un umbral q₇₅=**0,024247163097617024** calculado en entrenamiento de 2023. Esa variable contextual no se añade como entrada de las redes. Si un estrato queda vacío, el resultado es no estimable. La dependencia, no estacionariedad y la escasez de episodios limitan la precisión aunque existan cientos de filas.

Los intervalos descriptivos nominales al 95 % no se presentan como simultáneos ni sustituyen los p ajustados. Una diferencia entre regímenes no demuestra una causa arquitectónica, una solución transferible ni superioridad global. Un gap no demostrado se informa como tal; no se escoge después otro calendario, otra receta o un umbral más favorable.

## 13.3 Potencia condicional y dimensionamiento nuevo

Las proyecciones de semillas del P0 se hicieron con ganancias emparejadas sobre un calendario fijo. Su método y supuestos se conservan en el [paquete histórico](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip). Esas aproximaciones condicionales no estiman automáticamente potencia temporal ni dimensionan los modelos incorporados ahora.

El dimensionamiento nuevo requiere recetas congeladas, comparadores definidos y efectos de planificación explícitos respecto a las fronteras de las hipótesis. Debe distinguir capacidad entre semillas y capacidad temporal, incorporar las familias completas y, cuando corresponda, la decisión conjunta de error y coste. Las simulaciones deberán declarar sus supuestos, controlar el error bajo la nula y cuantificar la incertidumbre de la potencia estimada.

**No hay un R ni un presupuesto experimental nuevo fijados en esta actualización.** No se heredan tamaños de piloto, rejillas de R o asignaciones de cómputo del catálogo sustituido. Si el diseño no alcanza la potencia buscada, se informarán su límite y el efecto detectable; un número redondo de semillas no certifica suficiencia.

La variación entre modelos y semillas se reportará por separado. Un modelo mixto o descomposición de varianza deberá declarar efectos, identificabilidad y estimador; no se atribuyen componentes de varianza entre mercados a una única serie. Las pérdidas diarias y los folds con entrenamiento solapado requieren una estrategia temporal justificada.

## 13.4 Familias de hipótesis y diagnósticos

El [preregistro anterior](../_static/remediation_protocol/preregistration.json) y la [enmienda archivada](../_static/completion_protocol/analysis_additions.json) permanecen accesibles para verificar su procedencia. **Sus recuentos de hipótesis y asignaciones de error no están vigentes para la nueva selección.** La nueva enmienda deberá fijar cada comparación, su referencia, pertenencia a familia y nivel de error antes de observar los resultados relevantes.

La [sección 6](06_benchmark_final.md) describe el papel de las comparaciones predictivas, eficiencia y diagnósticos residuales. Un contraste DM requiere declarar horizonte, dependencia y, si corresponde, corrección HLN. Ljung–Box o BDS no se usan para escoger después el contraste que resulte significativo. Un rechazo de iid no identifica por sí solo no linealidad.

Para afirmar mejora simultánea de error y coste se necesita una prueba de intersección-unión: su p-valor conjunto conserva el mayor p válido de ambos componentes antes del ajuste familiar. La latencia se mide con calentamiento, bloques y orden intercalado en el mismo hardware; sus repeticiones no se mezclan con la variabilidad de la RMSE.

La ausencia de rechazo no prueba equivalencia ni ausencia de sobreajuste. Una afirmación de equivalencia necesita un margen justificado, un contraste registrado y un intervalo compatible con ese margen. Un diagrama de diferencia crítica o Friedman no convierte semillas del mismo mercado en datasets independientes.

### 13.4.1 Análisis adicionales con alcance propio

La brecha train–validación debe usar métricas y soportes comparables. No rechazar un exceso de brecha es inconcluso, no una certificación de generalización. Las pruebas de equivalencia, PIT, ablaciones, controles negativos y diagnósticos de distribución necesitan objetivos y familias propios; no quedan incluidos automáticamente en el ajuste de las preguntas principales.

Las ablaciones de CNN-1D + LSTM + atención temporal deberán aislar el aporte de convolución, memoria y atención con soporte D7×5 idéntico. Son controles del candidato 11, no nuevas incorporaciones al catálogo. Los resultados desfavorables se conservan y no se sustituye la receta después de evaluar para sostener una superioridad presupuesta.

La probabilidad de superioridad, un contraste HAC alternativo o una frontera de Pareto deben conservar su alcance condicionado y supuestos. Un IC de calendario sobre pesos fijos no incluye automáticamente incertidumbre de HPO o entre activos. Una referencia determinista tampoco adquiere variación de inicialización por duplicar sus filas.

## 13.5 Evidencia verificable

`scripts/analyze_e2_remediation_pilot.py` y el [paquete P0](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip) conservan el procedimiento histórico, sus comprobaciones de identidad, semillas, cohortes, hashes y cierre antes del scoring. Los resultados P0 se enlazan en §9. Las pruebas sintéticas acreditan comportamiento del software, no resultados nuevos de Bitcoin.

La [implementación estadística archivada](../_static/completion_download/completion_bundle.zip) conserva `e2_final_statistics.py` y su runner sin modificarlos. No es una implementación validada del nuevo catálogo por el hecho de seguir disponible. La nueva enmienda y cualquier adaptación necesitarán validación y recibos propios antes de afirmar resultados.

La [sección 12](12_incertidumbre_errores.md) distingue intervalos predictivos de σ₇, IC de métricas y análisis de casos únicos. Un IC de RMSE o la dispersión entre semillas no sustituyen la incertidumbre predictiva. Reutilizar periodos conocidos mantiene el alcance exploratorio, incluso con multiplicidad controlada.
