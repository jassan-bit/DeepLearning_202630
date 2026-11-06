# 13. Agregación estadística e interpretación

**Estado:** reglas registradas y código probado; el diagnóstico P0/OOF se informa en la [sección 9](09_protocolo_registrado.md). Este capítulo aplica la separación de niveles de §13 de la guía. Sus reglas no convierten el test histórico conocido en una evaluación confirmatoria nueva.

## 13.1 Qué se estima y qué se remuestrea

Para cada modelo $m$ y semilla $s$ se calcula $\operatorname{RMSE}_{ms}=\sqrt{N^{-1}\sum_t(y_t-\widehat y_{mst})^2}$. La métrica primaria es la **media de esas RMSE** entre semillas. La RMSE de un ensamble de pronósticos es otro estimando y se identifica por separado. Tampoco se promedian sin ponderación diez RMSE mensuales para sustituir la pérdida sobre todas las fechas elegibles.

| Unidad | Función | Límite |
|---|---|---|
| Semilla emparejada | Variación de inicialización/optimización con calendario fijo | No es otra trayectoria de mercado |
| Origen diario | Error predictivo sobre σ₇ | Etiquetas, covariables y regímenes dependen temporalmente |
| Fold mensual forward | Predictor congelado durante el mes, entrenado solo con historia admisible | Los folds comparten entrenamiento; no son diez datasets independientes |
| Serie BTCUSDT | Dominio de aplicación de este experimento | No permite estimar varianza entre activos, bolsas o mercados |

La receta GRU/D7-c03 ya fue seleccionada en E1 con desarrollo de 2024. Reajustarla con semillas nuevas y cortes causales no elimina esa selección histórica. OOF identifica la separación de los ajustes de pesos de esta etapa; no acredita un desarrollo nunca observado ni una estimación libre de toda selección previa.

Los modelos se comparan sobre las mismas fechas y semillas registradas. Las referencias deterministas se almacenan una vez; su repetición aritmética para emparejar una tabla no genera observaciones independientes ni una DE de entrenamiento estimada. Los faltantes mantienen su posición en el calendario y se excluyen de la métrica únicamente mediante la máscara común auditada.

## 13.2 Inferencia del diagnóstico P0

El análisis de P0 usa un panel modelo×semilla×fecha y bootstrap circular sobre **bloques del calendario**, con índices comunes entre modelos. Se conserva el emparejamiento y la pertenencia al régimen se filtra después de remuestrear; no se concatenan días de alta volatilidad separados como si fueran contiguos. El bloque principal es de siete días; catorce y veintiocho días son sensibilidades registradas, con **4.999** réplicas y semilla Monte Carlo 20261106. La elección de bloques para el benchmark definitivo deberá justificarse en desarrollo antes de su scoring.

El gap registrado contrasta, para TimeMixer, iTransformer y PatchTST, $H_0:G_m\leq\delta$ frente a $H_1:G_m>\delta$, donde $G_m$ es la media entre semillas de RMSE en alta volatilidad menos RMSE en el resto. Se utiliza exclusivamente OOF enero–octubre de 2024, Holm sobre tres hipótesis y **α=0,01**. Noviembre–diciembre aporta descripción y variación de entrenamiento; no se abre otra oportunidad para rechazar el mismo gap. El estadístico bajo la nula se centra en su frontera δ; no se transforma un contraste contra cero en una afirmación de mejora práctica.

La clasificación usa σ₇(t−1) y un umbral q₇₅=**0,024247163097617024** calculado en entrenamiento de 2023. Esa variable contextual no se añade como entrada de las redes. Si un estrato queda vacío, el resultado es no estimable. La dependencia, no estacionariedad y la escasez de episodios limitan la precisión aunque existan cientos de filas.

Los intervalos descriptivos nominales al 95 % no se presentan como simultáneos ni sustituyen los p ajustados. Una diferencia entre regímenes no demuestra una causa arquitectónica, una solución transferible ni superioridad global. Un gap no demostrado se informa como tal; no se escoge después otro calendario, otra receta o un umbral más favorable.

## 13.3 Potencia condicional y dimensionamiento definitivo

La proyección P0 de semillas usa las cinco ganancias RMSE emparejadas frente a GRU sobre un calendario fijo. Emplea t no central con alternativa de planificación **2δ**, nula en δ y α conservador **0,02/30**. La incertidumbre de su DE se muestra con cuatro grados de libertad, junto a proyecciones optimista y conservadora. Son aproximaciones condicionales bajo supuestos explícitos, no potencia temporal ni potencia del procedimiento final.

P1 solo procede tras seleccionar y congelar las recetas finales mediante HPO/NAS aplicable. La simulación registrada debe incorporar las familias completas, dependencia temporal y la decisión conjunta de RQ3. Se exige que el límite inferior unilateral de Wilson al 95 % de la potencia simulada alcance **0,80**. Si ninguna opción de R hasta 200 lo logra, se informarán el límite, la potencia estimada y el efecto mínimo detectable; no se certificará suficiencia por haber ejecutado un número redondo de semillas.

La variación entre modelos y semillas se reportará por separado. Un modelo mixto o descomposición de varianza que se ejecute deberá declarar efectos, identificabilidad y estimador; no se atribuyen componentes de varianza inexistentes a una única serie. Las pérdidas diarias y los folds con entrenamiento solapado requieren correcciones apropiadas o una justificación explícita de la estrategia temporal adoptada.

## 13.4 Familias finales y diagnósticos

El [preregistro](../_static/remediation_protocol/preregistration.json) fija treinta hipótesis frente al GRU original con α=0,02, treinta frente a GRU-E2 con α=0,02 y 199 diagnósticos con α=0,01. Holm se aplica dentro de cada familia completa y la suma de asignaciones es 0,05. La familia diagnóstica P0 es una etapa separada: no se afirma un control conjunto retrospectivo entre P0 y el benchmark final.

La [sección 6](06_benchmark_final.md) conserva los requisitos de DM con corrección HLN, Ljung–Box, BDS, sobreajuste, bondad de ajuste y comparación de eficiencia. Los diagnósticos no se usan para elegir después el contraste que resulte significativo. La ausencia de rechazo no prueba equivalencia, y un diagrama de diferencia crítica o Friedman no trata las semillas sobre el mismo mercado como datasets independientes. Para una afirmación de equivalencia se necesita el contraste registrado correspondiente y un intervalo compatible con su margen.

Para RQ3 se exige simultáneamente mejora predictiva y menor coste: el contraste de intersección-unión conserva el mayor p de ambos componentes, antes del ajuste familiar. Los tiempos batch 1 se miden con calentamiento, bloques y orden intercalado en el mismo hardware; no se mezclan con la variabilidad de la RMSE.

## 13.5 Evidencia verificable

`scripts/analyze_e2_remediation_pilot.py` verifica identidad del plan, semillas, cohortes, hashes de checkpoints, cierre de todos los ajustes antes del scoring y soportes de cada fold. Consume predicciones privadas cerradas y escribe únicamente agregados y procedencia. Sus trece pruebas usan casos sintéticos; los resultados de Bitcoin se publican por separado después de ejecutar el protocolo.

El [paquete actual](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip) conserva ese código, su diseño y los agregados disponibles. La [sección 12](12_incertidumbre_errores.md) especifica otra función distinta: intervalos predictivos de σ₇ con calibración disjunta y análisis de cien casos únicos por modelo. Un IC de RMSE o la dispersión entre semillas no sustituye esa incertidumbre predictiva.
