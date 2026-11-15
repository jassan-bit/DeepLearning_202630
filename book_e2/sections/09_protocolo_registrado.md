(registros-enmiendas-y-cierre-de-la-ejecucion)=
# 9. Preregistro, Potencia Estadística y Control de Decisiones

**Lectura vigente:** el registro reducido gobierna un benchmark ejecutado de **16 sistemas, 81 predictores y 608 fechas comunes**. Este capítulo aplica los campos de §§9.1–9.3 de la guía y separa lo registrado antes de cada etapa, la congelación anterior al scoring actual y los análisis posteriores. El piloto no alcanzó la potencia objetivo: **R=6 se conservó por recursos**, con la limitación y el efecto mínimo detectable publicados. La existencia de registros, resultados o reproducciones no acredita por sí sola adecuación estadística conjunta.

La métrica, δ, unidad temporal y partición heredadas de E1 se mantienen como exige la guía. El test **01/01/2025–31/08/2026 ya era conocido**; su herencia no obliga a inventar otro holdout ni recupera cegamiento. La [admisión de datos](../_static/e2_benchmark_final_20261113/data_admission.json) y la [congelación](../_static/e2_benchmark_final_20261113/final_freeze.json) acreditan la separación de las etapas de esta campaña, con ese alcance retrospectivo.

La [lista verificable](../_static/e2_repro_prereg_current_v1/guide_checklist.csv), la [trazabilidad](../_static/e2_repro_prereg_current_v1/decision_traceability.csv), el [índice de evidencias](../_static/e2_repro_prereg_current_v1/current_evidence_index.json) y el [manifiesto](../_static/e2_repro_prereg_current_v1/manifest.json) acompañan la lectura. El {download}`índice de esta consolidación <../_static/e2_repro_prereg_current_v1/README.md>` identifica las descargas. La {download}`versión exacta anterior del capítulo 9 <../_static/e2_repro_prereg_current_v1/09_protocolo_registrado_before_update.md>` y el bloque documental al final conservan sus estados, encabezados y enlaces.

(e2-preregistro-vigente)=
## 9.1 Plantilla de preregistro aplicada al protocolo ejecutado

### 9.1.1 Campos, valores y alcance temporal

| Campo de la guía | Valor y evidencia actuales | Cronología y límite |
|---|---|---|
| **Tarea y unidad de independencia** | F2a: pronóstico diario de σ₇ poblacional, h=1, entrada D7×5 OHLCV. Una serie BTCUSDT con etiquetas y ventanas dependientes. [Contrato y RQ](01_punto_partida.md) · [ruta](07_ruta_temporal.md). | Las semillas son inicializaciones del optimizador, no nuevas series. La reclasificación temporal conserva X/Y y la unidad heredada. |
| **Métrica primaria y secundarias** | Media de las RMSE individuales en unidades originales. [A2](../_static/e2_benchmark_final_20261113/benchmark_A2.csv) añade MAE, sMAPE, R², cobertura al 90 % y anchura. | La primaria se hereda de E1; no se sustituye por RMSE de un ensemble. sMAPE no acredita MAPE o MASE. Los suplementos identifican sus métricas y no las promueven retrospectivamente a primaria. |
| **Margen de relevancia δ** | **0,00025772842664981106** unidades de σ₇: tolerancia académica equivalente al 5 % de la RMSE CV inicial de GRU/D7. [§1](01_punto_partida.md) · [linaje](../_static/e2_point1_review_v1/margin_lineage.json). | No se recalcula como el 5 % del nuevo ganador ni representa rentabilidad o utilidad financiera demostradas. |
| **Margen de sobreajuste δg** | **0,00025772842664981106**, explícito en `599a29f`; `ef2be289` registra la dirección de tolerancia. Para RMSE, g=RMSE_validación−RMSE_train; H₀:θg≥δg frente a H₁:θg<δg. [Linaje](../_static/e2_point1_review_v1/margin_lineage.json). | El margen previo existe. θg corresponde a la mediana para Wilcoxon o a la media para t, bajo sus respectivos supuestos. El registro no transfiere automáticamente familias/resultados antiguos ni establece la prueba formal comparable actual. |
| **Familia confirmatoria** | **20 pares:** cada uno de los diez candidatos frente a GRU32 y Persistence; ganancia G=RMSE_referencia−RMSE_candidato, H₀:G≤δ, H₁:G>δ. **Holm20, α familiar=0,05.** [Protocolo reducido](../_static/reduced_protocol_20261112/protocol.json). | RQ1 conserva esa familia. RQ2 y RQ3 son secundarias exploratorias con reglas aclaradas posteriormente; sus pruebas específicas no se presentan como ejecutadas ni como nuevos preregistros históricos. |
| **Pruebas y supuestos** | Bootstrap pareado de semillas/calendario, bloques de 14 días, original B=4.999; t pareada y 64 cambios de signo, con un Holm20 separado por procedimiento. [Motor registrado](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/scripts/e2_reduced_final_analysis.py) · [mapa de los 20 pares](../_static/e2_route_current_v1/contrast_battery_map.csv). | Los tres métodos están ejecutados. La definición concreta quedó congelada antes del scoring actual; no se atribuye a antes de todos los ajustes. t y signos son diagnósticos condicionados al calendario. Normalidad, simetría y validez de bloques no se demuestran por contarlos. |
| **Número de ejecuciones R** | **Seis semillas 271801–271806 por red**, 78 ajustes; tres comparadores deterministas con R=1. Diez ajustes piloto: cinco semillas de iTransformer y cinco de GRU32. [Freeze](../_static/e2_benchmark_final_20261113/final_freeze.json) · [piloto](../_static/e2_benchmark_final_20261113/pilot_power_MDE.json). | R=6 es operativo y acotado por recursos. §9.2 conserva potencia 0,122953 frente al objetivo 0,80. Los clásicos no reciben repeticiones ficticias. |
| **Reglas de exclusión** | Fallos/divergencia se conservan con motivo; una oportunidad fallida no se reemplaza por otra configuración o semilla. Máscaras y soporte común anteriores al scoring. [§48](48_protocolo_reducido.md) · [admisión](../_static/e2_benchmark_final_20261113/data_admission.json). | **234 oportunidades HPO: 231 válidas, dos fallidas y una de desenlace desconocido.** No se descartan fechas por error observado ni se convierten ausencias en ceros. |
| **Reglas de parada y ajuste** | Seis plazas C01–C06 por red, tres folds y semilla HPO 271201; máximo 200 épocas y paciencia 20. La receta minimiza RMSE sobre 249 validaciones concatenadas; empate por ID estable. Final con épocas fijas: techo de la mediana de las tres mejores épocas ganadoras. [§48](48_protocolo_reducido.md). | El dominio y las reglas comunes no prueban igualdad de coste. SAE conserva AE1/AE2/supervisado. Las políticas de recursos y los cortes anteriores tienen fechas propias; no se escogen épocas con el test. |
| **Espacio de arquitectura** | Los trece cuerpos seleccionados y sus adaptaciones están identificados en [§44](44_arquitecturas_publicadas.md) y las [entradas congeladas](../_static/e2_flops_final_audit_20261114/frozen_architecture_inputs.json); TiDE vigente es `TIDE-D7-PUBLISHED-R0-v2`. | El bloqueo inicial de TiDE se cerró con evidencia de fuente antes de sus ajustes. NAS es condicional a cuerpos sin arquitectura definida. Seis plazas HPO no se renombran como 1.000 arquitecturas ni borran obligaciones de variantes propias históricas. |
| **Análisis exploratorios permitidos** | Sensibilidades y diagnósticos identificados en el registro; residuos, DM/MCS, robustez, explicaciones, ablaciones, calibración y errores conservan planes y familias posteriores. [§6](06_benchmark_final.md) · [§7.3](07_ruta_temporal.md). | §9.3 registra su alcance. Un plan previo a su propio cálculo, pero posterior a conocer A2, no los vuelve confirmatorios ni autoriza seleccionar otro modelo final. |
| **Congelamiento de datos** | Entrenamiento final de 655 ejemplos hasta el 31/10/2024; calibración de 53 fechas del 08/11 al 30/12/2024; evaluación de 608 del 01/01/2025 al 31/08/2026. [Admisión](../_static/e2_benchmark_final_20261113/data_admission.json) identifica SHA256 del mercado diario y soporte; el [análisis](../_static/e2_benchmark_final_20261113/benchmark_analysis.json) fija la máscara común. | El freeze vincula pesos, escaladores, calibradores, protocolo y análisis antes de admitir los datos finales actuales. El puente del 31/12 y la historia compartida calibración/test son causales; no implican refit ni independencia. Los hashes no constituyen una auditoría global de todos los archivos privados. |

### 9.1.2 Qué se registró antes de cada etapa

| Registro o transición | Qué acredita | Qué conserva su alcance propio |
|---|---|---|
| **E1 y registros del 06/11/2026** | Contrato heredado y registro explícito de δg en [`599a29f`](https://github.com/jassan-bit/DeepLearning_202630/commit/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0); dirección y familia histórica en [`ef2be289`](https://github.com/jassan-bit/DeepLearning_202630/commit/ef2be289903517e9d6f3b70bd72981990d58479a). | P0 y el catálogo anterior conservan sus recetas, cohortes y decisiones. No acreditan los ajustes ni las pruebas del catálogo reducido por transferencia automática. |
| **Registro reducido del 12/11/2026** | Dieciséis sistemas, seis plazas HPO, tres folds, calendario y R=6 en el [protocolo versionado](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/experiments/e2_reduced_protocol_20261112/protocol.json). Las extensiones de cuerpo y recursos preceden a los trabajos que autorizan. | Los campos iniciales `pending` o `test_battery_implementation_verified=false` describen ese corte de registro, no el resultado de la ejecución posterior. |
| **Congelación antes del scoring actual** | `analysis_frozen=true`, 81 identidades físicas y vínculo al registro concreto de análisis en [final_freeze.json](../_static/e2_benchmark_final_20261113/final_freeze.json). | Es anterior al acceso/scoring actuales; no se presenta como anterior a todos los fits ni como recuperación del test ciego de E1. |
| **Planes posteriores y refinamiento** | El plan [`60cd2bb`](https://github.com/jassan-bit/DeepLearning_202630/commit/60cd2bb23f6da22797e2c3a7bf8225250753f903) precede al refinamiento de 20.000, que cotejó exactamente los veinte resultados originales. [§56](56_inferencia_estabilidad_final.md). | Conserva hipótesis, δ, semillas y Holm; los veinte p ajustados siguen en 1. B=4.999 y B=20.000 son el mismo método con distinta resolución, no dos pruebas. |

El archivo denominado [current_protocol.json](../_static/current_protocol.json) conserva un **protocolo anterior de 20 sistemas y familias con α=0,02**. Su nombre no lo convierte en el ejecutor reducido vigente. El alcance actual se obtiene de §48, el protocolo reducido, el freeze y los recibos finales. Los comandos de P0 en archivos históricos corresponden a aquellos estudios; la reproducción actual entra por [§8](08_reproducibilidad.md) y [§61](61_reproduccion_consolidada.md).

Holm aporta el control múltiple junto a los **tres procedimientos reales por cada par**. No es una cuarta prueba, y la alternativa de la guía «global o con control de comparaciones múltiples» no obliga a añadir otro global por ese solo motivo. Tampoco se exige independencia entre los tres métodos. Se distinguen sus supuestos y fuentes de incertidumbre; los suplementos globales no se suman como confirmaciones retrospectivas del mismo H₀ práctico.

### 9.1.3 RQ, hipótesis y respuesta ante los supuestos

La [especificación de RQ](../_static/e2_point1_review_v1/rq_specification.json) conserva las reglas documentales actuales y su cronología. Las secundarias no heredan automáticamente la potencia o el control familiar de RQ1.

| Pregunta | Hipótesis, procedimiento y estado |
|---|---|
| **RQ1 · Ganancia relevante** | H₀:G≤δ frente a H₁:G>δ, Top 10 frente a GRU32/Persistence, Holm20. Bootstrap pareado de semillas y bloques del calendario; t y signos como diagnósticos condicionados. **Ejecutada, cero rechazos.** |
| **RQ2 · Erosión de ventaja en alta volatilidad** | Frente a GRU32, d=G_normal−G_alta; H₀:d≤δ frente a H₁:d>δ. Régimen causal definido con σ₇(t−1) y umbral de entrenamiento; bootstrap temporal pareado recalculando RMSE por estrato. Si se informan los diez candidatos, Holm10 propio. **Especificación secundaria aclarada después del scoring; contraste no ejecutado bajo esas reglas.** |
| **RQ3 · Mejora conjunta de error y coste** | H₀:(G≤δ) o (c≥0), frente a G>δ y c<0; c es latencia del candidato menos GRU32. Intersección–unión, p conjunto=max(p_ganancia,p_coste), con costes crudos comparables y emparejados; Holm10 si se informan diez candidatos. **No ejecutada; los perfiles agregados y Pareto no la sustituyen.** |

Para RQ1 se declaran dependencia temporal suficientemente corta, longitud de bloque adecuada e intercambiabilidad de trayectorias; los diagnósticos t/signos tienen supuestos adicionales de normalidad aproximada o simetría. Las sensibilidades no seleccionan el bloque que produzca un p favorable. Ante un supuesto no acreditado o soporte insuficiente se limita la interpretación y se declara lo no estimable; no se cambia retrospectivamente la nula, el margen, la familia o el método para declarar éxito. RQ2 requiere soporte temporal por estrato y RQ3 sesiones de latencia comparables; su ausencia no se rellena con otros suplementos.

(e2-potencia-vigente)=
## 9.2 Potencia y número de ejecuciones: resultado publicado

La guía propone un piloto de al menos cinco semillas sobre dos o tres representantes y el efecto estandarizado d=δ/σ̂d. Su fórmula ilustrativa para una **t bilateral** es:

$$
R\approx\left(\frac{z_{1-\alpha'/2}+z_{1-\beta}}{d}\right)^2,
\qquad d=\frac{\delta}{\widehat\sigma_d},\qquad 1-\beta\ge0,80.
$$

El [piloto ejecutado](../_static/e2_benchmark_final_20261113/pilot_power_MDE.json) usa una planificación distinta y explícita: **t no central unilateral**, con frontera nula de ganancia δ y alternativa planificada **2δ**. Por ello, la señal por encima de la nula es 2δ−δ=δ. No se atribuye a la fórmula bilateral anterior la potencia publicada, ni se interpreta ésta como potencia para rechazar en la propia frontera nula.

| Elemento del piloto o de la planificación | Valor publicado o derivación descriptiva |
|---|---|
| Representantes y semillas | iTransformer y GRU32; cinco pares 271701–271705, diez ajustes |
| Fechas de desarrollo | 53 fechas, 08/11–30/12/2024; sus etiquetas no entraron en HPO |
| σ̂d, DE muestral de las ganancias pareadas | **0,00025250638523446184** |
| δ heredado | **0,00025772842664981106** |
| d=δ/σ̂d | **≈1,02068**, cociente descriptivo de los dos valores anteriores; la señal planificada (2δ−δ)/σ̂d coincide numéricamente |
| R operativo / gl de la planificación | **6 / 5** |
| α usado para planificar | **0,0025=0,05/20** |
| Potencia para la ganancia planificada 2δ | **0,1229533091842126**; objetivo **0,80 no alcanzado** |
| Exceso mínimo detectable sobre δ al 80 % | **0,0006209451759122912** |
| Ganancia total mínima al 80 % en esa planificación | **0,0008786736025621022**=δ+exceso mínimo detectable |

El [desglose de diseño de potencia](../_static/e2_repro_prereg_current_v1/power_design_readout.json) distingue los valores originales de las divisiones descriptivas. El piloto publicado fija R=6 y obtiene su MDE; **no publica un R mínimo que alcance 0,80**. No se inventa ese número ni se traslada un R de P0 u otro protocolo. α=0,0025 es la referencia conservadora de planificación; no es una simulación exacta del procedimiento escalonado Holm20 ni una estimación de la potencia del bootstrap temporal primario.

La desviación se estima con sólo **cinco pares y dos representantes**, sobre un calendario de 53 fechas. La aproximación t no central condiciona en esa desviación estimada y en su modelo de ganancias; no incorpora toda su incertidumbre ni garantiza potencia para cada candidato, RQ2/RQ3 o cambios temporales. La ausencia de significancia con esa capacidad no demuestra equivalencia o ausencia de efecto. `automatic_R_increase=false` conserva R=6 después del piloto y del scoring, sin perseguir un p favorable mediante nuevas semillas.

Se distinguen **variación de inicialización, variación por partición y dependencia temporal**. Las seis semillas finales comparten entrenamiento de 655 y las 608 fechas; los tres folds HPO tienen entrenamiento expansivo solapado y sirven para seleccionar una receta. No se multiplican semillas × folds para afirmar 18 réplicas independientes. La inferencia sobre estas trayectorias y este calendario no identifica la variación de un algoritmo entre particiones o mercados; variar ambas fuentes requeriría otro diseño declarado, no renombrar los registros existentes.

### Recursos y regla de parada

El [cierre presupuestario](../_static/e2_reproduction_ablations_20261113/budget_final.json) registra **80 → 84 horas acumuladas**, una ampliación de **cuatro horas**, `budget_reset=false` y cero reservas abiertas. El cargo cerrado es **80,2561354201 horas**, con **13.477,9125 segundos de saldo** en ese corte. Son cargos de reloj de pared acumulados, con historia y gestión; **no GPU-horas activas**, energía o importe monetario. La ampliación autorizada no cambia δ, familias, R o épocas y no convierte su saldo histórico en una orden de ejecutar nuevos trabajos.

La política de fases nuevas fijó **1.800 segundos por fase**, con reservas completas de **1.860 segundos** para un ajuste normal y **5.460** para SAE. Es un techo de admisión, no su duración observada. Los resultados históricos conservan el límite y los cargos con que se obtuvieron, según [§48](48_protocolo_reducido.md).

(e2-exploratorios-vigente)=
## 9.3 Política de hallazgos exploratorios y decisiones

Los análisis no incluidos en la definición previa que gobierna su etapa se identifican como **exploratorios**. Un plan guardado antes de su cálculo mejora la trazabilidad del suplemento, pero no lo vuelve anterior al conocimiento de A2. Sus hallazgos generan hipótesis y no seleccionan otro modelo final con el periodo evaluado. Las [RQ de §1](01_punto_partida.md) y su [especificación documental](../_static/e2_point1_review_v1/rq_specification.json) mantienen separados RQ1, RQ2 y RQ3.

### 9.3.1 Inventario por alcance de modelos, configuraciones y análisis

| Ámbito | Cantidad y estado verificados | Interpretación para el control de decisiones |
|---|---|---|
| Selección neuronal reducida | **13 redes × seis plazas × tres folds × una semilla = 234 oportunidades**: 231 válidas, dos fallidas y una desconocida. Son 78 pares modelo–plaza antes de repetir por fold. [§48](48_protocolo_reducido.md). | El dominio reducido y sus fallos se conservan. Las reutilizaciones no se cuentan como entrenamientos GPU nuevos ni las recetas ganadoras como todo el esfuerzo de búsqueda. |
| Selección de clásicos | SeasonalNaive: tres periodos y nueve validaciones válidas. ARIMA: 18 órdenes × tres folds = 54 oportunidades; 51 válidas, dos no convergencias y un fallo de implementación; 15 órdenes con los tres folds válidos. [§6](06_benchmark_final.md). | Es otro dominio de búsqueda, no otras 54 oportunidades neuronales ni un motivo para sustituir fallos. Persistence tiene regla fija. |
| Ajustes y medidas finales | **78 ajustes neuronales + 3 clásicos = 81 predictores**, más los diez ajustes piloto. A2 publica seis medidas de desempeño: RMSE, MAE, sMAPE, R², cobertura y anchura; DE, R y n documentan variación/soporte. | Los pilotos no aumentan el R final. Métricas secundarias, semillas y fechas no son configuraciones alternativas ni permiten cambiar la primaria después de ver el ranking. |
| Familia primaria y variantes | Veinte pares, tres procedimientos originales por par, cada familia Holm20; refinamiento de 4.999 a 20.000 y sensibilidades de bloques de 7/28 días identificados. [§7.3](07_ruta_temporal.md). | No se cuentan variantes de resolución o bloques como métodos nuevos. La primaria mantiene **cero rechazos** de mejora mayor que δ. |
| Diagnósticos posteriores | Residuos: 826 hipótesis; DM: 120 pares; MCS: un conjunto de 16; rangos: 13 redes y 78 Wilcoxon; GR/constancia: 40 hipótesis. Cada familia tiene estimando y control propios. [§49](49_diagnosticos_f2a.md) · [§50](50_comparaciones_f2a.md) · [§56](56_inferencia_estabilidad_final.md). | Sus p-valores no se agregan como votos de superioridad práctica ni como una sola familia confirmatoria global. MSE diaria y ganancia de RMSE>δ responden a nulas distintas. |
| Robustez, explicaciones y complementos | Trece redes × cuatro perturbaciones × tres severidades; 52 áreas. Top 3 explicado en 24 fechas. **81 reproducciones + 18 ajustes de tres ablaciones**; tres contrastes Holm3. Calibración y ensemble mantienen planes posteriores. [§54](54_robustez_interpretabilidad.md) · [§55](55_reproduccion_ablaciones.md) · [§60](60_calibracion_temporal_condicional.md) · [§63](63_ensamble_calibracion.md). | Las reproducciones verifican los mismos predictores; las ablaciones son variantes experimentales separadas. No amplían el catálogo A2 ni autorizan retuning de su ganador. |
| Selección histórica anterior | [§41](41_avance_optimizacion_11_origenes.md) conserva **1.484 registros nuevos completed** de aquel corte y diseño, con su incidente físico original desconocido documentado. | No son 1.484 ajustes adicionales del dominio reducido ni sistemas finales. Se preservan los IDs y el alcance de caché/contabilidad; no se suman a 231 como si fueran ensayos nuevos de una misma campaña. |

Este inventario informa las cantidades que permiten las fuentes de cada ámbito; **no reconstruye un total único y exhaustivo de todas las configuraciones, variantes y métricas históricas examinadas**. Los diseños anteriores, P0, la réplica externa ETTh1 y los suplementos conservan sus identidades. Las sumas de cortes o recibos reutilizados no ofrecen por sí solas el número de decisiones independientes ni corrigen el efecto de explorar múltiples caminos.

### 9.3.2 Reglas de interpretación que se mantienen

La ganancia de RMSE, su IC y el p ajustado se leen respecto a **δ fijo**. Ninguna comparación primaria demuestra mejora práctica tras Holm20; el menor valor observado de SAE no se convierte en superioridad universal. Los resultados negativos y los fallos permanecen accesibles. Un no rechazo de GR, Wilcoxon o una ablación no prueba estabilidad, equivalencia o contribución nula.

RQ2/RQ3 conservan sus aclaraciones posteriores y sus requisitos de datos; no reciben automáticamente p-valores, potencia o control familiar de RQ1. La calibración temporal, el ensemble y el proxy de techo tienen reglas propias y no sustituyen A2. El δg previo se reconoce, mientras su cumplimiento formal actual y la consistencia validación–prueba siguen sin establecerse. La [discusión de §6](06_benchmark_final.md) y la [decisión de §7.5](07_ruta_temporal.md) separan benchmark ejecutado de adecuación conjunta.

Esta actualización utiliza registros guardados y aritmética descriptiva identificada. Mantiene modelos, semillas, recetas, datos y presupuesto; no incorpora nuevos ajustes, inferencia, pruebas de hipótesis o remuestreos. Las tareas pendientes conservan su diseño y alcance propios; no se presentan como análisis ya realizados.

````{admonition} Archivo documental: contenido anterior del capítulo 9
:class: dropdown

Este bloque conserva íntegramente el contenido anterior y sus encabezados. Sus fechas, estados y enlaces corresponden a esos cortes; la lectura vigente sigue los apartados 9.1–9.3 anteriores.

**Estado vigente del 14/11/2026:** están cerrados la selección reducida, los **78 ajustes finales neuronales**, los **diez pilotos**, los **tres comparadores** y la evaluación común de los **16 sistemas**. Se completaron además **81 reproducciones y 18 ablaciones**. [§48](48_protocolo_reducido.md) reúne los registros y transiciones del ejecutor; [§55](55_reproduccion_ablaciones.md), los recibos del cierre complementario.

## 9.1 Registros que definen la entrega

| Elemento | Fuente y alcance |
|---|---|
| Catálogo obligatorio | [§43](43_catalogo_obligatorio.md): diez candidatos, tres referencias E1 y tres comparadores; híbrido y Ridge históricos/opcionales |
| Cuerpos ejecutados | [§44](44_arquitecturas_publicadas.md) y actualización de [§48](48_protocolo_reducido.md), incluida TiDE publicada r=0 |
| Selección y calendario efectivos | [§48](48_protocolo_reducido.md): seis configuraciones por red y tres folds; ajuste final hasta 31/10/2024, embargo y calibración disjunta |
| Congelación anterior al scoring de esta campaña | [81 predictores](../_static/e2_benchmark_final_20261113/final_freeze.json), con pesos, escaladores y calibradores |
| Admisión y soporte común | [Recibo de datos](../_static/e2_benchmark_final_20261113/data_admission.json): 608 fechas conocidas de E1 |
| Resultados originales | [A2](../_static/e2_benchmark_final_20261113/benchmark_A2.csv) y [familia primaria](../_static/e2_benchmark_final_20261113/primary_contrasts.csv) |

Las enmiendas conservan sus fechas reales. Los cambios de adquisición, respaldo y recursos tienen recibos propios y no se renombran como cambios de receta científica. El registro original de P0, de noviembre de 2026, sigue disponible como antecedente y no acredita por sí solo los modelos incorporados posteriormente.

## 9.2 Familia primaria y análisis posteriores

La familia primaria contiene **20 pares**: cada candidato frente a GRU32 y frente a Persistence, con margen δ heredado y ajuste Holm. La evaluación original usó 4.999 sorteos. El [plan posterior](../_static/e2_temporal_final_audit_20261114/analysis_plan.json), registrado antes de sus cálculos en [`60cd2bb`](https://github.com/jassan-bit/DeepLearning_202630/commit/60cd2bb23f6da22797e2c3a7bf8225250753f903), amplió **la misma familia** a 20.000. Reprodujo exactamente los veinte resultados originales antes del refinamiento; todos los p Holm siguen siendo 1. [§56](56_inferencia_estabilidad_final.md) distingue esa precisión Monte Carlo de un preregistro anterior al conocimiento del test.

Los diagnósticos F2a, DM/MCS, robustez, explicaciones, ablaciones y taxonomía se identifican como suplementos posteriores. Sus familias y estimandos no se presentan como confirmaciones adicionales de cada contraste primario. Se conservan las hipótesis, recetas, semillas y resultados negativos.

## 9.3 Potencia, repeticiones y desviaciones

El piloto midió cinco semillas pareadas de iTransformer y GRU32. Con R=6 y el efecto planificado, su potencia condicional estimada fue **0,123**, inferior a 0,80. Se conservó R=6 por recursos y se declaró la limitación; no se amplió R tras ver el test para buscar significación. El [informe de potencia y MDE](../_static/e2_benchmark_final_20261113/pilot_power_MDE.json) conserva supuestos y alcance.

La partición se hereda de E1 conforme a la guía. La congelación anterior al scoring de E2 acredita el orden de esta campaña; el calendario conserva su condición histórica conocida. **δg=0,00025772842664981106 sí estaba registrado en el commit `599a29f` del 6 de noviembre**, antes de la campaña actual, y la ampliación `ef2be289` definió la dirección de la prueba de tolerancia. El [linaje registrado de δg](../_static/e2_point1_review_v1/margin_lineage.json) distingue ese antecedente de la prueba formal aún no acreditada para los modelos actuales. El [Apéndice B actual](61_reproduccion_consolidada.md) recoge esta aclaración sin alterar los registros ni los análisis guardados.

## 9.4 Cierre y trazabilidad

El [estado final](../_static/e2_reproduction_ablations_20261113/status.json) documenta 99/99 ajustes complementarios cerrados, sin reservas abiertas. Los fallos HPO, el desenlace desconocido T4, las incidencias ARIMA y las pausas de transporte conservan sus cargos y recibos. Un build del libro comprueba presentación; una auditoría aritmética comprueba los resultados guardados; los recibos de reajuste acreditan el entrenamiento realmente repetido. Sus alcances se separan en [§8](08_reproducibilidad.md).


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/09_protocolo_registrado.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).

````
