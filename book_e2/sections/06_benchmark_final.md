# 6. Benchmark Final

**Estado vigente:** P0 se conserva con su registro `599a29f`. La implementación final histórica y la [enmienda metodológica](../_static/completion_protocol/analysis_additions.json) se publicaron en [`dbb193a`](https://github.com/jassan-bit/DeepLearning_202630/commit/dbb193a72897c80cbd34b452ba3cfe9d82ce6ce0) y se [registraron](../_static/completion_preflight/registration.json) el 06/11/2026 a las 18:59:10 UTC. El recibo de las 19:00 UTC acredita HPO iniciado e incompleto, no actividad actual ni cierre; **resultados del benchmark final Bitcoin D7 pendientes**. P0/OOF está ejecutado y analizado según [§9](09_protocolo_registrado.md). Sus resultados de desarrollo tienen una sección propia; no se insertan como resultados de los veinte modelos finales.

La réplica y el piloto externo ETTh1 de [§5](05_paper_guia.md) conservan su dataset y finalidad. Las métricas históricas de E1 tampoco se convierten en nuevas corridas E2. El acceso anterior a la reserva limita a exploratoria la interpretación del futuro benchmark, aunque el procedimiento actual esté registrado antes de sus nuevos ajustes.

## 6.0 Alcance de la propuesta nueva y condiciones de reporte

**La tabla de veinte configuraciones siguiente pertenece al catálogo registrado histórico.** Los diez candidatos revisados y el híbrido de [§15](15_revision_modelos.md), junto con los prototipos de [§4.0](04_implementacion.md), no sustituyen sus filas ni heredan sus recibos. El nuevo experimento todavía no tiene catálogo completo congelado: diez candidatos más un híbrido no incluye automáticamente todas las anclas, baselines y controles HAR/red sola.

Para la propuesta nueva se conserva una [secuencia editorial de próximos pasos](../_static/literature_screening_followup_20261108/next_phases.json), **no un preregistro**:

| Fase | Evidencia necesaria antes de avanzar | Estado de la propuesta nueva |
|---|---|---|
| Selección y recetas | Fuentes, criterios, familias, arquitectura y condición NAS; controles y soportes comparables | Abierta; fichas y prototipos no son recetas congeladas |
| Pipeline | HAR/OLS, cross-fitting temporal real, enlace/loss residual, seis adaptadores retenidos, escalado y runner separado | Prototipo parcial; no listo para datos reales |
| Registro | Enmienda ejecutable publicada, hashes, datos ya observados, calendario, presupuesto y análisis definidos antes de ajustes | No creado para el catálogo nuevo |
| HPO y dimensionamiento | Búsquedas comparables cerradas, recetas fijadas y P1 con variación temporal y de entrenamiento | No iniciados para el catálogo nuevo |
| Evaluación y reporte | Pesos/calibradores congelados, cohortes comunes, métricas y costes, fallos y resultados negativos | Sin resultados de mercado de esta propuesta |

No se llenan celdas con pérdidas sintéticas, cifras UCR/bicicletas o resultados de otros artículos. Los resultados históricos P0, E1 y ETTh1 mantienen su alcance. La reserva ya conocida continúa siendo exploratoria; una publicación de código no recupera independencia ni acredita potencia.

## 6.1 Tabla comparativa de veinte configuraciones

La [sección 4](04_implementacion.md) y la [enmienda HPO](../_static/completion_protocol/hpo.json) distinguen quince recetas ajustables, tres anclas heredadas y dos referencias ingenuas. La siguiente tabla es una estructura de reporte, **sin un ranking observado**. Cada celda pendiente exige evidencia del procedimiento final, no del piloto reducido.

| Configuración | Papel | RMSE media ± DE e IC95 % | Secundarias y coste |
|---|---|---|---|
| TSMixer | Top 10 · MLP | PENDIENTE | PENDIENTE |
| TiDE | Top 10 · MLP residual | PENDIENTE | PENDIENTE |
| TimeMixer | Top 10 · MLP multiescala | PENDIENTE | PENDIENTE |
| TimesNet | Top 10 · convolución 2D temporal | PENDIENTE | PENDIENTE |
| PatchTST | Top 10 · atención por parches | PENDIENTE | PENDIENTE |
| Informer | Top 10 · atención | PENDIENTE | PENDIENTE |
| Autoformer | Top 10 · autocorrelación/descomposición | PENDIENTE | PENDIENTE |
| iTransformer | Top 10 · atención entre variables | PENDIENTE | PENDIENTE |
| Crossformer | Top 10 · atención jerárquica | PENDIENTE | PENDIENTE |
| S-Mamba | Top 10 · estado selectivo | PENDIENTE | PENDIENTE |
| GRU-E2 | Base retunada con presupuesto común | PENDIENTE | PENDIENTE |
| LSTM-E2 | Base retunada con presupuesto común | PENDIENTE | PENDIENTE |
| TCN-E2 | Base retunada con presupuesto común | PENDIENTE | PENDIENTE |
| GRU/D7-c03 | Ancla fija E1; baseline heredado | PENDIENTE | PENDIENTE |
| LSTM/D7-c02 | Ancla fija E1 | PENDIENTE | PENDIENTE |
| TCN/D7-c02 | Ancla fija E1 | PENDIENTE | PENDIENTE |
| Ridge | Referencia clásica ajustable | PENDIENTE | PENDIENTE |
| ARIMA | Referencia temporal ajustable | PENDIENTE | PENDIENTE |
| Persistencia σ₇(t−1) | Referencia trivial | PENDIENTE | PENDIENTE |
| Naive estacional σ₇(t−7) | Referencia semanal de la ruta | PENDIENTE | PENDIENTE |

**PENDIENTE no significa cero.** Se publicarán resultados por semilla/origen, elegibilidad, fallos y cobertura, además de los agregados. Las referencias deterministas se almacenan una vez por receta; repetir su predicción para una resta pareada no crea R entrenamientos independientes. Las anclas E1 permanecen diferenciadas de las bases retunadas.

La primaria se conserva:

$$
q_{m,s}=\sqrt{|\mathcal T|^{-1}\sum_{t\in\mathcal T}(Y_t-\widehat Y_{m,s,t})^2},
\qquad \widehat\mu_m=R^{-1}\sum_s q_{m,s}.
$$

No se reemplaza por la RMSE de predicciones promediadas, la raíz de errores agrupados entre semillas ni un promedio no ponderado de RMSE mensuales. MAE, MSE, R², MAPE y MASE, además de MSE sobre σ₇² y QLIKE, complementan la primaria, con denominadores y cobertura declarados. MASE usa una escala ingenua calculada sólo en entrenamiento; una recomendación de MASE en la ruta no cambia silenciosamente la RMSE heredada.

## 6.2 Rangos, eficiencia e incertidumbre

Los rangos medios se obtendrán **dentro de cada semilla**, con rango medio en empates exactos, y después se promediarán. No son el rango de la RMSE media agregada. Se mostrarán ambos órdenes si difieren, junto a errores y efectos; las fechas y diez meses solapados no se convierten en datasets independientes.

Las fronteras de Pareto compararán RMSE con latencia, parámetros, FLOPs, memoria y coste de entrenamiento por separado. Una medida ausente queda no estimable: no beneficia al modelo ni elimina su error del ranking descriptivo. Dominancia puntual y estabilidad por remuestreo no demuestran superioridad poblacional o equivalencia.

Los costes deben proceder del mismo hardware y protocolo registrado. La latencia por ventana batch 1, throughput por lote, tiempo HPO y ajuste definitivo se distinguen. Se declararán operaciones no contabilizadas, versiones y precisión; no se comparan cifras externas ETTh1 con inferencia D7 como si fueran mediciones equivalentes.

Los IC del score describen incertidumbre del calendario o del entrenamiento, según su unidad. **No son intervalos predictivos de σ₇.** La rama A53 de [§12](12_incertidumbre_errores.md) prepara **los veinte modelos** antes de conocer el ranking, para poder informar a los tres mejores con una rama válida. Su calibración usa 53 fechas disjuntas de selección/ajuste hasta 31/10, cinco miembros por red fijados antes de P1 y exactamente el mismo predictor A en evaluación, sin refit; los resultados de cobertura siguen pendientes. Los residuos OOF de desarrollo no constituyen automáticamente un split de calibración disjunto del modelo final que después use esas mismas etiquetas.

El código incorpora una **referencia condicional de predictibilidad**, no un techo universal demostrado: D7 contiene seis retornos conocidos y el séptimo se integra con una ley empírica de innovaciones aprendida exclusivamente en entrenamiento 2023 hasta 24/12. Si S y Q son suma y suma de cuadrados de los seis retornos conocidos, para una innovación r se evalúa σ₇²=(Q+r²)/7−((S+r)/7)². Se estima media/varianza condicional de σ₇ bajo esa ley; iid y transferencia temporal son supuestos, no resultados probados. La entropía de permutación usa dimensión3/retardo1.

La distancia γ=(RMSE_mejor−RMSE_proxy)/(RMSE_baseline−RMSE_proxy) conserva valores fuera de [0,1] y queda no estimable con denominador inestable. Mejor modelo y baseline se fijan en desarrollo. Bootstrap de 2.000 réplicas propaga ley de innovación, calendario y semillas emparejadas, con bloques de siete días y sensibilidades de catorce/veintiocho días; no elimina toda incertidumbre por selección de modelos. R²=1 no es un techo empírico. Los resultados efectivos del proxy/γ siguen pendientes.

QLIKE se calcula sobre varianzas con suelo fijo 10⁻¹²: log(max(σ̂²,10⁻¹²))+σ²/max(σ̂²,10⁻¹²). No se aplican pruebas VaR de rentabilidades cuando el modelo no especifica esa distribución condicional. Coste por mejora y equivalencia requieren sus márgenes, incertidumbre y denominadores válidos.

## 6.3 Familias registradas y control de decisiones

### 6.3.1 Dos comparaciones y un presupuesto global

El [preregistro](../_static/remediation_protocol/preregistration.json) conserva **δ=0,00025772842664981106** y registra tres familias. No se revisan al conocer resultados:

| Familia | Comparaciones | Número | Nivel y control |
|---|---|---:|---|
| Referencia heredada | Diez Top 10 frente a GRU/D7-c03 para RQ1–RQ3 | 30 | Holm, α=0,02 |
| Comparación de ajuste equivalente | Diez Top 10 frente a GRU-E2 para RQ1–RQ3 | 30 | Holm, α=0,02 |
| Diagnósticos finales registrados | 180 diagnósticos modelo/prueba y 19 DM frente a GRU/D7-c03 | 199 | Holm, α=0,01 |

La suma asignada es 0,05; la afirmación conjunta se limita a estas familias y depende de p-valores válidos. Los análisis de MCS, calibración, segmentos, equivalencia o factores NAS se identifican con su alcance propio; no quedan cubiertos automáticamente por esa suma. Tampoco se adoptan como principales los 190 pares posibles de veinte configuraciones.

Para cada referencia b, sea g=μ_b−μ_m, positivo si mejora RMSE; d=g_normal−g_alta, erosión de esa ventaja; y c=latencia_m−latencia_b. Las alternativas registradas son **RQ1: g>δ**, **RQ2: d>δ** y **RQ3: g>δ y c<0**. Para RQ3, el p de intersección-unión es el máximo de los dos p unilaterales válidos, antes de Holm {cite:p}`e2_holm`.

Si una hipótesis no se puede estimar, conserva internamente p=1 para la multiplicidad y se publica como no estimable, con motivo. No se reduce el tamaño de familia por fallos. El régimen alto usa σ₇(t−1)>0,024247163097617024, cuantil 0,75 calculado sólo en entrenamiento de 2023, sin añadirlo como canal de red.

### 6.3.2 Remuestreo, límites y potencia

El protocolo fija 99.999 remuestreos, semilla `202611064`, bloques principales de siete fechas y sensibilidades de catorce/veintiocho; el fundamento temporal se verifica en desarrollo, sin escoger la longitud que dé el menor p. Los bloques se comparten entre modelos, preservan el calendario y recalculan cada RMSE antes de promediar semillas.

La nula práctica se centra en la frontera δ. Los IC95 % nominales percentiles acompañan límites unilaterales simultáneos conservadores, con cola **α_familia/30**, inferiores para g/d y superiores para c. Se reutiliza el mismo límite de g en RQ1 y RQ3. No se aplican por inercia límites 0,05/30 tras haber dividido el presupuesto de error.

La resolución mínima Monte Carlo es 1/100.000. Más remuestreos reducen error numérico, sin crear más bloques o mercados. Las sesenta fechas conocidas del test siguen limitando la información temporal; el bootstrap requiere supuestos de dependencia/estabilidad y no garantiza cobertura exacta bajo cambio de régimen.

P1 determinará R **después de HPO y congelación de recetas**, sobre 53 fechas 08/11–30/12/2024, ajuste/selección hasta 31/10 y embargo 01–07/11. La proyección a sesenta días de test es un supuesto explícito. La meta es 0,80 para alternativas de ganancia/erosión 2δ —exceso δ sobre la frontera nula— y ahorro de latencia del 20 %. La selección usa simulación, límite inferior Wilson y ambas fuentes de variabilidad, con máximo R=200. No se promete potencia 0,80 en g=δ ni se extrapola automáticamente a modelos no pilotados. Los proxies de planificación y una meta incumplida se informan como tales.

### 6.3.3 Supuestos y diagnósticos

Las diferencias por semillas pueden estudiarse con t pareado/Wilcoxon sólo con su alcance condicional y supuestos declarados; no representan mercados nuevos. Se informará d_z pareado y su incertidumbre; DE cero implica efecto no definido. Shapiro no decide automáticamente qué prueba produce la conclusión. Con cinco pares sin empates ni ceros, el mínimo p bilateral exacto de Wilcoxon es 0,0625.

Para DM se promedian **pérdidas cuadráticas individuales por fecha**, no predicciones de un ensamble. El horizonte es h=1 y la corrección HLN es √((T−1)/T); el HAC Bartlett registrado tiene rezago siete porque el horizonte uno no elimina la dependencia de etiquetas σ₇ {cite:p}`e2_dm,e2_hln`. DM compara MSE media, no la mejora no lineal δ de RMSE entre semillas.

Los 180 diagnósticos son nueve por configuración: Ljung–Box en residuos y cuadrados a rezagos 7/14; ARCH-LM a rezago 7; BDS en dimensiones 2/3 con radio 1,5 DE; Jarque–Bera; y Mincer–Zarnowitz conjunto. Se usa una regla de semilla fijada, sin concatenar residuos de varias semillas. Para la red fuera de muestra, `model_df=0` no resta sus pesos como parámetros ARMA {cite:p}`e2_ljung`.

BDS rechaza iid, no demuestra por sí solo no linealidad aprovechable. Los tests no estimables o con supuestos insuficientes se informan con alcance y motivo. La descomposición de varianza, superioridad y sensibilidad con prueba alternativa requieren su ejecución; no se estima una varianza entre series con una única serie BTC.

### 6.3.4 Ruta temporal y métodos globales

La guía específica exige considerar DM y **Model Confidence Set** en la ruta temporal; MCS no se omite por describirse como opcional en un capítulo general. Requiere pérdidas comparables, nivel y bootstrap temporal, y su conjunto no prueba equivalencia ni una mejora mayor que δ. Su ejecución final permanece pendiente.

No se fabricará Friedman/Nemenyi o un diagrama de diferencia crítica tratando semillas o meses solapados como datasets independientes. La aplicabilidad se explica por unidad y dependencia; con este único mercado no se obtiene una comparación entre datasets {cite:p}`e2_demsar`. Las comparaciones planificadas con Holm no necesitan rechazar previamente Friedman. Nadeau–Bengio tampoco se traslada sin derivación a ventanas expansivas.

### 6.3.5 Familias exploratorias adicionales de la enmienda

Se mantienen las 30+30+199 hipótesis originales. Los siguientes análisis tienen controles **separados**; no se afirma un FWER global de 0,05 sobre todo el libro:

| Familia | Nula/decisión predefinida | Control |
|---|---|---|
| Brecha A53 por encima de tolerancia | RMSE_validación−RMSE_train≤δg, δg=δ; veinte modelos | Holm20, α=0,005; 19.999 bloques |
| Brecha A53 dentro de tolerancia | RMSE_validación−RMSE_train≥δg; rechazo y cota superior<δg para afirmar tolerancia | Holm20, α=0,005; no rechazo no acredita ausencia de sobreajuste |
| Equivalencia y menor coste | Diecinueve alternativas frente al mejor de los veinte elegido en desarrollo; TOST ±δ y ahorro de latencia, IUT | Holm19, α=0,01 |
| PIT predictivo | Uniformidad marginal en la cuadrícula fijada, proceso empírico temporal centrado | Holm20, α=0,01 |
| Ablaciones | Tres diferencias Δ=RMSE_sin_componente−RMSE_control, H₀:|Δ|≤δ | p práctico bilateral, Holm3 α=0,05; 9.999 bloques de siete días, sensibilidades de catorce/veintiocho días |
| Controles de etiquetas permutadas | Diecisiete alertas conjuntas: ganancia RMSE>δ y asociación Spearman positiva frente a media train | IUT, Holm17 α=0,01; 2.000 remuestreos, diagnóstico de fuga sin prueba de ausencia |
| Clasificador adversarial | Distinguibilidad de entradas de entrenamiento/evaluación, parámetros fijos | α=0,01, 2.000 permutaciones por bloques y CV reajustada; diagnóstico de shift sin retroalimentar HPO |
| MCS | Eliminación secuencial Tmax con pérdidas temporales comparables | Nivel90 %, 9.999 bloques; no demuestra equivalencia práctica |

La brecha formal A53 usa **media de RMSE por semilla** en modo evaluación, el mismo predictor entrenado hasta el 31/10 y DEV53. No se calcula sobre un ensamble ni se atribuye al ajuste primario hasta24/12. Para este último se informa aparte la brecha train−test descriptiva, confundida con deriva temporal. Las ablaciones usan únicamente DEV53, con reajuste desde cero y el mismo R final; no aportan nuevas consultas de test. Un R dimensionado para las familias principales no certifica potencia de ablaciones o calibración.

## 6.4 P0/OOF y discusión final: alcances separados

El diagnóstico P0 tiene **una familia propia de tres hipótesis**, Holm α=0,01, sobre enero–octubre de 2024 OOF: RMSE_alta−RMSE_normal>δ para TimeMixer, iTransformer y PatchTST. No es RQ2 del benchmark —que compara erosión de ventajas frente a una GRU— ni la brecha train–validación δg. Noviembre–diciembre aporta factibilidad/variabilidad y un gap descriptivo, sin segunda familia para elegir resultados favorables.

La ejecución registrada comprende cuatro modelos, cinco semillas nuevas, diez orígenes OOF y una rama de sesenta fechas: **220 trayectorias / 440 ajustes completados**. El cierre, los resultados y la validación de sus recibos están en [§9](09_protocolo_registrado.md). No certifica potencia de recetas futuras ni demuestra un ganador entre los veinte modelos.

| Pregunta crítica | Estado del benchmark final | Evidencia que falta |
|---|---|---|
| Mejor modelo y mecanismo | No determinado | Error, efectos, rangos, ablaciones y explicaciones verificadas |
| Generalización y calibración | No determinada | Predicciones e incertidumbre apropiadas; límites de un activo y test conocido |
| Robustez | Pendiente | Perturbaciones con tres severidades, controles y cambios de distribución |
| Significación y relevancia | Pendiente | Familias, efectos e intervalos respecto a δ sobre resultados finales |
| Trade-off de coste | Sin frontera medida | Mediciones comunes, Pareto e incertidumbre |
| Techo y margen γ | Sin estimación | Referencia de predictibilidad justificada, no R²=1 por decreto |
| Respuesta a RQ1–RQ3 | Sin decisión nueva | Efectos g/d/c y reglas de multiplicidad registradas |
| Amenazas de validez | Identificadas | Auditorías reales, sensibilidad, dependencia y presupuesto realizado |
| Limitaciones y trabajo futuro | Abiertas según evidencia | Casos adversos, incertidumbre, coste y brecha contrastada para E3 |

## 6.5 Fuentes actuales y antecedentes

- [Enmienda preparada](../_static/completion_protocol/analysis_additions.json), [diseño final](../_static/completion_protocol/design.json) y [descarga de implementación](../_static/completion_download/completion_bundle.zip).
- [Preregistro, potencia y piloto](09_protocolo_registrado.md), [reglas de implementación](04_implementacion.md) y [preregistro original P0](../_static/remediation_protocol/preregistration.json).
- [Descargar protocolo, código y evidencias](../_static/remediation_download/Remediacion_E2_protocolo_codigo_evidencias.zip).

El [plan v1](../_static/benchmark_analysis/proposal.json), la [tabla antigua de quince filas](../_static/benchmark_analysis/table_template.csv) y el [ZIP de plantillas v1](../_static/benchmark_analysis_download/Benchmark_E2_analisis_plantillas.zip) son antecedentes fechados el 5–6 de noviembre. Sus familias y números no sustituyen el protocolo registrado actual. Las primitivas probadas con fixtures sintéticos verifican software; no aportan métricas de mercado, potencia alcanzada o auditoría completa del benchmark.
