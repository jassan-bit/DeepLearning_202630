# 62. Elegibilidad focal y puntuación bibliográfica contextual

La revisión documental del **14/11/2026** resuelve decisiones individuales que estaban pendientes o dispersas entre E1 y E2. Se adjudican 30 objetivos de recuperación dirigida, se concilian las fuentes arquitectónicas actuales y se amplía a nueve modelos la información de desempeño contextual. Es un complemento retrospectivo: conserva el catálogo autorizado y los puntajes anteriores, y no utiliza la A2 para escoger literatura.

## Decisiones de elegibilidad con evidencia individual

El corte focal contiene **33 apariciones de fuentes y 30 objetivos únicos por DOI**: la unión de 18 objetivos de completitud, diez IDs de la segunda tanda y cinco fichas del seguimiento inicial. No es una muestra aleatoria de los 9.559 grupos recibidos ni representa el cribado completo de la cohorte de 1.078 o de los cuarenta candidatos anteriores.

| Disposición del corte focal | Fuentes |
|---|---:|
| Excluidas del ámbito aplicado por población: fotovoltaica, viento o tráfico | 4 |
| Sin original completo acreditado en los recibos examinados; elegibilidad pendiente | 8 |
| Con secciones primarias previamente evaluadas | 18 |
| De esas 18: incluidas en síntesis aplicada de volatilidad derivada de retornos | 11 |
| De esas 18: incluidas en estratos distintos de volatilidad implícita o condicional | 2 |
| De esas 18: contexto de mecanismo, con constructo y elegibilidad aplicada pendientes | 2 |
| De esas 18: excluidas del núcleo aplicado DL–volatilidad verificable | 3 |

Por tanto, se adjudican **13 inclusiones aplicadas y siete exclusiones**. Los ocho informes incompletos y las dos dudas de constructo mantienen su condición pendiente. No se infiere que todos los informes hayan sido buscados con el mismo protocolo ni que se hayan realizado dieciocho nuevas lecturas. La evidencia procede de fichas primarias ya existentes, con DOI, versión y localizadores identificados en el [registro de decisiones](../_static/e2_literature_delivery_closure_20261114_v2/eligibility/applied_target_decisions.csv).

La admisibilidad narrativa no exige que cada artículo pronostique exactamente σ₇ con D7×5. Las diferencias de proxy, frecuencia, horizonte e información sí impiden mezclar sus métricas como un único efecto. Semillas, código, arquitectura y presupuesto de ajuste se conservan como calidad o límites; su ausencia no se transforma automáticamente en exclusión de toda la discusión.

Algunas decisiones muestran por qué no basta el título o un estado anterior de recuperación:

- **E2-0173**, *Forecasting bitcoin volatility: exploring the potential of deep learning*, se excluye de la síntesis aplicada de volatilidad: la ecuación 14 define el objetivo como logretorno diario.
- **E2-0012**, DOI `10.3390/app10144768`, se conserva como comparador neuronal: sus ANN/HONN de una capa no acreditan el predictor profundo requerido para ese núcleo.
- **Miura 2019/E2-0671** sí cuenta con original y extracción previos para varianza realizada de tres horas. El enlace exacto con E1 corrige el estado parcial de una tanda anterior; no inventa una nueva descarga.
- **E2-0285** cuenta con secciones de la versión final IEEE 2025 documentadas en E1. Sus resultados no se intercambian con los del preprint SSRN 2022.
- **CryptoMamba-SSM** y el híbrido de DOI `10.3390/risks10120237` se mantienen como contexto de mecanismos mientras sus constructos no estén conciliados. No se cuentan como efectos aplicados elegibles.

La {download}`síntesis focal <../_static/e2_literature_delivery_closure_20261114_v2/eligibility/focused_synthesis.md>` conserva los resultados adversos y las diferencias entre MLP/HAR, LSTM y otros modelos, así como discordancias MSE/RMSE y la dependencia del ganador respecto a la métrica.

## Fuentes de mecanismos y enlace entre entregables

Los diez sistemas actuales corresponden a **nueve estudios primarios**, contando una sola vez la fuente compartida por FCN y ResNet. Los nueve se admiten como fuentes de mecanismo, con sus adaptaciones D7 y límites; Wang se conserva como excepción fundacional. Otros cuatro originales sirven como comparadores históricos. La [adjudicación de las trece fuentes](../_static/e2_literature_delivery_closure_20261114_v2/eligibility/mechanism_source_decisions.csv) distingue esa función de la eficacia aplicada sobre Bitcoin.

Un índice reproducible enlaza por **DOI exacto** 107 fichas históricas con grupos E2. **43 grupos tienen un recibo previo de evaluación de secciones primarias**. Sus [roles](../_static/e2_literature_delivery_closure_20261114_v2/eligibility/historical43_source_roles.csv) se revisaron documentalmente; no se importan automáticamente las decisiones estrictas de E1. Dieciséis solapan con el corte focal, por lo que los conteos no se suman. El [índice de procedencia](../_static/e2_literature_delivery_closure_20261114_v2/cross_entregable/cross_entregable_primary_links.csv) no se presenta como nuevas lecturas ni como un PRISMA de 43 estudios.

## Desempeño contextual: nueve modelos, con unidades separadas

El P anterior permanece intacto. El nuevo escenario usa resultados dentro de tablas publicadas y sus propios comparadores, con una regla retrospectiva declarada antes de generar el escenario. No exige que todos los trabajos estudien Bitcoin, y separa clasificación de pronóstico:

| Sistemas | Evidencia y comparación contextual |
|---|---|
| FCN y ResNet | MPCE agregado de los 44 datasets UCR, once métodos de la tabla I de Wang; evita escoger una sola fila favorable |
| LSTM-SAE | RMSE de los cuatro modelos de la tabla 2 de Beijing PM2.5; MAE y SMAPE como sensibilidades separadas |
| TimesNet, TiDE, PatchTST e iTransformer | MSE ETTh1/H96/L96 dentro de la tabla 10 de iTransformer, con once comparadores |
| TimeMixer | MSE ETTh1/H96/L96 dentro de su tabla 13, con once comparadores |
| TSMixer | Columnas de hiperparámetros unificados de la tabla 23 de TimeMixer, con cuatro comparadores |
| Inception individual | P pendiente: el resultado verificado pertenece al ensemble de cinco redes |

El [protocolo y las fuentes](../_static/e2_literature_delivery_closure_20261114_v2/citation_score/contextual_p/preferred_intrapaper_scenario_rule.json) fijan métrica, contexto, horizonte, elección de tabla y todos sus comparadores. Para pérdidas menores-mejor se normaliza por el rango de la misma tabla. Un valor de uno significa el menor error dentro de ese conjunto; no acredita estado del arte global. El rango depende de los comparadores, y la pertenencia a una misma tabla no demuestra igualdad de HPO, escalado o historia. SAE conserva historias de 25 y 30 pasos en sus filas, explicitadas en la extracción.

La sensibilidad es material: **SAE recibe P=0 con RMSE y P=1 con MAE o SMAPE** entre sus cuatro comparadores. Se conserva RMSE en el escenario principal y se publican las alternativas; no se elige la métrica más favorable.

| Escenario documental | C conocidas | P contextuales calculables | Componentes conocidos | Scores condicionales completos |
|---|---:|---:|---:|---:|
| Puntaje anterior conservado | 6 | 0 | 36/50 | 0/10 |
| Sensibilidad anterior ETTh1 nominal | 6 | 6 | 42/50 | 6/10 |
| Nuevo escenario dentro de tablas | 6 | 9 | **45/50** | **6/10** |

Los [scores e intervalos algebraicos de todos los sistemas](../_static/e2_literature_delivery_closure_20261114_v2/citation_score/conditional_score_by_scenario.csv) reutilizan los pesos previos únicamente como sensibilidad. Sumar escalas locales de tareas diferentes supone una equivalencia de utilidad que no está demostrada. **No se presenta ese resultado como ranking global controlado ni como la selección que produjo el Top 10.**

Las cuatro C de FCN, ResNet, Inception y SAE permanecen desconocidas. Se verificaron alias editoriales y las exportaciones recibidas; no se obtuvo un corte bibliométrico uniforme nuevo. Los conteos aislados de editoriales no se mezclan con Scopus ni un valor ausente se convierte en cero. La [auditoría bibliométrica](../_static/e2_literature_delivery_closure_20261114_v2/citation_score/citation_reconciliation_manifest.json) conserva esa búsqueda y sus límites.

## Estado que conserva la entrega

Se completaron las decisiones documentales focales, el enlace de evidencia histórica y la sensibilidad de desempeño contextual. La **elegibilidad global de los 9.559 grupos sigue incompleta**; los ocho accesos sin original acreditado y las dos dudas de constructo no se convierten en inclusiones. Tampoco se recupera una selección bibliográfica previa por puntaje ni eficacia demostrada sobre el contrato E2.

El [manifiesto de esta actualización](../_static/e2_literature_delivery_closure_20261114_v2/manifest.json) conserva los insumos, resultados y código. La revisión de fichas se identifica como asistida por IA, sin atribuir doble revisión humana independiente ni nuevas lecturas completas de originales. No hubo nuevos entrenamientos ni inferencia de modelos.
