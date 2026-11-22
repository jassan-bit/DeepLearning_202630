# 14 Evaluación y calificación

**Estado: reglas de la guía, matriz documental y valoración orientativa vigente en 14.13; cortes anteriores conservados en 14.10–14.12, sin calificación oficial asignada.** Este capítulo organiza la sección 14 de la guía —páginas 41–47, Cuadro 8 de la página 59 y Cuadro 9 de la página 60—. Las reglas pertenecen a esa guía. La valoración solicitada distingue el puntaje ponderado, los topes condicionales y el valor de sustentación aportado por el autor; no constituye una evaluación del docente.

La nota se expresa en la escala de **0 a 5** y se determina en dos pasos: primero se obtiene el puntaje ponderado de la rúbrica y después se aplica el tope que corresponda al estado **evaluado** de R1, R2 y R3. Una condición pendiente no se convierte en «no» para fijar ese tope, y una casilla sin evidencia no se marca cumplida para completar la rúbrica.

La evidencia se consulta en las [correcciones de E2](04_correcciones_e2.md), [artefactos](05_configuraciones_exportacion.md), [candidato](06_modelo_original.md), [diseño riguroso](07_diseno_riguroso.md), [verificación estadística](08_verificacion_estadistica.md) y [diseño del dashboard](09_dashboard_predictivo.md). Las actividades pendientes conservan su estado. Los cortes experimentales históricos y la validación temporal posterior de 8.29 conservan sus propios recibos. La valoración orientativa de 14.10 no modifica esos resultados ni rellena los campos de calificación oficial.

La matriz incorpora el corte posterior del **17 de noviembre de 2026**: diez semillas por red de E2 y de la referencia E3 de 1210 parámetros, tres trayectorias clásicas deterministas y análisis de los 17 sistemas sobre 608 fechas. Actualizar una casilla con evidencia retrospectiva no la convierte en cumplimiento confirmatorio ni transfiere las propiedades del primer módulo de 1211 parámetros a la receta evaluada.

La {ref}`auditoría ampliada cerrada en 8.18 <e3-final-extended-audit>` aporta **796 trabajos completos** y evaluación de los 19 sistemas en D00–D03, con 76 paneles, análisis terminado y 16 figuras. El corte parcial de 251 trabajos permanece en 8.17 como antecedente. La potencia prospectiva completa, la justificación de generalización y los restantes requisitos de R3 siguen pendientes. Estos datos actualizan la evidencia sin asignar nota, tope ni acreditación de R1/R3.

## 14.1 Tope de calificación según el modelo original

**Actualización por objeto:** {ref}`6.18 <e3-r1-substantive-verdict-20261120>` recomienda favorablemente R1 de V3 con alcance documental explícito. No acredita la aditiva ni R2/R3, y no asigna un tope de 4,0 o 5,0 automáticamente. La guía exige evaluar las condiciones y ponderar los componentes.

La guía establece estos casos **condicionales**, sujetos a evidencia y evaluación académica:

| Tope previsto por la guía | Condición que debe haberse evaluado | Alcance de la regla |
|---|---|---|
| 5,0 | R1 sí, R2 sí y R3 sí. | Novedad para el problema, núcleo algorítmico propio y defendible, propiedad formal exigida y superioridad estadísticamente verificada frente a toda la familia, con consistencia adicional. El tope no sustituye la rúbrica. |
| 4,0 | R1 sí y R2 sí, pero R3 no. | La contribución admisible y su sustento teórico están acreditados, pero no se demuestra la superioridad significativa exigida frente a toda la familia. |
| 3,0 | R1 no o R2 no, independientemente del desempeño. | La diferencia es sólo de aplicación, el mecanismo ya fue usado para el problema, la transferencia carece de atribución/adaptación sustantiva o el núcleo proviene de un import o código ajeno sin programación algorítmica propia. |

La condición incumplida de menor nivel determina el caso; **R3 sólo se evalúa cuando R1 y R2 están acreditadas**. «Pendiente de acreditar» no equivale a un «no» demostrado. V3 dispone del {ref}`dictamen documental por IA favorable y acotado de 6.18 <e3-r1-substantive-verdict-20261120>`; éste no acredita la aditiva ni constituye aceptación docente. La evaluación académica, la comprensión y defensa de R2 y la confirmación de R3 conservan sus requisitos, por lo que esta tabla no asigna una nota automática al avance. Las 35 pruebas del primer módulo y la demostración de su cabeza latente no acreditan por sí solas novedad, autoría responsable, defensa o superioridad de mercado.

La potencia se dimensiona **antes de congelar**: al menos 0,90 por contraste y potencia conjunta reportada, con diez semillas como mínimo o el número mayor requerido. La guía distingue estas situaciones:

| Situación | Regla de la guía | Evidencia que debe conservarse |
|---|---|---|
| El análisis previo muestra que las semillas disponibles no permiten detectar el margen práctico δ. | Aumentar las semillas antes de congelar el protocolo; el estándar no se reduce después de ver los resultados. | Piloto, estimación de dispersión, MDE, potencia por contraste y conjunta y número de semillas finalmente fijado. |
| El resultado es no concluyente y procede de un diseño sin potencia, declarado de antemano y con MDE reportada. | Con R1 y R2 acreditadas, se evalúa con tope 4,0. La falta de rechazo no se interpreta por sí sola como falta de calidad del trabajo. | Declaración previa, análisis de potencia, MDE y discusión del resultado no concluyente. |
| El resultado es no concluyente y no hay análisis de potencia. | Con R1 y R2 acreditadas, también corresponde el tope 4,0; además, el componente de pruebas de hipótesis y adecuación —15 %— se evalúa en «en desarrollo». | Identificar la ausencia del análisis; no rellenarla con potencia observada o con una justificación retrospectiva. |

Las dos últimas reglas son condicionales y no desplazan el tope 3,0 si se establece que R1 o R2 no se cumplen. Tampoco autorizan a omitir el dimensionamiento ni asignan una calificación a este avance. **El tope 5,0 no se concede por una media mejor cuando faltan los rechazos exigidos.**

## 14.2 Rúbrica ponderada

Se conservan exactamente los once componentes y pesos del Cuadro 9. Los criterios de clasificación se adaptan a la ruta temporal F2a cuando corresponde.

El {download}`registro estructurado de la rúbrica <../_static/e3_cierre_guia_20261118/rubric.json>` conserva los once pesos, las 45 condiciones y las reglas de topes/potencia. Sus campos de nota, tope y puntuaciones del proyecto están vacíos; el archivo permite comprobar la transcripción y no calcula ni atribuye una calificación.

| Componente | Peso | Evidencia que valora la guía |
|---|---:|---|
| Correcciones del Entregable 2 | 7 % | Trazabilidad, resultados consistentes recalculados cuando proceda, ranking antes/después, auditoría de fuga y determinismo. |
| Mejores configuraciones y artefactos | 7 % | Selección estadística, ajuste final, exportación, ida y vuelta, metadatos y coste de inferencia. |
| Dashboard: Contexto y EDA | 8 % | Contenido mínimo, callbacks, panel F2a e interpretación de cada figura. |
| Dashboard: Modelos | 9 % | Catálogo completo, predicción con explicación e incertidumbre, OOD, diagnósticos, comparación y ficha. |
| Dashboard: diseño, UX y despliegue | 5 % | Calidad visual, responsividad, recursos, robustez, seguridad, consistencia y URL funcional. |
| Modelo original: revisión y diseño | 8 % | Matriz/declaración de novedad, procedencia, paper guía, limitación con evidencia, hipótesis falsable, formulación y pseudocódigo. |
| Modelo original: implementación y demostración | 10 % | Núcleo propio o adaptación sustantiva, demostración/verificación numérica, complejidad, pruebas, ablación e interpretabilidad. |
| Evaluación ampliada y robustez | 10 % | Preregistro, equidad, conjuntos adicionales, familia completa, sensibilidad, perturbaciones, escalabilidad, calibración, incertidumbre y eficiencia. |
| Pruebas de hipótesis y adecuación | 15 % | Tres pruebas por contraste, supuestos, Holm, potencia por contraste/conjunta, IC, efectos y ajuste/sobreajuste. |
| Informe en Jupyter Book | 14 % | Estructura completa, contenedor, reproducibilidad, redacción, figuras y bibliografía. |
| Sustentación | 7 % | Claridad, dominio técnico, demostración en vivo y respuestas. |
| **Total** | **100 %** | Los pesos no son puntuaciones ya obtenidas. |

La [rúbrica estructurada y su inventario de condiciones](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_cierre_guia_20261118/rubric.json) conserva esos once pesos, las reglas condicionales y los nueve apartados de 14.5. Sus campos de nota y tope actuales están vacíos; el archivo sirve para comprobar concordancia, no para emitir una calificación automática.

## 14.3 Cálculo de la nota final

Si el evaluador determina un cumplimiento por componente $q_i$ entre 0 y 100 y aplica los pesos $w_i$ de la tabla, el puntaje ponderado queda en la escala de 0 a 100:

$$
P_{\mathrm{ponderado}}=\sum_{i=1}^{11}\frac{w_iq_i}{100},
\qquad
\mathrm{Nota\ final}=\min\left(\mathrm{Tope},\;5{,}0\times\frac{P_{\mathrm{ponderado}}}{100}\right).
$$ (eq-e3-final-grade)

En la ecuación {eq}`eq-e3-final-grade`, el tope se obtiene de las condiciones evaluadas de 14.1; no es un componente adicional de la suma. Alcanzar un tope de 5,0 no implica obtener esa nota: se necesita también el 100 % del puntaje ponderado. Los cuatro niveles de 14.4 no reciben equivalencias numéricas automáticas. La aplicación orientativa solicitada se documenta por separado en 14.10, con notas explícitas entre 0 y 5, mientras los campos oficiales del registro histórico permanecen vacíos.

## 14.4 Niveles de desempeño por componente

Los niveles describen criterios de la guía; no clasifican automáticamente los componentes actuales.

| Nivel | Criterio de la guía |
|---|---|
| Sobresaliente | Protocolo fechado antes de los experimentos; hipótesis falsable y predicciones verificadas; demostración formal con verificación numérica; fuente y adaptación justificadas; Nd y evidencia independiente; familia completa con Holm; potencia por contraste y conjunta; sensibilidad y reproducción independiente exitosa. |
| Satisfactorio | Protocolo correcto y reproducible, pruebas adecuadas e interpretación sólida, con omisiones menores de robustez o documentación. |
| En desarrollo | Implementación funcional con debilidades de método: presupuesto desigual, pocas semillas, falta de control familiar, bitácora o potencia, o conjuntos elegidos con conocimiento del resultado. |
| Insuficiente | Fuga, selección posterior de semillas/datos/métricas/rivales, relajación del estándar, copia de implementación, transferencia sin citar, resultados irreproducibles o veredicto sin evidencia. |

## 14.5 Condiciones para alcanzar la nota 5,0

La guía exige simultáneamente R1/R2/R3 acreditadas y 100 % de la rúbrica. **Una exigencia sin evidencia no puede marcarse cumplida**, aunque esté descrita en un plan. Las matrices siguientes distinguen evidencia parcial, pendiente y alcance no aplicable por justificar; no usan casillas aprobadas ni porcentajes de avance.

«Parcial» identifica evidencia real que todavía no cubre todo el requisito. «Pendiente» indica que falta ejecutarlo o acreditarlo, sin convertir esa ausencia en un fallo científico demostrado. Los documentos de diseño y protocolos en borrador son evidencia documental; no sustituyen resultados, firmas, recibos de CI o validación externa.

Se conservan las **45 condiciones** de la guía, distribuidas en nueve apartados: **7 + 6 + 6 + 3 + 4 + 6 + 5 + 7 + 1**. Son condiciones de evidencia que pueden apoyar más de un componente, no 45 componentes ponderados ni un porcentaje de avance que se obtenga contando filas. Los pesos repetidos en los encabezados de 14.5 identifican esa relación y no se vuelven a sumar. Una adaptación a F2a o un «no aplica» requieren justificación específica; no equivalen automáticamente a una casilla aprobada.

### 14.5.1 Contribución y novedad

Corresponde a R1/R2 y a revisión/diseño —8 %— e implementación/demostración —10 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Mecanismo nuevo y transferible para el problema, con diferencia algorítmica sustantiva. | Dictamen IA favorable y acotado para R1 de V3; no para la aditiva. | {ref}`6.18 <e3-r1-substantive-verdict-20261120>`: ley de U, mapa por A antes de la esperanza y MSE/CRPS, contrastados con MDN, FQF y regresión de desviaciones móviles. | Declaración revisable bajo nuevos antecedentes y su cobertura documentada; no es aprobación del docente ni prueba de ausencia mundial. No exige ganar el piloto. |
| Si hay transferencia: fuente citada, justificación y adaptación documentadas en `experiments/procedencia.md`. | Procedencia del corte anterior preservada y revisión versionada de V3 disponible. | {download}`Procedencia V3 revisada <../_static/e3_r1_substantive_20261120/procedencia_v3.md>` y matrices de 6.18; fe de erratas WP11/WP15. | No inventar inspiración histórica si la fuente se encontró después. El dictamen apoya la adaptación concreta, no novedad de cada ingrediente conocido. |
| Limitación de los métodos, preferiblemente del paper guía, citada o cuantificada por piloto. | Limitación matemática delimitada; utilidad empírica de la solución no confirmada. | La lectura puntual no identifica la ley y el segundo momento no determina en general la media transformada; propiedad y cien casos previos revisados en {ref}`7.15 <e3-r1-formal-scope-20261120>`. | No afirmar que la aditiva sea incorrecta o incapaz de pronosticar, ni trasladar sin evidencia una limitación de TimeMixer a BTC. El piloto adverso no demuestra beneficio de la solución completa. |
| Hipótesis falsable y predicciones secundarias anteriores a los experimentos finales. | Plan de desarrollo ejecutado con resultado adverso; confirmación pendiente. | {ref}`8.19 <e3-v3-pilot-completed-20261120>` conserva las cinco recetas, tres semillas finales, 55 fechas y la regla previa del 2 %, que no se superó. | Una confirmación futura necesita hipótesis, márgenes y reglas propios anteriores a sus resultados. No se cambia la regla ni se elige otra métrica tras este piloto. |
| Propiedad formal, supuestos explícitos y verificación numérica. | Revisada con alcances separados por arquitectura. | {ref}`7.15 <e3-r1-formal-scope-20261120>`: soporte y esperanza bajo ley finita, Jensen/oracle con supuestos; diez insumos, 28 coeficientes racionales y cien casos previos verificados. Las pruebas históricas conservan su objeto. | No prueba calibración de mercado, convergencia o menor RMSE que una aditiva flexible. No se exige un teorema inédito ni beneficio predictivo para dar por correcta la propiedad. |
| Núcleo escrito desde la formulación, sin importar el modelo ni copiar el núcleo ajeno; comprensión defendible. | Código inspeccionable y auditado; defensa del estudiante por observar. | Código V3 fijado al piloto, correspondencia de ecuaciones y funciones en 6.18; primitivas de PyTorch y módulos locales, con IA declarada. | La revisión técnica de R1 no sustituye comprensión, explicación y modificación por el estudiante para R2; tampoco imputa copia por no haber observado su defensa. |
| Matriz y declaración de novedad, búsqueda repetida al final y verificación por un tercero. | Revisión acumulada y dictamen específico por IA disponibles bajo la modalidad autorizada. | {ref}`6.18 <e3-r1-substantive-verdict-20261120>`: cobertura histórica de tres bases, lecturas focales, seguimiento desde cinco semillas, repositorios y dos cadenas repetidas por el revisor IA. | Las nuevas llamadas API bloqueadas no cuentan como respuestas bibliográficas; los accesos parciales y límites de cobertura se conservan. No se inventa una firma humana o aprobación institucional. |

### 14.5.2 Comparación justa

Corresponde a R3 y evaluación ampliada/robustez —10 %—. Los controles propuestos se detallan en la [sección 8](08_verificacion_estadistica.md).

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Benchmarks fuertes/recientes, ajustados y congelados después del Bloque A. | Parcial heredado. | Dieciséis sistemas, recetas y resultados E2; auditorías en [sección 4](04_correcciones_e2.md). | Cierre A, controles aplicables y anclas del benchmark confirmatorio; el freeze histórico no cierra A automáticamente. |
| Familia completa, sin selección posterior de rivales. | Familia histórica y auditoría exploratoria de 19 sistemas calculadas. | La familia histórica conserva **16 rivales** y 48 contrastes sobre 608 fechas. La auditoría nueva añade dos controles y completó sus 19 sistemas en D00–D03; {ref}`8.18 <e3-final-extended-audit>` enlaza los 76 paneles y análisis. | Cerrar la justificación de equidad, potencia y requisitos confirmatorios. El m≈12 ilustrativo no reduce ninguna familia ni sus resultados adversos. |
| Mismo presupuesto, particiones, preprocesamiento y fuente de pesos preentrenados. | Diferencia histórica documentada; procedimiento común exploratorio completado. | E2 histórico: 655 ejemplos de ajuste y 53 de calibración; E3: 532 de ajuste y 116 de validación. La auditoría nueva ejecutó la familia bajo el procedimiento y presupuesto prefijados; contratos, desviaciones y recursos se conservan en {ref}`8.18 <e3-final-extended-audit>`. | Revisar consumos y comparabilidad efectiva, incluida la medición completa de recursos; no atribuir resultados exclusivamente a la arquitectura ni asumir que diez semillas o una misma forma de entrada prueban equidad. |
| Capacidad controlada con parámetros/FLOPs y benchmark de tamaño comparable si el propio es mayor. | Parcial de inventario. | Parámetros del candidato y costes heredados con alcance, [secciones 5–6](05_configuraciones_exportacion.md). | Medición comparable y variante ajustada en tamaño si procede. Para esta casilla de 5,0, discutir el tamaño no sustituye esa variante. |
| Estándar/protocolo con fecha y tag previos; confirmatorio y exploratorio separados. | Auditoría exploratoria registrada antes de sus ajustes; cierre confirmatorio pendiente. | El borrador histórico `experiments/protocolo.md` conserva sus campos abiertos. La auditoría de 8.17 tiene un protocolo propio y tag anterior a los ajustes, con fuente y hashes fijados. | Completar el estándar y el dimensionamiento de la batería confirmatoria. El tag de la auditoría no registra retrospectivamente el análisis histórico ni acredita R3. |
| Bitácora Nd y desarrollo separado de confirmación. | Conciliación documental del corpus completada, con límites explícitos. | {ref}`7.14 <e3-design-inventory-reconciled>`: 15 recetas neuronales implementadas, cuatro controles, tres propuestas, 183 ensayos HPO y 844 trabajos de ámbitos separados; 21 fuentes cotejadas. | Un bosquejo Faithful requiere especificación para decidir su identidad. No se certifican ideas sin archivo, todas las invocaciones físicas ni un preregistro histórico. Los calendarios expuestos siguen siendo desarrollo. |

### 14.5.3 Evidencia estadística

Corresponde a R3 y pruebas/adecuación —15 %—. Las unidades, estimandos y reglas de decisión deberán quedar cerrados antes de confirmar.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Al menos diez semillas o más por potencia, media/DE/IC95 % bootstrap, MDE y potencia conjunta. | Réplicas y descripción disponibles; dimensionamiento pendiente. | Trece redes E2 con diez semillas más tres clásicos deterministas: **133 predictores E2**; diez semillas E3. Media, DE e intervalos con ámbitos declarados en [sección 8](08_verificacion_estadistica.md). | Piloto y diseño previo para R/calendario suficientes, MDE y potencia por contraste y conjunta. Diez semillas no prueban potencia; copiar un clásico determinista no crea réplicas. |
| Tres pruebas admisibles por contraste, Holm y potencia ≥0,90 por contraste. | Tres procedimientos posteriores ejecutados; batería confirmatoria y potencia pendientes. | Bootstrap temporal, influencia RMSE con HAC y t entre semillas: **48/48 rechazos** con Holm global48, máximo p ajustado 0,0019. | Justificar pertinencia F2a y supuestos de cada procedimiento, que condicionan ámbitos distintos. La guía exige pruebas distintas apropiadas, no identidad literal de ámbitos. Los rechazos retrospectivos no acreditan potencia ni toda R3. |
| Efecto con IC y relevancia frente a δ. | Disponible con alcance retrospectivo. | Ganancia de RMSE frente al mejor rival descriptivo LSTM-autoencoder: **0,0009334571**; IC95 % temporal marginal **[0,0006329359; 0,0012131746]**; δ heredado **0,0002577284**. Efectos de los 16 rivales descargables en [sección 8](08_verificacion_estadistica.md). | Los IC nominales no son simultáneos. Cohen dz entre semillas condiciona el calendario; no representa variación entre mercados ni permite convertir δ de RMSE en δ² de MSE. |
| Comparación global que incluya al original. | MCS auxiliar posterior calculado. | MCS90 Tmax de los 17 sistemas, sobre MSE diario medio entre semillas: conserva **diez sistemas, incluido E3**; p final 0,1442. | No identifica ganador único, equivalencia ni ganancia práctica de RMSE. No sustituye la selección histórica de E2 o una comparación confirmatoria bajo protocolo común. |
| Prueba entre conjuntos: Friedman–Iman–Davenport y Holm, o bayesiana con ROPE. | Comparación descriptiva entre conjuntos disponible; adecuación inferencial pendiente. | La auditoría completa aporta métricas y rangos entre D00–D03 en {ref}`8.18 <e3-final-extended-audit>`, con dependencia entre conjuntos explícita. | Justificar unidad independiente, método y límites de potencia y ejecutar la prueba admisible que corresponda; una tabla de rangos no es por sí sola Friedman–Iman–Davenport. No promediar RMSE de escalas diferentes. |
| Sensibilidad a prueba, métrica secundaria y subconjunto de semillas. | Disponible con alcance posterior. | Bloques/rezagos 7 y 28 conservan 48/48 rechazos; MAE secundaria con Holm16 rechaza 16/16 frente a cero; subgrupos descriptivos de primeras/últimas cinco semillas publicados. | No reemplazan sensibilidad a hiperparámetros, perturbaciones o nuevos conjuntos. No se inventa margen práctico de MAE ni se usa Nadeau–Bengio para días/semillas de una sola cronología. |

La potencia no se certifica por cumplir R≥10 ni multiplicando potencias marginales dependientes. El dimensionamiento debe considerar la regla completa —Holm, tres pruebas y familia—, la alternativa frente a la frontera δ y la dependencia temporal. Semillas×días no son observaciones iid. Estas exigencias no se relajan ante resultados adversos.

### 14.5.4 Generalización de la evidencia

Corresponde a R3 y evaluación ampliada/robustez —10 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Al menos tres conjuntos de naturaleza distinta, elegidos antes de conocer resultados y fuera del diseño. | **Tres archivos admitidos y evaluación exploratoria aditiva completa**; ninguno acredita confirmación de la candidata actual. | {ref}`8.18 <e3-final-extended-audit>`: D01 WIKI/AAPL, D02 ETHUSDT Spot y D03 BTCUSD_PERP, además de D00, completaron los 19 sistemas y el análisis. | Justificar dos ejes de diferencia por adicional y una evaluación no usada para diseñar, con dependencias y potencia documentadas. F00–F03 siguen sin admisiones verificadas; la {ref}`revisión de selección <e3-data-selection-review-20261121>` pausa la adquisición propuesta de acciones. Tres archivos o tickers distintos no acreditan automáticamente esas condiciones. |
| Evaluación bajo cambio de distribución: periodo, dominio o fuente. | Comparación exploratoria entre conjuntos disponible. | D01–D03 y diagnósticos temporales/perturbaciones calculados en {ref}`8.18 <e3-final-extended-audit>`, con exposición y dependencia declaradas. | Justificar qué cambio está aislado y qué datos se mantuvieron fuera del diseño; un tramo conocido no adquiere independencia al renombrarlo. |
| Mejor media en al menos dos de tres adicionales y ausencia de inferioridad significativa frente a cualquier rival en cualquiera. | Resultados por conjunto disponibles; R3 no acreditada. | Métricas y contrastes de la familia completa en {ref}`8.18 <e3-final-extended-audit>`; se conservan resultados favorables, adversos y no concluyentes. | Evaluar el criterio completo con pruebas apropiadas y potencia, sin escoger rivales o conjuntos posteriores. Falta de rechazo de inferioridad no equivale a equivalencia o no inferioridad demostradas. |

### 14.5.5 Validez del modelo

Corresponde a pruebas/adecuación —15 %— y correcciones E2 —7 %—. La ruta es **pronóstico temporal F2a**: ECE y Brier de clasificación no se trasladan directamente a σ₇ continua.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Auditoría de fuga: duplicados, unidad de partición, preentrenamiento y exposición del test. | Parcial, con auditoría posterior incorporada. | Cierre de datos E2: **12 PASS, 3 PARTIAL y 1 NA**, con 1316 ventanas y las 608 de prueba disponibles; [sección 4.2](04_correcciones_e2.md). Referencia E3 sin backbone preentrenado y exposición histórica explícita. | Cerrar controles parciales y la auditoría integral del pipeline candidato. Las ventanas verificadas o un control sin fallo no acreditan ausencia integral de fuga ni independencia de calendarios conocidos. |
| Tablas A y B de ajuste/sobreajuste superadas. | Brechas y diagnósticos retrospectivos disponibles; aprobación formal pendiente. | {ref}`8.20 <e3-existing-adequacy-close-20261120>` añade brechas entre particiones y controles triviales en D00–D03, residuos de los pesos históricos y de la auditoría ampliada, con versiones y familias diagnósticas separadas. | No existía un margen de brecha prospectivo del candidato. Cocientes de RMSE, no rechazar un diagnóstico o un error de prueba menor no completan las tablas ni acreditan potencia/adecuación confirmatoria. |
| Calibración antes/después de la recalibración, adaptada a la tarea. | Diagnóstico histórico, perfil individual de servicio y calibración exploratoria ampliada disponibles. | [8.11](08_verificacion_estadistica.md) conserva diez cuantiles y 530 forwards CPU; el servicio usa el q existente de 271801. {ref}`8.18 <e3-final-extended-audit>` añade calibradores y evaluación de intervalos en los cuatro conjuntos, separados de esos artefactos históricos. | Completar la calibración probabilística/regresión que exija la salida y su validación temporal confirmatoria. La aceptación pública histórica ya está comprobada en 10.10; no sustituye esa validación. Un punto y un intervalo no definen toda una distribución predictiva. |
| Incertidumbre: cobertura empírica frente a nominal y por subgrupo. | Diagnósticos históricos y ampliados calculados; aceptación pública histórica comprobada para H. | El diagnóstico histórico conserva 94,49 % global y 86,22 % en 2026Q1 entre diez semillas; el predictor servido 271801 tiene 94,08 % y 85,56 %. La auditoría nueva aporta cobertura/anchura y subgrupos temporales por conjunto en {ref}`8.18 <e3-final-extended-audit>`. | Justificar subgrupos, dependencia y cambio temporal sin convertir cobertura histórica en garantía futura. Las cifras de la campaña nueva no se atribuyen a los pesos o calibrador históricos del servicio. |

### 14.5.6 Análisis que explican el resultado

Corresponde a implementación/demostración —10 %— y evaluación ampliada/robustez —10 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Ablación, incluidas adaptaciones, con las mismas pruebas, familia propia y control de capacidad. | Tres componentes de V3 aislados en desarrollo; batería confirmatoria pendiente. | {ref}`8.19 <e3-v3-pilot-completed-20261120>`: retirar CRPS, fijar masas y sustituir la agregación exacta; mismo protocolo, selección y tres semillas, con menor capacidad declarada para masas fijas. La receta completa fue 2,70 % peor que el mejor control. | Este piloto de 55 fechas conocidas no aporta tres pruebas de superioridad confirmatoria por contraste ni su potencia. No se promociona automáticamente la ablación ganadora. |
| Sensibilidad a hiperparámetros propios y estabilidad por semilla. | Réplicas históricas y sensibilidad exploratoria ampliada calculadas. | Diez semillas históricas aditivas y nueva evaluación de las tres tasas de aprendizaje prefijadas con su semilla HPO; tablas y alcance en {ref}`8.18 <e3-final-extended-audit>`. | La sensibilidad usa una semilla HPO y no equivale a diez semillas en cada configuración. Justificar criterios, cubrir los hiperparámetros pertinentes y conservar todos los ensayos, sin escoger la región favorable. |
| Robustez a ruido, perturbaciones y reducción de muestra. | Estudios exploratorios ejecutados en D00–D03. | Tres severidades de ruido causal D7, escenario limpio y curvas con fracciones 25 %, 50 % y 100 % y tres semillas para candidato/TSMixer; tablas y 16 figuras en {ref}`8.18 <e3-final-extended-audit>`. | Separar estrés artificial de cambio plausible. En las curvas, tamaño y extensión temporal varían conjuntamente; los clásicos mantienen historia cruda limpia. Completar adecuación y alcance confirmatorio, no tratar estas curvas como un efecto aislado del tamaño. |
| Interpretabilidad con fidelidad, estabilidad, comparación con mejor benchmark y descarte de atajos. | Sensibilidad local disponible; estudio completo pendiente. | El dashboard calcula intervenciones sobre la entrada del mismo artefacto ONNX y describe sus efectos. | Comparación global con el mejor SAE, fidelidad, estabilidad entre semillas y descarte de atajos; la sensibilidad local no acredita explicación causal. |
| Errores y escenarios donde no supera a los benchmarks. | Diagnósticos históricos disponibles; estudio confirmatorio pendiente. | El dashboard muestra trayectorias y residuos de representantes fijos; se conserva el diagnóstico posterior de las 18 fechas de objetivo alto frente a M00 proyectado. | Escenarios prefijados, errores compartidos y causas discutidas sin inferencia causal injustificada. Un grupo definido por el objetivo realizado no estaba disponible al pronosticar. |
| Parámetros, FLOPs, latencia, memoria, entrenamiento y energía con Pareto. | Inventario de ingeniería y auditoría de recursos documentados; medición integral pendiente. | {ref}`7.14 <e3-design-inventory-reconciled>` y {ref}`8.20 <e3-existing-adequacy-close-20261120>` separan 900 fases de 796 trabajos y conservan tres registros con cómputo de caída desconocido; los tiempos de desconexión no se convierten en GPU activa. | FLOPs, energía y comparabilidad total de hardware/fronteras no se reconstruyen de registros incompletos. Las oportunidades nominales de presupuesto no prueban consumo efectivo idéntico. |

### 14.5.7 Reproducibilidad

Corresponde al informe —14 %— y configuraciones/artefactos —7 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Archivo con DOI, Docker, semillas y determinismo documentados. | Contenedores de servicio y reanálisis comprobados; DOI del suplemento documental publicado y ZIP exacto verificado en {ref}`11.10 <e3-zenodo-publication-verified-20261120>`. | [Sección 11](11_informe_tecnico.md): archivo determinista con SHA-256, Docker científico desde extracción limpia, once CSV idénticos y discrepancias por campo/unidad cero. Sin entrenamientos o inferencias. | Corrección de metadatos del DOI documental, archivo integral, derechos fuera de los componentes propios seleccionados y reproducción humana del alcance que fije el protocolo; los entrenamientos históricos conservados no se repiten por rutina. La igualdad de agregados no acredita determinismo universal o entre plataformas. |
| Maestro por ejecución que genere informe y dashboard. | Maestro histórico consistente y análisis ampliado disponibles; integración final parcial. | El catálogo de 17 sistemas deriva de su tabla histórica y tiene una prueba de igualdad de valores. La {ref}`auditoría de 8.18 <e3-final-extended-audit>` añade métricas y diagnósticos de la familia de 19 sistemas en D00–D03. | Completar recursos comparables y la integración explícita de la versión final con el servicio. No sustituir silenciosamente el catálogo, los pesos o el calibrador históricos; las curvas representantes no son medias de diez modelos. |
| Pruebas unitarias ejecutadas en CI. | Evidencia local y CI identificada por corte y alcance. | Los 42 casos estadísticos históricos conservan su recibo. El [CI del dashboard del corte de base](https://github.com/jassan-bit/DeepLearning_202630/actions/runs/44213072051) aprobó sus 25 casos y seis recorridos de navegador; el job científico aprobó 60 y omitió una referencia opcional SciPy, incluida en los 61 aprobados localmente. La comprobación A/B conserva 50 casos aprobados. La actualización de servicio conserva por separado 29 pruebas locales y seis recorridos de navegador, con cinco inferencias comprobadas en contenedor; [capítulo 9](09_dashboard_predictivo.md). | Conservar los recibos de la versión final sin sumar coberturas distintas como una campaña científica. Ni estos checks ni los casos históricos certifican reproducción humana, defensa académica o servicio Render. |
| Datos o instrucciones exactas de obtención y licencias verificadas. | Tres archivos adicionales admitidos y evaluados; derechos del paquete final parciales. | Cohorte e insumos históricos conservados; D01 WIKI/AAPL y D02/D03 disponen de fuentes, calendarios, comprobaciones y hashes públicos, con políticas declaradas en {ref}`8.18 <e3-final-extended-audit>`. | Ratificar condiciones/derechos de distribución del paquete final y proporcionar sus instrucciones exactas. La evaluación completa no equivale a reproducción independiente de todos los pesos. |
| Reproducción independiente desde cero en contenedor, por persona ajena al código, con discrepancia máxima. | Humana pendiente; ruta automática completa del libro y tablas principales comprobada. | La {ref}`sección 11.12 <e3-next-human-book-20261121>` reconstruyó el corte `f5c960e`: 39 páginas, 76 filas principales aditivas y siete V3 visibles, con comparaciones por campo/unidad. Los recibos A/B anteriores permanecen separados. | Persona ajena al código, ejecución propia desde cero, alcance declarado, discrepancias y concepto auténtico. La guía admite precálculos costosos versionados con su generador; no impone C universalmente. La automatización no se atribuye a un tercero. |

### 14.5.8 Productos del entregable

Corresponde a correcciones —7 %—, artefactos —7 %—, dashboard —8 %, 9 % y 5 %—, informe —14 %— y sustentación —7 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Correcciones E2 con trazabilidad completa y recálculo consistente. | Parcial. | Tabla de revisión interna y auditorías de [sección 4](04_correcciones_e2.md). | Cerrar los controles pendientes; recalcular dependencias y ranking antes/después si una corrección modifica resultados. No se recibieron observaciones del docente. |
| Mejores configuraciones exportadas, ida y vuelta y `metadata.json`. | Catorce representantes neuronales ONNX verificados; las vistas originales conservan cinco y `/perfil-14` sirve las catorce, con aceptación pública para H. | MLP16, LSTM32, GRU32, E3 y TSMixer conservan sus recibos de la semilla 271801. El cierre posterior verifica las nueve redes restantes en 608 ventanas de prueba y 53 de calibración por red, sin entrenamiento ni cambio de calibradores. Los tres clásicos requieren más historia/estado que siete barras. [Capítulo 5](05_configuraciones_exportacion.md) y {ref}`cierre técnico <e3-completion-20261120>`. | La paridad y la aceptación pública del perfil14 ya están ejecutadas en 10.10. Permanecen los costes completos comparables, la capacidad y disponibilidad sostenidas no medidas y los requisitos científicos de R3. Los tres clásicos conservan su justificación de contrato. |
| Dash Plotly en Render, prueba de humo, contingencia y funcionamiento en sustentación. | Aceptación pública aprobada para H `feb83dbc08197bde7076857bd8ff475de197efd2`; clientes de cinco y catorce modelos. | Las vistas originales de tres páginas y cinco modelos y el perfil ampliado de catorce conservan sus recibos públicos de HTTPS, predicciones, intervalos, explicaciones y guardas. El Manual Deploy de H y la aceptación con SHA efectivo están comprobados; {ref}`10.10 <e3-render-public-acceptance-20261120>` conserva la evidencia y el alcance, separado del SHA del libro. [URL de Render](https://deeplearning-e3-dashboard.onrender.com). | Repetir aceptación ante cambios del código o los artefactos; no hay un despliegue pendiente de H para este ensayo. La concordancia del SHA autodeclarado no es una atestación completa ni mide capacidad o disponibilidad sostenidas. Completar contingencia y funcionamiento durante la sustentación. |
| Jupyter Book sin errores, ficha de modelo y declaración de datos. | Publicación técnica de base comprobada; cierre científico pendiente. | Fuentes, resumen con aportes técnicos, mapa científico 1–16, ecuaciones numeradas y bibliografía en [11](11_informe_tecnico.md); MODEL_CARD.md y DATA_STATEMENT.md del prototipo. La [matriz de 12](12_lista_entregables.md) relaciona los trece productos con su cierre. | Reconstruir/verificar la versión final y ratificar derechos, contenidos y responsabilidades humanas; los archivos no son una firma ni cierran R1/R2/R3. |
| Procedencia del mecanismo transferido, cuando corresponda. | Documento histórico preservado y revisión V3 versionada con dictamen técnico favorable y acotado. | {download}`Procedencia V3 <../_static/e3_r1_substantive_20261120/procedencia_v3.md>`, matrices y {ref}`6.18 <e3-r1-substantive-verdict-20261120>` vinculan fuentes, cambios, lecturas y límites. | Defender la declaración sin atribuir invenciones a componentes conocidos ni inventar inspiración histórica. R2/R3 y la aprobación del docente no se acreditan por este concepto. |
| Declaración de autoría, contribuciones individuales y uso de IA. | Parcial documental; validación pendiente. | Asistencia de IA reconocida en código, teoría y procedencia. | Declaración final firmada, contribuciones individuales y comprensión responsable del autor; la asistencia declarada no acredita una firma ni defensa. |
| Sustentación: demostración en vivo y capacidad de explicar/modificar el algoritmo. | Ensayo y defensa pendientes. | [Guion de 20 minutos](13_sustentacion.md), con 18 de contenido y dos de margen, nueve preguntas y comandos de práctica comprobados. | Defensa efectiva por el estudiante, preguntas técnicas y modificación/ejecución en vivo sobre la versión identificada; el guion y sus tests no reemplazan esa evidencia. |

### 14.5.9 Decisión previa al congelamiento del protocolo

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Definir antes del cierre el hallazgo alternativo que sostendrá el trabajo si no se acredita superioridad significativa y consistente. | Propuesta pendiente de fijar. | Límites y resultados adversos se conservan en el diseño de [secciones 7–8](08_verificacion_estadistica.md). | Registrar el hallazgo alternativo, sus análisis y criterios antes de confirmar; no escogerlo después por ser favorable. |

Se propone como hallazgo alternativo un **benchmark reproducible y una explicación delimitada de los escenarios en que la regla candidata aporta o perjudica**. Si no supera a toda la familia, el informe conservaría las diferencias/IC, potencia y MDE, costes y ablaciones, y distinguiría falta de evidencia, efecto pequeño y deterioro observado. La relación entre acotar la cabeza y predecir colas de σ₇ sería una hipótesis a contrastar, no una explicación ya demostrada. Esta propuesta debe precisarse antes del congelamiento y no reemplaza el objetivo confirmatorio ni permite relajar R3.

La consecuencia de un R3 evaluado y no satisfecho, bajo R1/R2 acreditadas, es la regla condicional descrita en 14.1. No se aplica aquí a condiciones aún pendientes. La selección posterior de semillas, conjuntos, métricas, análisis o rivales para sostener superioridad contraviene la guía y anula esa afirmación; se conservarán el historial, los resultados desfavorables y las desviaciones declaradas.

La [sección 15](15_integridad_academica.md) desarrolla esas reglas de integridad y la [16](16_conclusiones.md) conserva las conclusiones compatibles con la evidencia actual. El [Apéndice A](apendice_a_plantillas.md) ofrece registros editables sin marcar requisitos como cumplidos por disponer de una plantilla.

## Evidencia posterior que modifica la interpretación

Los cortes de 6.11–6.13 descritos a continuación conservan el estado de la referencia y de aquella revisión. La evaluación específica posterior de V3 se expone en 6.18 y 14.6; sus conclusiones no se transfieren a la aditiva.

El [control común del capítulo 8](08_verificacion_estadistica.md) conserva la ventaja RMSE descriptiva histórica de **10,33 %** frente al mejor rival proyectado y el **3,70 %** de mejora MAE del control sólo piso frente a E3. Son cortes anteriores a la {ref}`auditoría exploratoria completa de 8.18 <e3-final-extended-audit>`, que aporta 796 trabajos, los 19 sistemas en D00–D03, análisis y figuras terminados. La revisión de antecedentes y los cálculos no acreditan R1 o R3 automáticamente. Persisten la justificación de independencia, la potencia prospectiva de la batería y los requisitos científicos restantes; completar la ejecución no asigna un tope mayor.

El [expediente de R1 de 6.11](06_modelo_original.md) conserva comparación formal, trazabilidad y un formulario externo vacío. La región admisible de la cabeza aditiva tiene un antecedente de guarda por muestra; la equivalencia escalar no identifica clases neuronales o entrenamiento. La ejecución posterior de **6.12** ya tiene respuestas verificadas de tres bases, con 180 ocurrencias de metadatos y un cribado de títulos separado del recibo original. Las consultas amplias y el examen limitado a primeras páginas no acreditan exhaustividad. Ese corte no cerró la diferencia sustantiva de la aditiva. La revisión posterior específica de V3 se incorpora en 6.18. Conforme a la indicación docente comunicada por el autor, 6.13 adopta la revisión por IA declarada como sustitución del revisor humano para ese componente y documenta sus dos consultas repetidas. No se marca R1 como acreditada por los metadatos, el cribado o los dictámenes asistidos; tampoco se asigna aquí una nota o un tope al proyecto.

La indicación de **6.13** retira el carácter humano como condición de modalidad para esa revisión externa, con uso de IA mencionado. El componente se documenta como completado mediante dictamen y dos cadenas repetidas por IA. Esto no convierte el concepto crítico en una acreditación favorable de novedad ni completa el resto de 14.5.1; R1, la defensa de R2 y R3 conservan sus pendientes científicos y académicos.

La {ref}`adecuación retrospectiva de 8.20 <e3-existing-adequacy-close-20261120>`
añade evidencia cuantitativa a las casillas de ajuste/sobreajuste: doce brechas
emparejadas por semilla en D00–D03, cinco controles triviales por conjunto,
diagnósticos de residuos sobre cuarenta trayectorias de prueba limpia de la
auditoría ampliada y cobertura por trimestre. Los cuatro paneles de origen y
los extractos están fijados por hashes, y sus cuarenta RMSE/MAE coinciden con las
tablas originales. Los diagnósticos Holm160 de esa ejecución permanecen
separados del antecedente de diez semillas y 608 fechas —Holm40— y del piloto V3.
Los resultados adversos de MAE, heterocedasticidad y calibración lineal se
conservan. No se declara aprobada la adecuación: continúan faltando un margen y
reglas de decisión oportunos, potencia prospectiva y las restantes condiciones
confirmatorias. Las pruebas de residuos no sustituyen la batería de
superioridad R3. La actualización no asigna una nota ni un tope.


(e3-r1-evaluation-scope-20261120)=
## 14.6 Alcance del dictamen específico de R1 de V3

La revisión técnica de 6.18 da por sustentada documentalmente la adaptación de lectura y pérdida de V3 bajo el criterio de la guía y la cobertura declarada. La aditiva no hereda esa conclusión; V3 no hereda los 796 trabajos. La utilidad del piloto sigue siendo desfavorable y no se cambia la regla ni la propuesta por una ablación que ganó después.

El concepto IA no acredita R2 o R3 ni completa las 45 condiciones. No es una nota del docente, no hace oficial un tope y no convierte la modalidad de tercero por IA en una defensa del estudiante. La cobertura histórica efectiva y las búsquedas nuevas se distinguen de los intentos API sin respuesta. Las declaraciones anteriores de R1 pendiente conservan su corte; la matriz actual está en 6.18.


## 14.7 Alcance posterior de la evidencia, sin nota automática

Los 20 nuevos ajustes CPU de V4 y sus 1.210 salidas son desarrollo expuesto y terminaron con abandono por la regla fijada. No heredan R1 de V3 ni validación de la aditiva. La validación controlada de los procedimientos detectó inflación de rechazo y no aprueba la batería; completar el código no acredita R3.

La guía pide pruebas de igualdad con dirección y margen práctico, efectos/intervalos y las demás condiciones simultáneas; no exige literalmente tres p-valores unilaterales del nulo más fuerte `theta <= delta`. La propuesta anterior de ese nulo conserva su identidad. MCS aparece entre los procedimientos admitidos, pero la extensión ensayada requiere una traducción contrastiva justificada y falló su calibración. La {ref}`sección 8.25 <e3-next-statistics-data-20261121>` conserva las distinciones y los datos de validación.

La nueva ruta de contenedor cierra una preparación técnica de reproducción del corte fijado; falta la persona real y su concepto. Las reglas condicionales R1/R2/R3 y los once pesos permanecen sin cambios, y este cierre no concede un tope o una nota por sí solo.


(e3-statistical-component-evaluation-20261121)=
## 14.8 Evidencia nueva del componente de hipótesis y adecuación

La {ref}`remediación de 8.26 <e3-statistical-remediation-20261121>` se valora por
lo que ejecutó y verificó, distinguiendo cuatro niveles: software, validación
estadística controlada, diagnóstico retrospectivo y confirmación. El código de
un método no acredita su tamaño o cobertura; una simulación favorable no elimina
sus supuestos ni demuestra potencia para los datos de mercado.

La nueva versión MCS pasó el cribado global individual en tres DGP nulos, con una tolerancia exploratoria explícita; la familia primaria P1/P2 mostró inflación en el caso de variación sólo entre semillas y no se certifica. La adecuación retrospectiva V4 aporta 88 diagnósticos con Holm88 y trece rechazos MZ, conservando el tamaño limitado de 55 fechas y sus supuestos. La auditoría de potencia verifica entradas y brechas, no dimensiona una campaña válida ni reanima los 340 ajustes de la receta abandonada. Las tres pruebas pertinentes, control válido de toda la familia y potencia ≥0,90 por contraste siguen sin acreditarse.

La valoración orientativa solicitada del componente del **15 %** se conserva
separada de la rúbrica oficial y no completa automáticamente sus condiciones.
Los once pesos, los campos vacíos de calificación y los topes condicionales de
14.1 permanecen intactos. No se atribuye una subida por terminar programas o
por seleccionar un resultado favorable.

La **sustentación de 4,5** es un valor indicado por el autor para el cálculo
orientativo, no una defensa observada ni una acreditación automática de R2.
La aceptación académica y la reproducción humana siguen requiriendo sus actos
auténticos. Las recetas V3/V4, sus resultados adversos y sus decisiones no se
reescriben al mejorar la herramienta estadística.


(e3-r3-route-verdict-20261121)=
## 14.9 Dictamen histórico de la ruta R3, anterior al motor temporal

La ruta de aquel corte documenta condiciones y bloqueos; **no acredita R3** ni afirma haber refutado toda propuesta futura. La guía condiciona la evaluación de R3 a R1/R2 satisfechas. Entonces no había candidata admitida, batería confirmatoria plenamente validada, datos futuros admitidos, dimensionamiento final o protocolo real anterior a resultados. La nominación input-only y el motor temporal se documentan después, en 8.29 y en la valoración actual de 14.10. Una reparación estadística no revierte la regla fallida de V3 o el abandono de V4.

El método global MCS requiere su propio nivel congelado y una correspondencia defendida con la guía; 0,10 fue una elección de desarrollo, no un valor universal obligatorio. Dos pruebas pareadas más una regla global singleton no se presentan automáticamente como cumplimiento literal de tres procedimientos por contraste con Holm. Se conservan las diferencias entre cobertura del conjunto óptimo, dirección, margen, multiplicidad y potencia de toda la decisión.

Los campos oficiales de nota y tope permanecen sin rellenar. La **sustentación 4,5** sigue siendo un dato indicado por el autor, sin afirmar una defensa observada o acreditar R2 por esa cifra. Los resultados nuevos y sus limitaciones se documentan en {ref}`8.27 <e3-r3-route-statistics-20261121>`; este corte no asigna una nueva nota global ni convierte una condición pendiente en incumplimiento demostrado.


(e3-evaluation-current-20261121)=
## 14.10 Evaluación orientativa anterior, con sustentación 4,5

**Valoración documental por IA del 21 de noviembre de 2026, hora de Bogotá, solicitada por el autor.** Se revisa la evidencia del corte `28a58e22bee25a9ab813afeac272279b91d5663a`, incorporando la aclaración de selección de datos. Es una evaluación orientativa en escala de 0 a 5; no sustituye la decisión académica ni afirma haber observado una sustentación. El **4,5 de sustentación es el valor proporcionado por el autor** y se usa exclusivamente para ese componente.

Se aplican los **once pesos del Cuadro 9, p. 60**, que suman 100 %. Las notas son juicios documentales explícitos por componente; no se inventa una equivalencia automática entre los cuatro niveles de la guía y una cifra. El {download}`registro versionado de notas, aportes y evidencia <../_static/e3_btc_scope_assessment_20261121/assessment.json>` permite repetir el cálculo y comparar futuras revisiones. Los campos oficiales del registro histórico de 14.2 permanecen sin diligenciar.

| Componente | Peso | Nota /5 | Aporte a la nota /5 | Fundamento de esta valoración |
|---|---:|---:|---:|---|
| Correcciones del Entregable 2 | 7 % | 4,4 | 0,308 | Trazabilidad, objetivo y auditorías conservados; controles D13–D15 y determinismo entre plataformas parciales. |
| Mejores configuraciones y artefactos | 7 % | 4,5 | 0,315 | Catorce representantes ONNX verificados; política común de ajuste final y costes comparables todavía parciales. |
| Dashboard: Contexto y EDA | 8 % | 4,6 | 0,368 | Contenido, interacción y diagnósticos temporales documentados, con límites explícitos. |
| Dashboard: Modelos | 9 % | 4,4 | 0,396 | Catálogo, perfil14, intervalos y explicaciones disponibles; fidelidad de explicaciones y eficiencia completa limitadas. |
| Dashboard: diseño, UX y despliegue | 5 % | 4,6 | 0,23 | Aceptación pública de Render H comprobada; capacidad, disponibilidad sostenida y accesibilidad integral no demostradas. |
| Modelo original: revisión y diseño | 8 % | 4,2 | 0,336 | Justificación propia y acotada de input-only, ecuaciones y antecedentes; nominación posterior al desarrollo declarada. |
| Modelo original: implementación y demostración | 10 % | 4,1 | 0,41 | Mecanismo, propiedad delimitada, equivalencia y recuperación verificadas; interpretación y evaluación completa de la candidata parciales. |
| Evaluación ampliada y robustez | 10 % | 3,5 | 0,35 | Auditoría exploratoria real y extensa; falta confirmación de la candidata actual y diversidad admitida para esa evaluación. |
| Pruebas de hipótesis y adecuación | 15 % | 2,8 | 0,42 | La admisión temporal condicional mejora el método; faltan dimensionamiento suficiente y ejecución confirmatoria de mercado. |
| Informe en Jupyter Book | 14 % | 4,6 | 0,644 | Evidencias, contenedor, hashes y DOI verificables; reproducción humana pendiente y navegación editorial mejorable. |
| Sustentación | 7 % | 4,5 | 0,315 | Valor indicado por el autor para esta evaluación; no corresponde a una defensa observada por el revisor. |
| **Total ponderado** | **100 %** | | **4,092** | **4,09/5 al redondear a dos decimales; antes de aplicar un tope.** |

El cálculo usa $N=\sum_i w_i n_i/100$, con $n_i$ entre 0 y 5. La sustentación aporta exactamente **0,315 puntos**. Primero se suma sin redondear y después se considera el tope; no se suman otra vez los pesos de las matrices de condiciones.

### Puntaje ponderado y regla de tope

El **Cuadro 8, p. 59**, requiere distinguir los escenarios de evaluación:

* **Con R1 y R2 aceptadas y R3 no acreditada:** $\min(4{,}092;4{,}0)=4{,}00$. La **nota orientativa bajo ese supuesto es 4,00/5**.
* **Si se determina que R1 o R2 no se cumplen:** corresponde el tope 3,0. Una condición pendiente de evaluación no se convierte automáticamente en ese incumplimiento.

La revisión técnica específica de input-only es una evidencia propia; no hereda el dictamen de V3 ni la comparación de la aditiva. La defensa de comprensión y modificación del código para R2 sigue requiriendo el acto auténtico del estudiante. Introducir 4,5 en sustentación no demuestra por sí mismo R2. El puntaje previo de 4,09 tampoco autoriza superar el tope 4,0 del primer escenario.

### Qué se reconoce y qué sigue limitando el resultado

Se reconocen el motor experimental completo, **282 pruebas y 40 subpruebas** del paquete fijado, la recuperación comprobada y la admisión estadística temporal **condicional a los DGP y ajustes documentados**. El asistente de preparación posterior pasó **21 pruebas de software**. Esos avances mejoran implementación y trazabilidad; no son ajustes de mercado, potencia de la campaña o confirmación de superioridad. Los 796 trabajos siguen perteneciendo a la aditiva. V3, V4 y V5 completo conservan sus resultados adversos y decisiones; no se les resta nota por declararlos honestamente ni por la modalidad autorizada de revisión bibliográfica con IA.

Los descuentos principales corresponden a **confirmación propia de la candidata**, **dimensionamiento y batería de mercado suficientes** y **reproducción humana independiente del libro y su tabla principal**. La guía admite precálculos costosos versionados con generadores documentados; no se añade una obligación universal de reentrenar toda la historia. Render H y el DOI del suplemento tienen evidencias públicas previas; no se vuelven a presentar como pendientes de despliegue o publicación.

La {ref}`revisión de conjuntos adicionales <e3-data-selection-review-20261121>` mantiene Bitcoin como principal y pausa la adquisición asistida de IBM/MSFT/XOM hasta justificarla o registrar una sustitución antes de nuevos resultados. La guía exige misma tarea y dos ejes por adicional, no acciones, Nasdaq ni una razón de tamaño 1,5×. Tampoco se da por válida una sustitución por criptomonedas sólo por cambiar ticker o periodo. Esta revisión metodológica no es un resultado experimental adverso ni una penalización por detener una descarga.


(e3-adequacy-component-reassessment-20261122)=
## 14.11 Revaloración de hipótesis y adecuación con análisis nuevos

**Evaluación orientativa por IA del 21 de noviembre de 2026, hora de Bogotá.** El componente del 15 % pasa de **2,8 a 3,0/5**. La revisión independiente compara la evidencia ejecutada en {ref}`8.30 <e3-hypothesis-adequacy-20261122>` con el registro anterior de 14.10, que se conserva sin reescribir. La sustentación permanece en **4,5 por indicación del autor**, sin afirmar una defensa observada ni acreditar automáticamente R2. La {download}`revaloración estructurada <../_static/e3_hypothesis_adequacy_20261122/guide_audit/COMPONENT_REASSESSMENT.json>` identifica el SHA del registro anterior, cada aporte, los archivos nuevos y las razones del cambio.

### Evidencia que justifica el incremento

El crédito corresponde a los **128 diagnósticos nuevos de residuos**, con sus 128 posiciones conservadas y ajuste Holm; los **64 intervalos puntuales de sesgo HAC**; y los **intervalos temporales pareados del funcional primario**, calculando la raíz por semilla antes de promediar. La revisión independiente comprobó el ZIP original, la aritmética y Holm sin repetir el remuestreo. Estas operaciones aportan análisis propios de la candidata actual que antes no estaban ejecutados.

Se mantienen los hallazgos adversos: dos de cinco trayectorias input-only rechazan MZ y los dos IC RMSE frente a global incluyen cero. No se descuenta por informar esos resultados, ni se acredita adecuación por los no rechazos de los otros diagnósticos. Los intervalos son exploratorios, puntuales y condicionados a las cinco trayectorias guardadas sobre 55 fechas; no incluyen toda la incertidumbre poblacional de inicializaciones ni demuestran cobertura finita.

**No se asignan puntos por escribir la matriz, preparar fuentes, completar software, reutilizar simulaciones antiguas o ser retenida en el MCS secundario.** El MCS mejora el reporte: siete recetas permanecen, sin ganador único. Su pérdida MSE y su familia de ocho recetas no se convierten en una inferencia de la primaria media-RMSE o de los 21 sistemas.

### Cálculo conservando los pesos y las otras notas

| Componente | Peso | Nota /5 | Aporte /5 |
|---|---:|---:|---:|
| Correcciones del Entregable 2 | 7 % | 4,4 | 0,308 |
| Mejores configuraciones y artefactos | 7 % | 4,5 | 0,315 |
| Dashboard: Contexto y EDA | 8 % | 4,6 | 0,368 |
| Dashboard: Modelos | 9 % | 4,4 | 0,396 |
| Dashboard: diseño, UX y despliegue | 5 % | 4,6 | 0,23 |
| Modelo original: revisión y diseño | 8 % | 4,2 | 0,336 |
| Modelo original: implementación y demostración | 10 % | 4,1 | 0,41 |
| Evaluación ampliada y robustez | 10 % | 3,5 | 0,35 |
| Pruebas de hipótesis y adecuación | 15 % | 3,0 | 0,45 |
| Informe en Jupyter Book | 14 % | 4,6 | 0,644 |
| Sustentación | 7 % | 4,5 | 0,315 |
| **Total** | **100 %** | | **4,122** |

El componente aporta $0{,}15	imes3{,}0=0{,}450$, frente a $0{,}15	imes2{,}8=0{,}420$ antes. La diferencia global es **+0,030**: $4{,}092+0{,}030=4{,}122$, que se muestra como **4,12/5 antes del tope**. Las otras diez notas y los once pesos no cambian. No se redondea cada aporte antes de sumar.

Con R1 y R2 aceptadas y R3 no acreditada, $min(4{,}122;4{,}0)=4{,}00$: **la valoración orientativa bajo ese supuesto sigue en 4,00/5**. Si se determina que R1 o R2 no se cumplen, corresponde el tope 3,0; una condición pendiente no se transforma automáticamente en ese incumplimiento. La nota oficial permanece sin asignar.

### Razones por las que el componente continúa en desarrollo

La nueva {download}`matriz de requisitos y acciones <../_static/e3_hypothesis_adequacy_20261122/guide_audit/results/requirements_matrix.csv>` contrasta 16 filas solapadas con 25 citas de la guía; no son 16 componentes ni nuevos pesos. La auditoría verifica 22 artefactos y conserva su versión anterior.

Persisten dos brechas de inferencia que la versión operativa bloquea antes de confirmar: **MSE agregado y media de RMSE individuales no son el mismo funcional**, y las cotas marginales de la calibración disponible no certifican por sí solas la regla conjunta principal más dos de tres adicionales. La cota conservadora $2q=0{,}09572$ no es una estimación de inflación observada ni prueba de que ese error ocurra; muestra que esas cotas no bastan para deducir control al 5 %. No se cambia la primaria, alfa o regla tras los resultados para salvar esa brecha.

Además faltan predicciones/recibos reales completos de planificación para los cuatro paneles y los 21 sistemas, dimensionamiento suficiente, Tablas A/B completas y calibración probabilística/cobertura propias de la candidata. Las nuevas fuentes son propuestas de desarrollo, no reserva auténtica ni datos confirmados. **R/T/B finales, potencia alcanzada y MDE permanecen nulos.** La implementación y la preparación de una nueva ejecución no acreditan R3.


(e3-primary-joint-rubric-scope-20261122)=
## 14.12 Evidencia nueva para revisar R1/R2 e hipótesis, sin nota automática

La {ref}`entrega de 6.21 <e3-input-only-defense-dossier-20261122>` aporta un expediente propio de la candidata y una demostración ejecutable. La {ref}`reparación estadística de 8.31 <e3-primary-joint-repair-20261122>` implementa inferencia de la media de RMSE por semilla y evalúa la decisión conjunta mediante un protocolo publicado antes de simular. El {ref}`cuaderno de 11.15 <e3-primary-joint-colab-20261122>` conserva fuentes, resultados y alcance. Estas evidencias se corresponden con revisión/diseño8 %, implementación/demostración10 % e hipótesis/adecuación15 %; sus pesos no son puntos concedidos por disponer de archivos o pruebas.

Para **hipótesis y adecuación 15 %**, existe ahora una ejecución metodológica completa: **4.608 réplicas V1**, protocolo anterior al cálculo y 4.410,373 s CPU observados. Los ocho nulos pasaron el techo de falsa aceptación conjunta, cuya mayor cota simultánea fue **0,03692582**. El control favorable pasó 512/512 veces, con cota inferior de potencia sintética conjunta y mínima por contraste **0,98513395**. Esto aporta evidencia sobre el funcional reparado y la decisión completa en los escenarios fijados; no estima potencia o superioridad de mercado.

El {download}`informe original V1 <../_static/e3_primary_joint_repair_20261122/results/VALIDATION_REPORT_V1.json>` conserva **`accepted=false`**: 16/18 límites de cobertura cumplieron exactamente y dos quedaron en 94,836605 %. La {download}`indicación académica reportada después por el autor <../_static/e3_primary_joint_repair_20261122/results/ACADEMIC_ROUNDING_DECISION.json>` permite redondear al porcentaje entero más próximo; aplicada consistentemente a los 18 límites, 18/18 cumplen. Los otros 89 límites permanecen sin redondeo. Se registra cumplimiento metodológico **bajo esa regla académica comunicada**, separado del veredicto científico original y sin inventar una aprobación docente o defensa observada. Los diagnósticos adversos de IC95 permanecen disponibles en 8.31.

V2 quedó preparada y archivada con 18 pruebas propias y 34 independientes, **sin réplicas formales ni nuevo Monte Carlo después de la instrucción del autor**. Las comprobaciones de software y el redondeo académico no acreditan R1/R2 ni R3 y no asignan puntos automáticamente.

La valoración numérica de **14.11** conserva su corte: **4,122 antes del tope**, sustentación **4,5 indicada por el autor**, once pesos y otras notas sin modificación. Esta entrega no asigna una nueva nota oficial ni registra una defensa observada. Las nuevas comprobaciones de software no se suman como entrenamientos, y la potencia del control sintético no se atribuye al problema de Bitcoin.

Con R1/R2 aceptadas y R3 evaluada como no satisfecha, permanece la regla condicional de tope4,0. Si una condición está pendiente, no se convierte automáticamente en un incumplimiento demostrado. Para evaluar R3 siguen siendo necesarios la aceptación correspondiente de R1/R2, un dimensionamiento válido con entradas reales, una reserva adecuada y la evaluación completa de la candidata; la recuperación opcional de pesos sólo aportaría adecuación retrospectiva.

El alcance de lo que se acredita es verificable: fuente original preservada, cálculo reparado, protocolo y resultados sintéticos auténticos, y material concreto para la defensa. La modalidad de IA declarada para la revisión no inventa un acto humano o una firma, y no transforma el expediente de otro modelo en aprobación de la candidata actual.


(e3-current-evaluation-20261122-v2)=
## 14.13 Evaluación actualizada con sustentación 4,5 y redondeo comunicado

**Valoración orientativa por IA del 22 de noviembre de 2026, hora de Bogotá:** **4,197/5 antes del tope, mostrado como 4,20/5**. Se revisaron las once filas de la guía sobre la fuente publicada `34c8e1c0d84a80dec8d8c318e85ef88515633c2b`. La sustentación permanece en **4,5 indicada por el autor**, sin afirmar una defensa observada. Esta revisión se solicita después del corte de 14.12, que se conserva como antecedente; no es una nota oficial.

| Componente | Peso | Nota /5 | Aporte ponderado |
|---|---:|---:|---:|
| Correcciones del Entregable 2 | 7 % | 4,4 | 0,308 |
| Mejores configuraciones y artefactos | 7 % | 4,5 | 0,315 |
| Dashboard: Contexto y EDA | 8 % | 4,6 | 0,368 |
| Dashboard: Modelos | 9 % | 4,4 | 0,396 |
| Dashboard: diseño, UX y despliegue | 5 % | 4,6 | 0,23 |
| Modelo original: revisión y diseño | 8 % | 4,2 | 0,336 |
| Modelo original: implementación y demostración | 10 % | 4,1 | 0,41 |
| Evaluación ampliada y robustez | 10 % | 3,5 | 0,35 |
| Pruebas de hipótesis y adecuación | 15 % | 3,5 | 0,525 |
| Informe en Jupyter Book | 14 % | 4,6 | 0,644 |
| Sustentación | 7 % | 4,5 | 0,315 |
| **Total** | **100 %** | | **4,197 → 4,20/5** |

La mejora se concentra en **Hipótesis y adecuación: 3,0 → 3,5/5**. La media de RMSE por semilla ya tiene inferencia propia y la decisión de cuatro paneles se evaluó con 4.608 réplicas prefijadas y 107 límites simultáneos. Los ocho controles nulos cumplen el límite de error; su mayor cota es 0,03692582. El control favorable registra 512/512 y una cota inferior 0,98513395. Se verificaron los cálculos y tres reproducciones exactas. Es evidencia metodológica ejecutada; no se concede el incremento por cantidad de archivos o pruebas de software.

Se aplica el redondeo académico que el autor comunicó: **94,836605 % → 95 %** al porcentaje entero, de forma uniforme para los 18 límites de cobertura. Los otros 89 límites permanecen exactos. No se vuelve a descontar por los dos límites aceptados bajo ese criterio; el protocolo y el resultado científico original `accepted=false` se conservan. Esta interpretación posterior no se presenta como calibración nueva ni firma docente.

Las otras diez notas permanecen idénticas. La simulación no añade robustez de mercado ni confirmación de la candidata, y la preparación del expediente o del recuperador no sustituye una defensa o una ejecución auténticas. No se concede crédito experimental por V2 —cero Monte Carlo— o por recuperación —cero inferencias reales—. Tampoco se penalizan otra vez los resultados adversos declarados o la modalidad de IA autorizada para revisar R1.

El componente estadístico aporta ahora `0,15 × 3,5 = 0,525`, frente a 0,450 antes: la mejora global es **+0,075 puntos**. La suma se realiza sin redondear cada aporte: **4,122 + 0,075 = 4,197 → 4,20**. Se mantienen los once pesos y las otras diez notas.

**Con R1 y R2 aceptadas y R3 no acreditada, la valoración bajo ese supuesto es 4,00/5:** `min(4,197;4,0)=4,00`. Si R1 o R2 se evalúan como incumplidas, la guía fija tope 3,0; una condición pendiente no se transforma en ese incumplimiento. La cifra4,20 antes del tope tampoco acredita R3.

Los descuentos principales siguen siendo la falta de dimensionamiento y potencia de mercado suficientes, una reserva adecuada con confirmación de la candidata frente a toda la familia y la adecuación/robustez real de esa candidata. Las Tablas A/B propias, la calibración y la reproducción humana conservan sus ámbitos pendientes. Los 796 trabajos aditivos y los pilotos de otras arquitecturas no completan esos requisitos por transferencia documental.

La {download}`evaluación estructurada <../_static/e3_evaluation_20261122/assessment.json>`, la {download}`rúbrica calculada <../_static/e3_evaluation_20261122/rubric.csv>` y la {download}`verificación aritmética <../_static/e3_evaluation_20261122/ARITHMETIC_VERIFICATION.json>` fijan fuente, evaluación anterior, notas, reglas condicionales y evidencia. La {ref}`validación de 8.31 <e3-primary-joint-repair-20261122>` y el {ref}`cuaderno de 11.15 <e3-primary-joint-colab-20261122>` permiten revisar resultados y redondeo por separado.
