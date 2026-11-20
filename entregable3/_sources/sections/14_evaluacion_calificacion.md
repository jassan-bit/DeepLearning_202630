# 14 Evaluación y calificación

**Estado: matriz documental de requisitos y evidencia, sin nota ni tope asignados al avance actual.** Este capítulo organiza la sección 14 de la guía —páginas 41–47, Cuadro 8 de la página 59 y Cuadro 9 de la página 60—. Las reglas de evaluación pertenecen a esa guía; describirlas no acredita su cumplimiento ni constituye una evaluación del docente.

La nota se expresa en la escala de **0 a 5** y se determina en dos pasos: primero se obtiene el puntaje ponderado de la rúbrica y después se aplica el tope que corresponda al estado **evaluado** de R1, R2 y R3. Una condición pendiente no se convierte en «no» para fijar ese tope, y una casilla sin evidencia no se marca cumplida para completar la rúbrica.

La evidencia se consulta en las [correcciones de E2](04_correcciones_e2.md), [artefactos](05_configuraciones_exportacion.md), [candidato](06_modelo_original.md), [diseño riguroso](07_diseno_riguroso.md), [verificación estadística](08_verificacion_estadistica.md) y [diseño del dashboard](09_dashboard_predictivo.md). Las actividades pendientes conservan su estado; esta ampliación no entrena modelos, ejecuta contrastes ni calcula calificaciones.

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

La condición incumplida de menor nivel determina el caso; **R3 sólo se evalúa cuando R1 y R2 están acreditadas**. «Pendiente de acreditar» no equivale a un «no» demostrado. En este proyecto siguen abiertas la revisión de novedad R1, la comprensión y defensa académica de R2 y la evaluación de R3; por ello esta tabla no se usa para fijar un tope actual. Las 35 pruebas del primer módulo y la demostración de su cabeza latente no acreditan por sí solas novedad, autoría responsable, defensa o superioridad de mercado.

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

En la ecuación {eq}`eq-e3-final-grade`, el tope se obtiene de las condiciones evaluadas de 14.1; no es un componente adicional de la suma. Alcanzar un tope de 5,0 no implica obtener esa nota: se necesita también el 100 % del puntaje ponderado. Los cuatro niveles de 14.4 no reciben aquí equivalencias numéricas inventadas. No se asignan valores a $q_i$, no se estima el puntaje y no se calcula una nota o tope del proyecto.

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
| Al menos tres conjuntos de naturaleza distinta, elegidos antes de conocer resultados y fuera del diseño. | **Tres archivos admitidos y evaluación completa de la familia**, con independencia por justificar. | {ref}`8.18 <e3-final-extended-audit>`: D01 WIKI/AAPL, D02 ETHUSDT Spot y D03 BTCUSD_PERP, además de D00, completaron los 19 sistemas y el análisis. | Justificar diversidad e independencia pertinentes, exposición previa y potencia. D02/D03 comparten exposición cripto y D03 se solapa con el subyacente BTC; tres archivos no representan automáticamente tres mercados independientes ni demuestran superioridad. |
| Evaluación bajo cambio de distribución: periodo, dominio o fuente. | Comparación exploratoria entre conjuntos disponible. | D01–D03 y diagnósticos temporales/perturbaciones calculados en {ref}`8.18 <e3-final-extended-audit>`, con exposición y dependencia declaradas. | Justificar qué cambio está aislado y qué datos se mantuvieron fuera del diseño; un tramo conocido no adquiere independencia al renombrarlo. |
| Mejor media en al menos dos de tres adicionales y ausencia de inferioridad significativa frente a cualquier rival en cualquiera. | Resultados por conjunto disponibles; R3 no acreditada. | Métricas y contrastes de la familia completa en {ref}`8.18 <e3-final-extended-audit>`; se conservan resultados favorables, adversos y no concluyentes. | Evaluar el criterio completo con pruebas apropiadas y potencia, sin escoger rivales o conjuntos posteriores. Falta de rechazo de inferioridad no equivale a equivalencia o no inferioridad demostradas. |

### 14.5.5 Validez del modelo

Corresponde a pruebas/adecuación —15 %— y correcciones E2 —7 %—. La ruta es **pronóstico temporal F2a**: ECE y Brier de clasificación no se trasladan directamente a σ₇ continua.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Auditoría de fuga: duplicados, unidad de partición, preentrenamiento y exposición del test. | Parcial, con auditoría posterior incorporada. | Cierre de datos E2: **12 PASS, 3 PARTIAL y 1 NA**, con 1316 ventanas y las 608 de prueba disponibles; [sección 4.2](04_correcciones_e2.md). Referencia E3 sin backbone preentrenado y exposición histórica explícita. | Cerrar controles parciales y la auditoría integral del pipeline candidato. Las ventanas verificadas o un control sin fallo no acreditan ausencia integral de fuga ni independencia de calendarios conocidos. |
| Tablas A y B de ajuste/sobreajuste superadas. | Brechas y diagnósticos retrospectivos disponibles; aprobación formal pendiente. | {ref}`8.20 <e3-existing-adequacy-close-20261120>` añade brechas entre particiones y controles triviales en D00–D03, residuos de los pesos históricos y de la auditoría ampliada, con versiones y familias diagnósticas separadas. | No existía un margen de brecha prospectivo del candidato. Cocientes de RMSE, no rechazar un diagnóstico o un error de prueba menor no completan las tablas ni acreditan potencia/adecuación confirmatoria. |
| Calibración antes/después de la recalibración, adaptada a la tarea. | Diagnóstico histórico, perfil individual de servicio y calibración exploratoria ampliada disponibles. | [8.11](08_verificacion_estadistica.md) conserva diez cuantiles y 530 forwards CPU; el servicio usa el q existente de 271801. {ref}`8.18 <e3-final-extended-audit>` añade calibradores y evaluación de intervalos en los cuatro conjuntos, separados de esos artefactos históricos. | Completar la calibración probabilística/regresión que exija la salida y su validación temporal confirmatoria, junto con aceptación pública del perfil. Un punto y un intervalo no definen toda una distribución predictiva. |
| Incertidumbre: cobertura empírica frente a nominal y por subgrupo. | Diagnósticos históricos y ampliados calculados; aceptación pública del servicio pendiente. | El diagnóstico histórico conserva 94,49 % global y 86,22 % en 2026Q1 entre diez semillas; el predictor servido 271801 tiene 94,08 % y 85,56 %. La auditoría nueva aporta cobertura/anchura y subgrupos temporales por conjunto en {ref}`8.18 <e3-final-extended-audit>`. | Justificar subgrupos, dependencia y cambio temporal sin convertir cobertura histórica en garantía futura. Las cifras de la campaña nueva no se atribuyen a los pesos o calibrador históricos del servicio. |

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
| Archivo con DOI, Docker, semillas y determinismo documentados. | Contenedores de servicio y reanálisis comprobados; DOI pendiente. | [Sección 11](11_informe_tecnico.md): archivo determinista con SHA-256, Docker científico desde extracción limpia, once CSV idénticos y discrepancias por campo/unidad cero. Sin entrenamientos o inferencias. | Depósito persistente con DOI, derechos del paquete final ratificados y reproducción humana del alcance que fije el protocolo; los entrenamientos históricos conservados no se repiten por rutina. La igualdad de agregados no acredita determinismo universal o entre plataformas. |
| Maestro por ejecución que genere informe y dashboard. | Maestro histórico consistente y análisis ampliado disponibles; integración final parcial. | El catálogo de 17 sistemas deriva de su tabla histórica y tiene una prueba de igualdad de valores. La {ref}`auditoría de 8.18 <e3-final-extended-audit>` añade métricas y diagnósticos de la familia de 19 sistemas en D00–D03. | Completar recursos comparables y la integración explícita de la versión final con el servicio. No sustituir silenciosamente el catálogo, los pesos o el calibrador históricos; las curvas representantes no son medias de diez modelos. |
| Pruebas unitarias ejecutadas en CI. | Evidencia local y CI identificada por corte y alcance. | Los 42 casos estadísticos históricos conservan su recibo. El [CI del dashboard del corte de base](https://github.com/jassan-bit/DeepLearning_202630/actions/runs/44213072051) aprobó sus 25 casos y seis recorridos de navegador; el job científico aprobó 60 y omitió una referencia opcional SciPy, incluida en los 61 aprobados localmente. La comprobación A/B conserva 50 casos aprobados. La actualización de servicio conserva por separado 29 pruebas locales y seis recorridos de navegador, con cinco inferencias comprobadas en contenedor; [capítulo 9](09_dashboard_predictivo.md). | Conservar los recibos de la versión final sin sumar coberturas distintas como una campaña científica. Ni estos checks ni los casos históricos certifican reproducción humana, defensa académica o servicio Render. |
| Datos o instrucciones exactas de obtención y licencias verificadas. | Tres archivos adicionales admitidos y evaluados; derechos del paquete final parciales. | Cohorte e insumos históricos conservados; D01 WIKI/AAPL y D02/D03 disponen de fuentes, calendarios, comprobaciones y hashes públicos, con políticas declaradas en {ref}`8.18 <e3-final-extended-audit>`. | Ratificar condiciones/derechos de distribución del paquete final y proporcionar sus instrucciones exactas. La evaluación completa no equivale a reproducción independiente de todos los pesos. |
| Reproducción independiente desde cero en contenedor, por persona ajena al código, con discrepancia máxima. | Humana pendiente; reproducción automática de agregados completada. | [Contenedor científico y recibo](11_informe_tecnico.md): 431 filas, 3963 pares numéricos, 101 grupos campo/unidad con error máximo cero. No reentrena 143 predictores. | Persona externa, ejecución desde cero, alcance, discrepancias y concepto firmado. El recibo automático no se atribuye a una persona ni se convierte en acta humana. |

### 14.5.8 Productos del entregable

Corresponde a correcciones —7 %—, artefactos —7 %—, dashboard —8 %, 9 % y 5 %—, informe —14 %— y sustentación —7 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Correcciones E2 con trazabilidad completa y recálculo consistente. | Parcial. | Tabla de revisión interna y auditorías de [sección 4](04_correcciones_e2.md). | Cerrar los controles pendientes; recalcular dependencias y ranking antes/después si una corrección modifica resultados. No se recibieron observaciones del docente. |
| Mejores configuraciones exportadas, ida y vuelta y `metadata.json`. | Cinco ONNX verificados; doce sistemas precalculados. | MLP16, LSTM32, GRU32, E3 y TSMixer, semilla 271801, con paridad en 608 fechas y tolerancia absoluta 10⁻⁶; TSMixer también en 53 de calibración. Los tres clásicos requieren más historia o estado que siete barras. [Capítulo 5](05_configuraciones_exportacion.md). | Nueve exportaciones neuronales verificadas pendientes, política final común y costes completos comparables; no se declara una exclusión por recursos sin medirla. Los tres clásicos conservan su justificación de contrato. |
| Dash Plotly en Render, prueba de humo, contingencia y funcionamiento en sustentación. | Perfil local comprobado; Render Live informado por el operador, sin recibo público aprobado. | Tres páginas, 17 sistemas, cinco ONNX en vivo y doce precálculos. La actualización pasó 29 pruebas, seis recorridos de navegador y cinco inferencias dentro del contenedor; [capítulo 9](09_dashboard_predictivo.md). V2 pasó también su recorrido local de cinco modelos. [URL de Render](https://deeplearning-e3-dashboard.onrender.com). | Con autoDeploy desactivado, Manual Deploy del último commit, esperar Live y ejecutar el cliente CPU V2 con SHA efectivo de 40 caracteres; comprobar páginas, cinco predicciones/intervalos/explicaciones, huellas y recibo público. La concordancia del SHA autodeclarado no es una atestación completa. Completar contingencia y funcionamiento durante la sustentación; [capítulo 10](10_despliegue_render.md). |
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
