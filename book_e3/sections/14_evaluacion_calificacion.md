# 14 Evaluación y calificación

**Estado: matriz documental de requisitos y evidencia, sin nota ni tope asignados al avance actual.** Este capítulo organiza la sección 14 de la guía —páginas 41–47, Cuadro 8 de la página 59 y Cuadro 9 de la página 60—. Las reglas de evaluación pertenecen a esa guía; describirlas no acredita su cumplimiento ni constituye una evaluación del docente.

La evidencia se consulta en las [correcciones de E2](04_correcciones_e2.md), [artefactos](05_configuraciones_exportacion.md), [candidato](06_modelo_original.md), [diseño riguroso](07_diseno_riguroso.md), [verificación estadística](08_verificacion_estadistica.md) y [diseño del dashboard](09_dashboard_predictivo.md). Las actividades pendientes conservan su estado; esta ampliación no entrena modelos, ejecuta contrastes ni calcula calificaciones.

La matriz incorpora el corte posterior del **17 de noviembre de 2026**: diez semillas por red de E2 y de la referencia E3 de 1210 parámetros, tres trayectorias clásicas deterministas y análisis de los 17 sistemas sobre 608 fechas. Actualizar una casilla con evidencia retrospectiva no la convierte en cumplimiento confirmatorio ni transfiere las propiedades del primer módulo de 1211 parámetros a la receta evaluada.

## 14.1 Tope de calificación según el modelo original

La guía establece estos casos **condicionales**, sujetos a evidencia y evaluación académica:

| Tope previsto por la guía | Condición que debe haberse evaluado | Alcance de la regla |
|---|---|---|
| 5,0 | R1 sí, R2 sí y R3 sí. | Novedad para el problema, núcleo algorítmico propio y defendible, propiedad formal exigida y superioridad estadísticamente verificada frente a toda la familia, con consistencia adicional. El tope no sustituye la rúbrica. |
| 4,0 | R1 sí y R2 sí, pero R3 no. | La contribución admisible y su sustento teórico están acreditados, pero no se demuestra la superioridad significativa exigida frente a toda la familia. |
| 3,0 | R1 no o R2 no, independientemente del desempeño. | La diferencia es sólo de aplicación, el mecanismo ya fue usado para el problema, la transferencia carece de atribución/adaptación sustantiva o el núcleo proviene de un import o código ajeno sin programación algorítmica propia. |

La condición incumplida de menor nivel determina el caso; **R3 sólo se evalúa cuando R1 y R2 están acreditadas**. «Pendiente de acreditar» no equivale a un «no» demostrado. En este proyecto siguen abiertas la revisión de novedad R1, la comprensión y defensa académica de R2 y la evaluación de R3; por ello esta tabla no se usa para fijar un tope actual. Las 35 pruebas del primer módulo y la demostración de su cabeza latente no acreditan por sí solas novedad, autoría responsable, defensa o superioridad de mercado.

La potencia se dimensiona **antes de congelar**: al menos 0,90 por contraste y potencia conjunta reportada, con diez semillas como mínimo o el número mayor requerido. Si el piloto muestra insuficiencia, la guía exige aumentar las semillas antes del cierre; no permite rebajar el estándar después de observar los resultados. Bajo el supuesto de R1 y R2 acreditadas, la guía contempla el caso de resultado no concluyente con potencia insuficiente declarada y MDE reportada dentro de la regla de 4,0. Si falta análisis de potencia, además sitúa el componente de pruebas de hipótesis del 15 % en «en desarrollo». Son reglas condicionales de evaluación, no una autorización para omitir el dimensionamiento ni una calificación de este avance. Una media mejor sin los rechazos exigidos no acredita R3.

## 14.2 Rúbrica ponderada

Se conservan exactamente los once componentes y pesos del Cuadro 9. Los criterios de clasificación se adaptan a la ruta temporal F2a cuando corresponde.

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

## 14.3 Cálculo de la nota final

Si el evaluador determina un cumplimiento por componente $q_i$ entre 0 y 100 y aplica los pesos $w_i$ de la tabla, el puntaje ponderado queda en la escala de 0 a 100:

$$
P_{\mathrm{ponderado}}=\sum_{i=1}^{11}\frac{w_iq_i}{100},
\qquad
\mathrm{Nota\ final}=\min\left(\mathrm{Tope},\;5{,}0\times\frac{P_{\mathrm{ponderado}}}{100}\right).
$$

El tope se obtiene de las condiciones evaluadas de 14.1; no es un componente adicional de la suma. Alcanzar un tope de 5,0 no implica obtener esa nota: se necesita también el 100 % del puntaje ponderado. Aquí no se asignan valores a $q_i$, no se estima el puntaje y no se calcula una nota o tope del proyecto.

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

### 14.5.1 Contribución y novedad

Corresponde a R1/R2 y a revisión/diseño —8 %— e implementación/demostración —10 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Mecanismo nuevo y transferible para el problema, con diferencia algorítmica sustantiva. | Pendiente R1. | Regla candidata y comparación de siete antecedentes en [sección 6](06_modelo_original.md). | Búsqueda efectiva y análisis de equivalencias; la combinación no se declara inédita. |
| Si hay transferencia: fuente citada, justificación y adaptación documentadas en `experiments/procedencia.md`. | Parcial; procedencia final pendiente. | Registro del candidato en `experiments/e3_bloque_c_20261115/procedencia.md` y antecedentes declarados. | Resolver origen y modificaciones verificables; consolidar el documento final cuando corresponda. No marcar «no aplica» sin justificarlo. |
| Limitación de los métodos, preferiblemente del paper guía, citada o cuantificada por piloto. | Parcial. | Limitación exploratoria y límites de extrapolar el estrés artificial, [secciones 6.1 y 7.1](07_diseno_riguroso.md). | Evidencia sobre la limitación específica de BTC que la regla pretende corregir. |
| Hipótesis falsable y predicciones secundarias anteriores a los experimentos finales. | Parcial documental. | D1–D5 propuestas en [sección 7](07_diseno_riguroso.md). | Fijar magnitudes, escenarios y éxito/fracaso antes del protocolo definitivo. |
| Propiedad formal, supuestos explícitos y verificación numérica. | Disponible con alcances separados por arquitectura. | Primer módulo y momentos conservan sus pruebas. La cabeza aditiva tiene una cota condicional independiente y verificación con 1728 pares y 96 gradientes sintéticos en [7.3](07_diseno_riguroso.md). | La cota aditiva requiere estados acotados y piso/parámetros fijos. No cubre el mapa OHLCV ni prueba novedad; falta mostrar un beneficio predictivo medible atribuible a la propiedad. |
| Núcleo escrito desde la formulación, sin importar el modelo ni copiar el núcleo ajeno; comprensión defendible. | Parcial. | `src/original_model.py` y sus 35 pruebas pertenecen al primer módulo; la referencia aditiva se identifica por separado en [sección 6](06_modelo_original.md). Asistencia de IA declarada. | Verificar la implementación y pruebas correspondientes al mecanismo que se presente como contribución, y su revisión, comprensión y modificación por el estudiante. Los tests no certifican la defensa R2. |
| Matriz y declaración de novedad, búsqueda repetida al final y verificación por un tercero. | Parcial, sin declaración acreditada. | Matriz y bitácora: las 18 solicitudes a tres índices se bloquearon antes de buscar. | Ejecutar búsquedas, completar revisión y repetición final, y concepto de una persona externa; resultados desconocidos no significan cero antecedentes. |

### 14.5.2 Comparación justa

Corresponde a R3 y evaluación ampliada/robustez —10 %—. Los controles propuestos se detallan en la [sección 8](08_verificacion_estadistica.md).

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Benchmarks fuertes/recientes, ajustados y congelados después del Bloque A. | Parcial heredado. | Dieciséis sistemas, recetas y resultados E2; auditorías en [sección 4](04_correcciones_e2.md). | Cierre A, controles aplicables y anclas del benchmark confirmatorio; el freeze histórico no cierra A automáticamente. |
| Familia completa, sin selección posterior de rivales. | Comparación retrospectiva calculada; confirmación pendiente. | Diez candidatos, tres referencias neuronales y tres clásicos: **16 rivales** frente a E3; 48 contrastes primarios sobre las mismas 608 fechas, sin excluir sistemas o semillas. | Campaña confirmatoria con requisitos de equidad, potencia y evaluación ampliada. El m≈12 ilustrativo no reduce esta familia. |
| Mismo presupuesto, particiones, preprocesamiento y fuente de pesos preentrenados. | Diferencia de tratamiento documentada; equidad pendiente. | E2: 655 ejemplos de ajuste y 53 de calibración; E3: 532 de ajuste y 116 de validación, sin calibración. Recetas y consumos separados en [sección 8.2](08_verificacion_estadistica.md). | Ejecutar un protocolo común futuro; no atribuir el resultado exclusivamente a la arquitectura ni asumir que diez semillas o una misma forma de entrada demuestran equidad. |
| Capacidad controlada con parámetros/FLOPs y benchmark de tamaño comparable si el propio es mayor. | Parcial de inventario. | Parámetros del candidato y costes heredados con alcance, [secciones 5–6](05_configuraciones_exportacion.md). | Medición comparable y variante ajustada en tamaño si procede. Para esta casilla de 5,0, discutir el tamaño no sustituye esa variante. |
| Estándar/protocolo con fecha y tag previos; confirmatorio y exploratorio separados. | Pendiente de cierre y registro. | Estándar operativo y `experiments/protocolo.md` canónico disponibles; este último es un borrador posterior, **no congelado**, con campos abiertos. El análisis histórico mantiene su alcance en [sección 8](08_verificacion_estadistica.md). | Cerrar decisiones y revisión y registrar fechas/tags reales antes de futuras ejecuciones confirmatorias; disponer del archivo no completa un preregistro. |
| Bitácora Nd y desarrollo separado de confirmación. | Registro consolidado parcial; Nd global sin conciliar. | [Bitácora fechada](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/design_log.md) e inventario del capítulo 7: variantes artificiales, cinco recetas v1/v2, tres por momentos y réplicas de una receta aditiva. | Conciliar propuestas, variantes, parametrizaciones, intentos y fallos sin usar ocho recetas como Nd global. El registro posterior no es preregistro; definir confirmación que no haya guiado el diseño. |

### 14.5.3 Evidencia estadística

Corresponde a R3 y pruebas/adecuación —15 %—. Las unidades, estimandos y reglas de decisión deberán quedar cerrados antes de confirmar.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Al menos diez semillas o más por potencia, media/DE/IC95 % bootstrap, MDE y potencia conjunta. | Réplicas y descripción disponibles; dimensionamiento pendiente. | Trece redes E2 con diez semillas más tres clásicos deterministas: **133 predictores E2**; diez semillas E3. Media, DE e intervalos con ámbitos declarados en [sección 8](08_verificacion_estadistica.md). | Piloto y diseño previo para R/calendario suficientes, MDE y potencia por contraste y conjunta. Diez semillas no prueban potencia; copiar un clásico determinista no crea réplicas. |
| Tres pruebas admisibles por contraste, Holm y potencia ≥0,90 por contraste. | Tres procedimientos posteriores ejecutados; batería confirmatoria y potencia pendientes. | Bootstrap temporal, influencia RMSE con HAC y t entre semillas: **48/48 rechazos** con Holm global48, máximo p ajustado 0,0019. | Justificar pertinencia F2a y supuestos de cada procedimiento, que condicionan ámbitos distintos. La guía exige pruebas distintas apropiadas, no identidad literal de ámbitos. Los rechazos retrospectivos no acreditan potencia ni toda R3. |
| Efecto con IC y relevancia frente a δ. | Disponible con alcance retrospectivo. | Ganancia de RMSE frente al mejor rival descriptivo LSTM-autoencoder: **0,0009334571**; IC95 % temporal marginal **[0,0006329359; 0,0012131746]**; δ heredado **0,0002577284**. Efectos de los 16 rivales descargables en [sección 8](08_verificacion_estadistica.md). | Los IC nominales no son simultáneos. Cohen dz entre semillas condiciona el calendario; no representa variación entre mercados ni permite convertir δ de RMSE en δ² de MSE. |
| Comparación global que incluya al original. | MCS auxiliar posterior calculado. | MCS90 Tmax de los 17 sistemas, sobre MSE diario medio entre semillas: conserva **diez sistemas, incluido E3**; p final 0,1442. | No identifica ganador único, equivalencia ni ganancia práctica de RMSE. No sustituye la selección histórica de E2 o una comparación confirmatoria bajo protocolo común. |
| Prueba entre conjuntos: Friedman–Iman–Davenport y Holm, o bayesiana con ROPE. | Pendiente. | Alternativas documentadas, sin elegir por resultados. | Conjuntos admisibles, unidad/dependencia y análisis con límites de potencia; no promediar RMSE de escalas diferentes. |
| Sensibilidad a prueba, métrica secundaria y subconjunto de semillas. | Disponible con alcance posterior. | Bloques/rezagos 7 y 28 conservan 48/48 rechazos; MAE secundaria con Holm16 rechaza 16/16 frente a cero; subgrupos descriptivos de primeras/últimas cinco semillas publicados. | No reemplazan sensibilidad a hiperparámetros, perturbaciones o nuevos conjuntos. No se inventa margen práctico de MAE ni se usa Nadeau–Bengio para días/semillas de una sola cronología. |

La potencia no se certifica por cumplir R≥10 ni multiplicando potencias marginales dependientes. El dimensionamiento debe considerar la regla completa —Holm, tres pruebas y familia—, la alternativa frente a la frontera δ y la dependencia temporal. Semillas×días no son observaciones iid. Estas exigencias no se relajan ante resultados adversos.

### 14.5.4 Generalización de la evidencia

Corresponde a R3 y evaluación ampliada/robustez —10 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Al menos tres conjuntos de naturaleza distinta, elegidos antes de conocer resultados y fuera del diseño. | Pendiente. | Criterios de fuente pública, dos ejes de diferencia, licencia/versión/hash en [sección 8.3](08_verificacion_estadistica.md). | Elegirlos y evaluar original y todos los rivales con presupuesto comparable. Ninguno está seleccionado. |
| Evaluación bajo cambio de distribución: periodo, dominio o fuente. | Pendiente E3. | Estrategia propuesta y exposición previa de periodos históricos identificada. | Evidencia mantenida fuera de las decisiones de diseño; otro tramo conocido no adquiere independencia al renombrarlo. |
| Mejor media en al menos dos de tres adicionales y ausencia de inferioridad significativa frente a cualquier rival en cualquiera. | Pendiente. | Criterio R3 conservado sin resultados del candidato. | Métricas y contrastes por conjunto; falta de rechazo de inferioridad no equivale a equivalencia o no inferioridad demostradas. |

### 14.5.5 Validez del modelo

Corresponde a pruebas/adecuación —15 %— y correcciones E2 —7 %—. La ruta es **pronóstico temporal F2a**: ECE y Brier de clasificación no se trasladan directamente a σ₇ continua.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Auditoría de fuga: duplicados, unidad de partición, preentrenamiento y exposición del test. | Parcial, con auditoría posterior incorporada. | Cierre de datos E2: **12 PASS, 3 PARTIAL y 1 NA**, con 1316 ventanas y las 608 de prueba disponibles; [sección 4.2](04_correcciones_e2.md). Referencia E3 sin backbone preentrenado y exposición histórica explícita. | Cerrar controles parciales y la auditoría integral del pipeline candidato. Las ventanas verificadas o un control sin fallo no acreditan ausencia integral de fuga ni independencia de calendarios conocidos. |
| Tablas A y B de ajuste/sobreajuste superadas. | Pendiente del candidato. | E2 conserva diagnósticos CPU condicionales: dos de trece redes establecen tolerancia δg. | Baseline trivial, riesgos comparables, contraste de brecha, residuos y consistencia del candidato; no heredar adecuación de otras redes. |
| Calibración antes/después de la recalibración, adaptada a la tarea. | Pendiente del candidato. | Diagnósticos F2a y reglas probabilísticas previstos. | Pendiente/intercepto, residuos y calibración probabilística aplicables, con recalibración sólo en datos autorizados y evaluación separada; pinball para cuantiles, CRPS/log-score cuando la salida lo permita. |
| Incertidumbre: cobertura empírica frente a nominal y por subgrupo. | E2 histórico descrito; candidato y servicio pendientes. | Tabla extendida de [sección 8](08_verificacion_estadistica.md): cobertura y anchura de E2 reconstruidas con sus cuantiles guardados, nominal 0,90. E3 carece de calibrador. | Cobertura/ancho de E3, subgrupos prefijados e IC temporales y verificación del servicio. El IC entre semillas no es un intervalo predictivo y nominal no es garantía individual. |

### 14.5.6 Análisis que explican el resultado

Corresponde a implementación/demostración —10 %— y evaluación ampliada/robustez —10 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Ablación, incluidas adaptaciones, con las mismas pruebas, familia propia y control de capacidad. | Controles históricos ejecutados; ablación confirmatoria pendiente. | Primer módulo: variantes artificiales y piloto completo frente a M00 sin ambas modificaciones. V2 y momentos conservan sus recetas y resultados adversos en [secciones 6–7](06_modelo_original.md). | Al menos tres componentes aislados del mecanismo que finalmente se presente, mismas pruebas estadísticas y controles de capacidad; las variantes históricas de otros núcleos no completan la ablación de la referencia aditiva. |
| Sensibilidad a hiperparámetros propios y estabilidad por semilla. | Estabilidad de una receta disponible; sensibilidad pendiente. | Diez semillas congeladas de la aditiva: RMSE media 0,0029394324 y DE muestral 0,0000471254, con intervalos condicionados al calendario; [sección 8](08_verificacion_estadistica.md). | La réplica de una receta no es HPO ni sensibilidad a sus hiperparámetros. Faltan estudios comparables y sus criterios, sin seleccionar sólo la región favorable. |
| Robustez a ruido, perturbaciones y reducción de muestra. | Pendiente E3. | Estudios pendientes identificados en [sección 8.8](08_verificacion_estadistica.md). | Al menos tres severidades, curvas de degradación y aprendizaje/escalabilidad con varias semillas; separar estrés artificial de escenarios plausibles. |
| Interpretabilidad con fidelidad, estabilidad, comparación con mejor benchmark y descarte de atajos. | Sensibilidad local disponible; estudio completo pendiente. | El dashboard calcula intervenciones sobre la entrada del mismo artefacto ONNX y describe sus efectos. | Comparación global con el mejor SAE, fidelidad, estabilidad entre semillas y descarte de atajos; la sensibilidad local no acredita explicación causal. |
| Errores y escenarios donde no supera a los benchmarks. | Diagnósticos históricos disponibles; estudio confirmatorio pendiente. | El dashboard muestra trayectorias y residuos de representantes fijos; se conserva el diagnóstico posterior de las 18 fechas de objetivo alto frente a M00 proyectado. | Escenarios prefijados, errores compartidos y causas discutidas sin inferencia causal injustificada. Un grupo definido por el objetivo realizado no estaba disponible al pronosticar. |
| Parámetros, FLOPs, latencia, memoria, entrenamiento y energía con Pareto. | Parcial de ingeniería e inventario. | Cuatro ONNX con parámetros y latencias CPU de preprocesamiento más inferencia caliente; los perfiles TorchScript históricos mantienen otro alcance. | FLOPs completos, recursos comparables de los 17 sistemas, energía y perfil del servicio público. La paridad de cuatro artefactos no establece una frontera de Pareto de toda la familia. |

### 14.5.7 Reproducibilidad

Corresponde al informe —14 %— y configuraciones/artefactos —7 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Archivo con DOI, Docker, semillas y determinismo documentados. | Contenedor de servicio local comprobado; paquete científico final pendiente. | Imagen Docker de ONNX Runtime sin Torch construida y probada localmente, README y semillas/precisión documentadas; estado de imagen y recursos en [sección 10](10_despliegue_render.md). | Archivo E3 persistente con DOI, contenedor científico integral y reproducción humana. No atribuir determinismo global NumPy o paridad entre plataformas no demostrados. |
| Maestro por ejecución que genere informe y dashboard. | Disponible para las métricas BTC actuales; maestro final parcial. | El catálogo de 17 sistemas deriva de la tabla ampliada del libro y tiene una prueba de igualdad de valores; fuentes y hashes del bundle publicados. | Completar campos de recursos, calibración E3 y conjuntos adicionales; las curvas de la semilla representante no son la media de diez modelos. |
| Pruebas unitarias ejecutadas en CI. | Evidencia local y CI identificada por corte y alcance. | Los 42 casos estadísticos históricos conservan su recibo. El [CI del dashboard del corte de base](https://github.com/jassan-bit/DeepLearning_202630/actions/runs/44213072051) aprobó sus 25 casos y seis recorridos de navegador; el job científico aprobó 60 y omitió una referencia opcional SciPy, incluida en los 61 aprobados localmente. La comprobación A/B conserva 50 casos aprobados. | Conservar los recibos de la versión final sin sumar coberturas distintas como una campaña científica. Ni estos checks ni los casos históricos certifican reproducción humana, defensa académica o servicio Render. |
| Datos o instrucciones exactas de obtención y licencias verificadas. | Parcial. | Cohorte auditada con ventanas train/cal/test, fuentes inmutables y SHA256, junto con el informe de admisión de los once insumos del análisis posterior. | Consolidar distribución autorizada, licencias e instrucciones de la entrega final; no reconstruir OHLCV desde predicciones ni confundir admisión de predicciones con deserialización independiente de todos los pesos. |
| Reproducción independiente desde cero en contenedor, por persona ajena al código, con discrepancia máxima. | Pendiente. | Auditores de software y [acta propuesta por campo/unidad](11_informe_tecnico.md), con emparejamiento de filas y tolerancias previas. | Persona externa, contenedor científico limpio, comandos, máxima discrepancia observada y concepto firmado; un subagente de IA no sustituye ese requisito y no hay una discrepancia cero que atribuirle. |

### 14.5.8 Productos del entregable

Corresponde a correcciones —7 %—, artefactos —7 %—, dashboard —8 %, 9 % y 5 %—, informe —14 %— y sustentación —7 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Correcciones E2 con trazabilidad completa y recálculo consistente. | Parcial. | Tabla de revisión interna y auditorías de [sección 4](04_correcciones_e2.md). | Cerrar los controles pendientes; recalcular dependencias y ranking antes/después si una corrección modifica resultados. No se recibieron observaciones del docente. |
| Mejores configuraciones exportadas, ida y vuelta y `metadata.json`. | Cuatro ONNX verificados; catálogo completo pendiente. | MLP16, LSTM32, GRU32 y E3, semilla 271801, con paridad en 608 fechas y tolerancia absoluta 10⁻⁶; TorchScript históricos conservados. | Exportaciones de otros trece sistemas, política final común y demás campos completos del catálogo. Los límites de semilla y de tratamiento no se borran al exportar. |
| Dash Plotly en Render, prueba de humo, contingencia y funcionamiento en sustentación. | Prototipo y contenedor locales comprobados; servicio público por crear. | Tres páginas y 17 sistemas, cuatro artefactos ONNX CPU y recorridos del [capítulo 9](09_dashboard_predictivo.md); guion reproducible y procedimiento público en [10](10_despliegue_render.md). | URL Render, recursos y prueba de humo públicos, contingencia y sustentación comprobadas. Un guion preparado o un resultado local conserva ese alcance; publicar el libro no despliega Dash. |
| Jupyter Book sin errores, ficha de modelo y declaración de datos. | Publicación técnica de base comprobada; cierre científico pendiente. | Fuentes, resumen con aportes técnicos, mapa científico 1–16, ecuaciones numeradas y bibliografía en [11](11_informe_tecnico.md); MODEL_CARD.md y DATA_STATEMENT.md del prototipo. La [matriz de 12](12_lista_entregables.md) relaciona los trece productos con su cierre. | Reconstruir/verificar la versión final y ratificar derechos, contenidos y responsabilidades humanas; los archivos no son una firma ni cierran R1/R2/R3. |
| Procedencia del mecanismo transferido, cuando corresponda. | Parcial. | Registro candidato y atribuciones en [sección 6](06_modelo_original.md). | Cerrar fuente/adaptación e integrar `experiments/procedencia.md` final, sin fingir novedad de componentes conocidos. |
| Declaración de autoría, contribuciones individuales y uso de IA. | Parcial documental; validación pendiente. | Asistencia de IA reconocida en código, teoría y procedencia. | Declaración final firmada, contribuciones individuales y comprensión responsable del autor; la asistencia declarada no acredita una firma ni defensa. |
| Sustentación: demostración en vivo y capacidad de explicar/modificar el algoritmo. | Ensayo y defensa pendientes. | [Guion de 20 minutos](13_sustentacion.md), con 18 de contenido y dos de margen, nueve preguntas y comandos de práctica comprobados. | Defensa efectiva por el estudiante, preguntas técnicas y modificación/ejecución en vivo sobre la versión identificada; el guion y sus tests no reemplazan esa evidencia. |

### 14.5.9 Decisión previa al congelamiento del protocolo

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Definir antes del cierre el hallazgo alternativo que sostendrá el trabajo si no se acredita superioridad significativa y consistente. | Propuesta pendiente de fijar. | Límites y resultados adversos se conservan en el diseño de [secciones 7–8](08_verificacion_estadistica.md). | Registrar el hallazgo alternativo, sus análisis y criterios antes de confirmar; no escogerlo después por ser favorable. |

Se propone como hallazgo alternativo un **benchmark reproducible y una explicación delimitada de los escenarios en que la regla candidata aporta o perjudica**. Si no supera a toda la familia, el informe conservaría las diferencias/IC, potencia y MDE, costes y ablaciones, y distinguiría falta de evidencia, efecto pequeño y deterioro observado. La relación entre acotar la cabeza y predecir colas de σ₇ sería una hipótesis a contrastar, no una explicación ya demostrada. Esta propuesta debe precisarse antes del congelamiento y no reemplaza el objetivo confirmatorio ni permite relajar R3.

La consecuencia de un R3 evaluado y no satisfecho, bajo R1/R2 acreditadas, es la regla condicional descrita en 14.1. No se aplica aquí a condiciones aún pendientes. La selección posterior de semillas, conjuntos, métricas, análisis o rivales para sostener superioridad contraviene la guía y anula esa afirmación; se conservarán el historial, los resultados desfavorables y las desviaciones declaradas.

La [sección 15](15_integridad_academica.md) desarrolla esas reglas de integridad y la [16](16_conclusiones.md) conserva las conclusiones compatibles con la evidencia actual. El [Apéndice A](apendice_a_plantillas.md) ofrece registros editables sin marcar requisitos como cumplidos por disponer de una plantilla.
