# 14 Evaluación y calificación

**Estado: matriz documental de requisitos y evidencia, sin nota ni tope asignados al avance actual.** Este capítulo organiza la sección 14 de la guía —páginas 41–47, Cuadro 8 de la página 59 y Cuadro 9 de la página 60—. Las reglas de evaluación pertenecen a esa guía; describirlas no acredita su cumplimiento ni constituye una evaluación del docente.

La evidencia se consulta en las [correcciones de E2](04_correcciones_e2.md), [artefactos](05_configuraciones_exportacion.md), [candidato](06_modelo_original.md), [diseño riguroso](07_diseno_riguroso.md), [verificación estadística](08_verificacion_estadistica.md) y [diseño del dashboard](09_dashboard_predictivo.md). Las actividades pendientes conservan su estado; esta ampliación no entrena modelos, ejecuta contrastes ni calcula calificaciones.

## 14.1 Tope de calificación según el modelo original

La guía establece estos casos **condicionales**, sujetos a evidencia y evaluación académica:

| Tope previsto por la guía | Condición que debe haberse evaluado | Alcance de la regla |
|---|---|---|
| 5,0 | R1 sí, R2 sí y R3 sí. | Novedad para el problema, núcleo algorítmico propio y defendible, propiedad formal exigida y superioridad estadísticamente verificada frente a toda la familia, con consistencia adicional. El tope no sustituye la rúbrica. |
| 4,0 | R1 sí y R2 sí, pero R3 no. | La contribución admisible y su sustento teórico están acreditados, pero no se demuestra la superioridad significativa exigida frente a toda la familia. |
| 3,0 | R1 no o R2 no, independientemente del desempeño. | La diferencia es sólo de aplicación, el mecanismo ya fue usado para el problema, la transferencia carece de atribución/adaptación sustantiva o el núcleo proviene de un import o código ajeno sin programación algorítmica propia. |

La condición incumplida de menor nivel determina el caso; **R3 sólo se evalúa cuando R1 y R2 están acreditadas**. «Pendiente de acreditar» no equivale a un «no» demostrado. En este proyecto siguen abiertas la revisión de novedad R1, la comprensión y defensa académica de R2 y la evaluación de R3; por ello esta tabla no se usa para fijar un tope actual. Las 35 pruebas de software y la demostración de la cabeza latente no acreditan por sí solas novedad, autoría responsable, defensa o superioridad de mercado.

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
| Propiedad formal, supuestos explícitos y verificación numérica. | Disponible con alcance latente delimitado. | Cota de la cabeza, derivadas y verificaciones en [secciones 6–7](06_modelo_original.md). | No atribuir esa cota al encoder o al pipeline OHLCV completo ni dar por probada utilidad de mercado. |
| Núcleo escrito desde la formulación, sin importar el modelo ni copiar el núcleo ajeno; comprensión defendible. | Parcial. | `src/original_model.py`, primitivas PyTorch y 35 pruebas artificiales; asistencia de IA declarada. | Revisión, comprensión y capacidad de explicar/modificar por el estudiante. Las pruebas no certifican la defensa R2. |
| Matriz y declaración de novedad, búsqueda repetida al final y verificación por un tercero. | Parcial, sin declaración acreditada. | Matriz y bitácora: las 18 solicitudes a tres índices se bloquearon antes de buscar. | Ejecutar búsquedas, completar revisión y repetición final, y concepto de una persona externa; resultados desconocidos no significan cero antecedentes. |

### 14.5.2 Comparación justa

Corresponde a R3 y evaluación ampliada/robustez —10 %—. Los controles propuestos se detallan en la [sección 8](08_verificacion_estadistica.md).

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Benchmarks fuertes/recientes, ajustados y congelados después del Bloque A. | Parcial heredado. | Dieciséis sistemas, recetas y resultados E2; auditorías en [sección 4](04_correcciones_e2.md). | Cierre A, controles aplicables y anclas del benchmark confirmatorio; el freeze histórico no cierra A automáticamente. |
| Familia completa, sin selección posterior de rivales. | Inventario definido; comparación pendiente. | Diez candidatos, tres referencias neuronales y tres clásicos: **16 rivales** frente al original. | Campaña completa y trazable. El m≈12 ilustrativo no reduce esta familia. |
| Mismo presupuesto, particiones, preprocesamiento y fuente de pesos preentrenados. | Pendiente E3. | Propuesta de equidad y contabilidad de HPO/tiempo; candidato sin pesos preentrenados. | Cerrar recursos y ejecutar protocolo común; no asumir que heredar formas de entrada demuestra equidad. |
| Capacidad controlada con parámetros/FLOPs y benchmark de tamaño comparable si el propio es mayor. | Parcial de inventario. | Parámetros del candidato y costes heredados con alcance, [secciones 5–6](05_configuraciones_exportacion.md). | Medición comparable y variante ajustada en tamaño si procede. Para esta casilla de 5,0, discutir el tamaño no sustituye esa variante. |
| Estándar/protocolo con fecha y tag previos; confirmatorio y exploratorio separados. | Pendiente. | Estándar descrito y borrador **no congelado** de [sección 8](08_verificacion_estadistica.md). | `experiments/estandar.md` y protocolo finales, decisiones cerradas y tags reales anteriores a las ejecuciones definitivas. |
| Bitácora Nd y desarrollo separado de confirmación. | Parcial documental. | Nd de mercado=0, cuatro variantes funcionales artificiales; exposición histórica registrada. | Conciliar esfuerzo total e intentos/fallos y definir confirmación que no haya guiado el diseño. Repetir E1/E2 no restaura cegamiento. |

### 14.5.3 Evidencia estadística

Corresponde a R3 y pruebas/adecuación —15 %—. Las unidades, estimandos y reglas de decisión deberán quedar cerrados antes de confirmar.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Al menos diez semillas o más por potencia, media/DE/IC95 % bootstrap, MDE y potencia conjunta. | Pendiente del candidato. | E2 tiene seis semillas por red y una trayectoria por clásico; límites declarados. | Piloto de al menos cinco pares contra cada rival, R final y calendario suficientes. Copiar un clásico determinista no crea semillas independientes. |
| Tres pruebas admisibles por contraste, Holm y potencia ≥0,90 por contraste. | Pendiente. | Batería por validar en [sección 8](08_verificacion_estadistica.md): 16 rivales reales. | Pruebas F2a coherentes con la media de RMSE y δ; p válidos, supuestos y máximo de p ajustados por rival. Superar a todos exige que todos pasen. |
| Efecto con IC y relevancia frente a δ. | Pendiente del candidato. | δ heredado y orientación de ganancia declarados. | Efectos/IC con unidades independientes y dirección favorable; no transferir efectos E2 ni convertir δ de RMSE en δ² de MSE. |
| Comparación global que incluya al original. | Pendiente. | MCS histórico E2 sin candidato. | MCS de los 17 sistemas bajo protocolo; pertenencia no demuestra por sí sola superioridad práctica por par. |
| Prueba entre conjuntos: Friedman–Iman–Davenport y Holm, o bayesiana con ROPE. | Pendiente. | Alternativas documentadas, sin elegir por resultados. | Conjuntos admisibles, unidad/dependencia y análisis con límites de potencia; no promediar RMSE de escalas diferentes. |
| Sensibilidad a prueba, métrica secundaria y subconjunto de semillas. | Pendiente. | Análisis propuestos y aplicabilidad de Nadeau–Bengio delimitada. | Reglas previas y resultados; comparación paramétrica/no paramétrica y con/sin corrección sólo cuando su diseño la justifique. |

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
| Auditoría de fuga: duplicados, unidad de partición, preentrenamiento y exposición del test. | Parcial. | Auditoría E2 train655/cal53: ocho controles verificados, siete parciales y uno no aplicable; candidato sin backbone preentrenado. | Datos/controles de prueba y auditoría del pipeline candidato. No se acredita ausencia integral de fuga por no hallar un fallo en lo inspeccionado. |
| Tablas A y B de ajuste/sobreajuste superadas. | Pendiente del candidato. | E2 conserva diagnósticos CPU condicionales: dos de trece redes establecen tolerancia δg. | Baseline trivial, riesgos comparables, contraste de brecha, residuos y consistencia del candidato; no heredar adecuación de otras redes. |
| Calibración antes/después de la recalibración, adaptada a la tarea. | Pendiente del candidato. | Diagnósticos F2a y reglas probabilísticas previstos. | Pendiente/intercepto, residuos y calibración probabilística aplicables, con recalibración sólo en datos autorizados y evaluación separada; pinball para cuantiles, CRPS/log-score cuando la salida lo permita. |
| Incertidumbre: cobertura empírica frente a nominal y por subgrupo. | Pendiente del candidato y del servicio. | Calibradores de referencias existentes, con nominal 0,90 y límites temporales, [secciones 5 y 9](09_dashboard_predictivo.md). | Cobertura/ancho del mismo predictor y cohorte, subgrupos prefijados e IC temporales; nominal no es garantía individual. |

### 14.5.6 Análisis que explican el resultado

Corresponde a implementación/demostración —10 %— y evaluación ampliada/robustez —10 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Ablación, incluidas adaptaciones, con las mismas pruebas, familia propia y control de capacidad. | Parcial de software. | Completo, sin penalización, sin saturación y sin contenido comprobados artificialmente. | Ablaciones de desempeño, base sin ambas modificaciones y controles igualados en tamaño cuando cambie capacidad; no declarar tres innovaciones por tres intervenciones. |
| Sensibilidad a hiperparámetros propios y estabilidad por semilla. | Pendiente de mercado. | Rangos y estudios propuestos, [secciones 7–8](07_diseno_riguroso.md). | Resultados con varias semillas, IC y sensibilidad de la conclusión, sin seleccionar sólo la región favorable. |
| Robustez a ruido, perturbaciones y reducción de muestra. | Pendiente E3. | Matriz de estudios de [sección 8.8](08_verificacion_estadistica.md). | Al menos tres severidades, curvas de degradación y aprendizaje/escalabilidad con varias semillas; separar estrés artificial de escenarios plausibles. |
| Interpretabilidad con fidelidad, estabilidad, comparación con mejor benchmark y descarte de atajos. | Pendiente. | Métodos y controles especificados en [sección 9](09_dashboard_predictivo.md). | Explicaciones reales del mismo caso/modelo y evidencia comparativa; atención no equivale por sí sola a explicación causal. |
| Errores y escenarios donde no supera a los benchmarks. | Pendiente del candidato. | Regímenes y análisis de errores previstos; límites económicos declarados. | Resultados adversos por escenarios prefijados, errores compartidos y causas discutidas sin inferencia causal injustificada. |
| Parámetros, FLOPs, latencia, memoria, entrenamiento y energía con Pareto. | Parcial de inventario. | Parámetros del candidato; costes E2 y tres forwards TorchScript con ámbitos distintos. | Mediciones comparables del conjunto completo y candidato, energía estimada y Pareto; RSS aislado no mide cabida Render ni RAM incremental. |

### 14.5.7 Reproducibilidad

Corresponde al informe —14 %— y configuraciones/artefactos —7 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Archivo con DOI, Docker, semillas y determinismo documentados. | Parcial; paquete final pendiente. | Repositorio, versiones CPU del candidato y política/semillas heredadas documentadas. | Archivo E3 persistente con DOI y contenedor final verificados; no atribuir determinismo global NumPy o paridad entre plataformas no demostrados. |
| Maestro por ejecución que genere informe y dashboard. | Parcial. | Tablas E2 y maestro de artefactos de 16 sistemas; diseño de vista de 17. | Fuente final común con procedencia/estados y prueba de igualdad app–libro; no añadir resultados ficticios del candidato al CSV histórico. |
| Pruebas unitarias ejecutadas en CI. | Parcial de alcance software. | 35 casos aprobados en informe de verificación y workflows CPU de original/handoff disponibles. | Recibos del paquete final e integración completa, exportaciones y app; los tests del módulo no acreditan el dashboard. |
| Datos o instrucciones exactas de obtención y licencias verificadas. | Parcial. | Fuentes, contratos y disponibilidad de train/cal documentados. | Insumos finales autorizados, hashes/licencias e instrucciones que resuelvan los faltantes; no reconstruir OHLCV desde predicciones. |
| Reproducción independiente desde cero en contenedor, por persona ajena al código, con discrepancia máxima. | Pendiente. | Auditores y comprobaciones de software con alcances propios. | Persona externa, entorno limpio, comandos y recibo verificable de reproducción final; un subagente de IA no sustituye ese requisito. |

### 14.5.8 Productos del entregable

Corresponde a correcciones —7 %—, artefactos —7 %—, dashboard —8 %, 9 % y 5 %—, informe —14 %— y sustentación —7 %—.

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Correcciones E2 con trazabilidad completa y recálculo consistente. | Parcial. | Tabla de revisión interna y auditorías de [sección 4](04_correcciones_e2.md). | Cerrar los controles pendientes; recalcular dependencias y ranking antes/después si una corrección modifica resultados. No se recibieron observaciones del docente. |
| Mejores configuraciones exportadas, ida y vuelta y `metadata.json`. | Parcial. | MLP16/LSTM32/GRU32 TorchScript, paridad puntual en 53 fechas de calibración. | Exportaciones/resto del catálogo, política final y prueba A2 aplicable. ONNX no está verificado; TorchScript es el formato disponible, no una conversión ONNX. |
| Dash Plotly en Render, prueba de humo, contingencia y funcionamiento en sustentación. | Pendiente. | Especificación de [sección 9](09_dashboard_predictivo.md). | App, URL, recursos medidos, prueba de humo y plan de contingencia comprobados; publicar el libro no despliega Dash. |
| Jupyter Book sin errores, ficha de modelo y declaración de datos. | Parcial documental. | Fuentes y evidencias del libro E3, con esta ampliación editorial. | Construcción y revisión del paquete final, `MODEL_CARD.md` y `DATA_STATEMENT.md` completos; documentos propuestos no son productos verificados. |
| Procedencia del mecanismo transferido, cuando corresponda. | Parcial. | Registro candidato y atribuciones en [sección 6](06_modelo_original.md). | Cerrar fuente/adaptación e integrar `experiments/procedencia.md` final, sin fingir novedad de componentes conocidos. |
| Declaración de autoría, contribuciones individuales y uso de IA. | Parcial documental; validación pendiente. | Asistencia de IA reconocida en código, teoría y procedencia. | Declaración final firmada, contribuciones individuales y comprensión responsable del autor; la asistencia declarada no acredita una firma ni defensa. |
| Sustentación: demostración en vivo y capacidad de explicar/modificar el algoritmo. | Pendiente. | Código, formulación y comandos de reproducción disponibles. | Defensa efectiva por el estudiante, preguntas técnicas y modificación/ejecución en vivo; el archivo y sus tests no reemplazan esa evidencia. |

### 14.5.9 Decisión previa al congelamiento del protocolo

| Exigencia de la guía | Estado actual | Evidencia disponible | Vacío que impide darla por cumplida |
|---|---|---|---|
| Definir antes del cierre el hallazgo alternativo que sostendrá el trabajo si no se acredita superioridad significativa y consistente. | Propuesta pendiente de fijar. | Límites y resultados adversos se conservan en el diseño de [secciones 7–8](08_verificacion_estadistica.md). | Registrar el hallazgo alternativo, sus análisis y criterios antes de confirmar; no escogerlo después por ser favorable. |

Se propone como hallazgo alternativo un **benchmark reproducible y una explicación delimitada de los escenarios en que la regla candidata aporta o perjudica**. Si no supera a toda la familia, el informe conservaría las diferencias/IC, potencia y MDE, costes y ablaciones, y distinguiría falta de evidencia, efecto pequeño y deterioro observado. La relación entre acotar la cabeza y predecir colas de σ₇ sería una hipótesis a contrastar, no una explicación ya demostrada. Esta propuesta debe precisarse antes del congelamiento y no reemplaza el objetivo confirmatorio ni permite relajar R3.

La consecuencia de un R3 evaluado y no satisfecho, bajo R1/R2 acreditadas, es la regla condicional descrita en 14.1. No se aplica aquí a condiciones aún pendientes. La selección posterior de semillas, conjuntos, métricas, análisis o rivales para sostener superioridad contraviene la guía y anula esa afirmación; se conservarán el historial, los resultados desfavorables y las desviaciones declaradas.

La [sección 15](15_integridad_academica.md) desarrolla esas reglas de integridad y la [16](16_conclusiones.md) conserva las conclusiones compatibles con la evidencia actual. El [Apéndice A](apendice_a_plantillas.md) ofrece registros editables sin marcar requisitos como cumplidos por disponer de una plantilla.
