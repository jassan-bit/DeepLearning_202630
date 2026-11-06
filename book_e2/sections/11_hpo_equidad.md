# 11. Entrenamiento, ajuste equitativo y coste computacional

**Estado:** HPO implementado con [enmienda preparada](../_static/completion_protocol/hpo.json), pendiente de registro y ejecución. El registro [`599a29f`](https://github.com/jassan-bit/DeepLearning_202630/commit/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0) conserva el procedimiento original P0; **HPO y benchmark final todavía no ejecutados**. P0/OOF de cuatro modelos representativos ya se ejecutó y tiene otro propósito, descrito en [§9](09_protocolo_registrado.md). No se presenta su receta pequeña como ganadora de una búsqueda.

Este capítulo desarrolla §§11.1–11.4 de la guía completa, páginas **46–48**, y el requisito de equidad del Apéndice B, página **63**. Las fuentes de la implementación final son el [HPO enmendado](../_static/completion_protocol/hpo.json), la [enmienda](../_static/completion_protocol/analysis_additions.json) y la [revisión arquitectónica](../_static/completion_protocol/architecture_review.json). El [preregistro original](../_static/remediation_protocol/preregistration.json) conserva las familias principales y el alcance histórico P0. Su existencia no acredita ejecución o igualdad de consumo realizada.

## 11.1 Qué recibe el mismo presupuesto

El catálogo vigente tiene **20 configuraciones**. La distinción entre receta ajustable y referencia fija evita atribuir una optimización nueva a modelos históricos:

| Grupo | Número | Regla de comparación |
|---|---:|---|
| Top 10 de §3 | 10 | TSMixer, TiDE, TimeMixer, TimesNet, PatchTST, Informer, Autoformer, iTransformer, Crossformer y S-Mamba; adaptadores finales implementados y revisados por modelo; pipeline real pendiente |
| Bases retunadas E2 | 3 | GRU-E2, LSTM-E2 y TCN-E2 reciben el mismo presupuesto asignado que el Top 10 |
| Referencias clásicas ajustables | 2 | Ridge y ARIMA: espacios propios de regularización/orden, con el mismo número de candidatos y límite asignado |
| Anclas históricas E1 | 3 | GRU/D7-c03, LSTM/D7-c02 y TCN/D7-c02 conservan sus recetas; se identifican como contexto fijo |
| Referencias ingenuas | 2 | Persistencia y naive semanal: sin búsqueda ni realizaciones aleatorias ficticias |

Son **15 recetas ajustables**. A cada una se asignan **12 candidatos** y **172.800 CPU-s**: el registro contiene **180 configuraciones predefinidas**. Ese número no cuenta folds, semillas, selección de épocas, reajustes o evaluación; no equivale a 180 entrenamientos.

Las variantes E2 permiten comparar modelos con ajuste comparable sin borrar las tres anclas. GRU/D7-c03 sigue siendo el baseline heredado; el HPO no lo renombra después de conocer test. La familia adicional contra GRU-E2 estudia otra comparación y conserva su nombre, nivel y alcance.

Se mantienen origen de pronóstico, objetivo, partición, elegibilidad y métrica. Ridge aplana las mismas 35 entradas; ARIMA y las referencias ingenuas usan historia causal del objetivo, una excepción explícita de la ruta de [§7](07_ruta_temporal.md), no un sexto canal para las redes. Esa diferencia de información se declara al interpretar resultados.

**Igualdad asignada no significa igualdad ejecutada.** Se informará cuántos candidatos completó cada modelo, su presupuesto consumido, fallos y censura. Si una familia no termina su programa, no se compensa con más intentos para otra ni se afirma que recibieron la misma optimización realizada. Tampoco se atribuye automáticamente la ganancia frente a una ancla fija sólo a la arquitectura.

## 11.2 Espacios de búsqueda y restricciones

### 11.2.1 Ejes registrados

Las trece recetas profundas conservan exactamente los **once primeros candidatos aleatorios** del registro original, generado con semilla `202611061`, y sustituyen únicamente c12 por un punto respaldado por fuente. Ridge y ARIMA conservan sus doce candidatos originales. El total sigue siendo 180. Los soportes de los once aleatorios son:

| Eje | Espacio |
|---|---|
| Learning rate | Loguniforme entre 10⁻⁴ y 0,003 |
| Weight decay | Loguniforme entre 10⁻⁶ y 0,01 |
| Dropout | Uniforme entre 0 y 0,3 en las posiciones documentadas del modelo |
| Batch de entrenamiento | {16, 32, 64} |
| Profundidad y ancho | Fijados por la estructura fuente/adaptación D7 revisada individualmente en §14; no se presenta como búsqueda un valor constante |
| Cabezas, parches, escalas y kernels | Restricciones de cada backbone y del contexto D7, con correspondencia y fuente explícitas; no se eligen mirando resultados |

La guía pide considerar profundidad/ancho y los parámetros de cada familia, y conserva las estructuras ya definidas. La [revisión individual](../_static/completion_protocol/architecture_review.json) resuelve la configuración final de los diez backbones con fuentes/presets fijados y modificaciones D7 explícitas. Sus dimensiones no se sustituyen por las pequeñas P0. La excepción estructural se fundamenta modelo por modelo; no equivale a una excepción de todo el HPO.

El candidato **c12** usa LR/dropout identificados en artículo o código fuente, con **weight decay=0,0001 y batch=32 comunes**. No se presenta como réplica literal del mejor resultado del paper. Las únicas salidas del soporte aleatorio permitidas son sus valores exactos: TSMixer dropout=0,9; TiDE LR=0,00003822279848104051; TimeMixer LR=0,01 y S-Mamba LR=0,00007. El manifiesto identifica también los puntos que caen dentro del soporte. Una excepción de c12 no autoriza candidatos aleatorios adicionales fuera del rango, ni cambia T tras observar resultados. Para PatchTST se prioriza dropout=0,2 explícito en el artículo frente al 0,3 de otro script; las condiciones se distinguen.

AdamW respeta únicamente las exenciones `_no_weight_decay` explícitas de la fuente Mamba para `A_log`/`D`; los demás parámetros reciben el decay del candidato. No se añaden exenciones generales de bias o normalización sin declararlas.

Si un caso queda realmente sin diseño definido, se activa el protocolo condicional de [§14](14_arquitecturas.md). La tasa y los patrones de dropout de ese NAS siguen su espacio fijo 0,2 y no se sustituyen por la cuadrícula general de tasas. Su HPO previo se hace sobre el control especificado antes de comparar arquitecturas.

Ridge recibe doce valores de α en escala logarítmica **10⁻⁶–10³**, con intercepto y representación fija. ARIMA usa el universo completo de doce órdenes **p∈{0,1,2}, d∈{0,1}, q∈{0,1}**; tendencia constante cuando d=0 y sin tendencia cuando d=1. A estos estimadores no se les inventan dropout, batch o learning rate para simular igualdad de parámetros.

Los estimadores clásicos conservan su salida cruda y aplican la transformación fija `max(0, predicción)` al publicar el pronóstico de volatilidad. Una salida no finita es un fallo registrado, no se transforma en cero. La regla está fijada antes de los resultados y no elimina fechas del panel.

### 11.2.2 Selección temporal anidada

El programa comparte dos semillas HPO, **2026110611 y 2026110612**, y diez orígenes externos mensuales de enero a octubre de 2024. Las referencias deterministas se ajustan una vez por candidato/fold, sin contar dos copias idénticas como ejecuciones independientes.

En cada origen externo se forman tres bloques internos de treinta días en su pasado permitido. Las etiquetas de ajuste terminan en q−8; la selección de épocas usa la ventana nominal q−90 a q−8, **83 fechas potenciales antes de elegibilidad**, con ajuste temprano hasta q−98. Se reestima cada transformación sólo en su entrenamiento autorizado y se reajusta desde una inicialización nueva durante las épocas elegidas.

Se concatenan las fechas elegibles de cada trayectoria, se calcula su RMSE en escala original y después se promedia entre semillas. No se optimiza una media no ponderada de RMSE mensuales ni la RMSE de un ensamble. Se minimiza ese valor; los desempates siguen latencia de desarrollo bajo el mismo protocolo, parámetros e identificador de candidato, sin crear empates por redondeo.

Los resultados de los meses externos describen el **procedimiento de selección**, no una receta única que haya sido elegida sin usarlos. La receta final procede de una búsqueda forward adicional en desarrollo, declarada como selección y congelada antes de P1. Ninguna de estas búsquedas usa la reserva histórica para decidir parámetros.

La igualdad de T no prueba que los espacios sean igual de informativos para cada familia. Se deben reportar el espacio, la posición de la receta seleccionada, las restricciones y las curvas de importancia. Un parámetro fijado, un óptimo en el borde o un rango insuficiente se explican como tales; no se amplía el espacio tras observar test.

## 11.3 Entrenamiento, incidencias y ficha por receta

Las reglas siguientes corresponden a las futuras recetas profundas ajustables; las anclas conservan su configuración E1 y P0 conserva su propio manifiesto:

| Control | Regla registrada |
|---|---|
| Optimización | AdamW; MSE con escala positiva de Y aprendida exclusivamente en train |
| Scheduler | Sin scheduler |
| Precisión y ejecución | FP32, un thread y un interop thread; mismo hardware para comparaciones pertinentes |
| Estabilidad | Gradient clipping de norma 1 y determinismo explícito |
| Parada | Máximo 100 épocas, paciencia 12, mejora mínima 0 en MSE de validación |
| Límite por ajuste | 900 s por ajuste de optimizador, dentro del presupuesto total asignado |
| Gradientes | No actualizar pesos con validación o test |

La ficha de modelo exigida por §11.3 incluirá nombre/familia/DOI; repositorio y commit; pesos preentrenados, corpus y licencia o «ninguno»; arquitectura, parámetros entrenables/totales y FLOPs; espacios y valores elegidos; epochs, checkpoint, precisión y hardware; costes por etapa; resultados por semilla/fold y todas las incidencias.

Se conserva el registro de **todas** las configuraciones, no sólo la ganadora. Los fallos por NaN, divergencia, checkpoint, auditoría causal, límite de recursos o interrupción quedan identificados. No se elimina una semilla por mal desempeño ni se reemplazan candidatos fallidos selectivamente. Sólo se permite reanudar la misma configuración/semilla ante una interrupción externa documentada, cargando ese consumo al mismo presupuesto.

P0/OOF consta de cuatro modelos y cinco semillas nuevas, con 220 trayectorias / 440 ajustes completados. Es una ejecución de factibilidad y diagnóstico de recetas fijas; **no son el HPO de quince familias ni su R final**. Tras el HPO se requiere P1 con recetas congeladas, dimensionamiento y nuevo cierre de pesos. Los modelos no pilotados no reciben una certificación empírica de potencia a partir de los representantes.

## 11.4 Eficiencia y coste por mejora

El consumo realizado se medirá por separado de la asignación. Se distinguen tiempo de búsqueda, selección de épocas, reajuste, entrenamiento definitivo e inferencia, junto a memoria pico, parámetros, FLOPs/MACs y throughput. Se documentan versiones y operaciones no soportadas por el contador; menos parámetros no implica menor latencia. El contador propio declara MAC real=2 y un convenio explícito por operación. FFT nativa u operadores flotantes no soportados dejan el total como **no estimable** y publican sólo un subtotal parcial, nunca un cero ni un total completo inventado. Throughput usa batch32, 32 calentamientos y 31 bloques de 32 llamadas, en CPU sin entrenamiento concurrente.

El protocolo de latencia fija **batch 1**, CPU común, un thread, 256 llamadas de calentamiento y **61 bloques intercalados de 256 llamadas**, con semilla de orden `202611065`. La unidad es el bloque de temporización; no una nueva observación de mercado. Se informa la precisión neural FP32 y la clásica correspondiente, sin afirmar que sean idénticas. El preprocesamiento se mide por separado del forward completo con la cabeza escalar.

Se construirán fronteras de Pareto RMSE–latencia, RMSE–parámetros y frente a cada otro coste pertinente. Un valor no medido queda ausente, no cero; no beneficia al modelo ni elimina su error del ranking descriptivo. Una frontera puntual no demuestra dominancia con incertidumbre.

La relación entre ganancia de error y coste incremental debe conservar signos y unidades. Si el coste incremental es cero o próximo a cero, se informa la indefinición o inestabilidad del cociente, sin épsilon arbitrario. Un modelo más preciso y barato se describe con ambas diferencias, sin ocultarlo tras un ranking de cocientes. El bootstrap debe propagar incertidumbre predictiva y de temporización con sus unidades respectivas.

La enmienda fija diecinueve comparaciones de equivalencia práctica y menor latencia frente al mejor modelo elegido entre los veinte en desarrollo. Usa TOST con ±δ y una intersección-unión con ahorro de coste, Holm α=0,01, en una familia exploratoria separada. La referencia no se elige en test y un no rechazo de diferencia no prueba equivalencia. La energía y emisiones pueden estimarse si hay método y hardware trazables, incluyendo HPO; una magnitud no medida no se publica como cero ni se presenta una estimación por TDP como medición directa.

## 11.5 Condiciones para abrir el HPO real

Las quince asignaciones y los 180 candidatos están especificados; los diez backbones y sus adaptaciones ya cuentan con revisión independiente y controles sintéticos. **Antes del HPO real faltan el registro de esta enmienda y los recibos integrados del entorno, datos, fuentes, código, candidatos y recursos.** Si aplica NAS, las fases posteriores exigen además la fase 2 completa y el ganador congelado.

El control compara los bytes con su commit registrado y exige recibos de evidencia con hashes. Esto permite detectar cambios y falta de prerrequisitos; no convierte un fichero existente en certificación científica. Los resultados negativos, la búsqueda incompleta y la potencia insuficiente siguen siendo desenlaces posibles y deben conservarse.

- [Registro nuevo de 180 candidatos sin evaluar](../_static/completion_protocol/candidate_registry.json), [HPO con puntos fuente](../_static/completion_protocol/hpo.json) y [enmienda](../_static/completion_protocol/analysis_additions.json).
- [Validación documental original P0](../_static/remediation_protocol/validation.json), conservada con su alcance.
- Antecedentes conservados: [HPO original](../_static/remediation_protocol/hpo.json), [decisiones arquitectónicas previas](../_static/remediation_protocol/architecture_decisions.json) y [preregistro P0](../_static/remediation_protocol/preregistration.json).
- [Protocolo y piloto](09_protocolo_registrado.md), [tabla de benchmark pendiente](06_benchmark_final.md) y [búsqueda arquitectónica condicional](14_arquitecturas.md).
