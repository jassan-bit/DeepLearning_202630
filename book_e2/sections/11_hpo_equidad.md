# 11. HPO y equidad para el catálogo vigente

**Estado al 08/11/2026: no hay HPO propio del nuevo catálogo.** Los diez modelos vigentes son FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer. HAR + red residual es un candidato adicional, no un ganador ni un modelo con presupuesto preferente.

Esta sección define las condiciones que debe satisfacer el nuevo diseño, **no un plan HPO ya registrado**. El número de candidatos, los límites de cómputo, los espacios, las semillas y R siguen pendientes de definición. No se reutilizan las asignaciones ni los compromisos de búsqueda del catálogo retirado.

## 11.1 Presupuesto comparable y completo

El preregistro debe fijar una unidad de comparación, hardware y entorno, límites por receta y reglas ante fallos antes de iniciar búsquedas. Igualar sólo el número de candidatos no asegura igualdad de coste ni de oportunidad de selección. Se deben publicar tanto la asignación como el consumo real.

| Componente | Trabajo que debe entrar en la contabilidad |
|---|---|
| FCN-1D y ResNet-1D | Selección de receta, ajuste de épocas y reajustes de cada configuración/fold |
| Inception-1D | Todas las redes si se adopta un ensemble; distinguirlo de una receta de un miembro |
| LSTM-autoencoder con predictor | Preentrenamiento de reconstrucción, selección de esa fase, ajuste supervisado y cualquier reajuste |
| TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer | Integración validada, HPO de su adaptación y cada ajuste previsto |
| HAR + red residual | Estimación HAR, generación temporal OOF, selección y entrenamiento del corrector, reajustes y controles |

La preparación de software se puede informar por separado del coste de entrenamiento, sin ocultar las etapas que sí usan datos para seleccionar modelos. Una receta inicializada o preentrenada con información externa debe declarar origen, acceso, licencia y alcance temporal; no se presupone que exista tal recurso aquí.

No se darán más intentos a una familia por resultados desfavorables. Timeouts, NaN, divergencias y ejecuciones incompletas consumen el presupuesto según una regla previa. Sólo se reanuda la misma configuración/semilla ante una interrupción externa documentada, con su coste acumulado; no se sustituyen semillas de peor desempeño.

## 11.2 Espacios y recetas por congelar

Las arquitecturas se fijarán a partir de fuentes y adaptaciones verificadas, con una justificación individual de lo que es estructura, hiperparámetro de optimización y modificación de tarea. Los defaults compactos de [§4](04_implementacion.md) no son ganadores seleccionados ni réplicas literales de las fuentes.

El registro de cada espacio debe indicar parámetros variables, valores fijos, distribución o cuadrícula, restricciones, orden o semilla de generación, número de oportunidades y parada. Las elecciones de optimizador, pérdida, learning rate, batch, regularización y épocas no se heredan por silencio del experimento histórico.

Para el autoencoder deben cerrarse preentrenamiento secuencial frente a otra estrategia, política de congelado y escalado de objetivos. Una pérdida conjunta sería una adaptación propia que requiere justificación. Para Inception-1D debe resolverse red individual frente a ensemble antes de comparar costes. Para HAR deben fijarse ventanas de calendario, transformación, enlace y cómo se generan los residuos disponibles.

Una fuente puede justificar un punto de partida; no demuestra que ese punto sea óptimo para Bitcoin D7. Si el óptimo observado cae en el borde del espacio o el presupuesto resulta insuficiente, se reportará esa limitación. No se amplía un rango después de mirar la evaluación.

La condición de búsqueda arquitectónica se resolverá por modelo en [§14](14_arquitecturas.md). No se inventa una búsqueda NAS realizada ni se hereda una exención de otra receta.

## 11.3 Selección temporal sin fuga

Cada origen externo necesita sus propios bloques internos de selección, completamente anteriores a su límite permitido. Escaladores, imputaciones, selección de épocas, preentrenamiento y ajuste supervisado se aprenden exclusivamente en ese ámbito. Un estado o checkpoint entrenado posteriormente no puede viajar a un fold anterior.

Las guardas preliminares exigen una separación de etiquetas del estimador base hasta q−8 para el origen q. El ejecutor nuevo debe comprobar soportes reales, huecos y disponibilidad; no basta confiar en metadatos declarados. La elección exacta de folds, corte y embargo debe constar en el nuevo manifiesto.

La selección por RMSE debe concatenar las fechas elegibles de cada trayectoria, calcular su RMSE en escala original y después promediar entre semillas. No equivale al promedio no ponderado de RMSE mensuales ni al error de un ensemble no especificado. El nuevo protocolo fijará desempates sin crearlos por redondeo.

La evaluación OOF del procedimiento de selección se distingue de la búsqueda final de una receta con desarrollo. El piloto para dimensionar repeticiones debe corresponder a las recetas congeladas que se desea evaluar. Los datos destinados a calibración predictiva necesitan su separación propia y no se reutilizan inadvertidamente para selección.

## 11.4 Controles específicos del candidato 11

El prototipo recibe un pronóstico HAR externo. **No implementa HAR/OLS ni cross-fitting real**, y su truncamiento puede anular el gradiente si la suma resulta negativa. Resolver ese problema forma parte del diseño previo al HPO, no de ajustes oportunistas tras observar resultados.

Dentro del train autorizado se deben generar predicciones HAR realmente fuera de muestra y entrenar la red sólo con sus residuos ya disponibles. Para validación se aplicará la receta de reajuste registrada sin usar residuos de esa validación como etiquetas del corrector. La diferencia entre HAR OOF y HAR reajustado debe quedar explícita.

HAR solo, red sola y el híbrido se compararán con calendario y tratamiento de salida consistentes. Los controles de soporte adicional son necesarios para distinguir una ventaja arquitectónica de disponer de más historia. Todos sus ajustes se contabilizarán sin reintroducir automáticamente modelos del catálogo descartado.

## 11.5 Medición de eficiencia

Se separarán tiempo de búsqueda, preentrenamiento, selección de épocas, reajuste, entrenamiento definitivo, preprocesamiento e inferencia. El protocolo común debe fijar precisión, threads, calentamiento, lotes, orden de temporización y repeticiones antes de medir.

Parámetros, FLOPs/MACs, memoria pico y throughput requieren método, versiones y operaciones no soportadas. Menos parámetros no garantiza menor latencia. Un total no estimable se presenta como tal; un subtotal no se rotula como total y una energía no medida no vale cero.

Las fronteras de Pareto se construirán con error y costes efectivamente observados. Una afirmación de equivalencia o de menor coste necesita margen, comparador e incertidumbre predefinidos; no basta no rechazar una diferencia de error. El coste por mejora conservará signos y unidades y será no estimable cuando su denominador sea inestable.

## 11.6 Estado y evidencia de cierre

| Requisito del nuevo HPO | Estado |
|---|---|
| Recetas y espacios de los once candidatos | Sin congelar |
| Presupuesto, hardware, semillas y límites | Sin preregistro propio |
| Runner integrado y cross-fitting HAR auditado | Pendiente |
| Preflight y registro anterior al primer ajuste de mercado | Pendientes |
| Búsquedas completas y consumo real | No ejecutados |
| Recetas seleccionadas y dimensionamiento de R | No disponibles |

El [recibo del 06/11/2026](../_static/completion_preflight/execution_progress.json) acredita sólo un HPO histórico iniciado e incompleto. No confirma actividad actual ni cuenta como búsqueda de este catálogo. Sus [espacios originales](../_static/completion_protocol/hpo.json), [candidatos originales](../_static/completion_protocol/candidate_registry.json) y [registro](../_static/completion_preflight/registration.json) se conservan para auditoría, sin transferir sus asignaciones al nuevo diseño. Los 440 ajustes P0/OOF también son antecedentes, no un nuevo presupuesto consumido.

El orden de registro y evaluación está en [§9](09_protocolo_registrado.md); el estado por modelo, en [§6](06_benchmark_final.md). Hasta cerrar esos requisitos no corresponde anunciar entrenamiento justo, HPO terminado, potencia suficiente ni superioridad del híbrido.
