# 11. HPO y equidad para el catálogo vigente

**Vigencia al 11/11/2026:** Se conservan los dominios, semillas y oportunidades HPO/NAS en los once orígenes originales. La reducción a tres queda pendiente; sólo se optimiza implementación y orden. [Decisión y ejecutor vigentes](40_calendario_original.md).



**Decisión posterior de diseño (09/11/2026):** [§27 fija seis variantes NAS, cinco cuerpos publicados, rangos HPO y presupuesto para Codex](27_nas_hpo_presupuesto.md). La implementación de nuevas variantes, el perfilado de horas y el preregistro completo siguen pendientes; el contenido siguiente conserva el alcance de su corte anterior.

**Guía original cotejada:** [correcciones y pendientes por página](20_contraste_guia.md). La cuadrícula reducida no cubre el espacio mínimo; controles F2a, NAS condicional y batería estadística requieren completar el plan antes de entrenar.


**Actualización de planificación:** [§19: auditoría real de desarrollo y plan propuesto](19_plan_auditoria.md). Sin entrenamiento de mercado, HPO ni preregistro nuevo.

**Avance de software:** fábrica de los once candidatos, híbrido y flujo sintético implementados y verificados; recetas provisionales, sin datos de mercado. Véase [§18: integración y pruebas sintéticas](18_integracion_sintetica.md). No hay HPO ni benchmark vigente ejecutado.

**Estado al 08/11/2026: no hay HPO propio del nuevo catálogo.** Los diez modelos vigentes son FCN-1D, ResNet-1D, Inception-1D, LSTM-autoencoder con predictor, TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer. CNN-1D + LSTM + atención temporal es el candidato adicional activo, no un ganador ni un modelo con presupuesto preferente.

Esta sección define las condiciones que debe satisfacer el nuevo diseño, **no un plan HPO ya registrado**. El número de configuraciones por receta, los límites de cómputo, los espacios, las semillas y R siguen pendientes de definición. Los diez modelos y el híbrido están seleccionados editorialmente, pero este último está implementado provisionalmente y probado con datos sintéticos, sin entrenamiento de mercado. No se reutilizan las asignaciones ni los compromisos de búsqueda del catálogo retirado.

## 11.1 Presupuesto comparable y completo

El preregistro debe fijar una unidad de comparación, hardware y entorno, límites por receta y reglas ante fallos antes de iniciar búsquedas. Igualar sólo el número de candidatos no asegura igualdad de coste ni de oportunidad de selección. Se deben publicar tanto la asignación como el consumo real.

| Componente | Trabajo que debe entrar en la contabilidad |
|---|---|
| FCN-1D y ResNet-1D | Selección de receta, ajuste de épocas y reajustes de cada configuración/fold |
| Inception-1D | Todas las redes si se adopta un ensemble; distinguirlo de una receta de un miembro |
| LSTM-autoencoder con predictor | Preentrenamiento de reconstrucción, selección de esa fase, ajuste supervisado y cualquier reajuste |
| TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer | Integración validada, HPO de su adaptación y cada ajuste previsto |
| CNN-1D + LSTM + atención temporal | Selección, entrenamiento de los bloques CNN/LSTM/atención y cabeza, reajustes y ablaciones registradas |

La preparación de software se puede informar por separado del coste de entrenamiento, sin ocultar las etapas que sí usan datos para seleccionar modelos. Una receta inicializada o preentrenada con información externa debe declarar origen, acceso, licencia y alcance temporal; no se presupone que exista tal recurso aquí.

No se darán más intentos a una familia por resultados desfavorables. Timeouts, NaN, divergencias y ejecuciones incompletas consumen el presupuesto según una regla previa. Sólo se reanuda la misma configuración/semilla ante una interrupción externa documentada, con su coste acumulado; no se sustituyen semillas de peor desempeño.

## 11.2 Espacios y recetas por congelar

Las arquitecturas se fijarán a partir de fuentes y adaptaciones verificadas, con una justificación individual de lo que es estructura, hiperparámetro de optimización y modificación de tarea. Los defaults compactos de [§4](04_implementacion.md) no son ganadores seleccionados ni réplicas literales de las fuentes.

El registro de cada espacio debe indicar parámetros variables, valores fijos, distribución o cuadrícula, restricciones, orden o semilla de generación, número de oportunidades y parada. Las elecciones de optimizador, pérdida, learning rate, batch, regularización y épocas no se heredan por silencio del experimento histórico.

Para el autoencoder deben cerrarse preentrenamiento secuencial frente a otra estrategia, política de congelado y escalado de objetivos. Una pérdida conjunta sería una adaptación propia que requiere justificación. Para Inception-1D debe resolverse red individual frente a ensemble antes de comparar costes. Para el híbrido deben fijarse anchos, kernels, profundidad, función de atención, enlace y entrenamiento. La estructura propuesta es CNN causal → LSTM unidireccional con secuencia de salida → atención sobre siete estados observados → contexto → cabeza escalar; no se han elegido hiperparámetros definitivos.

Una fuente puede justificar un punto de partida; no demuestra que ese punto sea óptimo para Bitcoin D7. Si el óptimo observado cae en el borde del espacio o el presupuesto resulta insuficiente, se reportará esa limitación. No se amplía un rango después de mirar la evaluación.

La condición de búsqueda arquitectónica se resolverá por modelo en [§14](14_arquitecturas.md). No se inventa una búsqueda NAS realizada ni se hereda una exención de otra receta.

## 11.3 Selección temporal sin fuga

Cada origen externo necesita sus propios bloques internos de selección, completamente anteriores a su límite permitido. Escaladores, imputaciones, selección de épocas, preentrenamiento y ajuste supervisado se aprenden exclusivamente en ese ámbito. Un estado o checkpoint entrenado posteriormente no puede viajar a un fold anterior.

El ejecutor nuevo debe comprobar soportes reales, huecos y disponibilidad de D7×5 `relative_window`, σ₇ poblacional y h=1 conforme al [contrato sincronizado E1–E2–E3](17_contrato_sincronizado.md); no basta confiar en metadatos declarados. La referencia E1 usa etiquetas hasta o−8 y embargo de siete días; el nuevo manifiesto deberá documentar y verificar sus folds y cortes. Retirar HAR no elimina las restricciones temporales del contrato común.

La selección por RMSE debe concatenar las fechas elegibles de cada trayectoria, calcular su RMSE en escala original y después promediar entre semillas. No equivale al promedio no ponderado de RMSE mensuales ni al error de un ensemble no especificado. El nuevo protocolo fijará desempates sin crearlos por redondeo.

La evaluación OOF del procedimiento de selección se distingue de la búsqueda final de una receta con desarrollo. El piloto para dimensionar repeticiones debe corresponder a las recetas congeladas que se desea evaluar. Los datos destinados a calibración predictiva necesitan su separación propia y no se reutilizan inadvertidamente para selección.

## 11.4 Controles específicos del candidato 11

El nuevo híbrido dispone de implementación provisional y pruebas propias en §18, sin evaluación de mercado. Las diez pruebas sintéticas previas corresponden a cuatro redes compactas y a HAR + red residual, que se conserva como código histórico inactivo. No validan el nuevo flujo CNN/LSTM/atención.

Los controles propuestos son **CNN sola**, **LSTM sola** y **CNN–LSTM sin atención**. Deben compartir el contrato de entrada, objetivo, cortes y calendario del híbrido; las diferencias de capacidad y coste se documentarán para interpretar cada ablación. La variante sin atención necesitará una regla de lectura predefinida, sin escogerla tras observar el test.

Sus recetas, oportunidades de selección y costes deben registrarse antes de la búsqueda. Estos controles no amplían tácitamente el catálogo de diez modelos más candidato 11. No se requieren estimación HAR, soporte de 22 días ni cross-fitting de residuos para el nuevo híbrido; conservar el código anterior no los reactiva.

## 11.5 Medición de eficiencia

Se separarán tiempo de búsqueda, preentrenamiento, selección de épocas, reajuste, entrenamiento definitivo, preprocesamiento e inferencia. El protocolo común debe fijar precisión, threads, calentamiento, lotes, orden de temporización y repeticiones antes de medir.

Parámetros, FLOPs/MACs, memoria pico y throughput requieren método, versiones y operaciones no soportadas. Menos parámetros no garantiza menor latencia. Un total no estimable se presenta como tal; un subtotal no se rotula como total y una energía no medida no vale cero.

Las fronteras de Pareto se construirán con error y costes efectivamente observados. Una afirmación de equivalencia o de menor coste necesita margen, comparador e incertidumbre predefinidos; no basta no rechazar una diferencia de error. El coste por mejora conservará signos y unidades y será no estimable cuando su denominador sea inestable.

## 11.6 Estado y evidencia de cierre

| Requisito del nuevo HPO | Estado |
|---|---|
| Recetas y espacios de los once candidatos | Sin congelar |
| Presupuesto, hardware, semillas y límites | Sin preregistro propio |
| Runner integrado, flujo CNN/LSTM/atención y ablaciones auditados | Smoke sintético cerrado en §18; auditoría del pipeline de mercado pendiente |
| Preflight y registro anterior al primer ajuste de mercado | Pendientes |
| Búsquedas completas y consumo real | No ejecutados |
| Recetas seleccionadas y dimensionamiento de R | No disponibles |

El [recibo del 06/11/2026](../_static/completion_preflight/execution_progress.json) acredita sólo un HPO histórico iniciado e incompleto. No confirma actividad actual ni cuenta como búsqueda de este catálogo. Sus [espacios originales](../_static/completion_protocol/hpo.json), [candidatos originales](../_static/completion_protocol/candidate_registry.json) y [registro](../_static/completion_preflight/registration.json) se conservan para auditoría, sin transferir sus asignaciones al nuevo diseño. Los 440 ajustes P0/OOF también son antecedentes, no un nuevo presupuesto consumido.

El orden de registro y evaluación está en [§9](09_protocolo_registrado.md); el estado por modelo, en [§6](06_benchmark_final.md). Hasta cerrar esos requisitos no corresponde anunciar entrenamiento justo, HPO terminado, potencia suficiente ni superioridad del híbrido.
