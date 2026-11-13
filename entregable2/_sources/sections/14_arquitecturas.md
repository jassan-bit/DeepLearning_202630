# 14. Arquitecturas del catálogo vigente y adaptación a D7

**Estado vigente al 13/11/2026:** el catálogo ejecutado conserva los diez cuerpos publicados documentados en [§44](44_arquitecturas_publicadas.md) y su actualización de [§48](48_protocolo_reducido.md), incluida **TIDE-D7-PUBLISHED-R0-v2**. La NAS de 1.000 diseños sigue siendo una obligación condicional para cuerpos no definidos; las seis plazas HPO no se presentan como NAS. El [benchmark final](06_benchmark_final.md) ya está completado. Las decisiones y estados anteriores de este capítulo permanecen como antecedentes fechados.

**Antecedente conservado:** **Vigencia documental (11/11/2026):** existe selección HPO/NAS registrada y ejecutada para el catálogo actual; las afirmaciones de ausencia de HPO o de falta de registro de selección corresponden a cortes anteriores. El benchmark final, R y su cierre inferencial siguen pendientes. [Registro y alcance actuales (§9)](09_protocolo_registrado.md) · [Calendario (§40)](40_calendario_original.md) · [Corte de resultados (§41)](41_avance_optimizacion_11_origenes.md) · [Auditoría y trabajo paralelo (§42)](42_auditoria_y_trabajo_paralelo.md).

## Antecedentes de arquitectura hasta el 11/11/2026

**Decisión histórica al 11/11/2026:** Se conservan los cinco cuerpos publicados y las seis variantes NAS. No se conceden cuatro exenciones nuevas ni se recortan arquitecturas; se mantiene el calendario original. [Decisión y ejecutor vigentes](40_calendario_original.md).



**Decisión posterior de diseño (09/11/2026):** [§27 fija seis variantes NAS, cinco cuerpos publicados, rangos HPO y presupuesto para Codex](27_nas_hpo_presupuesto.md). La implementación de nuevas variantes, el perfilado de horas y el preregistro completo siguen pendientes; el contenido siguiente conserva el alcance de su corte anterior.

**Once propuestas concretas:** [fichas de cuerpos, adaptaciones y decisiones pendientes](22_fichas_recetas.md). No se implementan ni entrenan las recetas nuevas en este corte.


**Siguiente especificación:** [comparadores temporales, matriz NAS y costes por escenarios](21_comparadores_nas.md). Propuesta sin entrenamiento, búsqueda ni cierre automático del protocolo.


**Guía original cotejada:** [correcciones y pendientes por página](20_contraste_guia.md). La cuadrícula reducida no cubre el espacio mínimo; controles F2a, NAS condicional y batería estadística requieren completar el plan antes de entrenar.


**Avance de software:** fábrica de los once candidatos, híbrido y flujo sintético implementados y verificados; recetas provisionales, sin datos de mercado. Véase [§18: integración y pruebas sintéticas](18_integracion_sintetica.md). No hay HPO ni benchmark vigente ejecutado.

**Selección vigente al 08/11/2026:** diez modelos neurales y CNN-1D + LSTM + atención temporal como candidato adicional activo. La selección reemplaza el catálogo anterior en la presentación; no convierte los prototipos en recetas definitivas ni transfiere registros, pruebas o resultados de otras arquitecturas.

Cuatro prototipos compactos y un corrector con HAR externo, ahora histórico inactivo, tienen diez pruebas sintéticas documentadas. **El nuevo híbrido está implementado provisionalmente y probado con datos sintéticos en §18, sin entrenamiento de mercado. Las diez pruebas del prototipo anterior no se transfieren al nuevo código.** Los seis adaptadores existentes están integrados y comprobados con datos sintéticos; su validación de mercado sigue pendiente. **Ninguno tiene resultados de mercado acreditados bajo el nuevo diseño.**

## 14.1 Diez arquitecturas, una tarea neural

La entrada común es D7×5 derivado de OHLCV con `relative_window` y la salida es σ₇ poblacional a un día, sin anualización y en su escala original, conforme al [contrato sincronizado E1–E2–E3](17_contrato_sincronizado.md). La fuente de una arquitectura, su configuración publicada, el código reutilizado y la adaptación de E2 deben identificarse por separado.

| N.º | Modelo | Mecanismo | Decisión arquitectónica pendiente |
|---|---|---|---|
| 1 | FCN-1D | Extracción por convoluciones y pooling temporal | Anchos, kernels, normalización, padding y cabeza de regresión |
| 2 | ResNet-1D | Bloques convolucionales y conexiones de salto | Profundidad, proyecciones, activaciones y campo receptivo compatible con D7 |
| 3 | Inception-1D | Ramas convolucionales de varias escalas | Kernels, bottleneck, módulos, residuales y red individual frente a ensemble |
| 4 | LSTM-autoencoder con predictor | Representación recurrente, reconstrucción y cabeza supervisada | Encoder/decoder, dimensión latente, preentrenamiento y ajuste de predicción |
| 5 | TimesNet | Periodicidad y convolución sobre representación temporal | Frecuencias disponibles en D7, extensión, bloques y salida escalar |
| 6 | TSMixer | Mezcla temporal y de canales mediante MLP | Bloques, normalización, dimensiones y lectura de volatilidad |
| 7 | TiDE | Encoder/decoder denso residual | Proyecciones, covariables permitidas y cabeza de σ₇ |
| 8 | TimeMixer | Descomposición y mezcla multiescala | Descensos válidos, escalas no vacías y combinación de predictores |
| 9 | PatchTST | Atención sobre parches temporales | Longitud/stride de parches, padding, atención y cabeza de tarea |
| 10 | iTransformer | Atención entre tokens de variables | Tokens OHLCV, normalización, encoder y proyección escalar |

La cobertura incluye convolución, recurrencia/autoencoder, mezcla MLP y atención. La correspondencia con las familias y configuraciones exigidas por el curso debe verificarse antes de afirmar cumplimiento; enumerar nombres no demuestra que cada implementación preserve su mecanismo.

## 14.2 Prototipos compactos frente a fuentes

Las [fichas bibliográficas](16_fichas_nuevas.md) distinguen los resultados publicados de la adaptación propuesta. FCN y ResNet parten de una fuente de clasificación de series; InceptionTime también es una arquitectura/ensemble de clasificación. Su precisión en esos datasets no equivale a pronóstico de volatilidad de Bitcoin.

| Modelo | Prototipo disponible | Diferencia que impide llamarlo réplica literal |
|---|---|---|
| FCN-1D | Tres convoluciones 3/3/1, ancho 16, ReLU y pooling promedio | La fuente usa otros kernels/anchos y batch normalization |
| ResNet-1D | Dos bloques con convoluciones 3/3 y proyección de atajo | Menor profundidad y receta distinta de la fuente |
| Inception-1D | Un módulo compacto con ramas 1/3/5, bottleneck y max-pooling | No es el ensemble publicado ni conserva sus kernels y profundidad |
| LSTM-autoencoder con predictor | Encoder y decoder LSTM, latente repetido, reconstrucción y cabeza Softplus | La fuente usa preentrenamiento apilado; el flujo común prueba dos fases sintéticas, pero no reproduce ese cuerpo ni su entrenamiento de mercado |

Los cuatro usan ancho predeterminado 16 y salida positiva. Son elecciones de prototipado, no valores seleccionados por HPO ni configuraciones atribuidas al profesor. Una pérdida conjunta de reconstrucción/predicción sería una adaptación propia; no se deduce del test de gradientes, que combina objetivos sólo para comprobar conexiones.

Kernels largos pueden requerir padding con D7. Repetir bordes o introducir ceros no aporta observaciones anteriores ni futuras. Aplicar una operación sobre una ventana ya cerrada es distinto de aplicarla a la serie completa antes de cortar cada origen; sólo el segundo procedimiento podría filtrar información posterior inadvertidamente.

## 14.3 Revisión de los seis adaptadores reutilizables

La [matriz histórica de revisión](../_static/completion_protocol/architecture_review.json) conserva fuentes, presets y cambios anteriores. Para el nuevo diseño se revisarán únicamente los seis modelos de esta selección y su código efectivamente integrado:

- **TimesNet:** verificar cuántas frecuencias no nulas existen con la longitud usada y seleccionar periodos por origen, sin usar otras observaciones del batch para decidirlos.
- **TSMixer:** comprobar mezclas y ejes de normalización; la predicción en evaluación no debe depender de los acompañantes del batch.
- **TiDE:** conservar bloques y conexiones declaradas, sin observaciones futuras en el decoder. Una salida en escala OHLCV no se desnormaliza como si fuera σ₇.
- **TimeMixer:** justificar cada descenso y escala con sólo siete entradas; no admitir longitudes vacías ni presentar padding como nueva historia.
- **PatchTST:** documentar los parches disponibles, normalización y lectura; una receta publicada para ventanas largas exige una adaptación visible a D7.
- **iTransformer:** los tokens representan las variables observadas. La cabeza de volatilidad y su escala deben diferenciarse de un proyector que pronostica cada variable original.

La inspección anterior de un adaptador no verifica automáticamente el nuevo runner, sus transformaciones o la ausencia de fuga. Los hashes deben corresponder a la receta concreta que se va a registrar.

## 14.4 Modelo 11: CNN-1D + LSTM + atención temporal

El candidato activo combina extracción local, recurrencia y agregación temporal. Es un **diseño editorial propuesto**, no una arquitectura exacta verificada del curso o de la literatura, una innovación acreditada ni un modelo superior demostrado. Tiene código y pruebas sintéticas propias en §18; no tiene entrenamiento ni resultados de mercado.

| Etapa propuesta | Operación y forma conceptual | Restricción |
|---|---|---|
| Entrada | Tensor `(batch, 7, 5)` bajo `relative_window` | Sólo siete días observados y cinco canales derivados de OHLCV; sin rezagos de σ₇ |
| CNN-1D causal | Extraer rasgos locales y conservar `(batch, 7, c)` | Cada posición usa sólo ella y posiciones anteriores de la ventana; padding únicamente a la izquierda si se necesita |
| LSTM unidireccional | Recorrer los rasgos en orden temporal y devolver `(batch, 7, d)` con `return_sequences=True` | Conservar los siete estados; no bidireccionalidad ni estado de otro origen |
| Atención temporal | Puntuar y normalizar pesos sobre los siete estados observados | Ningún acceso a la etiqueta futura ni a posiciones fuera de la ventana |
| Contexto | Suma ponderada de estados, forma `(batch, d)` | Agregar información ya observada, sin crear historia adicional |
| Cabeza escalar | Mapear contexto a un pronóstico de σ₇, forma `(batch,)` | Enlace, pérdida y cualquier transformación por definir y evaluar en unidades originales |

Las letras `c` y `d` representan dimensiones de la receta. §18 fija opciones provisionales explícitas (ancho8 para el smoke, kernel3, atención aditiva y Softplus); su selección definitiva, regularización, presupuesto e hiperparámetros para mercado siguen pendientes del nuevo registro. Sus pesos no se presentarán como una explicación causal del mercado.

La CNN conserva la resolución de siete posiciones para que la LSTM y la atención reciban los siete estados propuestos. El padding no aporta observaciones nuevas. La atención puede ponderar toda la ventana porque los siete días están disponibles en el origen de predicción; no es una atención a datos futuros. La causalidad completa también depende de la construcción de ventanas, transformaciones y cortes de entrenamiento.

Se proponen ablaciones **CNN sola**, **LSTM sola** y **CNN–LSTM sin atención**, con el mismo contrato de información y una regla de lectura fijada para cada una. No se añaden como posiciones del catálogo. Se comprobaron mecánicamente en las pruebas nuevas; no se entrenaron ni compararon con mercado. Cualquier afirmación de mejora debe apoyarse en el nuevo protocolo y permitir resultados nulos o desfavorables.

**Antecedente inactivo:** el corrector HAR anterior recibe un pronóstico externo y calcula una corrección truncada; su código, limitaciones y diez pruebas compartidas con las cuatro redes compactas se preservan. El soporte 1/7/22, HAR/OLS y el cross-fitting residual pertenecen a ese prototipo retirado, no al candidato activo. [§4](04_implementacion.md) delimita esa evidencia histórica.

## 14.5 Cuándo corresponde una búsqueda estructural

La guía exige resolver una arquitectura no definida mediante su procedimiento de búsqueda. Esa condición se debe evaluar **de nuevo y por receta**: ni un nombre de familia ni la exención de una versión anterior bastan para cerrar el caso actual.

Antes de cualquier búsqueda deben registrarse espacio, restricciones, control, criterio de selección, presupuesto, separación temporal y confirmación de los finalistas. Si la guía impone un tamaño o fases determinados, la nueva especificación debe demostrarlos explícitamente. No se reutiliza por defecto el espacio anterior ni se afirma que se haya ejecutado NAS.

Una estructura verificada en una fuente puede justificar una adaptación definida, siempre que se documenten sus cambios. Si la revisión detecta que se ha sustituido el mecanismo por otro genérico, debe corregirse o reclasificarse antes de entrenar. No se concede una excepción para evitar una obligación ni se aumenta la estructura después de mirar el test.

## 14.6 Evidencia necesaria para cerrar la arquitectura

Cada ficha ejecutable debe reunir:

- Fuente y localización exacta de cada decisión, repositorio/commit y licencia aplicable; «no verificado» cuando corresponda.
- Diagrama, dimensiones, ecuaciones o reglas de los bloques y adaptación explícita a D7×5 → σ₇.
- Cabeza, transformación del objetivo, enlace, parámetros entrenables y receta de entrenamiento; fases del autoencoder y flujo completo CNN/LSTM/atención con sus ablaciones.
- Formas, gradientes finitos, checkpoint, invariancia en evaluación y pruebas de causalidad sobre el pipeline real.
- Costes medidos con método común y limitaciones del contador, sin convertir complejidad asintótica o cifras de otro dataset en mediciones E2.
- Identidad de fuente, entorno y configuración vinculada al nuevo preregistro.

Las diez pruebas sintéticas históricas verifican aspectos de cuatro prototipos compactos y del corrector HAR retirado; no cubren el nuevo híbrido ni cierran estas fichas para los once modelos. El [protocolo pendiente](09_protocolo_registrado.md) y la [equidad de HPO](11_hpo_equidad.md) separan decisiones, implementaciones, ajustes y resultados. Los [artefactos históricos](../_static/completion_download/completion_bundle.zip) permanecen disponibles para auditoría, sin reintroducir su catálogo en la presentación vigente.
