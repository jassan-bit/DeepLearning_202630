# 14. Arquitecturas del catálogo vigente y adaptación a D7

**Selección vigente al 08/11/2026:** diez modelos neurales y HAR + red residual como candidato adicional. La selección reemplaza el catálogo anterior en la presentación; no convierte los prototipos en recetas definitivas ni transfiere registros, pruebas o resultados de otras arquitecturas.

Cuatro prototipos compactos y un corrector con HAR externo tienen diez pruebas sintéticas documentadas. Los seis modelos restantes cuentan con código histórico, pero requieren integración y validación nueva. **Ninguno tiene resultados de mercado acreditados bajo el nuevo diseño.**

## 14.1 Diez arquitecturas, una tarea neural

La entrada propuesta es D7×5 OHLCV y la salida es σ₇ a un día, en su escala original. La fuente de una arquitectura, su configuración publicada, el código reutilizado y la adaptación de E2 deben identificarse por separado.

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
| LSTM-autoencoder con predictor | Encoder y decoder LSTM, latente repetido, reconstrucción y cabeza Softplus | La fuente respalda preentrenamiento seguido de ajuste supervisado; el prototipo no integra esa política |

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

## 14.4 Modelo 11: HAR + red residual

El candidato combina agregaciones de historia de σ₇ con una corrección neuronal. HAR tiene un antecedente establecido en Corsi (2009), [DOI 10.1093/jjfinec/nbp001](https://doi.org/10.1093/jjfinec/nbp001). Esa fuente no valida por sí misma el uso de σ₇ diario de este proyecto ni acredita novedad o superioridad del híbrido.

La implementación preliminar recibe `HAR_raw` externo y calcula `max(0, HAR_raw + corrección_firmada(D7×5))`. No estima HAR ni genera residuos OOF. Si la suma es negativa, el truncamiento produce gradiente cero; la solución de entrenamiento sigue pendiente. No debe entrenarse con datos reales presentando ese enlace provisional como una pérdida ya resuelta.

El soporte preliminar 1/7/22 usa días calendario de σ₇, con cierres que pueden llegar hasta t−29. Deben fijarse las ventanas para Bitcoin, el calentamiento y los controles HAR solo/red sola/soporte comparable. La corrección se entrenará únicamente con residuos causalmente fuera de muestra y disponibles dentro del train autorizado, conforme al futuro preregistro.

La literatura de [§15](15_revision_modelos.md) incluye antecedentes de corrección residual con resultados desfavorables. El mecanismo no garantiza que el candidato 11 supere a ninguno de los diez modelos; una comparación registrada debe permitir y comunicar ese desenlace.

## 14.5 Cuándo corresponde una búsqueda estructural

La guía exige resolver una arquitectura no definida mediante su procedimiento de búsqueda. Esa condición se debe evaluar **de nuevo y por receta**: ni un nombre de familia ni la exención de una versión anterior bastan para cerrar el caso actual.

Antes de cualquier búsqueda deben registrarse espacio, restricciones, control, criterio de selección, presupuesto, separación temporal y confirmación de los finalistas. Si la guía impone un tamaño o fases determinados, la nueva especificación debe demostrarlos explícitamente. No se reutiliza por defecto el espacio anterior ni se afirma que se haya ejecutado NAS.

Una estructura verificada en una fuente puede justificar una adaptación definida, siempre que se documenten sus cambios. Si la revisión detecta que se ha sustituido el mecanismo por otro genérico, debe corregirse o reclasificarse antes de entrenar. No se concede una excepción para evitar una obligación ni se aumenta la estructura después de mirar el test.

## 14.6 Evidencia necesaria para cerrar la arquitectura

Cada ficha ejecutable debe reunir:

- Fuente y localización exacta de cada decisión, repositorio/commit y licencia aplicable; «no verificado» cuando corresponda.
- Diagrama, dimensiones, ecuaciones o reglas de los bloques y adaptación explícita a D7×5 → σ₇.
- Cabeza, transformación del objetivo, enlace, parámetros entrenables y receta de entrenamiento; fases separadas para autoencoder e híbrido.
- Formas, gradientes finitos, checkpoint, invariancia en evaluación y pruebas de causalidad sobre el pipeline real.
- Costes medidos con método común y limitaciones del contador, sin convertir complejidad asintótica o cifras de otro dataset en mediciones E2.
- Identidad de fuente, entorno y configuración vinculada al nuevo preregistro.

Las diez pruebas sintéticas actuales son un comienzo de verificación, no el cierre de estas fichas para los once modelos. El [protocolo pendiente](09_protocolo_registrado.md) y la [equidad de HPO](11_hpo_equidad.md) separan decisiones, implementaciones, ajustes y resultados. Los [artefactos históricos](../_static/completion_download/completion_bundle.zip) permanecen disponibles para auditoría, sin reintroducir su catálogo en la presentación vigente.
