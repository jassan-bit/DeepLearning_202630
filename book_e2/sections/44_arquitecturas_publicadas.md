# 44. Cuerpos publicados, adaptaciones D7 y aplicación de NAS

**Enmienda metodológica del 12/11/2026:** se mantiene el [catálogo obligatorio de 16 sistemas](43_catalogo_obligatorio.md) y se adjudica la arquitectura de cada red antes de ejecutar los nuevos ajustes afectados. La ruta prospectiva conserva o restaura **nueve cuerpos definidos del Top 10**; **TiDE mantiene su variante propia y requiere NAS**. Las tres referencias E1 conservan sus cuerpos fijos y no requieren NAS estructural.

Esta decisión no modifica la campaña sellada ni la sesión privada de Colab. Sus seis familias NAS y sus resultados pertenecen al diseño anterior. La transición a las recetas de este capítulo debe conservar el historial, el tiempo consumido y los intentos sin desenlace; [§41](41_avance_optimizacion_11_origenes.md) sigue publicando un corte histórico identificado.

## Criterio usado para decidir

La guía pide identificar si cada artículo define capas y dimensiones (§3.3, p. 12), preferir el código oficial y declarar las adaptaciones (§11, p. 46). En §14, pp. 52–53, distingue un cuerpo definido de una familia sin diseño: si el artículo define la arquitectura, se utiliza esa arquitectura; NAS se aplica a los diseños que siguen sin quedar definidos. La guía no exige completar el espacio NAS de una variante propia cuando se adopta y acredita un cuerpo publicado compatible.

Para justificar una exención se conserva el mecanismo del cuerpo: bloques, anchos, atención, residuales y normalizaciones. Las interfaces de longitud, canales y salida se fijan sin consultar nuevos resultados de mercado y se documentan abajo. Una forma de salida correcta o el nombre de un artículo no acreditan esa correspondencia. **«Cuerpo publicado adaptado» tampoco significa réplica numérica de la tarea original.**

## Adjudicación por modelo

| Modelo | Cuerpo y tratamiento vigente | Adaptación al contrato D7×5 → σ₇ | NAS estructural |
|---|---|---|---|
| FCN-1D | Tres bloques convolucionales 128/256/128, kernels 8/5/3, BN/ReLU y GAP; se conserva la receta existente | Convolución temporal equivalente al eje temporal del código fuente; cabeza escalar positiva | Exento: cuerpo definido |
| ResNet-1D | Tres bloques 64/128/128 y sus atajos proyectados; se conserva la receta existente | Entrada de siete días y cinco canales; cabeza de regresión | Exento: cuerpo definido |
| Inception-1D | Red individual de seis módulos, ramas 40/20/10 y conexiones residuales; se conserva la receta existente | Padding temporal explícito y cabeza escalar; el ensemble de cinco redes es una construcción distinta | Exento: cuerpo definido |
| LSTM-autoencoder con predictor | **Restauración fija:** dos encoders LSTM de 25 y 26 unidades, sin el tercer bloque Dense de SAENAS | AE1 → AE2 con encoder 1 congelado → transferencia → ajuste supervisado; reconstrucción original `X/s`, escala ajustada sólo en entrenamiento, D7 y Softplus | Exento para la nueva receta publicada; SAENAS anterior conserva otra identidad |
| TimesNet | **Restauración fija:** dos TimesBlocks, `d_model=32`, `d_ff=32`, seis kernels y `top_k=3`; cuerpo de forecasting completo | Proyección fuente 7→8; FFT sin DC; cinco salidas del cuerpo y cabeza externa `Linear(5,1)+Softplus` | Exento para la nueva receta definida |
| TSMixer | Seis bloques con mezcla temporal, MLP 5→512→5, residuales y BN del port fuente; se conserva la receta existente | Cabeza escalar existente y sus momentos de la misma ventana, declarados como adaptación de tarea | Exento: cuerpo definido |
| TiDE | Se conserva **TIDE-D7-NAS-v1**, con sus grupos y dominio registrados | La variante vigente modifica decoder y representación de covariables; no se atribuye al cuerpo publicado sin evidencia suficiente | **Requerido** para esta variante |
| TimeMixer | **Restauración fija:** preset corto M4, cuatro PDM, `d_model=32`, `d_ff=32`, un descenso medio de factor dos y FMM completo | Escalas 7/3; se conserva la salida fuente de cinco variables y se añade una cabeza externa escalar | Exento para la nueva receta definida |
| PatchTST | **Restauración fija:** encoder oficial de tres capas, cuatro cabezas, d16, FF128 y atención residual; `Flatten_Head` oficial | D7 con patch 3, stride 2 y padding final 2: cuatro parches; cinco salidas antes de RevIN-denorm y cabeza externa escalar | Exento para el cuerpo oficial con interfaz D7 declarada |
| iTransformer | Encoder oficial de dos capas, d256, FF256 y ocho cabezas; se conserva la receta existente | Cinco tokens de variables con proyección temporal 7→256; cabeza escalar existente | Exento: cuerpo definido |

Los cuerpos fijos de **MLP16, LSTM32 y GRU32** permanecen como referencias E1 reinicializadas. Persistence, SeasonalNaive y ARIMA completan los tres comparadores; esta enmienda no añade sistemas al catálogo.

## Fuentes y decisiones de adaptación

### Convolucionales y autoencoder

FCN y ResNet se contrastan con [Wang et al., §II.B–C, pp. 2–3](https://arxiv.org/pdf/1611.06455v4) y los archivos [FCN.py](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/3ecc8971fa768bd01ada7f3a613688255e6256c2/FCN.py) y [ResNet.py](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/3ecc8971fa768bd01ada7f3a613688255e6256c2/ResNet.py). Inception se contrasta con [InceptionTime, §3.1, pp. 5–7](https://arxiv.org/pdf/1909.04939v3) y [el cuerpo individual del autor](https://github.com/hfawaz/InceptionTime/blob/952d115e83fb3a66d75c5858c702f8dd5eeb18b7/classifiers/inception.py). El ensemble de cinco miembros se describe por separado en §3.2 y no se cuenta como otra arquitectura obligatoria.

Para el autoencoder se verificaron las secciones de arquitectura/aprendizaje, pp. 7–8, y la Tabla 2, p. 11, del [artículo de Sagheer y Kotb](https://doi.org/10.1038/s41598-019-55320-6), mediante la [copia de texto completo alojada por Alaa Sagheer](https://www.academia.edu/111461009/Unsupervised_Pre_training_of_a_Deep_LSTM_based_Stacked_Autoencoder_for_Multivariate_Time_Series_Forecasting_Problems). El [código autoral fijado](https://github.com/2M-kotb/LSTM-based-Stacked-Autoencoder/blob/8668132ef6f8855dbab289c5a2b2478f871c4c4b/2layer_evaluate.py) precisa los dos encoders 25/26, reconstrucción del original y transferencia al predictor. La receta fuente informa dropout 0,3. Se declara el canal de lectura utilizado: **no se atribuye un hash a un PDF que no se descargó**.

La restauración usa `SAERecipe` y `SAEPretraining`, con decoders LSTM de cinco canales y sin tercer Dense. El preentrenamiento sigue siendo parte del algoritmo: construir dos LSTM y entrenar sólo el predictor no equivale a ejecutar la receta. La longitud 25 de la tarea fuente pasa a D7 por el contrato de E2; la reconstrucción `X/s` y Softplus son adaptaciones declaradas. La escala auxiliar se ajusta con el conjunto de entrenamiento y permanece fija para validación.

### TimesNet y TimeMixer

[TimesNet v3](https://arxiv.org/abs/2210.02186v3), §3.1–3.2, define los TimesBlocks; la Tabla 7, p. 14, y el apéndice B, figura 8, documentan dimensiones y valores de k. Se conserva [el cuerpo oficial fijado](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/models/TimesNet.py). D7 con horizonte uno produce ocho posiciones latentes y cuatro frecuencias no DC; se fija `top_k=3`, valor dentro del dominio publicado. La selección mantiene el promedio entre ejemplos del código fuente y sus pesos por ejemplo: no se sustituye por una selección individual nueva.

La exclusión de DC es explícita incluso en empates de ventanas constantes. Durante entrenamiento se conserva el promedio del batch del autor. En evaluación se invoca ese mismo cuerpo con **una ventana por llamada**, en el orden del llamador, para que una ventana posterior no determine las frecuencias de una predicción anterior. Es una política de inferencia explícita que debe reflejarse también en la medición de latencia. También se registra la diferencia de la fuente: el artículo expresa un período con `ceil`, mientras el código oficial usa `floor(T/f)`. La implementación conserva la convención del código fijado; no presenta ambas fórmulas como iguales.

[TimeMixer v1](https://arxiv.org/abs/2405.14616v1), §3 y Tabla 7, p. 14, incluye configuraciones cortas. Se adopta el [preset M4 del autor](https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/scripts/short_term_forecast/M4/TimeMixer.sh), con cuatro PDM, 32/32 y **M=1**, compatible con 7/3. Se mantienen pooling medio de factor dos, `channel_independence=1`, normalización, media móvil 25 y FMM. No es necesario inventar una profundidad de escalas para hacer caber D7.

Ambos restauran el forecasting completo y añaden sólo la cabeza externa escalar. La desnormalización interna de las cinco salidas pertenece al cuerpo que produce características para esa cabeza; la desescala de **σ₇** la realiza el llamador con la escala del objetivo ajustada en entrenamiento. No se aplica directamente la media o desviación de OHLCV a σ₇. Los módulos fuente registrados pero no utilizados por esta llamada —por ejemplo, una rama temporal sin marcas de calendario— se identifican y congelan; no se eliminan bloques para ocultar gradientes ausentes.

### TSMixer, iTransformer y PatchTST

Se mantienen los cuerpos y recetas actuales de [TSMixer, código Google fijado](https://github.com/google-research/google-research/blob/d3a4bd8f0eceddda553e47085033b537935b283f/tsmixer/tsmixer_basic/models/tsmixer.py) y [iTransformer, encoder oficial fijado](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/model/iTransformer.py). Sus cabezas escalares existentes se documentan como interfaces de tarea. Conservarlas permite someter sus resultados anteriores a una comprobación de compatibilidad; no se cambian sus pesos, recetas o entradas para atribuirles después los mismos resultados.

PatchTST restaura [el backbone oficial](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised/layers/PatchTST_backbone.py) y la configuración pequeña de encoder documentada en el artículo, apéndice A.1.4, p. 14, y [preset ETTh1](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised/scripts/PatchTST/etth1.sh). **Patch 3/stride 2 es una adaptación de la interfaz D7, no el preset ETTh1 exacto de patch 16/stride 8.** Ese preset largo no cabe nativamente en siete días.

La geometría P3/S2 ya estaba propuesta en el [expediente de lectura anterior](../_static/literature_fulltext/models.json). Produce `[0:3]`, `[2:5]`, `[4:7]` y `[6,6,6]` con padding final del último observado. No se agregan nueve días ficticios por la izquierda. RevIN se ajusta dentro de cada ventana de siete observaciones; los scores de atención residual entre capas y el `Flatten_Head` oficial se conservan. La cabeza de σ₇ recibe sus cinco salidas antes de la desnormalización de covariables.

### Por qué TiDE conserva NAS

El [código TiDE fijado](https://github.com/google-research/google-research/blob/d3a4bd8f0eceddda553e47085033b537935b283f/tide/models.py) y [sus opciones de entrenamiento](https://github.com/google-research/google-research/blob/d3a4bd8f0eceddda553e47085033b537935b283f/tide/train.py) no validan automáticamente la variante D7 vigente. Ésta modifica el decoder temporal, representa covariables futuras no disponibles mediante sentinels y utiliza un puente propio.

Se examinó la alternativa sin covariables dinámicas (`r=0`). Con entrada vacía, biases cero y ReLU de la fuente, parte del encoder temporal queda permanentemente sin gradiente: la supuesta covariable se reduce a constantes aprendidas. Omitir esa subred o cambiar su inicialización sería otra adaptación propia. Añadir calendario o lags del objetivo modificaría el contrato de entradas y exigiría una enmienda aparte. Por ello **no se activa una exención para TiDE** y se conserva `TIDE-D7-NAS-v1` con su búsqueda registrada.

## Verificación, identidades y reutilización

La comprobación de estas adaptaciones incluye correspondencia con fuentes fijadas, formas y operaciones del cuerpo, paridad de salidas y gradientes con los mismos pesos, ventanas constantes y variables, atención/mezcla activas, transferencia SAE, actualización de parámetros y ausencia de covariables futuras. La sanidad con un lote pequeño comprueba entrenabilidad; no acredita rendimiento de mercado, generalización o potencia. El [recibo de verificación](../_static/published_architectures_20261112/verification.json) declara las comprobaciones efectivamente ejecutadas y sus límites.

**Verificación local CPU: 67 comprobaciones aprobadas, sin fallos ni omisiones.** Se verificaron 76 archivos autorales fijados y tres archivos auxiliares de evidencia. El recibo identifica versiones, pruebas y hashes; no informa entrenamiento de mercado ni integración observada sobre GPU.

El nuevo adaptador de entrenamiento se habilita en este paso para **fixtures CPU explícitas**. La integración conjunta de ajuste, inferencia y orquestación GPU queda pendiente de la migración versionada del controlador. La disponibilidad de un helper SAE para CUDA no acredita por sí sola esa integración ni una corrida de mercado.

Las cuatro restauraciones tienen identidades nuevas:

| Modelo | Receta prospectiva | Reutilización de resultados de la variante NAS anterior |
|---|---|---|
| LSTM-autoencoder con predictor | `SAE-D7-PUBLISHED-v1` | No se renombran ni se aceptan como ajustes de esta receta |
| TimesNet | `TN-D7-PUBLISHED-v1` | No se renombran ni se aceptan como ajustes de esta receta |
| TimeMixer | `TM-D7-PUBLISHED-v1` | No se renombran ni se aceptan como ajustes de esta receta |
| PatchTST | `PT-D7-PUBLISHED-v1` | No se renombran ni se aceptan como ajustes de esta receta |

FCN-1D, ResNet-1D, Inception-1D, TSMixer, iTransformer y las tres referencias E1 conservan sus recetas; sólo se reutiliza un resultado cuando coincidan arquitectura, receta, datos, partición, semillas y política de ejecución. TiDE conserva su identidad NAS. Los recibos anteriores se mantienen como evidencia del diseño que realmente se ejecutó.

Se conservan las **18 oportunidades HPO**, los ejes de tasa de aprendizaje, weight decay y batch, las semillas y los límites de entrenamiento registrados. En las restauraciones, los knobs estructurales antiguos se fijan al cuerpo publicado: SAE usa dropout 0,3; TimesNet y TimeMixer, 0,1; PatchTST usa dropout de encoder y `fc_dropout` 0,3 y `head_dropout` 0. Por ello no se afirma que todos sus dominios sean idénticos a los de la variante NAS. Los cuerpos conservados mantienen sus ejes existentes. La adjudicación de cuerpos conservó inicialmente los once orígenes. La [enmienda temporal posterior de §45](45_enmienda_protocolo.md) registra tres anclas HPO y una sola NAS inicial TiDE, con reajustes mensuales; conserva límites de épocas, semillas, gasto y techo autorizado. k y R siguen sujetos a sus pilotos y ninguna cota de corridas se presenta como un mínimo literal de la guía.

## Registro y aplicación

- [Política de arquitectura/NAS vigente](../_static/published_architectures_20261112/architecture_policy.json).
- [Catálogo con las identidades vigentes](../_static/published_architectures_20261112/catalogue.json).
- [Matriz descargable de arquitectura y adaptación](../_static/published_architectures_20261112/architecture_matrix.csv).
- [Fuentes auxiliares fijadas y sus hashes](../_static/published_architectures_20261112/supporting_sources.json): preset M4, defaults TimeMixer y código autoral SAE.
- [Registro versionado en el repositorio](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e2_published_architectures_20261112/architecture_policy.json).
- Implementaciones: [registro](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_published_registry.py), [TimesNet/TimeMixer](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_published_periodic.py), [PatchTST](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_published_patchtst.py), [SAE](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_published_sae.py) y [entrenamiento](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_published_training.py).

La política prospectiva deja NAS para **una variante del Top 10: TiDE**. Las exenciones se justifican para las recetas concretas descritas; añadir bloques propios, alterar sus mecanismos o usar otra receta requiere otra adjudicación. Esto no equivale a haber terminado esa búsqueda, el HPO o la evaluación final. Antes de reanudar con estas recetas, el controlador debe incorporar esta política y el [calendario de §45](45_enmienda_protocolo.md), y conciliar caché y saldo. Publicar el libro o importar un constructor no migra la campaña activa de Colab.
