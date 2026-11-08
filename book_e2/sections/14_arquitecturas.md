# 14. Arquitecturas fuente, adaptación D7 y búsqueda condicional

**Propuesta posterior del 08/11/2026:** véase la [revisión de diez modelos y un híbrido candidato](15_revision_modelos.md). El catálogo y las cifras de esta sección describen el diseño registrado anterior; no son evidencia de ejecución de la nueva propuesta.

**Estado:** los diez adaptadores finales están implementados y cuentan con [revisión independiente por modelo](../_static/completion_protocol/architecture_review.json), pruebas de gradientes/invariancia y [sanidad de memorización sintética](../_static/completion_preflight/tiny_batch_sanity.json). La enmienda se registró en [`dbb193a`](https://github.com/jassan-bit/DeepLearning_202630/commit/dbb193a72897c80cbd34b452ba3cfe9d82ce6ce0). El [recibo del 06/11/2026 a las 19:00 UTC](../_static/completion_preflight/execution_progress.json) acredita doce trayectorias y veinticuatro ajustes cerrados de TSMixer, con HPO incompleto. No confirma ejecución activa hoy ni resultados finales Bitcoin. P0/OOF y su registro `599a29f` conservan sus recetas y resultados históricos.

La guía, PDF pp52–58, exige búsqueda de 1.000 arquitecturas **cuando la estructura no esté definida**. La revisión resuelve fuentes, presets y modificaciones de los diez casos; `nas_required=false` está justificado individualmente. No significa que NAS se haya ejecutado ni exime de HPO, auditoría real o comparación justa.

## 14.1 Fuentes y adaptaciones finales verificadas

Se usa el artículo para los parámetros que declara; los no reportados se completan desde un preset/código oficial fijado, identificando sus condiciones. Mezclar campos explícitos de artículo y default de código se declara como una receta adaptada, **no como réplica exacta de una fila ganadora**. Los hashes completos, pruebas e identidad de archivos revisados están en la matriz descargable. La matriz antigua `architecture_decisions.json` conserva su estado previo P0 y no se edita para aparentar que estas revisiones existían entonces.

| Modelo | Estructura final y fuente | Adaptación D7 declarada |
|---|---|---|
| **TSMixer** | Artículo Tabla8 ETTh1/h96; Google `d3a4bd8`: seis bloques, hidden512, ReLU, pre-BatchNorm en ejes tiempo/canal, Glorot | Port TensorFlow→PyTorch de los bloques; cabeza de tarea escalar Softplus. Se conserva la inconsistencia impresa de hidden512 frente al rango ETT8–64. c12 dropout0,9 no se atribuye al espacio aleatorio0–0,3. |
| **TiDE** | Artículo Tabla8 + Google `d3a4bd8`: hidden256, flag num_layers2, decoderOutput8, temporalHidden128, width4, LayerNorm/RevIN. En el código, flag2 corresponde a **un bloque residual de dos densas**, no dos bloques | Cinco identidades OHLCV y mismas covariables pasadas D7; futuras fijadas a cero. Decoder temporal produce latente8 con LayerNorm8 para evitar colapso del escalar LayerNorm1; skip7→1 difundido al latente. Cabeza escalar, sin desnormalizar OHLCV como σ₇. |
| **TimeMixer** | Autor `e246105`, preset ETTh1 unificado: d16, FF32, dos PDM, media25 y channel independence | Tres descensos darían longitud cero. Se usa máxima profundidad válida **dos descensos, 7→3→1**, preservando PDM y FMM por escala; P0 tenía un descenso y sigue diferenciado. |
| **TimesNet** | Artículo Tabla7, C5 implica d32; TSL `4e938a1` aporta FF32 y seis kernels, dos bloques | Extensión temporal aprendida7→8 antes de FFT; top-k5 se limita a las **cuatro frecuencias no DC** disponibles, con pesos propios por origen. Inception2D conserva las seis parametrizaciones mediante fusión de kernels numéricamente equivalente y comprobación de gradientes. |
| **PatchTST** | Artículo A.1.4–A.1.5 y autor `204c21e`: tres bloques, d16, FF128, cuatro cabezas, patch16/stride8 | Normalizar sólo D7 observado, replicar borde izquierdo hasta16 y conservar padding final8: dos patches. La réplica de bordes no añade historia ni observaciones futuras. Encoder de atención residual y lectura escalar propios de esta tarea. |
| **Informer** | Artículo Tabla7 + autor `29f2a73`: d512, FF2048, encoder stacks3+1, cabezas16; decoder2 y atención cruzada8; factor5 | Mantiene stacks/destilación/decoder; seq7/label3/pred1, sin calendario futuro. La rama corta tiene un token; se define ese límite de ProbSparse y su régimen de consulta completa. Las cinco salidas latentes del decoder son características de tarea, no pronósticos OHLCV reutilizados como volatilidad. |
| **Autoformer** | Autor `51c7d41`, script/defaults originales: d512, FF2048, encoder2/decoder1, ocho cabezas, media25 y factor3 | Mantiene descomposición, autocorrelación, decoder y ramas aditivas; padding de borde para media25. Sin embeddings de calendario ni observaciones futuras; lectura escalar explícita. |
| **iTransformer** | Autor `c2426e6`, preset ETTh1_96_96: d256, FF256, dos bloques y ocho cabezas por default identificado | Cinco tokens de variables con siete datos observados; conserva encoder invertido y normalización. Sustituye proyector por variable/desnormalización OHLCV por cabeza σ₇. P0 d16 no se presenta como esta receta. |
| **Crossformer** | Autor `c10c8ea`, script ETTh1/CLI: seg6, d256, FF512, tres niveles, cuatro cabezas, diez routers y merge2 | Repetición causal izquierda D7→12, DSW/TSA y **cuatro etapas de decoder HED** con combinación aditiva. FF1024/merge4 del constructor son otra condición y no se atribuyen a este preset. |
| **S-Mamba** | Autor `e7e8bf0`, preset ETTh1H96: d256, FF256, dos bloques, state2, conv2, expand1 | Mantiene escaneo selectivo bidireccional de variables y FFN. Referencia CPU explícita contrastada en valores/gradientes con Mamba1.2.0; no se exige un kernel CUDA inexistente ni se sustituye por una RNN genérica. |

Las bases retunadas tienen una capa GRU/LSTM de ancho16 o tres bloques TCN de ancho8, kernel3/dilataciones1/2/4. Se distinguen de las tres recetas E1 fijas. Ridge y ARIMA son estimadores definidos; no requieren inventar una búsqueda de arquitectura neuronal.

Los backbones y las cabezas se prueban con salida positiva `[B]`, gradientes finitos de sus parámetros entrenables y predicción en evaluación independiente de acompañantes del batch. La prueba de memorización usa cuatro entradas sintéticas y un umbral fijado; pasó en los diez casos. **Es sanidad de software, no rendimiento ni potencia Bitcoin**, y no sustituye los controles de etiquetas permutadas u otras verificaciones del pipeline real.

## 14.2 Diferencia respecto al piloto y conservación del contrato

P0 usó TimeMixer d16/FF32 con escalas7/3, iTransformer d16/FF32/dos bloques y PatchTST d16/FF32/dos bloques/patch2. Las tres cabezas concatenaban 80 latentes y diez momentos de la misma entrada D7, `Linear(90,1)` y Softplus. Estas recetas se conservan con sus hashes; las nuevas no se seleccionaron por sus resultados P0 ni se renombran retroactivamente.

Cada adaptador final registra su lectura latente y cambios de normalización, escalas, parches o proyección. Conserva **D7×5 OHLCV → σ₇**, sin historia de σ₇ como canal nuevo, barras futuras ni embeddings con información posterior al origen. Las estadísticas de la misma entrada no son observaciones externas. Una salida OHLCV no se desnormaliza con Close para interpretarla como σ₇.

## 14.3 NAS sólo cuando corresponda

Si una receta queda definida únicamente por familia o incorpora una estructura sin especificación verificable, debe resolverse el caso antes de continuar.
La guía no pide alterar indiscriminadamente arquitecturas publicadas. Para un caso realmente sujeto a NAS, el espacio registrado es:

$$
W=\{16,32,64,128,256\},\qquad
\mathcal A=W^3\times\{0,1\}^3,\qquad |\mathcal A|=125\times8=1000.
$$

Cada patrón indica dropout de tasa 0,2 inmediatamente después de cada uno de los tres bloques; incluye ausencia total de dropout.
El significado del ancho depende de la familia: neuronas densas, unidades recurrentes, canales convolucionales o dimensión de bloques de atención.
La traducción del espacio y cualquier correspondencia aproximada deben declararse conservando las 1.000 combinaciones, no recortarse por conveniencia.

1. **Control y ajuste previo.** Usar el control 64–64–64 con patrón 010 para fijar mediante desarrollo LR, optimizador, scheduler, weight decay, batch,
   pérdida, activación y parada. Estos elementos permanecen iguales durante la búsqueda; sólo cambian anchos y bits de dropout.
2. **Piloto de ruido.** Repetir el control y las cuatro arquitecturas de contraste registradas con semillas 6301–6305, antes del cribado.
   Se estima la mayor DE dentro de arquitectura frente a la DE entre medias de arquitecturas: k=20 si la razón≤1, k=40 si es mayor o no definida.
   Entre modelos afectados se conserva el mayor k requerido por esa regla; no se escoge después de ver qué finalista interesa.
3. **Fase 1 exhaustiva.** Evaluar los 1.000 identificadores con semilla 6401 y el 25 % de las épocas máximas, con la misma regla de early stopping.
   No hay poda previa. El cribado es ruidoso y no demuestra que su ganador sea el mejor verdadero del espacio.
4. **Fase 2.** Entrenar los k finalistas y el control con presupuesto completo y semillas 6501–6505, únicas y distintas de fase 1 y del piloto de ruido.
   Registrar métrica de validación, brecha train–validación, parámetros y tiempo por ejecución; mantener el mismo k y R entre modelos afectados.
5. **Selección.** Minimizar la RMSE media de validación. Ante diferencia menor que δ, aplicar parsimonia por parámetros y después identificador lexicográfico.
   La selección no usa test. El ganador es el mejor entre los finalistas confirmados, sin atribuirle optimalidad garantizada sobre los 1.000.

Los presupuestos y semillas son decisiones registradas, **no evidencia de que esas fases se hayan ejecutado**.
El perfilado previo debe estimar el coste, para m modelos afectados y R=5 en fase 2:

$$
C_{\mathrm{total}}=m\left[1000\,\bar t_1+(k+1)\,5\,\bar t_2\right].
$$

Se publicará también el consumo real. Un límite insuficiente se registra como desviación y búsqueda incompleta, sin recortar el espacio silenciosamente.
La asignación HPO de [§11](11_hpo_equidad.md) no prueba que este coste adicional esté cubierto o que se hayan completado las 1.000 evaluaciones.

## 14.4 Cierre operativo y evidencia pendiente

La revisión arquitectónica por modelo y las pruebas sintéticas ya existen. La [enmienda](../_static/completion_protocol/analysis_additions.json), el [diseño](../_static/completion_protocol/design.json) y el [HPO](../_static/completion_protocol/hpo.json) se vincularon al [registro público](../_static/completion_preflight/registration.json) y al [preflight](../_static/completion_preflight/completion.json) antes del inicio documentado de HPO. Cualquier cambio experimental posterior requiere un registro nuevo; la propuesta de §15 todavía no lo constituye. El runner vincula fuente propia, dependencias de autores, configuración, datos de desarrollo y revisión mediante hashes; rechaza cambios respecto a su registro. Eso no convierte la existencia de un archivo en certificación de validez científica.

Si la verificación integrada rechaza un caso y éste queda sin estructura definida, se registra la corrección o se activa NAS antes de seguir; no se concede una excepción por conveniencia. En tal caso se entregarían los 1.000 identificadores, presupuesto estimado/real, cribado, piloto de ruido, k finalistas y control, cinco semillas de confirmación y selección congelada. No hay resultados de esas fases que informar ahora.

La [descarga de implementación](../_static/completion_download/completion_bundle.zip) permite revisar adaptadores, pruebas y procedencia. HPO completo, potencia P1, robustez, ablación y métricas finales se acreditarán únicamente mediante sus ejecuciones y recibos. No se infiere equivalencia de finalistas por no rechazar diferencias ni se amplía una estructura después de consultar test.
