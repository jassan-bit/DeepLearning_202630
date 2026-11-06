# 14. Arquitecturas fuente, adaptación D7 y búsqueda condicional

**Estado:** diez mecanismos fuente identificados; las diez revisiones de arquitectura final del Top 10 siguen pendientes.
P0 está ejecutando tres recetas pequeñas y la referencia GRU fijadas previamente; no ofrece resultados completos ni una exención para el benchmark final.
No se ha entrenado ninguna de las 1.000 arquitecturas de una búsqueda NAS.

La guía completa de 65 páginas ya está recibida. Sus páginas **52–58** definen la condición y el procedimiento de búsqueda;
la lista de verificación de la página **63** exige documentar los casos afectados y las dos fases.
El registro de esta etapa es el commit público [`599a29f`](https://github.com/jassan-bit/DeepLearning_202630/commit/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0).

## 14.1 Identificar una arquitectura no equivale a aprobar cualquier adaptación

La [matriz registrada](../_static/remediation_protocol/architecture_decisions.json) separa la estructura publicada,
su configuración numérica, la receta P0 y la adaptación final a **D7×5 → σ₇**.
`source_architecture_defined=true` indica que se recuperó una descripción estructural; no acredita que el adaptador final esté validado.
En los diez casos, `final_architecture_exception_verified=false` mantiene bloqueadas las fases posteriores correspondientes.

La decisión actual `nas_required=false` describe los backbones fuente identificados, no una exención general concedida por tener DOI o código.
Un hiperparámetro no reportado tampoco demuestra que el artículo sólo nombre una familia: debe resolverse la receta con evidencia identificable.
Si el diseño realmente queda sin definir, se activa la búsqueda condicional; si existe un diseño, se conserva y se justifican sus modificaciones necesarias.

Las páginas de la tabla siguiente son **páginas físicas del artículo original**, distintas de las páginas de la guía.
Los commits y los hashes completos de código/PDF están en la matriz descargable; las abreviaturas sólo facilitan la lectura.

| Modelo | Configuración o información numérica fuente | Código fijado | Revisión necesaria para el benchmark final |
|---|---|---|---|
| **TSMixer** | `2303.06053v5`, Tabla 8, pp. 21–23: ETTh1/h96, seis bloques, hidden 512, ReLU y pre-BatchNorm; dropout 0,9. La tabla presenta hidden 512 fuera del rango ETT 8–64 impreso; se conserva la inconsistencia. | `google-research/google-research@d3a4bd8`, `tsmixer/tsmixer_basic/models/tsmixer.py`, `res_block`/`build_model` | Conservar mezcla temporal y de atributos, residuales y normalización. Revisar contexto siete y lectura σ₇. No es uno de los tres Top 10 de P0. |
| **TiDE** | `2304.08424v5`, Tablas 7–8, pp. 20–21: hidden 256, encoder 2/decoder 2, decoderOutputDim=8, temporalDecoderHidden=128, temporalWidth=4, LayerNorm y RevIN activas. | `google-research/google-research@d3a4bd8`, `tide/models.py`, `MLPResidual`/`TideModel` | Mantener proyección de covariables, encoder, decoder y decoder temporal. La ausencia de historia de σ₇ y de covariables futuras exige declarar las ramas modificadas; no presentarlo como réplica literal. |
| **TimeMixer** | `2405.14616v1`, Tabla 7, p. 14, más preset original ETTh1 unificado: d=16, FFN 32, dos PDM, tres descensos por factor 2, media móvil 25, channel independence y normalización activas. FFN 32 procede del código, no de la tabla. | `kwuking/TimeMixer@e246105`, `models/TimeMixer.py`; preset `TimeMixer_ETTh1_unify.sh` | D7 no admite tres descensos no vacíos. P0 usa uno, con longitudes 7/3, preservando mezcla estacional/tendencia y suma FMM. Esa modificación no se acepta automáticamente para la receta final. |
| **TimesNet** | `2210.02186v3`, Tabla 7, p. 14: dos bloques, regla de ancho con mínimo 32 —32 para cinco canales— y top-k=5. FFN y número de kernels requieren completar el preset/versionado pertinente. | `thuml/Time-Series-Library@4e938a1`, `models/TimesNet.py`, `FFT_for_Period`/`TimesBlock` | El preset TSL posterior d=16/FFN 32 no se atribuye al mínimo 32 del PDF. Resolver parámetros y frecuencias admisibles con D7; preservar Inception 2D y selección de periodos por origen. No está en P0. |
| **PatchTST** | `2211.14730v2`, p. 14, A.1.4–A.1.5: configuración pequeña ETTh1 con tres bloques, d=16, cuatro cabezas, FFN 128 y GELU. Patches publicados de longitud=16/stride8 y dropout del encoder 0,2. | `yuqinie98/PatchTST@204c21e`, `PatchTST_supervised/models/PatchTST.py` | Sus patches no caben directamente en D7. P0 usa dos bloques, FFN 32, patch 2/stride 1; no es la configuración publicada ni decide la configuración final. |
| **Informer** | `2012.07436v3`, pp. 5/12/13, Tabla 7: d=512, FFN 2048, dieciséis cabezas de encoder y ocho de atención cruzada de decoder; stacks 3+1, decoder 2, factor 5 y dropout 0,1. La configuración ganadora exacta por fila no está reportada. | `zhouhaoyi/Informer2020@29f2a73`, `models/model.py`, variantes `Informer` y stack | Resolver variante/preset y conservar ProbSparse, destilación y decoder. Revisar longitudes y muestreo válidos para D7; el futuro usa placeholders, no OHLCV observadas después del origen. No está en P0. |
| **Autoformer** | `2106.13008v5`, pp. 7/19: d=512, encoder 2/decoder 1, ocho cabezas y media móvil 25. El texto presenta factor en 1–3; FFN y factor exacto requieren preset original identificable. Dropout numérico: NR en el PDF. | `thuml/Autoformer@51c7d41`, `models/Autoformer.py`, `series_decomp` | Conservar descomposición, autocorrelación y suma de ramas. La tendencia de precio no se reutiliza como tendencia de σ₇: la proyección de tarea debe estar declarada y revisada. No está en P0. |
| **iTransformer** | `2310.06625v4`, A.2, p. 14: búsqueda de dos/tres/cuatro bloques y d=256/512. La configuración ETTh1 seleccionada y sus cabezas/FFN no quedan fijadas por esa tabla. | `thuml/iTransformer@c2426e6`, `model/iTransformer.py`, embedding invertido y encoder | Recuperar un preset oficial exacto. P0 d=16/FFN 32/dos bloques no se convierte en el ganador original. Conservar tokens de las cinco variables y revisar lectura latente para σ₇. |
| **Crossformer** | PDF ICLR 2023, A.3, p. 14: ETTh1 d=256, tres niveles, cuatro cabezas y diez routers; segmentos entre 6/12/24, ganador exacto pendiente. Constructor fijado: merge=4 y FFN 1024, separados de la evidencia del PDF. | `Thinklab-SJTU/Crossformer@c10c8ea`, `cross_models/cross_former.py` | Resolver segmento y adaptación D7. Preservar DSW, TSA y HED completo: decoder futuro y combinación de escalas, no sólo pooling del encoder. No está en P0. |
| **S-Mamba** | `2403.11144v3`, Algoritmos 1–2, pp. 4–5: mecanismos definidos, pero receta numérica final por dataset **NR**. El código fija d_conv=2 y expand=1; faltan preset de capas, d, FFN y d_state. | `wzhwzhwzh0921/S-D-Mamba@e7e8bf0`, `model/S_Mamba.py` | No inventar d_state=8 o d=16 como parámetros del artículo. Verificar preset y runtime, manteniendo Mamba bidireccional sobre variables y FFN. NR bloquea el cierre; no activa NAS por sí solo. No está en P0. |

Las bases GRU-E2/LSTM-E2/TCN-E2 tienen estructuras locales heredadas explícitas: una capa recurrente de ancho 16 para GRU/LSTM y tres bloques TCN de ancho 8,
kernel 3 y dilataciones 1/2/4. Su cabeza es escalar Softplus. El ajuste E2 no debe cambiar esa estructura silenciosamente.
Ridge y ARIMA son estimadores especificados, sin una arquitectura neuronal oculta que requiera inventar un NAS.
Esta procedencia local se distingue de afirmar que sus dimensiones son una configuración canónica de un artículo.

## 14.2 Recetas P0 y modificaciones de tarea

P0 se ejecutó con TimeMixer, iTransformer, PatchTST y la GRU/D7-c03 de referencia, con cinco semillas nuevas.
Las tres recetas pequeñas del Top 10 se fijaron antes del entrenamiento y están separadas en `P0_fixed_recipe`.
El detalle de registro, calendario y alcance está en [§9](09_protocolo_registrado.md); este capítulo no presenta sus resultados.

| Receta P0 | Parámetros estructurales fijados | Diferencia que debe conservarse en el reporte |
|---|---|---|
| TimeMixer | d=16, FFN 32, dos PDM; channel independence=1; average pooling por 2 con un descenso 7→3; moving_avg=25 | Cambio de pirámide frente a los tres descensos del preset fuente; suma de contribuciones FMM antes de proyección/desnormalización OHLCV |
| iTransformer | d=16, FFN 32, dos bloques, cuatro cabezas y cinco tokens de variables | Receta de factibilidad distinta del espacio d=256/512 documentado en el original |
| PatchTST | d=16, FFN 32, dos bloques, cuatro cabezas, patch 2/stride 1 y siete patches con padding final | Cambios explícitos frente a tres bloques/FFN 128 y patch 16/stride 8 de la configuración pequeña publicada |

Los tres adaptadores P0 producen latentes `[B,5,16]`. Su cabeza concatena los 80 latentes con cinco medias y cinco desviaciones poblacionales
calculadas desde la **misma entrada D7 escalada con train**, y aplica `Linear(90,1)` más Softplus; la inversión de escala de Y pertenece al pipeline.
Esos diez resúmenes no añaden información externa. La cabeza de 90 entradas es propia de estas recetas P0, no una dimensión impuesta a backbones finales d=256/512.

La adaptación final debe preservar el objetivo y la información disponible: siete barras OHLCV, ninguna historia de σ₇ como canal nuevo y ningún valor futuro en decoders.
Una proyección que originalmente predice OHLCV no se interpreta como volatilidad ni se desnormaliza con media/DE de Close como si fuera σ₇.
Debe registrarse el punto de lectura latente y cualquier cambio de normalización, marcas temporales, parches, escalas o proyecciones.

TiDE debe conservar y describir sus bloques centrales, declarando la sustitución de ramas que requieran historia del objetivo o covariables futuras inexistentes.
Crossformer debe conservar su decoder HED y fusión multiescala. TimesNet debe evitar selección de periodos compartida entre orígenes de inferencia;
batch 1 o una selección independiente requieren pruebas del adaptador. La factibilidad de forma no sustituye esas comprobaciones.

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

## 14.4 Evidencia y controles antes de continuar

La revisión final debe unir fuente, configuración numérica, adaptación, implementación y pruebas sobre el contrato D7.
En el estado registrado, **HPO, P1, ajustes definitivos y scoring están bloqueados por las revisiones finales pendientes del Top 10**.
Si algún caso activa NAS, P1/ajustes definitivos/scoring exigirán además los recibos de fase 2 completa y del ganador congelado.
Un flag de arquitectura conocida, un forward sintético o un archivo con hash no equivalen a validar todo el pipeline.

| Evidencia de una búsqueda que llegue a ejecutarse | Contenido que debe entregarse |
|---|---|
| Espacio y cómputo | Los 1.000 identificadores/versiones, configuración, parámetros/FLOPs, presupuesto estimado y consumo real |
| Dos fases | Todas las métricas del cribado, semilla/presupuesto, finalistas, control y resultados completos por semilla de confirmación |
| Fiabilidad de selección | Ruido piloto, regla k, correlación de rangos entre fases, regla de desempate y posición del ganador en el borde del espacio |
| Análisis agregado | Superficies, efectos principales/interacciones con IC, varianza explicada y multiplicidad por factores; no 499.500 pruebas entre pares |
| Calidad y límites | Pareto de error/coste, brechas de generalización, conjunto de finalistas indistinguibles con procedimiento justificado y desviaciones |

No se infiere equivalencia de finalistas a partir de no rechazar diferencias; TOST/ROPE requieren margen y supuestos, y pertenecer a MCS no prueba equivalencia.
Un óptimo en el borde se declara como limitación; no se amplía W tras consultar test para buscar un resultado favorable.
Estos análisis pendientes deberán aportar sus datos y recibos; enumerarlos aquí no constituye un resultado de búsqueda.

Las fuentes vigentes son la [matriz arquitectónica](../_static/remediation_protocol/architecture_decisions.json),
el [preregistro](../_static/remediation_protocol/preregistration.json) y el [diseño HPO](../_static/remediation_protocol/hpo.json).
La [implementación prevista](04_implementacion.md) y el [registro del piloto](09_protocolo_registrado.md) conservan la separación entre P0 y el benchmark final.
