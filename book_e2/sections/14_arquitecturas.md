(arquitecturas-ejecutadas-y-condicion-de-nas)=
# 14. Arquitecturas definidas: búsqueda NAS no aplicable

**Aplicabilidad vigente: NO APLICA.** La búsqueda de **1.000 arquitecturas y el protocolo NAS en dos fases** de la sección 14 de la guía corresponden a modelos sin diseño estructural definido. El catálogo actual tiene **trece arquitecturas neuronales definidas**: los diez candidatos de E2 y las tres referencias MLP16, LSTM32 y GRU32 de E1. Incluye **TiDE TIDE-D7-PUBLISHED-R0-v2**, con su grafo publicado y sus adaptaciones registradas. **Esta búsqueda NAS no es un requisito pendiente del benchmark actual.**

La adjudicación se sustenta en el estado vigente y la tabla de cuerpos de [§44](44_arquitecturas_publicadas.md), en el [protocolo ejecutado de §48](48_protocolo_reducido.md) y en la [extensión de TiDE anterior al ajuste](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/experiments/e2_tide_published_r0_20261112/protocol_extension.json). La [lista verificable](../_static/e2_agregacion_entrega_current_v1/guide_checklist.csv), el [índice de evidencia](../_static/e2_agregacion_entrega_current_v1/current_evidence_index.json), el {download}`índice de esta actualización <../_static/e2_agregacion_entrega_current_v1/README.md>` y el [manifiesto](../_static/e2_agregacion_entrega_current_v1/manifest.json) permiten revisar esta decisión junto con la entrega consolidada.

(e2-nas-no-aplica-catalogo-actual)=
## 14.1 Catálogo actual y decisión de aplicabilidad

| Cuerpos definidos del catálogo | Modelos | Aplicación de NAS de 1.000 diseños y dos fases |
|---|---|---|
| Convolucionales y bloques temporales convolucionales | FCN-1D, ResNet-1D, Inception-1D y TimesNet | **No aplica** a las recetas definidas actuales. |
| Representación recurrente autoencoder | LSTM-autoencoder con predictor (SAE) | **No aplica** a la receta publicada actual. |
| Mezcla MLP y encoder-decoder denso | TSMixer, TiDE R0-v2 y TimeMixer | **No aplica** a las recetas definidas actuales. |
| Atención Transformer | PatchTST e iTransformer | **No aplica** a las recetas definidas actuales. |
| Referencias neuronales de E1 | MLP16, LSTM32 y GRU32 | **No aplica** a sus cuerpos fijos. |

La entrada **D7×5 OHLCV** y la salida escalar **σ₇ con horizonte h=1** están especificadas para estas recetas. Los bloques, dimensiones, mecanismos y adaptaciones de interfaz se documentan en §44; la disponibilidad causal de las ventanas se fija en [§17](17_contrato_sincronizado.md). Tener una adaptación a la tarea no convierte por sí solo el cuerpo en una arquitectura indefinida.

## 14.2 Fuentes, ports y adaptaciones que siguen aplicando

La identificación del artículo y del código de referencia, los commits y hashes de las fuentes, los ports y las adaptaciones D7 siguen formando parte de la evidencia de implementación. La decisión **«NAS no aplica»** se refiere a la búsqueda estructural condicional; las recetas empleadas mantienen su trazabilidad en [§44](44_arquitecturas_publicadas.md) y su entrenamiento en [§11](11_hpo_equidad.md).

**TiDE R0-v2** conserva el grafo oficial completo y declara la rama r=0 inactiva. La extensión fija sus dimensiones y adaptaciones antes del ajuste; su decisión indica que no quedan elecciones estructurales sin especificar para esa receta fija. La [comprobación de paridad funcional](https://github.com/jassan-bit/DeepLearning_202630/blob/1e8745497b484518f766b6da5c19be8362d9add7/experiments/e2_tide_published_r0_20261112/source_parity.json) y el cierre de §48 sustentan esta identidad actual.

Las comprobaciones de formas, gradientes, mecanismos y paridad verifican aspectos de implementación. Los resultados predictivos y sus seis semillas por red se reportan en [§6](06_benchmark_final.md), y la reproducción independiente de los ajustes, en [§55](55_reproduccion_ablaciones.md). La definición de un cuerpo y su paridad funcional tienen ese alcance propio.

## 14.3 HPO ejecutado y condición futura de NAS

Las **seis configuraciones HPO por red y los tres folds internos** seleccionan hiperparámetros de entrenamiento de cuerpos ya definidos. Se registran en [§48](48_protocolo_reducido.md) y se describen en [§11](11_hpo_equidad.md). **Seis plazas HPO no equivalen a una búsqueda NAS de 1.000 arquitecturas.** La cobertura y equidad del dominio HPO se evalúan como tales en el capítulo 11; la NAS que no aplica no se contabiliza como una búsqueda incompleta del catálogo actual.

Si en un trabajo futuro se introduce un cuerpo cuya estructura siga sin quedar definida, se deberá evaluar para ese nuevo caso la ruta NAS de 1.000 diseños y dos fases prevista en la guía. Esa condición futura no es una obligación actual de estas trece recetas.

## 14.4 Parámetros y operaciones de las recetas seleccionadas

La [auditoría de §58](58_flops_predictores.md) conserva las trece recetas ganadoras exactas, sus parámetros y la concordancia con los recibos. El conteo de matrices y convoluciones usa **2 FLOPs/MAC** y declara las exclusiones de FFT, activaciones, normalización y otros operadores. Las tres sondas sintéticas tienen el alcance documentado para TimesNet. Estas medidas caracterizan los cuerpos seleccionados; no constituyen una búsqueda de arquitectura.

La {download}`versión exacta anterior a esta actualización <../_static/e2_agregacion_entrega_current_v1/14_arquitecturas_before_update.md>` y su manifiesto conservan el corte precedente. El texto anterior se mantiene también a continuación, con sus encabezados, fechas y descargas originales.

````{admonition} Archivo documental: contenido anterior del capítulo 14
:class: dropdown

Este bloque conserva el contenido anterior. La decisión vigente es **NAS no aplicable al catálogo actual**, según las secciones precedentes.

# 14. Búsqueda de Arquitectura cuando No Hay Diseño Predefinido

**Estado vigente del 14/11/2026:** el catálogo obligatorio ejecutado contiene diez cuerpos de literatura/adaptaciones y tres referencias neuronales de E1. Las identidades, fuentes primarias y adaptaciones D7 de los diez candidatos se adjudican en [§44](44_arquitecturas_publicadas.md) y la actualización de [§48](48_protocolo_reducido.md), incluida **TIDE-D7-PUBLISHED-R0-v2**. El benchmark final está cerrado.

## 14.1 Cuerpos y familias

| Familia funcional | Candidatos |
|---|---|
| Convolucional | FCN-1D, ResNet-1D, Inception-1D y TimesNet |
| Recurrente con representación autoencoder | LSTM-autoencoder con predictor |
| Mezcla MLP / encoder-decoder denso | TSMixer, TiDE y TimeMixer |
| Atención Transformer | PatchTST e iTransformer |

Los tres controles MLP16, LSTM32 y GRU32 se cuentan adicionalmente. D7×5 describe la entrada de los modelos actuales y h=1 la lectura escalar σ₇. Padding y adaptaciones de cabeza no añaden historia de mercado. La disponibilidad causal de las ventanas se fija en [§17](17_contrato_sincronizado.md).

## 14.2 Adjudicación y pruebas

Los archivos de autores se fijaron por commit y hash. La restauración de SAE, TimesNet, TimeMixer y PatchTST conserva identidades propias; no se renombran resultados de prototipos anteriores como resultados de esos cuerpos. TiDE r=0 usa una receta publicada y una comprobación de paridad funcional propia, separada de su antigua variante NAS.

La comprobación de formas, gradientes, mecanismos y paridad es evidencia de implementación, no de superioridad predictiva. Los resultados de mercado y sus seis semillas por red se reportan en [§6](06_benchmark_final.md); la repetición independiente de sus ajustes se reporta en [§55](55_reproduccion_ablaciones.md).

## 14.3 Aplicabilidad de la búsqueda de 1.000 arquitecturas

La Sección 14 de la guía aplica el espacio de 1.000 diseños y el protocolo en dos fases a **modelos sin arquitectura predefinida**. La exención de los cuerpos adjudicados no surge únicamente de citar un artículo: requiere que la receta y sus adaptaciones estén suficientemente definidas y verificadas, como se documenta en §§44/48.

Las seis configuraciones HPO por red no se presentan como NAS de 1.000 arquitecturas. Las propuestas anteriores de cinco cuerpos y seis familias NAS conservan su identidad histórica en el archivo. No se elimina una obligación NAS mediante una reducción tácita de candidatos; si se introdujera un cuerpo nuevo indefinido, necesitaría su protocolo y evidencia correspondientes.

## 14.4 Parámetros y operaciones de los cuerpos seleccionados

[§58](58_flops_predictores.md) audita las trece recetas ganadoras exactas y la concordancia de sus parámetros con los recibos. El conteo de matrices y convoluciones usa 2 FLOPs/MAC, con FFT, activaciones, normalización y otros operadores excluidos e inventariados. Las tres sondas sintéticas no equivalen a inferencia de mercado ni acotan todas las rutas dependientes de valores de TimesNet.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/14_arquitecturas.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).

````
