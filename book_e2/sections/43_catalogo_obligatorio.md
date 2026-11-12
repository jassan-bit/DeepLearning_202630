# 43. Catálogo obligatorio: dieciséis sistemas

**Decisión de alcance del 12/11/2026, autorizada por el usuario:** completar **diez modelos del Top 10, tres referencias de E1 y tres comparadores temporales**. Son **13 redes y 3 comparadores: 16 sistemas**. El híbrido adicional y Ridge quedan fuera del alcance obligatorio; sus implementaciones, configuraciones y resultados previos se conservan como antecedentes opcionales.

La guía exige Top 10 con **al menos cuatro familias** y tres referencias de E1 (§3.2, p. 11). La ruta F2a exige naive, naive estacional y **ARIMA o ETS** (p. 30). Se conserva ARIMA como el tercer comparador; implementar también ETS no es necesario para cubrir esa alternativa. El catálogo satisface el recuento y la diversidad funcional documentada; completar sus resultados y los demás ítems de la guía sigue siendo obligatorio.

## Los sistemas que se deben entregar

| N.º | Identidad del sistema | Papel | Familia funcional |
|---|---|---|---|
| 1 | FCN-1D | Top 10 | Convolucional |
| 2 | ResNet-1D | Top 10 | Convolucional |
| 3 | Inception-1D | Top 10 | Convolucional |
| 4 | LSTM-autoencoder con predictor | Top 10 | Recurrente / autoencoder |
| 5 | TimesNet | Top 10 | Convolucional con representación de periodicidad |
| 6 | TSMixer | Top 10 | Mezcla mediante MLP |
| 7 | TiDE | Top 10 | Encoder-decoder denso / MLP |
| 8 | TimeMixer | Top 10 | Mezcla mediante MLP y descomposición multiescala |
| 9 | PatchTST | Top 10 | Atención Transformer |
| 10 | iTransformer | Top 10 | Atención Transformer |
| 11 | MLP16 | Referencia E1 reinicializada | MLP |
| 12 | LSTM32 | Referencia E1 reinicializada | Recurrente |
| 13 | GRU32 | Referencia E1 reinicializada | Recurrente |
| 14 | Persistence | Naive | Persistencia del objetivo disponible |
| 15 | SeasonalNaive | Naive estacional | Persistencia estacional |
| 16 | ARIMA | Comparador clásico | Autorregresivo integrado de media móvil |

Los nombres son las identidades ejecutables; «persistencia» y «naive estacional» son las etiquetas de presentación de `Persistence` y `SeasonalNaive`. Las referencias E1 mantienen su arquitectura y se reajustan para E2 con pesos reinicializados y el protocolo común. Sus métricas antiguas no sustituyen una evaluación de E2.

La asignación de **cuatro familias dentro del Top 10** se fundamenta en [§38](38_cumplimiento_guia.md): convolucional, recurrente/autoencoder, mezcla MLP y atención. Los controles E1 y los dos sistemas retirados no se usan para completar ese mínimo. La selección bibliográfica, los puntajes, las fichas y la revisión de las adaptaciones continúan sujetos a sus evidencias de cierre.

## Sistemas retirados del alcance obligatorio

| Sistema | Tratamiento desde esta enmienda |
|---|---|
| CNN-1D + LSTM + atención temporal | Candidato adicional opcional; no exige completar su HPO/NAS ni su estudio propio para cerrar el catálogo obligatorio. No se declara ganador ni mejora validada. |
| Ridge | Comparador opcional; el comparador clásico obligatorio se cubre con ARIMA. No se programan ajustes Ridge para la ruta del catálogo requerido. |

No se borran los ensayos, las fuentes ni las descargas anteriores. El trabajo compatible de los **sistemas retenidos** se puede reutilizar cuando coincidan receta, arquitectura, datos, partición, semilla y política de ejecución. Excluir dos sistemas no autoriza cambiar la procedencia de un resultado ni contarlo como un ajuste nuevo.

## Registro ejecutable y plantillas

- [Manifiesto vigente, con la política de arquitectura de §44](../_static/published_architectures_20261112/catalogue.json).
- [Manifiesto del cierre de catálogo del paso 1](../_static/required_catalogue_20261112/catalogue.json), conservado con sus dieciséis sistemas y exclusiones explícitas.
- [Plantilla A.2 vigente: dieciséis filas](../_static/published_architectures_20261112/benchmark_A2_template.csv). Los campos sin evidencia permanecen vacíos; esta plantilla no contiene resultados finales. Se conserva la [plantilla del paso 1](../_static/required_catalogue_20261112/benchmark_A2_template.csv) con sus bytes originales.
- [Módulo de catálogo obligatorio](https://github.com/jassan-bit/DeepLearning_202630/blob/main/scripts/e2_required_catalogue.py) y [registro versionado en el repositorio](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e2_required_catalogue_20261112/catalogue.json).

La capa vigente identifica los sistemas requeridos, aplica la [política de arquitectura/NAS de §44](44_arquitecturas_publicadas.md) y permite seleccionar los tres comparadores sin ejecutar Ridge. Los constructores históricos de catorce redes y sus protocolos sellados permanecen disponibles para verificar los recibos anteriores. La existencia de este catálogo tampoco acredita por sí sola la integración del benchmark final histórico ni su ejecución completa.

## Qué cambia en la campaña y qué queda pendiente

**El notebook y la campaña sellada de Colab todavía corresponden al plan anterior de catorce redes.** Publicar este libro no modifica el proceso ni las variables de una sesión privada. Para aplicar el alcance nuevo a la búsqueda se necesita una transición explícita, que conserve los resultados compatibles, el presupuesto consumido y los intentos sin desenlace. Hasta entonces, los contadores de [§41](41_avance_optimizacion_11_origenes.md) y el techo de **85.074 intentos** de [§40](40_calendario_original.md) describen ese plan anterior; no son el número de corridas exigido por el PDF ni el total actualizado del catálogo de dieciséis sistemas.

El cierre de catálogo establece **qué sistemas son obligatorios**. El paso siguiente ya se documenta en [§44](44_arquitecturas_publicadas.md): nueve cuerpos definidos del Top 10 y TiDE como variante que conserva NAS. Las cuatro restauraciones tienen recetas nuevas; sus ensayos NAS anteriores no se renombran. Se conservan los once orígenes, las oportunidades HPO y los ejes aplicables; no se reducen k, semillas, épocas ni presupuesto, ni se fija R. Tampoco se omiten evaluación común, potencia, inferencia, costes, robustez, incertidumbre, explicabilidad del Top 3, ablaciones del mejor, análisis de errores ni reproducción.

**Próximo cierre operativo:** incorporar el catálogo y la política de arquitectura al controlador, verificar qué resultados son compatibles y conciliar el saldo antes de nuevas corridas afectadas. La adjudicación de cuerpos de §44 no completa el benchmark ni el cumplimiento integral de E2.
