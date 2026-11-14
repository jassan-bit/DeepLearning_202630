# 14. Arquitecturas ejecutadas y condición de NAS

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
