# 15. Catálogo ejecutado y procedencia bibliográfica

**Estado vigente del 14/11/2026:** el catálogo obligatorio de E2 comprende **16 sistemas**: diez candidatos de cuatro familias, tres referencias E1 y tres comparadores. El híbrido CNN-1D + LSTM + atención y Ridge son antecedentes opcionales; no constituyen posiciones obligatorias adicionales ni ganadores demostrados.

## 15.1 Diez candidatos y fuentes

| Modelo | Familia y mecanismo principal | Evidencia vigente |
|---|---|---|
| FCN-1D | Convoluciones temporales y agregación global | Wang; receta adaptada D7 |
| ResNet-1D | Bloques convolucionales con conexiones residuales | Comparte fuente Wang con FCN; es otro sistema, no otro estudio |
| Inception-1D | Ramas convolucionales de distintos kernels | InceptionTime; identidad de miembro/adaptación declarada |
| LSTM-autoencoder con predictor | Preentrenamiento reconstructivo y transferencia recurrente | Kotb et al.; fases auxiliares y predictor explícitos |
| TimesNet | Periodicidad y representación 2D con convoluciones | Cuerpo restaurado y adaptaciones registradas |
| TSMixer | Mezcla temporal y de características mediante MLP | Cuerpo publicado y lectura D7 |
| TiDE | Encoder-decoder denso con residuales | Receta publicada r=0, distinta de la variante NAS anterior |
| TimeMixer | Descomposición y mezcla multiescala | Receta adaptada al soporte D7 |
| PatchTST | Parches temporales y atención | Cuerpo restaurado y cabeza de regresión registrada |
| iTransformer | Atención sobre variables | Fuentes de autor fijadas y adaptación de salida |

Los identificadores, archivos, commits, locadores y decisiones de arquitectura se consultan en [§44](44_arquitecturas_publicadas.md) y [§48](48_protocolo_reducido.md). La comparación final y sus seis semillas por red están en [§6](06_benchmark_final.md). MLP16, LSTM32 y GRU32 son las referencias E1; Persistence, SeasonalNaive y ARIMA completan los dieciséis.

## 15.2 Alcance de la revisión y de las puntuaciones

[§59](59_auditoria_literatura_final.md) concilia **9.559 grupos recibidos**, manteniendo aparte las recuperaciones dirigidas. Diez sistemas actuales se vinculan con nueve fuentes primarias; compartir una fuente no multiplica estudios. La adjudicación documental no convierte el corpus recibido en un PRISMA global cerrado.

El puntaje anterior dispone de **36/50 componentes observados**. La sensibilidad contextual ETTh1/H96 añade información de P para seis modelos —cinco al estratificar por lookback— y conserva diferencias de split, escala e historia. Es una sensibilidad retrospectiva, no el ranking que determinó el catálogo. Los componentes desconocidos no se imputan como cero ni se rellenan con resultados A2.

## 15.3 Distinción de antecedentes

El catálogo editorial de noviembre de 2026 incluyó antes un candidato 11 híbrido y prototipos compactos. Su fuente exacta anterior se conserva debajo y en los registros históricos. Los recibos de esas variantes no se reasignan a los cuerpos ejecutados. La elección de una continuidad temática de E3 queda separada del cierre de E2 y de la guía específica de E3.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/15_revision_modelos.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).
