# Cribado por título de la búsqueda Colab de R1

Preparado con asistencia de IA el 18 de noviembre de 2026. **No es una revisión humana independiente.** Se cribaron los títulos de las 180 ocurrencias recuperadas en las primeras páginas; no se declara que se hayan leído sus resúmenes o textos completos en esta etapa.

## Procedencia y conservación

Ejecución `20261118T202323797581Z`, código ejecutado `314f6e7e888b0748c8790cf3c0d9f4fc71bd6a0b`, commit de resultados `edf6fe312987fd7299f6ca850a20a67e4f1c3cbc`. Los originales se conservan en [el registro inmutable de la ejecución](https://github.com/jassan-bit/DeepLearning_202630/tree/edf6fe312987fd7299f6ca850a20a67e4f1c3cbc/runs/r1_search/20261118T202323797581Z). No se modificaron `search_receipt.json`, `screening.csv` ni las respuestas originales. Este cribado es un conjunto de archivos derivado, con autoría asistida por IA y alcance propio.

Las 30 solicitudes produjeron seis respuestas válidas en cada una de Crossref, arXiv y OpenAlex. La cobertura de respuestas nativas en tres bases es real; los conteos de resultados nativos no son conteos de trabajos pertinentes. Cada consulta recuperó sólo los primeros diez registros, con ordenación y semántica propias de cada base.

## Criterios del cribado

- `include_for_abstract`: el título justifica examinar el resumen por tarea, arquitectura, restricción de salida o método próximo. No confirma equivalencia ni relevancia final.
- `context_only`: referencia contextual; el título no identifica una diferencia cercana del mecanismo. Se conserva para seguimiento y no se usa como evidencia de novedad.
- `exclude_component`: figura o tabla con identificador propio; no se cuenta como artículo independiente. Su trabajo padre puede seguirse si resulta pertinente.
- `exclude_irrelevant`: el título corresponde inequívocamente a otro tema o a una coincidencia léxica ajena al objeto del cribado. La exclusión tiene alcance por título, no un juicio sobre toda contribución del texto.
- `uncertain`: el título no permite resolver la pertinencia; requiere resumen.
- `exclude_duplicate`: el identificador DOI normalizado o arXiv ya está representado por otra ocurrencia. Se mantiene la fila y su consulta de origen; la decisión inicial por título queda en `title_stage_decision`.

Se consideraron pertinentes tanto antecedentes de volatilidad y CNN–LSTM–atención como mecanismos de restricción en otras aplicaciones. Por ello las redes de control físico con límites de salida se remiten a resumen: no se presupone que un límite de la planta equivalga a una guarda de la salida neuronal.

## Resultado limitado a los registros guardados

| Decisión | Ocurrencias |
|---|---:|
| Examinar resumen | 48 |
| Contexto | 59 |
| Componente, no artículo independiente | 14 |
| Duplicado por identificador | 14 |
| Fuera de alcance por título | 43 |
| Incierto | 2 |
| Total | 180 |

Quedan **166 identificadores distintos**, que no se presentan como 166 trabajos únicos comprobados. Tres grupos comparten título con identificadores diferentes: las tres entradas SSRN de *Asymmetric Reverting and Volatility in Cryptocurrency Markets*, la versión SSRN/editorial de *Cryptocurrency Volatility Markets* y los registros 2011/2017 de *Do Phillips Curves Conditionally Help to Forecast Inflation?*. Se conservan como posibles versiones relacionadas; su relación no se decide sólo por título.

## Calidad de recuperación y siguiente etapa

Las primeras páginas contienen ruido importante. Crossref devuelve DOIs de figuras y tablas; la cadena española sobre «piso positivo» recupera inmuebles, trabajo en hoteles y otros homónimos. Las expresiones arXiv históricas `all:` seguidas de palabras sin una conjunción explícita no aseguran la intersección de todos los conceptos: Q4 recupera PISO, planetas y gravitación; Q6 incluye difusión de vídeo y física. Esa observación exige refinar la estrategia y conservar las consultas anteriores, no reinterpretarlas como búsquedas exhaustivas o como ausencia de antecedentes.

`priority_for_abstract.csv` propone quince primeras lecturas de las 48 remitidas a resumen. Destacan *Incorporating Interpretable Output Constraints in Bayesian Neural Networks*, el pronóstico probabilístico de volatilidad cripto, redes para volatilidad cripto y métodos de proyección. ConstraintNet y Peng reaparecen como antecedentes ya identificados; volver a recuperarlos no es una nueva lectura completa. La lectura posterior de fuentes se documenta aparte, con localizadores reales, y no se atribuye retrospectivamente a este cribado.

El cierre requiere examinar los resúmenes seleccionados y los dos inciertos, obtener los cuerpos pertinentes, completar el seguimiento de citas y tratar los límites de cobertura. La búsqueda deberá repetirse antes de la entrega definitiva. **`screening_complete=false`, R1 sin acreditar y revisión humana pendiente** siguen siendo los estados correctos del expediente global. No hubo ajustes, predicciones de mercado o contrastes nuevos.

## Archivos

- `title_screening.csv`: las 180 ocurrencias con decisión, razón, procedencia y deduplicación.
- `title_screening_summary.json`: conteos, cobertura y límites explícitos.
- `deduplication.json`: duplicados exactos y grupos de posibles versiones no fusionadas.
- `priority_for_abstract.csv`: cola inicial de quince fuentes; no certifica lecturas ejecutadas.
