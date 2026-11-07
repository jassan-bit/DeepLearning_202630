# Cribado del corpus recibido E2 — 2026-11-06

**Primera pasada por reglas completada; adjudicación humana y elegibilidad final abiertas.** No es una nueva consulta WoS/Scopus ni una revisión sistemática global cerrada.

Se procesaron los **7,833 grupos** de los archivos recibidos (8,802 ocurrencias, 969 duplicaciones). Scopus sigue en 7,500/7,874; los **374 no recibidos** no son exclusiones PRISMA. El reenvío CSV tiene el mismo hash del original y no se suma.

## Decisiones por registro

`screening_ledger.csv` registra cada grupo, regla, señales literales, huella del abstract privado, procedencia, estado y motivo. `priority_queue.csv` ordena por cercanía temática; ninguna prioridad es un resultado predictivo o una inclusión final.

- exclude_title_abstract_provisional: **4,323** grupos.
- manual_review_pending: **1,074** grupos.
- retain_for_fulltext: **2,436** grupos.

Las exclusiones son recomendaciones automáticas provisionales; los términos de volatilidad pueden ser sólo contexto. Objetivo real, rol de comparador autónomo, temporalidad y calidad se verifican en texto completo. Hay cero adjudicaciones humanas independientes afirmadas y no se calcula κ de Cohen.

## Originales y apoyo aplicado

Se conservan 10 originales extraídos (215 páginas). Coinciden en el corpus nativo **8**; las obras no localizadas en el corpus nativo (Autoformer, Informer) corresponden a ampliación dirigida. No se insertan como nuevos registros nativos.

Los ocho apoyos representativos se enlazan por DOI; S-Mamba coincide con un original. Los siete cuerpos aplicados adicionales siguen pendientes de extracción; sus abstracts no se convierten en evidencia metodológica completa. Las cifras históricas no se reescriben.

## Búsqueda transversal local

Se ejecutaron tres conjunciones de mecanismo + modalidad de origen + transferencia/adaptación sobre título y abstracts privados recibidos. Las expresiones y resultados se guardan en `local_transversal_searches.json` y `local_transversal_hits.csv`. No reemplazan nuevas búsquedas nativas con sus fechas, totales y cobertura global.

- T1_attention_vision: **15** grupos coincidentes.
- T2_mixing_vision: **3** grupos coincidentes.
- T3_ssm_language: **90** grupos coincidentes.

## Reconsideración del Top 10 y alternativas

`candidate_reassessment.csv` contiene 29 modelos/familias, incluidos alternativas y baselines. Las cantidades son **menciones por grupo**, no recurrencia verificada como comparador. Se mantienen separados los originales y sus citas de los estudios de apoyo; no se suman WoS y Scopus. Los intervalos anteriores de los diez propuestos son históricos; no se impone un ranking nuevo a alternativas con código, identidad, cuartil o rendimiento desconocidos.

Además se leyeron editorialmente **17 resúmenes nativos** seleccionados: las verificaciones están en `editorial_title_abstract_checks.json`. Se identifican explícitamente trabajos de precio/retorno/trading que mencionan volatilidad como contexto; no se suman como respaldo directo de pronóstico de volatilidad. Esta lectura asistida no equivale a doble revisión humana ni a lectura de sus cuerpos completos.

Los campos desconocidos permanecen vacíos/null; no reciben puntuación cero. La pertenencia al índice se conserva cuando consta en WoS, pero no acredita Q1. DLinear figura como contexto lineal y no completa un cupo profundo; MLP/CNN requieren verificar profundidad/dimensionalidad.

## Cierre pendiente

Validar decisiones por lectura editorial, recuperar textos retenidos priorizando volatilidad cripto/financiera, confirmar objetivo/roles y calidad, ejecutar búsquedas nativas transversales, completar los 374 Scopus y el historial de consultas, verificar cuartiles por año, y comparar seleccionados/excluidos con componentes verificables. La elegibilidad final y el PRISMA completo quedan abiertos.

## Reproducción

```bash
python scripts/screen_e2_literature.py --private-combined /RUTA_PRIVADA/private_combined_records.json
python -m unittest discover -s tests -p 'test_e2_literature_screening.py'
```

El input privado debe corresponder exactamente a los 7.833 IDs y metadatos públicos. Un desfase, duplicación o ausencia aborta; no hay descarte silencioso. El output público no contiene abstracts completos, referencias nativas ni registros científicos íntegros.
