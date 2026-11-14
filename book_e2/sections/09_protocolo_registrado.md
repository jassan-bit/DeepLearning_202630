# 9. Registros, enmiendas y cierre de la ejecución

**Estado vigente del 14/11/2026:** están cerrados la selección reducida, los **78 ajustes finales neuronales**, los **diez pilotos**, los **tres comparadores** y la evaluación común de los **16 sistemas**. Se completaron además **81 reproducciones y 18 ablaciones**. [§48](48_protocolo_reducido.md) reúne los registros y transiciones del ejecutor; [§55](55_reproduccion_ablaciones.md), los recibos del cierre complementario.

## 9.1 Registros que definen la entrega

| Elemento | Fuente y alcance |
|---|---|
| Catálogo obligatorio | [§43](43_catalogo_obligatorio.md): diez candidatos, tres referencias E1 y tres comparadores; híbrido y Ridge históricos/opcionales |
| Cuerpos ejecutados | [§44](44_arquitecturas_publicadas.md) y actualización de [§48](48_protocolo_reducido.md), incluida TiDE publicada r=0 |
| Selección y calendario efectivos | [§48](48_protocolo_reducido.md): seis configuraciones por red y tres folds; ajuste final hasta 31/10/2024, embargo y calibración disjunta |
| Congelación anterior al scoring de esta campaña | [81 predictores](../_static/e2_benchmark_final_20261113/final_freeze.json), con pesos, escaladores y calibradores |
| Admisión y soporte común | [Recibo de datos](../_static/e2_benchmark_final_20261113/data_admission.json): 608 fechas conocidas de E1 |
| Resultados originales | [A2](../_static/e2_benchmark_final_20261113/benchmark_A2.csv) y [familia primaria](../_static/e2_benchmark_final_20261113/primary_contrasts.csv) |

Las enmiendas conservan sus fechas reales. Los cambios de adquisición, respaldo y recursos tienen recibos propios y no se renombran como cambios de receta científica. El registro original de P0, de noviembre de 2026, sigue disponible como antecedente y no acredita por sí solo los modelos incorporados posteriormente.

## 9.2 Familia primaria y análisis posteriores

La familia primaria contiene **20 pares**: cada candidato frente a GRU32 y frente a Persistence, con margen δ heredado y ajuste Holm. La evaluación original usó 4.999 sorteos. El [plan posterior](../_static/e2_temporal_final_audit_20261114/analysis_plan.json), registrado antes de sus cálculos en [`60cd2bb`](https://github.com/jassan-bit/DeepLearning_202630/commit/60cd2bb23f6da22797e2c3a7bf8225250753f903), amplió **la misma familia** a 20.000. Reprodujo exactamente los veinte resultados originales antes del refinamiento; todos los p Holm siguen siendo 1. [§56](56_inferencia_estabilidad_final.md) distingue esa precisión Monte Carlo de un preregistro anterior al conocimiento del test.

Los diagnósticos F2a, DM/MCS, robustez, explicaciones, ablaciones y taxonomía se identifican como suplementos posteriores. Sus familias y estimandos no se presentan como confirmaciones adicionales de cada contraste primario. Se conservan las hipótesis, recetas, semillas y resultados negativos.

## 9.3 Potencia, repeticiones y desviaciones

El piloto midió cinco semillas pareadas de iTransformer y GRU32. Con R=6 y el efecto planificado, su potencia condicional estimada fue **0,123**, inferior a 0,80. Se conservó R=6 por recursos y se declaró la limitación; no se amplió R tras ver el test para buscar significación. El [informe de potencia y MDE](../_static/e2_benchmark_final_20261113/pilot_power_MDE.json) conserva supuestos y alcance.

La prueba ya era conocida desde E1. La congelación actual no recupera cegamiento prospectivo; tampoco existe un margen de sobreajuste δg previo que pueda reconstruirse a posteriori. El [Apéndice B actual](61_reproduccion_consolidada.md) registra estas limitaciones como tales. No se transforma un corte fechado ni una justificación posterior en una decisión anterior.

## 9.4 Cierre y trazabilidad

El [estado final](../_static/e2_reproduction_ablations_20261113/status.json) documenta 99/99 ajustes complementarios cerrados, sin reservas abiertas. Los fallos HPO, el desenlace desconocido T4, las incidencias ARIMA y las pausas de transporte conservan sus cargos y recibos. Un build del libro comprueba presentación; una auditoría aritmética comprueba los resultados guardados; los recibos de reajuste acreditan el entrenamiento realmente repetido. Sus alcances se separan en [§8](08_reproducibilidad.md).


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/09_protocolo_registrado.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).
