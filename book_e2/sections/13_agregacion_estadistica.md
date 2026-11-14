# 13. Estimandos, agregación e inferencia actual

**Estado vigente del 14/11/2026:** la A2 compara **16 sistemas y 608 fechas**, con R=6 para las trece redes y R=1 para los tres comparadores deterministas. El [refinamiento de §56](56_inferencia_estabilidad_final.md) usa 20.000 sorteos de la misma familia primaria; los análisis posteriores se identifican por su estimando y su familia.

## 13.1 Qué se estima

Para un modelo m y semilla s se calcula la RMSE de toda su trayectoria sobre las 608 fechas comunes. La primaria de una red es la media de sus seis RMSE individuales. La RMSE del pronóstico promedio es otro estimando. No se promedian RMSE mensuales sin ponderación para sustituir el error del calendario completo.

Hay una sola serie de mercado. Las seis semillas describen variación de optimización, no seis mercados independientes. Se emparejan las fechas y las semillas cuando corresponde; los comparadores deterministas conservan una única trayectoria. Las máscaras mantienen las posiciones reales del calendario sin comprimir huecos.

## 13.2 Familia primaria y refinamiento

La familia tiene **20 comparaciones**: cada candidato frente a GRU32 y Persistence. La ganancia es RMSE de la referencia menos RMSE del candidato; el margen práctico es **δ=0,00025772842664981106**. Se usa bootstrap pareado de semillas y posiciones temporales, bloque principal de 14 días y Holm20.

El cálculo original de 4.999 réplicas queda disponible. [§56](56_inferencia_estabilidad_final.md) lo reproduce exactamente antes de recalcular las mismas hipótesis y el mismo estimando con **20.000 réplicas**. Los veinte p Holm siguen siendo 1. Este refinamiento mejora la resolución Monte Carlo; no cambia recetas, márgenes ni el carácter retrospectivo de la prueba.

## 13.3 Comparaciones y diagnósticos posteriores

[§50](50_comparaciones_f2a.md) aporta DM–HLN/HAC14 y MCS sobre pérdidas cuadráticas diarias, además de rangos condicionados a semillas. Son objetivos distintos de una mejora práctica de RMSE mayor que δ. [§56](56_inferencia_estabilidad_final.md) separa Giacomini–Rossi bajo igualdad predictiva de un diagnóstico centrado de constancia; ajusta conjuntamente 40 hipótesis. Un rechazo de igualdad puede deberse a una ventaja persistente y no demostrar un quiebre. Un no rechazo no demuestra equivalencia ni estabilidad universal.

Las [ablaciones](55_reproduccion_ablaciones.md) usan 10.000 remuestreos y Holm3, conservando sus seis pares de semillas. La taxonomía de [§57](57_errores_techo.md) informa IC marginales de frecuencias bajo dependencia local; no ofrece cobertura simultánea de todas las categorías ni define causas de error.

## 13.4 Potencia, brechas y lenguaje de conclusión

El piloto de cinco pares de semillas estimó potencia **0,123** para el efecto planificado con R=6, frente al objetivo 0,80. Su alcance se limita a dos representantes y 53 fechas de desarrollo. No se amplían repeticiones después de observar la prueba para buscar un resultado favorable.

No hubo un margen de sobreajuste δg definido previamente. Las 78 brechas guardadas y las tolerancias ilustrativas de [§56](56_inferencia_estabilidad_final.md) son descriptivas; no acreditan no inferioridad train–validación. La calibración no se presenta como una nueva validación independiente. Menor RMSE observada, ausencia de rechazo y compatibilidad numérica de una réplica son conclusiones diferentes.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/13_agregacion_estadistica.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).
