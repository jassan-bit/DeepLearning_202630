# Plan publicado antes de los ajustes definitivos de 608 fechas

Fuente científica inmutable:98e0cbcf4bb09dba863289261677dd6e7534d259.
HPO nuevo:24 trayectorias,240 bloques,480 etapas,4 candidatos por familia.
Arquitecturas fijas GRU32/LSTM32/MLP16. Baseline seleccionado por CV:GRU.
Piloto nuevo:10 semillas por familia,30 trayectorias,60 etapas sobre60fechas de2024.

R requerido=R final=309 por familia; semillas18101–18409.
Son927 trayectorias y1854 etapas definitivas, todas anteriores a la lectura de CSV
de evaluación. El piso30 estaba fijado antes de HPO/piloto y no altera este R mayor.
plan.json es copia exacta del cálculo privado; SHA256
6295b408623c811aa707b7c55355ab5fb529306f91049b047bac8dcc4d0fd15f.

Este registro se publica antes del primer ajuste definitivo. No es un informe
final ni una recuperación retroactiva del piloto del benchmark original.
La evaluación será histórica sobre608fechas ya conocidas; no recupera un holdout
confirmatorio independiente. Los valores de potencia suponen normalidad y
transferencia de la dispersión del piloto; no son potencia temporal o alcanzada.

Los CSV contienen métricas y configuración; se excluyen escalas aprendidas,
precios, objetivos diarios, pronósticos diarios y pesos.
