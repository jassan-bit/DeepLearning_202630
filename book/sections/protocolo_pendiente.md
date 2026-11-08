# Decisiones, evidencia y alcance de E1

**El nuevo benchmark principal compara GRU32, LSTM32 y MLP16 sobre las mismas 608 fechas del 01/01/2025–31/08/2026.** Su HPO equitativo, piloto propio y dimensionamiento **R=309** preceden a los ajustes definitivos. El {ref}`capítulo 11 <remediation-608-e1>` documenta la nueva ejecución coherente de las tres redes. El [capítulo 5](05_benchmark.md) conserva las diez semillas originales y las cuatro entradas; el {ref}`capítulo 10 <final-core-mlp-e1>` mantiene la comprobación complementaria de sesenta días con otras recetas. Se conservan las identidades y resultados de cada estudio.

## Literatura: selección y trazabilidad

| Requisito | Evidencia entregada | Alcance |
|---|---|---|
| **§3.1: WoS, Scopus e IEEE** | {ref}`Bitácora nativa <current-native-execution-e1>`: cadenas, fecha confirmada del 06/11/2026 en Colombia, campos, filtros y totales **271/538/32**; exportados **271/538/25** | Consulta acreditada; resultados, exportaciones y textos leídos se cuentan por separado |
| **§3.2: recurrencia** | {ref}`Selección de familias <core-selection-e1>`: diez experimentos cripto con reporte suficiente, **LSTM 10, MLP 4, GRU 2, TCN 1** | Las tres primeras familias del corpus examinado; cobertura y búsquedas dirigidas explícitas |
| **§3.2: simplicidad** | Una capa oculta MLP16 de **593 parámetros**, una capa GRU32 de **3.777** y una LSTM32 de **5.025** | Capacidad y profundidad declaradas; adaptaciones propias con respaldo aplicado, no réplicas exactas |
| **§3.2: diversidad** | Ventana ordenada sin recurrencia, actualización/reset GRU y memoria LSTM separada; misma información D7×5 | Mecanismos distintos, sin afirmar independencia de errores |
| **Originales y fuentes gratuitas** | {ref}`Síntesis de 57 obras <homogeneous-expansion-e1>`, 35 textos primarios completos y una transcripción; catálogo de 70 referencias | Catorce informes no recuperados conservan disposición y motivo; alternativas verificadas sin inventar lecturas |

TCN conserva sus resultados como complemento. Los cortes bibliográficos fechados mantienen sus denominadores. La selección documental no se calcula a partir de las puntuaciones de test.

Las sesiones antiguas conservan cadenas y horas no recuperadas. La ronda nativa posterior documenta consultas reales de las tres bases sin reconstruir esos valores. IEEE exportó 25/32 resultados; se distingue el universo exportado de la consulta efectiva.

## Nueva comparación principal de 608 días

La ejecución **e1_608_remediation_20261107** mantiene cinco covariables OHLCV, σ₇ poblacional, D7, horizonte de un día, transformación relativa y escalas aprendidas del entrenamiento. Las arquitecturas originales se conservan con nuevos pesos y una HPO común de LR/dropout. El {ref}`capítulo 11 <remediation-608-e1>` identifica las fases, sus hashes y los resultados de la evaluación cerrada.

| Requisito | Evidencia de la nueva ejecución | Alcance |
|---|---|---|
| HPO comparable, §§4.1–4.3 | Cuatro candidatos por familia × dos semillas × diez bloques enero–octubre de 2024; **480 etapas**, 160 por red | Igual esfuerzo de búsqueda dentro de LR/dropout; anchos 32/32/16 conservados |
| Recetas elegidas | **GRU32 c03**, LR 0,003/dropout 0,1; **LSTM32 c02**, LR 0,003/dropout 0; **MLP16 c03**, LR 0,003/dropout 0,1 | Baseline **GRU32** por menor RMSE CV, cerrado antes del piloto |
| Piloto propio | Diez semillas **17101–17110** por familia sobre 01/11–30/12/2024; 30 trayectorias y 60 etapas | Distinto de HPO y de las semillas definitivas; media, DE y CV informados |
| Dimensionamiento previo | δ=0,00025772842664981106; α*=0,05/3, potencia marginal 0,80, cota simultánea del 80 % de DE; **R requerido=R final=309** | Plan calculado tras el piloto y publicado/verificado antes del primer ajuste definitivo |
| Registro público | [Fuente 98e0cbcf](https://github.com/jassan-bit/DeepLearning_202630/commit/98e0cbcf4bb09dba863289261677dd6e7534d259) y [plan exacto 0f5f5eae](https://github.com/jassan-bit/DeepLearning_202630/commit/0f5f5eaeb7131b1413a062cb0cce104a88f3ddf3) | Fuente verificada por GET antes de HPO; plan verificado por GET 08/11/2026 01:17:26 UTC antes de registrar el ajuste definitivo |
| Entrenamiento definitivo | **927 modelos y 1.854 etapas ejecutados**, semillas comunes **18101–18409**; 709 etiquetas hasta 24/12/2024; **2.394 etapas totales** con HPO/piloto | Etapa interna causal 619/83; reajuste desde cero; los 927 pesos y preprocesadores cerrados antes de leer CSV de test |
| Inferencia temporal, §5.4 | **563.616 predicciones** de las tres redes sobre las mismas **608 fechas**; DM/HAC6/HLN/Holm-6, bootstrap circular y BDS/Ljung–Box ejecutados conforme al diseño | Vectores nuevos conservados privadamente; resultados y firmas públicos. No se reconstruyen los originales ausentes |
| Curvas, costes y contraste entre semillas | **33.141 registros de pérdida**, parámetros y tiempos; t/Holm-3, Q–Q, Friedman/Kendall W y Wilcoxon/Holm-3 publicados | Evidencia propia de la ejecución; semillas condicionadas al calendario, sin mercados independientes |

La nueva ejecución corrige el diseño mediante trabajo nuevo verificable. El {download}`resumen de cierre <../_static/e1_608_remediation_20261107/summary.json>` y la {download}`cronología <../_static/e1_608_remediation_20261107/chronology.json>` acreditan las fases; el {download}`manifiesto de publicación <../_static/e1_608_remediation_20261107/manifest.json>` identifica los archivos públicos.

| Modelo de la nueva ejecución | RMSE media | R² medio |
|---|---:|---:|
| GRU32 | 0,005212686 | 0,709833 |
| LSTM32 | 0,006251147 | 0,594658 |
| MLP16 | 0,005228132 | 0,717998 |
| Persistencia | **0,004085596** | **0,829605** |

Persistencia presenta menor error medio que las tres redes. DM/Holm-6 rechaza GRU–LSTM y las tres redes frente a persistencia; no rechaza GRU–MLP ni LSTM–MLP. Los límites inferiores temporales familiares del bloque siete para la mejora relativa de persistencia sobre GRU/LSTM/MLP son **8,60/21,27/10,22 %**. Los diagnósticos residuales rechazan BDS en 927/927 trayectorias y Ljung–Box en 1.854/1.854 pruebas después de sus ajustes Holm separados: el cierre metodológico no demuestra adecuación residual o superioridad neuronal. La {download}`tabla de seis métricas e IC <../_static/e1_608_remediation_20261107/metric_summary.csv>` mantiene la incertidumbre entre semillas; los contrastes temporales tienen su propia unidad e interpretación.

La comparación nueva satisface HPO común, piloto/dimensionamiento previo a sus propios ajustes y análisis diarios conjuntos; conserva su resultado adverso a las redes. El calendario ya se había utilizado: registrar el código y plan antes de los nuevos ajustes no recupera una reserva temporal independiente ni acredita potencia alcanzada o generalización a otros regímenes.

## Referencia histórica original de 608 días

El {download}`expediente histórico <../_static/e1_primary_608_20261107/README.md>` concilia resultados auténticos de las mismas arquitecturas y fechas. Conserva las diez semillas **4101–4110**, CV de doce bloques de 2024 y cuatro entradas; la MLP original usa LR **0,001**, frente a **0,003** en la nueva receta elegida por HPO. GRU/LSTM usan LR 0,003 y las tres redes originales dropout 0,1. La conciliación verifica **480 igualdades exactas de LSTM** entre fuentes.

Los R² medios originales son GRU **0,854173**, MLP **0,764012**, LSTM **0,741108** y persistencia **0,829605**. GRU tiene menor RMSE y mayor R² medios en esa ejecución; persistencia mejores MAE y MAPE. Esos números se conservan completos en el capítulo 5 y no se atribuyen a los nuevos pesos.

**En la ejecución original** el esfuerzo de HPO fue desigual y las diez semillas fueron un presupuesto sin piloto previo. Los vectores diarios y pesos originales de MLP no están disponibles aquí. La nueva ejecución aporta sus propios vectores, HPO, piloto y plan; no afirma haber recuperado los archivos originales ni retrotrae su registro a aquellas corridas.

## Comprobación complementaria de sesenta días

El [protocolo publicado](https://github.com/jassan-bit/DeepLearning_202630/commit/91acdeae1efd92f098ff2ce9a32f90c5b0a4b98a) y el [plan exacto](https://github.com/jassan-bit/DeepLearning_202630/commit/593b4ec943dc0f7e257d348f3145e71810f665e4) conservan sus fechas anteriores a los ajustes correspondientes. Este estudio compara **GRU16–LSTM16–MLP32**, con HPO equitativo de 960 etapas, piloto de treinta trayectorias/sesenta etapas y R requerido331 elevado por el piso previo a **R final465**.

Se cerraron 1.395 modelos y 2.790 etapas definitivas antes de abrir observaciones: **83.700 predicciones y 44.201 registros de pérdida**. La revisión independiente recalculó seis métricas con diferencia máxima cero y verificó t/Holm, DM, bootstrap y archivos públicos. Estos resultados, curvas y tamaños pertenecen exclusivamente al periodo **01/09–30/10/2026** y sus recetas.

En ese periodo persistencia tiene el menor error puntual. GRU16−MLP32 no rechaza tras Holm (p=0,243); ningún DM rechaza con Holm-6 y ningún límite familiar temporal acredita mejora relativa superior al 5 %. Se conservan estos resultados adversos e inconclusos, sin trasladar contrastes o conclusiones al test de 608 fechas.

## Margen y reproducibilidad

δ=0,00025772842664981106 conserva el ancla del 5 % del RMSE CV del GRU32 inicial. Su [justificación académica](01_problema_dataset.md) presenta contexto de variabilidad y sensibilidad al 2,5/10 %. La argumentación y planificación posteriores no se retrotraen para dimensionar las diez semillas iniciales ni demuestran utilidad financiera o equivalencia.

La nueva ejecución de 608 fechas registra sus propias curvas de entrenamiento/validación y de reajuste; sus configuraciones, dependencias y costes se identifican por estudio. Las curvas del capítulo 10 pertenecen a las recetas complementarias **16/16/32**. La {ref}`reproducción del estudio con TCN <curvas-finales-reproducidas>` y la {ref}`auditoría de repetibilidad <auditoria-numerica-e1>` también mantienen sus identidades. Ninguna se presenta como recuperación exacta de tensores privados históricos ausentes.

La guía §4.3 exige semillas, dependencias, determinismo declarado y procedimiento ejecutable. Las descargas distinguen compilación de métricas guardadas, métodos originales y ejecuciones posteriores. Construir HTML o empaquetar no entrena ni vuelve a puntuar modelos.

## Alcance temporal y continuidad a E2

Los calendarios de 608 y de sesenta fechas ya son conocidos. Las semillas describen aleatoriedad de entrenamiento sobre las mismas fechas y no son mercados independientes. El solapamiento de σ₇, los cambios temporales y las diferencias de receta limitan extrapolar resultados entre periodos.

**El benchmark principal es la nueva ejecución GRU32–LSTM32–MLP16, R=309 y 608 fechas; el original R=10 y el de sesenta días siguen visibles con sus identidades.** La nueva planificación y sus resultados no modifican los archivos científicos históricos. E2 mantiene sus fuentes, recetas y baseline registrados; esta actualización de E1 no modifica automáticamente experimentos ya cerrados.
