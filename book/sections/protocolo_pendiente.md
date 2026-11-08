# Decisiones, evidencia y alcance de E1

**El benchmark principal compara GRU32, LSTM32 y MLP16 sobre las mismas 608 fechas del 01/01/2025–31/08/2026.** Las ejecuciones originales ya existen: esta actualización verifica y prioriza sus resultados. El [capítulo 5](05_benchmark.md) reúne seis métricas y las cuatro entradas; el {ref}`capítulo 10 <final-core-mlp-e1>` conserva la comprobación complementaria de sesenta días con otras recetas. Las versiones y datos de cada estudio permanecen separados.

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

## Comparación principal verificada de 608 días

El {download}`expediente principal <../_static/e1_primary_608_20261107/README.md>` concilia resultados auténticos bajo el mismo contrato: cinco covariables OHLCV, σ₇ poblacional, D7, horizonte de un día, transformación relativa y escalas aprendidas del entrenamiento. La CV usa doce bloques de 2024 y el test las mismas 608 fechas para todos.

| Elemento | Evidencia principal |
|---|---|
| Recetas | GRU32/LSTM32, LR 0,003; MLP16, LR 0,001; dropout 0,1 |
| Entrenamiento final | 709 etiquetas comunes, 08/01/2023–24/12/2024; embargo de siete días |
| Repeticiones | Diez semillas 4101–4110 por red; presupuesto histórico fijo |
| Evaluación | Pesos y escaladores fijos; veinte meses de reporte, sin reajustes en test |
| MLP bajo las mismas condiciones | Diez corridas D7 originales verificadas en métricas y eventos; no se afirma una corrida nueva |
| Control entre fuentes | 480 igualdades exactas de LSTM: cuatro entradas × diez semillas × seis métricas × dos fases |
| Selección por desarrollo | D7 minimiza RMSE CV por familia; GRU32 es el baseline neuronal principal |
| Resultados | R² medios GRU **0,854173**, MLP **0,764012**, LSTM **0,741108**, persistencia **0,829605** |

GRU tiene menor RMSE y mayor R² medios en este periodo; persistencia conserva mejores MAE y MAPE. La tabla completa y los IC se publican en el capítulo 5. Los contrastes entre semillas son exploratorios y condicionados al calendario.

**La búsqueda histórica no tuvo el mismo esfuerzo de HPO entre familias.** Se comparan las recetas originales con reglas de evaluación comunes. Las diez semillas no proceden de un piloto previo y no se les atribuye potencia prospectiva. La prioridad actual de este benchmark se decide después de conocer los resultados; veinte meses aportan mayor cobertura temporal, sin acreditar por sí solos representatividad universal o independencia confirmatoria.

Los vectores diarios y pesos originales de MLP no están disponibles aquí. Los agregados verificados bastan para esta comparación de seis métricas; no reconstruyen DM, bootstrap temporal o residuos originales. La nueva compilación no ajusta modelos, calcula predicciones ni modifica los resultados fuente.

## Comprobación complementaria de sesenta días

El [protocolo publicado](https://github.com/jassan-bit/DeepLearning_202630/commit/91acdeae1efd92f098ff2ce9a32f90c5b0a4b98a) y el [plan exacto](https://github.com/jassan-bit/DeepLearning_202630/commit/593b4ec943dc0f7e257d348f3145e71810f665e4) conservan sus fechas anteriores a los ajustes correspondientes. Este estudio compara **GRU16–LSTM16–MLP32**, con HPO equitativo de 960 etapas, piloto de treinta trayectorias/sesenta etapas y R requerido331 elevado por el piso previo a **R final465**.

Se cerraron 1.395 modelos y 2.790 etapas definitivas antes de abrir observaciones: **83.700 predicciones y 44.201 registros de pérdida**. La revisión independiente recalculó seis métricas con diferencia máxima cero y verificó t/Holm, DM, bootstrap y archivos públicos. Estos resultados, curvas y tamaños pertenecen exclusivamente al periodo **01/09–30/10/2026** y sus recetas.

En ese periodo persistencia tiene el menor error puntual. GRU16−MLP32 no rechaza tras Holm (p=0,243); ningún DM rechaza con Holm-6 y ningún límite familiar temporal acredita mejora relativa superior al 5 %. Se conservan estos resultados adversos e inconclusos, sin trasladar contrastes o conclusiones al test de 608 fechas.

## Margen y reproducibilidad

δ=0,00025772842664981106 conserva el ancla del 5 % del RMSE CV del GRU32 inicial. Su [justificación académica](01_problema_dataset.md) presenta contexto de variabilidad y sensibilidad al 2,5/10 %. La argumentación y planificación posteriores no se retrotraen para dimensionar las diez semillas iniciales ni demuestran utilidad financiera o equivalencia.

Las curvas públicas del capítulo 10 son de las recetas complementarias16/16/32. La {ref}`reproducción del estudio con TCN <curvas-finales-reproducidas>` y la {ref}`auditoría de repetibilidad <auditoria-numerica-e1>` también mantienen sus identidades. Ninguna se presenta como recuperación exacta de tensores privados históricos ausentes.

La guía §4.3 exige semillas, dependencias, determinismo declarado y procedimiento ejecutable. Las descargas distinguen compilación de métricas guardadas, métodos originales y ejecuciones posteriores. Construir HTML o empaquetar no entrena ni vuelve a puntuar modelos.

## Alcance temporal y continuidad a E2

Ambos calendarios ya son conocidos. Las semillas describen aleatoriedad de entrenamiento sobre las mismas fechas y no son mercados independientes. El solapamiento de σ₇, los cambios temporales y las diferencias de receta limitan extrapolar resultados entre periodos.

**El benchmark principal es GRU32–LSTM32–MLP16 de 608 días; el de sesenta días sigue visible como complemento.** Esta prioridad editorial preserva los resultados completos y su cronología. E2 mantiene sus fuentes, recetas y baseline registrados; esta actualización de E1 no modifica automáticamente experimentos ya cerrados.
