# Diagnósticos post hoc de los predictores E2 congelados

Este paquete ejecuta inferencia adicional sobre los pesos y el caché privados ya
presentes en Colab. No entrena, selecciona, recalibra, descarga datos, instala
paquetes ni modifica el resultado A2. Sus resultados son nuevos diagnósticos
descriptivos del calendario histórico conocido, no una nueva prueba prospectiva.

El motor original es `1e8745497b484518f766b6da5c19be8362d9add7`; se exige su
checkout preservado, su entorno Python 3.12/Torch 2.8.0+cu126 y una A100.
Se verifican las identidades del caché, los 81 pesos/predictores, sus escaladores,
calibradores y las 81 series guardadas antes de ejecutar los diagnósticos. Cada
predictor neural debe reproducir las 608 predicciones originales con tamaño de
lote **1** y tolerancia absoluta `1e-10` antes de admitir sus perturbaciones.
Los constructores originales verifican las fuentes externas de las arquitecturas.

Se evalúan las 13 arquitecturas neurales con sus seis semillas: los diez candidatos
y las tres referencias neurales. Cada una recibe ruido gaussiano de magnitud
`0.1/0.3/0.5` desviaciones de entrenamiento, valores atípicos en proporciones
`0.01/0.05/0.10` con desplazamiento de cinco desviaciones, sustituciones por
medianas de entrenamiento en proporciones `0.05/0.15/0.30`, y un quiebre sintético
de media de `0.25/0.5/1.0` desviaciones desde el **1 de enero de 2026**.
La referencia usa exclusivamente las ventanas de entrenamiento hasta el 31 de
octubre de 2024. Se conservan las coordenadas invariantes y el objetivo original.
Las corrupciones corresponden a entradas de medición estandarizadas por ventana;
no reconstruyen una trayectoria OHLCV coherente ni una ley financiera alternativa.

Se informa RMSE medio entre semillas, degradación pareada e intervalos mediante
2.000 remuestreos de fechas en bloques circulares de 7, 14 y 28 días. Las semillas
comparten los índices de remuestreo y no se tratan como mercados independientes.
Se retiene el cuantil conforme original para medir cobertura y anchura bajo
perturbaciones, sin recalibración. El área de degradación normalizada usa la regla
trapezoidal y se interpreta por separado para cada perturbación.

Para SAE y GRU32 se calculan oclusiones de las 35 coordenadas y agregaciones por
variable y rezago. Para ARIMA se repiten solamente los siete rezagos de sigma7
previos al pronóstico desde el estado original anterior a la ventana, conservando
los parámetros y el estado más antiguo. Se explica así a los tres primeros sistemas
del A2: SAE, ARIMA y GRU32, con entradas distintas explícitas. Las 24 fechas de
explicación se fijan por posiciones igualmente espaciadas desde el 8 de enero de
2025; no se eligen por errores. Se comparan borrado e inserción por relevancia con
el mismo control de orden aleatorio; se reportan respuesta del predictor, RMSE y
estabilidad de rangos entre las seis semillas neurales. ARIMA es determinista R=1
y no tiene estimando de estabilidad entre semillas. La oclusión no identifica
efectos causales del mercado ni sustituye una ablación de componentes con ajuste.

Cada predictor neural recibe un perfil nuevo de parámetros, almacenamiento de
estado, latencia y memoria con lote 1. La latencia usa eventos CUDA, 32 llamadas de
calentamiento y 100 repeticiones, entrada residente y sincronización; excluye
preprocesamiento y transferencias. Se declara el alcance de los máximos del
asignador y se mantienen FLOPs y energía en `null`.

La ejecución admite `run(source_root, output_root, destination, client,
wall_seconds=5400, budget=shared_ledger, configuration_commit=commit)`. Se exige
el diario común con persistencia remota del controlador unificado y el commit
publicado inmutable de 40 caracteres, que se incorpora a la identidad de resultados.
Se conserva el ancla
original de 80 horas, cuyo saldo inicial cerrado es 18.169,090972 s. No se reinicia
ni amplía el presupuesto. Cada invocación reserva hasta 5.400 s y carga el tiempo
real completo. Una interrupción conserva su reserva hasta reconciliación.

Las salidas residen en
`/content/entregable2_gpu_workspace/e2-guide-closure-posthoc-v1/robustness/`.
Cada predictor terminado se respalda inmediatamente mediante el `GitHubBackup`
original, con identidades y recibos inmutables. Los reintentos de transporte son
acotados y reutilizan cómputo terminado; un predictor iniciado sin cierre se
preserva y bloquea la repetición automática. El controlador integrado debe respaldar
también el diario compartido después de cargar cada etapa. No se usa Drive ni se
publican OHLCV, ventanas de entrada o pesos. El token existente viaja por stdin.

El punto de ejecución de producción es exclusivamente la celda del lanzador
unificado de cierre E2. Tanto la CLI independiente como
`run_e2_frozen_posthoc_colab(commit, execute=True)` se detienen antes de acceder a
secretos o ejecutar trabajo, e indican usar esa celda para conservar un único
presupuesto remoto. `execute=False` permite inspeccionar la acción solicitada.

La validación local es sintética y no acredita que los diagnósticos privados se
hayan ejecutado. Sus pruebas cubren proyección, causalidad y replay ARIMA,
perturbaciones, fidelidad, bootstrap, estabilidad, diario, reintentos y reuso de
resultados completos. La evidencia real debe provenir de los recibos de Colab.
