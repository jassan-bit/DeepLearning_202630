# Análisis ejecutable de evaluación ampliada y robustez

Código preparado con asistencia de IA. No contiene resultados nuevos de mercado:
la verificación local utiliza únicamente datos generados. Los CSV científicos
reales deben proceder del ejecutor de evaluación y quedar ligados al protocolo,
los datos y los checkpoints que los produjeron.

## Contrato del panel principal

`analyze.run(panel_csv, protocol_json, output, run_manifest=None,
resources_csv=None, learning_csv=None, learning_index=None,
hpo_sensitivity_csv=None)` analiza un panel
guardado; nunca obtiene datos ni llama a modelos o entrenadores. También ofrece
CLI con `--panel`, `--protocol`, `--output`, `--run-manifest`, `--resources`,
`--learning-panel`, `--learning-index` y `--hpo-sensitivity`.

`--panel` admite `panel_manifest.json`: cada CSV por conjunto/modelo se lee
por separado y se comprueban su SHA-256, tamaño y número de filas. La familia,
semillas y todos los grupos se verifican después de leerlos; un manifiesto de
progreso incompleto no sirve como evaluación final. No se necesita materializar
un CSV único de cientos de megabytes.

Columnas obligatorias: `dataset,split,scenario,model,seed,date,target,
raw_prediction,prediction,known_floor`. Los splits son `fit`, `validation`,
`calibration` y `test`. La predicción primaria debe ser
`max(raw_prediction, known_floor)`; el piso se calcula con las observaciones
realmente disponibles en cada escenario. Las métricas crudas se conservan como
diagnóstico secundario de la proyección, sin reemplazar la comparación primaria.

La familia real tiene 19 sistemas: los 17 históricos y los dos controles
`floor_only`/`floor_plus_mean_excess`. Las catorce redes usan las mismas diez
semillas como mínimo; Persistence, SeasonalNaive, ARIMA y los dos controles
analíticos usan `seed=-1`, una trayectoria cada uno. Se requieren D00 —BTC
histórico conocido— y los tres conjuntos adicionales D01/D02/D03.

No se admiten fechas, modelos, semillas, escenarios u objetivos omitidos o
añadidos. Los objetivos y calendarios coinciden exactamente entre modelos,
semillas y escenarios; las particiones son cronológicas y disjuntas. La purga
del soporte de entradas/etiquetas se verifica en el ejecutor de ajuste, no se
deduce aquí de fechas de objetivos. El manifiesto opcional vincula SHA-256 del
protocolo, panel y commit; esos enlaces de contenido no constituyen por sí solos
una prueba independiente de registro temporal anterior a resultados.

## Métricas e incertidumbre

El estimando primario es la **media de RMSE individuales por semilla**, no la
RMSE de un ensamble. Se reportan RMSE, MAE, sMAPE y R² por trayectoria y sus
agregados. R² permanece ausente si el objetivo es constante; un denominador
cero no se sustituye por una puntuación favorable.

Se reutilizan las funciones genéricas de remuestreo de
`experiments/e3_estadistica_posterior_20261117/analysis.py`, registrando su hash.
El bootstrap circular usa la misma selección de fechas para todos los modelos
y semillas de un dataset. Recalcula las raíces dentro de cada trayectoria,
antes de promediar. Los bloques de 7/14/28 observaciones son análisis de
sensibilidad prefijados; una sesión bursátil y un día UTC son relojes distintos.

Los IC temporales condicionan en las trayectorias ajustadas. La DE y Cohen dz
entre semillas describen otra fuente de variación sobre el calendario fijo;
no se combinan días y semillas como réplicas iid. Los IC nominales y las bandas
Bonferroni exploratorias identifican familias distintas: **72** comparaciones
RMSE no perturbadas y **216** bajo los tres niveles de ruido. Se expone cuántos
remuestreos caen en cada cola simultánea: con 9999 repeticiones esas colas pueden
ser escasas, por lo que no se usa su signo para acreditar una decisión R3.
No se producen p-valores ni se presenta esto como las tres pruebas de la guía.

Se ofrecen rangos dentro de cada conjunto y rangos medios descriptivos. No se
promedian RMSE de escalas diferentes ni se presenta un p-valor de Friedman como
generalización a mercados independientes: D02/D03 comparten exposición cripto
y D03 comparte subyacente y parte del calendario con BTC histórico.

## Calibración y estrés de entradas

Cada modelo/semilla estima el cuantil exacto de residuos sólo en calibración
limpia: rango `ceil((n_cal+1)*0.9)`. El intervalo es
`[max(0,prediction-q), prediction+q]`. Se conservan q y los checkpoints durante
el test limpio y los tres escenarios de ruido; no se recalibra para mejorar un
resultado de prueba. Se publican cobertura, anchura, puntuación de intervalo
y cuartos de calendario, incluidos subgrupos pequeños. No hay garantía de
cobertura futura, por grupo o bajo dependencia/cambio temporal.

Un intervalo único del 90 % no determina una distribución predictiva: CRPS y
PIT quedan ausentes en vez de inventarlos a partir del punto y q.

`perturb.py` implementa ruido de **medición**, con factores 0,25/0,50/1,00 de
escalas estimadas sólo en ajuste. Cada fecha tiene un error fijo y compartido
entre modelos, semillas y ventanas solapadas. Un multiplicador común a OHLC
conserva su orden; el volumen mantiene su unidad, ceros y contratos enteros.
La verdad futura permanece limpia. El piso obtenido desde entradas corruptas
puede superar esa verdad: ya no es una cota física del objetivo limpio, y esas
observaciones no se eliminan como supuestos errores.

Los tres clásicos consumen historia de sigma, que este estrés D7 no altera.
Deben declarar `source_raw_forecast_invariant=true` o el alias
`classical_raw_forecast_invariant_by_input_contract=true`. Se comprueba igualdad
de sus pronósticos **crudos**; su proyección sí puede cambiar con el piso
corrupto. No se afirma que se haya perturbado su historial completo ni que todos
los contratos de entrada hayan recibido la misma corrupción informativa.

## Curvas de tamaño y recursos

Las curvas de 0,25/0,50/1,00 requieren predicciones separadas del candidato y
TSMixer, con las primeras tres semillas prefijadas. El 100 % reutiliza el ajuste
principal; las fracciones menores necesitan optimización real y un escalador
estimado en su propio subconjunto. `sample_curve_jobs.json` contiene una lista
de registros —o un objeto con clave `fits`— con dataset/model/seed/fracción,
`fit_id` único, `training_rows`,
`full_training_rows`, `scaler_fit_rows`, `optimizer_steps`, `actual_new_fit`,
`reused_base_fit`, `fit_receipt_path` relativo y `fit_receipt_sha256`. Se comprueba
cada recibo y la alineación de todos los objetivos y fechas. No se acepta
renombrar predicciones de un ajuste completo como curvas de aprendizaje.

Usar prefijos más cortos cambia también la cobertura temporal del ajuste; las
tablas declaran ese factor de confusión y no atribuyen toda la diferencia al
número de ejemplos. La elección del comparador se fija antes de resultados de
test, no se escoge el rival más conveniente después.

`resources.csv` conserva mediciones por dataset/modelo/semilla/fase. Los campos
ausentes de FLOPs, energía o perfil comparable se mantienen ausentes; contar
parámetros no mide por sí solo esos recursos. Las ausencias se reportan como
evidencia pendiente, sin convertirlas en ceros o en una frontera global falsa.

El recibo de salida distingue estas tareas pendientes y mantiene
`R1_accredited=false`, `R3_accredited=false` y
`robustness_component_complete=false`. Ejecutar este análisis mejora evidencia
concreta, pero no certifica automáticamente toda la rúbrica o un veredicto R3.

## Sensibilidad de hiperparámetros

`HPO_sensitivity.csv` contiene los tres ensayos prefijados de tasa de aprendizaje,
una semilla HPO fija y todos los modelos neuronales/conjuntos. El ejecutor cierra
y sincroniza la selección por validación **antes** de inferir test con esos
pesos. Se describen todos los ensayos y se comprueba que el indicador seleccionado
sea el argmínimo de validación, incluso si otro ensayo tiene menor RMSE de test.
Es una tabla secundaria de sensibilidad, no una selección nueva ni el resultado
primario de diez semillas. Este análisis no vuelve a optimizar.

## Figuras exportables

`plots.py --analysis-dir analysis --output figures` verifica primero el manifiesto
de tablas y produce PNG/SVG estáticos con Matplotlib. Se usa el perfil científico
fijado de `deployment/science/requirements.lock.txt` en un entorno separado del
kernel. Las barras RMSE incluyen los IC temporales nominales ya calculados, con
un panel por dataset y toda la familia; esos IC no se presentan como simultáneos.
Los gráficos de ruido muestran comparadores preseleccionados, y las curvas de
aprendizaje usan DE entre sus tres semillas, identificada como tal. La cobertura
trimestral se muestra como diagnóstico descriptivo. El manifiesto de figuras liga
sus bytes a las tablas, protocolo y código. Las figuras de las pruebas de software
llevan una marca visible de datos sintéticos y se generan en directorios temporales.
