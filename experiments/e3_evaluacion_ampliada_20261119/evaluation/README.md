# Evaluación ampliada con ajuste común y controles de piso

Este ejecutor está preparado para producir resultados nuevos con la GPU de Colab. La implementación y sus pruebas usaron asistencia de IA declarada. **Aún no se han ejecutado ajustes de mercado de esta campaña.** Las predicciones y entrenamientos anteriores permanecen intactos.

## Familia y dominios

La auditoría conserva los 17 sistemas históricos: 13 redes E2, Persistence, SeasonalNaive, ARIMA y el candidato aditivo E3 de 1210 parámetros. Añade `floor_only` y `floor_plus_mean_excess`, fijados antes del resultado, para un total de **19 sistemas**. V3 no se introduce retrospectivamente como candidato de esta comparación.

Se evalúan D00 BTC y los tres conjuntos adicionales D01 WIKI/AAPL RAW, D02 ETHUSDT Spot y D03 BTCUSD_PERP COIN-M. Los tres adicionales requieren admisión física completa. D00 es un calendario histórico conocido; D02/D03 comparten exposición cripto y D03 comparte subyacente y algunas fechas del entrenamiento BTC. No se afirma que sean cuatro mercados independientes o calendarios globalmente ciegos.

## Preespecificación y ajuste

La CLI exige un protocolo ligado por SHA-256 a datos y código, más el recibo de un **tag anotado real** comprobado por el controlador. La GPU y sus versiones deben coincidir con el perfil fijado antes de ajustar. Se ejecuta una auditoría exploratoria preespecificada; su tag no acredita potencia, novedad ni R3.

Todas las redes usan el mismo preprocesamiento OHLCV D7, escalador estimado sólo en ajuste, pérdida MSE de sigma cruda y selección por MSE de validación con proyección física común. La proyección `max(predicción_cruda, sqrt(A_entrada))` se aplica a **los 19 sistemas**; el panel conserva también los puntos crudos. No se deja ese factor favorable únicamente al candidato.

Por modelo y conjunto, la búsqueda automática prueba las tres tasas `0.0003`, `0.001` y `0.003`, con una semilla HPO distinta de las diez finales. La mejor validación y el desempate por índice quedan escritos antes de calcular sensibilidad sobre prueba. Máximo 40 épocas, paciencia ocho, lote 32, AdamW sin decaimiento y norma de gradiente uno. El protocolo predetermina un presupuesto de 120 segundos por ajuste, comprobado en límites de minibatch; SAE reparte **ese único presupuesto** entre AE1, AE2 y supervisión. No obtiene tres presupuestos completos. Los tiempos de serialización y posibles excesos de frontera se reportan; no hay una promesa de timeout duro del proceso.

ARIMA(1,0,1) tiene sus parámetros congelados durante el filtrado causal. Se aplica el mismo límite de oportunidad en las fronteras de iteración de su optimizador. Si no converge o se agota antes de un resultado válido, se conserva el fallo; no se sustituye por otra serie. Persistence, SeasonalNaive y `floor_only` no ejecutan optimización. `floor_plus_mean_excess` estima **un escalar** con etiquetas de ajuste: se reporta como estadístico analítico estimado, no como cero parámetros ni como un ajuste de optimizador.

Con cuatro conjuntos, el plan completo comprende 168 trabajos neuronales HPO, 560 ajustes finales y 48 ajustes adicionales de curvas de tamaño, más cuatro ARIMA: **780 intentos de ajuste numérico** si todos se ejecutan. SAE suma dos fases adicionales por trabajo propio, sin contarlas como modelos distintos. El techo nominal de fases asignadas es aproximadamente 26 horas a 120 segundos por ajuste, además de datos, inferencia, análisis y persistencia. Los tiempos reales pueden ser menores; el límite no es un cobro ni una garantía de duración.

## Separación temporal y huecos

D00 conserva 532 ventanas de ajuste, 116 de validación, 53 de calibración y las 608 de prueba conocidas. Los otros conjuntos usan fracciones cronológicas 55/15/10 por ciento y el resto de prueba, con siete soportes purgados entre segmentos. La construcción de objetivos reservados ocurre después del control del protocolo.

Los diagnósticos de ajuste de **todos** los sistemas usan una máscara fijada por disponibilidad: omitir los primeros siete objetivos y aquellos sin sigma rezagada uno o siete. Se publican fechas y razones. Esta máscara no elimina filas de entrenamiento ni de validación/calibración/prueba por su rendimiento. Para AAPL, SeasonalNaive usa siete observaciones de sesión, no siete días naturales.

La historia diaria BTC conserva el hueco de precios del 24 de marzo de 2023 y sus etiquetas sigma no disponibles. ARIMA avanza esos días con observaciones NaN en el filtro Kalman, sin fabricar valores ni comprimir el calendario. El pronóstico causal se escribe y sincroniza antes de incorporar la observación de ese día.

## Robustez y curvas

Las perturbaciones de cotizaciones sólo se aplican a prueba, con tres severidades y escala estimada en barras de ajuste. Se conservan las etiquetas sigma limpias. El piso calculado desde cotizaciones perturbadas puede dejar de ser una cota del objetivo limpio: ese efecto mide sensibilidad al error de observación. Los controles clásicos mantienen sus pronósticos crudos basados en historia, pero la proyección común puede cambiar con el piso perturbado.

Las curvas usan el candidato y TSMixer, prefijados, con fracciones 25/50/100 por ciento y las tres primeras semillas finales. Las fracciones pequeñas **sí reajustan** el modelo y su escalador usando prefijos de ajuste; la fracción completa reutiliza el ajuste final. Cambia también la cobertura del calendario, lo que se declara en el índice. La tasa de aprendizaje permanece seleccionada por la validación del ajuste completo. La sensibilidad HPO puntúa los tres modelos de tasa sólo después de congelar el ganador de validación, sin elegir un nuevo ganador por prueba.

## Salidas y reanudación

- `jobs/<dataset>/<modelo>__<fase>__<semilla>/`: solicitud, pesos seleccionados, checkpoints, optimizador/RNG, metadatos y resultado o fallo. Los resultados terminados se admiten sólo si coinciden identidad, semilla, protocolo y hashes.
- `panels/<dataset>/<modelo>.csv`: paneles por modelo, divididos para evitar un CSV de cientos de MB. `panel_manifest.json` vincula sus hashes, tamaños, semillas y filas. El controlador debe persistir un panel cuando contiene todas las semillas previstas para ese modelo.
- `HPOselection.json` y `HPO_sensitivity.csv`: selección por validación y estudio secundario posterior, separados de la familia primaria.
- `sample_curve_predictions.csv` y `sample_curve_jobs.json`: puntos y evidencia de los ajustes reales o reutilización completa.
- `resources.csv`, `resources.json`, `dataset_geometry.json`, `failures.json`, `run_manifest.json` y `evaluation_receipt.json`: recursos, soportes, integridad y alcance. FLOPs completos y energía no se inventan si no se midieron.

Las fases recuperan pesos, mejor época, optimizador y RNG desde el último límite de época durable. El plazo original es inmutable y cuenta el tiempo de reinicio conservadoramente; una fase sin actualización validada no se convierte en un pronóstico. ARIMA con marcador de inicio pero sin archivo de ajuste requiere conciliación explícita. Un fallo mantiene la ejecución parcial y nunca elimina al rival o una fecha de prueba para aparentar finalización.

API:

```python
run(output, source_commit, admission_path, protocol_path,
    freeze_receipt_path, npz_path,
    btc_history_paths=(development_csv, evaluation_csv), device="cuda:0")
```

CLI: `python evaluation/run.py --output ... --source-commit SHA --admission ... --protocol ... --freeze-receipt ... --npz ... --btc-history DEVCSV EVALCSV --device cuda:0`.

## Verificación local

`verification.json` y `tests.log` documentan **26 pruebas sintéticas**: ajustes reales de legado/SAE sobre datos generados, presupuesto compartido, gradientes, recuperación de optimizador/RNG, errores, escalador sin etiquetas reservadas, ARIMA con huecos y causalidad, y coordinación de 19 sistemas por cuatro dominios con trabajadores simulados. El escape CPU rechaza paneles de tamaño de producción. Esta verificación no es una campaña de mercado ni una ejecución real de los 14 cuerpos neuronales en GPU.
