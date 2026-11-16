# Reproducción real desde datos: primera implementación

El programa `scripts/e2_reproduce_from_raw.py` implementa una cadena nueva de
**78 ajustes neuronales y tres reconstrucciones clásicas**, calibración, freeze,
admisión de los veinte archivos crudos, pronósticos de608fechas, A2 y análisis
primario sobre esos pronósticos nuevos. **Aquí no se ha ejecutado ningún ajuste.**
La comprobación más reciente está en `v2/readiness_cap12000/readiness.json` y el
plan ejecutable vigente es `v2/plan.json`. Los artefactos de preparación v1 se
conservan con su fuente y sus fallos de preflight; no son resultados científicos.

## Comandos

Preparar otro plan, sin ajustar, en una carpeta nueva:

```sh
python scripts/e2_reproduce_from_raw.py --prepare-plan --output NUEVO_PLAN --python PYTHON_CON_TORCH --raw-root RUTA_RAW --author-root RUTA_AUTORES --cap-seconds 12000
```

Comprobar un plan existente, sin ajustar:

```sh
python scripts/e2_reproduce_from_raw.py --readiness --plan RUTA_PLAN/plan.json --output NUEVA_COMPROBACION --python PYTHON_CON_TORCH --raw-root RUTA_RAW --author-root RUTA_AUTORES --cap-seconds 12000
```

Ejecutar después en un runtime CUDA real y compatible:

```sh
python scripts/e2_reproduce_from_raw.py --run --plan RUTA_PLAN/plan.json --output NUEVA_REPRODUCCION --python PYTHON_CUDA --raw-root RUTA_RAW --author-root RUTA_AUTORES --cap-seconds 12000
```

`RUTA_RAW` contiene `BTCUSDT/1m/monthly/BTCUSDT-1m-AAAA-MM.zip`; son exactamente
los veinte archivos fijados por el manifiesto original. `RUTA_AUTORES` contiene
los seis árboles de autores y sus76archivos verificados. `--source-root` permite
indicar el árbol local que conserva scripts, experimentos y los dos CSV diarios
de desarrollo fijados por hash. No se requiere una sesión Colab ni se descarga
ningún archivo durante la ejecución.

La comprobación de datos se ejecuta en `--python`, bajo una reserva corta propia
del diario vigente. La ejecución completa reserva su límite global por sí
misma: **no envolver estos comandos en otro contador**. El saldo real se vuelve
a consultar; el límite no amplía las84horas ya autorizadas ni garantiza que todos
los trabajos quepan. Si falta CUDA, datos, fuentes, piloto o saldo, se detiene
antes del primer ajuste. Nunca habilita `allow_cpu_for_tests` para entrenar.

## APIs y resultados

- Neuronales: `e2_reduced_final.neural_final_worker`, que llama al optimizador
  durable `fit_reduced_job` con las recetas y épocas seleccionadas originales.
  Produce pesos, escalador train-only y calibrador de53fechas por semilla.
- Clásicos: `classical_worker(mode="classical-final")`, con lag1, periodo7 y
  ARIMA(2,0,1), sin buscar otros órdenes. Los coeficientes ARIMA permanecen fijos
  durante el recorrido causal posterior.
- Congelación: `freeze_final` verifica81predictores y sus calibradores. Reutiliza
  el piloto histórico auténtico y fijado por hash; no lo presenta como un piloto
  recién entrenado. No se lee numéricamente test antes de esta puerta.
- Test: `score_final`, con archivos locales y `allow_download=False`. La ruta
  explícita al derivado evita su fallback implícito a GitHub. Produce nuevos
  `score/predictions_*.json`, `benchmark_A2.csv` y análisis4999; después se genera
  `benchmark_analysis_20000.json` exclusivamente desde esos nuevos pronósticos
  y los cierres derivados del raw admitido.

Los resultados quedan en una carpeta nueva. Cada proceso registra sus fuentes,
receta, entorno, tiempos de pared y, en CUDA, máximo asignado aislado tras reset.
Los registros durables originales conservan los recursos por fase. El tiempo de
scoring/análisis es un sobre compartido; no se reparte ficticiamente entre
modelos ni se suma otra vez a sus componentes. No se inventan energía, dinero,
memoria CPU o coste histórico integral que no se haya medido.

`--resume` sólo omite trabajos completos cuyos artefactos y bindings vuelven a
verificarse. Un trabajo a medias requiere reconciliación explícita: no reinicia
su optimizador, no lo reemplaza por otra semilla y no oculta su coste. El worker
interno exige la reserva activa del padre y no puede iniciar ajustes directamente
sin presupuesto.

## Alcance que sigue pendiente

Esta ejecución reproduce **las recetas ganadoras fijas**. No repite el HPO de
234oportunidades/78configuraciones. El API real para repetirlo sería
`fit_reduced_job(train, validation, fixed_epochs=None)`, con sus tres particiones,
pero esa fase no está implementada como opción ejecutable aquí ni se declara
realizada. La conciliación180/231de costes es recuperación de metadatos.

Tampoco repite automáticamente controles con ajustes propios, todas las pruebas
de robustez/explicabilidad o las18ablaciones. Por ello mantiene
`full_original_campaign_passed=false`, incluso si termina fit→cal→raw→test→A2.
Las arquitecturas definidas no requieren NAS. Ningún estado de readiness se
presenta como una ejecución completa o como recuperación de cegamiento histórico.
