# Primera ejecución DL de desarrollo — 04/11/2026

Se ejecutaron MLP, LSTM y CNN1D poco profunda sobre OHLCV diario BTCUSDT, objetivo σ7 no anualizado, horizonte de un día. Entrenamiento exclusivamente de 2023; test reservado de 2025–2026 sin abrir. Esta ejecución no cierra el respaldo bibliográfico aplicado pendiente para MLP/CNN1D ni constituye confirmación fuera del desarrollo.

`code_manifest_before_search.json` enlaza protocolo y tres scripts firmados con el commit previo a los ajustes `3c3c481b8b3f543064ffa7d616ef6c739cb2a783`. `events.jsonl` conserva registro, búsqueda, selección, piloto, dimensionamiento y comparación con semillas nuevas. El registro es posterior al EDA histórico.

| Fase | Fechas evaluadas | Corridas | Semillas por configuración |
|---|---|---:|---:|
| Búsqueda: cuatro D y cuatro configuraciones por familia | Enero–junio de 2024 | 96 | 2 |
| Piloto con configuración seleccionada | Julio–diciembre de 2024 | 15 | 5 |
| Comparación con semillas nuevas | Las mismas 184 fechas de julio–diciembre | 90 | 30 |

Cada corrida comprende selección interna de duración y reajuste desde inicialización nueva sobre el entrenamiento permitido de 2023: **201 corridas, 402 ajustes**. Sin actualización diaria de pesos en 2024. La purga omite D resultados diarios antes del corte; las transformaciones usan solo el entrenamiento correspondiente. No se imputan huecos.

Se eligió D=7. MLP (H=16, lr=0,001): 593 parámetros; LSTM unidireccional (H=32, lr=0,003): 5.025; CNN1D causal de dos capas (H=16, lr=0,001): 1.057. Referencia operacional elegida en enero–junio: **MLP**; líder neuronal descriptivo posterior: **LSTM**. No se reemplazó retrospectivamente la identidad de referencia. RMSE medio julio–diciembre: MLP 0,015990; LSTM 0,008903; CNN1D 0,017587; persistencia 0,004822. Las tres redes quedaron por detrás de persistencia.

El piloto dio un máximo de **1.212** semillas para detectar contra cero el efecto planificado de 5 % del RMSE de persistencia, con aproximación y multiplicidad registradas. Se aplicó el límite de **30** del primer ciclo: no se alcanzó la potencia planificada para esa pregunta. Ese cálculo no demuestra mejora mayor que 5 % ni fija margen entre redes o frente a modelos avanzados.

## Archivos

- `runs.csv`: configuración, semilla, métricas y tiempos de las 201 corridas.
- `predictions/`: predicciones individuales de cada fase, en unidades originales.
- `analysis/`: métricas/IC, contrastes, diagnósticos, bootstrap temporal y cuatro figuras; el README explica su alcance.
- `runtime.json`, `data_audit.json`, `protocol_initial.json`, `selected_configuration.json`, `pilot_summary.json`, `protocol_final.json`: entorno, fuentes y decisiones. El protocolo final congela la comparación **de desarrollo**, no el test.
- `run_artifacts.tar.gz`: **1.512 archivos** con curvas, configuración, escaladores, máscaras, predicciones y resúmenes; incluye **105 checkpoints** de piloto y comparación. La búsqueda conserva registros y predicciones sin pesos. `run_artifacts_manifest.json` contiene SHA-256 de cada archivo y del comprimido. `python scripts/package_dl_benchmark.py` empaqueta sin entrenar.

Los directorios individuales `runs/` se conservan localmente y se excluyen de Git porque su contenido completo está en el archivo comprimido. Restauración en un checkout nuevo:

```bash
tar -xzf results/dl_benchmark_20261104/run_artifacts.tar.gz -C results/dl_benchmark_20261104
```

## Reproducción y alcance

El entorno y los comandos se documentan en `BUILD.md`. El entrenador admite únicamente dos CSV diarios de desarrollo registrados, verifica hashes y firma de código/entorno y rechaza mezclar resultados existentes con otra firma. Las semillas controlan aleatorización en el entorno registrado; no garantizan identidad numérica entre plataformas.

El análisis usa predicciones guardadas, sin ajustes nuevos ni lectura de mercado bruto:

```bash
python scripts/analyze_dl_benchmark.py --results-dir results/dl_benchmark_20261104
```

Los IC t entre semillas condicionan al mismo calendario. El bootstrap circular de fechas (2.000 réplicas, bloque principal 14 días, sensibilidad 7/28) condiciona a los modelos guardados: no remuestrea semillas ni reentrena. Ajustar p-valores no convierte IC marginales en simultáneos. Los contrastes entre semillas, pérdidas temporales y residuos responden preguntas distintas.

Persistencia usa σ7(d−1), disponible al origen y calculada con ocho cierres históricos. Las redes D=7 observan siete días OHLCV: hay una diferencia de una observación histórica para calcular esa referencia de contexto. Las tres redes sí comparten exactamente la misma información entre ellas.
