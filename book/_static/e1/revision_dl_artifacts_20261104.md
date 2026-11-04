# Auditoría de los resultados reales de desarrollo DL

Fecha: 2026-11-04 UTC. Repositorio: `/workspace/DeepLearning_202630`. Artefactos: `results/dl_benchmark_20261104`. Resultado: **PASS; no se detectaron discrepancias de integridad**. La auditoría no entrenó modelos, no alteró los scripts firmados y no leyó archivos raw ni conjuntos de prueba.

## Evidencia comprobada

- **201 corridas únicas y completas**: 96 de búsqueda, 15 de piloto y 90 de benchmark de desarrollo. Cada corrida contiene ajuste interno y refit; los 402 ajustes declarados coinciden con los dos conjuntos de curvas por corrida.
- **1.311 hashes SHA-256** de artefactos por corrida coinciden con sus `summary.json`. No hay directorios faltantes, duplicados, incompletos o adicionales en `runs/`.
- **36.792 filas de predicción** y **7.248 filas de curvas** verificadas. Los CSV agregados por fase coinciden con los archivos de cada corrida. RMSE, MSE, MAE y MASE recalculados coinciden con `runs.csv` y los resúmenes individuales.
- Los hashes de los tres scripts de entrenamiento/datos/modelos coinciden con `code_manifest_before_search.json`; la firma común es `74dc4b3f4e43ab88c10e6e7d66331b82cc1e0f41287162504cf692f97790a9fc`. El protocolo inicial, la selección y el resumen del piloto coinciden con sus hashes congelados. Las marcas de tiempo respetan el orden registro → búsqueda → selección → piloto → protocolo final → benchmark.

## Datos, cortes y transformaciones

Se leyeron solamente los dos derivados de desarrollo autorizados, con 731 días de calendario UTC entre 2023-01-01 y 2024-12-31:

| Archivo | SHA-256 |
| --- | --- |
| `results/eda_development/daily_market_development.csv` | `bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171` |
| `results/eda_development/daily_volatility_development.csv` | `20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a` |

La carga vuelve a comprobar sigma_7 como la desviación estándar poblacional de siete retornos logarítmicos diarios, conserva el día incompleto 2023-03-24 y no comprime el calendario. Se utilizan exclusivamente cinco canales OHLCV.

| Historia D | Entrenamiento externo | Último objetivo permitido | Entrenamiento interno | Validación interna | Último día de entrada del scaler externo |
| --- | ---: | --- | ---: | ---: | --- |
| 7 | 343 | 2023-12-24 | 251 | 85 | 2023-12-23 |
| 14 | 322 | 2023-12-17 | 230 | 78 | 2023-12-16 |
| 21 | 301 | 2023-12-10 | 209 | 71 | 2023-12-09 |
| 28 | 280 | 2023-12-03 | 188 | 64 | 2023-12-02 |

Para las 201 corridas se reconstruyeron los índices y los parámetros de preprocesamiento. Coinciden exactamente con los guardados. Todos los objetivos usados para entrenamiento y todos los días usados para ajustar scalers pertenecen a **2023**. La validación interna comienza el 2023-10-01 y respeta el embargo D. Se ajustan log(OHLC), log1p(volumen), media y desviación por canal sobre días de entrada únicos; el objetivo se divide por su media positiva de entrenamiento sin centrar. No se emplean días de 2024 para ajustar esas transformaciones.

La búsqueda evalúa los 182 días de **2024-01-01 a 2024-06-30**. El piloto y el benchmark evalúan los 184 días de **2024-07-01 a 2024-12-31**. Para cada predicción se verificó que la última entrada es el día anterior al objetivo y que el objetivo se declara disponible al día siguiente. La persistencia coincide con sigma_7 desplazada un día en el calendario completo. La escala MASE usa exclusivamente pares de objetivos adyacentes válidos dentro del entrenamiento.

## Selección y checkpoints

Se reprodujo la regla de selección a partir de `runs.csv`: mismo presupuesto por familia, cuatro configuraciones por cada una de cuatro historias y dos semillas por configuración. La historia común seleccionada es **D=7**. Las configuraciones congeladas son MLP ancho16/lr0,001; LSTM ancho32/lr0,003; CNN1D ancho16/lr0,001. Se conservaron en el piloto y en las 30 semillas nuevas del benchmark por familia. Las épocas de refit coinciden con las épocas elegidas por la validación interna de 2023.

Se cargó un checkpoint guardado de cada familia, semilla 1001, con `weights_only=True` y CPU; se ejecutó solamente inferencia. Las **184 predicciones de cada modelo coinciden exactamente** con sus CSV guardados, con diferencia absoluta máxima **0**. Los parámetros entrenables son **593 MLP**, **5.025 LSTM** y **1.057 CNN1D**.

## Runtime y reserva de prueba

Los registros declaran Python3.12.14, PyTorch2.15.0+cpu, NumPy2.5.4, pandas3.1.0, SciPy1.18.1, statsmodels0.15.0, CPU, algoritmos deterministas y un hilo por worker. La búsqueda usa concurrencia3; piloto y benchmark, concurrencia1. No hay GPU. La auditoría reprodujo los checkpoints en ese entorno.

`test_files_opened=0` figura en los manifiestos, ejecución, selección, protocolos, piloto y las 201 configuraciones. El cargador firmado abre únicamente los dos CSV de desarrollo. **La auditoría tampoco abrió archivos de prueba.** Esta es evidencia de código y artefactos; los contadores son metadatos del programa y no equivalen a un registro externo de auditoría de todas las operaciones del sistema operativo.

## Limitaciones que deben quedar visibles en el libro

1. **Es evidencia de desarrollo.** No hay evaluación reservada de 2025–2026 ni confirmación externa de generalización. Los resultados corresponden a ajuste estático en 2023, sin actualización diaria; cambiar ese diseño requiere otro protocolo.
2. **La potencia planificada no se alcanzó para el efecto de planificación.** El piloto estimó R necesarios de 1.212 MLP, 172 LSTM y 724 CNN1D para potencia0,80 con la aproximación registrada. Se aplicó el límite previamente registrado de R=30. Las potencias aproximadas en ese margen son 0,0142; 0,1034 y 0,0211, respectivamente. Esto no invalida que puedan observarse efectos mucho mayores; impide presentar el diseño como dimensionado al80% para la mejora pequeña planificada.
3. **El piloto y el benchmark comparten las fechas de calibración.** Sus semillas de entrenamiento son distintas y las configuraciones no se vuelven a seleccionar, pero las 30 semillas no aportan 30 calendarios independientes. Los intervalos entre semillas describen la aleatoriedad de entrenamiento condicionada a esas fechas. No prueban por sí solos incertidumbre temporal o generalización a futuros periodos.
4. **Persistencia sigue siendo una referencia más fuerte que los DL ejecutados.** En calibración, RMSE de persistencia=0,00482248; medias de las30 semillas: LSTM0,00890346, MLP0,01599031 y CNN1D0,01758669. No hay evidencia en estas corridas de que DL supere persistencia. La familia operativa fijada por búsqueda sigue siendo MLP; el menor RMSE descriptivo posterior de LSTM no autoriza sustituirla retroactivamente como si hubiera sido elegida de antemano.
5. La búsqueda con dos semillas y cuatro configuraciones por familia es limitada, aunque es simétrica y reproducible. Los costes son medidas de esta CPU y de las concurrencias registradas; no son estimaciones de despliegue GPU.

**Bloqueadores de integridad detectados: ninguno.** Estas limitaciones condicionan las afirmaciones científicas, no requieren modificar retroactivamente artefactos ni reabrir prueba.

Salida verificable: `/tmp/dl_actual_artifact_audit.json`. Script de auditoría: `/tmp/audit_dl_actual_artifacts.py`.
