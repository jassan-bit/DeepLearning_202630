# CV efectiva de Filipe y revisión de R² — 04/11/2026

El ciclo original `dl_benchmark_20261104` se conserva completo. Su R² negativo estaba bien calculado: los precios de 2024 estaban fuera del soporte estandarizado de 2023, MLP/CNN sobreestimaban y LSTM predecía casi un nivel constante. Este segundo ciclo compara entrenamiento creciente real y representación relativa, sin modificar la fórmula de R², el objetivo σ7/ddof0/h1 ni los cinco canales OHLCV.

## Particiones realmente utilizadas

`scripts/prepare_dl_r2_cv.py` se ejecutó en el entorno separado NumPy1.26.4/timeseries-cv0.1.5. Invoca `split_train_val_forwardChaining(np.arange(731),7,1,1)`, función de DidierRLopes con coautoría de Filipe R. Ramos. La librería proporciona ventanas y orígenes diarios. El adaptador reúne los orígenes de 2024 en **12 bloques mensuales** y aplica elegibilidad y embargo explícitos: resultado de entrenamiento u≤inicio_del_mes−8.

`folds.json` conserva candidatos de la librería, claves, ventanas, fechas, índices admitidos y hashes. `scripts/train_dl_r2_cv.py` utiliza esos mismos índices para entrenar modelos y escaladores. En cada mes se selecciona duración sobre una validación interna pasada (90 días antes del origen, 83 etiquetas después del embargo) y se reajusta desde cero con todo el entrenamiento permitido. Los pesos quedan fijos durante el mes; cada predicción diaria h1 recibe solo OHLCV anterior a su día de resultado. Se hicieron **12 procedimientos mensuales por trayectoria**, no 366 reajustes diarios ni pronósticos de un mes adelante.

Calendario conservado: 365 días iniciales de 2023; 366 orígenes de validación de 2024; ajuste final con desarrollo cuando se cierre la configuración; test previsto 01/01/2025–31/08/2026, 608 días. Septiembre–octubre de 2026 queda sin asignar al test. Este runner no abrió test ni ejecutó el ajuste final posterior a la CV.

## Comparación controlada

- Variantes `absolute_log` y `relative_window`, ambas con los mismos folds mensuales. La segunda usa log(OHLC)−log(último cierre observable de la propia ventana), más log1p del volumen. Estadísticas aprendidas únicamente con entrenamiento; las escalas relativas ponderan posiciones de ventanas, mientras las absolutas usan días únicos. No se añaden rezagos de volatilidad.
- D=7 y configuraciones V1 fijas: MLP H16/lr0,001/593 parámetros; LSTM H32/lr0,003/5.025; CNN1D H16/lr0,001/1.057. No hubo nuevo HPO.
- Diez semillas comunes 4101–4110 por variante/modelo; semilla de ajuste mensual=base+ordinal_del_mes×100000 y reajuste=semilla_mensual+10000000.
- **60 trayectorias, 720 segmentos modelo–fold, 1.440 ajustes efectivos y 21.960 predicciones**. Se guardaron 720 checkpoints.

El protocolo y código se versionaron en `1f0ed23` antes de ajustar. `protocol_erratum.json` corrige una frase descriptiva heredada que decía «full2023»: los índices efectivos siempre usaron la cohorte mensual creciente. El optimizador no utiliza esa frase; no se alteró el protocolo firmado ni ningún cálculo por la errata.

## Resultados

| Modelo | R² con precios relativos, todo 2024 | R² relativo, julio–diciembre | R² V1, mismo semestre |
|---|---:|---:|---:|
| MLP | 0,615670 | 0,657808 | −2,739982 |
| LSTM | 0,578892 | 0,609298 | −0,128090 |
| CNN1D | 0,592088 | 0,689848 | −3,758601 |
| Persistencia | 0,759571 | 0,669995 | 0,669995 |

Persistencia sigue siendo la mejor referencia anual. CNN1D reduce el RMSE semestral aproximadamente un 3,10 % frente a persistencia; no alcanza el margen del 5 % ni dispone aquí de una prueba temporal confirmatoria. V1 tiene 30 semillas distintas, la CV diez: su comparación en el mismo semestre es descriptiva. Las tres pruebas/Holm comparan únicamente las variantes absoluta y relativa de CV con semillas emparejadas. Los R² anuales se calculan desde las 366 predicciones reunidas, sin promediar R² de folds individuales.

Las configuraciones V1 se eligieron usando enero–junio de 2024 y el EDA ya examinó 2024. Este ciclo, diseñado después de ver los fallos, es **desarrollo exploratorio**, no validación anidada o confirmación externa. Diez semillas son un presupuesto de diagnóstico, sin dimensionamiento de potencia 0,80. La identidad operacional MLP del ciclo previo no se reemplaza retrospectivamente.

## Archivos y reproducción

`runs.csv`, `predictions.csv`, `fold_metrics.csv` y `analysis/` permiten reconstruir métricas, IC marginales por semillas, comparación controlada y figuras. MASE utiliza la escala de entrenamiento de cada mes, registrada por predicción; la persistencia determinista no representa diez entrenamientos. Los tiempos V2 se midieron con tres trabajadores concurrentes y no se comparan directamente con tiempos seriales de V1.

`run_artifacts.tar.gz` contiene **5.940 archivos** por trayectoria/fold; `run_artifacts_manifest.json` contiene sus hashes. Restaurar y comprobar sin entrenar:

```bash
tar -xzf results/dl_r2_cv_20261104/run_artifacts.tar.gz -C results/dl_r2_cv_20261104
python scripts/verify_dl_r2_cv.py
python scripts/analyze_dl_r2_v2.py
python scripts/plot_dl_cv_folds.py
```

Usar el entorno de entrenamiento fijado en `requirements-training.lock.txt`. `BUILD.md` explica los dos entornos, la regeneración independiente de índices y una replicación aislada. No regenerar encima de `folds.json` firmado: su fecha de generación forma parte de su hash; comprobar los índices en otro directorio.
