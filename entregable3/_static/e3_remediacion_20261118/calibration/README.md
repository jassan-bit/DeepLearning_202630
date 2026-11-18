# Diagnóstico retrospectivo de intervalos E3 — 18 de noviembre de 2026

Se calcularon diez cuantiles nuevos para los diez predictores E3 congelados, sin
reentrenar ni cambiar sus pronósticos. La cobertura media sobre los 608 días
históricos fue **94.4901 %**, con anchura media **0.01096344 σ7** e interval score
**0.01536077 σ7** a α = 0.10. La media entre semillas del RMSE puntual permanece
en **0.0029394324 σ7**. La cobertura cae a **86.22 % en 2026Q1** y a **84.53 % en
el grupo de objetivos altos**, por lo que el resultado agregado no acredita
calibración uniforme bajo cambio temporal.

## Alcance y separación de la campaña original

El plan original de `estabilidad_aditiva_v1` prohibía abrir la calibración:
`calibration_members_allowed:false`; su cierre declaró
`calibration_members_opened:false`. Ambos hechos se verifican antes de ejecutar
este análisis y conservan su significado. Este directorio registra un **análisis
posterior separado** que sí abre esas ventanas. No modifica el protocolo, las
hipótesis, los checkpoints, los metadatos del servicio ni el cierre original.

El calendario de evaluación y sus resultados ya eran conocidos al plantearlo.
El archivo `analysis_plan.json` documenta las decisiones de este diagnóstico;
**no es un protocolo prospectivo ni una preinscripción confirmatoria**. La
cobertura se calcula sobre los mismos resultados históricos E3. No resuelve R1,
la evaluación ampliada R3, la reproducción humana independiente ni la
aceptación pública del despliegue. El dashboard conserva su declaración de que
E3 no dispone de intervalo calibrado: estos cuantiles **no están activados**.

## Método y procedencia

- Semillas fijas 271801–271810, sin selección de la más favorable y sin ensemble.
- Diez `best_model_state` seleccionados por la validación original, con 1210
  parámetros y bytes/SHA-256 verificados antes de `torch.load(weights_only=True)`.
- Estadísticas originales estimadas en 532 ventanas de ajuste hasta 2024-06-30;
  116 ventanas de selección desde 2024-07-08 hasta 2024-10-31. Se reconstruyeron
  las ocho matrices originales y sus SHA-256 coinciden con la admisión original.
- Calibración: 53 ventanas, 2024-11-08–2024-12-30. Sus siete días de entrada
  empiezan después del último objetivo de validación. Se ignoran los tensores
  escalados por E2; la transformación aplica exclusivamente el escalador E3.
- Inferencia solo CPU, una ventana por llamada, encoder/atención `float32` y
  readout/escalas `float64`, tal como quedaron guardados. Se comprueba el piso
  observable de varianza y la coherencia del objetivo. **530 forwards** nuevos.
- Evaluación: 608 días, 2025-01-01–2026-08-31, por cada predictor. Se reutilizan
  las 6080 predicciones puntuales guardadas de sus checkpoints correspondientes;
  no se ejecuta ningún forward de evaluación ni se modifica su CSV.
- Para cada predictor, `q` es el **49.º** residuo absoluto de las 53 ventanas:
  `ceil((53+1)*(1−0.10)) = 49`. Se usa el intervalo
  `[max(0,predicción−q), predicción+q]`, comparable con la convención histórica de
  E2. α = 0.10 no se eligió buscando cobertura favorable en evaluación.
- Se reportan cobertura, anchura y score de intervalo propio:
  `(U−L) + (2/α)(L−y)·1[y<L] + (2/α)(y−U)·1[y>U]`.
  Las unidades son σ7 original, sin anualizar; menor score es mejor.

`input_manifest.json` fija quince entradas públicas por commit, ruta, bytes y
SHA-256. Los pesos y las predicciones proceden del commit
`510c3bddfdc9926322e3210ac61ee1f5eeffad2e`; el NPZ procede de
`af3f1223ad5899ed63016d2af2d09bae8470ab00` y su admisión E3 de
`1ca029e869a517cb8de704e573c78ca1e0cbad3c`. Las descargas son solo lecturas y
fallan si difiere cualquier hash. No se escriben secretos ni se necesita GPU.

## Resultados

| Semilla | q (σ7) | Cobertura (%) | Anchura media (σ7) | Interval score (σ7) |
|---:|---:|---:|---:|---:|
| 271801 | 0.00540246 | 94.0789 | 0.01080287 | 0.01519440 |
| 271802 | 0.00539822 | 94.2434 | 0.01079435 | 0.01523761 |
| 271803 | 0.00540951 | 94.4079 | 0.01081713 | 0.01512961 |
| 271804 | 0.00561775 | 94.4079 | 0.01123326 | 0.01553168 |
| 271805 | 0.00539453 | 94.2434 | 0.01078715 | 0.01502375 |
| 271806 | 0.00545629 | 94.4079 | 0.01091060 | 0.01542291 |
| 271807 | 0.00539060 | 94.2434 | 0.01077921 | 0.01528602 |
| 271808 | 0.00544118 | 94.4079 | 0.01088046 | 0.01553853 |
| 271809 | 0.00560353 | 95.2303 | 0.01120576 | 0.01553263 |
| 271810 | 0.00571274 | 95.2303 | 0.01142367 | 0.01571061 |
| **Media** | **0.00548268** | **94.4901** | **0.01096344** | **0.01536077** |

La desviación estándar entre las diez semillas es 0.4047 puntos porcentuales
de cobertura, 0.00023424 σ7 de anchura y 0.00021910 σ7 de score. Es dispersión
de semillas sobre el mismo calendario, no diez réplicas independientes del
mercado. No se construyó un intervalo de ensemble.

El bootstrap circular de bloques de 14 días, 4999 remuestras y RNG 20261118
resamplea los mismos días conjuntamente para las diez semillas. El intervalo
marginal de 95 % de cobertura media es [92.4342 %, 96.3158 %] y el de score es
[0.01313811, 0.01826591] σ7. **Condicionan en los modelos y q ya estimados**:
no incluyen la incertidumbre de estimar q con solo 53 ventanas y no son
intervalos simultáneos ni una garantía fuera de esta muestra.

| Diagnóstico posterior | Días por predictor | Cobertura media (%) |
|---|---:|---:|
| 2025Q1 | 90 | 95.00 |
| 2025Q2 | 91 | 93.41 |
| 2025Q3 | 92 | 97.72 |
| 2025Q4 | 92 | 96.20 |
| 2026Q1 | 90 | **86.22** |
| 2026Q2 | 91 | 97.36 |
| 2026Q3 (julio–agosto) | 62 | 95.81 |
| Objetivo ≤ q25 original de ajuste | 139 | 99.42 |
| q25 < objetivo ≤ q75 original | 383 | 94.93 |
| Objetivo > q75 original | 86 | **84.53** |

Los umbrales de nivel proceden del ajuste original, pero asignar el grupo con
el objetivo realizado utiliza información posterior. Estos grupos describen
fallos históricos; no son reglas disponibles al emitir un pronóstico. Ningún
subgrupo se utilizó para cambiar q o elegir una semilla. Todos los fallos de
cobertura de este intervalo fueron por el extremo superior.

Los objetivos de calibración y evaluación son disjuntos y están ordenados. Hay
solapamiento de historia conocida: la entrada del primer pronóstico de enero
contiene 2024-12-25–31 y la calibración termina el 30 de diciembre. No es una
entrada futura en el origen del pronóstico, pero los objetivos móviles y la
dependencia temporal no acreditan intercambiabilidad. La corrección de rango
finito por sí sola **no garantiza cobertura del 90 %** en esta serie.

## Reproducción y verificación

Entorno usado: Python 3.12.14, NumPy 2.3.5 y PyTorch 2.8.0+cpu; un hilo CPU.
El análisis no hace llamadas de entrenamiento, actualizaciones del escalador,
bisecciones de sesgo, cambios del calibrador previo ni escrituras al servicio.
La admisión, descarga de las entradas faltantes y 530 forwards tardaron
3.77 segundos en este entorno; no es una medición de latencia de producción.

Desde la raíz del repositorio, con esas dependencias disponibles:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python experiments/e3_remediacion_20261118/calibration/run.py \
  --input-dir /tmp/e3_retrospective_calibration_inputs --download \
  --output /tmp/e3_retrospective_calibration_results

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python experiments/e3_remediacion_20261118/calibration/verify_results.py \
  --input-dir /tmp/e3_retrospective_calibration_inputs \
  --results /tmp/e3_retrospective_calibration_results

PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s experiments/e3_remediacion_20261118/calibration -p test_calibration.py -v
```

Las seis pruebas verifican el rango 49 sin interpolación, el rechazo de tamaños
o residuos inválidos, la independencia de q frente a cambios en los objetivos
de evaluación, la penalización del score, la conservación de las predicciones
puntuales y la cronología. `verify_results.py` verifica los quince hashes,
reconstruye ocho matrices originales y sus hashes, comprueba las precisiones
de los checkpoints, reconstruye q y todas las métricas desde los CSV, con
**cero forwards adicionales**. No es una reproducción humana independiente.

Los archivos completos están en `results/`: `per_seed_intervals.csv`,
`calibration_predictions.csv`, `subgroup_diagnostics.csv`, `results.json` y
`verification.json`. Este último fija los hashes de los cuatro archivos de
resultados y del verificador. Los tiempos y la fecha cambian al repetir el
script; las métricas y los CSV constituyen el resultado numérico a contrastar.
