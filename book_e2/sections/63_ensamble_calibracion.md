# 63. Referencia empírica mediante un ensemble ajustado en calibración

Se construyó **una única combinación convexa de los dieciséis sistemas**, usando solamente las 53 fechas de calibración del 08/11–30/12/2024 para aprender sus pesos. Los pronósticos originales y los resultados de A2 se conservan. El [plan](../_static/e2_calibration_ensemble_20261114/analysis_plan.json) fue versionado en el [commit 8d9cc8c](https://github.com/jassan-bit/DeepLearning_202630/commit/8d9cc8c3397242a07378fb1a678196dab36744da) antes de este ajuste, aunque el periodo de prueba y sus resultados originales ya eran conocidos. Es un suplemento retrospectivo.

## Construcción y separación de datos

Los pesos son comunes a las seis semillas, no negativos y suman uno. Los comparadores deterministas repiten su trayectoria al emparejarse con cada semilla neuronal; esto no los convierte en seis ejecuciones independientes. Se minimizó la MSE de calibración media de las seis trayectorias del ensemble, normalizada por la media finita de σ₇ del desarrollo 2023–2024, con penalización fija 1e-4 hacia pesos uniformes. Se realizó una sola optimización SLSQP con gradiente analítico, tolerancia 1e-12 y máximo 1000 iteraciones; no hubo HPO, prueba de múltiples penalizaciones ni selección del ensemble según su RMSE de prueba.

Se efectuó **un ajuste estadístico de los pesos de combinación en CPU**, con duración observada de aproximadamente 2 segundos. No se reajustó ninguna de las trece redes ni de los tres comparadores, ni se cargaron sus pesos para realizar nuevas inferencias. Los pronósticos del ensemble son combinaciones de las trayectorias ya guardadas. Los calibradores originales permanecen intactos. Estos costes CPU se documentan separados de los cargos históricos A100.

Ocho pesos son superiores a 1e-9; corresponden a FCN, ResNet, Inception, SAE, TimesNet, TimeMixer, iTransformer y Persistence. Los [dieciséis pesos y sus ceros numéricos](../_static/e2_calibration_ensemble_20261114/weights.csv) se publican completos. La optimización terminó en 43 iteraciones. La [auditoría independiente de factibilidad y condiciones KKT](../_static/e2_calibration_ensemble_20261114/independent_fit_audit.json) verifica suma 1, no negatividad, residuo activo máximo 4,73e-7 y holgura inactiva positiva. Las diferencias en validación de 2024 explican por qué un modelo con buena posición en la A2 puede recibir peso cero aquí; no se modificaron pesos para favorecer su posición en 2025–2026.

## Resultado en las mismas 608 fechas

La métrica conserva el estimando de A2: **media de seis RMSE individuales**, una por ensemble emparejado con semilla. No se calcula la RMSE de promediar las seis predicciones.

| Sistema | RMSE media | IC95 temporal marginal, bloques 14 días |
|---|---:|---|
| Ensemble ajustado en calibración | 0,003689532 | [0,003043221;0,004419618] |
| SAE original | 0,003803430 | [0,003176916;0,004566666] |
| Persistence original | 0,004085596 | [0,003313647;0,004960634] |

La RMSE de calibración del ensemble es 0,003186257. Su error en prueba es mayor, lo que se conserva como resultado; no se volvieron a aprender sus pesos. El SAE continúa siendo el mejor modelo individual observado del catálogo original. La combinación complementaria no se añade a A2 ni modifica el catálogo obligatorio de 16 sistemas.

| Referencia | Ganancia RMSE de la combinación | IC95 marginal de la ganancia | p Holm (2 contrastes) |
|---|---:|---|---:|
| SAE | 0,000113898 | [−0,000213924;0,000426550] | 0,243776 |
| Persistence | 0,000396064 | [0,000022275;0,000817232] | 0,063794 |

Son dos contrastes posteriores de ganancia mayor que cero, con 10.000 remuestreos pareados de semillas y bloques circulares de 14 días, y Holm sobre ambos. **Ninguno rechaza a 0,05 después de Holm.** Los IC son marginales, no simultáneos. No se contrastó aquí superar el margen práctico δ ni se sustituyeron los veinte contrastes originales. Se conservan las sensibilidades a 7 y 28 días en el [resumen completo](../_static/e2_calibration_ensemble_20261114/summary.json). Los pesos aprendidos permanecen fijos durante el remuestreo: la inferencia está condicionada a esa calibración y no incorpora la incertidumbre de volver a estimarlos en otra muestra.

## Distancia frente a una referencia alcanzable e incertidumbre

La construcción proporciona una **referencia empírica alcanzable bajo este procedimiento**, con RMSE puntual menor que la del SAE y Persistence. Usando esa referencia como proxy operativa, la fórmula orientada a pérdida es:

$$
\gamma_{emp}=\frac{RMSE_{SAE}-RMSE_{ensemble}}{RMSE_{Persistence}-RMSE_{ensemble}}=0,287574.
$$

Este número expresa la fracción puntual del margen Persistence–ensemble que aún separa al SAE de esta combinación concreta. **No identifica el riesgo Bayes ni un techo óptimo sobre todos los predictores.**

La incertidumbre del cociente es grande: el intervalo percentil de su valor algebraico es **[−1,49210;2,37128]** con bloques de 14 días. Sólo 6.588 de 10.000 réplicas conservan el orden `ensemble < SAE < Persistence`. Se conservaron todas las réplicas algebraicamente definidas, incluidas las que cambian el orden o el signo del denominador; no se truncó el intervalo a [0,1] ni se condicionó silenciosamente a los resultados favorables. Por ello no se interpreta este intervalo como una fracción poblacional estable ni se concluye que quede exactamente 28,8 % de margen real.

El ensemble uniforme de [§57](57_errores_techo.md) permanece como resultado anterior negativo, con otra identidad. Esta construcción de pesos es nueva y tiene su protocolo propio; no se elige retrospectivamente entre ambas para declarar un techo demostrado. Se mantienen el calendario conocido desde E1, la ausencia de δg previo y la incertidumbre sobre generalización futura.

## Artefactos y reproducción

- [Plan registrado](../_static/e2_calibration_ensemble_20261114/analysis_plan.json).
- [Pesos aprendidos](../_static/e2_calibration_ensemble_20261114/weights.csv) y [pronósticos derivados](../_static/e2_calibration_ensemble_20261114/ensemble_saved_forecasts.csv).
- [Dos contrastes e intervalos](../_static/e2_calibration_ensemble_20261114/contrasts.csv).
- [Resumen completo y sensibilidad](../_static/e2_calibration_ensemble_20261114/summary.json) y [manifiesto](../_static/e2_calibration_ensemble_20261114/manifest.json).

```bash
python scripts/e2_calibration_ensemble.py \
  --input book_e2/_static/e2_final_audit_inputs_20261114/e2_f2a_input.json \
  --development book_e2/_static/e2_final_audit_inputs_20261114/daily_volatility_development.csv \
  --plan experiments/e2_calibration_ensemble_20261114/analysis_plan.json \
  --registration 8d9cc8c3397242a07378fb1a678196dab36744da \
  --output work/calibration_ensemble_repetition
```

La repetición vuelve a ajustar los pesos estadísticos con la misma calibración. La auditoría documental de [§61](61_reproduccion_consolidada.md) comprueba los artefactos guardados sin ese ajuste; ambos procedimientos tienen alcances distintos.
