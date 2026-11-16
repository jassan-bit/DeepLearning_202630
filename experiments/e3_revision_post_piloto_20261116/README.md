# Revisión posterior al primer piloto de E3

Corte: 16 de noviembre de 2026. Este registro conserva un resultado adverso y
una revisión focal posterior. Sigue siendo desarrollo exploratorio; no es un
prerregistro, una búsqueda sistemática terminada ni una confirmación de R1/R3.

## Resultado que motiva la revisión

La corrida `9e40ce379a7b4ee1a3e2fb306071cdcb`, guardada en el commit
[`f7826d1`](https://github.com/jassan-bit/DeepLearning_202630/tree/f7826d120f3a6eb8e9a0952dd9842cd77f152f34/runs/desarrollo_v1),
completó 40 épocas por modelo en una A100. Con la semilla 20261116, `full`
obtuvo RMSE 0.0060453143 y `M00` 0.0053767665 en la validación interna de
116 fechas. El tiempo registrado fue 369.47796 segundos del límite de 1200.
La época se seleccionó con esa misma validación. El registro original permanece
intacto; no se convierte esta diferencia descriptiva en significancia.

## Hallazgo de coherencia

Las siete barras de entrada contienen seis de los siete retornos que componen
la variable objetivo. Su dispersión fija un piso observable para la volatilidad
del día siguiente. Ambas redes lo incumplen en 32 de 116 predicciones. La
descomposición y la prueba de proyección se documentan en [theory.md](theory.md).

El diagnóstico aplica la misma proyección a los dos modelos, sin reentrenarlos.
El piso solo también se presenta como referencia analítica. Estas alternativas
se examinaron después de observar el piloto: son análisis posteriores, no nuevos
benchmarks confirmatorios ni modelos originales. La revisión identifica una
limitación medible; no demuestra una contribución algorítmica nueva.

## Reproducir el diagnóstico

Desde la raíz del repositorio, con NumPy 2.3.5:

```bash
python scripts/e3_audit_development.py --check
python -m pytest tests/test_development_audit.py -q
```

El script verifica el SHA-256 del NPZ y del manifiesto del resultado; comprueba
los CSV y el resumen contra ese manifiesto. Sólo abre `training_raw`,
`training_y_original` y `training_dates`. La auditoría utiliza exclusivamente
las 116 fechas de validación de desarrollo ya expuestas. No carga checkpoints,
no ajusta modelos y no puntúa calibración ni evaluación.

Los insumos y salidas quedan en [audit/](audit/); la matriz de nueve fuentes,
sus alcances de lectura, bibliografía y búsquedas están en [literature/](literature/).
La decisión de continuación se registra en [decision.md](decision.md), y el
recuento acumulado de variantes en
[la bitácora de desarrollo](../e3_desarrollo_v1/design_log.md).

La revisión y el código se elaboraron con asistencia de IA. Las revisiones
internas de agentes no se presentan como verificación humana externa de novedad
ni como reproducción independiente por una persona ajena al código.
