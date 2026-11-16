# Piloto v2 listo para Colab: misma CNN-LSTM con atención

La [ficha anterior](18_ficha_cnn_lstm_atencion_v2.md) ya tiene una implementación
de desarrollo y un
[notebook para Colab](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable3_innovacion_v2_colab.ipynb).
**Al publicar este capítulo no se ha ejecutado el piloto de mercado de v2.**
El [plan operativo y guía](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_innovacion_v2_20261116)
conservan la formulación y sus límites.

## Modelo y controles

Se mantiene la base **CNN causal → LSTM unidireccional → atención temporal de
M00**, con 1210 parámetros por red. La modificación está en la cabeza y la
pérdida para σ₇. La atención continúa aprendiendo sus ponderaciones temporales.

El candidato combina MSE de σ₇ con NLL auxiliar de la magnitud de innovación.
Se compara con cinco controles: la misma cabeza sin NLL, una cabeza positiva
aditiva, piso solo, piso más incremento medio de ajuste y M00 v1 proyectado.
Se entrenan tres redes nuevas; las otras referencias son analíticas o conservan
los pesos ya seleccionados en v1.

## Cómo ejecutarlo

1. Abrir el notebook nuevo en Colab y seleccionar GPU; se recomienda la A100.
2. Habilitar `GITHUB_TOKEN` con permiso **Contents: Read and write** para el
   repositorio, o introducirlo en el campo oculto.
3. Ejecutar las celdas en orden: código y datos verificados, entorno aislado,
   recuperación o inicialización, entrenamiento y resultados.

El notebook fija el código por commit y SHA-256. Los resultados se conservan
en `e3-desarrollo-resultados-v2/runs/innovacion_v2`, con pesos, optimizadores,
generadores aleatorios, curvas, predicciones y saldo. Una reconexión restaura
la misma corrida. Una reserva activa de otra sesión impide el entrenamiento
simultáneo; una publicación sin confirmar detiene el avance.

## Presupuesto, datos y decisión

El tope adicional es **600 segundos conjuntos**, incluyendo estadísticas de
ajuste, inicialización de biases, entrenamiento, validación y checkpoints.
Se mantienen los **369.48 segundos** registrados en v1 y el límite acumulado
de **1200 segundos**. La instalación y las descargas pueden añadir tiempo de
pared. Cualquier exceso de una operación indivisible se registra y detiene
nuevos ajustes; no se reinicia el contador para reanudar.

Sólo se admiten los tres arreglos originales de entrenamiento: 532 ejemplos de
ajuste, siete purgados y 116 de validación interna ya expuesta. Las particiones
de calibración y evaluación permanecen reservadas. El entrenamiento usa una
semilla, épocas intercaladas, máximo 40 por red y paciencia 8.

La continuación requiere al candidato al menos **2 % menos RMSE que el mejor
de los cinco controles**, con las tres recetas completas. Un piloto incompleto
no supera esa heurística. El umbral no es un margen confirmatorio ni una prueba
de significancia. Los intervalos son nominales bajo la familia asumida.

## Alcance de las pruebas

La implementación se verifica con datos sintéticos: soporte, pérdidas,
gradientes, precisión mixta, inicialización, checkpoints, recuperación y
coordinación del presupuesto remoto. La ejecución real con CUDA se realizará
en Colab. **La novedad, la calibración y la superioridad frente al Top 10
siguen sin demostrar.** La asistencia de IA se declara en la guía y en el código.
