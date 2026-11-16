# Piloto implementado: CNN-LSTM con atención, innovación v2

**Software preparado; sin ejecución de mercado de v2 al publicarlo.**
La [ficha de diseño](../e3_ficha_cnn_lstm_atencion_v2_20261116/README.md)
se conserva como corte anterior. Este directorio contiene su
[plan operativo](plan.json), con el hash de esa ficha y los mismos componentes
científicos. El notebook se abre en
[Colab](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable3_innovacion_v2_colab.ipynb).

## Qué se conserva y qué se implementa

Se conserva la base **CNN causal → LSTM unidireccional → atención temporal**
de M00, con **1210 parámetros entrenables** por red. M00 mantiene la atención.
La lectura escalar final y las cabezas usan FP64; el codificador y la atención
usan FP32. Las constantes de cuadratura no se entrenan.

Se implementan tres recetas con inicialización fresca común:

| Receta | Predicción y pérdida |
|---|---|
| Candidato `innovation_density_mean` | Media de σ₇ por cuadratura positiva; MSE normalizada y NLL auxiliar de la magnitud, peso 0.1. |
| Ablación `innovation_mean_only` | Misma cabeza; sólo MSE normalizada. |
| Control `additive_positive_head` | Piso más incremento positivo; sólo MSE normalizada. |

Los otros tres controles son el piso solo, el piso más incremento medio
calculado en ajuste y el **M00 v1 conservado y proyectado**. Así, el candidato
se compara con cinco controles. Las dos cabezas probabilísticas comparten
exactamente el bias inicial; el aditivo puede necesitar otro bias para igualar
el incremento medio inicial. El ajuste inicial usa sólo la partición de ajuste
y consume el mismo presupuesto.

Las implementaciones están en
[e3_innovation_model.py](../../src/e3_innovation_model.py) y
[e3_innovation.py](../../src/e3_innovation.py). Se escribieron con asistencia de
IA a partir de la formulación documentada. La revisión de novedad y la defensa
del código por el autor siguen pendientes; estas pruebas no acreditan R1/R3.

## Ejecución en Colab

1. Abrir el notebook nuevo y seleccionar GPU; se recomienda la A100.
2. Habilitar el secreto `GITHUB_TOKEN` con acceso **Contents: Read and write**
   a este repositorio, o introducirlo en el campo oculto cuando se solicite.
3. Ejecutar las celdas en orden. Se verifica el código fijado por commit y
   SHA-256, se prepara Python 3.12.12 / Torch 2.8.0+cu126 / NumPy 2.3.5 /
   SciPy 1.17.0 y se ejecuta la comprobación matemática sintética.
4. Recuperar o inicializar la corrida. Esta fase ya reserva y consume cómputo.
5. Ejecutar la celda de entrenamiento y revisar la tabla, curvas y decisión.

Los resultados se guardan en
[`e3-desarrollo-resultados-v2/runs/innovacion_v2`](https://github.com/jassan-bit/DeepLearning_202630/tree/e3-desarrollo-resultados-v2/runs/innovacion_v2)
cuando se ejecute el notebook. La rama se crea durante la primera reserva,
no se presenta como una corrida ya realizada. Una reconexión vuelve a abrir
este mismo notebook y ejecuta las celdas en orden para restaurar la misma corrida.

## Datos y límites

El NPZ permanece fijado al commit y SHA-256 del plan. Sólo se leen
`training_raw`, `training_y_original` y `training_dates`: **532 ejemplos de
ajuste, siete purgados y 116 de validación interna expuesta**. No se usan los
arreglos de calibración ni evaluación. La etiqueta de magnitud U no entra como
predictor. Una etiqueta que contradiga el piso más allá de la tolerancia
documentada detiene la corrida, sin descartar filas.

Una semilla, AdamW, hasta 40 épocas por red, paciencia 8, sin AMP ni TF32.
Las épocas se intercalan una por red activa. Los bloques pueden reunir varios
ciclos para reducir arranques de procesos y publicaciones; eso no cambia el
orden científico ni el número máximo de épocas.

El tope adicional es **600 segundos conjuntos**, no por red ni por sesión.
Se conserva el cargo v1 de **369.4779596389991 segundos** y el límite global
de **1200 segundos**. El estado remoto canónico vincula una sola corrida al
plan y al snapshot v1 terminal; cambiar el directorio o la VM no abre otro saldo.
Se comprueba el presupuesto entre operaciones; cualquier exceso por una
operación indivisible o por el cierre se registra y detiene nuevos ajustes.
La instalación y las descargas añaden tiempo de pared fuera del contador.
Los cargos aplicados durante una reserva activa también consumen el saldo.

Antes de ajustar estadísticas, biases o pesos, la reserva ACTIVE debe estar
confirmada en GitHub. Antes del siguiente bloque debe confirmarse el snapshot
CLOSED. Un conflicto, una credencial inválida o una publicación sin confirmar
detienen el avance. La restauración de una reserva interrumpida conserva su
fecha y aplica el cargo conservador previsto; puede agotar el saldo.

Los snapshots conservan estados del optimizador y generadores aleatorios,
pesos, curvas, predicciones, admisión de datos y registro de cómputo. Los hashes
se comprueban antes de cargar checkpoints. La credencial permanece en el
coordinador y se excluye del entorno del trabajador y de los snapshots.

## Interpretación

La heurística de continuación exige **al menos 2 % menos RMSE que el mejor de
los cinco controles**, con las tres recetas neuronales completas. Si alguna
queda interrumpida por presupuesto, el piloto es incompleto y no supera la
heurística. No se amplía la corrida para buscar una victoria.

Se reportan MAE, sesgo, errores de alta volatilidad y conteo bajo el piso. Las
cabezas probabilísticas añaden NLL de U y cobertura nominal del 90 %, sin
afirmar calibración. No se producen p-valores ni conclusiones confirmatorias
frente al Top 10. Antes de esa comparación siguen pendientes la revisión de
novedad, potencia y protocolo congelado.

## Verificación de software

Las pruebas usan **datos sintéticos y transportes de Git de prueba**, sin
ajustes o inferencias de mercado. Cubren datos, pérdidas, gradientes, precisión,
inicialización, reanudación, registro conjunto, conflictos y aislamiento de
credenciales. La ejecución CUDA real queda para el notebook de Colab.

```bash
python -m pip install -r requirements-e3-design-cpu.lock.txt
python -m pytest tests/test_innovation_model.py tests/test_innovation_training.py \
  tests/test_innovation_github.py tests/test_innovation_notebook.py -q
python scripts/e3_verify_innovation_design.py --check
```
