# Evaluación retrospectiva de la referencia aditiva v2

La referencia conserva la CNN causal, LSTM y atención temporal de v2, el checkpoint de la época 38 y su escalador de 532 ejemplos. Este procedimiento aplica ese predictor a las 608 fechas del calendario histórico de prueba de E2 (2025-01-01 a 2026-08-31).

No entrena, selecciona épocas, ajusta escaladores, iguala sesgos ni recalibra. La elección de esta referencia procede de resultados de desarrollo ya expuestos. El resultado es descriptivo: los modelos neuronales de E2 tienen seis semillas y 655 objetivos de entrenamiento, frente a una semilla y 532 objetivos de esta referencia; tampoco se han igualado los presupuestos de ajuste. No acredita R1 ni R3.

## Uso en Colab

El lanzador verifica los bytes del código, prepara o reutiliza el entorno aislado de v2 y ejecuta `worker.py` en un subproceso sin credenciales. El secreto `GITHUB_TOKEN` permanece en el coordinador y necesita permiso de escritura de contenidos en `jassan-bit/DeepLearning_202630`.

Antes de abrir los miembros de prueba del NPZ, el trabajador debe reproducir las 116 predicciones de validación publicadas, con tolerancia absoluta de `1e-7` en sigma7 original. Los pesos se cargan con `weights_only=True` después de comprobar su SHA256, identidad de corrida, plan, datos, fuentes, variante y época.

Los datos de prueba se transforman con el escalador guardado de E3. No se usan los tensores escalados de E2. La salida del modelo ya está en sigma7 original, sin una segunda multiplicación por la escala del objetivo.

El coordinador fija el descriptor antes de ejecutar y guarda `predictions.csv`, `comparison.csv` y `summary.json` en la rama `e3-evaluacion-aditiva-v2`, prefijo `runs/evaluacion_aditiva_v2`. Conserva los pilotos anteriores y se detiene si encuentra un descriptor distinto. Si la evaluación ya está publicada, recupera ese resultado.

## Fuentes inmutables

- Código original: `cf891213f1acb6967aef5e0857fa0ac63768f836`.
- Resultados v2: `1ca029e869a517cb8de704e573c78ca1e0cbad3c`.
- Checkpoint aditivo: SHA256 `65641a377ab3e6051081d8a8d15f55e7eece3765b305a0029b032418c8095916`.
- NPZ auditado: commit `af3f1223ad5899ed63016d2af2d09bae8470ab00`, SHA256 `186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683`.
- Partición y tabla A2 de E2: `f65b14d38e38f8d5fd80c717d71f2abd9ea708e2`.

## Verificación de preparación

La comprobación local del checkpoint real sobre las 116 fechas de validación, con PyTorch 2.8.0 en CPU y NumPy 2.3.5, dio un error absoluto máximo de aproximadamente `1.24e-8` respecto a las predicciones de GPU publicadas. Esa comprobación no abrió los miembros de evaluación. Las pruebas de evaluación completa usan datos artificiales; la ejecución real de las 608 fechas corresponde al usuario en Colab.

Pasaron 13 pruebas locales: cuatro del trabajador con datos artificiales y nueve del coordinador con GitHub simulado. Incluyen acceso restringido a miembros NPZ, conservación de resultados, recuperación después de fallos de publicación, repetición de una evaluación ya publicada, conflictos de escritura y errores de validación HTTP con credenciales redactadas. Desde la raíz del repositorio y con PyTorch 2.8.0/NumPy 2.3.5 instalados:

```bash
python -m unittest discover -s notebooks/e3_evaluacion_aditiva_v2 -p 'test_*.py' -v
```

## Corrección del guardado HTTP 422

Una ejecución en Colab recibió HTTP 422 al intentar crear el primer commit del descriptor, antes de ejecutar el trabajador. El estado HTTP por sí solo no identifica la causa exacta. La publicación corregida usa un padre existente de la rama dispersa de resultados v2 y declara la identidad automática `E3 development storage`, igual que el transporte Git anterior. La creación del árbol sigue incluyendo solamente los archivos de esta evaluación, sin copiar el árbol principal. Las actualizaciones posteriores conservan el padre esperado y `force=False`.

Se añaden los campos acotados `message`, `resource`, `field` y `code` de los errores de validación, con redacción de credenciales. La celda corregida usa `launcher_v2/` para conservar el código descargado anteriormente, junto con los mismos `inputs/`, `output/` y descriptor científico. El worker y su SHA256 permanecen sin cambios.

Este código fue preparado con asistencia de IA. Debe declararse esa asistencia en el informe, junto con la responsabilidad del estudiante de comprender y defender su implementación.
