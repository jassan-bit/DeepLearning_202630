# Completar E2 a diez semillas

Esta etapa posterior conserva las seis semillas originales de cada red de E2 y entrena únicamente las semillas 271807–271810 de sus trece redes: 52 ajustes adicionales. Conserva también las tres referencias deterministas. La CNN-LSTM con atención aditiva de E3 ya completó sus diez semillas y se reutilizan sus resultados publicados.

El cuaderno `notebooks/entregable3_e2_extension_colab.ipynb` descarga código fijado por commit y verifica sus SHA256 antes de ejecutarlo. Seleccione una GPU en Colab —los tiempos de referencia corresponden a una A100— y habilite el secreto `GITHUB_TOKEN` con permiso de escritura de contenido en `jassan-bit/DeepLearning_202630`. Ejecute la celda del cuaderno. No utiliza Google Drive.

## Cálculo y conservación

Se mantienen las arquitecturas, los hiperparámetros, las épocas, el escalado y el procedimiento de entrenamiento publicados de E2. Cada ajuste utiliza un proceso aislado y calibra sus intervalos con las 53 observaciones de calibración. Los 52 ajustes y sus calibradores deben quedar confirmados en GitHub antes de generar sus predicciones de prueba. La predicción utiliza lotes de tamaño uno, también para TimesNet.

El límite de esta etapa es 7200 segundos de procesos científicos, incluidos los procesos fallidos. La estimación con los tiempos históricos de A100 es de unos 64 minutos de entrenamiento; la preparación, la evaluación y la publicación añaden tiempo. El cómputo cerrado de E3 conserva sus 948.439625042 segundos y no se reinicia.

El progreso se publica mediante Git nativo en la rama `e2-extension-10-semillas-v1`, bajo `runs/e2_extension_10_semillas_v1`. La publicación y la lectura de confirmación preceden al siguiente proceso. Si falla la publicación tras completar un ajuste, la siguiente ejecución conserva y publica ese resultado antes de continuar. El entrenador original no reanuda un optimizador parcialmente ejecutado: una interrupción dentro de un ajuste conserva el estado y detiene la etapa para revisar ese caso.

## Productos e interpretación

La salida final conserva los 81 predictores de E2 y agrega 52, para un total de 133: diez semillas por red y una trayectoria por referencia determinista. Produce `combined_predictors.json`, `per_seed_metrics.csv`, `comparison.csv`, `summary.json` y sus comprobantes de integridad. La comparación incorpora las diez semillas ya publicadas de E3 sobre las mismas 608 fechas.

La etapa es posterior al conocimiento de ese calendario de prueba. E2 usa 655 ejemplos de entrenamiento y E3 usó 532. Los resultados son descriptivos; esta ejecución no realiza pruebas de hipótesis ni acredita R1 o R3.

## Validación del código

Las pruebas locales verifican los archivos fijados, los trece constructores originales con tensores artificiales, el aislamiento del entorno, el registro de consumo, la publicación Git y la conservación de resultados. Los ajustes de prueba utilizan únicamente datos sintéticos. Las pruebas no ejecutan los 52 entrenamientos reales ni sustituyen la verificación con CUDA en Colab.

Archivos de implementación: `prepare.py`, `worker.py`, `aggregate.py`, `bootstrap.py` y `plan.json`. Las evidencias y pruebas permanecen en este directorio; no se añaden capítulos al libro.
