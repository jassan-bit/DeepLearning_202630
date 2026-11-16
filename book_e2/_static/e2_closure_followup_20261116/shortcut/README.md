# Diagnóstico exploratorio de dependencia de entradas

Se evalúan los 78 predictores neuronales congelados sobre las 608 fechas conocidas de prueba, sin entrenamiento, selección ni recalibración. El grupo volumen se eligió como candidato por la deriva temporal previamente documentada; esa deriva no acredita una señal espuria ni fuga.

Antes de nuevos forwards se congelaron código, fuentes, condiciones, donantes y análisis en `study_v1/analysis_plan.json`. El SHA-256 de ese plan es `0bc277124baab88a00d6e281f7c759f4bab36bed5ceb66628526dce806cb843a`.

Las tres condiciones son:

1. Entradas originales, regeneradas en el mismo entorno CPU. Cada pronóstico debe ser exactamente igual al pronóstico CPU previo correspondiente; cualquier discrepancia detiene la admisión.
2. Sustitución de los siete valores de volumen por una ventana completa de train655.
3. Control de sustitución de los 28 valores OHLC por los de la misma ventana donante.

Los 608 donantes se sortean una sola vez con semilla `202611166401`, antes de los forwards. Sólo se usan ventanas de entrenamiento, con los mismos índices para todos los modelos, semillas y condiciones. Se conservan los caminos de siete días dentro de cada grupo. Se verifica la identidad del checkpoint, escalador, configuración y estado antes y después de cada condición.

## Interpretación y límites

La tabla muestra cambios de RMSE y cambios absolutos de predicción. Sus IC95 % marginales se obtienen mediante 3.000 remuestreos comunes por bloques móviles no circulares de 14 fechas, reteniendo las seis trayectorias de inicialización. Son condicionales a una realización fija de donantes y al calendario conocido; no incorporan incertidumbre entre mercados, nuevas particiones de entrenamiento o realizaciones de donantes. No se usan para selección ni para afirmaciones simultáneas de significancia.

La sustitución rompe relaciones entre OHLC y volumen y la consistencia entre ventanas solapadas. Puede crear entradas fuera del soporte conjunto, aunque cada camino donante proceda de train. Por eso diagnostica dependencia bajo esta intervención: no identifica causas económicas, demuestra un atajo espurio, prueba fuga ni estima el beneficio de volver a entrenar sin volumen. Una ablación entrenada tendría otro estimando y requeriría un estudio separado.

A2 y RQ1 originales se mantienen. Se compara cada intervención con su propio original CPU, sin mezclar predicciones A100 con las nuevas CPU.

## Ejecución registrada

Desde la raíz del repositorio, en el entorno CPU fijado, después de verificar las fuentes del plan:

```bash
python scripts/e2_shortcut_followup.py prepare --folder NUEVA_CARPETA
python scripts/e2_shortcut_followup.py infer --folder NUEVA_CARPETA --plan-sha256 SHA_DEL_PLAN
python scripts/e2_shortcut_followup.py statistics --folder NUEVA_CARPETA --plan-sha256 SHA_DEL_PLAN
```

Las operaciones científicas deben pasar por el controlador presupuestario acumulado. El runner científico conserva las rutas y condiciones del estudio registrado; no se atribuye a estos comandos portabilidad fuera de las fuentes admitidas. Los archivos de predicción permiten un reanálisis separado sin deserializar ni volver a ajustar pesos.
