# Verificación de las ejecuciones con CV de Filipe — 04/11/2026

**Resultado: PASS, sin discrepancias detectadas.** Se auditaron los archivos completos del experimento `results/dl_r2_cv_20261104`. Esta verificación no entrenó modelos ni abrió archivos brutos del test reservado de 2025–2026. El script solo escribe el informe JSON indicado explícitamente.

El detalle se conserva en [revision_dl_r2_cv_20261104.json](revision_dl_r2_cv_20261104.json). El verificador reproducible es `scripts/verify_dl_r2_cv.py`:

```bash
python scripts/verify_dl_r2_cv.py
```

Por defecto escribe `/tmp/dl_r2_cv_verification.json`; `--output` permite elegir otra ruta. En un checkout nuevo deben restaurarse previamente los directorios de corrida desde su paquete de artefactos.

## Fuentes y protocolo

Se contrastaron el commit previo a los ajustes `1f0ed23aa40c8160f3a04ae66094aee037fb0128`, las copias del protocolo firmado, los hashes de los seis módulos de entrenamiento/adaptación y los hashes de datos, selección V1 y manifiesto de folds. Se reconstruyó la firma de ejecución y se comprobó que el runtime coincide con el snapshot registrado. Los módulos firmados de V1 permanecen intactos.

El manifiesto identifica `timeseries-cv==0.1.5`, su función `tsxv.splitTrainVal.split_train_val_forwardChaining` y los hashes de su procedencia. La adaptación conserva los orígenes diarios reales de la biblioteca y los agrupa en doce bloques mensuales. Se verificaron las matrices de índices guardadas; no se regeneró el manifiesto con otra fecha ni se afirmó una API mensual nativa inexistente.

Cada bloque usa exclusivamente las muestras históricas admisibles anteriores a su origen mensual, con embargo D=7: la última etiqueta de entrenamiento es como máximo el origen menos ocho días. La validación interna comienza noventa días de calendario antes del origen, termina en el corte externo y contiene 83 fechas; el entrenamiento interno tiene su propio embargo. Las ventanas, etiquetas y hashes de cada cohorte coinciden con el manifiesto de Filipe.

La evaluación diaria h=1 utiliza OHLCV hasta el día previo a cada resultado. Los pesos se mantienen fijos dentro del mes; no es una predicción de todo el mes emitida al inicio ni un reentrenamiento diario. No se comprimen los huecos de marzo de 2023.

## Comprobaciones ejecutadas

| Evidencia | Cantidad verificada |
|---|---:|
| Trayectorias completas: dos variantes × tres familias × diez semillas | 60 |
| Bloques mensuales por trayectoria | 12 |
| Configuraciones/segmentos de modelo por fold | 720 |
| Ajustes efectivos registrados, dos por segmento | 1.440 |
| Hashes de artefactos contenidos en los manifiestos por trayectoria | 5.880 |
| Filas de predicción verificadas | 21.960 |
| Fechas únicas por trayectoria | 366 |
| Filas de curvas de ajuste y validación | 63.358 |
| Checkpoints recargados sin entrenar | 6 |
| Predicciones de diciembre reproducidas por checkpoint | 31 |

Las semillas base 4101–4110 están emparejadas entre variantes y familias. Se verificó la derivación de la semilla de cada mes y la inicialización nueva del reajuste. Las anchuras, learning rates, D=7 y conteos de parámetros se mantienen iguales a las configuraciones seleccionadas de V1.

Para los 720 segmentos se recalcularon los escaladores a partir de sus índices efectivos: tanto los internos como los del reajuste completo. En `absolute_log`, los parámetros se estiman sobre los días únicos de entrada cubiertos por train. En `relative_window`, se estiman sobre posiciones de ventanas de train, pues cada ventana tiene un ancla distinta. Las dos variantes conservan las cinco fuentes OHLCV y dividen Y por su media positiva de entrenamiento, sin centrarla.

Se verificaron los metadatos y las fechas de cada predicción, las etiquetas contra el derivado diario de desarrollo, la persistencia observable y las escalas MASE calculadas en cada entrenamiento mensual. Las métricas se recalcularon con fórmulas independientes, incluido R² como `1 − SSE/SST`. El valor anual se calcula sobre las 366 predicciones agrupadas de cada trayectoria; no es una media de R² mensuales. MASE se calcula con la escala mensual correspondiente a cada fecha.

Se contrastaron las predicciones y métricas agregadas con todos sus archivos mensuales. Las duraciones del reajuste coinciden con la época elegida en la parada interna y las curvas se convierten correctamente a unidades originales. El reajuste completo no contiene evaluación de validación para elegir otra duración.

Finalmente se cargaron los checkpoints del último fold, diciembre, para MLP, LSTM y CNN1D de ambas variantes, semilla base 4101. Sus 31 predicciones por checkpoint coincidieron **exactamente** con los CSV almacenados: error absoluto máximo **0** en los seis casos. Se utilizó `torch.load(..., weights_only=True)` y solo inferencia.

## Errata y límites

La copia inicial de `training.refit` conserva la prosa heredada «full2023». La errata fechada corrige esa descripción a la cohorte mensual permitida, que incluye el pasado ya observado de 2024. Se verificó que ambos registros de errata coinciden y apuntan al protocolo firmado original, sin modificarlo. Los índices, escaladores y configuraciones efectivos de los 720 segmentos corresponden a las cohortes crecientes correctas; el optimizador no utiliza esa cadena descriptiva para elegir datos.

La auditoría valida integridad y coherencia de este experimento, no superioridad confirmatoria ni potencia suficiente. Todo 2024 es desarrollo ya expuesto. Los hiperparámetros V1 se eligieron con enero–junio, por lo que la evaluación anual no es una validación anidada de todo el procedimiento de selección. Julio–diciembre también fue observado durante el desarrollo y se informa con ese alcance.

Las diez trayectorias estudian variación del entrenamiento condicionada al calendario; los doce folds no son doce mercados independientes ni justifican una potencia 0,80. La comparación `relative_window` frente a `absolute_log` comparte CV e hiperparámetros y estudia el paquete de representación, incluido el cambio de ponderación al ajustar escalas. Comparar cualquiera de ellas con V1 estático cambia además la política de actualización, los cortes internos y las semillas; esa diferencia no se atribuye únicamente a la normalización.
