# Capacidades de exportación de los 13 sistemas pendientes

Esta revisión añade **una exportación comprobada de TSMixer**, justifica mantener los tres controles clásicos como resultados históricos bajo el contrato actual de siete barras y deja **nueve exportaciones neuronales pendientes**. No se entrenó ni se recalibró ningún predictor. Estos archivos no modifican por sí solos el catálogo del servicio.

La [matriz por sistema](capability_matrix.csv) distingue la información necesaria, el checkpoint recuperado, las operaciones reales, los intentos efectivamente realizados y el trabajo que falta. [provenance.json](provenance.json) registra los hashes de las fuentes consultadas y delimita el inventario local. La ausencia de un peso en ese inventario no demuestra que falte en las entregas originales.

## Exportación real de TSMixer

Se recuperó el checkpoint de la semilla fija 271801, identificado por SHA256 `9ead27bd519975e8a62abc79c92ee326aec8bece71170828666b9964ae454be7`. Coincide con el predictor de la tabla congelada y con su calibrador. Se reconstruyó el constructor original sin modificarlo, con el punto HPO `TSM-D7-r1__hpo_v1:hpo-008`, dropout 0.3 y sus 35044 parámetros. La semilla es la menor identificada de antemano, no una elección por su resultado en prueba.

| Comprobación | Resultado |
|---|---:|
| Formato / entorno | ONNX opset 17 / ONNX Runtime CPU, un hilo |
| Tamaño | 165263 bytes |
| Predicciones de prueba contrastadas | 608 |
| Error absoluto máximo frente a las predicciones congeladas, sigma7 original | 2.2850587297340574e-8 |
| Error absoluto máximo frente al módulo PyTorch CPU | 9.99158632913888e-9 |
| Tolerancia declarada, sigma7 original | 1e-6 |
| Predicciones de calibración contrastadas | 53 |
| Cuantil heredado, sin reajuste | 0.007345522909209999 |
| Cobertura histórica individual, semilla 271801 | 89.30921052631578 % |
| Mediana / p95, preprocesamiento + ORT caliente | 0.0820635 / 0.1033561 ms |

Los tiempos corresponden a esta máquina y excluyen explicación, arranque, transporte y medición de RAM; no son un SLA de Render. El intervalo usa `[max(0, predicción−q), predicción+q]`, con cobertura **nominal** del 90 %. La calibración usa los 53 días del 8 de noviembre al 30 de diciembre de 2024, con el mismo checkpoint y escalador. La dependencia temporal impide afirmar una garantía exacta de intercambiabilidad, cobertura futura o cobertura por grupo. La cobertura individual no debe sustituirse por la media de las diez semillas.

Los resultados completos están en [tsmixer/roundtrip.json](tsmixer/roundtrip.json), la identidad del artefacto en [tsmixer/metadata.json](tsmixer/metadata.json) y la procedencia de cada entrada en [tsmixer/provenance.json](tsmixer/provenance.json). El artefacto es [tsmixer/model.onnx](tsmixer/model.onnx). El servicio sólo debe adoptarlo después de enlazar su metadata mediante hash y verificar el flujo real de entrada, predicción y explicación.

Además, `python verify_cpu_examples.py` ejecuta tres ventanas reales divulgadas con NumPy y ONNX Runtime, sin Torch. Su [recibo guardado](tsmixer/cpu_verification.json) comprueba los hashes, las predicciones y los extremos de los intervalos con el cuantil heredado.

El exportador emitió avisos de trazado sobre validaciones Python de forma/valores y `len(x)`. El grafo admitido tiene forma **fija `[1,7,5]`**; la admisión de valores finitos y la forma corresponden al runtime. No se anuncia soporte para lotes de tamaño variable. La paridad se comprobó ventana por ventana con todos los casos congelados de prueba y calibración.

## Por qué los tres clásicos necesitan otro contrato

La interfaz actual recibe siete barras completas consecutivas hasta `t−1`. Eso entrega siete cierres y seis retornos conocidos. El objetivo es la desviación estándar poblacional de siete retornos, que requiere ocho cierres.

| Control congelado | Información requerida por su receta |
|---|---|
| Persistence, rezago 1 | `sigma7(t−1)`: cierres `t−8..t−1`, o la etiqueta histórica exacta. |
| SeasonalNaive, período 7 | `sigma7(t−7)`: cierres `t−14..t−7`; una carga que termine en `t−1` necesita al menos catorce cierres consecutivos. |
| ARIMA, orden `[2,0,1]` | Parámetros congelados y estado de filtrado fechado hasta `t−1`; la construcción original exige al menos 90 etiquetas válidas y actualización causal secuencial con parámetros fijos. |

[input_contract_counterexamples.json](input_contract_counterexamples.json) contiene dos verificaciones constructivas: historias con las mismas siete barras admitidas y distinto cierre anterior producen pronósticos originales distintos. No interviene ningún ajuste. Por ello, la interfaz actual no determina estos pronósticos. Mostrar sus resultados históricos con una etiqueta explícita es defendible; habilitarlos en vivo exige una admisión de historia o estado adicional y su propia prueba de paridad. Esto no implica que su ejecución en CPU sea inviable ni que requieran ONNX obligatoriamente.

## Nueve redes que siguen pendientes

FCN-1D, ResNet-1D, Inception-1D, el autoencoder LSTM, TiDE, TimeMixer, PatchTST e iTransformer tienen contratos neuronales compatibles con la ventana D7, pero no se recuperó su checkpoint congelado en el directorio inspeccionado. Su exclusión definitiva por recursos **no está acreditada**. Hace falta recuperar los pesos con el hash especificado, usar el cuerpo y fuentes originales de esa receta, exportar y comprobar las 608 predicciones congeladas. Se deben medir latencia y RAM del runtime si se propone una exclusión por recursos.

TimesNet también sigue pendiente. Su fuente usa `torch.fft.rfft`, magnitudes complejas, `topk` sobre frecuencias no DC y conversión a NumPy para elegir períodos y formas de ramas dependientes de la entrada. Un trazado que congelara esos períodos podría parecer válido para un ejemplo y cambiar el predictor en otros. Se requiere un exportador que conserve esa semántica o una implementación CPU fiel y comprobada. No se ejecutó aquí un intento con sus pesos y no se declara una imposibilidad universal de ONNX.

No se intercambian recetas históricas por cuerpos nuevos, se eliminan mecanismos ni se reentrenan sustitutos bajo el mismo nombre. En particular, exportar el predictor de un autoencoder correctamente preentrenado no necesita incluir sus decodificadores de entrenamiento, pero sí recuperar sus pesos finales originales.

## Reproducción

Se utilizó un entorno ya existente con Torch, ONNX y ONNX Runtime. El exportador comprueba el hash del checkpoint **antes** de abrirlo con `torch.load(weights_only=True)`. Todas las rutas externas se reciben como argumentos; los hashes esperados están fijados en el script.

```bash
python export_tsmixer.py \
  --repo /ruta/al/repositorio \
  --checkpoint /ruta/al/selected_model.pt \
  --npz /ruta/a/current_cohort_arrays.npz \
  --predictions /ruta/a/combined_predictors.json \
  --protocol /ruta/a/protocol.json \
  --selection /ruta/a/TSMixer.json \
  --scaler /ruta/a/scaler.json \
  --calibrator /ruta/a/calibrator.json \
  --output /ruta/nueva/tsmixer
```

`audit_capabilities.py` regenera el inventario, la matriz y los contraejemplos a partir de las fuentes y recibos originales. Un nuevo checkpoint recuperado detiene la generación para exigir que se actualice el dictamen; no lo clasifica automáticamente como validado.

La revisión, el código de exportación y esta documentación se prepararon con asistencia de IA (Codex). Esta es una auditoría técnica de exportación y contrato de entrada; no acredita R1 ni completa la comparación confirmatoria R3.
