# Exportación CPU de cinco checkpoints E2 congelados

Se recuperaron y comprobaron por SHA256 los checkpoints originales de TimesNet, TiDE, TimeMixer, PatchTST e iTransformer, semilla 271801. La regla representativa es la menor semilla congelada, fijada antes de examinar su resultado. No se ajustó ningún modelo, hiperparámetro, escalador ni calibrador; no se utilizó GPU. Estos artefactos no cambian el modelo vigente del dashboard ni acreditan R1 o R3.

Cada ONNX conserva la receta y los pesos de su checkpoint E2. La entrada es una ventana **normalizada float32 [1,7,5]** en orden OHLCV, y la salida `normalized_prediction` debe multiplicarse por `output_scale` para recuperar sigma7 poblacional diaria, sin anualización. El preprocesamiento usa precios logarítmicos relativos al último cierre de la propia ventana, log1p del volumen y las medias/desviaciones congeladas del ajuste; no se reestima con las 608 fechas de evaluación ni las 53 fechas de calibración. Los q originales se reutilizan, con sus hashes y fechas; las coberturas históricas no garantizan cobertura bajo dependencia temporal o cambio de distribución.

La validación parte de las ventanas OHLCV originales del NPZ inmutable, aplica el preprocesamiento congelado y compara, de una ventana por llamada, CPU en memoria y ONNX Runtime con **608 predicciones y 53 predicciones de calibración previamente guardadas**. La tolerancia absoluta se fijó en **1e-6 sigma7 original**. Se añaden cinco casos sintéticos de contrato de entrada: cero, constante positiva, constante negativa, alternante y simétrica. Son pruebas del exportador, no nuevos diseños o entrenamientos ni nueva evidencia de generalización.

`payload/<modelo>/` contiene ONNX, metadatos, procedencia, recibo de ida y vuelta y licencia de la fuente autora. `registry/*.json` es un inventario local para el benchmark común del auditor; no es el catálogo productivo. Las mediciones incrustadas son provisionales con posible contención: los costes comparables deben provenir del benchmark común, aislado y secuencial, realizado por el auditor raíz.

## Adaptadores de operadores

- **TiDE, TimeMixer, iTransformer:** forward original de la receta congelada. La importación opcional de `reformer-pytorch==1.4.4` requerida por el archivo autor de iTransformer no introduce un modelo Reformer ni pesos externos.
- **PatchTST:** replicación explícita de los dos últimos valores y cuatro cortes P3/S2 sustituyen exactamente `ReplicationPad1d`/`Unfold`, que el exportador no resolvía por inferencia de tamaño. Se mantienen RevIN sobre los siete días reales, encoder, cabeza, readout y pesos. El wrapper se fija en eval y se comprueba que ningún tensor registrado cambie.
- **TimesNet:** la inferencia original E2 ya invocaba cada ventana por separado para evitar que el FFT medio de un lote dependiera de ventanas posteriores. El exportador mantiene ese contrato. Para longitud efectiva ocho se calculan las cuatro frecuencias no DC mediante mariposas FFT reales y **todas** las ramas de periodo 8/4/2/2. Las comparaciones de amplitud y Gather eligen dinámicamente las tres ramas; no se congela el periodo observado al trazar. Los 75 órdenes débiles posibles de cuatro amplitudes están representados en una tabla que conserva el desempate de TopK4/k3 de Torch 2.8 CPU, sin sumar epsilon ni usar los resultados de mercado. Los 256 casos discretos exhaustivos pasan en ORT. Catorce perfiles latentes, incluidos ceros, constantes, simétricos y espectros empatados, verifican coeficientes y selección; el mayor error de bloque ONNX es 7.16e-7 en espacio latente, separado del error final sigma7. Véanse `fft_edge_diagnostics.json` y `topk_order_contract.json`.

Las identidades FFT y de cortes son matemáticas; los recibos cuantifican diferencias numéricas en este perfil. No se afirma igualdad bit a bit para cualquier entrada de coma flotante o cualquier hardware. Entradas que desborden operaciones o contengan NaN quedan fuera del contrato finito del servicio. El primer adaptador con DFT genérica pasó los datos de mercado pero falló ante empates internos; sus contraejemplos se conservan en `attempt_history` y no se presenta como artefacto aceptado.

## Reproducción portable sin entrenamiento

Perfil verificado: Python 3.12.14, Torch 2.8.0+cpu, NumPy 2.3.5, ONNX 1.19.1 y ONNX Runtime 1.23.2. Instalar las dependencias en un entorno de exportación separado, nunca en el dashboard:

```bash
python -m pip install -r requirements-cpu-export.txt
python ci_export_temporal.py --workdir /tmp/e3-temporal-private --output /tmp/e3-temporal-public
```

El wrapper recupera fuentes propias y autoras, cinco checkpoints, NPZ histórico, predicciones, metadatos y selecciones desde URLs públicas inmutables; valida sus hashes antes de deserializar (`weights_only=True`), exige el SHA exacto de cada ONNX y repite la paridad de 661 fechas. Además ejecuta los contratos de TimesNet. El resultado público incluye `temporal_export_manifest.json`. El árbol de trabajo debe conservarse aparte y **no debe publicarse**: contiene datos OHLCV ya existentes, checkpoints, código completo de terceros y muestras auxiliares.

Archivos necesarios junto al wrapper: `export_temporal.py`, `source_input_manifest.json`, `test_timesnet_edges.py`, `test_timesnet_topk.py` y `requirements-cpu-export.txt`. La descarga original y el exportador sólo leen insumos y crean evidencia; no envían mensajes ni promueven modelos. Se declara asistencia de IA en el código y los metadatos; la revisión de los resultados y la decisión de promoción corresponden al autor/evaluador.
