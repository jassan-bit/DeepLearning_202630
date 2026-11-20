# Costes ORT comparables

Asistencia de IA declarada. Se midieron15procesos secuenciales con el mismo host/CPU lógico, un hilo ORT, lote1, diez calentamientos y100 forwards. Los14representantes históricos y V3 experimental conservan su escalador. La ventana física común corresponde a evaluation_raw[0], objetivo2025-01-01, del NPZ fijado por input_reference.json. No se incluyen HTTP, Dash, explicación por oclusión, GPU, FLOPs ni energía; RSS es el pico del proceso completo. El host es compartido y no se garantiza exclusividad, caché de disco fría o concurrencia de producción.

measured/ es la pasada final enlazada a los metadatos congelados. superseded_pre_metadata_freeze/ conserva la primera pasada: sus pesos fueron iguales, pero cinco recibos metadata cambiaron durante la exportación; no se usa como comparación final. Ningún coste de entrenamiento ni energía se reconstruye desde desconexiones.

Para repetir, descargue el NPZ original usando reproduction/download_inputs.py, extraiga los nueve ONNX del archivo de ingeniería verificado, reemplace {repository}, {packet} y {exports_bundle} en registry_template.json, prepare un JSON con raw_window = evaluation_raw[0] y provenance=input_reference.json, y ejecute benchmark_ort.py --registry registro.json --input-json ventana.json --output nueva_salida. No usa Torch ni entrena.
