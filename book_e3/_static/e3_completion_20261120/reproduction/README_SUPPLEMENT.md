# Suplemento documental: pesos públicos y exportación local

Preparado con asistencia de IA. Este paquete público documental omite los archivos `.pt` y `.onnx`. **No publica el ONNX experimental derivado de V3**, ni los nuevos binarios de las otras exportaciones. El paquete original de 122 archivos y sus binarios se conserva localmente con su manifiesto original; no se modifica ni se presenta el subconjunto documental como si lo reemplazara byte a byte.

`public_weight_locators.json` identifica los 15 checkpoints finales que **ya estaban publicados** en el commit original `8eaa581bc34b615a8507c9e384a3e4be29084720`, snapshot 93 del piloto `20261120T031424176339Z`. Para cada checkpoint fija URL pública, commit, ruta, tamaño y SHA256. Incluye también el destino duplicado del representante: semilla mínima 20261118 de `atom_crps`, elegida por regla, no por métrica.

`hydrate_public_weights.py` descarga únicamente esos pesos originales, sin credenciales, y verifica tamaño y SHA256 antes de escribir. Conserva un archivo existente si su hash coincide y falla si difiere. Copia el representante desde el mismo checkpoint verificado. **No entrena, infiere, recalibra, exporta ni sube archivos.** La recuperación de pesos ya públicos no constituye publicación del ONNX derivado pendiente de aprobación.

## Repetición local

Trabaje en una copia local del paquete documental; conserve aparte su manifiesto público original. Desde la raíz de esa copia, con los scripts originales de reproducción disponibles:

```bash
python hydrate_public_weights.py --packet-root .
python download_inputs.py --output inputs
python -m pip install -r requirements-export-host.txt
python export_v3.py --npz inputs/cohort.npz --root .
```

El último comando reconstruye **localmente** el ONNX experimental desde el estado `best` del checkpoint congelado y comprueba paridad frente a las 55 predicciones de desarrollo guardadas. No cambia pesos ni ejecuta un ajuste. La tolerancia es `1e-6` en sigma7; no se hereda ningún calibrador de la referencia aditiva. La exportación local y sus recibos no deben confundirse con una descarga de un ONNX público ni con autorización para publicarlo.

Después de esa exportación, prepare el replay Docker de las instrucciones del README original: descargue las nueve ruedas fijadas en `requirements-runtime.lock`, construya la imagen con `Dockerfile` y ejecute `replay_onnx.py` con datos y código de sólo lectura, sin red. El grafo se monta desde su copia local, sin subirlo a GitHub. Si el replay usa `MANIFEST.sha256.json`, su copia de trabajo debe tener un **manifiesto local nuevo** que incluya los archivos hidratados y derivados; preserve el manifiesto documental publicado y registre el SHA256 del manifiesto local por separado. No sustituya la identidad del paquete original ni trate hashes nuevos como si fueran los publicados.

Los recibos del paquete describen ejecuciones ya observadas: reproducción B de 15 checkpoints/825 predicciones y un representante ONNX/55 predicciones en contenedor. Las instrucciones anteriores están preparadas para una nueva ejecución; no equivalen a otra reproducción ya ejecutada. No se reentrenaron los 796 trabajos anteriores ni los 30 ajustes del piloto, y no se acredita una reproducción humana independiente.

La variante completa `atom_crps` sigue habiendo fallado la heurística original y fue 2,7043 % peor que `atom_mse_only` en el piloto. La exportación no la promociona, no activa serving y no acredita R3.
