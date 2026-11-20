# Reproducción B de V3 y diagnóstico de transferencia

Preparado con asistencia de IA. Se conservan los 796 trabajos de la referencia aditiva y los 30 ajustes del piloto V3; no se ejecutó ningún entrenamiento ni recalibración.

## Qué quedó ejecutado

- **B, pesos y preprocesamiento:** los 15 checkpoints finales del piloto reproducen las 825 predicciones de desarrollo guardadas; error máximo absoluto `1.7901474420878305e-8` sigma7 frente a la tolerancia fijada `1e-6`. Se usa el estado `best` validado, no el último estado. Los hashes se contrastan con el snapshot congelado del piloto.
- **B, contenedor nuevo:** un perfil ONNX experimental de `atom_crps`, semilla final mínima 20261118 elegida por regla y no por rendimiento, reproduce 55 fechas del 7 de septiembre al 31 de octubre de 2024. Error máximo `7.828650949726068e-9` sigma7; cero violaciones del piso. Se construyó una imagen nueva a partir de Python 3.12.14 por digest y nueve ruedas fijadas por versión y SHA256. La ejecución usó red deshabilitada, sistema de archivos de sólo lectura, UID65534 y ninguna GPU.
- **Diagnóstico nuevo:** tras el commit `6aca875bd94b0a1c3457b5b8586f14a4d0e16dbe` de la decisión y su SHA256 `6bda16b7e322e03a1494420f4bedae58a38cad0aed790a8c6493d5d15637df29`, se ejecutaron exactamente 51 trayectorias y 20.400 predicciones en los tres paneles clean-test ya expuestos. Se reutilizan los 15 pesos del piloto y dos controles analíticos; el escalador y el exceso medio siguen siendo los de BTC. Las 51 trayectorias fueron finitas, sin fallos ni filas eliminadas. Esto evalúa transferencia sin ajuste, no entrenamiento de V3 en esos dominios ni confirmación R3.

La candidata `atom_crps` no supera al mejor control en ninguno de los tres dominios. Consulte la tabla completa `diagnostic_transfer/summary.csv`, incluidas todas las ablaciones. No se añade un umbral nuevo, no se selecciona un ganador nuevo y no cambia el resultado adverso original: `atom_crps` fue 2,7043 % peor que `atom_mse_only` y no pasó la heurística del piloto.

## Perfil experimental separado

`artifacts/atom_crps_seed20261118/model.onnx` ocupa 15.368 bytes. Entradas: `values` float32 de forma `[1,7,5]` y `A` float64 de forma `[1]`; salida: sigma7 float64 `[1]`, poblacional ddof0, diaria sin anualizar. El encoder opera en float32 y la cabeza en float64. El adaptador ONNX representa `hypot` mediante máximo/mínimo y norma escalada, y Softplus float64 mediante primitivas estables equivalentes. No cambia pesos. Se comprobaron las 55 fechas y seis casos numéricos sintéticos de borde; éstos no son datos de mercado adicionales.

**No tiene calibrador**, no hereda el calibrador de la aditiva, no se activa en serving y no acredita R3. La primera exportación directa encontró un operador Softplus float64 sin implementación en ORT; se conserva la incidencia y la equivalencia del adaptador corregido en `execution_scope.json` y `roundtrip.json`.

## Repetición limpia del replay ONNX

Requisitos: Docker y un host Linux x86_64 con Python3.12/pip para descargar las ruedas. Los OHLCV originales no se duplican en este paquete; `external_input_manifest.json` fija commit, URL, tamaño y SHA256. La descarga no implica derechos nuevos sobre los datos.

```bash
python download_inputs.py --output inputs
python -m pip download --only-binary=:all: --no-deps --require-hashes -r requirements-runtime.lock --dest wheelhouse
docker build --network none --pull=false -t e3-v3-frozen-replay:local .
mkdir -p replay_output
chmod 777 replay_output
docker run --rm --network none --read-only --cap-drop=ALL --security-opt no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,size=32m --user 65534:65534 -v "$PWD:/payload:ro" -v "$PWD/inputs/cohort.npz:/inputs/cohort.npz:ro" -v "$PWD/replay_output:/output:rw" e3-v3-frozen-replay:local --npz /inputs/cohort.npz --output /output
```

Los ficheros del paquete y la entrada deben ser legibles por UID65534. El script verifica el modelo, las entradas y el manifiesto antes de comparar. `receipt.json` registra discrepancia, unidades, versiones y alcance. El benchmark usa CPU de un hilo, lote1, diez llamadas de calentamiento y cien medidas para forward y para validación+preprocesamiento+forward. El RSS es el pico del proceso completo, no memoria incremental del modelo. Las medidas comparten host y no constituyen una comparación aislada de costes con otros modelos.

Para reproducir los 15 checkpoints desde PyTorch en el entorno fijado de `requirements-export-host.txt`, ejecute `python replay_fifteen.py --npz inputs/cohort.npz --output fifteen_output`. No necesita descargar datos adicionales. Para repetir el diagnóstico nuevo es necesario descargar las entradas adicionales con `--include-additional` y conservar la decisión registrada; sus instrucciones no equivalen a una ejecución futura observada.

## Niveles y límites

**A:** ya existía una reconstrucción automática de once tablas desde predicciones congeladas, registrada el 18 de noviembre, fuente `73d8958c50a69fbea46deb828f03f89385318789`. Aquí sólo se conserva su referencia y hash; no se repitió ese trabajo.

**B:** se ejecutaron los replays de pesos descritos arriba. **C:** no se reejecutó entrenamiento alguno. Una ejecución automatizada asistida por IA no es una reproducción humana independiente observada. `REPRODUCCION_HUMANA_DRAFT.md` queda preparado, sin nombres ni firmas inventados.

Los dominios D01, D02 y D03 ya estaban expuestos por la auditoría aditiva. D01 mide sesiones y volumen en acciones AAPL RAW, con saltos corporativos; D02 mide días UTC y ETH; D03 días UTC y contratos. Se conserva el transformador de volumen de BTC sin conversión silenciosa. Esos calendarios y mercados no son tres réplicas independientes.
