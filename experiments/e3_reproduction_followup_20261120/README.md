# Reproducción del binario V3 público aceptado

Preparado y ejecutado con asistencia de IA. Esta ruta evita la reexportación: descarga el ONNX experimental público ya aceptado y comprueba sus bytes, metadatos, preprocesamiento y 55 predicciones de desarrollo guardadas. No incorpora nuevos días de evaluación y no activa V3 en el servicio.

## Ejecución limpia del tercero

Desde este directorio en un checkout del commit publicado, con Linux x86_64, Docker y Python 3.12/pip:

```bash
python hydrate_accepted.py --output inputs
python audit_existing_receipts.py --output existing_levels_audit_fresh.json
python check_integrity_guards.py --inputs inputs --output integrity_checks.json
python -m pip download --only-binary=:all: --no-deps --require-hashes -r requirements-runtime.lock --dest wheelhouse
docker pull python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
docker build --network none --pull=false --no-cache -t e3-accepted-v3-replay:local .
mkdir -p replay_output
chmod a+rx . inputs
chmod a+r ./*.py ./*.json inputs/*
chmod 777 replay_output
docker run --rm --network none --read-only --cap-drop=ALL --security-opt no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,size=32m --user 65534:65534 --cpus 1 --memory 256m --pids-limit 64 -v "$PWD:/payload:ro" -v "$PWD/inputs:/inputs:ro" -v "$PWD/replay_output:/output:rw" e3-accepted-v3-replay:local --inputs /inputs --output /output
sha256sum replay_output/receipt.json replay_output/discrepancies.csv
```

Las descargas comprueban tamaño y SHA256; un archivo existente distinto se conserva y causa error. La cohorte original completa ocupa 460.090 bytes, pero el replay abre únicamente `training_raw` y `training_dates`. No abre las particiones de calibración/prueba. El volumen y escalador BTC permanecen congelados. No existe un calibrador V3 y no se utiliza el de la aditiva.

El contenedor no instala Torch ni ONNX/exportadores. Usa Python 3.12.14 por digest y nueve ruedas fijadas por versión y hash; ORT 1.23.2 y NumPy 2.3.5. La construcción se realiza sin red después de obtener la base y las ruedas. La ejecución usa red deshabilitada, raíz de sólo lectura, UID65534, un CPU lógico y límite de memoria de 256 MiB. El manifiesto ata el modelo de 15.368 bytes a `881dd637837265829aa96900d7a0c864dcdab113a57a91485feb20fd2a443abc`, estado validado `best`, época37, semilla20261118. Esa semilla se fijó por regla mínima, no por su rendimiento.

`discrepancies.csv` compara por fecha la salida actual con la GPU congelada y con ORT aceptado, cada una en sigma7 diaria poblacional ddof0. La tolerancia absoluta es `1e-6` por salida. Las fechas y su orden son exactos; se comprueban tipo, forma, finitud y piso estructural. El recibo separa discrepancias numéricas, recursos del proceso completo y alcance científico. El RSS no es memoria incremental ni una comparación de recursos del servicio.

## Qué está ejecutado y qué sigue pendiente

`execution/receipt.json`, `execution/discrepancies.csv` y `execution/container_context.json` documentan una ejecución nueva del binario público aceptado: error máximo `7.828650949726068e-9` sigma7 frente a GPU y cero frente a ORT aceptado; cero violaciones del piso. El proceso final midió 0,1509 s para las verificaciones y 55 forwards, y pico RSS de 69.404 KiB. Esas cifras no incluyen la construcción/descarga ni miden el servicio web. Se conservó el primer recibo numérico correcto y se repitió sólo el replay CPU para corregir la serialización de escalares NumPy en el CSV. El CSV final contiene 55 filas con campos numéricos estándar. `execution/integrity_checks.json` prueba que una alteración de un bit en cualquiera de las cuatro entradas se rechaza antes del forward; los originales quedan intactos. El primer intento de contenedor falló por permisos de lectura de las carpetas y no ejecutó inferencia; el log se conserva y el procedimiento explicita los permisos mínimos.

| Nivel | Evidencia disponible | Alcance y estado |
|---|---|---|
| A: tablas desde predicciones | Once tablas y discrepancias por campo de la reproducción automática de 2026-11-18 | Existente y auditada por hash; no repetida en este cierre. No reconstruye entrenamiento. |
| B: predicciones desde pesos | Quince checkpoints V3 y 825 predicciones; nueve ONNX históricos con 5.949 replays remotos; nuevo replay aceptado de 55 fechas | Automático; pesos/preprocesamiento verificados. Sin nuevas fechas ni ajuste. |
| C: entrenamiento | Fuentes, planes y recibos históricos del piloto y maestro | No ejecutado aquí. Los 796 trabajos y los 30 ajustes se conservan; no se repiten. |
| Reproducción humana independiente | Este paquete, comandos y `REPRODUCTION_FORM_DRAFT.md` | Pendiente de que una persona ajena al diseño ejecute y aporte su constancia propia. |

`existing_levels_audit.json` fija los recibos previos por hash y conserva sus límites. La ejecución en un contenedor limpio es independencia de proceso/entorno, no independencia humana ni confirmación de superioridad. R3 permanece sin acreditar. Las 51 trayectorias de transferencia y 20.400 filas siguen siendo diagnóstico retrospectivo; no se vuelven independientes al reproducir el forward.

Para la reproducción B ampliada de los quince checkpoints se conserva la ruta inmutable `experiments/e3_completion_20261120/reproduction/README_SUPPLEMENT.md`, con hidratos públicos y hashes de pesos. No es necesario reexportar el ONNX para la ruta rápida de este directorio. Ningún paso de este directorio inicia C.

Los archivos de datos se referencian en su origen y no se atribuye una licencia nueva. La publicación de un depósito Zenodo requiere resolver derechos por componente y autorización del propietario; el replay no suple esos requisitos.
