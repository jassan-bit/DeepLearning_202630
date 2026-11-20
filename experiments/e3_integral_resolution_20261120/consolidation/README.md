# Consolidación del archivo revisable 0.3.1

Herramienta stdlib preparada con asistencia de IA. No importa modelos, no carga
checkpoints, no infiere ni entrena. Tampoco publica en GitHub/Zenodo ni concede
una licencia global al repositorio.

## Entradas fijadas

1. Artifact de Actions44833679006, ID14676474833, descargado por el runner.
   El tar0.3.0 debe tener58823392 bytes y SHA256
   `d7a97766af592629efafcc64c386c9aeae39b4ab107b909217d9629b9d8a6d35`.
   `execution_receipt.json` debe estar junto al tar. Se comprueban todos los
   hashes internos,18564 miembros y281641001 bytes expandidos antes de añadir
   material. El recibo acredita9227 rutas lógicas (8841payloads+386metadatos)
   y9217 objetos SHA únicos; no se repite aquella descarga de4,27GB.
2. ZIP público de las nueve exportaciones ya aprobadas,12987835 bytes,
   SHA256 `0fb857c407250949bfc8c96fd7b9525bab624b650b5c94dd4d1d1d8c60f62c66`.
   Se verifican también el manifiesto de11779 bytes (SHA256
   `e8eb71d5356b20b2950544e52a7a63fcfc22cfa323cadda98e9365db120f90bf`),
   las66 entradas y los nueve hashes ONNX independientes. Pesos, escaladores,
   calibradores, recibos y notices se copian byte por byte.
3. Opcionalmente, el kit C con un manifiesto final cuyo SHA256 proporciona el
   llamador. Sólo se incluyen los miembros enumerados y verificados. El único
   CSV permitido es `references/assessment_predictions.csv`,71967 bytes y
   SHA256 `0399c7c80f46ac2fdfccaf218a2866fe110f7ec5051d8484e488299c679ad193`.
   Sus935 filas son825 salidas neuronales+110 de controles ya públicos del
   desarrollo, no predicciones nuevas. No se incluyen NPZ, raw inputs o pesos
   de entrenamiento. La preparación del kit C no acredita su ejecución humana.

## Ejecución

```bash
python consolidate_archive.py \
  --base-archive /tmp/base/e3-source-evidence-8d03a73-v0.3.0-REVIEW.tar.gz \
  --output /tmp/e3-consolidated \
  --C-addon-root /ruta/al/kit-C \
  --C-addon-manifest /ruta/al/kit-C/MANIFEST.json \
  --C-addon-manifest-sha256 SHA256_DEL_MANIFIESTO \
  --launcher-commit COMMIT_DEL_LANZADOR
```

El directorio de salida debe ser nuevo o estar vacío. Sin `--nine-zip`, se
descarga sólo el ZIP público fijado, sin cabecera de autorización. Una copia
local se admite mediante `--nine-zip` y debe tener exactamente el mismo hash.
El addon es opcional técnicamente: omitir sus tres argumentos lo registra como
no incluido. Si se solicita, su raíz, manifiesto y SHA son obligatorios juntos.

Las salidas son el tar0.3.1-REVIEW, `capsule_manifest.json`,
`consolidation_receipt.json` y `RELEASE_NOTES.md`. El tar se reconstruye con
rutas ordenadas, mtime/uid/gid cero y gzip de cabecera estable. Los miembros
originales se conservan; el README y manifiesto raíz previos se guardan bajo
`provenance/` antes de generar los de la nueva revisión. Se verifica de nuevo
cada miembro del archivo final y que contiene exactamente15 hashes ONNX:
catorce históricos y V3 experimental aparte. No se activan modelos en Render.
Se elimina cualquier tar incompleto si hay un fallo; la herramienta no puede
subir un archivo por sí sola.

## Límites de interpretación

El tar final sigue siendo un archivo de fuentes/evidencias con exclusiones
explícitas, no un depósito de todo el maestro científico. El DOI0.2.0 y su ZIP
permanecen intactos. Los derechos/notices originales se conservan; esta
consolidación no extiende MIT/CC BY a terceros ni resuelve una licencia global.
Tampoco es un preregistro, nueva candidata, reproducción humana o R3 acreditada.

Pruebas: `python -m unittest discover -s . -p test_consolidate_archive.py -v`.
Son fixtures de software declarados, sin datos de mercado o ejecución de grafos.
El resultado real sólo lo establece el recibo de una ejecución completa.
