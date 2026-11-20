# Recreación CPU estricta de nueve exportaciones existentes

Este harness descarga entradas públicas inmutables y sólo reconstruye
**artefactos e inferencia de checkpoints ya entrenados**. No repite entrenamiento,
HPO, selección de variantes ni calibración. Conserva la semilla 271801 y la
tolerancia 1e-6 fijadas antes de validar.

Archivos del bloque estándar que deben conservarse juntos:

- `ci_export_standard.py`
- `standard_source_input_manifest.json`
- `export_standard.py` (CLI original, también verificado por SHA)

El manifiesto fija los checkpoints y 19 módulos/JSON propios del proyecto por
SHA, los cuatro recibos de selección, el archivo original de scalers y
calibradores, 608+53 ventanas reales y las predicciones originales. Todas las
URLs son públicas y fijan commits o assets de releases. No se necesita token
para descargar estas entradas. El bloque estándar no requiere fuentes de
terceros para sus cuatro cuerpos independientes.

El bloque temporal mantiene juntos `ci_export_temporal.py`,
`source_input_manifest.json`, `export_temporal.py`,
`test_timesnet_edges.py` y `test_timesnet_topk.py`, preparados por el agente
temporal. Sus fuentes autora también se recuperan por hashes fijados, sólo
en el directorio privado temporal.

Instalar `requirements-cpu-ci.txt` en **un runner limpio**, con Python
**3.12.14**. No se instala ni modifica nada en el entorno compartido local.
El fichero incluye el conjunto de dependencias de ambos bloques; Torch usa
su distribución CPU. El wrapper estándar comprueba las versiones críticas.

```bash
python -m pip install -r ci/requirements-cpu-ci.txt
python ci/ci_export_all.py \
  --standard-wrapper ci/standard/ci_export_standard.py \
  --temporal-wrapper ci/temporal/ci_export_temporal.py \
  --workdir "$RUNNER_TEMP/e3-private-inputs" \
  --output "$RUNNER_TEMP/e3-public-exports"
```

Las rutas de los wrappers pueden adaptarse a la estructura del repositorio.
`--workdir` y `--output` deben estar en árboles distintos. Sólo se sube
`--output`, después de una salida satisfactoria. Nunca se sube el directorio
privado: contiene OHLCV original, checkpoints, fuentes autora e informes de
ejecución. El harness comprueba nueve modelos, sus hashes ONNX exactos, 661
casos por modelo, tolerancia fija y cero entrenamientos/recalibraciones, y
genera `nine_export_release_manifest.json`.

**La identidad binaria entre este host y GitHub Actions todavía debe ser
ejecutada y comprobada en CI.** Los metadatos y recibos nuevos tienen su propia
fecha/plataforma; no se promete que esos JSON tengan el hash de los recibos
locales. Los cuatro SHA ONNX aceptados sí son un requisito estricto. Una
diferencia de SHA detiene la publicación, aunque la paridad numérica pase.
En ese caso se conserva el ONNX local ya verificado mediante el mecanismo
alternativo de partes de 128 KiB; no se acepta automáticamente otro binario.

`--prepare-only` del wrapper estándar permite comprobar descargas públicas,
hashes y disponibilidad de constructores sin exportar ni ejecutar forwards.
Esta comprobación se ejecutó localmente; el informe registra su alcance.
No se presenta como un ensayo de GitHub Actions ni como la recreación completa
del pipeline de entrenamiento. La asistencia de IA se declara en los recibos.
