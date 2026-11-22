# Datos cripto y adaptador de admisión

`DATA_SELECTION_DECISION.md` explica la propuesta de tres mercados adicionales,
sus dos ejes, calendarios, exposición y límites. `data_panels_v2.json` contiene
el contrato final de metadatos; `data_panels_initial_proposal_v2.json` conserva
la propuesta previa y la enmienda de F03 se registra en el contrato final.
Ningún archivo contiene precios de mercado nuevos, reserva o resultados de
modelos. El estado real de admisión continúa en cero.

`engine_data_v2.py` es el archivo completo de `engine.data` para la fuente nueva,
con versión `crypto-markets-bounded-archives-base-volume-v2.1`. Se instala como
`runtime/engine/data.py`, bajo un manifest propio; no se parchea dinámicamente el
motor A ni se cambian sus identidades. El runtime proporciona también
`engine.confirmatory_gate`: la CLI de preparación final está bloqueada antes de
leer entradas, aunque la función de bajo nivel permanece comprobable con
fixtures explícitos. La candidata, las semillas y el objetivo no cambian.

Los comandos operativos conservan sus argumentos originales. El notebook de la
entrega crea los formularios y ejecuta, sólo tras activación, esta secuencia:

```bash
python -m engine.data record-owner-evidence --form private/owner-development.json --evidence private/owner-development.json --output private/exposure-development.json
python -m engine.data collect-development --config private/acquisition.json --rights private/rights.json --exposure private/exposure-development.json --output private/raw --enable-market-acquisition
python -m engine.data prepare-development --manifest private/raw/raw_manifest.json --split-config private/development-split.json --rights private/rights.json --exposure private/exposure-development.json --output private/panel
```

Los paths de ese ejemplo representan un panel; el notebook usa namespaces
separados F00–F03. Los documentos deben contener la declaración real del actor,
no copiar identidades de los tests. No hace falta una clave Nasdaq para este
adaptador. GitHub y la clave independiente de cifrado siguen siendo necesarios
para el respaldo de campaña, mediante el canal privado del notebook.

`binance_archive_plan(config)` y
`inspect_binance_pending(config, directory, rights_receipt=..., exposure_receipt=...)`
son operaciones de lectura/metadatos. La segunda sólo valida un estado parcial
ya vinculado a configuración y evidencia; no lo admite como dataset.
`validate_binance_manifest(record, manifest_path=None)` verifica el vínculo
canónico de fuente; con una ruta verifica también archivos y compañeros
CHECKSUM. El colector emite progreso JSON seguro y permite reanudar bytes
parciales verificados sin reescribirlos.

## Repetir la verificación de software

Desde esta carpeta y con el Python aislado del paquete (NumPy fijado):

```bash
python -m unittest -v test_crypto_data_v2.py
```

Para la regresión del replay legado, el test busca el archivo original en el
checkout del repositorio. Si se trabaja con una extracción que no lo contiene,
definir `E3_SOURCE_A_DATA` con la ruta del `engine/data.py` verificado del commit
`c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c`. Se exige su SHA256 original; un
archivo actual distinto no se acepta como referencia histórica.

`SOFTWARE_TEST_OUTPUT.txt` conserva 20 pruebas nuevas. Incluyen unidades,
calendarios, recuperación de tres proveedores, fuente Spot A real, detección de
relabelado con hashes recalculados y preparación final bajo autorización de
software explícita. `LEGACY_DATA_COMPATIBILITY_TESTS.txt` conserva 23 pruebas del
código de datos y recuperación anterior. La revisión independiente de los dos
defectos encontrados está enlazada por `DATA_QA.json`.

El generador local del archivo extendido se reproduce con:

```bash
python build_data_v2.py
python optimize_archive_plan.py
python bind_manifest_identity.py
```

Estos generadores necesitan la misma referencia A (`E3_SOURCE_A_DATA` si no está
en el checkout); producen código y procedencia, sin red ni entrenamiento. Se
ejecutan en orden sobre una carpeta de trabajo nueva, no sobre una fuente ya
publicada o durante una adquisición activa. `qa_v2_initial/` conserva la primera
versión y las 16 pruebas que no cubrían los dos defectos hallados posteriormente;
no describe la versión corregida final.

Asistencia de IA declarada. Las pruebas no autentican actores, licencias de
fixtures o reservas y no acreditan R3.
