# Recuperación opcional de pesos originales para adecuación retrospectiva

Material ejecutable para reutilizar los **cinco checkpoints originales seleccionados** de `input_only_mse`, sin entrenar, cambiar sus reglas o promover el V5 completo. El código usa exclusivamente lectura remota GET del commit original `05c1a637f13356dd1df6b8c5aa8f4676dda61c12`. La nueva inferencia y sus recibos son independientes de la reanudación del entrenamiento: el guard original de cohorte no se modifica ni se llama con una cohorte distinta.

**Estado: preparado; recuperación con clave real e inferencia real pendientes.** No existe una clave privada en esta sesión, y no se solicita en el chat. Las verificaciones ejecutadas que se describen abajo no son resultados experimentales nuevos ni acreditan adecuación, calibración o R3.

## Acción concreta

Abrir `E3_V5_FrozenWeights_PrivateRecovery.ipynb` en Colab, revisar el protocolo y activar `RUN_RECOVERY=True` cuando el propietario desee ejecutar el análisis. `INSTALL_RUNTIME=False` por defecto; una instalación CPU aislada con el lock original sólo ocurre si el propietario activa también esa opción. El cuaderno estadístico NumPy/SciPy principal no importa ni necesita Torch.

La celda privada usa los secretos de Colab `GITHUB_TOKEN` (lectura) y **`E3_V5_BACKUP_KEY`**, que debe ser la clave original del respaldo V5. Si faltan, ofrece entrada oculta al ejecutarse voluntariamente. No se incluyen claves en el código, argumentos, entorno, registros o salidas. Un caso ya restaurado no pide secretos ni consulta GitHub. Un resultado existente se verifica sin Torch, red ni inferencia adicional.

No se llama `setup`, `run`, `assess`, `backup`, `load_case` ni a un optimizador. La restauración original sólo escribe un caso privado local con su recibo; no escribe respaldos remotos, no modifica los pesos y mantiene la identidad original. El script bloquea cualquier POST/PATCH/PUT/DELETE. No descarga mercado ni accede lógicamente a las matrices de reserva del NPZ histórico: sólo `training_raw`, `training_y_original` y `training_dates`, conforme al lector original verificado.

## Verificación antes de exportar

`REANALYSIS_PLAN.json` fija antes de una eventual ejecución privada las fuentes, el respaldo, los checkpoints y las tolerancias. Es un plan de **reanálisis posterior a los experimentos**, no un preregistro retrospectivo del V5.

1. Manifest original, todos sus archivos, ZIP público, NPZ y cifrado, con commits inmutables, tamaño y SHA256.
2. Plan/caso original y sus hashes; datos, admisión, escaladores y metadatos `job.json` exactos frente al ZIP público congelado.
3. Cinco snapshots con su nombre, SHA256, solicitud y `best_epoch` auténticos: **39, 3, 2, 31 y 19**. Se usa `state['best']`, nunca `current` ni un checkpoint local parecido.
4. Por semilla, se reproduce la RMSE de selección (54 puntos) y cada una de las 55 predicciones de assessment ya guardadas. Tolerancias fijas: `atol=1e-8`, `rtol=1e-5`. Si falla una, no se exportan resultados ni se ajusta la tolerancia después de ver el fallo.
5. Sólo tras pasar las cinco paridades se evalúan los 532 puntos fit. Se comprueba que ningún peso cambió y se registran las versiones reales de inferencia. El entrenamiento original usó Python3.13.16, Torch2.8.0+cpu y NumPy2.3.5; el reanálisis admite otra versión de Python sólo como inferencia independiente con paridad obligatoria. Torch y NumPy conservan esas versiones.

Hay **2930 puntos nuevos fit/selección** (5×586), más **275 puntos de comprobación** de assessment, total **3205**. Ni las cinco semillas ni las 55 fechas expuestas constituyen muestras nuevas de confirmación.

## Salidas privadas

`WORK/fit_selection_PRIVATE` conserva cuatro archivos y un manifiesto:

- `fit_selection_predictions_PRIVATE.csv`: 2660 predicciones fit y 270 de selección, con fecha, semilla, objetivo y predicción.
- `fit_selection_metrics_PRIVATE.csv`: RMSE, MSE, MAE y sesgo descriptivo por partición y semilla.
- `frozen_assessment_and_selection_parity.csv`: diferencias y comprobación de las cinco paridades.
- `execution_receipt.json`: commits/hashes, cohorte original y observada, tiempos reales, conteos y alcance.
- `OUTPUT_MANIFEST.json`: tamaño/SHA256 de los cuatro archivos anteriores.

La carpeta tiene permisos 0700; los archivos, 0600. No se reemplaza una salida existente. No se publican pesos ni CSV privados y no se emite una decisión de adecuación. Para conservarlos tras cerrar Colab, descargarlos a un dispositivo privado desde el panel de archivos. El reanálisis puede aportar entradas para las Tablas A/B y el gap de desarrollo, pero no fija retrospectivamente un margen prospectivo, no produce intervalos calibrados y no acredita confirmación/R3.

Para verificar una salida existente, sin dependencias científicas ni secretos:

```bash
python frozen_v5_reanalysis.py --output-dir /ruta/privada/fit_selection_PRIVATE --verify-only
```

## Verificaciones realmente ejecutadas

`SOFTWARE_TEST_OUTPUT.txt`: **14 pruebas aprobadas**. Incluyen lectura y verificación de fuente/ZIP/NPZ originales y hashes reales de preprocesamiento, con geometría 532/54/55. Las demás usan fixtures explícitos: selección `best` frente a `current`, evaluación sin gradientes/optimizer, conteos, errores de época/hash/paridad, permisos e inmutabilidad de salidas; cifrado/descifrado original con clave ficticia, rechazo de clave/contexto incorrectos; GET-only y ausencia de acciones con la opción desactivada. No se usó clave real, no se restauraron pesos reales ni se generaron predicciones de mercado.

Comando ejecutado en el runtime CPU científico existente:

```bash
/workspace/e3_engine_clean_cache/venv-cpu-652abf93aba295c3-py312/bin/python -m unittest -v test_frozen_v5_reanalysis.py
```

`NOTEBOOK_DISABLED_EXECUTION.json`: ambas celdas compiladas y ejecutadas con `python -S`, red bloqueada y sin Torch importado, instalaciones, secretos, inferencia o carpeta privada. El estado desactivado sólo materializa los dos archivos de código público embebidos, cuyos SHA se verifican antes de importarlos.

`SOFTWARE_VERIFICATION_RECEIPT.json` y `MANIFEST.json` separan la verificación del programa de la ejecución científica todavía no realizada. Las pruebas de integración de lectura usan las rutas del caché científico existente en `/workspace`; no deben confundirse con recuperación con contraseña.

Fuentes originales y condiciones de datos/licencias conservadas; el recuperador no concede derechos sobre ellas. Asistencia de IA declarada en el código, el protocolo y los recibos.
