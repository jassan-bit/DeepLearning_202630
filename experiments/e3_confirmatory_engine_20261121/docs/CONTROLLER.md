# Controlador ejecutable de campaña

`engine/controller.py` conecta las admisiones de datos, los 21 sistemas reales, el trabajador con checkpoints, GitHub, la planificación y la evaluación única. Importarlo no ejecuta ninguna etapa. El entrenamiento y la apertura de prueba exigen `activate=True`; el notebook y la CLI lo dejan desactivado inicialmente. Los estados de preparación o resultados de fixtures no acreditan R3.

## API y etapas

```python
from engine import controller

panels = {"F00": "/content/panels/F00", "F01": "/content/panels/F01",
          "F02": "/content/panels/F02", "F03": "/content/panels/F03"}
controller.prepare_case(source_root, case_root, panels, "planning",
                        immutable_source_commit, protocol_path)
controller.run_case(source_root, case_root, panels, store, activate=True)
planning = controller.predict_planning(source_root, case_root, panels, store)
# planning['predictions'] y planning['receipt'] alimentan engine.power.dimension.
```

`prepare_case(source_root, case_root, panel_dirs, stage, source_commit, protocol_path, power_receipt=None, *, software_fixture=False)` fija un caso nuevo y no entrena. `panel_dirs` es un diccionario F00–F03 o una lista de exactamente cuatro rutas en ese orden. Verifica el manifest de fuentes, el ancla pública, las implementaciones ejecutadas y el vendor, la identidad de candidata, la familia completa, el banco exacto y cada partición de desarrollo. No lee el payload test. Las admisiones iniciales contienen ajuste, selección, planificación y calibración; la prueba se prepara por separado después del sello de entrenamiento.

El plan inmutable usa `r3-engine-plan-<UUID>` o `r3-engine-final-<UUID>`. Los directorios son `jobs/F00-m00-s0000`, etc. Para planificación se usan los cinco índices 4091–4095 del banco archivado; para final, su prefijo de R entradas y R≥10. Los 16 métodos neuronales usan cada semilla y los cinco deterministas se ejecutan una sola vez por panel: 340 trabajos de planificación (320 ajustes neuronales y 20 sistemas deterministas). La comprobación de la campaña vuelve a exigir el producto cartesiano completo, sin omisiones o duplicaciones. `global_mse` conserva su lugar en la familia y su papel de comparación principal.

Los únicos cambios comunes de entrenamiento autorizados en campaña son presupuestos y política de parada; las recetas históricas siguen en `models`. `software_fixture=True` permite overrides pequeños y admisiones sintéticas para pruebas de planificación. Se marca en cada identidad y recibo y nunca puede cerrar un caso final. No existe un flag para entrenar sin respaldo.

`run_case(..., store, *, activate=False, max_jobs=None, checkpoint_interval_seconds=60.)` puede terminar un lote de trabajos y continuar después. `max_jobs` limita trabajos nuevos de esa llamada; no reduce el plan ni cambia R. Antes del primer ajuste, de cada trabajo y de reanudar, publica y relee un respaldo remoto autenticado. El callback síncrono del trabajador guarda también los checkpoints intermedios y el final/fallo de trabajo. Un fallo corta la llamada antes de otro ajuste. Los completos se omiten sólo después de `training.verify_job`, que vuelve a comprobar configuración, fuentes, puntero y estado seleccionado. Cada proceso verifica de nuevo los bytes de datos externos; una ruta diferente se admite si las identidades de desarrollo y la evidencia congelada coinciden.

La sesión remota abierta y la recuperación conservan incertidumbre física de cómputo. `unobserved_remote_compute_possible` se propaga del historial de recuperación al estado, sello y recibo final. Un checkpoint no convierte pasos perdidos o reservas de tiempo en consumo físico exacto.

## Export de planificación y potencia

`predict_planning(source_root, case_root, panel_dirs, store)` exige todos los trabajos completos antes de predecir. Lee sólo la partición dedicada `planning`. Crea:

- `predictions/planning_predictions.csv`: `panel,system,seed,date,y,prediction,known_floor_A,rfit,raw_prediction`.
- `predictions/planning_predictions.receipt.json`: contrato real de `power.load_planning`, hashes de datos, protocolo, métodos, manifest, checkpoint de cada trabajo y registros de intentos.
- `receipts/planning_data.json`: identidades de los cuatro paneles y auditoría de purgas.

`prediction` es sigma7 proyectada, `raw_prediction` conserva el pronóstico original y `known_floor_A` es una varianza conocida cuya raíz define el soporte. `rfit` se obtiene exclusivamente del preprocesamiento de ajuste. Las semillas deterministas se exportan como `deterministic`. Un segundo export reutiliza exactamente el CSV verificado, sin nuevos forwards. Un recibo sintético lleva `synthetic:true` y es rechazado por el dimensionador real.

Los eventos de intentos se retienen. El `active_seconds` del recibo completo es el acumulado cobrado por el trabajador; `active_seconds_is_charge_not_exact_elapsed` identifica reservas/incertidumbre. Eventos ya incluidos en ese acumulado aportan cero al sumatorio adicional, conservando su ledger y observaciones para evitar contabilizar el mismo trabajo dos veces. El planificador separa cargas, medición exacta y cómputo incierto.

## Contrato real de cierre final

La configuración entregada tiene `final.closed:false`. El controlador rechaza su cierre final actual; no cambia ese estado por renombrar una prueba o aceptar una afirmación `admitted:true`.

Un protocolo final nuevo debe fijar `final.R/T/B`, `final.test_panels` con `start/end/count` de F00–F03, `final.power_receipt:{path,sha256}` y `final.method_admission:{path,sha256}`. Los parámetros de `statistical_settings` y los del dimensionamiento deben coincidir. Debe conservar la familia e identidad de candidata.

La potencia debe proceder del dimensionador real con estado `DIMENSIONED_CONDITIONAL_EMPIRICAL_DGP`, ganador hasheado antes de su validación independiente, prefijo exacto de semillas, archivos de planificación verificables y el mismo diseño congelado de `configs/power_design.json`. Se verifican los 80 pares panel/rival únicos, los contadores MC de cada alternativa y sus límites Clopper–Pearson simultáneos; cada límite debe alcanzar 0,90. La simulación usa exactamente la regla operacional finita en B; la incertidumbre de contadores de cola se informa como diagnóstico separado, sin sustituir esa regla por otra imposible de aprobar. La potencia conjunta se exige reportada. Esto conserva el alcance condicional del DGP empírico, sin prometer potencia universal de mercado.

La admisión metodológica exige un reporte `PASSED_PREDECLARED_VALIDATION` con `validation_scope:"single_panel_temporal_family"`, `panels_per_replicate:1`, `final_panels_application:4`, 20 rivales, los tres métodos, nulos completos/parciales, cobertura y colas familiares. `validated_settings` debe contener exactamente R/T/B/bloque/HAC/horizonte/margen/referencia/alpha de la ejecución futura. El número cuatro declara dónde se aplicará la batería; **no dice que se simularon cuatro paneles**. Una réplica marginal no se copia cuatro veces para aumentar N o anunciar calibración conjunta.

El reporte enlaza `statistics_source_sha256` y el bundle `evidence` con `plan`, `freeze`, `runner` y `replicates`, cada uno `{path,sha256}` relativo al reporte o absoluto. Se admite JSONL o gzip con hash de los bytes comprimidos y del JSONL íntegro descomprimido. El plan identifica una réplica por escenario y parámetros exactos; el congelamiento enlaza sus fuentes. Cada fila contiene los 120 p/contadores de cola crudos y ajustados en el orden rival × método × EPA/práctico, los intervalos 99 % y su verdad simulada, y los gates de los veinte rivales.

El controlador recalcula los p `(tail_count+1)/(MC_reps+1)`, Holm120, los eventos FWER bajo cada nulo declarado y la cobertura de los intervalos 99 % desde sus límites/verdades. Exige CP95 superior ≤0,05 para el FWER por escenario y CP95 inferior ≥0,95 para la cobertura operacional, además de CP95 inferior ≥0,90 por rival bajo la alternativa fuerte prefijada. Los intervalos 95 % diagnósticos se conservan por separado y no se relabelan. No acepta un booleano de aprobación en lugar de esos contadores. Los archivos y hashes no prueban por sí solos la honestidad del custodio o la fecha material de una declaración.

**Paso de un panel a cuatro.** Bajo p marginalmente superuniformes para todos los nulos aplicables, Holm conserva control fuerte a nivel α bajo dependencia arbitraria: en el primer nulo verdadero rechazado quedan al menos m₀ nulos verdaderos, su p debe ser ≤α/m₀ y la cota de unión limita la probabilidad a α. No requiere independencia entre paneles o modelos. Esta justificación permite la familia Holm180 de inferioridad EPA de los tres adicionales; no convierte una simulación marginal finita en prueba universal de superuniformidad. El veto direccional usa DM/bootstrap; GW condicional sigue en la multiplicidad, pero su rechazo unsigned no demuestra inferioridad de la media. La decisión favorable además requiere la conjunción principal y consistencia adicional. La potencia conjunta real de toda esa decisión se estudia aparte sobre los cuatro paneles de planificación y su acoplamiento declarado, con cotas cuando faltan covarianzas identificadas.

V1/V2 fallidos no satisfacen ese contrato. Una validación acotada que no cubra esta familia y configuración tampoco lo satisface, incluso si pasa sus propios escenarios. No hay un bloqueo permanente codificado: una construcción nueva con evidencia admisible puede cumplir el contrato.

## Sello y evaluación única

```python
controller.seal_training(source_root, final_case, development_panels, store)
# El custodio prepara ahora test en nuevos directorios con autorización real.
result = controller.evaluate_once(source_root, final_case, released_test_panels,
                                  store, activate=True)
```

`seal_training` verifica todos los checkpoints seleccionados, fija `selected_manifest.json`, guarda los pronósticos de ajuste/selección/calibración y ajusta `metrics/calibrators.json` exclusivamente con calibración. Exige calibradores suficientes para todos los sistemas. `training_seal.json` enlaza los pesos, los CSV y el hash de archivo de calibradores. Se publica y se relee antes de permitir test. Repetir el sello verifica los artefactos deterministas guardados.

El test-spec del protocolo fija `source_identity_sha256 = controller.source_identity_sha256(provenance)`, el digest canónico de `{provider,instrument,calendar}` conocido antes de adquirir el futuro. Los hashes de archivos y versiones raw se fijan aparte al adquirirlos; no se introducen en esa identidad previa. `exposure_evidence_sha256` corresponde a un inventario independiente de exposición/reserva anterior al cierre, no al formulario final que contiene `protocol_sha256`. Al liberar la prueba, `exposure.form.exposure_inventory_sha256` y `authorization.reserve_inventory:{path,sha256}` deben identificar ese mismo archivo real. El controlador lee y comprueba sus bytes antes de cualquier carga de test. Esto evita un ciclo de hashes y evita dejar los dos compromisos del protocolo sin comprobación material.

La nueva admisión final puede contener sólo test o copiar idénticas particiones de desarrollo. Su autorización auténtica debe enlazar el protocolo y el hash del manifest de checkpoints ya sellado, fijar `sources_methods_size_frozen:true`, aportar actor y evidencia con bytes verificables, y autorizar preparación de test. Se exigen fechas y tamaño exactos, y soporte test posterior al soporte de desarrollo con el mismo índice de barras. El custodio debe preservar `provenance.bar_index_origin`, la primera sesión raw observada, al preparar el raw manifest completo. El origen de la nueva admisión debe ser exactamente el del desarrollo congelado: reiniciar el contador en una colección test-only no demuestra una purga. Antes de crear un caso final, el controlador exige ese origen y comprueba que el inicio de test sea posterior a todas las fechas de desarrollo, sin leer etiquetas finales. No se exige que el hash de toda la nueva admisión sea idéntico al de desarrollo.

Antes del primer `data.load_partition(...,"test",allow_test=True)`, el controlador persiste y respalda un `evaluation_receipt.json` con `eval_id`, identidad de protocolo/plan/paneles, `sealed:true`, hashes de calibración/selección y estado `started`. La identidad final incluye los metadatos de custodia en `receipts/final_admissions.json`, sin arrays de entrada.

Cada trabajo tiene `evaluation/<job_id>.pending.json`, guardado remotamente **antes** de cargar test o ejecutar el forward, y después `evaluation/<job_id>.json` con sus predicciones y hash. Si queda pendiente sin cache, la ejecución se detiene: no repite automáticamente una consulta cuyo estado quedó incierto. Se conservan el fallo y la identidad; una nueva corrida con otra identidad no es una forma de borrar esa exposición. Una cache válida se recupera sin nuevo forward. Un resultado completo vuelve a exportar el mismo CSV sin cargar de nuevo datos test; las rutas de CSV de desarrollo se resuelven contra el caso restaurado y sus hashes originales.

La finalización conserva todas las predicciones en `predictions/test_predictions.csv`, cambia el mismo recibo a `completed` y respalda de nuevo. Incluye ambos hashes de calibradores: `calibrators_sha256` del payload semántico para assessment y `calibrators_file_sha256` de bytes para el sello. Las banderas `calibrators_frozen_before_test`, `selected_checkpoints_verified_before_test` y `evaluation_sealed` documentan las barreras comprobadas. La conclusión permanece `FINAL_EVALUATED_ANALYSIS_PENDING`, `R3_accredited:false`; `engine.assessment` consume sólo esos CSV y artefactos guardados para comprobar la regla completa, sin forwards.

## Publicación y recuperación

La clase real de almacenamiento está descrita en [GITHUB_STORAGE.md](GITHUB_STORAGE.md). La publicación de derivados exige `allow_derived_publication:true` en los cuatro recibos de derechos. De lo contrario los resultados se conservan sólo cifrados. La política pública predeterminada es fija y acotada: recibos/metadata y un inventario path/SHA256/bytes; nunca CSV completos. Todos los CSV íntegros permanecen en los fragmentos cifrados y en `export_case`, sin muestrear filas o elegir resultados favorables. El ZIP público tiene un techo prospectivo de 2 MiB sin comprimir y cada recibo un máximo de 256 KiB, por debajo del límite de transporte de 8 MiB. Antes del sello de evaluación de un caso final, el controlador conserva sus CSV de desarrollo en el respaldo privado. Nunca publica NPZ de inputs, fuentes o pesos en claro.

`status_case(source_root, case_root, panel_dirs=None, store=None)` es de sólo lectura y no abre test. Los marcadores de completitud mostrados no se presentan como verificación de modelos. `export_case(source_root, case_root, output, *, panel_dirs=None, store=None)` empaqueta derivados/recibos guardados, excluye pesos e inputs y no vuelve a consultar modelos. El ZIP se escribe fuera del caso. Si hay respaldo disponible, también reconcilia su estado remoto. Las identidades de plan/fuentes no se modifican para reubicar datos; se proporcionan las rutas reacquiridas por separado.

## Pruebas ejecutadas

```bash
cd /workspace/e3_confirmatory_engine_20261121
/workspace/e3_fixed_state_runtime/bin/python -m unittest discover -s tests -p test_controller.py -v
```

El log está en `qa/storage/CONTROLLER_UNITTEST.log`. Incluye una cadena real controlador → trabajador de candidata → `GithubStore` → Git REST simulado → restauración y `verify_job`, con una época de datos sintéticos. Las pruebas unitarias de ciclo final aíslan con mocks la admisión metodológica y el producto cartesiano para estudiar la consulta única; su resultado **no admite** datos finales ni demuestra validez estadística. La batería independiente de almacenamiento ejecuta cifrado real, fragmentos mayores del antiguo límite de 20 MiB y recuperación de fallos. No hubo ajustes de mercado, lecturas de test real o escrituras remotas con credenciales del propietario.

## Progreso y análisis durable

El controlador emite JSON con `flush=True` al verificar cada respaldo y al iniciar, completar o reutilizar un trabajo. Sólo contiene identificadores, sistema, panel, semilla, estado y commit remoto; no imprime arrays o credenciales. El notebook transmite esas líneas para que una campaña larga tenga progreso observable.

La CLI `assess` escribe `metrics/assessment.json` y `analysis_receipt.json` con hashes; no atribuye durabilidad remota a una escritura local. Después se ejecuta `sync` sin `--panels` para verificar el respaldo de análisis/recibos. `sync_case` y `export_case` pueden funcionar sin directorios de datos y no consultan modelos. Las pruebas de integración crean un análisis de software explícitamente ficticio, lo respaldan mediante Git REST simulado y comprueban sus bytes después de restaurar. Esas pruebas no son un informe estadístico real ni una declaración auténtica del autor.
