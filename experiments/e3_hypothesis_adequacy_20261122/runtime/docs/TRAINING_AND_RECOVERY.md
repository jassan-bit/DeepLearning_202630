# Entrenamiento y recuperación

`engine.training.fit_job(job_spec, fit_data, selection_data, job_dir,
on_checkpoint=barrier)` implementa entrenamiento real en CPU, con dos hilos.
No descubre paneles, abre particiones ni consulta la reserva. Recibe `raw[N,7,5]`,
`y[N]` y metadatos suministrados por el controlador. `y` es **un sigma7 por
ventana**, no siete horizontes. Las estadísticas se estiman exclusivamente con
ajuste. La pérdida pertenece al adaptador congelado; la selección supervisada
minimiza MSE de sigma7 proyectado, dividido por la escala de etiquetas de ajuste
al cuadrado. Los empates conservan la primera época.

La identidad mínima es `{model, seed, protocol_sha256, identity}`. `identity`
debe contener las identidades admitidas de fuente, candidata y datos del
controlador. La solicitud inmutable incluye además hashes reales de **todos**
los arrays suministrados, preprocesamiento, configuración de receta, fuentes
verificadas, Torch, NumPy y fases. Cambiar cualquiera de esos elementos requiere
otro directorio/identidad; nunca sobrescribe una campaña anterior.

Los ajustes de receta proceden de `models.model_config`: tasa, batch, decay,
épocas y paciencia propios de cada sistema. E2 conserva AdamW con exclusión de
bias/norm del decay y el scheduler histórico warmup/coseno, incluido warmup
0,1 de PatchTST. Candidata y controles propios conservan AdamW constante. No
hay búsqueda ni selección de hiperparámetros en el entrenador. Los overrides
explícitos de `job_spec.training` se fijan dentro de la identidad del protocolo;
los tests usan overrides mínimos declarados como `software_fixture`.

## Un presupuesto por sistema

El presupuesto por defecto es **120 segundos nominales de tiempo activo**, con
reservas de caída de cinco segundos por época. Son límites de control,
no consumo medido ni garantías de interrupción a los 120 segundos de reloj.
Cada operación atómica puede sobrepasar un límite. Ajuste/selección de todas las
fases comparten el presupuesto; SAE asigna previamente un tercio a AE1, otro a
AE2 y otro al predictor. Se registran separadamente setup, I/O de checkpoints,
respaldo y pausas; esos costes no se presentan como incluidos en los 120
segundos. El recibo declara expresamente las exclusiones. El límite de coste
real de una campaña debe sumar esos conceptos en su presupuesto externo.

Las reglas de parada se fijan dentro del hash de solicitud: máximo de épocas,
paciencia o presupuesto nominal. La parada por presupuesto devuelve el mejor
estado sólo si existe al menos una época completa validada en **cada** fase.
Una época parcial al alcanzar el límite se revierte, se registra y se cobra;
el sobrepaso cooperativo observado permanece en el recibo. Si falta una fase
validada, el trabajo sigue incompleto.

Las fases auxiliares reconstruyen entradas; cada una restaura su mejor estado
validado antes de transferir los encoders. Sólo después se admite la siguiente
fase. No se aceptan pesos iniciales ni una época parcial como un ajuste
completado. No se reinician las cargas acumuladas al reanudar.

Los sistemas deterministas producen un solo sistema, no réplicas independientes
por semilla. Las reglas analíticas no hacen pasos de optimizador. ARIMA calibra
una vez sus parámetros exclusivamente con ajuste, con su límite cooperativo de
iteraciones; la evaluación causal posterior aplica esos parámetros sin volver
a calibrarlos. Una calibración interrumpida reserva el presupuesto restante
completo, por lo que no puede repetirse gratis tras una caída.

## Contrato durable

Cada `snapshot_<uuid>.pt` contiene pesos actuales, mejor estado validado por
fase, estados AdamW, RNG Python/NumPy/Torch, época, pasos comprometidos,
preprocesamiento, cargas y estado de fases. Los optimizadores de fases
completadas permanecen dentro del último payload. Se escribe en archivo
temporal, se ejecuta fsync y se publica un payload inmutable. Sólo entonces se
mueve atómicamente `checkpoint.json`, que contiene `file`, `bytes`, `sha256` y
`request_sha256`. El payload al que apunta basta para recuperar; los anteriores
pueden retenerse para auditoría. Los lectores verifican bytes y hashes antes de
deserializar el checkpoint propio. No se admiten snapshots arbitrarios de
terceros.

`epoch_pending.json` se hace durable **antes** de cualquier paso. Si no existe
un compromiso completo al reiniciar, el modelo/Adam/RNG vuelven al último
snapshot y se descuenta una vez su reserva declarada, o un tiempo abortado
observado mayor. Se conservan intentos, fallos y pasos abortados conocidos,
pero el total real perdido queda marcado como desconocido. No se inventan
tiempos exactos a partir de la reserva. Si la caída ocurre entre publicar el
snapshot y borrar el marcador, el identificador ya comprometido evita cobrar
dos veces.

`running.json` conserva el inicio original y `done.json` certifica el checkpoint
seleccionado. `verify_job(job_spec, job_dir)` comprueba identidad, fuentes,
payload y recibo antes de omitir un completo. Si se publica el payload completo
antes de su recibo, `fit_job` puede reconstruir exactamente ese recibo sin
entrenar de nuevo. Un bloqueo por directorio evita dos escritores simultáneos.
Se rechazan directorios, punteros o archivos que atraviesen symlinks.

`on_checkpoint(directory, event)` es una barrera **sincrónica**: el controlador
de GitHub debe guardar, leer remotamente y verificar hashes antes de retornar.
Se llama antes del primer paso, tras cada época, en los límites de batch si
transcurren 60 segundos, y después de cada trabajo completo. La garantía
temporal es cooperativa; una operación larga no puede interrumpirse a mitad.
Una excepción o retorno `False` produce `DurabilityError`, conserva evidencia
local y detiene inmediatamente los pasos posteriores. El controlador debe
propagarlo para detener otros trabajos. `None` permite pruebas locales aisladas;
el controlador de campaña exige la barrera GitHub inicial verificada.

Archivos privados por trabajo: `request.json`, `checkpoint.json`, el snapshot
apuntado, `epoch_pending.json` si existe, `running.json`, `done.json`,
`resources.json`, `attempts.json`, `failures.json`, `curve.json`. Los temporales y
`job.lock` no son evidencia remota necesaria. El último I/O no puede medir su
propia duración dentro del mismo payload; el recibo identifica esa limitación.

## Predicción y auditoría

`predict_job(job_spec, partition, job_dir, batch_size=256)` carga los pesos
**seleccionados**, devuelve `raw_sigma7`, `projected_sigma7`, `floor` y hashes,
y restaura el RNG del llamante. No requiere etiquetas ni abre archivos de datos.
Las matrices de historia por fila se cortan por batch; el eje causal global
`observed_sigma_since_fit` permanece intacto para ARIMA. El loader de datos
valida el corte exclusivo de historia de cada fila. La predicción no modifica
el checkpoint ni recalibra ningún parámetro.

`export_job_resources(job_dir)` conserva separadamente cargas comprometidas,
reservas pendientes, intentos y fallos, también para trabajos incompletos.
Un archivo de recursos nunca convierte un intento en ajuste completado.

Pruebas de software (sin datos de mercado ni apertura de test):

```bash
python -m pytest -q tests/test_training.py
```

Las pruebas verifican pasos AdamW reales en fixtures pequeños, reanudación
idéntica, mejor checkpoint, tres fases, reservas de épocas interrumpidas,
fallo de respaldo antes/después de entrenar, omisión verificada, corrupción,
identidades, symlinks y predicción reproducible. Los registros de ejecución se
guardan en `qa/training/`. No constituyen experimentos confirmatorios ni
acreditación R3.

## Identidad de software al recuperar un trabajo

La versión `e3-confirmatory-training-2` fija en `request.json` el objeto
`immutable_runtime`: implementación y versión **mayor.menor de Python**, versiones
exactas de Torch, NumPy, SciPy y statsmodels, y el perfil CPU de dos hilos que
posee el trabajador. Esa identidad se comprueba antes de reanudar y también al
verificar un trabajo completo o cargar sus predicciones. Cambiar el intérprete
Colab o una biblioteca numérica obliga a conservar el caso anterior y preparar
una identidad de campaña distinta; no se omiten trabajos completos de otra
cohorte como si pertenecieran a la actual.

La versión completa de Python, su build, plataforma, arquitectura y procesador se
registran en el checkpoint inicial y en los recibos/recursos como observaciones
informativas. No son una restricción a un modelo concreto de GPU o CPU, ni una
medición de coste. La observación inicial permanece idéntica al recuperar un
estado: no se reescribe retrospectivamente con los datos del nuevo anfitrión.
Una discrepancia crítica bloquea la operación antes de pasos nuevos y conserva
los bytes de solicitudes, checkpoints y recibos existentes.

`qa/training/RUNTIME_IDENTITY_TESTS.log` conserva la prueba con fixtures del
bloqueo de cohortes distintas, incluyendo trabajos completos e incompletos, y la
recuperación permitida cuando sólo cambian observaciones informativas. No hubo
ajustes de mercado ni acceso a reserva final.
