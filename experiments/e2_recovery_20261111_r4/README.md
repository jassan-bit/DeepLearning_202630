# R4: conciliar un intento pendiente antes del calendario reducido

Registro del 11/11/2026. La campaña fijada en `e397b493def8f783e5be43d743af87861d6ad53f` conserva 1.801 resultados y un inicio sin resultado: evento 3632, trabajo `acace045a403bc4bb9a8962c5069b8c5c7a41228606ced185c62320e2795c6be`. El usuario informó una lista vacía de procesos de búsqueda en el Colab preparado, después de que el calendario reducido terminara con código 3.

R4 permite una sola repetición física del mismo trabajo lógico, configuración y semilla, con recibos de creación única. No inicia otra campaña, no altera los 1.801 resultados, no habilita otra recuperación R3 y no aumenta las 48 horas efectivas ya registradas. El resultado, tiempo y pasos del intento interrumpido siguen desconocidos; las mediciones del nuevo intento se registran aparte.

El controlador usa `--repair-only`: después de conciliar termina con `status: recovery_completed`. No continúa automáticamente el calendario antiguo de once orígenes. Sólo entonces corresponde iniciar de nuevo la sección 8 del notebook de tres orígenes. Un fallo de preparación, sincronización o disponibilidad GPU no autoriza otro ajuste. Un recibo físico iniciado sin observación bloquea otro reintento automático.

## Verificación realizada

- Inventario Merkle y hashes de los 5.436 registros públicos del snapshot validado.
- 1.801 resultados y exactamente un trabajo pendiente confirmados por el lector del diario.
- Límites iniciales de 24 horas y extensiones históricas 24→36→48 preservados. La extensión 36→48 se identifica por el evento 3516 y su SHA256 registrado.
- Pruebas de reconciliación, recibos, controlador y launcher con fixtures sintéticos; sin GPU, mercado ni escrituras a GitHub en esas pruebas.
- Fuente científica A3 inmutable. Las ayudas R4 se versionan por separado.

La lectura del snapshot carga conservadoramente 156.331,361 segundos frente a 172.800 disponibles. Esa carga incluye una asignación conservadora de sesión sin cierre; no es una medición de horas GPU ni consumo facturado.

## Ejecución en el Colab ya preparado

El launcher requiere la misma T4, fuente A3, Python aislado 3.12.12, datos e iTransformer preparados por las secciones 1–7. La nueva opción explícita es `RECOVER_PENDING_ONCE=True`; la opción antigua `RETRY_LOST_ONCE` no autoriza R4. Se fija el commit del controlador y el SHA256 del launcher y de `authorization.json` desde la celda de ejecución.

Al terminar se debe leer el informe único de esta invocación y sus campos `status`, `error_code`, `pending_job_ids` y `logical_completed_jobs`. No interpretar un `status.json` antiguo como resultado de R4. Este repositorio no tiene acceso directo al kernel privado ni acredita la repetición hasta que se observen sus recibos.
