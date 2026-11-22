# Respaldo GitHub del motor R3

Implementado: `engine.storage.GithubStore`, versión `e3-temporal-confirmation-github-1`. Es una implementación REST real de GitHub, probada mediante objetos Git y referencias simuladas, sin escrituras remotas en estas pruebas. La autorización, el repositorio y los secretos del propietario todavía son necesarios para demostrar un respaldo real de esta campaña. Ninguna prueba de software inicia ajustes ni abre etiquetas reservadas.

La referencia de diseño atribuible es `e3_seoi_v5_pilot_20261121/seoi_v5/github_store.py` y su controlador. Se conservan cifrado autenticado, transacción pendiente, lectura remota y protección frente a otra VM; la identidad de almacenamiento y los casos son nuevos. Se rechazan casos `seoi-v5-*`. Sólo se aceptan `r3-engine-plan-*` y `r3-engine-final-*`.

## Contrato

```python
from engine.storage import GithubStore

store = GithubStore(
    token, backup_secret, repository, branch,
    source_commit=immutable_source_commit, prefix="runs/r3/case-001",
)
remote = store.status()                    # autentica un respaldo existente
receipt = store.backup(case_root, public_files={}, stage="before_any_fit")
assert receipt["remote_verified"] is True
restored = store.restore(new_case_root)     # recibo autenticado o None si no existe rama
```

`repository` tiene forma `propietario/repositorio`; `branch`, `prefix` y `source_commit` son explícitos. La rama debe ser nueva para el primer respaldo. Una rama existente sin estado administrado en el prefijo se rechaza. El commit de fuentes es un SHA completo de 40 caracteres y debe coincidir con `plan.json.source_commit`. El SHA canónico de todo `plan.json` enlaza los datos, protocolo, familias, semillas y oportunidades de ajuste sin depender de nombres opcionales de esos campos.

`backup(case_root, public_files=None, *, stage="checkpoint")` devuelve `remote_verified`, `commit`, `head_commit`, `source_commit`, `case_id`, `plan_sha256`, `checkpoint_sha256` y `public_results_sha256`. Después de un fallo con `.github_pending`, la primera llamada reconcilia **exactamente** la transacción pendiente. El controlador debe llamar de nuevo antes de continuar si el estado local avanzó desde ese snapshot. Nunca debe interpretar ese primer recibo como cobertura de cambios locales posteriores.

`restore(case_root)` exige un destino inexistente. Reconstruye y verifica todos los archivos, punteros y hashes antes de instalarlo por renombrado atómico. No deserializa checkpoints de PyTorch durante el respaldo o restauración. El trabajador debe verificar la identidad del checkpoint antes de cargarlo. Los datos de entrada se deben adquirir de nuevo y verificar por sus hashes; no están en el respaldo.

## Contenido y límites

El índice privado `checkpoint.aesgcm` contiene un manifiesto autenticado de archivos y fragmentos. Los datos privados viven en `chunks/<dos-caracteres>/<sha256-del-cifrado>.aesgcm`. Cada objeto Git ocupa como máximo 8 MiB, incluido el sobre criptográfico. El tamaño de fragmento predeterminado es `8 MiB - 8192` bytes; sólo se retiene un fragmento de pesos en memoria a la vez. No existe el antiguo límite de ZIP privado de 20 MiB. Los límites de defensa son 16 GiB por archivo, 512 GiB por snapshot y un manifiesto menor de 8 MiB. Esto es capacidad del formato, no una promesa de cuota ni de viabilidad económica de GitHub para cientos de GiB. El dimensionamiento debe presupuestar bytes y tiempo reales. Un archivo sintético de 21 MiB se verifica en la prueba de recuperación.

Se incluyen exclusivamente:

- El plan y los metadatos de estado, sesión, fallos, selección, protocolo, cómputo y recibos enumerados en `ROOT_PRIVATE`.
- Para cada `plan.jobs[].id` —también se admite `job_id`—, los JSON conocidos del trabajador enumerados en `JOB_META`.
- El `.pt` que identifica `checkpoint.json`, o los punteros `latest`, `best` y `selected` explícitos. Cada puntero exige `file` o `path`, `bytes` y `sha256`. El snapshot del trabajador puede contener estado actual, optimizador, RNG y todos los mejores estados de sus fases; no se duplican snapshots históricos sin referencia.
- Resultados derivados y recibos de los directorios privados `receipts`, `evaluation`, `evaluations`, `planning`, `predictions` y `metrics`, con extensiones definidas en código.

Se excluyen inputs y datasets, fuentes, tokens, locks, temporales e históricos de pesos no referenciados. Se rechazan symlinks y rutas con escape, incluso después de autenticar el manifiesto. El plan debe mantener las identidades inmutables; una configuración local separada puede señalar dónde se reacquirieron los datos.

`public_results.zip` siempre se crea, pero está vacío salvo la selección explícita `public_files={"metrics/resumen.json": bytes, "docs/README.md": Path(...)}`. Las únicas raíces públicas son `predictions`, `metrics`, `status`, `docs` y `receipts`. Se aceptan archivos derivados JSON/JSONL/CSV/TSV/MD/TXT/NPY/NPZ; el código no permite publicar `.pt`, código fuente o directorios de inputs. El llamador sigue siendo responsable de que los bytes autorizados sean realmente resultados derivados. Se escanean todas las publicaciones y archivos privados para detectar los secretos activos, incluso cuando caen entre dos fragmentos.

El ZIP público ocupa como máximo 8 MiB. Seleccione resúmenes si el export completo supera ese límite; todas las predicciones pueden conservarse en el snapshot privado y exportarse localmente. En un caso final, **cualquier** archivo de `predictions/` exige `evaluation_sealed.json` o `evaluation_receipt.json` con `sealed: true` o `evaluation_sealed: true` y el `case_id` exacto. También se exige sello para nombres públicos que declaren test/final en casos de planificación. No se considera que cambiar el nombre de un archivo autorice publicar predicciones finales.

## Cifrado, atomicidad y concurrencia

Cada fragmento nuevo y cada índice usa AES-256-GCM, sal aleatoria de 16 bytes, nonce aleatorio de 12 bytes y Scrypt fijo `n=32768, r=8, p=1, length=32`. El AAD enlaza versión, repositorio, rama, prefijo, fuentes, caso, hash de plan, tipo de objeto y hash del contenido. Los parámetros KDF recibidos nunca pueden elevarse para agotar memoria. Una clave errónea falla **antes** de crear pendientes o escribir en GitHub cuando existe un respaldo. No hay rotación implícita de claves.

`checkpoint.aesgcm`, `status.json`, `public_results.zip` y los nuevos shards se publican en un solo commit descendiente del HEAD previamente verificado. Se relee la referencia inmediatamente antes de actualizarla. Las actualizaciones usan `force: false`; GitHub rechaza actualizaciones concurrentes que no sean fast-forward. La API de referencias Git REST no ofrece un parámetro literal compare-and-swap/lease: no se afirma protección ante una fuerza externa que mueva la rama hacia un ancestro entre lectura y PATCH. Evite administradores que hagan force-push en esta rama y utilice un único controlador de campaña.

La construcción de árboles usa subárboles inmediatos y lotes de como máximo 128 entradas. Nunca vuelve a enviar como JSON todo el árbol gigante de fuentes. Los shards idénticos ya autenticados se reutilizan; los shards obsoletos desaparecen del árbol actual. Los commits históricos conservan versiones cifradas anteriores, como corresponde a Git. Esta retención debe incluirse en la estimación de almacenamiento.

Después de actualizar la referencia se vuelven a descargar los bytes publicados y comprobar SHA-1 de objeto Git, SHA-256, longitud y HEAD. La lectura inicial de otro proceso autentica el índice y cada fragmento, reconstruye los hashes completos y verifica los punteros. La instancia guarda en memoria un inventario de objetos inmutables verificados para evitar subir otra vez los fragmentos sin cambios. Cambiar HEAD obliga a comprobar el estado remoto; una VM cuyo recibo local apunta a otro HEAD se detiene, sin sobrescribirlo. Los bytes de punteros ya autenticados se reutilizan únicamente para el mismo contexto y registro completo de manifiesto durante verificaciones sin destino. Sus referencias se vuelven a analizar en cada snapshot contra el inventario actual, incluyendo hash y tamaño del `.pt`: no se omite ese control. Así, un checkpoint intermedio no vuelve a descargar todos los punteros de trabajos antiguos. Ante un HEAD desconocido se relee la asociación ruta/objeto Git; durante restauración siempre se descargan, descifran y verifican los bytes completos, aunque exista caché en esa instancia.

La transacción se instala atómicamente en `.github_pending` **antes** de cualquier mutación remota. Un fallo de subida o actualización conserva esos mismos bytes para reintento. Si GitHub confirmó el commit pero se perdió la respuesta o el proceso cayó antes del recibo, el siguiente proceso reconcilia el snapshot ya publicado. La limpieza se retira mediante renombrado; una interrupción de limpieza no crea un pendiente incompleto.

## Controlador y secretos

El controlador del motor debe exigir un recibo remoto verificado antes del primer ajuste y antes de reanudar, invocar `backup` en checkpoints intermedios y al terminar/fallar cada trabajo, y detener nuevos ajustes ante cualquier error de guardado. La clase de almacenamiento **no inicia entrenamientos** y no puede imponer por sí misma que un llamador respete esa barrera. Las pruebas de integración del controlador verifican esa política por separado.

Use Colab Secrets (`userdata.get`) o `getpass.getpass`/stdin privado. Nunca pase secretos como argumentos de CLI, celdas guardadas, URL, contenido de plan, variables impresas o archivos publicados. El token necesita lectura del repositorio y permisos `Contents: Read and write` para crear objetos y actualizar la rama. Un repositorio privado reduce exposición de metadatos; los pesos permanecen cifrados incluso en uno público. La contraseña independiente tiene al menos 32 caracteres, se conserva fuera de GitHub y no puede coincidir con el token.

Los errores contienen códigos fijos. Los 401/403 de permisos no se reintentan como límites. Los 403 con `Retry-After`/secondary-limit o cuota primaria agotada y los 429 esperan según los encabezados; 408/5xx y fallos de transporte usan reintentos acotados del mismo cuerpo. Un 404 se diagnostica como objeto/repositorio ausente o permiso de lectura, porque GitHub oculta recursos privados con ese código. Las mutaciones se espacian al menos 1,25 s; las esperas se dividen en segmentos de 30 s. El controlador puede conservar evidencia local y reportar el código saneado, nunca el objeto HTTP ni su cuerpo.

Una recuperación añade `remote_restore.json` y `recovery_history.json`. Si la sesión guardada estaba abierta, costes y pasos son cotas inferiores: puede existir cómputo perdido después del último checkpoint. La recuperación no inventa consumo ni declara automáticamente comparabilidad presupuestaria.

## Evidencia ejecutada

```bash
/workspace/e3_fixed_state_runtime/bin/python -m unittest discover \
  -s tests -p test_storage.py -v
```

El resultado y las versiones realmente usados están en `qa/storage/UNITTEST.log` y `qa/storage/RESULTS.json`. Son pruebas offline de criptografía, Git REST simulado, recuperación y barreras; no constituyen una verificación remota con credenciales del propietario ni una evaluación confirmatoria.

## Política pública acotada, fijada antes de la campaña

El controlador no coloca predicciones CSV completas en `public_results.zip`. El ZIP público contiene sólo recibos de una lista fija y un inventario determinista de archivos derivados con ruta, SHA256, tamaño y ubicación privada, siempre que los cuatro recibos permitan publicar derivados. Hay un techo de 256 KiB por recibo y 2 MiB sin comprimir para el conjunto; un recibo demasiado grande permanece íntegro en el respaldo cifrado. El inventario público conserva su hash y conteo si su lista completa supera el límite de metadata; el manifiesto privado autenticado permite reconstruirla. Ninguna fila se selecciona por su rendimiento.

Todas las predicciones y análisis completos permanecen en los fragmentos cifrados de GitHub y en el ZIP completo que produce `export_case`. Se requiere la clave independiente para restaurarlos. La política evita que la suma de las tres condiciones de evaluación y los diagnósticos de cientos de trabajos bloquee un guardado al superar el límite de 8 MiB del archivo público. La prueba sintética conserva y restaura un CSV completo mayor de 8 MiB mientras verifica que el ZIP público queda por debajo de 2 MiB. No modifica ni resume los archivos privados originales.
