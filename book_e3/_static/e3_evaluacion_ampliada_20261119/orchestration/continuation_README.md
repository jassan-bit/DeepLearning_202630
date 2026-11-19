# Continuación verificable de la auditoría ampliada

Este módulo conserva bytes y recupera archivos; no entrena modelos ni acredita
R1/R3. Se preparó con asistencia de IA declarada. Las pruebas usan una API GitHub
falsa en memoria y checkpoints sintéticos: no realizaron escrituras externas.

## API

- `discover_runs(token, api, source=None)` enumera ejecuciones lógicas y devuelve
  su fuente, hash del protocolo, última parte, URL, `stage_status` y
  `analysis_complete`. Un resumen completado permite informar su URL sin
  descargar de nuevo todos los pesos.
- `save_changed(output, token, source, master_run_id, protocol_sha256, api,
  statefile=None)` guarda sólo archivos nuevos o cambiados. Devuelve el commit
  de cabecera, partes confirmadas y ruta del estado local.
- `restore(output, token, source, protocol_sha256, api, master_run_id=None,
  run_url=None, statefile=None)` reconstruye la versión más reciente de cada
  archivo, incluso cuando la URL suministrada identifica una parte anterior.
  Devuelve las identidades verificadas y la carpeta restaurada.

`api` es el módulo histórico `colab_github.py`, con el token mantenido en memoria.
La copia cargada de `github_store.py` recibe el perfil `audit` y permite SVG para
las figuras científicas. Ningún archivo histórico se modifica.

## Cadena y presupuestos

Cada parte ocupa una rama nueva `e3-extended-audit-<UTCstamp>` y un subárbol
`runs/e3_extended_audit/<UTCstamp>/`. La primera tiene como padre el commit del
código. Las siguientes tienen como padre el commit de la parte anterior. De
este modo, una ref de cabecera mantiene alcanzables los checkpoints antiguos,
incluso si desaparece una ref intermedia.

Cada recibo liga fuente, protocolo SHA-256, ejecución lógica, número secuencial,
recibo anterior y manifiesto de archivos. Se comprueban SHA-256, objeto Git,
tamaño y modo regular. La preservación de los demás archivos del repositorio se
verifica comparando hashes de subárboles hermanos; sólo se recorre el subárbol
nuevo. No se descarga repetidamente todo el árbol fuente.

Límites por parte: menos de 192 MiB, hasta 900 archivos de carga útil y 48 MiB por
archivo. Los `.pt` se conservan como bytes originales; no se deserializan ni se
envuelven en ZIP anidados. Los archivos grandes se reparten entre partes. Una
sola entrada que supere el límite se rechaza explícitamente.

## Uso con un trabajador activo

La carpeta entregada a `save_changed` debe permanecer estable mientras se
construye su parte. El controlador puede materializar una **vista de
persistencia**, separada de la carpeta donde siguen escribiéndose checkpoints.
Copiará allí los trabajos cerrados y metadatos coherentes. No necesita detener
la GPU ni consumir artificialmente su presupuesto de entrenamiento durante la
subida.

El estado liga fuente, protocolo y ejecución, no una ruta concreta. Por ello se
puede restaurar en la carpeta del trabajador y guardar luego desde otra vista,
usando el mismo `statefile` externo. La prueba de cambio de carpeta cubre ese
ciclo y una nueva restauración posterior. La vista debe conservar todos los
archivos previamente confirmados: eliminar uno no se interpreta como una
solicitud autorizada de borrado remoto.

La recuperación verifica todas las partes y todos los conflictos antes de
escribir. No sustituye ediciones locales desconocidas. Las bifurcaciones
ambiguas, cadenas incompletas, modos symlink y bytes alterados se rechazan. Los
tokens se buscan también dentro de ZIP/NPZ/PT comprimidos, sin ejecutar pickle.

Una interrupción de la subida conserva un snapshot local exacto y pendiente
antes de publicar la ref. Su reintento recupera el acuse perdido y no cuenta
otra vez el mismo trabajo como ajuste nuevo. Reiniciar Colab pierde la copia
local aún no confirmada; las partes confirmadas en GitHub siguen recuperables.
No se promete recuperar un checkpoint que nunca llegó a guardarse.

## Verificación

`python -m unittest discover -s experiments/e3_evaluacion_ampliada_20261119/orchestration -p test_continuation.py -v`

Las 15 pruebas cubren restauración entre sesiones, avance de checkpoints,
cambio de carpeta con vista estable, estado perdido, reintento con acuse
perdido, figuras SVG, límites de partes, conservación del código fuente y
rechazo de corrupción, ambigüedad, secretos y cambios locales. El recibo
`continuation_verification.json` registra hashes y alcance.
