# Inspección forense de la reproducción E1

Esta salida se obtuvo leyendo los 1395 ajustes ya existentes de la reproducción
del 6–7 de noviembre; **cero nuevos ajustes** y cero accesos a la reserva. Sus 8370
artefactos científicos se verificaron contra los hashes de la exportación ya
publicada antes de calcular los 4185 hashes canónicos nuevos. No se exportaron
pesos, escaladores, observaciones ni predicciones.

| Evidencia | Resultado | Alcance |
|---|---|---|
| Configuración, preprocesamiento y partición | 4185/4185 hashes originales coinciden | Identidad de esos archivos |
| Historias inner/refit | 0/2790 hashes coinciden | Diferencia de bytes; no cuantifica diferencias de valores |
| Checkpoints | 0/1395 hashes coinciden | El contenedor no demuestra por sí solo tensores distintos |
| Épocas seleccionadas | 1366/1395 coinciden; 29 diferencias de TCN | Divergencia científica real en al menos esas 29 trayectorias |
| Método de procesos | Ambos ejecutores usan `spawn` | Descarta el cambio supuesto de `fork` a `spawn` |
| Número de trabajadores | Original 3; reproducción 4 | Diferencia observada, causa no demostrada |
| Entorno BLAS | Original `setdefault`; reproducción fuerza uno | Entorno original efectivo desconocido |

Los CSV históricos de curvas no contienen tiempos; los tiempos no explican sus
hashes distintos. La coincidencia de época en GRU/LSTM tampoco acredita pérdidas
o pesos idénticos. La información original de plataforma, versiones, afinidad y
cuota no registra CPU/ISA y bibliotecas binarias con precisión suficiente para
atribuir una causa. **La causa exacta no está identificada.**

Una prueba sin entrenamiento cargó un checkpoint existente con `weights_only`
y lo guardó de nuevo bajo el mismo nombre `model.pt`: produjo bytes idénticos.
Guardarlo en `BytesIO` cambió el prefijo ZIP de `model/` a `archive/` y el hash,
con los mismos tensores. Los tests sintéticos también verifican la invariancia
canónica ante ese cambio. Esto demuestra una posibilidad general de
serialización; no demuestra que explique los 1395 hashes históricos distintos.

`epoch_discordances.csv` conserva las 29 diferencias, ordenadas por magnitud
absoluta y desempates por familia/semilla. `replay_canonical_hashes.csv` contiene
hashes de los valores de las dos curvas y del `state_dict`, con el algoritmo
declarado en el protocolo nuevo. Sólo se dispone de hashes de archivo originales:
no pueden reconstruirse los valores ni comparar canónicamente el original.

`recovery_search.json` documenta una búsqueda acotada: directorio original ausente,
archivos ZIP locales, árbol GitHub completo del commit de base, releases y cuatro
ejecuciones de publicación posteriores al original. No se recuperó ningún
payload original en esas ubicaciones. Las limitaciones de búsqueda remota quedan
explícitas; no se concluye que el estudio original no se ejecutara ni que no
existan copias en otro lugar.

El experimento separado `experiments/e1_replay_audit_20261107/protocol.json`
predeclara un subpanel dirigido de seis celdas y dos procesos independientes para
comprobar control numérico actual. Esta carpeta de inspección no contiene sus
resultados ni anticipa si superará los criterios exacto y numérico.
