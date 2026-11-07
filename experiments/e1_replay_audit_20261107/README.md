# Auditoría numérica y repetibilidad actual del E1 D7

La reproducción completa ya publicada conserva las recetas, preprocesamientos y
particiones originales, pero no coincide con los hashes históricos de las curvas
y checkpoints. Hay 29 cambios de época seleccionada, todos de TCN: al menos esos
ajustes tienen una diferencia científica real. Los otros hashes distintos no
permiten medir diferencias numéricas sin los archivos originales.

La fuente registrada original y el ejecutor de reproducción usan `spawn`.
El original usó tres trabajadores y la reproducción cuatro. El original conserva
valores de entorno BLAS preexistentes mediante `setdefault`; la reproducción los
fija a uno. Estas diferencias son hipótesis de trabajo, no causas demostradas.
Las versiones, plataforma, afinidad y cuota declaradas de CPU coinciden en los campos
comunes registrados. No se registraron el modelo de CPU, despacho ISA ni las
bibliotecas binarias originales; esos campos no acreditan hardware idéntico.
Las curvas CSV no tienen columnas de tiempo. Un cambio de serialización puede
cambiar un hash de checkpoint sin cambiar tensores; se verificó esa posibilidad
con objetos sintéticos y una reserialización de un checkpoint existente.

Este plan se publica después de esos resultados y antes de los nuevos ajustes.
Comprueba repetibilidad actual en seis celdas diagnósticas: las tres familias con
la primera semilla registrada, 8101, y los tres mayores cambios absolutos de época
entre el original y la reproducción, desempates por familia y semilla. Resultan
TCN 8110, 8416 y 8552. La selección usa resultados ya conocidos y se declara como
diagnóstico dirigido; no estima la repetibilidad del panel completo de 465
semillas ni la exactitud del estudio original.

Se ejecutan dos réplicas desde cero en invocaciones independientes de Python, cada
una con un pool nuevo de tres procesos `spawn`, las versiones originales y el
entorno fijado. Son doce trayectorias y 24 etapas de ajuste. Las recetas y los
módulos científicos históricos permanecen intactos. Sólo se leen los derivados de
desarrollo 2023–2024; no se consulta ni puntúa la reserva y no se emiten pronósticos.

El criterio principal exige idénticas épocas y hashes canónicos de pérdidas,
tensores y metadatos entre las réplicas. Un segundo criterio, fijado antes de estos
ajustes, informa equivalencia numérica elemento a elemento con `rtol=1e-6` y
`atol=1e-8`: `abs(a-b) <= atol + rtol*max(abs(a),abs(b))`. Deben coincidir también
épocas seleccionadas, filas, etiquetas de etapa/época/mejor época, valores
ausentes, nombres/tipos/formas de tensores y metadatos del checkpoint. Una forma o
época seleccionada diferente impide declarar equivalente el ajuste completo.
El criterio secundario nunca convierte un fallo de igualdad exacta en un éxito
del criterio principal. No se adaptan tolerancias ni semillas a los resultados.

`e1-csv-values-v1` conserva filas y columnas en orden, enteros int64, etapas UTF-8,
ausentes explícitos y los bits IEEE754 binary64 de cada número tras interpretar
su decimal. Ignora comillas, fin de línea y distintas escrituras del mismo número.
`e1-state-dict-v1` ordena las claves y encadena nombres, tipos, formas y bytes de
tensores CPU contiguos con longitud explícita y orden little-endian. Ignora el
contenedor ZIP, identificadores de almacenamiento y strides. Los otros metadatos
del checkpoint tienen un hash JSON ordenado independiente. Los hashes históricos
de archivo no permiten reconstruir esos valores ni sus hashes canónicos.

## Ejecución

Primero se publican este protocolo, el ejecutor y sus pruebas. El responsable
verifica que el commit exista en GitHub antes de registrar. Desde un checkout
limpio e inmutable de ese commit, usando el entorno de versiones fijadas:

```bash
PYTHONHASHSEED=0 python scripts/audit_e1_replay_forensics.py register --private-dir /ruta/privada/nueva --published-commit SHA_COMPLETO_PUBLICADO
PYTHONHASHSEED=0 python scripts/audit_e1_replay_forensics.py fit --private-dir /ruta/privada/nueva --replicate a
PYTHONHASHSEED=0 python scripts/audit_e1_replay_forensics.py fit --private-dir /ruta/privada/nueva --replicate b
PYTHONHASHSEED=0 python scripts/audit_e1_replay_forensics.py compare --private-dir /ruta/privada/nueva --prior-private-dir /ruta/privada/reproduccion-completa --output-dir /ruta/publica/nueva
```

No se vuelve a ejecutar `prepare` sobre un protocolo publicado. La CLI comprueba
el commit local, limpieza, hashes de fuentes y protocolo, entorno efectivo de
cada trabajador y directorios independientes vacíos. El argumento
`--published-commit` no es por sí mismo prueba de publicación: se conserva aparte
la verificación remota previa. Ante un fallo se conserva la salida parcial y se
detiene; no se reintenta ni sustituye la semilla. Cualquier ajuste adicional
requiere otro plan publicado.

La comparación pública contiene épocas originales, de la reproducción completa y
de las réplicas A/B, hashes y diferencias máximas, estados de aceptación, ejecución
y entorno. Nunca exporta pesos, escaladores, observaciones ni predicciones diarias.
La búsqueda de originales y la inspección de los 1395 ajustes existentes se
documentan aparte en `results/e1_replay_forensics_20261107`; no cuentan como nuevos
ajustes. No se modifica el estudio original ni se exige identidad bit a bit para
cumplir la guía académica: esta comprobación acredita su alcance diagnóstico
actual y conserva la limitación de no disponer de los archivos privados originales.
