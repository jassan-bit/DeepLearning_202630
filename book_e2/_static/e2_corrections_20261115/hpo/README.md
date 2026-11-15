# Cobertura y recibos del HPO ejecutado

Se recuperan fuentes públicas ya existentes del cierre de selección, sus hashes,
recibos por oportunidad y diario de presupuesto. No se realizan ajustes nuevos,
no se cambian ganadores, no se ejecuta NAS y TiDE conserva su cuerpo publicado R0-v2.

`opportunities_234.csv` distingue resultado válido, fallo conocido y desenlace
desconocido. Mantiene métricas disponibles de configuraciones incompletas sin
inventar las que faltan. `configurations_78.csv` recalcula la RMSE concatenada de
los tres folds (83 fechas por fold) y verifica los trece ganadores registrados.

`parameter_coverage_65.csv` y `parameter_value_outcomes.csv` identifican valores,
variación/fijación, posición del ganador y éxito por nivel. Las figuras muestran
rendimiento por configuración y cobertura observada. Seis tuplas que cambian
varias dimensiones a la vez no identifican importancia aislada, causal o fANOVA:
no se publica una falsa importancia para cerrar esa casilla.

Los costes se separan: duración de fases medida, cargo del diario de esta campaña
y coste histórico del ajuste cacheado. Un resultado reutilizado no es un nuevo
entrenamiento. El cargo conservador de la sesión T4 perdida no es tiempo de
entrenamiento medido. Los campos desconocidos se mantienen vacíos, sin asignar cero.
Las sumas conocidas no se presentan como una atribución integral de todo el
HPO histórico ni como coste GPU activo; no se suman categorías que pueden solaparse.

Las fuentes originales y su procedencia están en `sources/` y su manifiesto.
Reproducción con Git y Python/Matplotlib instalados, en una carpeta nueva:

```sh
python scripts/e2_hpo_coverage_closure.py --output /tmp/e2-hpo-repeat --recover-only
python scripts/e2_hpo_coverage_closure.py --output /tmp/e2-hpo-repeat --recover-dispositions
python scripts/e2_hpo_coverage_closure.py --output /tmp/e2-hpo-repeat --register-plan
python scripts/e2_hpo_coverage_closure.py --output /tmp/e2-hpo-repeat --execute
```

La recuperación lee objetos Git del commit fijado; no consulta resultados nuevos.
La fase de cálculo requiere los mismos bytes registrados y no sobrescribe salidas.
