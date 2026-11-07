# Reproducción nueva de las curvas del modelo final D7

Este protocolo se preparó después de la evaluación original. La búsqueda local
no encontró las historias privadas `history_inner.csv` y `history_refit.csv` de
la ejecución declarada en `/workspace/e1_new_study_20261105/run`. Eso no prueba
que no se ejecutara ni que el autor no conserve otra copia.

La nueva ejecución reproduce las tres recetas finales congeladas con las mismas
465 semillas por familia y sólo los derivados públicos de desarrollo 2023–2024.
Usa los módulos históricos sin modificarlos, con sus seis versiones de paquetes
fijadas y un entorno/hardware registrado de nuevo. Cada trayectoria selecciona
épocas mediante validación interna causal y se reajusta desde cero con esa
duración; no consulta ni puntúa la reserva de 2026.

`protocol.json` y el ejecutor deben publicarse antes del primer ajuste nuevo.
Los campos de preparación con cero ajustes se conservan como el estado de ese
momento; la ejecución tendrá recibos separados y no los reescribirá.

## Secuencia

Desde un checkout limpio del commit publicado, con Python 3.12.14:

```bash
python -m pip install -r requirements-e1-curves.lock.txt
python scripts/reproduce_e1_final_curves.py register --output-dir /ruta/privada/nueva --published-commit SHA_COMPLETO_PUBLICADO
python scripts/reproduce_e1_final_curves.py fit --output-dir /ruta/privada/nueva
```

No se ejecuta `prepare` de nuevo para reproducir este plan: se verifica el JSON
publicado. Las salidas privadas contienen las curvas, pesos, escaladores y
completados; el exportador público separado sólo permite pérdidas agregadas,
resúmenes del ajuste, figuras y hashes. No publica observaciones, predicciones,
pesos o valores de escaladores.

Los hashes de 2.790 CSV de curvas y de otros artefactos científicos se compararán
con el manifiesto histórico ya publicado. Una coincidencia significa reproducción
idéntica según ese manifiesto, nunca recuperación del archivo privado original.
Las diferencias se conservarán; no se sustituirán semillas ni recetas para
forzar coincidencias. El baseline y los resultados históricos siguen siendo los
del estudio original.

Las pruebas `tests/test_e1_curve_reproduction.py` verifican las guardas y el plan
sin crear, ajustar o puntuar modelos.
