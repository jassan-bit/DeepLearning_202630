# Entrega de protocolo y potencia V3

Esta carpeta no contiene entrenamientos, nuevas predicciones ni una campaña
confirmatoria autorizada. Conserva el resultado adverso del piloto y aporta
material incorporable al capítulo 8 y a la matriz de pendientes.

| Archivo | Uso y estado |
|---|---|
| `decision_zero_shot.json` / `.md` | Decisión nueva para el diagnóstico exploratorio de transferencia con pesos fijos; root debe registrar commit antes de que otro proceso prediga. |
| `protocol.DRAFT.json` / `.md` | Protocolo concreto pero bloqueado: faltan decisión científica, datos no expuestos, margen, batería revisada, potencia final, presupuesto, motor V3 y tag real. |
| `planning_registry.json` | Escenarios matemáticos registrados antes del cálculo CPU; no es prerregistro experimental. |
| `planning.py` | Reproduce sólo sensibilidad prospectiva condicional a partir de predicciones V3 guardadas. |
| `results/` | SD de seis controles, sensibilidad de semillas y calendario, límites de potencia conjunta y recibo de cálculo sin modelos. |
| `preflight.py` | Preparación CPU reanudable, identidad y hashes; no implementa entrenamiento ni inferencia. |
| `colab_cpu_preparation.py` | Celda preparada que exige commit inmutable y SHA publicados; no descarga fuentes flotantes. |
| `book8_insert.md` | Sección lista para adaptar al libro con enlaces del destino final. |
| `requirements_matrix.csv` | Requisito, evidencia y pendiente concreto. |
| `verification.json` / `delivery_manifest.json` | Comprobaciones ejecutadas y hashes de la entrega. |

Reproducción de los cálculos, con NumPy 2.3.5 y SciPy 1.17.0 ya instalados:

```bash
python planning.py --output results
python -m unittest test_preparation.py
python preflight.py --receipt CPU_preparation_receipt.json --prepare-only
```

`--prepare-only` sólo permite que la operación documental termine con código
cero cuando el protocolo esté bloqueado; no convierte ningún `false` en `true`.
Sin esa opción el bloqueo devuelve código 2. Reanudar el mismo recibo exige
idénticos bytes de protocolo y adaptador. Cambiar el diseño requiere otra
preparación, conservando la anterior y los masters existentes.

El helper histórico se conserva con hash y se reutilizan únicamente funciones
matemáticas puras, pasando explícitamente `α=.05/57`. Nunca se invoca su
`build()` ni se cargan varianzas aditivas. Los CSV de entrada son pronósticos
de desarrollo ya existentes; las trayectorias deterministas no se cuentan
como nuevos ajustes o fechas independientes.

La celda Colab permanece fijada a valores sin resolver porque aún no existe
el commit de integración. Quien publique debe proporcionar ese SHA real y
el hash de `delivery_manifest.json`; no se inventa un commit ni una etiqueta.
La preparación resultará bloqueada con el estado científico actual.

Se usó IA para redactar el borrador, auditar requisitos y escribir/revisar
el cálculo matemático y las puertas de ejecución. No se atribuye esta
preparación a una revisión humana ni a una aprobación institucional.
