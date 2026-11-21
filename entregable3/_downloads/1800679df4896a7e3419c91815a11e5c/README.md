# Suplemento1.1: tabla principal aditiva19×4, pilotoV3 y libro completo

El kitv1 y su MANIFEST SHA256
`a35a5f56d6288ad5640f347b59415f00968277dceb96096fda0dc83db75b0206`
permanecen intactos. Este suplemento amplía el alcance preparado; no convierte
las7 recetas del piloto en la maestra19sistemas×4conjuntos. Usa la misma edición
`f5c960eefedb5fc49b313dd795b1ec84e784bd78`, contenedor y lock del libro.
Se declara asistencia de IA en la preparación, el código y la auditoría de
software. Esa asistencia no representa la intervención de una persona externa.

## Alcance aritmético exacto

- Auditoría aditiva histórica:4.060 métricas por semilla versionadas,4060registros
  de resultados costosos ya calculados, se reagregan en532filas modelo/conjunto/
  split/escenario. Contienen76filas de prueba limpia (19×4), construidas desde580
  registros por trayectoria, más456rangos descriptivos y4filas del resumen
  publicado en8.18. Cobertura/anchura del resumen se reagregan desde2.320resúmenes
  de intervalos guardados; no se vuelve a calibrar.
- PilotoV3:935 predicciones puntuales guardadas,17trayectorias/55fechas/7recetas,
  se recalculan en RMSE,DE,MAE y sesgo; se mantiene el NO-GO y el CRPS_U etiquetado
  de fuente, no reconstruido desde puntos.
- Libro E3 entero: construcción limpia con notebooks desactivados. Todos los
  CSV y manifiestos históricos permanecen byte a byte intactos. Los derivados
  se instalan en `_static/human_reproduction_19x4/`; sólo se añade al capítulo8
  de la copia del caso un apéndice que muestra efectivamente las76filas
  regeneradas y las7del piloto, con enlaces descargables nuevos. Se comprueba
  la identidad de esas descargas del HTML y que las cuatro filas aditivas y
  las siete filas originales visibles coincidan al redondeo publicado.

**No se recalculan las métricas por semilla desde1.874.560 predicciones de la
auditoría**, ni se rehidrata el maestro4,27GB, ni se entrenan modelos, ejecutan
forward passes, remuestreos o nueva confirmación. Las demás tablas/figuras son
precálculos versionados renderizados con su código generador conservado en el
checkout. La guía p38 permite resultados costosos precalculados si se conserva
y documenta ese código. Se informa el nivel real de reproducción; las firmas
humanas y cualquier exigencia de un nivel mayor permanecen externas.

## Preparación nueva o reutilización de un caso aún sin ejecutar

Desde el directorio que contiene `human_reproduction`:

```bash
python3 human_reproduction/addon_v1_1/prepare_extended.py --source-root e3-source --case-root e3-human-comprehensive-001
```

También se admite el caso que v1 creó y que todavía no tiene salidas ni request
editado por una persona. Se conserva su copia nueva de1,50GB; no se duplica:

```bash
python3 human_reproduction/addon_v1_1/prepare_extended.py --case-root e3-human-case-001 --attach-to-unexecuted-case
```

El preparador verifica el manifiesto v1 exacto y el nuevo manifiesto del
suplemento, copia éste a `CASE/addon` y crea el request ampliado con identidad,
justificación y tolerancias todavía vacías. Rechaza reutilizar salidas o
sobrescribir una declaración humana ya editada. No modifica los kits originales.
La única modificación permitida de un archivo ya versionado en la copia es
el apéndice explícito del capítulo8. Los archivos nuevos de datos derivados se
inventarían en el recibo; no se reescriben los hashes de resultados anteriores.

## Ejecución

Construya la imagen según el Dockerfile/lockv1; la misma capa de dependencias
sirve a ambos procedimientos. Para ejecutar el alcance1.1 use este entrypoint:

```bash
docker run --rm --read-only --network none --cap-drop=ALL --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,size=512m --memory=2g --cpus=2 --pids-limit=256 --user "$(id -u):$(id -g)" --mount "type=bind,src=$(pwd)/e3-human-comprehensive-001,dst=/case" --entrypoint python e3-book-reproduction:f5c /case/addon/run_extended_case.py --case /case --mode human-external
```

El operador real debe completar antes `CASE/HUMAN_REQUEST.json`: aceptación del
alcance19×4+V3, identidad/relación/independencia, fecha previa y todos los campos
`tolerances` y `audit_tolerances`. Cada número finito se compara mediante
`|observado−referencia|≤absoluta+relativa×|referencia|`; claves, conteos, estados,
blancos, semillas, datasets y escenarios se verifican estrictamente. Un control
determinista tiene una trayectoria y DE no aplicable; las semillas neuronales
no se cuentan como diseños nuevos. Se conserva toda la familia, no sólo Top10.

La prueba automática del equipo usa otro caso nuevo y cambia únicamente
`--mode software-validation`. Su perfil está fijado antes de ejecutar:
V3 absoluta1e-15/relativa1e-12; agregación aditiva absoluta1e-12/relativa1e-12 por
campo. No es el perfil humano ni se copia como aceptación del tercero. Un recibo
con `human_execution_claimed=false` nunca constituye el acta humana.

Si la red de Docker no resuelve PyPI, se puede descargar en el host el mismo
lock con `pip download --only-binary=:all: --no-deps`, inventariar cada wheel por
hash y construir una variante offline con la misma base fijada y
`pip install --no-index --find-links ... --no-deps --report ... -r ...`.
Conserve Dockerfile efectivo, hashes de wheels, informe pip y digest de imagen.
Una instrucción preparada no se registra como prueba ejecutada. La variante
no cambia los generadores ni permite red durante la ejecución científica.

## Informe de la persona

Use el formulariov1 con el alcance actualizado19×4+V3. Añada los CSV y
`output/audit/audit_regeneration.json`, los máximos por campo de532/456/4filas,
los76registros principales, el recibo de ejecución y el HTML/logs del libro.
Declare expresamente el uso de métricas por semilla precalculadas y de resultados
puntuales del piloto. La persona comprueba y firma su propia ejecución; el
procedimiento no autentica identidades ni acredita automáticamente R2/R3.
La excepción docente de IA para R1 no se extiende a este acto ni a sustentación.
