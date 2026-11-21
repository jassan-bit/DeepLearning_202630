# Reproducción de la edición final d6: suplemento 1.2

Asistencia de IA declarada. Fija la fuente **d6cf7f1af6800ec1ac85a26347bdde52380e362f** del libro final, sin relabelar la ejecución anterior f5c. Reutiliza fuentes v1 y v1.1 byte por byte, verificadas por manifiestos; cambia de forma explícita su contexto de edición y perfil en memoria. No cambia fórmulas, referencias o tolerancias históricas. El contenedor compila el libro completo actual y reconstruye las tablas aditiva19×4 y V3. Añade una comparación de seis recetas V4 desde las1.210 predicciones guardadas, sin inferencias, entrenamiento o remuestreo.

## Preparar y ejecutar una persona externa

Obtener un checkout completo y limpio del commit d6 indicado, bajo los términos de sus entradas. Desde esta carpeta:

```bash
python prepare_current.py --source-root /ruta/al/checkout-d6 --case-root /ruta/nuevo-caso
```

Antes de ejecutar, la persona completa **HUMAN_REQUEST.json** en el caso: identidad y relación auténticas, independencia, aceptación del alcance19×4+V3+V4, explicación de adecuación, fecha previa y todas las tolerancias por campo. El campo V4_development_scope_accepted debe ser true por decisión de la persona. La guía permite resultados costosos precalculados con generadores versionados; la persona y docente acuerdan si este alcance satisface la tabla principal o debe ampliarse. No se presupone entrenamientoC.

Reconstruir la imagen con el perfil offline publicado en experiments/e3_next_evidence_20261121/container_profile y sus89wheels fijados, obtenidos bajo sus condiciones. No se publica una imagen OCI ni se promete el mismo ID binario. Ejecutar con la imagen construida y registrar ID/logs propios:

```bash
docker run --rm --read-only --network none --cap-drop=ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,nosuid,size=512m --memory=2g --cpus=2 --pids-limit=256 \
  --user $(id -u):$(id -g) --mount type=bind,src=/ruta/nuevo-caso,dst=/case \
  --entrypoint python IMAGEN_PROPIA /case/current/run_current.py --case /case --mode human-external
```

El modo software-validation usa su propio perfil y declara human=false. Nunca suplanta la solicitud humana. Conservar output/CURRENT_EDITION_EXECUTION_RECEIPT.json, logs, HTML, CSV y discrepancias; emitir después un concepto propio con el formulario congelado frozen_base/HUMAN_REPORT.template.md. Ni un recibo automatizado ni la excepción de IA de R1 acreditan R2/R3 o una persona externa.

## Integridad y alcance

MANIFEST.json enumera fuentes/perfiles congelados antes de esta comprobación. PUBLIC_MANIFEST.json añade después los recibos públicos de la ejecución, sin datos brutos, pesos, wheels o checkout completo. Los nuevos CSV V4 son operaciones aritméticas sobre resultados ya publicados; las seis recetas son desarrollo y V4 mantiene ABANDON_CURRENT_V4_RECIPE. El cómputo V4 ocurre después de construir el libro y queda como comparación adicional, no como nueva inferencia de mercado. Todas las fuentesCSV/manifiestos originales permanecen intactas en el caso, salvo el apéndiceMD local que instala el generador v1.1.
