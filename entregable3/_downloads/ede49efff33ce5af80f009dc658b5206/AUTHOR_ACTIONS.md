# Acciones del propietario después de las pruebas

Asistencia de IA declarada. No se necesita repetir los 796 trabajos ni los 30 ajustes.

## Render: identidad y aceptación pública

El probe real del 20 de noviembre, 12:46 Bogotá, obtuvo salud HTTP 200, pero ningún SHA. No aprobó el cliente de aceptación. La integración local no cambia ese resultado.

1. Consulte el commit de esta entrega y la ejecución `E3 optional historical14 profile, CPU only`. Use el **SHA fuente de 40 caracteres** del commit con las pruebas aprobadas; no el SHA de `gh-pages` ni el del libro generado.
2. Para conservar únicamente el perfil habitual de cinco modelos, mantenga `Dockerfile`. Para habilitar además `/perfil-14`, aplique la configuración revisada `render.profile14.yaml`/`Dockerfile.profile14` según `experiments/e3_product_followup_20261120/README.md`. V3 no se habilita con ninguno.
3. En el servicio existente, use Manual Deploy → Deploy a specific commit. Conserve `autoDeploy: false`; no reentrene ni cambie pesos, escaladores o cuantiles. Registre en Render el SHA efectivo y espere `Live`.
4. Consulte `/healthz`: `source_commit` debe concordar con aquel despliegue y `source_commit_source` debe identificar `RENDER_GIT_COMMIT` o el origen explícito admitido. Es una observación del servicio, no atestación criptográfica de cada fichero.
5. Ejecute el workflow existente `Read-only E3 public Render and DOI probe` con Run workflow: el cliente V2 comprobará tres páginas, callbacks, cinco modelos, intervalos, explicaciones, huellas y errores de consola/HTTP; no genera casos OOD ni CSV inválidos. Con el perfil14, ejecute además el cliente nuevo siguiendo el README del perfil para los catorce OOD, guardas temporales y diez entradas inválidas. Conserve recibos y capturas. No declare aceptación si la guarda de identidad o una comparación falla.

## Reproducción por un tercero

Entregue `experiments/e3_reproduction_followup_20261120/README.md`, manifiesto y `REPRODUCTION_FORM_DRAFT.md`. El tercero debe ejecutar sus propios comandos, informar versiones y discrepancias y completar su propia constancia. A/tablas y B/predicciones no se convierten en C/entrenamiento integral; acuerde el alcance con el docente. La autorización de IA para R1 no es una autorización general para sustituir esa persona.

## Depósito y DOI

El ZIP candidato es documental, no el maestro científico de 4,27 GB. Revise `experiments/e3_archive_followup_20261120/AUTHOR_RIGHTS_AND_AUTHORIZATION.md`: complete titularidad, derechos de cada componente, licencia propia y autorización del depósito exacto. No herede una licencia global de fuentes externas ni invente firmas o afiliación.

Una vez resueltos esos campos, el propietario puede cargar el candidato en el borrador `https://zenodo.org/uploads/24356217`, revisar metadatos y publicar. El script `verify_zenodo_deposit.py` es de sólo lectura y verifica después registro, resolución y hashes de los bytes depositados. Si el contenido cambia, construya y revise otro manifiesto; no reutilice el hash del candidato anterior. Un DOI reservado no es un DOI publicado.

## Ciencia

La receta V3 actual queda en NO-GO confirmatorio. No hay una campaña GPU pendiente de lanzar automáticamente. Para reconsiderarla, complete primero el checklist científico y justifique una pregunta nueva. No promueva la ablación ganadora ni cambie el 2 % del piloto; un algoritmo diferente será otra versión y requerirá revisar R1.
