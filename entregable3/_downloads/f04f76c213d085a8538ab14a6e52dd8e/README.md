# Cierre preparado: comprobación pública de Render y depósito persistente

Se preparó este material con asistencia de IA. No se entrenó ni recalibró ningún modelo. No se publicó en Zenodo ni se cambió la configuración de Render.

## Evidencia de esta comprobación

`network_observations.json` conserva tres consultas públicas de sólo lectura: `/healthz`, el API público de Zenodo para 24356217 y la resolución del DOI reservado. Las tres fallaron **en el túnel del proxy del entorno con 403**, antes de recibir una respuesta del servidor de origen. El proveedor web alternativo tampoco abrió esas URL. Esta observación no permite concluir que Render esté caído o que el depósito no se haya publicado.

`render_readiness_attempt.json` registra el uso del **cliente V2 existente**, conservado sin modificaciones y cotejado con sus blobs en el commit `29f9a26f1fd05484c6581ea6538ad7cb5a4781cb`. Su `load_references` verificó 17 archivos, los cinco ONNX, los metadatos y el intervalo posterior E3, sin inferencia. La sonda de su helper `wake_service` tuvo un presupuesto acotado de diez segundos y no obtuvo HTTP de origen. **No es la aceptación pública completa de 180/30/30 segundos**, no ejecutó el navegador y mantiene `render_public_smoke_passed=false`.

El SHA efectivo actual del despliegue no fue aportado por el operador. No se sustituyó por `main`, por el SHA antiguo de las capturas ni por un valor autodeclarado por el servicio. La identidad remota completa tampoco se considera atestada por el cliente V2.

## Acción preparada para el propietario de Render

1. Una vez publicado el cierre, elegir **Manual Deploy → Deploy latest commit** en el servicio existente y esperar **Live**. El blueprint conserva `autoDeploy: false`; este material no cambia el plan ni contrata recursos.
2. Copiar del log de ese despliegue el **SHA efectivo de 40 caracteres** y conservar el log. No basta un prefijo como el de las capturas antiguas.
3. Ejecutar `render_public_acceptance_colab.py` en Colab con CPU. El script pregunta ese SHA, obtiene un checkout mínimo de esa versión, verifica los hashes de los dos clientes revisados, crea un entorno aislado sólo para Playwright y ejecuta el V2 original. Las operaciones Git tienen timeout de 90 segundos. Si una descarga anterior dejó la caché sin HEAD o incompleta, se conserva aparte y se intenta una copia limpia de la misma versión. No usa GPU ni instala Torch; las cinco predicciones ocurren en el servicio para el caso fijo de aceptación.
4. Conservar el ZIP que se descarga: recibo, contexto y capturas. Antes de cerrar la sesión, subir esos archivos mediante la interfaz de GitHub a `runs/render_acceptance/<identificador>/` o incorporarlos con la siguiente actualización del repositorio. El helper no promete persistencia remota que no haya ejecutado y un ZIP local puede perderse al reiniciar Colab.
5. Sólo un recibo `status=passed`, con tres rutas y cinco modelos completados y concordancia del SHA esperado, permite registrar aceptación pública. Se conservan los fallos y capturas si no pasa. Memoria pico, concurrencia, continuidad del servicio y arranque frío acreditado tienen mediciones propias y no se deducen del recibo.

El código del helper se publicará en `experiments/e3_evidence_close_20261120/release/`. La celda de arranque fijada al commit de publicación puede generarse desde `colab_cell.py.template`; el marcador de commit debe resolverse por el publicador antes de presentar esa celda como ejecutable.

## Material preparado para el propietario del borrador Zenodo

El registro suministrado por el autor reserva `10.5281/zenodo.24356217` en `https://zenodo.org/uploads/24356217`. Esa reserva no registra por sí sola un DOI publicado. La verificación pública actual quedó bloqueada por el proxy y no equivale a un 404 de Zenodo.

El ZIP anterior `e3-source-audit-1bae73b-v0.1.0-DRAFT.zip` conserva sus 126.780 bytes y su alcance histórico de 34 fuentes. No se reemplaza silenciosamente ni se presenta como reproducción integral de los 796 trabajos.

`zenodo_upload_fields.DRAFT.json` prepara los campos conocidos del formulario para un **suplemento documental delimitado**; no es una exportación autenticada de Zenodo ni un payload listo para su API. `CITATION.cff.DRAFT` conserva autor, repositorio y título sin anunciar el DOI reservado como publicado. No se inventan ORCID, afiliación, fecha de publicación, firma o licencia.

`build_documentary_archive.py` y `make_documentary_spec.py` permiten preparar el ZIP concreto del cierre con una lista explícita de fuentes, tamaños y SHA-256. El archivo preserva estados de derechos por miembro y declara que no contiene todos los datos ni pesos ni reproduce entrenamientos. Se excluyen credenciales, pesos, datos binarios de mercado y cuerpos completos de artículos. No se empaqueta el checkout completo de forma indiscriminada.

Hay una excepción estricta para que el suplemento pueda repetir sus propios análisis sin red: sólo con tipo explícito `frozen_documentary_input_archive` se admiten `sources.zip` y archivos `*.csv.gz` bajo `experiments/e3_evidence_close_20261120/inventory/` o `adequacy/`. Son registros textuales del proyecto y extractos de predicciones ya guardadas, con derechos pendientes de ratificar; no son pesos ni barras OHLCV crudas. El constructor inspecciona el ZIP interior: rutas seguras, miembros regulares sin cifrado ni duplicados, sólo JSON/CSV/Markdown/Python/texto UTF-8, tamaños acotados y ningún archivo comprimido interior. Descomprime gzip de forma acotada y comprueba su cabecera CSV; busca credenciales después de descomprimir ambos formatos. Cada miembro comprimido conserva su hash original y el manifiesto registra además los hashes/tamaños expandidos y las cabeceras pertinentes. Los demás ZIP/GZ, los duplicados bajo `_static` y los archivos de entrenamiento siguen excluidos. Repetir estos análisis documentales no reproduce los 796 entrenamientos.

Después de ensamblar el cierre y conocer su commit fuente, el publicador debe ejecutar, con los miembros explícitos que efectivamente haya revisado:

```bash
python experiments/e3_evidence_close_20261120/release/make_documentary_spec.py \
  --root . \
  --include book_e3/intro.md \
  --include experiments/procedencia.md \
  --include experiments/design_log.md \
  --output work/documentary_archive_spec.json

python experiments/e3_evidence_close_20261120/release/build_documentary_archive.py \
  --root . --spec work/documentary_archive_spec.json \
  --source-commit "$SHA_FUENTE_DEL_CIERRE" \
  --output work/e3-documentary-closure-20261120-DRAFT.zip

python experiments/e3_evidence_close_20261120/release/build_documentary_archive.py \
  --root . --spec work/documentary_archive_spec.json \
  --source-commit "$SHA_FUENTE_DEL_CIERRE" \
  --output work/e3-documentary-closure-20261120-DRAFT.zip --check
```

La lista anterior muestra tres miembros de ejemplo; el ZIP final del suplemento debe usar el inventario completo revisado que prepare el publicador, incluidas las evidencias y análisis nuevos que realmente se incluyan. El manifiesto externo fija nombre, bytes, SHA-256 y número de miembros. El commit fuente puede pertenecer a una etapa previa de publicación; no debe incluirse el propio SHA del commit que contiene al ZIP de forma circular.

Antes de pulsar **Publish**, corresponde al autor confirmar el archivo exacto y su alcance, seleccionar las licencias del código y la documentación propios y ratificar los derechos de cada insumo incluido. La disponibilidad pública en GitHub no se convierte en una licencia abierta. Para publicar el entregable completo, usar además `full_archive_plan.DRAFT.json`: el suplemento por sí solo no cierra ese requisito.

El plan distingue la cadena incremental de la auditoría anterior —el último delta 193 no contiene por sí solo todo el master— del snapshot acumulado V3. También separa los archivos preparados de una reproducción independiente efectivamente ejecutada. Al publicar, conservar el enlace público, el DOI resuelto, los metadatos finales y los hashes descargados; actualizar entonces la ficha y la cita de la versión realmente archivada.

## Verificación local del material

`test_results.log` y `verification.json` documentan quince pruebas sintéticas de archivo reproducible, integridad, rechazo de cambios, rutas inseguras, pesos, credenciales, estados de derechos, separación de la entrada del operador y recuperación de caché incompleta sin perder sus bytes. Incluyen las excepciones comprimidas explícitas, su inspección interna, límites de expansión y rechazo de credenciales ocultas u OHLCV crudo. El cliente público y el DOI conservan sus estados no verificados. Las pruebas no ejecutan GPU, entrenamiento, inferencia de mercado ni solicitudes públicas.
