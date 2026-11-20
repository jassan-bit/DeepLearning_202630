<!-- Instrucciones de ensamblaje: aplicar estos bloques a las fuentes de capítulos10/11;
     este archivo es una propuesta de contenido, no una fuente MyST para el TOC. -->

## Bloque para añadir como 10.2

(e3-current-public-acceptance-handoff)=
## 10.2 Comprobación externa y aceptación preparada para el cierre

La comprobación del 19 de noviembre de 2026, hora de Bogotá, verificó las referencias locales del cliente V2 existente: **17 archivos para cinco ONNX**, metadatos y el intervalo posterior E3, sin inferencia. Sus blobs se cotejaron con la fuente publicada. La {download}`sonda de disponibilidad <../_static/e3_evidence_close_20261120/release/render_readiness_attempt.json>` ejecutó el helper `wake_service` de ese cliente durante diez segundos. No ejecutó el navegador ni obtuvo HTTP del servidor de origen; no sustituye el recorrido público completo de 180/30/30 segundos.

Las {download}`consultas públicas <../_static/e3_evidence_close_20261120/release/network_observations.json>` fallaron en el túnel del proxy del entorno con 403. Ese fallo corresponde al acceso del cliente y no permite afirmar que Render esté caído. Se mantiene **aceptación pública sin acreditar**, junto con la observación anterior Live comunicada por el operador. Tampoco se dispone del SHA efectivo actual confirmado en su log; no se adivinó a partir de `main` ni de la autodeclaración de `/healthz`.

Queda preparado el [helper de aceptación CPU para Colab](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_evidence_close_20261120/release/render_public_acceptance_colab.py) y su {download}`procedimiento completo <../_static/e3_evidence_close_20261120/release/README.md>`. Tras **Manual Deploy**, el propietario copia el SHA de 40 caracteres del despliegue y lo introduce en el helper. Éste obtiene las referencias de esa versión, verifica los clientes revisados y ejecuta el V2 original, con sus tres páginas, cinco predicciones, intervalos, explicaciones y huellas. El entorno de Playwright es separado, no requiere GPU y no reentrena ni recalibra.

El ZIP de salida conserva recibo, contexto y capturas; debe persistirse antes de cerrar Colab. Sólo su aprobación permite registrar el humo público. La concordancia de SHA autodeclarado conserva el límite de no ser una atestación completa del código remoto. El plan del servicio y los pesos/calibradores permanecen como estaban; este cierre no contrata recursos ni convierte el catálogo histórico en el de la auditoría nueva.

## Bloque para añadir como 11.7

(e3-documentary-archive-handoff)=
## 11.7 Suplemento documental y cierre del depósito persistente

El cierre prepara un suplemento documental con fuentes, evidencias y análisis de alcance explícito. Su {download}`constructor reproducible <../_static/e3_evidence_close_20261120/release/build_documentary_archive.py>` utiliza una lista revisada de miembros, tamaños y SHA-256; rechaza fuentes cambiadas, rutas inseguras, credenciales, pesos, datos binarios de mercado y cuerpos completos de artículos. Para repetir sus propios análisis sin red puede incluir únicamente los archivos textuales congelados `sources.zip` y `*.csv.gz` de inventario/adecuación, identificados expresamente en la especificación. Sus miembros, cabeceras, rutas, tamaños expandidos y ausencia de credenciales se verifican al descomprimir; no se permiten otros archivos comprimidos interiores. **El suplemento no contiene la reproducción integral de los 796 trabajos** ni sustituye sus datos, pesos, fuentes y cadena de recibos.

El {download}`plan del archivo completo <../_static/e3_evidence_close_20261120/release/full_archive_plan.DRAFT.json>` distingue el código de la versión final, la cadena de deltas de la auditoría de 796 trabajos y el snapshot acumulado de los 30 ajustes V3. El último delta193 de la auditoría no se presenta como si fuera por sí solo un snapshot completo del master. Se conservan los localizadores de los insumos y los requisitos de derechos; no se vuelven a entrenar modelos para preparar el archivo.

Los {download}`campos de Zenodo preparados <../_static/e3_evidence_close_20261120/release/zenodo_upload_fields.DRAFT.json>` y la {download}`cita de borrador <../_static/e3_evidence_close_20261120/release/CITATION.cff.DRAFT>` sólo usan la identidad y los datos conocidos. El DOI `10.5281/zenodo.24356217` se conserva como **reservado según la evidencia aportada por el autor**, sin presentarlo como una publicación comprobada. Las consultas actuales al API y al resolvedor fallaron en el proxy antes de alcanzar sus servidores; no se obtuvo un 404 del registro ni una confirmación de publicación.

El propietario debe revisar el archivo concreto y su manifiesto, confirmar las licencias y derechos por miembro, escoger si archiva este suplemento o el entregable integral y completar la fecha real de publicación. No se inventan ORCID, afiliación, firma o licencia. Después de publicar se deben preservar metadatos, hashes, enlace del registro y resolución pública del DOI para actualizar la cita. Mientras falten esas evidencias, el requisito del depósito persistente y la reproducción independiente mantienen su estado pendiente.
