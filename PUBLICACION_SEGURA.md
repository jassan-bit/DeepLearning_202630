# Publicación segura de E1, E2 y E3

## Estado y alcance

Esta propuesta conserva E1 en la raíz, E2 en `entregable2/`, E3 en
`entregable3/` y el selector `entregables.html`. No modifica fuentes de los
libros, resultados científicos, experimentos, descargas ni configuración de
Pages. No entrena modelos.

La activación autorizada del 09/11/2026 permite publicar **solo en el primer
intento de un nuevo push a main**. PRs, ejecuciones manuales y reruns compilan
y verifican, pero no publican.
La condición queda versionada en el workflow y no depende de una variable de
configuración. Esta activación no autoriza relanzar publicaciones históricas
ni cambiar protecciones de seguridad.

Antes de activar se verificó el ruleset activo [Proteger gh-pages](https://github.com/jassan-bit/DeepLearning_202630/rules/26826297),
sin bypass, con bloqueo de borrado y de actualizaciones no fast-forward solo
para `refs/heads/gh-pages`. El último sitio servido, fuente `cab9143a`, se conserva
en la [rama de respaldo](https://github.com/jassan-bit/DeepLearning_202630/tree/backup/pages-served-20261109-cab9143a),
anclada al commit de Pages `3c539138907ca94c25af20eee6731700054651ea`.

## Barreras implementadas

- Construcción y publicación en trabajos separados. Construcción con
  `contents: read`; solo el publicador recibe `contents: write`.
- Acciones fijadas por SHA. Checkout sin credenciales persistentes.
- PR, workflow_dispatch y reruns: construcción y pruebas, nunca publicación. Primer intento de push a cualquier commit de
  main: construcción. Se elimina el filtro por rutas para que un avance de main
  no deje una compilación obsoleta sin reemplazo.
- Fuente fijada al SHA del run. Verificación de checkout limpio y main vigente
  antes de construir y antes del push. Se rechazan cambios versionados y
  archivos no versionados que no estén ignorados.
- Los tres libros deben existir completos, con identidades, rutas, manifiesto,
  hashes de las introducciones, selector y `.nojekyll` correctos.
- Inventario SHA-256 de todos los archivos. Se rechazan enlaces simbólicos,
  hard links, archivos especiales y metadatos de repositorio.
- Todas las rutas ya presentes en gh-pages deben seguir existiendo. Las
  descargas existentes en `_downloads` y archivos con extensiones ZIP, PDF,
  CSV, TSV, XLS/XLSX, IPYNB, TAR, GZ, 7Z, PKL y PARQUET deben conservar sus bytes.
  Una nueva descarga debe usar una URL versionada, conservando la anterior.
- El `source_commit` anterior debe ser ancestro de la fuente nueva. No se
  permite sustituir silenciosamente la fuente ni crear gh-pages si falta.
- Se crea un commit descendiente del gh-pages observado, mediante un índice
  temporal separado. Todos los blobs del árbol preparado se cotejan con el
  inventario validado antes de crear el commit. La copia de trabajo no se
  transforma ni se borra.
- Push normal, sin force ni force-with-lease. Si cambian main o gh-pages, se
  detiene. No hay rebase, rollback ni reintento automático de escrituras.
- Un único publicador en el grupo `publish-book`, sin cancelar publicaciones
  en curso. Todas las futuras rutas de recuperación deben usar ese mismo grupo.
- Comprobación posterior de ambos refs. Si main avanza durante el push, el
  informe registra que la rama cambió, pero ya no corresponde al main actual.
- Un push correcto no declara éxito web. `verify-live` exige el manifiesto
  exacto y hashes de las tres introducciones y del selector servidos por HTTP.
  Las comprobaciones web pueden repetirse; las escrituras no se repiten.
- El informe distingue `pushed` de `web_deployment_verified`. El artefacto de
  publicación deja `pushed: null` y `push_outcome: unknown` si no puede confirmar
  el resultado de una escritura; nunca lo convierte en un «no publicado» seguro.
  El artefacto de
  sitio se conserva 30 días y el informe 90, sujetos a políticas de GitHub.
  Los plazos no son una copia de seguridad permanente.

## Comandos de comprobación

Usar Python 3.12 y las versiones exactas de `requirements-book.lock.txt`.

```sh
python -m pip check
python tests/test_multi_book_publication.py
python tests/test_safe_publish_books.py
python tests/test_books_alignment.py
for book_dir in book book_e2 book_e3; do
  jupyter-book build "$book_dir" --all --warningiserror --keep-going
  python scripts/check_book_links.py --book-dir "$book_dir"
done
python scripts/assemble_books_site.py --source-commit "$(git rev-parse HEAD)"
python scripts/safe_publish_books.py inventory --source-commit "$(git rev-parse HEAD)"
python scripts/safe_publish_books.py validate --source-commit "$(git rev-parse HEAD)"
```

El ensamblado requiere una carpeta nueva o vacía: al repetir, elegir otra
ruta con `--output-dir` y pasarla a los demás comandos mediante `--site`.

`publish` sin `--execute` es una comprobación sin push. Lee refs remotos y
obtiene objetos Git necesarios; también puede escribir el informe local:

```sh
python scripts/safe_publish_books.py publish --source-commit "$(git rev-parse HEAD)"
```

Este comando rechaza una rama de propuesta cuyo SHA no sea main. No se debe
cambiar su remote para eludir una denegación de publicación. Las pruebas usan
exclusivamente repositorios bare temporales locales para ejercitar escrituras.

## Lista de comprobación para una activación autorizada

1. Confirmar nuevamente main, gh-pages, los jobs de Pages y la versión realmente
   servida. No confundir último commit generado con último despliegue exitoso.
2. Conservar y verificar una copia del último sitio realmente servido, con
   manifiesto, hashes, SHA fuente y SHA de Pages. Conservar también el gh-pages
   vigente. No confiar en que un objeto huérfano remoto seguirá disponible.
3. Aprobar y aplicar una protección que prohíba force-push y borrado de gh-pages,
   sin bypass para el publicador, permitiendo sus pushes normales. Revisar
   interacción con reglas ya existentes antes de cambiar nada.
4. Revisar el parche en una rama/PR; su construcción no publica. Los workflows
   antiguos siguen teniendo código antiguo: no relanzarlos. La protección de
   rama es la barrera del servidor frente a esos force-push históricos.
5. Solo con autorización de activación, habilitar en el workflow la publicación
   de nuevos pushes a main y adaptar su prueba de regresión. Verificar por
   separado push, workflow dinámico de Pages y web.
   No borrar ni despublicar el sitio existente para realizar esta transición.

## Recuperación

- Si falla construcción, validación, preservación o vigencia, no cambiar la
  rama; la versión ya servida permanece fuera de esta operación.
- Si falla la confirmación web, consultar el estado real de Pages y el
  manifiesto servido. El reporte puede mostrar `pushed: true` y
  `web_deployment_verified: false`. No borrar deployments ni relanzar
  automáticamente un workflow viejo.
- Si se necesita una restauración, seleccionar un paquete previamente
  verificado y obtener aprobación de esa versión y sus consecuencias. Preferir
  una corrección nueva sobre main vigente para conservar la cronología fuente.
- Una restauración directa, si se autoriza, debe ser un **nuevo commit** hijo
  del gh-pages entonces vigente, con motivo y procedencia documentados y bajo
  la misma exclusión mutua. Nunca reset/force-push. Esa ruta debe diseñarse y
  verificarse separadamente; el publicador normal no tiene un parámetro para
  saltarse los controles ni presentar una fuente antigua como actual.

## Límites conocidos

- main y gh-pages no se actualizan en una misma transacción. Se comprueban antes
  y después del push, pero main puede avanzar inmediatamente después de una
  comprobación. Esto detecta carreras observables; no garantiza que el sitio
  corresponda al último commit en cada instante. No hay rollback automático.
- Concurrency controla únicamente trabajos que usen ese grupo. La protección
  del servidor y el rechazo fast-forward cubren otras escrituras divergentes.
- Las reglas de preservación son deliberadamente estrictas. Un cambio legítimo
  de tema, nombre, descarga o retirada de contenido puede requerir conservar
  rutas antiguas o una migración específica aprobada; no se reduce la regla
  automáticamente para hacer pasar el build.
- El validador de enlaces revisa href, src y anclas HTML, no URLs dentro de
  CSS/JavaScript ni enlaces externos. Se preservan los archivos anteriores,
  pero no se afirma una prueba visual completa en navegador.
- La verificación web comprueba manifiesto y cuatro páginas de entrada. Todas
  las descargas se validan localmente por hash; no se descargan de nuevo cientos
  de MB desde Pages ni se certifica cada recurso HTTP individual.
- En las reintentos de entrega web se contemplan hasta 24 observaciones con
  30 segundos entre ellas, más el tiempo de las peticiones HTTP; el job tiene
  un límite global de 25 minutos. Un timeout no identifica por sí solo su causa.
- El sitio sigue usando publicación desde rama. GitHub documenta que pushes
  con GITHUB_TOKEN no disparan un build de Pages; la configuración observada
  del repositorio sí ha generado runs dinámicos. Debe comprobarse la entrega,
  no asumirla. Una migración a upload-pages-artifact/deploy-pages es otra tarea,
  con cambios de configuración y autorización propios.
- El historial empieza a preservarse desde el primer commit hijo creado por
  este flujo. No recupera automáticamente los commits huérfanos anteriores.

## Fuentes técnicas

- [Reruns y SHA original](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).
- [Concurrencia](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).
- [Protección de ramas](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
- [Fuente de publicación y GITHUB_TOKEN](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
- [Workflows oficiales de Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
