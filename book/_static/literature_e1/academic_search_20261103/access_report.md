# Intentos actuales de acceso académico — 3 de noviembre de 2026

Se realizaron cuatro solicitudes GET. El proxy rechazó el establecimiento del túnel con 403 en las cuatro; no hubo respuesta de los buscadores. Esto no acredita un error HTTP 403 emitido por WoS, Scopus, IEEE o arXiv, ni un fallo de autenticación de esas plataformas.

| Fuente | Instante America/Bogota | Ruta | Estado | HTTP de origen | Total bibliográfico |
|---|---|---|---|---|---|
| Web of Science | 2026-11-03T11:51:02.450758-05:00 | ui_entry | CONNECT del proxy: 403 | No recibido (`null`) | Desconocido (`null`) |
| Scopus | 2026-11-03T11:51:02.449245-05:00 | search_ui | CONNECT del proxy: 403 | No recibido (`null`) | Desconocido (`null`) |
| IEEE Xplore | 2026-11-03T11:51:02.450270-05:00 | search_ui | CONNECT del proxy: 403 | No recibido (`null`) | Desconocido (`null`) |
| arXiv | 2026-11-03T11:51:02.451060-05:00 | Atom_API | CONNECT del proxy: 403 | No recibido (`null`) | Desconocido (`null`) |

Cadenas preparadas: 24 (ocho generales/de sensibilidad y dieciséis dirigidas opcionales). Consultas bibliográficas verificadas: ninguna. No se recuperaron exportaciones académicas; no se interpretan esos accesos fallidos como cero coincidencias o artículos excluidos.

Los mensajes seguros, URLs intentadas, cadenas literales y separación entre estado del proxy y del origen se conservan en `access_attempts.json`. La respuesta falló antes de producir un cuerpo bibliográfico para archivar o calcular su hash. El historial GitHub permanece separado y conserva sus cifras y hashes.
