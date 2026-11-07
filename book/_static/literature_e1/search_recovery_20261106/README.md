# Recuperación de trazabilidad y fechas de publicación de E1

El 6 de noviembre de 2026, hora de Bogotá (inicio 20:49:33; 7 de noviembre 01:49:33 UTC), se intentó recuperar evidencia nativa adicional de Web of Science, Scopus e IEEE Xplore mediante el lector web disponible. Se accedió a las rutas públicas y al enlace institucional de Uninorte, sin sesión autenticada ni credenciales de biblioteca.

## Resultado verificable

Los cuatro intentos no permitieron inspeccionar resultados nativos. WoS público devolvió HTTP 403; las otras rutas se declararon no accesibles por la herramienta. La causa concreta de estas últimas respuestas no queda establecida. No se recuperaron totales, rangos ni historial literal: **desconocido no significa cero resultados**. Abrir una ruta de consulta no acredita que la plataforma ejecutara la expresión solicitada.

Se volvieron a leer las cuatro exportaciones originales retenidas: coinciden sus SHA-256 y sus 19/158/8/59 registros con el manifiesto previo, sumando 244 ocurrencias. Las familias TCN/GRU–LSTM y el filtro 2019–2026 permanecen como declaraciones confirmadas por el autor. Un timestamp en el nombre de un CSV no prueba la fecha de la consulta. La ingesta anterior y sus 182 grupos no se reescriben.

## Fechas nominales 2027

Las páginas primarias de Springer corroboran **First Online 02/09/2026 para E1N025 y 02/08/2026 para E1N050**, aunque sus años nominales son 2027. Se preservan ambos campos. Para E1N082 se corroboró la publicación 2026 del libro padre, sin recuperar la fecha exacta del capítulo. Para E1N002 coincide el volumen LNNS 2055 con un libro publicado en 2026; la identidad y fecha exactas de su registro genérico de conferencia siguen parcialmente abiertas. E1N137 conserva fecha efectiva no resuelta. Estas comprobaciones no demuestran qué filtro se aplicó originalmente en Scopus y no cambian decisiones temáticas por sí solas.

La tabla de contenidos del volumen 2055 también permitió identificar una pista de LSTM–TCN+RL de Nayak. Su resumen editorial declara Optiver y un backbone conjunto: se registra sólo como contexto, sin contar TCN autónomo, extraer cifras o declarar lectura íntegra.

## Archivos y uso

- `search_log.json`: intentos efectivos, respuestas de acceso, búsquedas web complementarias y límites de captura. Las búsquedas web son externas al índice nativo.
- `source_reconciliation.csv`: identidad, conteo y huellas verificadas de los cuatro archivos recibidos, sin redistribuirlos.
- `query_plan.csv`: doce consultas **propuestas y no ejecutadas** para una nueva ronda, general y por familia. No reconstruyen las consultas históricas. La configuración nativa efectiva y sus totales tendrán que capturarse cuando exista acceso; los campos de las tres plataformas no tienen cobertura idéntica.
- `publication_dates.json`: dos fechas exactas corroboradas y tres casos que mantienen límites de identificación o fecha exacta.
- `related_lead.json`: pista contextual del editor, fuera de las frecuencias verificadas.
- `summary.json` y `manifest.json`: recuentos, entradas, alcance e integridad.

## Requisito que permanece abierto

Para cerrar íntegramente §3.1 de la guía se necesita la evidencia nativa de consulta: cadena literal generada por la plataforma, fecha/hora, colección y ediciones, filtros, total visible y rangos exportados. El lector disponible no puede recuperar el historial privado por Internet. Esta etapa **no acredita una consulta efectiva a IEEE Xplore ni declara cerrado ese requisito**. Conserva un expediente comprobable de lo recuperado y de lo no disponible, sin pedir ni publicar contraseñas, cookies o resúmenes de exportación.
