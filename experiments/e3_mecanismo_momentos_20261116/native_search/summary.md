# Intentos nativos de acceso bibliográfico

**Fecha: 16 de noviembre de 2026. Resultado: tres solicitudes reales, ninguna respuesta de las APIs. R1 y el cumplimiento de la revisión de §6.3 siguen pendientes.** El registro exacto, las URLs y las cinco familias de consultas previstas están en [requests.json](requests.json).

## Qué exige la guía original

Se verificó directamente `Entregable3DeepLearningProject.pdf`, SHA-256 `d6048aebd7b3db85a59231a5a89b01bc3f8ad8cb581a96b462e341d21ff1f364`, §6.3, páginas físicas 11–12. La página 12 exige al menos tres bases de datos y al menos cinco cadenas que combinen mecanismo, problema y sinónimos en inglés y español; también pide buscar otras aplicaciones y registrar la fecha. Añade matriz de cercanos, búsqueda por citas y repositorios, repetición antes de entregar y verificación por otra persona ajena al diseño.

La guía presenta Scopus o Web of Science, IEEE Xplore o ACL Anthology como ejemplos según el problema, y arXiv, OpenReview o SSRN como fuentes complementarias. No impone exclusivamente esos nombres. Tampoco menciona OpenAlex o Crossref ni declara que tres servicios públicos de metadatos satisfagan por sí solos sus requisitos. Su uso como apoyo bibliográfico **no certifica equivalencia** con la revisión exigida.

## Solicitudes efectivamente realizadas

Se usó Python `urllib.request` desde el entorno de ejecución para consultar directamente tres APIs, con un máximo de tres registros por solicitud, timeout de doce segundos y sin credenciales. No se sustituyeron por búsquedas web generales.

| Servicio | Operación nativa intentada | Resultado observado |
|---|---|---|
| OpenAlex | `https://api.openalex.org/works`, búsqueda `moment bounds neural volatility`, `per-page=3` | `URLError`: `Tunnel connection failed: 403 Forbidden`. |
| Crossref | `https://api.crossref.org/works`, `query.bibliographic=moment bounds neural volatility`, `rows=3` | El mismo error de túnel. |
| arXiv | `https://export.arxiv.org/api/query`, consulta booleana sobre `moment bounds`, `neural/learned` y `volatility/volatilidad`, `max_results=3` | El mismo error de túnel. |

El rechazo corresponde a la conexión mediante el túnel del entorno, **no a una respuesta HTTP 403 demostrada del servidor de cada API**. No se recibió un cuerpo bibliográfico ni un total de resultados; no consta que los motores llegaran a ejecutar las consultas. Los conteos de artículos son desconocidos, no cero. El primer intento no instrumentó hora UTC exacta; el JSON no la reconstruye. Los dos siguientes conservan sus horas reales de inicio y fin.

Tras el primer bloqueo de cada servicio no se repitieron consultas contra ese servicio, no se cambiaron hosts o proxies y no se solicitaron credenciales ni excepciones. Las cinco familias previstas abarcan cotas de momentos, elicitación conjunta, QLIKE/cuasi-Gamma, interpolación entre cotas y aplicaciones a opciones/riesgo. Permanecen **planificadas, no completadas**; las formulaciones bilingües del registro no se cuentan como cadenas ejecutadas. Por la restricción observada no se alcanzaron cinco consultas completas.

## Alcance del registro

Los dieciocho intentos denegados anteriormente en E3 son historia previa; no se suman como búsquedas nuevas ni como respuestas exitosas. Estos tres intentos tampoco acreditan tres bases consultadas con resultados ni la matriz de novedad. La búsqueda nativa, la equivalencia del alcance bibliográfico y la verificación humana externa siguen abiertas. No se afirma ausencia de antecedentes para la cabeza por momentos.

Este registro no consulta datos de mercado, no ajusta modelos, no realiza inferencias de mercado y no modifica los resultados V1/V2. Documenta una limitación de acceso; no constituye aprobación de R1 ni autorización de entrenamiento.
