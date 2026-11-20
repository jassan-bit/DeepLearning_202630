# Aceptación pública posterior al Manual Deploy

## Revisión v2 del fixture del ejecutor

La primera ejecución pública conserva su recibo: catorce predictores pasaron la aceptación completa. El cliente de cinco llegó a tres páginas y falló antes de las predicciones porque el checkout sparse del ejecutor omitía su CSS local de referencia. No fue un fallo de Render ni de los modelos. Esta revisión añade `assets` al checkout y verifica por hash antes del navegador sus tres comparadores públicos locales —MODEL_CARD, DATA_STATEMENT y CSS—; ambos clientes siguen sin cambios. El paquete v1 no se reescribe.

Un paso posterior conserva únicamente capturas públicas mediante fragmentos base64 de hasta 49152 caracteres, con ruta, número de fragmento, tamaño y SHA256. No imprime pesos, cookies, variables o archivos arbitrarios. Permite recuperar los mismos bytes si el acceso al ZIP de Actions queda bloqueado por el transporte del consumidor.

Este lanzador ejecuta los clientes existentes de cinco y catorce predictores sin modificar sus fuentes. La revisión y preparación tienen asistencia de IA declarada. No entrena, exporta, calibra, despliega servicios ni abre una campaña científica.

La fuente esperada es `feb83dbc08197bde7076857bd8ff475de197efd2`; la URL fija es `https://deeplearning-e3-dashboard.onrender.com`. Antes de cualquier predicción exige checkout Git exacto, SHA/tamaños de ambos clientes y sus referencias, salud con aquel SHA, perfil14 y V3 deshabilitada. Los artefactos de cinco modelos se verifican además mediante el cliente V2. No necesita una consulta REST GitHub sin autenticar: la identidad del checkout y los archivos fijados son su comprobación de repositorio.

El workflow sólo lee el repositorio y el servicio público. Usa Playwright 1.56.0, greenlet 3.5.6, pyee 13.0.1 y Python 3.12.14; instala Chromium, no bibliotecas de entrenamiento. El evento `create` queda limitado a ramas `e3-render-public-acceptance-*`, permitiendo ejecutar el paquete sin modificar `main`. No configure simultáneamente `push` para evitar dos aceptaciones por la misma creación de rama.

Los recibos, stdout/stderr, salud y capturas se retienen aunque falle la aceptación. El JSON también se imprime con prefijos `E3_PUBLIC_*` para su recuperación semántica desde los logs públicos. El manifiesto registra hashes de los bytes originales guardados. La serialización recuperada de logs no debe presentarse como el hash del ZIP original de Actions.

`scope=public_acceptance` y `public_Render_acceptance_passed=true` sólo se acreditan al pasar las comprobaciones del cliente sobre el origen HTTPS real, incluidos catorce modelos por API/navegador, OOD, guardas temporales, errores seguros, huellas e intervalos, más las tres rutas/cinco modelos del cliente original. No acredita R1/R2/R3, confirmación, reproducción humana integral, memoria, concurrencia remota o disponibilidad sostenida.

Las pruebas locales de guardas no llaman a red o predictores. Debe distinguirse este paquete preparado de una ejecución pública aprobada: el recibo real de Actions será la evidencia de la ejecución.
