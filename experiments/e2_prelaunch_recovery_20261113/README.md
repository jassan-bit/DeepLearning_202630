# Conciliación de una reserva local que no llegó a publicarse

Incidente verificado: 72 ajustes terminados y respaldados (54 reproducciones y
18 ablaciones), diario remoto cerrado de 252 filas y ninguna reserva activa.
La siguiente reserva de iTransformer, semilla 271801, sólo se añadió al diario
local: 253 filas, el prefijo remoto exacto y un directorio que contiene únicamente
worker_spec.json. La traza falló dentro de budget.reserve, antes del lanzamiento.
La lectura posterior de procesos no encontró controladores ni workers activos.

Esta recuperación se limita a ese incidente y a los hashes registrados. No
reconcilia ejecuciones desconocidas ni modifica los ocho archivos científicos
originales o el adaptador ambiental histórico. Requiere la misma VM con el
lanzador b2199a8bc5df0e392430343b7891b11bedcd5306 y su entorno ya preparado.

Antes de modificar la caché local, bloquea ambos diarios, verifica sus hashes,
el estado remoto, la ausencia de procesos y de artefactos de ejecución; respalda
los dos diarios, la especificación y un recibo en GitHub. Sólo entonces pone el
directorio nunca ejecutado en cuarentena fuera de la campaña y recupera el
prefijo remoto como caché local. El registro de la reserva local se conserva
íntegro en ese respaldo. El diario original de 80 horas permanece intacto.

El controlador registra el tiempo real de recuperación y un cargo conservador
adicional de 300 segundos por la publicación fallida anterior: no se presenta
como duración medida de entrenamiento. El margen cubre hasta dos intentos de
publicación con timeouts HTTP de 20 segundos (hasta 220 segundos en la ruta del
fallo observado), más preparación y cierre. No se añade crédito ni se amplía
el límite de 80 horas.

El adaptador de transporte fuerza lecturas de ref sin caché. Sólo puede
reconocer una publicación cuyo estado completo sea idéntico al propuesto o
repetir esa misma publicación una vez al comprobar la base exacta. Un estado
ajeno sigue bloqueando la ejecución. No repite ajustes ni fuerza refs.

Después se invoca el controlador científico original, que vuelve a verificar
los resultados terminados y admite el siguiente trabajo sólo si cabe su reserva.
Cada ajuste de una fase mantiene su reserva conservadora de 1.860 segundos;
el saldo no garantiza terminar los 27 ajustes pendientes. Una interrupción de
entrenamiento mantiene las reglas originales de conciliación.

Validación: 27 pruebas (11 de consistencia, 12 de conciliación y 4 de lanzador),
con red simulada y archivos temporales. No se ejecutó la reparación ni nuevos
ajustes en el Colab privado desde este workspace. El usuario debe ejecutar la
celda verificada una sola vez en la misma sesión.
