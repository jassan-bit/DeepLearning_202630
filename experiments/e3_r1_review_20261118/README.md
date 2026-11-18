# Expediente de revisión de R1

Fecha de preparación: 18 de noviembre de 2026. Preparado con asistencia de IA; la revisión humana independiente no se ha realizado.

Este expediente examina la referencia CNN causal–LSTM–atención por contenido con cabeza aditiva positiva de **1210 parámetros**, que ya tiene diez entrenamientos conservados. El corte de partida es el commit `76b2397d99027457b922225cec8a4f0aab1cc04e`. No corresponde al módulo inicial de 1211 parámetros ni a la propuesta por momentos de 1236 parámetros. No cambia modelos, predicciones o hipótesis históricas.

## Resultado y alcance

**R1 permanece pendiente de acreditación.** Completar registros y preparar un formulario no demuestra una contribución sustantiva ni constituye una revisión humana. El expediente puede terminar en un concepto favorable, desfavorable o no concluyente, de acuerdo con la evidencia.

La atención por contenido y las guardas de salida por muestra tienen antecedentes. La posible diferencia debe precisarse frente a ellos y superar una sustitución de enlace positivo o de aplicación. La identidad algebraica del soporte de la desviación estándar móvil se distingue de una innovación algorítmica.

## Contenido

- `search/`: acceso bibliográfico observado en este entorno y recolector reproducible para otras ejecuciones. Los errores de transporte conservan conteo desconocido. Recuperar metadatos no equivale a cribar títulos/resúmenes o leer textos completos.
- `mechanism/`: comparación del mecanismo, fuentes primarias, diferencias y límites del dictamen. Las propuestas de controles futuros no son experimentos ejecutados ni preregistros anteriores.
- `reviewer/`: instrucciones y formulario para una persona ajena al diseño. La identidad, búsquedas, fecha, concepto y declaración deben proceder de esa persona.
- `status.json`: estado explícito de cada condición de cierre, sin aprobar R1 por un contador de consultas.
- `manifest.json`: integridad SHA-256 de los archivos de este expediente; la integridad de bytes no certifica novedad.

## Condiciones de cierre

1. Consultar efectivamente al menos cinco cadenas en inglés y español en tres bases pertinentes; registrar consulta, fecha, alcance, conteos y cribado.
2. Revisar las fuentes cercanas, sus citas hacia atrás y adelante e implementaciones; documentar qué se ha leído y qué sigue sin comprobarse.
3. Delimitar una contribución en mecanismo, pérdida o entrenamiento y su diferencia sustantiva, procedencia y relación con el problema.
4. Repetir la búsqueda antes de la entrega definitiva.
5. Obtener el concepto de una persona ajena al diseño que repita al menos dos cadenas y conserve sus evidencias.

La búsqueda parcial previa se conserva en `experiments/e3_remediacion_20261118/literature/`. No se reescribe ni se cuentan búsquedas bloqueadas como consultas efectivas a otras bases. El dictamen de R1 tampoco concede R2 o R3.

## Ejecución y conservación

La recolección bibliográfica funciona con Python y CPU. No necesita GPU, datos de mercado ni nuevos entrenamientos. La ejecución en otro entorno se guardará en una rama nueva de resultados para mantener su código fuente, momento de consulta y recibos; no sustituye los archivos históricos ni aprueba automáticamente el cribado o la revisión externa.

No se incluyen artículos completos de terceros, firmas inventadas, credenciales ni datos personales del futuro revisor. No se envían invitaciones o mensajes a terceros desde este expediente.
