# Diagnóstico estadístico del suplemento CPU

Esta campaña conserva los pesos congelados y usa únicamente sus nuevos riesgos
CPU train655/calibración53 y sus nuevas latencias CPU. Las diferencias frente a
las proyecciones GPU originales se publican completas: 1e−7 es una alerta, no un
filtro ni una puerta aprobada. Los dos intentos originales fallidos se conservan.

La tolerancia δg usa seis brechas por red, t unilateral inferior con5gl y Holm13.
El IC95 % es marginal y describe optimización condicionada a estos periodos y a
la aproximación t; no demuestra ausencia de sobreajuste del benchmark original.
La calibración no se convierte en validación independiente de selección.

El coste usa100bloques completos de medición pareada, manteniendo seis semillas
por candidato/referencia. Se publican medias, diferencias, IC percentiles20.000 y
p unilaterales con Holm10 propio. Requiere estabilidad suficiente entre bloques;
no convierte repeticiones del reloj en mercados independientes. Esta comparación
CPU no sustituye los costes A100 ni acredita una mejora conjunta de error y coste.

No se ejecuta la IUT del plan714, no se mezcla calibración CPU con test GPU y no
se modifica A2, RQ1, ningún peso, calibrador o margen. No rechazo no demuestra
equivalencia. El plan estadístico y el de inferencia se vinculan antes de los
forwards de esta campaña; el registro es posterior al benchmark y a dos fallos.
