# Familia y dimensionamiento: alcance posterior al piloto

La familia de 19 rivales y 57 componentes del corte anterior corresponde a V3.
Una nueva candidata V4 no recibe automáticamente esa cobertura. Antes de sus
resultados se propuso conservar los 19 rivales y añadir V3 original y MSE-only:
**21 rivales, 22 sistemas y 84 celdas panel/rival** para cuatro paneles futuros.
Tres procedimientos por rival darían 63 componentes propuestos, cuando sus
nulos, funciones y ajustes estuvieran justificados. No es una batería aprobada.

`V4_FAMILY_AND_POWER.DRAFT.json` y su plantilla conservan esa propuesta previa.
La referencia a 340 ajustes neuronales de planificación es aritmética nominal
de 17 recetas por cuatro paneles por cinco semillas; no es un presupuesto
medido, una recomendación de ejecución o una tarea autorizada tras el resultado.

**El piloto falló su regla y se abandonó V4.**
`POST_PILOT_DISPOSITION.json` detiene la planificación de esta receta. No se
lanzan los 340 ajustes ni se rellena la plantilla con varianzas de otra versión.
Las referencias pareadas nuevas sobre cinco semillas y 55 fechas expuestas de
D00 son descriptivas; no completan las 84 celdas futuras sin datos admitidos.

La guía exige tres pruebas pertinentes de igualdad junto con dirección, margen,
efecto, intervalos y sus otras condiciones de R3. No exige literalmente tres
nulos unilaterales `theta <= delta`; ésa fue una propuesta más fuerte del
borrador y de los ensayos controlados, que conservan su interpretación. MCS
está admitido en la guía, pero la extensión funcional ensayada necesita una
traducción contrastiva justificada y falló su calibración. No se cambia de
interpretación después de ver los resultados para declararla aprobada.

Una línea futura científicamente justificada necesitará datos admisibles,
hipótesis/nulos y familia explícitos, métodos calibrados, márgenes y alternativas
fundamentados, estimaciones conjuntas de semillas y calendario y R/T finales.
Debe dimensionarse al menos 0,90 por contraste y reportar la potencia conjunta;
no se inventa un mínimo conjunto de 0,90. Las familias adicionales y el análisis
entre conjuntos requieren su propio alcance previo, sin promediar RMSE de
escalas distintas. No se creó protocolo o Colab de confirmación.

Preparado con asistencia de IA; R3 permanece no acreditada.
