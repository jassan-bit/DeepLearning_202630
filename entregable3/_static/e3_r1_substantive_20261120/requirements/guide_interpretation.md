# Lectura operativa de R1 y mapa de requisitos

Revisión documental por IA declarada, 20 de noviembre de 2026, America/Bogota. Corte auditado: `29cedb5228ebcecce2a354a1d0ab89adf3936f12`. Fuente: guía aportada por el usuario, SHA-256 `d6048aebd7b3db85a59231a5a89b01bc3f8ad8cb581a96b462e341d21ff1f364`. Las páginas citadas son las impresas y coinciden con las páginas del PDF inspeccionado. No se realizó aquí otra búsqueda bibliográfica, entrenamiento, inferencia o remuestreo.

## Qué exige realmente R1

La definición de **§6, p.7** contiene dos condiciones simultáneas: novedad para el **mismo fenómeno y la misma tarea** de predicción, y diferencia de mecanismo, pérdida o entrenamiento, en lugar de cambiar sólo la aplicación. La transferencia está permitida si cita con precisión la fuente y añade una modificación algorítmica sustantiva; cambiar datos o hiperparámetros no basta.

R1 no exige inventar por primera vez en todo campo científico un componente elemental, ni un teorema inédito. CNN, LSTM, atención, distribuciones condicionales, CRPS, Jensen o una identidad de varianza pueden utilizarse con atribución. Lo que debe defenderse es la adaptación concreta para el problema, sin presentar esos elementos conocidos como invenciones propias.

Tampoco puede exigirse una demostración matemática de ausencia de todo antecedente mundial. **§6.3, p.12** prescribe una revisión verificable y una declaración acotada a la búsqueda descrita. Eso no convierte cualquier búsqueda sin hallazgos en prueba suficiente: deben examinarse los trabajos realmente cercanos, describirse sus diferencias y conservarse los límites de acceso y cobertura.

## Cómo se sustenta la revisión

La sección 6.3 pide tres bases, cinco cadenas como mínimo que combinen familia del mecanismo, problema y sinónimos en inglés y español, búsqueda en otras aplicaciones, fecha, matriz de títulos/resúmenes y textos completos pertinentes, seguimiento de citas desde tres a cinco trabajos cercanos, repositorios/tablas de resultados, declaración precisa y repetición antes de la entrega. La matriz debe clasificar la diferencia como mecanismo, pérdida/entrenamiento o sólo aplicación.

Las respuestas históricas válidas de Crossref/arXiv/OpenAlex permanecen válidas dentro de su alcance. Dieciocho intentos nativos posteriores bloqueados no son dieciocho búsquedas con cero estudios ni una nueva cobertura efectiva. Un acceso a fuentes primarias mediante otro canal puede proporcionar evidencia de lectura, pero no se renombra como respuesta de una base que no contestó.

**La modalidad de tercero mediante IA ya fue autorizada y documentada.** El expediente de 6.13 contiene un dictamen y dos repeticiones léxicas de la referencia aditiva. No hay que pedir otra persona o firma humana para ese componente. Sin embargo, ese dictamen no acredita automáticamente una candidata distinta: el concepto técnico actual debe identificar expresamente V3 y las fuentes que compara. Eso es actualizar el objeto del dictamen bajo la modalidad aceptada, no reabrir la autorización.

## Qué pertenece a requisitos conjuntos o a otras condiciones

El encabezado de **§14.5.1, pp.43–44** dice condiciones R1 **y R2**. Sus siete casillas comprenden adaptación sustantiva, procedencia, limitación cuantificada, hipótesis/predicciones secundarias oportunas, propiedad formal, núcleo propio y revisión de literatura. No deben leerse todas como definición exclusiva de novedad:

- La programación del núcleo desde la formulación y su comprensión defendible son **R2** (§6, pp.7–8). El uso de PyTorch o capas primitivas está permitido. La sustentación no observada no prueba copia o un import del modelo.
- Una propiedad con supuestos, demostración y verificación es un requisito del modelo original y del conjunto de §14.5.1; **no acredita la novedad por sí sola**. La guía no exige que esa propiedad sea un teorema nuevo (§7.3, p.19; Cuadro 8, p.59).
- La comparación significativa contra toda la familia es **R3** (§6, p.8). No hay que ganarla para argumentar R1. Potencia, diez semillas y tres pruebas no son sustitutos de la revisión de novedad.
- La bitácora `N_d` y la frontera entre desarrollo y confirmación pertenecen a la disciplina de diseño y comparación (§7.1 y §14.5.2). La ambigüedad de un bosquejo histórico no refuta automáticamente la novedad de V3 ni obliga a inventar un total de todos los pensamientos informales.

Los topes de Cuadro 8 son condicionales: 3,0 si R1 o R2 no se cumplen; 4,0 si R1/R2 se cumplen y R3 no; 5,0 sólo si se satisfacen las condiciones y la rúbrica. Un estado pendiente no demuestra por sí solo uso previo o copia. Un juicio de no acreditación debe identificar qué evidencia sustantiva falta, sin atribuir conductas no observadas.

## Dictamen sobre la evidencia base, antes de integrar la revisión focal actual

**Aditiva:** las matrices anteriores no establecen una adaptación sustantiva frente a los antecedentes conocidos de atención y guarda de soporte. Una buena RMSE no completa ese requisito. Se mantiene como referencia, sin convertir la falta de acreditación en prueba de inexistencia mundial de cualquier composición novedosa.

**V3:** existe una diferencia documentada de estructura y entrenamiento: ley finita del retorno faltante, transformación dependiente de seis retornos observados antes de promediar y señal auxiliar CRPS. Se debe decidir si esa adaptación es sustantiva frente a los antecedentes cercanos reales y si la búsqueda respalda novedad para el fenómeno/tarea. Esta extracción de requisitos no sustituye ese dictamen; las matrices deben integrarse con la lectura focal y la auditoría del código.

La etiqueta implementada es la **desviación poblacional** de siete retornos (`ddof=0`), con `A = sum((r_i-m)^2)/7` y `c = 6/49`. No se usa aquí la fórmula de desviación muestral `ddof=1`.

El piloto completado aporta 30 ajustes de desarrollo y muestra que `atom_crps` fue 2,70 % peor que `atom_mse_only`, sin superar su heurística. **Ese resultado debilita la hipótesis de utilidad de la receta completa; no demuestra ausencia de novedad.** Se conservan los 796 trabajos anteriores, la regla y todas las ablaciones. El criterio previo del plan dice explícitamente que la heurística no significa R1 o R3.

## Ajustes editoriales concretos que evita esta lectura

1. En 6.16 y la fila actual de R1 de 6.1, separar «prioridad para el fenómeno/tarea» de «utilidad» en vez de pedir ambas como definición de R1.
2. No presentar como pendientes actuales «utilidad por contrastar» o «controles futuros» de 7.12 sin el rótulo de corte histórico: el piloto posterior ya terminó y fue desfavorable. No reescribir las decisiones originales; añadir una aclaración con fecha actual.
3. No exigir «mejora demostrada frente a la red aditiva» como propiedad formal mínima. Es una afirmación empírica más fuerte y distinta de las cotas de soporte/Jensen o del riesgo bajo ley verdadera.
4. Conservar las hipótesis de la propiedad: ley fijada, `A>0` cuando la cota lo requiere y ley verdadera para la identidad de riesgo-oráculo. La identidad no prueba una ventaja de una ley aprendida mal especificada ni incapacidad expresiva de la aditiva.
5. Mantener dos estados separados: la modalidad IA externa ya aceptada y el dictamen sustantivo particular sobre V3. La nueva revisión no inventa firma humana o aprobación del profesor.

## Material mínimo si el contraste actual todavía no permite acreditar R1

El vacío no se resuelve con otra campaña grande. Debe prepararse una sola ficha comparativa que identifique: antecedente más cercano; ecuación original; ecuación y código de la adaptación; cambio efectivo en mecanismo/pérdida/entrenamiento; limitación abordada; evidencia de que la misma composición no fue encontrada en la búsqueda delimitada para ese fenómeno/tarea; y objeción más fuerte con respuesta sustentada. Si un antecedente ya tiene esa composición, hay que decirlo y retirar la afirmación correspondiente, sin escoger retrospectivamente un nuevo modelo ganador.

Para la identidad Faithful se adjuntan `faithful_alias_check.md` y un formulario sin diligenciar. Sólo un registro existente puede resolver una receta histórica. No hace falta entrenar ni completar hoy una fórmula para fingir que ya estaba congelada.
