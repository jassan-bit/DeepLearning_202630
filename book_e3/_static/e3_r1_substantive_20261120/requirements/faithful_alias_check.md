# Auditoría documental de la referencia «su propia media sola»

Fecha de esta aclaración: 20 de noviembre de 2026, America/Bogota. Asistencia de IA declarada. Este dictamen es una lectura retrospectiva de registros del proyecto; no es una implementación, entrenamiento, búsqueda bibliográfica nueva ni preregistro.

**Dictamen: no se puede resolver la identidad de una receta histórica completa con los registros inspeccionados.** Se conservan **22 entradas documentadas y un bosquejo ambiguo**, sin sumar una arquitectura ni restar una entrada por semejanza de nombres. El intervalo 22–23 sólo representa esta ambigüedad dentro del corpus; no acota todas las ideas o decisiones informales.

## Evidencia que se cotejó

Se verificaron los SHA-256 de los **21 miembros propios** de `inventory/sources.zip` contra `source_manifest.json`; todos coinciden. No se abrieron checkpoints ni datos de mercado. `faithful_alias_check.json` conserva las coincidencias de texto y los hashes.

En esos 21 insumos, las referencias a Faithful aparecen en `moment_pilot_plan.json` y `moment_design_spec.json`: describen un control inspirado en Faithful aún no implementado y niegan heredar una garantía de trayectoria. La frase específica «su propia media sola» se encuentra en la tabla del **capítulo 7, línea 314**, y en el **anexo G, línea 40**. Esas fuentes del libro son adicionales al ZIP y no se cuentan como nuevos diseños por ser nuevos archivos consultados.

Además se recuperaron dos documentos propios del mismo commit `29cedb5228ebcecce2a354a1d0ab89adf3936f12`, conservados en `faithful_sources/` y verificados por su SHA de blob Git:

| Fuente propia congelada | Parte decisiva | Blob Git |
|---|---|---|
| [Propuesta de momentos](https://github.com/jassan-bit/DeepLearning_202630/blob/29cedb5228ebcecce2a354a1d0ab89adf3936f12/experiments/e3_mecanismo_momentos_20261116/README.md) | Líneas 74–82: media directa con auxiliares, control detenido al contexto, adaptación y comparación exacta por especificar. | `fe105629ef23dafd0b4db1afb4cb8404443c182a` |
| [Comparación focal primaria ya existente](https://github.com/jassan-bit/DeepLearning_202630/blob/29cedb5228ebcecce2a354a1d0ab89adf3936f12/experiments/e3_dictamen_momentos_20261116/primary_comparison.md) | Líneas 7 y 15–18: definición del comparador y diferencia entre ancestros de la media y auxiliares separados. | `1e7d251522604ac093c1dc6250e938490c027d6e` |

Estos son registros propios de lecturas anteriores. **No se declara haber releído ahora el artículo Faithful ni se transforma una lectura focal histórica en lectura íntegra nueva.** El requisito humano de aquel texto conserva su fecha: la autorización posterior de revisión por IA ya documentada en 6.13 no se reabre aquí.

## Lo que sí permite inferir la formulación

La propuesta describe una media directa `sqrt(A) + s_y softplus(r)` con dos lecturas auxiliares `m,q`. Si esas auxiliares no son ancestros de la media, eliminarlas junto con sus scores dejaría una sola lectura de media. **Si además coincide exactamente el codificador, la lectura, las escalas, la transformación positiva y la pérdida de N06**, ese mapa puntual sería el control aditivo ya inventariado.

Esa afirmación es condicional; no certifica que dos recetas congeladas sean idénticas. El código aditivo real usa `sqrt(A) + y_scale * (1e-4 + softplus(score))`, no sólo la fórmula abreviada sin epsilon: véase [e3_innovation_model.py, líneas 26 y 294–300](https://github.com/jassan-bit/DeepLearning_202630/blob/29cedb5228ebcecce2a354a1d0ab89adf3936f12/src/e3_innovation_model.py#L294). También fija lectura FP64 y codificador FP32. El bosquejo P01 no congela todos estos detalles ni un ejecutable de su referencia de media sola. La equivalencia simbólica no acredita la misma inicialización, actualización u optimizador.

Tampoco hay que confundir este bosquejo con **N10 `moment_mean_only`**, que ya está contado y conserva las tres lecturas de la cabeza acoplada de momentos. En esa cabeza, `m,q` sí son ancestros de la media. Quitar sólo sus scores no elimina esos ancestros ni produce la cabeza directa N06.

Por ello, no se fuerza un alias con N06, no se añade un diseño 23, no se afirma una garantía Faithful y no se inventan ejecuciones. Una nueva formulación propuesta hoy no demostraría qué diseño se había considerado históricamente.

## Material mínimo preparado para el autor

`faithful_author_decision.DRAFT.json` es un formulario sin diligenciar ni firmar. Ofrece tres decisiones posibles:

1. **Alias respaldado por un registro existente.** Aportar ruta, commit o archivo original y su fecha; identificar ecuación completa de la media, epsilon/escalas, ancestros, pérdida y relación con N06. Ese registro permite justificar la deduplicación; una afirmación retrospectiva sin soporte no se convierte en preregistro.
2. **Diseño distinto respaldado por un registro existente.** Aportar ese registro y localizar el cambio concreto de mecanismo, pérdida o entrenamiento. Añadir evidencia de código/ejecución sólo si realmente existió. No contar una semilla o inicialización como arquitectura nueva.
3. **Sin registro suficiente.** Mantener 22 identidades documentadas más el bosquejo ambiguo. No completar hoy una especificación y presentarla como histórica.

No hace falta ejecutar otro piloto para emitir esta aclaración documental. La ambigüedad de esta referencia histórica tampoco demuestra, por sí sola, que V3 carezca de una diferencia sustantiva.
