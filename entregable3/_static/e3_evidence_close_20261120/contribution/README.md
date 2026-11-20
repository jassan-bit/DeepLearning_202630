# Consolidación de contribución y procedencia con evidencia existente

Paquete de seguimiento que deja inmutable el expediente bibliográfico anterior. Prepara el contenido final de `experiments/procedencia.md`, una declaración precisa, matrices de componentes/fuentes/requisitos y propuestas de actualización del libro. Se declara asistencia de IA; no se firman actos humanos ni se acreditan R1/R3.

El piloto V3 de 30 ajustes ya terminó. Su candidata completa no superó la heurística y fue 2,70 % peor que el mejor control, sin CRPS. Este paquete conserva ese resultado y las diferencias teóricas sin promocionar la variante ganadora. El análisis y la decisión son de desarrollo, no una confirmación estadística.

- `procedencia.md`: contenido a publicar en la ruta canónica `experiments/procedencia.md`.
- `contribution_declaration.md/json`: contribución admisible, resultados y límites.
- `mechanism_loss_training_application.csv`: mecanismo, pérdida/entrenamiento y aplicación separados.
- `literature_crosswalk.csv/md`:15 registros de antecedentes, con acceso anterior preservado.
- `r1_evidence_matrix.csv/md`: cierre documental y requisitos que siguen abiertos.
- `source_references.json`: versiones y enlaces inmutables.
- `book_replacements.json` y `book_updates.md`: propuestas para capítulos6/7; el ensamblador verifica cada coincidencia literal antes de aplicar.
- `verify.py`, `manifest.json` y `verification.json`: integridad y controles de coherencia; no entrenan, infieren ni hacen peticiones de red.

No se modificaron los módulos, pesos, reglas de continuación, semillas, particiones o recibos anteriores. Las referencias de acceso se reutilizan de registros guardados; no se anuncian nuevas búsquedas o lecturas. El inventario global de diseños conserva categorías separadas y no usa semillas como diseños nuevos.
