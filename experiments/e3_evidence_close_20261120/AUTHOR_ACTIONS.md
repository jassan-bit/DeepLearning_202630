# Acciones concretas del autor después del cierre documental

Los archivos están preparados; no deben completarse con firmas, fechas o
aprobaciones supuestas. El registro de una decisión nueva conserva su fecha real.

1. **Una identidad de diseño pendiente.** En `inventory/reconciliation.json`
   y `design_inventory.csv`, el bosquejo «su propia media sola» inspirado en
   Faithful no tiene formulación suficiente. Indicar ecuación, pérdida,
   código y relación con la aditiva para decidir si es otro diseño o un alias.
   Documentar cualquier ensayo adicional conservado fuera del corpus; no
   declarar que 183 HPO enumera todas las decisiones informales.
2. **Defensa de la contribución.** Revisar `contribution/r1_evidence_matrix.csv`,
   `experiments/procedencia.md` y las propiedades de las distintas arquitecturas.
   Explicar la diferencia estructural/de entrenamiento y los límites de prioridad.
   No atribuir novedad a CNN/LSTM, CRPS, Jensen o la identidad de varianza.
   El fracaso de la heurística no es prueba de ausencia mundial de novedad;
   tampoco autoriza a convertir la ablación ganadora en la hipótesis original.
3. **Render.** Después de la publicación, efectuar Manual Deploy del commit
   final y esperar Live. Copiar su SHA completo del log, ejecutar la celda CPU
   preparada en `release/colab_cell.py` e introducir ese SHA cuando se solicite.
   Conservar el ZIP de recibo y capturas antes de cerrar Colab y subirlo al
   repositorio. El proxy de este entorno impidió obtener HTTP de origen.
4. **DOI y derechos.** Revisar el ZIP documental concreto, su manifiesto y
   `release/zenodo_upload_fields.DRAFT.json`. El suplemento es de alcance
   limitado. Para archivar el entregable integral, seguir también
   `release/full_archive_plan.DRAFT.json`: el delta 193 no contiene por sí solo
   todo el master de 796 trabajos. Ratificar derechos y elegir las licencias
   propias sin extender las de Bootstrap a todo el proyecto. No publicar
   automáticamente el borrador por disponer de un DOI reservado. Conservar
   el enlace público, metadatos finales y los hashes descargados al publicar.
5. **Confirmación futura.** Los calendarios ya expuestos, los márgenes no
   registrados y la potencia prospectiva no se recuperan retrospectivamente.
   Una eventual propuesta necesita datos adecuados, familia completa,
   hipótesis y márgenes, dimensionamiento y protocolo previos. Este cierre
   no ordena ni autoriza automáticamente otra campaña.
6. **Reproducción y sustentación.** Obtener un concepto independiente con
   alcance y discrepancias explícitos y realizar la defensa del estudiante.
   El Docker histórico reproduce once agregados desde 143 trayectorias;
   ni ese recibo ni este suplemento equivalen a reproducir todos los pesos
   o entrenamientos de la campaña. La explicación y modificación en vivo
   del algoritmo no se sustituyen por pruebas o declaraciones de IA.

Las nueve exportaciones neuronales restantes y los estudios completos de
interpretabilidad, FLOPs/energía y recursos conservan sus requisitos. No se
declaran excluidos por falta de recursos sin una medición que lo justifique.
