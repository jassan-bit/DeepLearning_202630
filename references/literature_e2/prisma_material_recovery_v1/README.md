# Restauración verificable de material bibliográfico

Scopus7500 y WoS302/500/500 coinciden por SHA con los cuatro originales iniciales registrados. WoS340 coincide con el original enfocado. Otros WoS aportados se enlazan sólo por identidad, sin atribuirles una consulta histórica no verificada. El ZIP histórico contiene derivados públicos y no añade los originales privados enfocados.

El paquete reconstruye la disponibilidad de material para los mismos9559grupos recibidos. Hay8396grupos con resumen auténtico:20celdas «[No abstract available]» no cuentan como resumen. No se inventa una exclusión científica por ausencia de material. Los530resúmenes arXiv y13enfocados recuperados después se documentan en el paquete de flujo recibido, con8939grupos de cobertura almacenada disjunta.

Los archivos públicos contienen metadatos, enlaces, hashes y conteos; los originales y los cuerpos de resúmenes permanecen fuera de este paquete. `source_coverage.csv` registra archivos lógicos y sus identidades, `identifier_links.csv` conserva posición y regla de enlace y `identity_conflicts.csv` separa conflictos sin resolverlos por similitud. La coincidencia subsidiaria exige título normalizado exacto, año concordante, candidato único y ausencia de DOI contradictorio.

Para reconstruir desde las copias del autor, utilizar `scripts/restore_e2_literature_materials.py --input-map <mapa_privado.json> --output-dir <directorio_privado_externo>`. El mapa asocia los source_id del recibo a los archivos originales. La salida privada queda fuera del repositorio público. `scripts/audit_e2_prisma_material_recovery.py` verifica los derivados y las identidades históricas sin requerir los cuerpos.

Los374Scopus mostrados pero no recibidos y la correspondencia incompleta de totales/rangos WoS son límites de cobertura. Esta restauración no amplía el universo histórico, no crea un preregistro ni ejecuta entrenamiento.
