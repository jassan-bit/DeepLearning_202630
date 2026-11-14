# Cierre focal de literatura E2, 14/11/2026

Contabilidad de30objetivos dirigidos y adjudicación de18informes con secciones primarias previamente evaluadas:13incluidos aplicados,2contextos de mecanismo con elegibilidad aplicada pendiente y3excluidos. Otros4objetivos se excluyen por población y8no tienen original completo acreditado. No cierra9.559grupos,1.078registros ni40candidatos.

`eligibility/` conserva decisiones individuales, síntesis, criterios, versiones, límites y43roles de fichas primarias históricas. `cross_entregable/` enlaza107fichasE1 porDOI exacto y43recibos primarios, sin importar automáticamente elegibilidad.

`citation_score/` amplía Pcontextual a9modelos dentro de tablas; mantiene Cdesconocida para4sistemas y PdeInception individual pendiente.45/50componentes contextuales y6scores condicionales completos no equivalen a ranking controlado global ni selección previa. Los puntajes y artefactos anteriores están intactos.

Auditoría sin red desde la raíz del repositorio:

```bash
python scripts/audit_e2_literature_delivery.py
python -m unittest tests.test_e2_literature_cross_entregable
```

Los scripts de `citation_score/` son portables y verificaron8ejecuciones en ubicación original y relocalizada. Para regenerar sus derivados, copie primero la carpeta a un directorio nuevo y establezca `E2_REPOSITORY_ROOT` al checkout: no sobrescriba el paquete sellado. Los scripts `build_citation_audit.py`, `contextual_p/build_contextual_p.py`, `contextual_p/build_preferred_intrapaper_scenario.py` y `build_conditional_score_scenarios.py` reconstruyen tablas desde los insumos documentales; los recibos de ejecución conservan sus propias fechas. `eligibility/build_eligibility.py --root <checkout> --output <nuevo_directorio>` reconstruye el ledger focal y los roles desde fuentes públicas previas.

El código no entrena ni infiere modelos. `manifest.json` sella todos los archivos y los insumos del repositorio.
