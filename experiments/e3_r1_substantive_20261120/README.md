# Revisión de la contribución sustantiva de V3 para R1

Revisión por IA declarada, **20 de noviembre de 2026, America/Bogota**.
Base del libro: `29cedb5228ebcecce2a354a1d0ab89adf3936f12`.
La revisión separa novedad para el problema, implementación defendible y
superioridad estadística. No incorpora entrenamientos, inferencias de modelos,
recalibraciones, remuestreos o una campaña nueva.

## Resultado y objetos separados

- **Referencia aditiva:** la revisión no acredita una adaptación sustantiva.
  Sus resultados favorables y los 796 trabajos siguen siendo su evidencia.
- **V3 completa `atom_crps`:** el dictamen independiente por IA apoya,
  con alcance explícito, la adaptación de lectura y pérdida para esta tarea.
  No reclama nuevas mezclas, CRPS, una identidad de varianza o un teorema inédito.
- **Utilidad:** el piloto de 30 ajustes no superó su heurística; `atom_crps`
  tuvo 2,70 % más RMSE que `atom_mse_only`. Se conserva el resultado y no se
  promociona esa ablación. El dictamen de novedad no cambia esa decisión.
- **R2/R3:** no se acredita una defensa del estudiante ni una confirmación
  de superioridad. Tampoco se asigna una nota o un tope automáticamente.

## Evidencia de la revisión

`contribution_declaration.md` y `procedencia_v3.md` fijan la pretensión actual.
La matriz actual está en `r1_evidence_matrix.csv`; la extracción de requisitos
en `requirements/` conserva expresamente el diagnóstico de la base anterior
a esta integración. `independent_review/` identifica el objeto V3, criterio,
lecturas y dictamen. Es revisión técnica por IA bajo la modalidad autorizada
comunicada por el autor, no una firma humana ni aprobación institucional.

`code_theory/` vincula ecuaciones, líneas y diez insumos; verifica exactamente
28 coeficientes de la identidad poblacional y la coherencia de los cien casos
artificiales ya guardados. No vuelve a construir o ajustar una red.

`literature_distribution/` y `literature_application/` conservan fuentes,
partes efectivamente leídas, diferencias, accesos fallidos y seguimiento desde
cinco trabajos semilla. Los repositorios sólo se inspeccionaron en metadata,
README o licencia: no se copió o ejecutó su implementación.

La cobertura histórica efectiva de Crossref, arXiv y OpenAlex se reutiliza
dentro de su alcance. Las consultas web focales actuales y las dos cadenas
repetidas por el revisor IA se registran aparte. Los nuevos intentos API
bloqueados por el proxy **no** se cuentan como respuestas bibliográficas,
resultados vacíos ni otras tres bases efectivas. No se promete exhaustividad.

## Conservación de versiones

El mecanismo auditado es el del código `8f65eac34aaf49293924ae8328c235cedbdd2866`;
el piloto es `20261120T031424176339Z`, resultados
`8eaa581bc34b615a8507c9e384a3e4be29084720`. Los 796 trabajos no se atribuyen a V3.
`experiments/procedencia.md`, `experiments/design_log.md` y los expedientes
anteriores conservan sus bytes y sus fechas. Esta procedencia V3 versionada
es su actualización posterior, no una reconstrucción de inspiración histórica.
La fe de erratas distingue los dos trabajos MDN de 1998 sin reescribir el PDF
o los recibos de la revisión anterior.

El ZIP documental del cierre anterior no contiene esta revisión nueva; sus
miembros y hash siguen siendo los del corte declarado en su propio recibo.
Aquí se entregan los archivos y manifiestos por separado.

## Comprobación

Desde la raíz del repositorio:

```bash
python experiments/e3_r1_substantive_20261120/verify_packet.py
```

Se verifican integridad, espejos publicados y preservación del cierre anterior.
El verificador no decide la novedad por comparar hashes. El dictamen de R1
es una evaluación razonada que se entrega como documento independiente.
`AUTHOR_ACTIONS.md` y el formulario Faithful dejan listas las decisiones del autor.
