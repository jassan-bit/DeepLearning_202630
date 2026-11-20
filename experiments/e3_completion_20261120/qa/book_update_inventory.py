"""Locate current-state prose requiring reconciliation; no book mutations."""
from pathlib import Path
import csv
import json

ROOT = Path('/workspace/e3_completion_worktree/book_e3')
OUT = Path('/workspace/e3_completion_delivery/qa')
entries = [
 ('intro.md',13,'current_inventory','Añadir nueve ONNX verificados sin activarlos: catorce representantes históricos exportados en total; V3 experimental por separado.'),
 ('intro.md',68,'current_pending_export','Mantener cinco habilitados y doce precalculados; sustituir pendiente de exportación de nueve redes por exportación offline verificada y habilitación/aceptación de producto pendiente.'),
 ('intro.md',88,'current_pending_export','Quitar el pendiente técnico de nueve exportaciones y conservar política final común, capacidad, producto y confirmación.'),
 ('sections/01_introduccion.md',11,'current_R1_global','Precisar: la aditiva carece de acreditación sustantiva; V3 tiene dictamen IA favorable acotado en6.18, sin aceptación institucional, defensaR2 niR3.'),
 ('sections/01_introduccion.md',42,'current_pending_export','Sustituir nueve redes esperan exportación por nueve exportadas offline; perfil servido sigue de cinco hasta integración y aceptación verificadas.'),
 ('sections/01_introduccion.md',58,'current_R1_global','Distinguir dictamen favorable documental V3 de ausencia de acreditación de aditiva y de evaluación académica R2/R3.'),
 ('sections/03_panorama_general.md',8,'current_inventory','Actualizar bloqueB con nueve exportaciones offline adicionales y alcance608+53; no cambiar cinco activas a catorce activas.'),
 ('sections/05_configuraciones_exportacion.md',3,'current_pending_export','Cambiar nueve pendientes a nueve exportaciones verificadas offline, conservando doce precalculados del perfil operativo.'),
 ('sections/05_configuraciones_exportacion.md',90,'current_pending_export','Sustituir faltan nueve por cierre9/9 con recibos,14.312.258bytes totales y error máximo3,1492e-7 sigma7; matriz antigua del18 conserva su corte.'),
 ('sections/05_configuraciones_exportacion.md',104,'current_inventory','Actualizar evidencia de exportación y separar pendiente de despliegue/política común de la paridad ya ejecutada.'),
 ('sections/05_configuraciones_exportacion.md',115,'current_pending_export','Sustituir sus exportaciones siguen pendientes por exportaciones recuperadas; mantener FLOPs completos y costes que realmente falten.'),
 ('sections/09_dashboard_predictivo.md',193,'current_pending_export','Quitar nueve exportaciones pendientes; mantener que aceptación local histórica no valida habilitación nueva, producto público niR3.'),
 ('sections/10_despliegue_render.md',40,'current_pending_export','Conservar doce sistemas precalculados en registro vivo; explicar que nueve redes ya tienen ONNX offline verificadas y tres clásicos mantienen justificación de contrato.'),
 ('sections/11_informe_tecnico.md',16,'current_inventory','Actualizar filabenchmark con nueve ONNX nuevas y paridad661/modelo, cinco habilitadas según registro, no capacidad pública supuesta.'),
 ('sections/11_informe_tecnico.md',17,'current_R1_global','Actualizar fila revisión: dictamen sustantivo favorable acotado V3 en6.18; no transferible a aditiva; aceptación institucional distinta.'),
 ('sections/11_informe_tecnico.md',167,'current_pending_export','Mantener TSMixer histórico y añadir cierre posterior9/9; no reescribir cuantil ni recibo de TSMixer.'),
 ('sections/12_lista_entregables.md',17,'current_inventory','Actualizar producto9 con nueve representantes adicionales y V3 experimental separado; señalar revisión offline y activación no inferida.'),
 ('sections/13_sustentacion.md',22,'current_pending_export','Nueve redes exportadas offline, aún precalculadas en perfil del ensayo; nuevo ensayo sólo si se ejecuta y guarda evidencia.'),
 ('sections/14_evaluacion_calificacion.md',25,'current_R1_global','La revisión V3 ya tiene dictamen documental favorable acotado; conservar pendiente de evaluación docente y R2/R3 sin nota automática.'),
 ('sections/14_evaluacion_calificacion.md',186,'current_inventory','Actualizar casilla exportaciones: nueve cerradas técnicamente; conservar política común/costes públicos y ausencia de herenciaR3.'),
 ('sections/15_integridad_academica.md',57,'current_inventory','Añadir que cierre posterior exporta nueve sin cambiar pesos/scalers/q; paridad no representa nuevos diseños ni entrenamiento.'),
 ('sections/16_conclusiones.md',27,'current_inventory','Conservar cinco operativos y distinguir nueve adicionales offline; no sustituir operativos por catorce sin activar.'),
 ('sections/16_conclusiones.md',47,'current_pending_export','Cambiar completar nueve exportaciones por integrar si procede y verificar su contrato/capacidad; mantener humo público y SHA efectivo.'),
 ('sections/06_modelo_original.md',266,'historical_cut_keep','Conservar dictamen fechado de aditiva y su enlace; añadir remisión inequívoca a6.18V3 sólo si el lector pudiera atribuirlo globalmente.'),
 ('sections/06_modelo_original.md',325,'historical_cut_keep','Conservar estado histórico de sustitución porIA; matizar R1global como estado de aquel corte y remitir a6.18.'),
 ('sections/06_modelo_original.md',348,'historical_cut_keep','La frase novedadV3sigue sin acreditar pertenece al corte inicial; marcar pasado explícito/remisión6.18 sin alterar fuente histórica.'),
 ('sections/08_verificacion_estadistica.md',439,'historical_cut_keep','No borrar cero candidatos admitidos del corte preparatorio; ya se enlaza cierre posterior8.18.'),
 ('sections/cierre_reproducibilidad.md',17,'historical_cut_keep','No borrar cero conjuntos del corte anterior fechado; enlazar nuevo cierre y diagnóstico V3 cuando esté ejecutado.'),
]

rows=[]
for rel,line,kind,action in entries:
 text=(ROOT/rel).read_text().splitlines()
 rows.append({'file':str(Path('book_e3')/rel),'line_at_review':line,'classification':kind,
    'excerpt':text[line-1], 'suggested_action':action})
with (OUT/'book_update_inventory.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'status':'read_only_editorial_inventory','book_root':str(ROOT),'staging_edited':False,
 'current_rows':sum(r['classification']!='historical_cut_keep' for r in rows),
 'historical_rows_to_preserve':sum(r['classification']=='historical_cut_keep' for r in rows),
 'important_distinction':'14 historical ONNX representatives verified overall does not mean14models enabled or accepted on Render; V3experimental is a separate15th export.',
 'diagnostic_followup':'Once zero-shot finishes, current summaries saying V3has no D01D02D03forecasts require a new retrospective-diagnostic paragraph; do not edit old dated statements or call this independent confirmation.',
 'public_status':'Render/DOI statements await current workflow results; transport failure before origin is not service404/DOIunpublished.'}
(OUT/'book_update_inventory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
