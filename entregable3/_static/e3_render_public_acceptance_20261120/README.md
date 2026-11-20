# Aceptación pública de Render ejecutada

El 20 de noviembre de 2026, de 14:22:40 a 14:23:35 de Bogotá, aprobaron los clientes originales de cinco y catorce predictores sobre `https://deeplearning-e3-dashboard.onrender.com`. La fuente efectiva del servicio es `feb83dbc08197bde7076857bd8ff475de197efd2`, informada por `RENDER_GIT_COMMIT`; el perfil suplementario contiene catorce redes históricas y V3 está deshabilitada.

La ejecución real es Actions [44801411372](https://github.com/jassan-bit/DeepLearning_202630/actions/runs/44801411372), job `136514212866`, launcher `63b2220ead51e5f755c421a0253644390c0a94e6`. Los clientes se tomaron sin modificaciones del checkout inmutable del servicio. El SHA del launcher y el futuro SHA de publicación del libro son distintos del SHA de Render.

## Resultado y alcance

- Cliente original: tres páginas y cinco predictores, huellas de recursos/modelos/metadatos, intervalos y explicaciones.
- Cliente ampliado: catorce predictores por API y navegador, catorce casos OOD, catorce guardas temporales y diez entradas inválidas. Cero errores de navegador o consola.
- Tolerancia: 1e-6 sigma7. Error máximo API de catorce: 1.5144635804112405e-8; diferencia máxima de la salida visual de seis decimales de cinco: 4.393076896667958e-7. Son comparaciones con el caso histórico fijado, no nuevos resultados de confirmación.
- Cuantiles E2 originales y calibrador posterior E3 conservados. No se habilita V3 ni se le asigna el calibrador de la aditiva.
- Cero ajustes de datos reales, recalibraciones, reexportaciones, GPU o fechas científicas nuevas. Se conservan los 796 trabajos de la aditiva y los treinta ajustes del piloto V3, incluido su resultado adverso.

La aceptación acredita este ensayo funcional de la versión indicada. No acredita R3, reproducción humana, memoria/CPU/energía de Render, concurrencia remota o disponibilidad sostenida. Las mediciones de 198,5 MiB pertenecen al contenedor local anterior, no al servidor público.

## Historial conservado

`initial_attempt/` conserva las observaciones previas. El acceso directo del entorno falló en el proxy antes de llegar al origen. El primer wrapper remoto observó SHA H y perfil14, pero detuvo su cliente por cuota de la API GitHub sin autenticar. Se verificó entonces el repositorio mediante checkout Git y archivos fijados, sin relajar las guardas.

El ensayo `44800163328` aprobó el cliente14; el cliente5 navegó tres páginas y se detuvo, antes de sus predicciones, por la ausencia de una referencia CSS en su checkout local. La corrección añadió `assets` y comprobó sus tres recursos estáticos antes del navegador. Se preservan el fallo, los recibos parciales y la aprobación posterior; no se atribuye aquella dependencia local ausente a un fallo de Render.

## Procedencia de los archivos

`accepted_attempt/` contiene los JSON recuperados semánticamente de logs públicos y seis PNG/JPEG recuperados de fragmentos base64. Cada imagen coincide con el tamaño y SHA256 original; no se redimensionó ni recodificó. Los hashes originales de los JSON/ZIP de Actions se conservan en el resumen, sin afirmar que su nueva serialización JSON tenga los mismos bytes.

`public_acceptance_summary.json` reúne fuente, run/job, recuentos, tolerancias y límites. `requirements_matrix_current.csv/json` proyecta diecisiete requisitos a seis campos y actualiza el cierre de producto. Las quince filas heredadas se interpretan respecto del expediente H `experiments/e3_followup_20261120/`; no respecto de esta carpeta. DOI, derechos/licencia, defensa y reproducción humana mantienen sus estados. El NO-GO confirmatorio de V3 y R3 no acreditada no cambian.

Los paquetes y manifiestos anteriores no se reescriben. En particular, el manifiesto de 203 entradas del seguimiento anterior incluye el Markdown de su corte: se verifica en el checkout `feb83dbc08197bde7076857bd8ff475de197efd2`. Los cambios editoriales posteriores se verifican con el manifiesto y el build de este nuevo corte. La publicación del libro no ordena otro Manual Deploy de Render.

La asistencia de IA en ingeniería, revisión y auditoría automática está declarada. No se presenta como firma de un tercero humano.

```bash
python experiments/e3_render_public_acceptance_20261120/verify_packet.py --root .
```
