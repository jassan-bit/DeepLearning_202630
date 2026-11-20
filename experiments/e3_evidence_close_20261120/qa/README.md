# Revisión independiente por IA del cierre de evidencia

Revisión de lectura y aritmética, con asistencia de IA declarada. No es una revisión humana, una defensa del estudiante ni acreditación R1/R3. No ejecutó entrenamientos del pronosticador, inferencias de mercado o remuestreos. La revisión se refiere al código y a los once documentos del libro anteriores al commit que añadirá el ZIP final y su enlace; la prueba posterior del ZIP y la verificación web corresponden a sus recibos separados.

## Resultado

No quedan incidencias científicas bloqueantes en el material examinado. Se comprobó de forma independiente:

- Piloto: 19 objetos Git/manifestados; cinco hashes de resultados; plan canónico; 17 trayectorias, 935 filas y 55 fechas. La media de las RMSE individuales conserva el orden: `atom_crps` ocupa el puesto5/7, con2,7043% más error que `atom_mse_only` y0,7419% más que el piso más exceso de ajuste. No se modifica la regla ni se promociona la ablación ganadora.
- Inventario: 21 registros fuente con bytes/SHA256/blobGit; 15 recetas neuronales implementadas,12 con ajuste y3 sólo software. Los22 registros documentados,183 ensayos HPO y844 trabajos conservan unidades distintas. La ambigüedad22–23 no limita las ideas informales no archivadas ni certifica todas las oportunidades de selección.
- Adecuación: 13 entradas y seis tablas derivadas; cuatro extractos de prueba limpia,40 trayectorias y18.080 filas. Se recalcularon80 métricas RMSE/MAE y los ajustes Holm40/Holm160 por separado. El antecedente608×10 no se mezcla con los pesos nuevos o V3. Las hipótesis diagnósticas y sus supuestos no se convierten en R3, potencia prospectiva o aprobación formal de adecuación.
- Recursos: 900 fases corresponden a796 trabajos; tres registros conservan cómputo de caída desconocido. El exceso máximo35.283,63s incluye desconexiones y no se interpreta como horasGPU activas.
- Producto: cinco archivos editoriales; se cotejaron21 callbacks/helpers con AST idéntico y27 archivos de datos/artefactos con hashes iguales. Las547 descargas estáticas anteriores del libro mantienen sus bytes.
- Handoff: el fallo403 del proxy no se atribuye al origen. No se finge aceptación Render ni publicación DOI. La caché parcial del cliente CPU se conserva antes del reintento;15 pruebas sintéticas de archivo/handoff pasaron. Las entradas comprimidas permitidas son textuales, acotadas y verificadas; el ZIP sigue siendo un suplemento limitado.

Los valores p nuevos son200 diagnósticos retrospectivos sobre50 trayectorias guardadas, no cero nuevas evaluaciones estadísticas ni pruebas confirmatorias de superioridad. Mincer–Zarnowitz y ARCH usan regresiones auxiliares; no reentrenan el modelo de pronóstico.

## Repetir la comprobación portable

Desde la raíz del repositorio:

```bash
python experiments/e3_evidence_close_20261120/qa/verify_closure_evidence.py \
  --repo-root . --book-root book_e3
```

El programa utiliza biblioteca estándar. El parámetro `--root` permite situar el directorio de evidencias en otro lugar. El cotejo adicional con callbacks del código anterior requiere `--baseline-source` apuntando a una copia sin modificar de la base29f9a26; no debe señalar a la publicación recién editada. `--book-base` permite repetir el cotejo completo de las descargas antiguas. Si no se proporcionan esos inputs, el recibo lo indica y no afirma haber repetido esas comparaciones.

`verification.json` registra la revisión con las bases locales disponibles; `publication_layout_verification.json` demuestra que el mismo checker funciona con las rutas públicas ensambladas sin aquellos cotejos opcionales. Los recibos incluyen hashes de los once documentos revisados; no anticipan los bytes del ZIP pendiente de construir ni una versión servida que aún no se haya comprobado.

## Límites preservados

R1/prioridad, defensaR2, confirmaciónR3, derechos/licencia, publicación del depósito y reproducción independiente conservan sus requisitos. Las referencias con acceso parcial mantienen esa condición; no se anuncian nuevas lecturas. El código fuente, las reglas y la evidencia histórica no se reescriben para favorecer el resultado.
