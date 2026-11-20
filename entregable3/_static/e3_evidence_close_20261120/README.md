# Cierre de evidencia existente de E3

Consolidación retrospectiva con asistencia de IA declarada. Base documental:
`29f9a26f1fd05484c6581ea6538ad7cb5a4781cb`. Se conservan el master de 796
trabajos `20261119T031104515520Z` y el piloto de 30 ajustes
`20261120T031424176339Z`, commit `8eaa581bc34b615a8507c9e384a3e4be29084720`.
Esta consolidación no entrena ni ejecuta nuevos pronósticos, recalibraciones,
remuestreos o contrastes confirmatorios de superioridad.

## Evidencia incorporada

- `pilot/`: 19 archivos originales, 17 trayectorias/935 filas, siete recetas y
  la heurística previa incumplida. La V3 completa tuvo 2,70 % más RMSE media
  que el mejor control; no se cambia de métrica ni se promueve la ablación ganadora.
- `inventory/`: 22 identidades del corpus, 15 recetas neuronales implementadas,
  cuatro controles y tres propuestas. Las 183 oportunidades HPO y 844 registros
  de trabajo se separan de los diseños. Hay un bosquejo conceptual ambiguo;
  el intervalo 22–23 no limita ideas sin archivo ni corrige el sesgo de selección.
- `contribution/`: procedencia, matrices y declaración de contribución delimitada.
  Mantiene las lecturas previas con sus límites y no acredita R1/R3.
- `adequacy/`: doce tablas originales, cuatro paneles verificados y sus extractos
  de prueba limpia; 40 trayectorias/18.080 filas de la auditoría ampliada.
  Se añaden 200 evaluaciones diagnósticas retrospectivas en dos familias
  distintas: Holm160 de esa auditoría y Holm40 de los pesos históricos sobre
  608 fechas. Son regresiones auxiliares sobre predicciones guardadas, no
  entrenamiento del pronosticador, potencia prospectiva ni pruebas de R3.
- `dashboard/`: cinco cambios editoriales con 21 callbacks/helpers y 27 archivos
  de datos/artefactos intactos. El servicio conserva su catálogo histórico.
- `release/`: comprobaciones de acceso, cliente CPU y preparación de un suplemento
  documental. Los fallos de proxy no acreditan caída de Render o ausencia de DOI.

## Repetir las comprobaciones pequeñas

Desde la raíz del repositorio, con Python 3.12 y los paquetes fijados por el
workflow `verify-e3-existing-evidence-close.yml`:

```bash
python experiments/e3_evidence_close_20261120/verify_closure.py
python experiments/e3_evidence_close_20261120/pilot/analyze_saved_pilot.py --verify-only
python experiments/e3_evidence_close_20261120/inventory/verify_inventory.py
python experiments/e3_evidence_close_20261120/contribution/verify.py --upstream book_e3/_static/e3_r1_close_20261120
python experiments/e3_evidence_close_20261120/adequacy/audit_existing_predictions.py --verify-only --source-dir book_e3/_static/e3_audit_completed_20261119T234320461006Z
python experiments/e3_evidence_close_20261120/adequacy/verify_diagnostic_reference.py
python -m unittest discover -s experiments/e3_evidence_close_20261120/release -p 'test_*.py' -v
```

Los verificadores de integridad y el inventario sólo usan biblioteca estándar;
el piloto usa NumPy. Adecuación usa NumPy/SciPy y su contraste independiente
statsmodels/pandas/patsy. No requieren Torch ni GPU. La creación original de
la figura del piloto usa Matplotlib, no requerido por `--verify-only`.

El ZIP documental permite repetir estos análisis pequeños con sus entradas
de texto congeladas. No contiene todas las imágenes/activos del libro, todos
los datos o los pesos. Para comprobarlo en ese suplemento se usa
`verify_closure.py --source-only --allow-missing-protected`; el resultado
declara cuántos archivos históricos no están incluidos y no certifica
preservación del repositorio entero. La compilación del libro completo y
los checks de todos sus archivos corresponden al repositorio publicado.

## Pendientes que la evidencia actual no reconstruye

`AUTHOR_ACTIONS.md` fija las decisiones concretas. La revisión por IA sustituye
al revisor externo de R1 conforme a la indicación docente comunicada por el
autor; no representa una defensa del estudiante o una reproducción humana
observada. Se mantienen separados resultados de desarrollo, diagnóstico
retrospectivo, confirmación y funcionamiento público.
