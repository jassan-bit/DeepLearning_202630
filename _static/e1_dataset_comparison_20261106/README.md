# Comparabilidad del dataset y margen académico

Síntesis documental elaborada por Codex el 06/11/2026 a partir de las matrices de evidencia existentes, sin nuevas búsquedas, entrenamientos, predicciones o réplicas externas.

- `comparison.csv`: contrato E1, siete fuentes históricas y dos fuentes nuevas en estratos separados. Cada fila conserva su archivo y localización de evidencia. Los campos desconocidos permanecen explícitos. No se exige el mismo periodo o versión exactos y no se ordenan RMSE de distintas tareas.
- `margin_sensitivity.csv`: cuatro conversiones algebraicas de 1/2,5/5/10 % del mismo RMSE GRU-CV original. La fila del 5 % coincide exactamente con `design.json/statistics/delta`. No son pruebas nuevas ni un nuevo dimensionamiento.
- `summary.json`: reglas, límites y SHA-256 de las fuentes usadas.
- `manifest.json`: integridad de estos derivados; no demuestra reproducción de originales ni cronología independiente.

El 5 % es una convención académica de planificación previa a fase II y posterior a fase I; no acredita beneficio financiero. Para una aplicación futura deberán fijarse acción, umbrales y costos, y medirse utilidad/calibración e incertidumbre en una cohorte nueva.
