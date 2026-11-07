# Contexto académico del margen fijo de E1

El autor confirmó que el uso es la comparación académica de modelos. Este suplemento
explica el margen ya fijado con agregados y etiquetas públicas de desarrollo 2023–2024.
Es una fundamentación posterior: no modifica δ, R, las recetas, el baseline o los
resultados de los estudios cerrados y no utiliza observaciones de reserva/test.

`margin_context.csv` normaliza 2,5/5/10% respecto del ancla original GRU/D7-CV, su
variabilidad de inicialización y la dispersión observada de σ7 en 2024. Esa dispersión
no estima ruido de medición ni error irreducible. `paired_development.csv` contextualiza
diferencias de las recetas iniciales sobre las mismas 366 fechas; sus IC son marginales
y condicionados a semillas, sin pruebas nuevas ni certificación de estabilidad temporal.

La elección del 5% es normativa y académica: exige una reducción no trivial del error
para considerar una mayor complejidad como avance, entre escenarios menor y mayor.
Las razones y escalas permiten deliberar esa decisión, pero no prueban un 5% óptimo.
Los costes son los ya publicados para las recetas iniciales, bajo concurrencia CPU;
no se convierten en precios ni se usan para ajustar un umbral al resultado conocido.

La fuente metodológica consultada es Lakens (2022), *Sample Size Justification*,
https://doi.org/10.1525/collabra.33267, apartados sobre efecto de interés y detectabilidad.
Distingue interés sustantivo y detección; no prescribe el 5% para este problema.

Reproducción desde la raíz, con `requirements-e1-supplement.lock.txt`:

```bash
python scripts/justify_e1_academic_margin.py
```

Las entradas admitidas se enumeran de forma cerrada en el script y sus hashes constan
en `source_manifest.json`. No hay entrenamiento, lectura de archivos de mercado
reservados, nuevas predicciones, búsqueda de umbrales ni pruebas de hipótesis.
