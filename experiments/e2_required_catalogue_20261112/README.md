# Catálogo obligatorio E2: 16 sistemas

Enmienda de alcance solicitada por el usuario el 12/11/2026, tras contrastar la
guía (§3.2, p. 11; F2a, p. 30). Su fecha no se presenta como un preregistro
anterior a los resultados ya obtenidos.

Se conservan los diez candidatos del Top 10, MLP16/LSTM32/GRU32 de E1 y
Persistence/SeasonalNaive/ARIMA. El híbrido adicional y Ridge quedan fuera del
trabajo obligatorio; sus implementaciones y resultados históricos se conservan.
Las cuatro familias funcionales se cuentan dentro del Top 10.

`catalogue.json` es el manifiesto vigente. Se puede verificar sin leer mercado:

```bash
python scripts/e2_required_catalogue.py
```

La API `run_required_neural_selection` rechaza el híbrido antes del entrenamiento.
`select_required_classical` omite los 18 ajustes Ridge y `refit_required_classical`
omite su reajuste. Las APIs históricas mantienen su comportamiento previo.
La implementación prospectiva está en `scripts/e2_required_classical_training.py`;
`scripts/e2_classical_training.py` conserva exactamente sus bytes históricos,
vinculados a los recibos publicados y al registro CUDA.

Esta enmienda cierra **qué sistemas son obligatorios**, no los experimentos ni
los demás requisitos. Mantiene recetas, HPO, NAS aplicable, orígenes, semillas,
épocas y el presupuesto autorizado. La selección y evaluación final siguen
pendientes de cierre.

La campaña sellada `campaigns/e2-cuda-crop-11-v2` continúa definida con 14 redes.
No se altera su plan, su caché o el proceso privado de Colab con esta publicación.
Antes de lanzar una campaña con 13 redes se necesita una migración versionada
de controlador, resultados y saldo. No ejecutar un segundo controlador ni usar
el ejecutor histórico `run_e2_final_benchmark.py` (catálogo antiguo de 20 sistemas)
como si ya evaluara estos 16.
