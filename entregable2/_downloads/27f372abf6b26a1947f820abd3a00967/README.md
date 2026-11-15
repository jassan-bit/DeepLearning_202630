# Evidencias vigentes de implementación y experimentación

Este expediente acompaña los apartados 4.1–4.6 de E2. Consolida el protocolo ejecutado, sus registros y los análisis complementarios; conserva las recetas, los resultados científicos y los cargos anteriores.

## Contenido

| Archivo | Alcance |
|---|---|
| `guide_checklist.json` y `.csv` | 68 requisitos/subcondiciones, con evidencia, estado y limitaciones. Una fila documentada no equivale a cumplimiento incondicional. |
| `decision_traceability.csv` | Decisiones y verificaciones derivadas de cada requisito, con sus riesgos. |
| `current_evidence_index.json` | Índice de 28 fuentes con tamaños y SHA-256, cifras vigentes y límites. |
| `prediction_trajectories.png` y `.svg` | Objetivo observado frente a las 81 trayectorias individuales guardadas: 16 paneles y 608 fechas completas, en escala original de σ₇. |
| `prediction_trajectories.json` | Procedencia, versiones, identidades, comprobaciones y hashes del gráfico. |
| `04_implementacion_before_update.md` | Copia exacta del capítulo anterior; sus prototipos y planes se conservan como antecedentes. |
| `manifest.json` | Tamaños y hashes del expediente y del script de la figura. |

La selección reducida registra 234 oportunidades: 231 resultados válidos, dos fallos conocidos y un desenlace desconocido. Las trece recetas ganadoras tienen sus tres folds. El cotejo no reconstruye un desglose exhaustivo caché/nuevo ni interpreta candidatos previstos de caché como resultados automáticamente admitidos. La reutilización no es entrenamiento nuevo.

El calendario final tiene 655 ejemplos de entrenamiento, 53 de calibración y 608 de evaluación heredada de E1. Trece redes × seis semillas y tres comparadores deterministas forman 81 predictores de 16 sistemas. La A2 usa la media de las RMSE individuales de cada red; no evalúa un ensamble. Los 81 ajustes de reproducción y 18 ajustes de tres ablaciones pertenecen al plan complementario. Las ablaciones tienen tres contrastes, con 10.000 remuestreos pareados y Holm3.

Las oportunidades HPO comunes no acreditan igual cómputo acumulado: se declaran caché, transición T4/A100 y las tres fases SAE. R=6 no alcanzó la potencia objetivo en el piloto. La adopción previa de δg está documentada, pero la prueba formal correspondiente para los predictores actuales sigue pendiente. La descomposición vigente de componentes de varianza no se sustituye por resultados históricos del P0.

El cierre complementario conserva un límite de 84 horas totales y sus cargos. Los tiempos contables, los tiempos instrumentados de entrenamiento, la inferencia residente en A100, la memoria y los FLOPs de núcleo son magnitudes distintas. La energía permanece sin medir; su estimación es opcional en la guía.

## Reproducir la figura guardada

Desde un checkout completo, usar Python 3.12, NumPy 2.3.5 y Matplotlib 3.10.8, versiones registradas en la procedencia:

```bash
python scripts/plot_e2_saved_predictions.py --output /tmp/e2-predicciones-guardadas
```

Elegir una ruta de salida nueva. El script también admite salidas idénticas y rechaza sobrescribir archivos diferentes. Comprueba los hashes de caché, congelación y protocolo; los 81 IDs/checkpoints; las seis semillas por red y el único predictor de cada clásico; las 608 fechas, máscara y valores finitos. Los PNG/SVG/JSON se reprodujeron byte por byte en el entorno registrado.

El gráfico lee exclusivamente los vectores originales guardados. Muestra todas las semillas, sin promediarlas, seleccionar una semilla, suavizar o recortar una ventana favorable. No calcula métricas, pruebas o puntuaciones nuevas, ni ejecuta modelos o abre ZIP brutos. Es visualización descriptiva posterior; no recupera cegamiento ni establece ajuste, causalidad o ausencia de sobreajuste por inspección visual.

## Verificar el cierre original

```bash
python scripts/audit_e2_delivery.py --require-supplements --output /tmp/e2-entrega-verificacion.json
```

El auditor comprueba los resultados y manifiestos guardados del benchmark; la figura tiene su procedencia separada. Los enlaces del capítulo permiten acceder a curvas, recursos, robustez, explicaciones y ablaciones. Las fuentes conservan sus fechas y alcances originales; los derivados científicos anteriores no se reescriben.
