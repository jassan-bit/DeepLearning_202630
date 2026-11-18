# Preparación verificable de una futura evaluación E3

Estado al 18 de noviembre de 2026: **dimensionamiento y auditoría ejecutados; ninguna campaña confirmatoria nueva**. Hay cero ajustes de modelos, cero inferencias nuevas, cero recalibraciones, cero contrastes nuevos y cero conjuntos adicionales admitidos. No se acredita R3 ni se modifica el protocolo histórico. Los resultados anteriores siguen siendo retrospectivos.

## Qué se calculó

`power_planning.py` verifica los once insumos inmutables de la auditoría estadística publicada y recupera las diez coordenadas de inicialización 271801–271810 sobre las 608 fechas ya conocidas. Calcula la desviación estándar del **gain pareado en RMSE**, rival menos candidato. Los tres comparadores deterministas conservan una trayectoria real cada uno: sus diez gains reflejan únicamente variación del candidato, no diez reentrenamientos del rival.

La familia contiene 16 rivales y tres procedimientos por rival: 48 decisiones. Se usa el umbral conservador unilateral α/48 = 0.05/48 como referencia de cada prueba de Holm. No se calcula la potencia exacta del procedimiento adaptativo de Holm.

La hipótesis nula es (g\leq\delta\), con margen heredado BTC (\delta=0.00025772842664981106\). Para la planificación se declaran alternativas (g_1=1.25\delta,1.5\delta,2\delta\): la separación es **(g_1-\delta\)**. La media observada del gain no se usa como alternativa. Esos valores ilustran decisiones futuras; no representan efectos garantizados ni márgenes ya justificados para otros datos.

El componente por semillas se dimensiona con la distribución t no central, suponiendo gains pareados normales e independientes y una σ futura igual a la asumida. Se invierte también esa distribución para obtener la diferencia mínima detectable sobre la frontera nula con R=10. Los CSV conservan cada rival y escenario; no se seleccionan los que favorecen al candidato.

## Resultado condicional y sensibilidad

Para la alternativa declarada (g_1=2\delta\), el máximo de R sobre los 16 rivales para **el componente t por semillas** es:

| Objetivo marginal de cada procedimiento | σ histórica | Cota superior simultánea de σ | Sensibilidad 2σ |
|---|---:|---:|---:|
| 0.90 | 280 | 1614 | 1103 |
| 0.9666667 | 354 | 2048 | 1399 |
| 0.9979167 | 515 | 2996 | 2046 |

TiDE determina esos máximos. La cota superior de σ usa la distribución χ² bajo normalidad, con cola 0.05/16: multiplica cada σ histórica por 2.4210. Con sólo diez pares su incertidumbre es material. Las cifras son referencias condicionales, **no una recomendación de ejecutar cientos de ajustes**, ni el tamaño definitivo de una campaña.

Con R=10 y σ histórica, la separación mínima detectable para potencia marginal 0.90 va desde 0.3452δ (SeasonalNaive) hasta 7.1759δ (TiDE). Es separación **adicional a δ**: los gains alternativos mínimos correspondientes son 1.3452δ y 8.1759δ. No se puede asignar una sola potencia al hecho de tener diez semillas.

El objetivo marginal 0.9666667 daría una cota conjunta 0.90 para tres decisiones de un rival si las tres alcanzaran esa potencia; 0.9979167 daría una cota conjunta 0.90 para las 48 decisiones si todas la alcanzaran. Es la desigualdad de la unión y no requiere independencia. Aquí sólo se dimensiona el componente t y una referencia temporal asintótica: **la potencia conjunta real de la batería de tres procedimientos sigue sin establecerse**. Los productos 0.9^16 y 0.9^48 se guardan exclusivamente como ejemplos bajo independencia hipotética. La correlación descriptiva de gains históricos (−0.5993 a 1.0) no identifica la dependencia futura de los p-valores.

Las fechas se tratan por separado. Se estima una escala temporal con la influencia delta-method del RMSE y HAC Bartlett con rezagos 7, 14 y 28. Una aproximación normal proporciona referencias de sensibilidad de longitud temporal. Para HAC14, (g_1=2\delta\) y objetivo marginal 0.90, la peor referencia sobre la familia asciende a 420677 fechas, determinada por SeasonalNaive. **Esta extrapolación extrema no constituye un horizonte recomendado ni demuestra cuánto necesitaría realmente un nuevo estudio**: impone el mismo gain cercano a la frontera a rivales muy distintos y prolonga estacionariedad mucho más allá de las 608 fechas observadas. Señala que se necesitan alternativas y varianzas justificadas en un piloto nuevo antes de congelar un diseño viable. Aumentar semillas no crea nuevos periodos de mercado independientes.

## Entrenamiento común: auditoría, no equivalencia acreditada

`campaign_readiness.py` verifica los planes históricos y analiza sintácticamente once puntos de entrada, sin importar Torch ni ejecutar recetas. Produce una fila por cada uno de los 17 sistemas, con receta, parámetros, épocas e identidad donde constan. Esta inspección no demuestra que todos se puedan entrenar en el entorno actual: faltan metadatos y árboles de autor en este checkout parcial, cuya ausencia local no equivale a ausencia en GitHub.

Los planes muestran tratamientos distintos:

| Tratamiento histórico | Rivales E2 | Candidato aditivo E3 |
|---|---|---|
| Ajuste supervisado | 655 ejemplos, hasta 2024-10-31 | 532 ejemplos, hasta 2024-06-30 |
| Selección interna | recetas/épocas previamente seleccionadas por arquitectura | 116 ejemplos de validación; parada y selección de época |
| Calibración | 53 ejemplos separados | no permitida en el plan |
| Presupuesto | extensión separada de 7200 segundos para 52 ajustes | límite global de 1200 segundos con cargos previos conservados |
| Evaluación histórica | mismas 608 fechas conocidas | mismas 608 fechas conocidas |

Ambos resúmenes históricos declaran `equal_training_budget_claimed=false`. Igualar el número de semillas no iguala datos, selección, calibración ni cómputo. La reproducción CPU de recetas fijas existente contempla 78 ajustes neuronales y tres reconstrucciones clásicas bajo su propio control de recursos; no repite el HPO histórico ni incluye automáticamente un entrenamiento común del candidato E3. No se ejecuta aquí.

También hay una distinción necesaria: las 143 trayectorias congeladas incluyen el **candidato aditivo de 1210 parámetros**, construido por `InnovationModel('additive_positive_head')`. `src/original_model.py:OriginalModel` implementa otro candidato de desarrollo con `BoundedDiscordanceAttention`. Sus formulaciones, propiedades y evidencia de novedad no se transfieren por compartir el nombre E3. El siguiente protocolo debe fijar la identidad del candidato que evaluará.

Antes de nuevas salidas confirmatorias hacen falta decisiones concretas: candidato; tres datos técnicamente admitidos y sus márgenes; particiones y escaladores comunes; política de calibración; recetas fijas o nueva búsqueda con presupuesto comparable; tratamiento causal de ARIMA y naïve; semillas y procedimientos; potencia individual y conjunta; estudios de robustez; y recibos de tiempo de cada arquitectura sobre el hardware elegido. Los trabajos fallidos deben consumir el mismo registro de presupuesto. No se reajustan comparadores ni se cambian familia, semillas o estándar tras ver resultados.

El inventario [dataset_admission.md](dataset_admission.md) y [dataset_candidates.json](dataset_candidates.json) describe tres candidatos por metadatos y sus bloqueos de licencia, calendario y verificación. **Tres candidatos no significan tres conjuntos admitidos o evaluados.**

## Reproducir

Desde la raíz del repositorio, con Python 3.12, NumPy 2.3.5 y SciPy 1.17.0:

```bash
python experiments/e3_remediacion_20261118/evaluation/test_power_planning.py
python experiments/e3_remediacion_20261118/evaluation/power_planning.py \
  --download --data /tmp/e3_planning_inputs \
  --output /tmp/e3_planning_results
python experiments/e3_remediacion_20261118/evaluation/campaign_readiness.py \
  --data /tmp/e3_planning_inputs \
  --output /tmp/e3_planning_readiness.json
```

`--download` usa sólo los once insumos históricos publicados, atados a commit, tamaño y SHA-256 por el cargador congelado. No descarga datos candidatos ni pesos para entrenar. Sin `--download`, se puede reutilizar un directorio ya verificado. Si una salida existente contiene otros bytes, los ejecutores exigen un destino nuevo.

Las nueve pruebas verifican el tamaño nominal en la frontera nula, inversión t/MDE, minimalidad del entero R, invariancia de unidades, ajuste por multiplicidad, incertidumbre de σ, separación de semillas y deterministas, alineación de identidades y que desplazar el gain observado sin cambiar su varianza no cambie el diseño. No son pruebas del desempeño ni de R3.

Los resultados detallados están en `results/seed_planning.csv` y `results/calendar_planning.csv` (432 escenarios cada uno), `results/seed_variance_inputs.csv`, `results/power_planning.json` y `results/campaign_readiness.json`. El primer manifiesto `results/manifest.json` verifica las cuatro tablas/productos de dimensionamiento; cada JSON registra sus fuentes y código. No se han sustituido artefactos históricos.
