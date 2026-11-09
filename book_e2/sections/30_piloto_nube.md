# 30. Piloto de desarrollo en la nube de Codex

**Resultado:** las 14 referencias completaron el reconocimiento. El extremo TimesNet se censuró por tiempo. El piloto principal de tres modelos quedó incompleto; el análisis secundario usa las cinco semillas completas del híbrido y GRU32. k=40 es conservador y R=30 es un techo provisional de coste; el R final permanece abierto.

Este corte incorpora ajustes reales sobre los derivados diarios de desarrollo de Bitcoin. Se ejecutaron en el servidor remoto de **Codex Cloud**, con cuatro vCPU de cuota, 16 GiB de RAM y sin GPU visible. El equipo del usuario no ejecuta estos entrenamientos.

## Registro previo y alcance

El [registro del piloto](https://github.com/jassan-bit/DeepLearning_202630/blob/cf68db00f4d7c3626bdb517bf922265a3b229ea4/experiments/e2_development_pilot_20261109/registration.json) se publicó antes de los ajustes. Su SHA256 es `cc10ddba82f0c59cfcd8caf70e713481ec0701e71860862ed13ead0df0bf4797`.

La [enmienda de análisis secundario](https://github.com/jassan-bit/DeepLearning_202630/blob/d511a8de3b9e9d5fa559d761ce9f2ac732921a80/experiments/e2_development_pilot_20261109/analysis_amendment.json) se publicó antes de cualquier predicción del periodo retenido. Permite analizar dos modelos completamente observados si el tercero no termina, conservando el resultado principal incompleto y todos sus costes.

Hay dos trabajos distintos:

1. **Reconocimiento de recursos:** los once candidatos y tres referencias neuronales E1, en sus arquitecturas de referencia, más el extremo TimesNet de 225.075.755 parámetros. Receta fija de §27: AdamW, LR 0,001, WD 0,0001, batch 32, warmup 0, paciencia 20 y hasta 200 épocas por etapa. Los intentos son secuenciales y usan la semilla 271601. El autoencoder conserva sus dos etapas auxiliares y su etapa supervisada.
2. **Variación antes de seleccionar recetas:** TimeMixer, híbrido y GRU32, cinco semillas nuevas 271901…271905 por modelo, arquitecturas de referencia y receta fija. No se eligen hiperparámetros ni arquitecturas a partir de estos resultados.

El reconocimiento limita cada etapa a 150 segundos comprobados en fronteras y cada proceso completo a 180 segundos mediante supervisor. El supervisor muestrea RSS y termina procesos que superen 12 GiB; ese muestreo no constituye una barrera de asignación del kernel. Estos límites operativos se registraron como una enmienda para medir viabilidad: **no sustituyen** las 7200/1800 segundos de los presupuestos completo/corto del diseño científico. Todo fallo o censura conserva el coste consumido y no recibe una semilla o intento de reemplazo. Los pasos desconocidos por terminación externa permanecen `null`.

## Datos, calendario y objetivo

Se admiten únicamente `daily_market_development.csv` y `daily_volatility_development.csv`, derivados de E1 para 2023–2024 y fijados por hash. El objetivo es **σ₇ poblacional de siete retornos logarítmicos diarios, ddof=0, sin anualización**. No es volatilidad realizada intradía. La reconstrucción independiente desde los cierres coincide con el derivado.

La única barra OHLCV incompleta es 24/03/2023. Se mantienen sus posiciones en el calendario y los objetivos afectados; no se interpolan ni se comprimen los días. El cargador hashea los archivos completos y comprueba sus fechas, pero convierte valores numéricos sólo dentro de la admisión de cada fase.

| Trabajo | Origen | Entrenamiento/selección | Validación | Reajuste/evaluación |
|---|---|---|---|---|
| Recursos | 01/10/2023 | 161 ventanas, hasta 25/06/2023 | 83 fechas, 03/07…23/09/2023 | Sin evaluación externa; las fechas posteriores se representan como no disponibles |
| Selección de épocas para variación |01/10/2024 | 527 ventanas, hasta 25/06/2024 | 83 fechas, 03/07…23/09/2024 | Mediana superior de las mejores épocas de las cinco semillas por modelo |
| Reajuste fresco del piloto |08/11/2024 | 655 ventanas, hasta 31/10/2024 | Sin parada con etiquetas de evaluación | 53 fechas de desarrollo, 08/11…30/12/2024 |

Los reajustes reinician modelos, optimizadores y escaladores calculados sólo con entrenamiento. Las predicciones se registran antes de observar cada resultado; las huellas de pesos y escaladores permiten comprobar que permanecen congelados. Se guardan checkpoints de inferencia con carga segura, sin presentarlos como reanudación de optimizadores.

La auditoría de trazabilidad verificó los derivados completos de desarrollo 2023–2024. Este trabajo no convierte esos datos históricamente utilizados en una muestra confirmatoria intacta. No se abren archivos de mercado 2025–2026 ni se evalúan las 608 fechas externas conocidas.

## Resultados observados y costes

En total hubo **40 intentos: 34 completos, cinco fallos por límite interno de tiempo y una censura externa**. Cinco reajustes de TimeMixer quedaron bloqueados sin intentarse; no se sustituyeron semillas. Los recibos contienen **50.009 pasos de optimización conocidos**, incluidos 3.645 de los cinco intentos fallidos. Los pasos del extremo TimesNet permanecen desconocidos, por lo que 50.009 no es un total exacto del trabajo ejecutado.

El reconocimiento suma 586,745 segundos y el piloto de variación 1323,587 segundos: **31,84 minutos de procesos medidos**. Incluyen carga, verificaciones, construcción, ajuste y, cuando corresponde, replay; no son tiempo neto de AdamW ni duración total de esta sesión.

| Red / caso | Estado | Segundos del proceso | Pico RSS muestreado (GiB) | Pasos conocidos |
|---|---|---:|---:|---:|
| FCN-1D | Completo | 10.074 | 0.350 | 180 |
| ResNet-1D | Completo | 65.662 | 0.363 | 552 |
| Inception-1D | Completo | 29.869 | 0.374 | 186 |
| LSTM-autoencoder con predictor | Completo | 47.905 | 0.354 | 2490 |
| TimesNet | Completo | 81.120 | 0.736 | 132 |
| TSMixer | Completo | 35.966 | 0.350 | 552 |
| TiDE | Completo | 15.676 | 0.330 | 696 |
| TimeMixer | Completo | 57.056 | 0.377 | 414 |
| PatchTST | Completo | 9.998 | 0.353 | 126 |
| iTransformer | Completo | 25.483 | 0.368 | 330 |
| CNN-1D + LSTM + atención temporal | Completo | 6.944 | 0.352 | 210 |
| MLP16 | Completo | 5.192 | 0.326 | 252 |
| LSTM32 | Completo | 6.387 | 0.340 | 270 |
| GRU32 | Completo | 8.928 | 0.328 | 684 |
| TimesNet (extremo256/256/256) | Censura externa | 180.485 | 4.941 | Desconocidos |

El extremo TimesNet alcanzó 180,485 segundos con 4,94 GiB muestreados; no fue un OOM observado. El resultado muestra que ese límite corto no basta para esta configuración, no que ningún presupuesto mayor pueda entrenarla.

En el piloto de variación, las cinco selecciones de TimeMixer consumieron su límite interno y conservaron sus pasos conocidos. Sin cinco selecciones completas no se calculó una mediana ni se ejecutaron sus reajustes. El híbrido y GRU32 completaron diez selecciones y diez reajustes, con **37 y 151 épocas fijas**, respectivamente. Se verificaron los 20 checkpoints con `weights_only=True` y sus hashes; diez replays conservaron 53 fechas comunes, con predicción durable anterior a observación. El informe cotejó 750 observaciones de hashes de fuentes sin discrepancias.

### Réplicas: resultado secundario y límites

El único par observado en este análisis es híbrido−GRU32. La media descriptiva de sus diferencias de RMSE fue +0,000258586: el híbrido presentó mayor error en esta muestra. Eso no constituye una comparación ajustada ni demuestra superioridad del candidato 11.

| Cantidad de planificación | Resultado |
|---|---:|
| Desviación de diferencias pareadas entre cinco semillas |0,000291733 |
| δ / desviación |0,88344 |
| R aproximado normal, estimación puntual |26 |
| R aproximado al usar la sensibilidad superior 95% de la desviación |142 |
| Escenario máximo conservado para costes |30 |
| R final de las recetas seleccionadas |Pendiente |

Las aproximaciones 26/142 dependen del par y de supuestos normales con sólo cinco semillas. No son tamaños requeridos demostrados para los 18 sistemas. Con R=30, la sensibilidad temporal produjo tasa de rechazo 0,0032 bajo un desplazamiento δ y 0,002 bajo la nula, frente a α/153≈0,000327. **La simulación no está calibrada bajo la nula**; esas tasas y su MDE simulado no se presentan como potencia ni como umbrales científicos validados. Se conserva R=30 únicamente como techo provisional de recursos y no se afirma potencia 0,80. El R final y la inferencia deberán resolverse con las recetas seleccionadas y un procedimiento temporal calibrado.

### Presupuesto de la campaña que falta

Con k=40, los escenarios contables de §27 son:

| R como escenario de coste | Etapas completas, máximo | Etapas cortas, máximo | Ajustes clásicos |
|---:|---:|---:|---:|
| 6 | 25935 | 88096 | 420 |
| 10 | 26639 | 88096 | 420 |
| 20 | 28399 | 88096 | 420 |
| 30 | 30159 | 88096 | 420 |

Las etapas auxiliares SAE se cuentan por separado. Son asignaciones de búsqueda, pilotos y reajustes de once orígenes; no incluyen controles, robustez, calibración, interpretabilidad ni reproducción independiente. Los límites por etapa permiten contabilizar asignaciones máximas, pero no estimar horas reales. El promedio de coste de mil arquitecturas y sus cohortes aún no fue medido; **horas totales y coste monetario siguen sin una estimación validada**.

## Cómo interpretar k, réplicas y horas

El piloto completo de ruido de §27 exige doce arquitecturas por cinco semillas en cada una de las seis familias NAS. Medir referencias y un extremo no completa esa matriz ni demuestra estabilidad de rankings. Cuando no sea estimable, se utiliza **k=40 como límite operativo conservador**, dejando visible el motivo y la cobertura pendiente.

El análisis de réplicas de esta sección es un **piloto previo a la selección**, registrado como enmienda. La guía §9.2 permite dos o tres modelos representativos; el piloto original de §27, con recetas ganadoras de la selección de octubre, sigue pendiente. Una recomendación basada en referencias no cierra automáticamente `final_seed_count` para las futuras recetas ganadoras.

Se estima la desviación de las diferencias pareadas de RMSE entre cinco semillas, con δ=0,00025772842664981106. El dimensionamiento usa α=0,05/153 como escenario conservador de los 153 pares de 18 sistemas en la métrica primaria. Se complementa con 5000 simulaciones, bloques temporales circulares de siete días y sensibilidad de la desviación por la escasez de semillas. La variación de calendario permanece como componente que no desaparece al aumentar réplicas de inicialización.

Las probabilidades simuladas son **sensibilidades condicionales de planificación**, no potencia alcanzada del experimento final ni evidencia de superioridad. Detectar una diferencia δ frente a cero es un objetivo distinto de demostrar una ganancia mayor que δ. Aplicar una corrección de 153 comparaciones no estima la varianza de los 152 pares ausentes del análisis secundario.

Las horas medidas describen este piloto y este entorno. Los límites acumulados por etapa son techos de asignación, no una predicción de duración. Faltan perfiles del espacio de arquitecturas, otros orígenes y presupuestos adicionales antes de afirmar un coste total de campaña.


## Evidencia y verificaciones

La integración automática comprueba **47 pruebas nuevas del piloto**, las 59 pruebas de entrenamiento anteriores y los 41 casos NAS. Los nuevos controles cubren admisión, fuentes, cronología, costes fallidos, exclusividad, snapshots, estadísticas y recibos. La construcción estricta de los tres libros y sus enlaces permanece como condición de publicación.

- [Resumen y hashes](../_static/development_pilot_20261109/summary.json).
- [Decisiones, escenarios y sensibilidades condicionales](../_static/development_pilot_20261109/decisions.json).
- [Tabla de recursos](../_static/development_pilot_20261109/resources_table.json) e [inventario de 20 checkpoints](../_static/development_pilot_20261109/checkpoints.json).
- [Recibos completos de recursos](../_static/development_pilot_20261109/resources_full.json.gz), [variación y fallos](../_static/development_pilot_20261109/variance_full.json.gz) y [diarios verificables](../_static/development_pilot_20261109/ledgers.json.gz).

Los recibos y los hashes se publican; los archivos de pesos del piloto permanecen en las salidas de ejecución. No son checkpoints de reanudación de optimizadores.

## Reproducción

En una copia del repositorio y un entorno Python3.12, use el lock de CPU de §29 y recupere las ocho fuentes fijadas de iTransformer. La instantánea de `experiments/e2_development_pilot_20261109/source_protocol.json` conserva los bytes usados: SHA256 `e0030c99…152abe`. Difiere del archivo histórico sólo en un salto de línea final; ambos tienen el mismo JSON canónico SHA256 `1bf33fb2…85c7e18`. El helper comprueba los dos hashes admitidos, guarda un respaldo exclusivo y prepara esos bytes exactos en la copia de trabajo; no cambia configuraciones ni los archivos históricos publicados. Los directorios de salida deben ser nuevos; los comandos conservan los intentos fallidos y no los reemplazan.

```bash
python -m pip install -r requirements-e2-training-cpu.lock.txt
python -m pip check
python scripts/prepare_e2_pilot_snapshot.py --root .
python scripts/fetch_e2_itransformer_sources.py --output work/author_sources/iTransformer
python scripts/run_e2_development_pilot.py \
  --data-root . --source-root-itransformer work/author_sources/iTransformer \
  --registration experiments/e2_development_pilot_20261109/registration.json \
  --output work/resources_reproduction
python scripts/run_e2_reference_variance.py \
  --data-root . \
  --registration experiments/e2_development_pilot_20261109/registration.json \
  --output work/reference_variance_reproduction
python scripts/build_e2_pilot_report.py \
  --resources work/resources_reproduction \
  --variance work/reference_variance_reproduction \
  --registration experiments/e2_development_pilot_20261109/registration.json \
  --output work/pilot_report_reproduction
```

Estos comandos usan datos de desarrollo y ejecutan optimización real en el entorno donde se lancen. Los pesos del piloto quedan en sus salidas de reajuste; la publicación conserva recibos y hashes. El entorno de esta sesión fue el servidor remoto de Codex Cloud.
