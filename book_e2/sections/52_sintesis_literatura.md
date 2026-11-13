# 52. Síntesis verificable de literatura y selección vigente

**Corte documental del 13/11/2026.** La revisión permite sostener mecanismos candidatos y una brecha de evaluación concreta. La elegibilidad global continúa sin determinar. Esta síntesis reúne evidencia ya extraída y una lectura adicional de fuentes primarias; el [expediente reproducible](../_static/e2_literature_closure_20261113/manifest.json) conserva las entradas y sus huellas. No añade ajustes ni resultados Bitcoin.

## 52.1 Dos cohortes con denominadores diferentes

La cohorte arquitectónica histórica contiene **diez estudios originales completos, 215 páginas físicas**. Sus identidades se recuperaron de forma dirigida. El [registro por estudio](../_static/e2_literature_closure_20261113/directed_originals10.csv) permite cerrar una evaluación local para síntesis de mecanismos: diez identidades distintas, diez textos completos recibidos, diez evaluados y diez admitidos con limitaciones de reporte. No hubo duplicados ni exclusiones **dentro de esas diez identidades fijadas**. Este tramo no representa el cribado del universo bibliográfico.

El Top 10 vigente de [§43](43_catalogo_obligatorio.md) corresponde a **diez sistemas y nueve estudios primarios**: FCN y ResNet comparten Wang et al. Sólo seis sistemas coinciden con la cohorte histórica: TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer. Wang se conserva como fuente fundacional de 2016, fuera de la ventana nominal 2019–2026, por definir los cuerpos convolucionales transferidos. La fuente de Inception comenzó en 2019 y la versión examinada es de 2020.

| Cobertura del Top 10 vigente | Estudios distintos | Alcance acreditado |
|---|---:|---|
| Seis originales modernos | 6 | Extracción previa de PDF completo y apéndices, con recibos |
| Wang, FCN y ResNet | 1 | Texto principal HTML completo, tablas y apéndice de contrastes examinados hoy |
| InceptionTime | 1 | Texto principal HTML completo, tablas y pies de figuras examinados hoy |
| Sagheer y Kotb, LSTM-SAE | 1 | Texto principal completo y tablas examinados hoy en PMC; sin recibo de un PDF descargado |
| Total vigente | **9** | Seis recibos de PDF completo y tres revisiones de texto principal web; no se inventa verificación visual de nuevos PDFs |

El [flujo local estructurado](../_static/e2_literature_closure_20261113/bounded_flow.json) mantiene `global_eligible_studies=null` y `global_included_studies=null`. Los 374 registros no recibidos de la consulta Scopus histórica, la conciliación entre rondas y las adjudicaciones pendientes de [§3](03_literatura.md) conservan sus denominadores. No se calcula κ: no hay dos revisores humanos independientes acreditados.

El [registro de las tres lecturas adicionales](../_static/e2_literature_closure_20261113/additional_source_reading_ledger.json) documenta versiones, canal de acceso y alcance asistido por IA. No redistribuye los textos completos.

## 52.2 Qué resultados sustentan realmente los diez sistemas

La [hoja vigente de extracción, CSV](../_static/e2_literature_closure_20261113/current10_extraction.csv) y su [versión JSON](../_static/e2_literature_closure_20261113/current10_extraction.json) contienen identidad, versión, localizador, arquitectura, adaptación, código, calidad y costes documentados. La política de [§44](44_arquitecturas_publicadas.md), actualizada por [§48](48_protocolo_reducido.md), fija los diez cuerpos: TiDE usa **TIDE-D7-PUBLISHED-R0-v2**, conservando separada su variante NAS histórica. Los diagramas, ecuaciones y recibos de los seis originales modernos permanecen en §3 y los de las cuatro altas en [§16](16_fichas_nuevas.md); las adaptaciones publicadas actuales se documentan en §44/§48.

| Sistema vigente | Resultado externo localizado | Interpretación y límite |
|---|---|---|
| FCN-1D | Wang, Tabla I, Adiac: error de clasificación **0,143** | Serie univariante completa; sustituir softmax por regresión cambia la tarea |
| ResNet-1D | Misma tabla y dataset: **0,174** | Comparación dentro del mismo artículo; no es error de volatilidad |
| Inception-1D | InceptionTime, §5/Fig. 10: **54/8/23** victorias/empates/derrotas frente a ResNet(5), p<0,01 | Resultado del **ensemble de cinco redes**; el sistema vigente usa una red individual |
| LSTM-autoencoder con predictor | Sagheer/Kotb, Tabla 2, PM2.5, dos capas: **RMSE 24,041; MAE 12,060** | Pronóstico siguiente con lag 25 e historia del objetivo; DLSTM obtiene menor RMSE: 23,750 |
| TimesNet | ETTh1 L96/H96, Tabla 13, PDF p.21: **MSE 0,384; MAE 0,402** | Periodicidad y convolución 2D; selección de frecuencias por batch exige controlar la inferencia |
| TSMixer | ETTh1 L512/H96, Tabla 3, PDF p.10: **0,361; 0,392** | Mezcla temporal y de canales; la mezcla de variables no mejora todas las filas del artículo |
| TiDE | ETTh1 L720/H96, Tabla 2, PDF p.8: **0,375; 0,398** | Historia del objetivo y covariables conocidas; E2 usa el cuerpo fuente r=0 y una cabeza escalar declarada |
| TimeMixer | ETTh1 L96/H96, Tabla 13, PDF p.17: **0,375; 0,400** | Configuración unificada; el cuerpo D7 vigente usa el preset corto documentado en §44 |
| PatchTST | ETTh1 L512/H96, /64, Tabla 3, PDF p.7: **0,370; 0,400** | P16/S8 de esta fila no cabe directamente en siete días |
| iTransformer | ETTh1 L96/H96, Tabla 10, PDF p.23: **0,386; 0,405** | Tokens de variables; cinco variables no heredan automáticamente una ventaja de escalabilidad |

Los pares modernos son MSE/MAE. **No se ordenan modelos comparando estas filas**: cambian contexto, partición, escalado y tarea. Las fuentes nuevas son [Wang et al., v4](https://arxiv.org/html/1611.06455v4), [InceptionTime, v3](https://arxiv.org/html/1909.04939v3) y [Sagheer/Kotb, texto primario PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6911101/). Wang declara normalización con estadísticas de entrenamiento y selección del checkpoint por pérdida de entrenamiento; sus contrastes entre datasets no miden incertidumbre entre semillas. InceptionTime emplea Friedman y Wilcoxon-Holm sobre clasificación. En SAE, Hyperopt usa RMSE de validación; las cantidades publicadas de bike suman 17.378 frente a 17.379 declaradas, y los conteos de partición no establecen por sí solos orden cronológico. Se conserva esa discrepancia.

### Calidad de reporte y reproducción

La [cobertura metodológica](../_static/e2_literature_closure_20261113/quality_coverage_architectural10.csv) usa exclusivamente los **diez originales históricos completos**, con el alcance de la fila numérica seleccionada:

| Campo | Hallazgo con denominador explícito |
|---|---|
| Scaler ajustado sólo en entrenamiento | Reportado en 2/10; NR en 8/10 |
| Identificadores numéricos de semillas | Reportados en 1/10; NR en 9/10 |
| Repeticiones de la condición seleccionada | Reportadas en 7/10; NR en 2/10; NV en 1/10 |
| Incertidumbre de esa misma fila | Reportada en 1/10; NR en 9/10 |
| Inferencia formal de esa comparación | Parcial en 1/10; NR en 9/10 |
| Igual presupuesto HPO | NR en 10/10 |

**NR** expresa ausencia de reporte en la versión y alcance examinados; **NV**, ausencia de verificación. Estos conteos no demuestran fuga, cero repeticiones ni mala calidad predictiva. La implementación de autor tiene identidad documentada para 10/10 sistemas vigentes, que representan 9/9 estudios. Las descargas originales de datasets y la reproducción ejecutable de esos nueve estudios no se validaron uniformemente; no se publica una proporción conjunta de código y datos reproducidos. Cuartiles por revista y año siguen sin verificar.

Las tres lecturas web añaden valoración propia, separada del denominador anterior. Wang no informa semillas ni repeticiones exactas y excluye HPO/CV de sus baselines, sin acreditar igualdad frente a comparadores importados. Inception informa varias ejecuciones sin número exacto y no precisa la población del ajuste de normalización; los cinco miembros del ensemble no son cinco ensembles repetidos. SAE no informa semillas, repeticiones ni incertidumbre; el coste de convergencia descrito cubre fine-tuning y deja un coste adicional de preentrenamiento. Estos límites se conservan en cada fila de extracción.

La [réplica de §5](05_paper_guia.md) merece crédito experimental separado: cinco semillas de TimeMixer en ETTh1 produjeron MSE **0,377838**, con IC95 % de discrepancia **[−0,000726; 0,006401]**, compatible con la tolerancia registrada. También existe un piloto de tres modelos. Esa reproducción parcial y el diagnóstico de marcadores fortalecen la trazabilidad operativa; mantienen su alcance ETTh1 y no asignan un efecto Bitcoin a TimeMixer.

## 52.3 Puntajes completos, incertidumbre y sensibilidad

La tabla siguiente es una **auditoría de soporte fijada el 13/11**, posterior al catálogo autorizado del 12/11. No reconstruye una selección previa ni convierte una preferencia autorizada en un óptimo bibliográfico. El [protocolo de puntaje](../_static/e2_literature_closure_20261113/support_score_protocol.json) declara:

$$S=100(0{,}15C+0{,}15U+0{,}25P+0{,}25A+0{,}20F).$$

`C` es citación Scopus por año, normalizada por **487**, el mayor valor observado con esta convención de edad: 2.435 citas de PatchTST divididas por cinco años. Se conserva el corte recibido de citas de noviembre 2026, sin fingir consulta viva. Para edad y recencia se toma el primer año arXiv de la identidad verificada; cuando no existe esa identidad se usa el año de publicación documentado. Esta convención se declara ahora; las tablas anteriores conservan sus valores. `U=clip((año−2019)/7,0,1)`. `P` requiere desempeño comparable en el objetivo, horizonte, información y protocolo E2. `A=1` documenta una identidad de código autoral fijada. `F=1/n_familia` explicita representatividad dentro del catálogo: cuatro convolucionales, tres MLP, un recurrente/AE y dos de atención. La diversidad mínima de cuatro familias se verifica aparte.

`P` permanece desconocido para todos los originales; `C` también para los cuatro sistemas añadidos. **No se imputan cero ni uno**. Los extremos siguientes sólo evalúan algebraicamente esos desconocidos dentro de [0,1]. La [tabla completa de componentes](../_static/e2_literature_closure_20261113/current10_support_scores.csv) conserva valores, citas originales, edades, intervalos y rangos admisibles.

| Sistema | C | U | A | F | P | Intervalo S |
|---|---:|---:|---:|---:|---|---|
| FCN-1D | NV | 0 | 1 | 0,250 | NV | [30,00; 70,00] |
| ResNet-1D | NV | 0 | 1 | 0,250 | NV | [30,00; 70,00] |
| Inception-1D | NV | 0 | 1 | 0,250 | NV | [30,00; 70,00] |
| LSTM-autoencoder con predictor | NV | 0 | 1 | 1,000 | NV | [45,00; 85,00] |
| TimesNet | 0,764 | 0,429 | 1 | 0,250 | NV | [47,89; 72,89] |
| TSMixer | 0,125 | 0,571 | 1 | 0,333 | NV | [42,11; 67,11] |
| TiDE | 0,126 | 0,571 | 1 | 0,333 | NV | [42,12; 67,12] |
| TimeMixer | 0,417 | 0,714 | 1 | 0,333 | NV | [48,63; 73,63] |
| PatchTST | 1,000 | 0,429 | 1 | 0,500 | NV | [56,43; 81,43] |
| iTransformer | 0,693 | 0,571 | 1 | 0,500 | NV | [53,96; 78,96] |

La [sensibilidad a cuatro ponderaciones](../_static/e2_literature_closure_20261113/score_sensitivity.csv) incluye pesos iguales, énfasis en implementación y énfasis en desempeño. **Ningún escenario identifica un orden único:** para cada sistema el rango conservador sigue siendo 1–10. Es evidencia de selección insuficientemente determinada, no un empate experimental.

La [tabla de originales excluidos](../_static/e2_literature_closure_20261113/excluded_originals_context.csv) conserva Informer, Autoformer, Crossformer y S-Mamba. Crossformer tiene 1.675 citas verificadas en ese corte y S-Mamba 313; las cantidades ausentes para Informer y Autoformer quedan desconocidas. Su exclusión responde al alcance actual y a la cobertura funcional de familias, no a inferioridad demostrada. La justificación mundial de los más citados excluidos permanece abierta porque la cobertura de candidatos y citas es incompleta.

## 52.4 Mecanismos transferibles y búsquedas nativas ejecutadas

La [matriz de mecanismos](../_static/e2_literature_closure_20261113/transverse_mechanisms.csv) conecta fuentes de origen con una propiedad compartida y una limitación concreta. Los textos de origen se identificaron por las referencias de los originales y sus fuentes primarias enlazadas; no se atribuye hoy una nueva lectura completa a todos ellos.

| Mecanismo y origen | Fuente primaria | Transferencia y condición D7 |
|---|---|---|
| Convolución, residuales y pooling; visión | [Long et al.](https://arxiv.org/abs/1411.4038), [He et al.](https://arxiv.org/abs/1512.03385) → Wang | Localidad y optimización residual; controlar padding y campo receptivo |
| Ramas con escalas distintas; visión | [Szegedy et al.](https://arxiv.org/abs/1602.07261) → InceptionTime | Detectar patrones a distintas escalas; kernels 40/20/10 exceden siete observaciones |
| Tokens de segmentos; visión | [ViT](https://arxiv.org/abs/2010.11929), [Swin](https://arxiv.org/abs/2103.14030) → PatchTST/Crossformer | Relaciones entre segmentos observados; redefinir parches válidos sin añadir días |
| Mezcla de posiciones y canales; visión | [MLP-Mixer](https://arxiv.org/abs/2105.01601) → TSMixer | Separar ejes temporal y de variables; contrastar mediante ablación la mezcla entre canales |

El 13/11 a las **18:52:37 UTC** se intentaron tres consultas nativas arXiv, sin filtro de fecha, `start=0,max_results=10`:

```text
all:"residual network" AND all:vision AND (all:transfer OR all:adaptation)
all:attention AND all:"image patches" AND (all:transfer OR all:adaptation)
all:"LSTM autoencoder" AND all:industrial AND (all:transfer OR all:adaptation)
```

Los intentos de las **18:52:37 UTC** fallaron por `Tunnel connection failed: 403 Forbidden`; el conector web tampoco recuperó sus URLs API. Su [registro histórico](../_static/e2_literature_closure_20261113/native_query_attempts.json) conserva los conteos `null` de aquella ejecución fallida.

**Recuperación nativa posterior desde Colab, 19:29:26–19:29:33 UTC.** Las mismas cadenas se ejecutaron realmente contra la API arXiv, sin filtro de fechas, con `start=0,max_results=10`. El [resumen verificado del recibo](../_static/e2_native_transverse_20261113/source_summary.json), la [tabla de consultas](../_static/e2_native_transverse_20261113/query_counts.csv) y el [manifiesto del suplemento](../_static/e2_native_transverse_20261113/manifest.json) documentan esta ronda separada.

| Consulta | Hora UTC del 13/11 | Total nativo | Metadatos recuperados |
|---|---|---:|---:|
| Residuales, visión y transferencia/adaptación | 19:29:26 | 106 | 10 |
| Atención, parches de imagen y transferencia/adaptación | 19:29:30 | 78 | 10 |
| LSTM-autoencoder, industria y transferencia/adaptación | 19:29:33 | 3 | 3 |

El recibo está fijado en el commit [`04d0947b1b043b7d389a742d7c3e06a97338d5b4`](https://github.com/jassan-bit/DeepLearning_202630/blob/04d0947b1b043b7d389a742d7c3e06a97338d5b4/runs/posthoc-native-search-v1/artifacts/native_transverse_receipt.json), con SHA-256 `1593584d6fd37d999f300f2879b7fb8213565062162d7fa287a5da9e904a8329`. Los 17 derivados anteriores permanecen como corte previo; sus valores desconocidos no se reescriben.

Esta ronda acredita **identificación nativa y recuperación parcial de metadatos**. La suma de totales, 187, no es una unión deduplicada ni un número de estudios incluidos; las 23 ocurrencias recuperadas tampoco son 23 estudios elegibles. El número de estudios únicos permanece desconocido. Siguen pendientes la recuperación de las páginas restantes, la deduplicación entre consultas y rondas, el cribado y la adjudicación por texto completo. Las búsquedas web auxiliares conservan su alcance complementario; la revisión global y el flujo PRISMA siguen parciales.

## 52.5 Brecha contrastable derivada de la extracción

**En los nueve estudios primarios que sustentan el catálogo vigente no se identificó una evaluación de estos mecanismos para pronosticar σ₇ no anualizada a h=1 usando exclusivamente siete días OHLCV de Bitcoin, con presupuesto común e inferencia temporal pareada.** La [matriz de refutación](../_static/e2_literature_closure_20261113/gap_falsification_matrix.csv) enlaza cada sistema con su tarea, dataset y localizador que sostienen esa delimitación. Basta localizar en cualquiera de esos estudios una evaluación que sí cumpla esas condiciones para corregir la afirmación. La revisión global pendiente impide extenderla a toda la literatura.

La evidencia financiera complementaria de §3 reduce aún más el alcance de la brecha: TimeMixer ya se ha aplicado a BTCUSD, pero con volatilidad móvil de **21 días anualizada** ([arXiv:2410.09062](https://arxiv.org/abs/2410.09062), §3/Tabla 4); otros estudios usan retornos al cuadrado, volatilidad intradía o implícita. Estos objetivos no se convierten en σ₇ por compartir el término volatilidad. Por ello, la contribución comprobable es estimar la transferencia bajo el contrato E2 y estudiar sus límites de contexto, régimen y coste.

La brecha se conecta con [RQ1–RQ3 de §1](01_punto_partida.md): ganancia RMSE superior al δ heredado frente a GRU32/Persistence, erosión por régimen y mejora conjunta de error/coste. Las pruebas y multiplicidad siguen [§45](45_enmienda_protocolo.md); los resultados futuros mantienen el alcance exploratorio del test conocido. Esta síntesis no cambia δ, no añade una hipótesis confirmatoria retroactiva y no selecciona todavía una arquitectura original de E3.

Para reproducir los derivados del primer corte: `python scripts/e2_literature_closure.py --check`. El suplemento de identificación nativa posterior conserva su propio manifiesto y procedencia inmutable. Quedan sin resolver la elegibilidad global con motivos por etapa, la conciliación de rondas/versiones, la recuperación y evaluación transversal completas, la verificación de calidad editorial por año y la selección comparativa frente al universo de candidatos. Los intervalos y campos desconocidos hacen visibles esas limitaciones.
