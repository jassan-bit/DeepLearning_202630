# Síntesis verificable de literatura y selección vigente

> **Archivo de antecedentes:** documento **§52** del registro anterior. Conserva su fecha, alcance y resultados históricos. [Ver organización del informe](concordancia_guia.md).

**Estado vigente en [§62](62_cierre_literatura_focal.md):** el flujo del corpus recibido está reconciliado: **9.559 grupos cribados y 42 estudios aplicados únicos**, con doce dimensiones de extracción y disposiciones de acceso explícitas. Los puntajes de los diez sistemas son calculables dentro de sus escenarios; la comparabilidad global permanece condicionada. **Esta página conserva el corte histórico y sus derivados**; sus estados parciales se interpretan en ese corte, sin sustituir el registro vigente.

**Lectura de los cortes conservados.** El [expediente de síntesis inicial](../_static/e2_literature_closure_20261113/manifest.json) mantiene sus tablas y huellas; la actualización de [§62](62_cierre_literatura_focal.md) cierra después la disposición del material recibido y amplía citas, calidad y disponibilidad. En el flujo vigente hay **9.559 grupos cribados, 42 estudios aplicados únicos y cero pendientes operativos**; **174 informes no recuperados** conservan elegibilidad desconocida. Los límites de cobertura —374 Scopus no recibidos y totales/rangos WoS incompletos— permanecen. Esta página distingue el corte inicial de esa evidencia posterior, sin añadir resultados Bitcoin.

## Dos cohortes con denominadores diferentes

La cohorte arquitectónica histórica contiene **diez estudios originales completos, 215 páginas físicas**. Sus identidades se recuperaron de forma dirigida. El [registro por estudio](../_static/e2_literature_closure_20261113/directed_originals10.csv) permite cerrar una evaluación local para síntesis de mecanismos: diez identidades distintas, diez textos completos recibidos, diez evaluados y diez admitidos con limitaciones de reporte. No hubo duplicados ni exclusiones **dentro de esas diez identidades fijadas**. Este tramo no representa el cribado del universo bibliográfico.

El Top 10 vigente de [§43](43_catalogo_obligatorio.md) corresponde a **diez sistemas y nueve estudios primarios**: FCN y ResNet comparten Wang et al. Sólo seis sistemas coinciden con la cohorte histórica: TimesNet, TSMixer, TiDE, TimeMixer, PatchTST e iTransformer. Wang se conserva como fuente fundacional de 2016, fuera de la ventana nominal 2019–2026, por definir los cuerpos convolucionales transferidos. La fuente de Inception comenzó en 2019 y la versión examinada es de 2020.

| Cobertura del Top 10 vigente | Estudios distintos | Alcance acreditado |
|---|---:|---|
| Seis originales modernos | 6 | Extracción previa de PDF completo y apéndices, con recibos |
| Wang, FCN y ResNet | 1 | Texto principal HTML completo, tablas y apéndice de contrastes examinados en la lectura adicional registrada |
| InceptionTime | 1 | Texto principal HTML completo, tablas y pies de figuras examinados en la lectura adicional registrada |
| Sagheer y Kotb, LSTM-SAE | 1 | Texto principal completo y tablas examinados en la lectura adicional registrada en PMC; sin recibo de un PDF descargado |
| Total vigente | **9** | Seis recibos de PDF completo y tres revisiones de texto principal web; no se inventa verificación visual de nuevos PDFs |

El [flujo local del primer corte](../_static/e2_literature_closure_20261113/bounded_flow.json) conserva sus valores históricos `global_eligible_studies=null` y `global_included_studies=null`; no se reescribe con los resultados posteriores. El [flujo vigente del corpus recibido](../_static/e2_prisma_received_v1/received_flow.json) reconcilia las rondas, fija 42 estudios aplicados únicos y documenta los informes inaccesibles. Ese cierre acotado no resuelve la cobertura de las bases ni la elegibilidad de los 174 no recuperados. No se calcula κ: no hay dos revisores humanos independientes acreditados.

El [registro de las tres lecturas adicionales](../_static/e2_literature_closure_20261113/additional_source_reading_ledger.json) documenta versiones, canal de acceso y alcance asistido por IA. No redistribuye los textos completos.

## Qué resultados sustentan realmente los diez sistemas

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

**NR** expresa ausencia de reporte en la versión y alcance examinados; **NV**, ausencia de verificación. La tabla anterior conserva el denominador de diez originales históricos y no describe las nueve fuentes actuales. Sus conteos no demuestran fuga, cero repeticiones ni mala calidad predictiva.

Para las **nueve fuentes actuales**, la [matriz vigente](../_static/e2_literature_delivery_closure_20261114_v3/quality/current_nine_quality.csv) registra escalado train-only reportado en 2/9, NR en 6/9 y NV en 1/9; repeticiones reportadas en 4/9, NR en 3/9, parcialmente en una y NV en otra; igualdad de presupuesto HPO NR en 9/9. La [verificación oficial de disponibilidad](../_static/e2_literature_delivery_closure_20261114_v3/quality/current_nine_data_access.csv) identifica datos públicos y código de autor fijado para **9/9 estudios** en ambos casos. Disponibilidad no verifica snapshot, preparación, splits o reproducción de esos experimentos. La calidad aplicada tiene otro denominador: **42 estudios × doce dimensiones = 504 filas**, enlazadas en §62. Cuartiles por revista y año no están verificados de forma uniforme.

Las tres lecturas web añaden valoración propia, separada del denominador anterior. Wang no informa semillas ni repeticiones exactas y excluye HPO/CV de sus baselines, sin acreditar igualdad frente a comparadores importados. Inception informa varias ejecuciones sin número exacto y no precisa la población del ajuste de normalización; los cinco miembros del ensemble no son cinco ensembles repetidos. SAE no informa semillas, repeticiones ni incertidumbre; el coste de convergencia descrito cubre fine-tuning y deja un coste adicional de preentrenamiento. Estos límites se conservan en cada fila de extracción.

La [réplica de §5](05_paper_guia.md) merece crédito experimental separado: cinco semillas de TimeMixer en ETTh1 produjeron MSE **0,377838**, con IC95 % de discrepancia **[−0,000726; 0,006401]**, compatible con la tolerancia registrada. También existe un piloto de tres modelos. Esa reproducción parcial y el diagnóstico de marcadores fortalecen la trazabilidad operativa; mantienen su alcance ETTh1 y no asignan un efecto Bitcoin a TimeMixer.

## Puntajes completos, incertidumbre y sensibilidad

La tabla siguiente es una **auditoría de soporte fijada el 13/11**, posterior al catálogo autorizado del 12/11. No reconstruye una selección previa ni convierte una preferencia autorizada en un óptimo bibliográfico. El [protocolo de puntaje](../_static/e2_literature_closure_20261113/support_score_protocol.json) declara:

$$S=100(0{,}15C+0{,}15U+0{,}25P+0{,}25A+0{,}20F).$$

En este primer corte, `C` es citación Scopus por año, normalizada por **487**, obtenido entonces de 2.435 citas de PatchTST divididas por cinco años. **487 se conserva como referencia histórica fija**, no como máximo del corte posterior: en la exportación uniforme actual PatchTST tiene 2.477 citas, 495,4 por año con la misma edad y `C` se limita a 1 conforme a la regla publicada en §62. No se recalibra el denominador ni se finge una consulta viva. Para edad y recencia se toma el primer año arXiv de la identidad verificada; cuando no existe esa identidad se usa el año de publicación documentado. Esta convención se declara ahora; las tablas anteriores conservan sus valores. `U=clip((año−2019)/7,0,1)`. `P` requiere desempeño comparable en el objetivo, horizonte, información y protocolo E2. `A=1` documenta una identidad de código autoral fijada. `F=1/n_familia` explicita representatividad dentro del catálogo: cuatro convolucionales, tres MLP, un recurrente/AE y dos de atención. La diversidad mínima de cuatro familias se verifica aparte.

**Estado del primer corte, conservado en la tabla siguiente:** P era desconocido para todos los originales y C para los cuatro sistemas añadidos. Sus extremos sólo evalúan algebraicamente los desconocidos dentro de [0,1], sin imputar cero ni uno. La [tabla original de componentes](../_static/e2_literature_closure_20261113/current10_support_scores.csv) permanece íntegra. **La evidencia vigente sí dispone de citas uniformes Scopus para 9/9 fuentes y diez sistemas**, con escenarios C/U/P/A/F calculables en [§62](62_cierre_literatura_focal.md). La [tabla estricta actual](../_static/e2_literature_closure_final/selection/tabla_principal.csv) distingue ese cálculo documental del P global no identificado y publica límites algebraicos; todos los modelos admiten rangos 1–10, sin dominancia robusta. La guía §3.2 no exige que el puntaje anteceda al benchmark; se declara la cronología sin atribuir al cálculo posterior haber causado la elección anterior.

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

La [sensibilidad inicial a cuatro ponderaciones](../_static/e2_literature_closure_20261113/score_sensitivity.csv) conserva pesos iguales, énfasis en implementación y énfasis en desempeño. La [sensibilidad estricta vigente](../_static/e2_literature_closure_final/selection/robust_global_P_bounds.csv) conserva además el análisis por estratos y límites globales: los rangos 1–10 expresan que el mérito global no está identificado, no un empate experimental. Los P locales de tareas o tablas distintas no se calibran automáticamente por estar entre cero y uno.

La [tabla de originales excluidos](../_static/e2_literature_closure_20261113/excluded_originals_context.csv) conserva Informer, Autoformer, Crossformer y S-Mamba. Crossformer tiene 1.675 citas verificadas en ese corte y S-Mamba 313; las cantidades ausentes para Informer y Autoformer quedan desconocidas. Su exclusión responde al alcance actual y a la cobertura funcional de familias, no a inferioridad demostrada. La justificación mundial de los más citados excluidos permanece abierta porque la cobertura de candidatos y citas es incompleta.

## Mecanismos transferibles y búsquedas nativas ejecutadas

La [matriz de mecanismos](../_static/e2_literature_closure_20261113/transverse_mechanisms.csv) conecta fuentes de origen con una propiedad compartida y una limitación concreta. Los textos de origen se identificaron por las referencias de los originales y sus fuentes primarias enlazadas; no se atribuye una nueva lectura completa a todos ellos. La tabla alimenta la consideración de mecanismos para E3 y no compromete su arquitectura final.

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

Esta ronda acredita **tres cadenas ejecutadas nativamente y recuperación parcial de metadatos**. La suma de totales, 187, no es una unión deduplicada ni un número de estudios incluidos; las 23 ocurrencias recuperadas tampoco son 23 estudios elegibles. El número de estudios únicos permanece desconocido. La recuperación completa y la adjudicación de esta ronda transversal conservan su alcance parcial, separado del **PRISMA ya reconciliado de 9.559 grupos recibidos**. No se presentan los intentos fallidos anteriores como el resultado final de las consultas ni se añaden sus totales al corpus cerrado. Las búsquedas web auxiliares mantienen su alcance complementario.

## Brecha contrastable derivada de la extracción

**En los nueve estudios primarios que sustentan el catálogo vigente no se identificó una evaluación de estos mecanismos para pronosticar σ₇ no anualizada a h=1 usando exclusivamente siete días OHLCV de Bitcoin, con presupuesto común e inferencia temporal pareada.** La [matriz de refutación](../_static/e2_literature_closure_20261113/gap_falsification_matrix.csv) enlaza cada sistema con su tarea, dataset y localizador que sostienen esa delimitación. Basta localizar en cualquiera de esos estudios una evaluación que sí cumpla esas condiciones para corregir la afirmación. La revisión global pendiente impide extenderla a toda la literatura.

La evidencia financiera complementaria de §3 reduce aún más el alcance de la brecha: TimeMixer ya se ha aplicado a BTCUSD, pero con volatilidad móvil de **21 días anualizada** ([arXiv:2410.09062](https://arxiv.org/abs/2410.09062), §3/Tabla 4); otros estudios usan retornos al cuadrado, volatilidad intradía o implícita. Estos objetivos no se convierten en σ₇ por compartir el término volatilidad. Por ello, la contribución comprobable es estimar la transferencia bajo el contrato E2 y estudiar sus límites de contexto, régimen y coste.

La brecha se conecta con [RQ1–RQ3 de §1](01_punto_partida.md): ganancia RMSE superior al δ heredado frente a GRU32/Persistence, erosión por régimen y mejora conjunta de error/coste. El calendario y la selección ejecutados están en [§48](48_protocolo_reducido.md), los resultados en [§6](06_benchmark_final.md) y la misma familia primaria refinada en [§56](56_inferencia_estabilidad_final.md). Las reglas secundarias aclaradas en §1 se identifican como posteriores, sin atribuirles un preregistro o resultados nuevos. El test conserva su herencia de E1 y exposición declarada. Esta síntesis no cambia δ ni selecciona una arquitectura original de E3.

Para reproducir los derivados del primer corte: `python scripts/e2_literature_closure.py --check`. El suplemento de identificación nativa conserva su propio manifiesto; la auditoría del flujo recibido y la extracción vigente se reproducen con los comandos de {ref}`§62 <e2-lit-coverage-audit>`. El registro actual cierra la conciliación y las disposiciones del material recibido, sin afirmar elegibilidad conocida para informes inaccesibles. Permanecen límites de cobertura de las bases, recuperación transversal parcial, calidad editorial no uniforme por año y comparabilidad del puntaje frente al universo de candidatos. La [lista del punto 3](../_static/e2_literature_current_v1/guide_checklist.csv) distingue esas limitaciones de las tareas documentales ya completadas.
