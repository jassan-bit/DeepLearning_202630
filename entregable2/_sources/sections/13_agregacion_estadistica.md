(estimandos-agregacion-e-inferencia-actual)=
# 13. Estrategia de Agregación y Modelado Estadístico

**Lectura vigente:** la A2 compara **16 sistemas sobre las mismas 608 fechas**, con seis semillas por cada una de trece redes y una trayectoria por cada clásico. La inferencia primaria conserva **veinte contrastes de mejora de RMSE mayor que δ**, con Holm20: ningún par rechaza, tanto en el cálculo original B=4.999 como en su refinamiento B=20.000. Los análisis posteriores tienen sus propias unidades, pérdidas, hipótesis y familias; no se suman como confirmaciones del mismo efecto. La [ampliación del 16/11 (§65)](65_cierre_cientifico_real.md) añade la comparación formal de los periodos originales, los controles reales de datos y una IUT CPU completa; A2 y RQ1 permanecen iguales.

Este capítulo desarrolla **13.1–13.4 de la guía, pp. 50–51**. La [lista verificable](../_static/e2_agregacion_entrega_current_v1/guide_checklist.csv), la [trazabilidad](../_static/e2_agregacion_entrega_current_v1/decision_traceability.csv), el [índice de evidencias](../_static/e2_agregacion_entrega_current_v1/current_evidence_index.json) y el [manifiesto](../_static/e2_agregacion_entrega_current_v1/manifest.json) distinguen los requisitos acreditados y los pendientes. El {download}`índice de esta consolidación <../_static/e2_agregacion_entrega_current_v1/README.md>` y la {download}`versión exacta anterior de este capítulo <../_static/e2_agregacion_entrega_current_v1/13_agregacion_estadistica_before_update.md>` conservan fuentes y estados, sin cálculos científicos nuevos. Esas listas conservan su corte documental; la evidencia adicional del 16/11 se consulta en [§65](65_cierre_cientifico_real.md).

(e2-unidades-agregacion-vigentes)=
## 13.1 Jerarquía de unidades y elección de la prueba

### 13.1.1 Estimando y fuentes de variación

Para la trayectoria de la semilla s del sistema m:

$$
q_{m,s}=\sqrt{\frac{1}{608}\sum_{t\in\mathcal T}(Y_t-\widehat Y_{m,s,t})^2},
\qquad \widehat\mu_m=\frac{1}{R_m}\sum_{s=1}^{R_m}q_{m,s}.
$$

La primaria es **la media de las RMSE individuales**, en unidades originales de σ₇, menor=mejor. No es la RMSE de promediar los pronósticos ni la raíz de errores cuadrados mezclados entre semillas. Los promedios mensuales no sustituyen el cálculo sobre todo el calendario. Para las redes $R_m=6$; Persistence, SeasonalNaive y ARIMA conservan $R_m=1$.

| Nivel o unidad | Papel y dependencia que se conserva |
|---|---|
| **Serie de mercado** | Una única serie BTCUSDT y una cronología compartida. No hay una muestra de mercados o datasets independientes para extrapolar una distribución poblacional entre activos. |
| **Fecha de objetivo/origen de pronóstico** | 608 fechas, con entradas y objetivos de ventanas solapadas. No se tratan como 608 observaciones iid. Las pérdidas diarias requieren bloques temporales o una aproximación de dependencia justificada. |
| **Semilla de inicialización** | Seis ejecuciones por red sobre los mismos datos y partición; describen optimización condicionada a esa cronología. La independencia/intercambiabilidad de trayectorias y el emparejamiento entre arquitecturas son supuestos, no consecuencia de compartir un número de semilla. |
| **Fold de selección HPO** | Tres orígenes, con entrenamientos expansivos solapados y 83 fechas de validación cada uno. Seleccionan la receta; no se multiplican por seis para afirmar 18 réplicas finales independientes. |
| **Comparador determinista** | Una trayectoria. Reutilizarla dentro de una resta pareada no crea entrenamientos adicionales, DE entre semillas o independencia temporal. |
| **Trimestre/ventana diagnóstica** | 112 filas sistema-trimestre y 8.304 ventanas móviles de 90 días en [§49](49_diagnosticos_f2a.md). Son descripciones dependientes, no nuevos datasets o réplicas iid. |

La [auditoría de partición](../_static/e2_eda_current_v1/partition_audit.json) conserva IDs, soportes y máscaras. Que calibración y test reutilicen historia observable con predictores fijos es legítimo; no implica independencia ni permite ajustar con etiquetas futuras. La métrica, δ y partición se heredan de E1 como exige la guía, con exposición previa declarada y alcance retrospectivo.

### 13.1.2 Unidad de cada comparación ejecutada

El **bootstrap primario** remuestrea semillas emparejadas y bloques del calendario conjuntamente dentro de cada par; recalcula la RMSE de cada trayectoria y después su media. Persistence mantiene una trayectoria determinista. Es un remuestreo de dos fuentes de variación condicionadas a una serie, **no un bootstrap jerárquico de varios mercados independientes**.

DM y MCS usan la pérdida diaria

$$
L_{m,t}=\frac{1}{R_m}\sum_s(Y_t-\widehat Y_{m,s,t})^2.
$$

Su media es una **MSE media de trayectorias fijas**. No equivale a la media primaria de RMSE ni a la pérdida de un ensemble. En esas comparaciones temporales las semillas quedan fijas: se modela dependencia mediante HAC14 o remuestreo común de fechas por bloques, respectivamente.

Friedman/Iman–Davenport y Wilcoxon se presentan como diagnósticos de **trece redes y seis vectores de semilla**, condicionados al calendario; excluyen los clásicos para no fabricar repeticiones. No constituyen Friedman sobre series u orígenes independientes. Su [diagrama de rangos](../_static/e2_benchmark_current_v1/conditional_rank_groups.svg) conecta no rechazos y no declara equivalencia. Las condiciones y p completos se conservan en [§50](50_comparaciones_f2a.md).

(e2-modelado-jerarquico-vigente)=
## 13.2 Efectos mixtos y descomposición de la varianza

La guía recomienda, cuando la estructura de observaciones repetidas lo justifique, un modelo como

$$
y_{ijr}=\mu+\alpha_j+u_i+v_r+\varepsilon_{ijr},
$$

con efecto fijo del modelo j, efecto aleatorio de unidad i y efecto de semilla/partición r. **No se ajustó ese modelo para el catálogo actual**, ni se publicaron contrastes mixtos, sus IC o proporciones de varianza actuales atribuibles a modelo, unidad, semilla y residuo.

Las DE entre semillas de A2 y las curvas por fecha muestran fuentes de variación, pero **no constituyen una descomposición de componentes de varianza**. Tampoco la sustituye una ANOVA o partición descriptiva del P0/catálogo anterior. Con una sola serie no se identifica varianza entre mercados; tratar sus 608 fechas correlacionadas como pacientes independientes produciría otra estructura estadística sin justificación. El emparejamiento por identificadores de semilla tampoco demuestra un efecto aleatorio común entre arquitecturas.

Un análisis jerárquico requeriría definir la cantidad por caso —por ejemplo, pérdida diaria, cuya media difiere de RMSE—, su estructura temporal, efectos anidados o cruzados y supuestos de residuos. La formulación ilustrativa no estima por sí sola esos componentes ni sus porcentajes. Los contrastes entre efectos de modelo necesitarían su corrección familiar e intervalos. Para métricas acotadas habría que justificar enlace, distribución o transformación; no se les asigna un modelo beta/logit que no se ha ejecutado.

Por tanto, se conserva como **recomendación metodológica y análisis no realizado**, sin inventar su cumplimiento ni exigir un ajuste mixto como requisito universal de todas las pruebas temporales. El límite real es que la descomposición solicitada no está acreditada para la ejecución vigente.

(e2-intervalos-efectos-vigentes)=
## 13.3 Intervalos de confianza, diferencias y tamaños de efecto

### 13.3.1 Qué intervalos están publicados

| Resultado | Intervalo y unidad que describe |
|---|---|
| **RMSE media individual de una red** | IC t del 95 % entre seis semillas, ya guardados en [el análisis A2](../_static/e2_benchmark_final_20261113/benchmark_analysis.json). Variación de optimización condicionada al calendario, con aproximación de pequeño tamaño; no incertidumbre entre mercados. No se atribuye ese IC a un clásico R=1. |
| **Ganancia primaria de RMSE** | Percentiles del 95 % del bootstrap pareado semillas/calendario, bloque 14. [Original B=4.999](../_static/e2_benchmark_final_20261113/primary_contrasts.csv) y [refinamiento B=20.000](../_static/e2_temporal_final_audit_20261114/primary_refinement_20000.csv) con la misma definición y δ. |
| **Diferencia de MSE diaria** | IC nominal del 95 % que invierte DM–HLN con HAC14 y referencia t607, en [las 120 parejas](../_static/e2_f2a_s8_20261113/dm_hln_hac14_all120.csv). No se convierte a un IC de ganancia RMSE>δ. |
| **Ablación menos SAE completo** | IC percentiles individuales de tres diferencias, seis pares de semillas, B=10.000 y bloques 14; p ajustados Holm3. [Resultados](../_static/e2_reproduction_ablations_20261113/ablation_comparisons.csv). |
| **Cobertura, errores y otros suplementos** | IC por calendario/bloques con método y soporte propios en [§49](49_diagnosticos_f2a.md), [§57](57_errores_techo.md) y [§60](60_calibracion_temporal_condicional.md). La cobertura condicional mantiene el calendario y la membresía, sin convertir grupos en series independientes. |

Los IC anteriores son **marginales**, salvo que una fuente identifique explícitamente otro alcance. No se convierten en intervalos simultáneos por publicar al lado un p Holm. La diferencia pareada tiene su propio IC: **el solapamiento de dos IC de scores individuales no decide si difieren**.

El bloque principal de 14 días es una elección registrada ante dependencia temporal; se guardan sensibilidades 7/28. Los residuos muestran estructura, pero esa observación no demuestra que 14 sea una longitud óptima o garantice cobertura ante dependencia arbitraria o cambios de régimen. Tampoco se remuestrean filas retenidas comprimiendo huecos del calendario.

Por ejemplo, para SAE con B=20.000:

| Referencia | Ganancia RMSE (referencia−SAE) | IC95 % marginal de la ganancia | p unilateral | p Holm20 |
|---|---:|---|---:|---:|
| GRU32 |0,000197733|[−0,000134704;0,000550757]|0,646118|1,000000|
| Persistence |0,000282166|[−0,000169219;0,000720013]|0,442778|1,000000|

La ganancia frente a GRU32 es menor que **δ=0,00025772842664981106**. Frente a Persistence la estimación supera δ, pero el IC incluye cero y no rechaza después de Holm. El ejemplo ilustra por qué magnitud, intervalo, margen y decisión se comunican conjuntamente, sin elegir una conclusión sólo por el promedio.

### 13.3.2 Efectos estandarizados y superioridad descriptiva

La familia primaria publica **efectos en unidades originales e IC de ganancia**, con p brutos/ajustados. La [ampliación ejecutada el 15/11/2026](64_correcciones_metodologicas.md) añade **d de Cohen pareado e IC percentiles para los veinte contrastes**, calculados directamente de las seis ganancias brutas por semilla, con 20.000 remuestreos de pares. Esta nueva estimación es posterior y describe variación de optimización condicionada al calendario; no reemplaza los IC temporales ni las decisiones primarias. El [CSV completo](../_static/e2_corrections_20261115/statistics/standardized_effects_20.csv) conserva los sorteos no definidos por varianza nula y sus límites; no se exige calcular simultáneamente Cohen, Cliff y A12.

El CSV de DM sí conserva dos campos descriptivos adicionales. Con $d_t=L_{A,t}-L_{B,t}$:

$$
\mathrm{standardized\_daily\_difference}=\frac{\bar d}{s_d},
\qquad
\widehat P_{\mathrm{fecha}}=\frac1{608}\sum_t
\left[\mathbf1(d_t<0)+\tfrac12\mathbf1(d_t=0)\right].
$$

El primer campo estandariza diferencias diarias de MSE y **no tiene IC estandarizado guardado**; la dependencia serial y su estimando impiden presentarlo como tamaño primario de la ganancia RMSE>δ. El segundo es la fracción empírica de fechas donde A tiene menor pérdida, contando medio empate: responde a una comparación sobre las unidades observadas de ese calendario, **no a una probabilidad inferencial de ganar en otro mercado o en una nueva ejecución**. No hay IC de esa fracción en el archivo.

| Par A frente a B | Diferencia diaria estandarizada | Fracción empírica de fechas favorables a A |
|---|---:|---:|
| SAE frente a GRU32 |−0,077709|0,544408|
| SAE frente a Persistence |−0,048849|0,330592|

Un valor negativo del primer campo favorece a A. SAE puede tener menor RMSE media y, a la vez, perder frente a Persistence en muchas fechas: el tamaño de los errores y la proporción de días responden a preguntas diferentes. Los [campos originales](../_static/e2_f2a_s8_20261113/dm_hln_hac14_all120.csv) conservan toda la precisión y la definición del [motor](https://github.com/jassan-bit/DeepLearning_202630/blob/2769b25574f4406bf8afcc44778a2a20f04977e5/scripts/e2_f2a_s8_comparisons.py).

La proporción de remuestreos con ganancia positiva **no se interpreta automáticamente como probabilidad posterior, predictiva o de superioridad poblacional**. Las colas bootstrap del contraste, las proporciones entre semillas, la fracción diaria anterior y la pertenencia a MCS tienen definiciones distintas. La tabla inferencial de [§61](61_reproduccion_consolidada.md) puede reunir p e IC primarios y diagnósticos disponibles, manteniendo los campos ausentes y separando estos descriptores MSE de los efectos RMSE.

(e2-reporte-familias-vigente)=
## 13.4 Reporte estadístico, familias y sensibilidad

### 13.4.1 Familias identificadas y decisiones

| Familia o procedimiento | Estimando, control y resultado publicados |
|---|---|
| **Primaria, 20 pares** | Ganancia RMSE>δ, candidatos frente a GRU32/Persistence, **Holm20**, α familiar 0,05. Cero rechazos; B=4.999 y B=20.000 conservan la misma hipótesis y método. |
| **Dos diagnósticos de semillas** | t pareada y 64 cambios de signo, 20 pares en cada lista, **Holm20 separado por procedimiento**. Todos sus p ajustados son 1. Son dos métodos ejecutados, condicionados al calendario, con normalidad aproximada/simetría e independencia pertinentes. |
| **Residuos y estabilidad trimestral** | 826 hipótesis, **Holm826**, 707 rechazos. No son 826 muestras independientes ni otra prueba primaria de mejora. |
| **DM–HLN** | 120 parejas sobre igualdad de MSE diaria, **Holm120**, 35 rechazos. El IC individual no sustituye el ajuste de p. |
| **MCS 90 %** | Procedimiento conjunto secuencial con B=10.000 y bloques 14. Conserva FCN, SAE, GRU32, Persistence y ARIMA. No se añade Holm a sus eliminaciones ni se afirma equivalencia. |
| **Rangos condicionados** | Friedman/Iman–Davenport global y comprobación por permutación de la misma nula; post-hoc de 78 parejas Wilcoxon, **Holm78**, cero rechazos. El diagnóstico no representa varias series independientes. |
| **GR y constancia** | 20 contrastes de igualdad predictiva y 20 de constancia, B=20.000, **Holm40 conjunto**, cero rechazos. No rechazo no demuestra estabilidad universal. |
| **Ablaciones SAE** | Tres contrastes, **Holm3**, un rechazo: retirar dropout supervisado aumenta RMSE. Sus 18 ajustes no son 18 contrastes. |
| **Cobertura por política/grupo** | 1.344 oportunidades, 1.336 estimables, **Holm1344**; las ocho no estimables conservan estado y p=1 para el tamaño de familia, sin atribuirles aprobación científica. [§60](60_calibracion_temporal_condicional.md). |
| **Ensemble calibrado** | Dos comparaciones posteriores de ganancia>0, **Holm2**, cero rechazos; no prueban superar δ ni se añaden a A2. [§63](63_ensamble_calibracion.md). |
| **RQ2 posterior** | Diez erosiones de ventaja frente a GRU32, régimen causal, B=20.000 y bloques14, **Holm10**: siete deterioros mayores que δ. [§64](64_correcciones_metodologicas.md). |
| **Brecha CPU respecto a δg** | Trece medias de brecha calibración−train, seis semillas, t con 5 gl, **Holm13**: dos tolerancias condicionales establecidas y once no establecidas. No acredita adecuación conjunta. [§64](64_correcciones_metodologicas.md). |
| **Periodos A100 originales** | Trece diferencias prueba608−calibración53 de los mismos predictores, t bilateral sobre seis semillas, **Holm13**: seis rechazos; sensibilidad por 64 signos con su propio Holm13: cero. Los tres clásicos se reportan descriptivamente. [Resultados](../_static/e2_scientific_closure_20261116/period_comparison/summary.json). |
| **RQ3 CPU completa** | Diez IUT de ganancia RMSE>δ y reducción de latencia, ambos componentes CPU, B=20.000, **Holm10**: cero mejoras conjuntas; p de coste, IUT y ajustado iguales a 1. [Resultados](../_static/e2_scientific_closure_20261116/cpu_test_iut/statistics/CPU_full_IUT_Holm10.csv). |
| **Etiquetas permutadas** | Trece controles entrenados con una semilla y etiquetas de train permutadas por bloques7. IUT de ganancia>δ y correlación positiva, B=10.000/bloques14, **Holm13**: cero alertas ajustadas; los p brutos no son todos mayores que 0,05. [Resultados](../_static/e2_scientific_closure_20261116/controls/negative_summary.json). |
| **Adversarial de dominio** | Una comparación train655/test608, cinco folds purgados, 999 permutaciones por bloques7 con reajuste: AUC OOF=0,95254, p=0,001. Los folds pueden incluir filas posteriores al bloque retenido: es un diagnóstico de dominio, no un backtest causal. La explicación por grupos de atributos es descriptiva y no añade nuevos contrastes confirmatorios. [§65](65_cierre_cientifico_real.md). |

Se conservan las familias registradas y las posteriores, con sus planes y fechas. La guía recomienda **FWER para confirmación** y FDR para exploraciones desagregadas; los análisis indicados aplicaron Holm o su procedimiento conjunto declarado. No se afirma haber ejecutado Benjamini–Hochberg donde no existe, ni se cambia de corrección después de observar los resultados. Las frecuencias de errores y el estrés con IC descriptivos no se convierten silenciosamente en otra familia confirmatoria.

El análisis primario tiene **tres procedimientos reales por par** —bootstrap, t y cambios de signo—, con control Holm en cada lista. Se reconoce su presencia y el componente de control múltiple de la guía; Holm no es una cuarta prueba ni obliga por sí solo a añadir otro global. La cantidad de métodos no demuestra sus supuestos ni tres confirmaciones temporales homogéneas. Los contrastes posteriores de MSE o rangos no se suman como votos ni como confirmaciones adicionales de la ganancia>δ.

### 13.4.2 Reporte mínimo y sensibilidad de la conclusión

El reporte por contraste debe identificar par, unidad, dirección, estimando, margen, efecto, IC95 %, p sin ajustar, p ajustado, regla y familia. Las tablas conservan los campos efectivamente publicados; los IC o tamaños de efecto ausentes se marcan como tales, sin afirmar que todas las familias estén completas. Los CSV conservan la precisión disponible; «p exacto» en el reporte **no convierte una aproximación asintótica o Monte Carlo en una prueba exacta de muestra finita**. En el auditor residual existen 35 salidas `p_raw=0.0` por precisión/subdesbordamiento: no se interpretan como probabilidades matemáticas nulas. Las ausencias de IC o efecto estandarizado permanecen señaladas.

Los diagnósticos alternativos t/signos no producen rechazos tras sus Holm y no cambian la conclusión primaria. Las sensibilidades temporales con bloques 7/28 están guardadas como diagnósticos **sin ajuste**, y el refinamiento a 20.000 conserva las veinte decisiones; ninguno autoriza elegir el bloque, método o familia con el p menor. La corrección Nadeau–Bengio se considera para comparaciones pertinentes de particiones con entrenamiento solapado, no como una opción que deba ejecutarse automáticamente en todas las trayectorias temporales.

La [potencia](09_protocolo_registrado.md) es limitada: 0,122953 en la alternativa planificada de 2δ, con R=6, dos representantes y 53 fechas de desarrollo. No se aumenta R tras observar la prueba. El [δg previo](../_static/e2_point1_review_v1/margin_lineage.json) se reconoce; las brechas anteriores no establecían su cumplimiento. [§64](64_correcciones_metodologicas.md) añade trece contrastes comparables en CPU, con dos tolerancias condicionales; no certifica ausencia de sobreajuste original. [§65](65_cierre_cientifico_real.md) sí ejecuta la comparación formal de riesgos originales entre calibración y prueba, manteniendo que calibración no es validación HPO independiente. Sus seis rechazos t/Holm13 y cero rechazos por signos se reportan juntos, sin elegir el procedimiento más favorable ni convertir no rechazo en equivalencia. No se heredan análisis de varianza o p del P0 para cerrar esas condiciones.

La conclusión se expresa según el criterio fijado: **menor RMSE observada y ausencia de superioridad práctica confirmada**. No se escribe «tendencia a la significancia», ni se equipara no rechazo con equivalencia. TOST/ROPE no están ejecutados; la reproducción numérica, un IC amplio o permanecer en MCS no los sustituyen. Las limitaciones estadísticas y los resultados negativos se mantienen junto a los favorables.

````{admonition} Archivo documental: contenido anterior del capítulo 13
:class: dropdown

Este bloque conserva íntegramente el texto anterior, sus encabezados y enlaces. Sus fechas y estados corresponden a sus cortes originales; la lectura vigente sigue 13.1–13.4 arriba.

**Estado vigente del 14/11/2026:** la A2 compara **16 sistemas y 608 fechas**, con R=6 para las trece redes y R=1 para los tres comparadores deterministas. El [refinamiento de §56](56_inferencia_estabilidad_final.md) usa 20.000 sorteos de la misma familia primaria; los análisis posteriores se identifican por su estimando y su familia.

## 13.1 Qué se estima

Para un modelo m y semilla s se calcula la RMSE de toda su trayectoria sobre las 608 fechas comunes. La primaria de una red es la media de sus seis RMSE individuales. La RMSE del pronóstico promedio es otro estimando. No se promedian RMSE mensuales sin ponderación para sustituir el error del calendario completo.

Hay una sola serie de mercado. Las seis semillas describen variación de optimización, no seis mercados independientes. Se emparejan las fechas y las semillas cuando corresponde; los comparadores deterministas conservan una única trayectoria. Las máscaras mantienen las posiciones reales del calendario sin comprimir huecos.

## 13.2 Familia primaria y refinamiento

La familia tiene **20 comparaciones**: cada candidato frente a GRU32 y Persistence. La ganancia es RMSE de la referencia menos RMSE del candidato; el margen práctico es **δ=0,00025772842664981106**. Se usa bootstrap pareado de semillas y posiciones temporales, bloque principal de 14 días y Holm20.

El cálculo original de 4.999 réplicas queda disponible. [§56](56_inferencia_estabilidad_final.md) lo reproduce exactamente antes de recalcular las mismas hipótesis y el mismo estimando con **20.000 réplicas**. Los veinte p Holm siguen siendo 1. Este refinamiento mejora la resolución Monte Carlo; no cambia recetas, márgenes ni el carácter retrospectivo de la prueba.

## 13.3 Comparaciones y diagnósticos posteriores

[§50](50_comparaciones_f2a.md) aporta DM–HLN/HAC14 y MCS sobre pérdidas cuadráticas diarias, además de rangos condicionados a semillas. Son objetivos distintos de una mejora práctica de RMSE mayor que δ. [§56](56_inferencia_estabilidad_final.md) separa Giacomini–Rossi bajo igualdad predictiva de un diagnóstico centrado de constancia; ajusta conjuntamente 40 hipótesis. Un rechazo de igualdad puede deberse a una ventaja persistente y no demostrar un quiebre. Un no rechazo no demuestra equivalencia ni estabilidad universal.

Las [ablaciones](55_reproduccion_ablaciones.md) usan 10.000 remuestreos y Holm3, conservando sus seis pares de semillas. La taxonomía de [§57](57_errores_techo.md) informa IC marginales de frecuencias bajo dependencia local; no ofrece cobertura simultánea de todas las categorías ni define causas de error.

## 13.4 Potencia, brechas y lenguaje de conclusión

El piloto de cinco pares de semillas estimó potencia **0,123** para el efecto planificado con R=6, frente al objetivo 0,80. Su alcance se limita a dos representantes y 53 fechas de desarrollo. No se amplían repeticiones después de observar la prueba para buscar un resultado favorable.

El margen de sobreajuste **δg sí estaba definido previamente**; su [linaje registrado de δg](../_static/e2_point1_review_v1/margin_lineage.json) conserva el valor y el alcance de las familias anteriores. Las 78 brechas guardadas y las tolerancias ilustrativas de [§56](56_inferencia_estabilidad_final.md) son descriptivas; no acreditan no inferioridad train–validación. La calibración no se presenta como una nueva validación independiente. Menor RMSE observada, ausencia de rechazo y compatibilidad numérica de una réplica son conclusiones diferentes.


## Archivo de la versión anterior

La {download}`versión exacta anterior a esta consolidación <../_static/e2_delivery_reproduction_20261114/archived_chapters/13_agregacion_estadistica.md>` se conserva con sus estados y fechas históricos. El [manifiesto de archivo](../_static/e2_delivery_reproduction_20261114/chapter_archive_manifest.json) fija su commit y SHA256. Esos antecedentes no sustituyen el protocolo ejecutado de [§48](48_protocolo_reducido.md) ni su Apéndice B actual de [§61](61_reproduccion_consolidada.md).


````
