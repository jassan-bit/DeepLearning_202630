# 12. Techo de desempeño, incertidumbre y análisis de errores

**Estado vigente al 13/11/2026:** la [A2 final de §6](06_benchmark_final.md) ya contiene las métricas e intervalos de 16 sistemas. [§49](49_diagnosticos_f2a.md) añade cobertura con incertidumbre temporal, calibración y desagregación trimestral sobre predicciones guardadas. La referencia de techo, cobertura condicional completa y clasificación de errores requerida siguen pendientes. Los estados anteriores conservados debajo son históricos.

**Vigencia documental (11/11/2026):** existe selección HPO/NAS registrada y ejecutada para el catálogo actual; las afirmaciones de ausencia de HPO o de falta de registro de selección corresponden a cortes anteriores. El benchmark final, R y su cierre inferencial siguen pendientes. [Registro y alcance actuales (§9)](09_protocolo_registrado.md) · [Calendario (§40)](40_calendario_original.md) · [Corte de resultados (§41)](41_avance_optimizacion_11_origenes.md) · [Auditoría y trabajo paralelo (§42)](42_auditoria_y_trabajo_paralelo.md).

**Estado:** análisis P0 histórico ejecutado; incertidumbre y errores del catálogo vigente pendientes de la nueva enmienda y de su evaluación. El catálogo contiene diez modelos y CNN-1D + LSTM + atención temporal como candidato 11, sin superioridad acreditada; las referencias MLP, LSTM y GRU de E1 se mantienen aparte.

Esta sección desarrolla §12.1–12.3 de la guía (PDF pp48–50). Los intervalos predictivos describen valores futuros de σ₇; los IC de RMSE describen incertidumbre de una métrica. Ni la dispersión entre semillas ni un IC de RMSE se presentan como un intervalo predictivo. La [enmienda anterior](../_static/completion_protocol/analysis_additions.json) y el [plan P0](https://github.com/jassan-bit/DeepLearning_202630/blob/599a29f87f8ebb1d67cd925ec09bf6bdbc732fa0/experiments/e2_remediation_20261106/uncertainty_errors.json) quedan como archivo de trazabilidad, sin trasladar al catálogo vigente sus cohortes, tamaños de ensamble o familias estadísticas.

## 12.1 Techo de desempeño y margen de mejora real

**Estado: estimación pendiente.** La guía (§12.1, pp48–49) exige al menos una referencia de techo o predictibilidad y cuantificar el margen restante antes de interpretar diferencias entre modelos. La volatilidad σ₇ se calcula mediante una regla determinista; esto permite auditar su reconstrucción, pero no implica que pueda pronosticarse sin error con D7×5.

### 12.1.1 Referencia aplicable a esta serie temporal

Se propone estudiar la predictibilidad intrínseca en desarrollo mediante entropía de permutación y error fuera de muestra de predictores diversos, con selección temporal causal. Antes de estimar deben registrarse la serie analizada, transformación, dimensión y retardo de la entropía, segmentos continuos, ventanas, tratamiento de empates y tamaño efectivo; la sensibilidad a esos parámetros también debe quedar fijada.

La entropía es un diagnóstico, no una cota numérica de RMSE. El error de un conjunto de modelos aporta una referencia empírica alcanzable, no prueba el error de Bayes ni un techo universal. La selección y cualquier combinación se ajustarán sólo en entrenamiento/validación admisibles. No se elegirá la referencia por su desempeño en test. La receta, los soportes y el coste del conjunto deberán registrarse si se utiliza; su propuesta no añade un modelo al catálogo por sí sola.

### 12.1.2 Distancia normalizada y sentido de la métrica

La expresión de la guía usa un score donde mayor es mejor. Para RMSE, donde menor es mejor, su equivalente es:

$
\gamma=\frac{e_{\mathrm{mejor}}-e_{\mathrm{techo}}}
{e_{\mathrm{base}}-e_{\mathrm{techo}}}.
$

Aquí los tres errores deben compartir objetivo, escala, cohorte y regla de agregación; el baseline y el procedimiento de selección del mejor modelo se fijan de antemano. El término $e_{\mathrm{techo}}$ requiere una estimación defendible del error límite. Si sólo se dispone de una referencia empírica, se identificará como tal y no se declarará un techo científico demostrado.

Se reportarán las estimaciones de los tres errores, incertidumbre temporal y sensibilidad de γ. Si el denominador es cero, no positivo o demasiado incierto para sostener la comparación, γ se declara no interpretable. No se recortan valores a [0,1] ni se transforma una referencia empírica superada en una afirmación de haber excedido el límite alcanzable. Tampoco se calculará γ mezclando resultados históricos con los del catálogo vigente.

### 12.1.3 Ruido y criterio de cierre

Se distinguirá error de reconstrucción de etiquetas, calidad de OHLCV y variabilidad futura no explicada por las entradas. La coincidencia de etiquetas reconstruidas no demuestra ausencia de incertidumbre irreducible. Si la auditoría identifica errores de etiquetas relevantes, se documentará una submuestra corregida y el análisis de sensibilidad del ranking exigido por la guía, sin corregir selectivamente casos según qué modelo favorezcan.

El cierre requiere método registrado, evidencia reproducible de desarrollo, incertidumbre y límites explícitos. No hay aún un techo estimado ni una cifra de margen restante para los once candidatos. Su seguimiento está en [R10 de la matriz de cierre](25_matriz_cierre.md).

## 12.2 Cuantificación de incertidumbre

### 12.2.1 Predictor congelado y calibración disjunta

Un predictor ajustado con etiquetas de un periodo no puede calibrarse usando los errores de ese mismo periodo como si fueran residuos fuera de muestra. Por ejemplo, el ajuste histórico principal podía usar resultados hasta el 24/12/2024; sus errores de noviembre–diciembre no constituyen automáticamente calibración disjunta.

La nueva enmienda deberá separar selección, ajuste, calibración y evaluación. Para cada modelo que reciba intervalos se requiere:

- Identificar receta, soporte, escalador, pesos y cualquier combinación de miembros antes de observar sus residuos de calibración.
- Fijar fechas de selección y ajuste, embargo y calibración, con etiquetas causalmente separadas. Las etiquetas σ₇ de calibración se solapan entre sí y no se suponen independientes.
- Mantener exactamente el mismo predictor durante calibración y evaluación. Un refit o un cambio en el número de miembros necesita otra justificación y no hereda los intervalos anteriores.
- Cerrar y registrar los calibradores antes de evaluar. Los timestamps reales de cierre se conservan en UTC y no se confunden con los orígenes del replay.
- Definir de antemano cómo se cubrirá a los modelos que finalmente se reporten, para evitar seleccionar una receta y descubrir después que no tiene calibración válida.

La incertidumbre de una media de pronósticos se atribuye a esa media, no a otra trayectoria ni al predictor reajustado. El error puntual del predictor calibrado se informa junto al primario sobre las mismas fechas; esa comparación no sustituye la media primaria de RMSE por semilla. La exposición histórica de recetas E1 a los datos debe declararse sin asignarles fechas de selección ficticias.

### 12.2.2 Intervalos temporales y evaluación

Para un predictor congelado A, los residuos absolutos de calibración son $s_i=|y_i-\widehat y_i^A|$. Un procedimiento residual puede construir $[\max(0,\widehat y^A-q_\alpha),\widehat y^A+q_\alpha]$, con el cuantil, ponderaciones y tratamiento de masa finita definidos antes de evaluar. La nueva enmienda deberá fijar esos parámetros y las coberturas nominales; no se heredan automáticamente del plan sustituido. Si la muestra no permite un radio estimable, se informa ese límite en lugar de inventar un intervalo útil.

Una calibración inspirada en el enfoque conforme no garantiza cobertura exacta si faltan sus supuestos. La dependencia de σ₇, el cambio de régimen, la selección de modelos y la separación temporal entre calibración y evaluación limitan la transferencia. Los pesos de recencia no prueban intercambiabilidad ni son cocientes de densidades por el mero hecho de ponderar. Una evaluación retrospectiva tampoco se presenta como una ventana online actualizada con etiquetas futuras.

La evaluación debe informar cobertura empírica frente a nominal, anchura media, score de intervalo y pérdidas pinball de las cotas. WIS combina intervalos con un pronóstico identificado del cuantil 0,5; no convierte cualquier media en la verdadera mediana condicional. **CRPS requiere una distribución predictiva declarada.** Una CDF empírica de residuos firmados y un intervalo simétrico basado en residuos absolutos son objetos diferentes. La variabilidad de optimización no es por sí sola una distribución de futuras volatilidades.

La cobertura condicional requiere grupos causales y umbrales fijados con datos admisibles antes de calibrar. Un grupo vacío o demasiado pequeño se informa como no estimable, sin asignarle cobertura cero. El remuestreo mantiene el calendario y filtra los grupos después de extraer bloques; no concatena fechas separadas como si fueran vecinas. Las longitudes de bloque, repeticiones y sensibilidades deberán justificarse y registrarse para el diseño vigente.

Los IC descriptivos no son simultáneos por defecto. El nivel de cobertura predictiva, el nivel de error de una familia de contrastes y el margen δ de RMSE tienen funciones distintas. Si se incluyen PIT, distancia respecto a uniforme o pendiente/intercepto de calibración, se declararán su inferencia temporal y multiplicidad; la referencia iid de Kolmogorov–Smirnov no se aplica automáticamente a residuos dependientes.

Los casos de menor confianza pueden identificarse por anchura predictiva únicamente cuando dispongan de una calibración válida para su origen. No se asigna retrospectivamente confianza OOF de 2024 usando residuos futuros de noviembre–diciembre.

## 12.3 Casos de error: unidad, taxonomía y muestreo

### 12.3.1 Requisitos del nuevo análisis

La unidad es un **origen diario único**, no cada semilla. Cinco semillas sobre cien fechas siguen siendo cien casos. El requisito de al menos cien casos auditables por modelo debe comprobarse sobre fechas elegibles comunes, manteniendo las posiciones de días no elegibles vacías.

La nueva enmienda debe fijar el panel OOF, sus cortes y la selección temporal anidada. Un fold mensual mantiene la identidad de su `run_id`, checkpoint, escalador y soporte durante el mes; no recibe un refit diario oculto. Toda exposición histórica de una receta a resultados posteriores a un origen OOF se declara y limita su interpretación.

Las categorías de error deben definirse con una referencia de entrenamiento identificada, no con los errores que luego se analizarán ni con el test. La magnitud puede estratificarse mediante cuantiles de esa referencia y la dirección distinguir subestimación consistente, sobreestimación consistente o signos mezclados entre semillas. Esto evita que errores grandes de signo opuesto desaparezcan al promediar pronósticos. Los umbrales exactos y los empates requieren una regla previa; δ no se transforma en una tolerancia arbitraria por caso.

Se clasifica el panel completo y, si se extrae una muestra de auditoría estratificada sin reemplazo, se conserva la probabilidad de inclusión. Las frecuencias poblacionales del panel no se estiman como porcentajes sin ponderar de una muestra estratificada. Los casos y predicciones individuales quedan en la evidencia privada; los agregados públicos identifican denominadores, categorías e incertidumbre temporal.

El solapamiento de errores grandes, Jaccard por pares y correlación de errores firmados pueden describir complementariedad. Un ensamble que se compara con su mejor miembro en el mismo OOF conserva el sesgo de esa referencia elegida y no prueba una ganancia futura. El análisis de errores del P0 ya ejecutado sirve como antecedente, sin sustituir los casos del catálogo vigente ni seleccionar retrospectivamente a sus mejores modelos.

### 12.3.2 Análisis P0 histórico ejecutado

Este análisis pertenece al registro anterior y no evalúa el catálogo vigente. Se analizaron **305 fechas únicas por cada una de las cuatro recetas**, conservando cinco semillas y los diez checkpoints mensuales por receta. La taxonomía se fijó a las **15:14:04 UTC**, antes del primer scoring OOF, usando los **341 errores de persistencia** admisibles de 09/01–24/12/2023. Sus umbrales son **q₅₀=0,00101245** y **q₉₀=0,00683008**; no se estimaron con errores de 2024 ni cambiando δ.

| Receta P0 | Fechas únicas clasificadas | Casos únicos en muestra privada | Errores grandes, total | Subestimaciones grandes consistentes |
|---|---:|---:|---:|---:|
| TimeMixer | 305 | 100 | 68 (22,30 %) | 61 |
| iTransformer | 305 | 100 | 73 (23,93 %) | 61 |
| PatchTST | 305 | 100 | 54 (17,70 %) | 49 |
| GRU | 305 | 100 | 38 (12,46 %) | 23 |

Los casos se clasificaron automáticamente en las siete categorías registradas; la muestra estratificada de cien fechas sin reemplazo por modelo conserva pesos de inclusión en el ledger privado. Esto acredita análisis cuantitativo P0 y disponibilidad de cien casos auditables, **no cien explicaciones causales escritas ni el análisis de los tres mejores finales**. La tabla usa como denominador las 305 fechas completas, no la muestra estratificada. Las categorías y sus IC nominales por bloques de catorce días, con 2.000 remuestreos, están en el [resumen agregado](../_static/remediation_error_results/error_summary.json).

Las categorías sin casos tienen frecuencia observada cero; un intervalo bootstrap degenerado [0,0] solo refleja esta muestra y no acredita probabilidad poblacional nula.

Los errores grandes de las tres recetas neuronales candidatas de P0 se concentran descriptivamente en subestimación consistente, aunque el patrón también aparece en GRU. El ensamble exploratorio de pesos iguales obtuvo RMSE **0,00569964**, frente a **0,00478278** del mejor miembro medio entre semillas en el mismo OOF: su ganancia fue **−0,00091685**. Estos son estimandos de ensamble, distintos de la media primaria de RMSE por semilla, y **no respaldan una mejora por combinar los cuatro modelos**.

Se publican la [taxonomía y su procedencia](../_static/remediation_error_results/training_taxonomy.json) y el [recibo de análisis](../_static/remediation_error_results/analysis_receipt.json). Los casos individuales se conservan privados. Los cuatro modelos P0 no se renombran los tres mejores del benchmark ni sus errores se convierten en intervalos predictivos calibrados.

### 12.3.3 Reproducción de la evidencia histórica

Los comandos siguientes se conservan para la reproducción del **procedimiento P0 anterior**:

El módulo histórico `scripts/e2_predictive_uncertainty.py` contiene validadores de soportes, calibración ponderada, métricas predictivas, remuestreo por bloques y análisis OOF. El comando por defecto solo describe el plan; no carga datos ni escribe resultados:

```bash
python scripts/e2_predictive_uncertainty.py
python tests/test_e2_predictive_uncertainty.py
```

Las funciones numéricas usan NumPy; los tests trabajan con fixtures sintéticos. Su éxito verifica software y restricciones del procedimiento, no acredita cobertura ni cien errores nuevos de cada modelo. Cualquier evaluación nueva requiere las predicciones y los recibos causalmente admisibles indicados en el plan.

Los OOF P0 registrados ya existen; la entrada explícita siguiente permite reproducir el análisis de su CSV y una taxonomía aprendida solo en entrenamiento; los identificadores deben coincidir con la cohorte efectiva, no con una selección posterior de ganadores:

```bash
python scripts/e2_predictive_uncertainty.py --oof-csv RUTA_PRIVADA/oof_predictions.csv --taxonomy-json RUTA_PRIVADA/training_taxonomy.json --models MODELOS_REGISTRADOS --seeds SEMILLAS_REGISTRADAS --private-output ../e2_private/error_analysis
```

La CLI exige los hashes y timestamps de checkpoints presentes en el CSV y su cierre antes del primer scoring, y escribe solo en una carpeta privada nueva fuera del repositorio. Produce un resumen agregado, un ledger privado de cien casos por modelo y un recibo de hashes; no vuelve a entrenar. Los hashes identifican insumos, pero no prueban por sí solos que sus declaraciones científicas sean verdaderas. El modo actual admite P0 diagnóstico; no lo presenta como un benchmark final ni decide los tres mejores modelos.
