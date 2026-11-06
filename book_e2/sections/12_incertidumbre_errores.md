# 12. Incertidumbre predictiva y análisis de errores

**Estado: análisis P0 de 305 fechas por receta ejecutado; incertidumbre predictiva de los tres mejores finales pendiente.** Esta sección desarrolla §12.2–12.3 de la guía completa (páginas PDF 49–50). Los intervalos predictivos describen valores futuros de σ₇; los IC de RMSE describen incertidumbre de una métrica. Ni la dispersión entre semillas ni un IC de RMSE se presentan como un intervalo predictivo.

El [plan versionado](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e2_remediation_20261106/uncertainty_errors.json) distingue una rama secundaria de calibración de la receta primaria. No modifica la entrada D7×5, el objetivo, la partición principal, la RMSE ni el margen absoluto heredado. Los 305 orígenes de desarrollo de enero–octubre de 2024 ya tienen errores P0 analizados para las cuatro recetas registradas; no son resultados de las recetas finales seleccionadas por HPO. La referencia Ridge del EDA tampoco se convierte en resultados del benchmark E2.

## 12.2 Cuantificación de incertidumbre

### 12.2.1 Predictor congelado y calibración disjunta

El ajuste principal puede usar resultados hasta el 24/12/2024. Sus errores de noviembre–diciembre son errores de entrenamiento o de un predictor diferente: **no sirven para calibrarlo como si fueran residuos disjuntos**. Para evitar esa sustitución, se define la **rama A**:

| Operación | Regla de la rama A |
|---|---|
| Selección de candidatos | Tres modelos/recetas por desarrollo temporal anidado, sobre una misma cohorte cuyo último resultado sea **24/10/2024**. Es la selección de esta rama; no reemplaza el ranking primario enero–octubre hasta el día 31 |
| Ajuste y escalado | Resultados admitidos hasta **24/10/2024**, origen nominal **01/11/2024**. Hiperparámetros, selección de modelos y épocas deben respetar ese mismo corte |
| Embargo de resultados | **25–31/10/2024**, ambos inclusive |
| Calibración | Sesenta orígenes **01/11–30/12/2024**; sin reentrenar, seleccionar pesos ni cambiar escaladores con sus etiquetas |
| Predictor puntual | Media de al menos cinco miembros con semillas y checkpoints fijados antes del scoring de calibración; una referencia determinista se almacena una sola vez |
| Evaluación secundaria posterior | Las mismas sesenta fechas conocidas **01/09–30/10/2026**, solo cuando la rama y sus insumos estén cerrados. Sigue siendo evaluación histórica conocida |

La primera etiqueta de calibración σ₇(01/11) utiliza cierres del 25/10 al 01/11. Sus cierres no se superponen con los de las etiquetas de ajuste, cuyo máximo es el 24/10. Las entradas de calibración pueden usar las covariables observables de los días de embargo; el embargo no borra esa historia disponible. Las etiquetas de calibración posteriores sí se solapan entre sí, por lo que su independencia no se supone.

Las fechas de origen son las del replay histórico. Los timestamps de cierre real de pesos y comienzo de scoring se registrarán en UTC cuando se ejecute el procedimiento; no se retrofechan a 2024. El verificador exige que todos los miembros cierren antes del scoring real, que las filas provengan de sus mismos `run_id` y que la identidad del predictor evaluado coincida con la del calibrado. Cambiar al ajuste primario entrenado hasta diciembre exige otra calibración: no se reutilizan estos radios.

Los modelos representativos P0 no se declaran los tres mejores finales. Sus posibles predicciones OOF permiten diagnósticos de desarrollo, pero la selección definitiva de la rama A requiere la cohorte de candidatos completa y comparable. Si los tres elegidos aquí difieren de los del ranking primario, se publicará la diferencia y la incertidumbre de los tres principales seguirá pendiente hasta disponer de una rama válida para ellos.

### 12.2.2 Intervalos temporales y evaluación

Para el mismo predictor congelado A, se calculan sesenta errores absolutos de calibración $s_i=|y_i-\widehat y_i^A|$. Se fijan pesos de recencia $w_i=2^{-d_i/28}$, donde $d_i$ son los días entre el origen de calibración y el 30/12/2024. Con masa de predicción fijada en uno, el radio es el menor residuo ordenado cuya masa acumulada alcanza $(1-\alpha)(\sum_i w_i+1)$.

Se proponen coberturas nominales **80 %, 90 % y 95 %** e intervalos $[\max(0,\widehat y^A-q_\alpha),\widehat y^A+q_\alpha]$. Si la masa no permite un radio finito, se informa **no estimable**; no se inventa un cero ni una cota infinita presentada como un intervalo útil.

Esta es una calibración residual temporal con ponderación declarada, inspirada en el enfoque conforme; **no se afirma cobertura exacta de muestra finita**. Los pesos no son cocientes de densidades estimados ni prueban intercambiabilidad. La dependencia de σ₇, la selección de modelos, solo sesenta observaciones y la brecha entre 2024 y 2026 limitan la transferencia. Los pesos permanecen referidos al final de calibración; no se simula una ventana online actualizada con etiquetas de test.

Se reportarán cobertura empírica frente a nominal, anchura media, score de intervalo y pérdidas pinball de las cotas. WIS combina los scores de los intervalos con el error absoluto del valor presentado como pronóstico del cuantil 0,5; no se afirma que ese valor sea la verdadera mediana condicional. **CRPS solo se calcula si se declara una distribución predictiva**: el código admite una CDF empírica de residuos firmados de calibración desplazados por el predictor congelado, con truncamiento explícito en cero y pesos documentados. No transforma la variabilidad de cinco optimizaciones en una distribución de futuros precios o volatilidades.

La cobertura condicional se evaluará por grupos causales y rangos de predicción con umbrales aprendidos en entrenamiento antes de calibración. Un grupo sin casos no recibe cobertura cero: se registra no estimable. El remuestreo mantiene el calendario completo y filtra la pertenencia al grupo después de extraer bloques; no concatena fechas separadas como si fueran vecinas.

Los IC descriptivos emplean bootstrap circular por bloques diarios, al menos **2.000** remuestreos, semilla 20261106, bloque principal **14 días** y sensibilidades **7/28**. Sesenta fechas contienen pocos bloques efectivos y no aseguran estacionariedad. Los IC nominales no son intervalos simultáneos ni gastan por sí solos el presupuesto de contrastes global **0,02 + 0,02 + 0,01**. El α de cobertura predictiva tampoco es el α de esas familias ni el margen δ de RMSE.

Para interpretar el coste de separar la calibración, la evaluación publicará lado a lado el error puntual de A y el del predictor primario comparable sobre las mismas fechas. Una comparación de medias de pronósticos se identificará como tal; no sustituirá la RMSE primaria promediada entre trayectorias de semillas.

## 12.3 Casos de error: unidad, taxonomía y muestreo

El diagnóstico requiere un panel OOF de **305 orígenes diarios**, del 01/01 al 31/10/2024, conservando posiciones no elegibles vacías y la misma elegibilidad entre modelos. Se conserva el ajuste externo **mensual**: resultados usados por ajuste, escaladores y selección de configuración terminan como máximo ocho días antes del primer origen de cada mes. El mismo `run_id`, checkpoint y soporte se mantienen durante ese fold; no se introduce un reajuste diario solo por cumplir $t-8$. Se exigen **al menos cien orígenes elegibles únicos por modelo**. Cinco semillas sobre cien fechas siguen siendo cien casos; no quinientos.

Las categorías se fijan con los cuantiles 0,50 y 0,90 del error absoluto de una referencia de entrenamiento identificada, con al menos veinte fechas distintas y resultados hasta el **24/12/2023**. No se calculan esos umbrales a partir de los errores de 2024 que se analizarán ni del test. El margen δ de mejora de RMSE no se convierte en una tolerancia arbitraria por caso.

| Magnitud del caso | Categorías mutuamente excluyentes |
|---|---|
| MAE entre semillas ≤ q₅₀ de referencia | Error bajo respecto a la referencia |
| q₅₀ < MAE entre semillas ≤ q₉₀ | Subestimación consistente; sobreestimación consistente; signos mezclados entre semillas |
| MAE entre semillas > q₉₀ | Subestimación consistente; sobreestimación consistente; signos mezclados entre semillas |

La dirección consistente exige que todas las semillas tengan el mismo signo estricto; las demás combinaciones pertenecen a «mezclados». Así se evita que errores grandes de signos opuestos desaparezcan al promediar pronósticos. Los empates y el error cero pertenecen a una única categoría.

Se clasifican todos los orígenes elegibles y se conserva una muestra de auditoría de **cien casos sin reemplazo**, estratificada por categoría y trimestre calendario, con semilla 20261106. Se guarda la probabilidad de inclusión. Las frecuencias publicadas se calculan sobre todos los casos elegibles, no como porcentajes sin ponderar de esa muestra estratificada. Los casos individuales y sus predicciones se conservarán en la evidencia privada.

Los agregados previstos incluyen frecuencias por categoría con IC temporales, errores grandes compartidos, Jaccard por pares y correlación de errores firmados. El ensamble de pesos iguales es un diagnóstico de complementariedad; su mejora observada respecto al mejor miembro en el mismo OOF no prueba una ganancia futura y conserva el sesgo de esa referencia elegida.

El módulo también identifica diez orígenes de mayor pérdida por modelo para la revisión privada. No se les asigna «menor confianza» usando intervalos calibrados con noviembre–diciembre posteriores a esos OOF: eso requeriría una calibración causal propia por origen. La función `public_error_summary` elimina los listados privados de auditoría y frontera antes de publicar los agregados.

Los OOF P0 de recetas fijas se identificarán como **diagnóstico**, sin sustituir una comparación final anidada, la selección de los mejores modelos o la evaluación del benchmark. Los casos reales P0 ya se analizaron; las coberturas y los casos de los modelos principales finales siguen pendientes. Las pruebas sintéticas y los resultados E1/ETTh1 no se presentan como su ejecución.

### 12.3.2 Análisis P0 ejecutado

Se analizaron **305 fechas únicas por cada una de las cuatro recetas**, conservando cinco semillas y los diez checkpoints mensuales por receta. La taxonomía se fijó a las **15:14:04 UTC**, antes del primer scoring OOF, usando los **341 errores de persistencia** admisibles de 09/01–24/12/2023. Sus umbrales son **q₅₀=0,00101245** y **q₉₀=0,00683008**; no se estimaron con errores de 2024 ni cambiando δ.

| Receta P0 | Fechas únicas clasificadas | Casos únicos en muestra privada | Errores grandes, total | Subestimaciones grandes consistentes |
|---|---:|---:|---:|---:|
| TimeMixer | 305 | 100 | 68 (22,30 %) | 61 |
| iTransformer | 305 | 100 | 73 (23,93 %) | 61 |
| PatchTST | 305 | 100 | 54 (17,70 %) | 49 |
| GRU | 305 | 100 | 38 (12,46 %) | 23 |

Los casos se clasificaron automáticamente en las siete categorías registradas; la muestra estratificada de cien fechas sin reemplazo por modelo conserva pesos de inclusión en el ledger privado. Esto acredita análisis cuantitativo P0 y disponibilidad de cien casos auditables, **no cien explicaciones causales escritas ni el análisis de los tres mejores finales**. La tabla usa como denominador las 305 fechas completas, no la muestra estratificada. Las categorías y sus IC nominales por bloques de catorce días, con 2.000 remuestreos, están en el [resumen agregado](../_static/remediation_error_results/error_summary.json).

Las categorías sin casos tienen frecuencia observada cero; un intervalo bootstrap degenerado [0,0] solo refleja esta muestra y no acredita probabilidad poblacional nula.

Los errores grandes de las tres recetas Top10 se concentran descriptivamente en subestimación consistente, aunque el patrón también aparece en GRU. El ensamble exploratorio de pesos iguales obtuvo RMSE **0,00569964**, frente a **0,00478278** del mejor miembro medio entre semillas en el mismo OOF: su ganancia fue **−0,00091685**. Estos son estimandos de ensamble, distintos de la media primaria de RMSE por semilla, y **no respaldan una mejora por combinar los cuatro modelos**.

Se publican la [taxonomía y su procedencia](../_static/remediation_error_results/training_taxonomy.json) y el [recibo de análisis](../_static/remediation_error_results/analysis_receipt.json). Los casos individuales se conservan privados. Los cuatro modelos P0 no se renombran los tres mejores del benchmark ni sus errores se convierten en intervalos predictivos calibrados.

### 12.3.3 Reproducción del procedimiento

El módulo propio `scripts/e2_predictive_uncertainty.py` contiene validadores de soportes, calibración ponderada, métricas predictivas, remuestreo por bloques y análisis OOF. El comando por defecto solo describe el plan; no carga datos ni escribe resultados:

```bash
python scripts/e2_predictive_uncertainty.py
python tests/test_e2_predictive_uncertainty.py
```

Las funciones numéricas usan NumPy; los tests trabajan con fixtures sintéticos. Su éxito verifica software y restricciones del procedimiento, no acredita cobertura ni cien errores nuevos de cada modelo. La evaluación efectiva requiere las predicciones y los recibos causalmente admisibles indicados en el plan.

Cuando existan los OOF P0 registrados, la entrada explícita siguiente permite analizar su CSV y una taxonomía aprendida solo en entrenamiento; los identificadores deben coincidir con la cohorte efectiva, no con una selección posterior de ganadores:

```bash
python scripts/e2_predictive_uncertainty.py --oof-csv RUTA_PRIVADA/oof_predictions.csv --taxonomy-json RUTA_PRIVADA/training_taxonomy.json --models MODELOS_REGISTRADOS --seeds SEMILLAS_REGISTRADAS --private-output ../e2_private/error_analysis
```

La CLI exige los hashes y timestamps de checkpoints presentes en el CSV y su cierre antes del primer scoring, y escribe solo en una carpeta privada nueva fuera del repositorio. Produce un resumen agregado, un ledger privado de cien casos por modelo y un recibo de hashes; no vuelve a entrenar. Los hashes identifican insumos, pero no prueban por sí solos que sus declaraciones científicas sean verdaderas. El modo actual admite P0 diagnóstico; no lo presenta como un benchmark final ni decide los tres mejores modelos.
