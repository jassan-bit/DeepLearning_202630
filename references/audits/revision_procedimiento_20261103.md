# Revisión de la guía de procedimiento — 3 de noviembre de 2026

Se contrastaron las rutas 6.1–6.11 de la guía. Para el contrato actual se aplica únicamente 6.4: regresión temporal, tipo 4. La tabla de once rutas conserva la numeración original, cuya lista inicial agrupa dos tareas bajo el ítem 10. La revisión consolida los capítulos 2–5 sin ejecutar modelos, estadísticas empíricas, datos de mercado o test.

## Aplicabilidad

| Ruta | Tarea | Principal para este proyecto | Motivo |
|---|---|---|---|
| 6.1 | Clasificación tabular | No | Objetivo continuo sin clases. |
| 6.2 | Regresión tabular | No | No es la ruta principal: covariables numéricas con reloj causal y dependencia temporal. |
| 6.3 | Series de tiempo | No | No es la ruta principal del contrato sin rezagos explícitos del objetivo; se revisaría si se añaden. |
| 6.4 | Regresión temporal | Sí | Ventanas OHLCV rezagadas y objetivo continuo, con disponibilidad y partición cronológica. |
| 6.5 | Regresión espacial | No | Sin coordenadas o unidades espaciales de predicción. |
| 6.6 | Regresión espacio-temporal | No | Sin ubicaciones múltiples o estructura espacial. |
| 6.7 | Clasificación de imágenes | No | No hay imágenes de entrada ni clases visuales. |
| 6.8 | Segmentación de imágenes | No | No hay máscaras ni objetivos por píxel. |
| 6.9 | Clasificación de texto | No | No hay documentos o etiquetas textuales en la entrada. |
| 6.10 | Etiquetado secuencial | No | No se asigna una etiqueta por elemento de la entrada. |
| 6.11 | Aprendizaje secuencia a secuencia | No | La salida actual es una volatilidad escalar a un día. |

## Requisitos de la ruta y coherencia

| ID | Requisito | Estado | Evidencia o corrección | Límite o pendiente |
|---|---|---|---|---|
| P01 | Seguir únicamente la ruta pertinente | Aplicabilidad explicitada | Matriz de once rutas, únicamente guía 6.4 principal; tipo admitido 4. | No reproduce bloques de modalidades ajenas ni cambia la lista inicial de diez ítems. |
| P02 | Numeración de guía frente a libro | Correspondencia conservada | Guía 6.4 se aplica dentro del §6.1 del libro; se preservan §6.1.1–6.1.6,6.2 y 6.3. | La guía agrupa dos tareas bajo su último ítem de admisión; once rutas no son una nueva clasificación. |
| P03 | Justificación del tipo de problema | Contrato precisado | OHLCV rezagadas y salida continua; LSTM/Transformer o un tensor no convierten la tarea en seq2seq. | Ausencia de rezagos explícitos de σ no implica independencia o exogeneidad de las covariables. |
| P04 | Partición cronológica estricta | Regla y evidencia existentes | 2023 como historia inicial; 2024 para selección; 01/01/2025–31/08/2026 reserva de test. | Etiquetas pasadas solo se incorporan cuando son observables y permitidas; no hay evaluación de test. |
| P05 | Embargo igual a máximo rezago | Convención explicitada | D días de resultado omitidos [t−D,t−1]; candidatos u≤t−D−1, con calidad y disponibilidad. | La separación entre resultados es D+1; no se confunde con D−1 días interiores. |
| P06 | Índices frente a entrenamiento y cadencia | Alcance delimitado | Manifiesto v2: 5.856 filas de 16 configuraciones por 366 orígenes; ejemplo de frontera 01/01/2024 con D=7. | No son corridas ni una decisión de reajuste diario; no se regeneraron índices. |
| P07 | Construcción de rezagos de covariables | Regla vigente conservada | X conserva D×5 OHLCV de días anteriores; no se añaden columnas de volatilidad objetivo. | Cambiar información o añadir σ requiere nuevo contrato y revisar ruta antes de ejecutarlo. |
| P08 | Disponibilidad de cada atributo | Reloj precisado | Resultado d, origen al inicio UTC de d y etiqueta al terminar d, equivalente a notación Y_t=σ_(t+1). | OHLCV completo de d no está disponible al inicio de d; n7 no son siete días enteramente futuros. |
| P09 | Escalado solo con pasado | Subentrenamiento delimitado | Transformaciones de X/Y y selección/PCA solo en subentrenamiento, no en unión con validación de parada. | Regla, tamaño interno, escaladores y parámetros operativos aún por cerrar. |
| P10 | Embargo interno y parada | Regla sincronizada con capítulo 4 | Para origen t, validación interna termina en ≤t−D−1; si empieza en v, fit usa u≤v−D−1. | La etiqueta externa no detiene su propio ajuste; no se fijan fechas internas arbitrarias. |
| P11 | Validación de selección frente a evaluación externa | Alcances separados | 2024 usado para elegir hiperparámetros/D se identifica como selección; evaluación externa exige anidación de todo selector. | Una parada causal no elimina por sí sola el optimismo de seleccionar con todo el año. |
| P12 | Calidad y faltantes | Política conservada | 24/03/2023 excluido, 80 minutos ausentes y un cierre no estándar documentados; calendario intacto. | Sin imputación nueva ni compresión de fechas; no se abrieron brutos o datos de mercado. |
| P13 | Ventanas de entrada, objetivo y horizonte | Decisiones preservadas | n=7 y h=1 principales; D=7/14/21/28 sin D principal; n=14/21/28 como sensibilidad. | No ranking entre RMSE de objetivos distintos ni horizontes nuevos por el ejemplo sintético. |
| P14 | Familias base típicas | Orientación frente a evidencia | Guía: MLP, LSTM/GRU y CNN1D; cinco textos: LSTM 4/5, GRU 2/5, TCN 1/5 y encoder 2/5. | Trío final abierto; TCN no equivale a CNN pequeña y atención requiere respaldo, no selección automática. |
| P15 | Pérdida cuadrática, absoluta o Huber | Opción adoptada identificada | Se conserva MSE con RMSE primaria, salida escalar no negativa. | Las opciones de guía no obligan a probarlas todas; escala e implementación pendientes. |
| P16 | Evaluación de regresión | Reglas sincronizadas | RMSE/MAE/R² y complementarias MSE/MAPE/MASE; RMSE agrupada por corrida y media entre semillas para ranking. | No RMSE de semillas×fechas o ensemble como sustituto; métricas propias aún no medidas. |
| P17 | Cobertura y casos indefinidos | Síntesis de métricas precisada | MAPE positivos; MASE con I_M, N_M y cobertura con escala train positiva; R² unitario indefinido. | No ε arbitrario ni MAE global dividida por escala elegida después. |
| P18 | Referencia lineal regularizada con mismas covariables | Evidencia existente delimitada | Cuatro Ridge con α=1 y 5D entradas, ajuste de 2023: 343/322/301/280 muestras; 366 predicciones de 2024 por D, 1.464 en total. | Son diagnósticos de EDA; no fijan penalización, biblioteca o cadencia del benchmark futuro. |
| P19 | Reconsiderar tipo ante alta autocorrelación | Decisión explícita | Revisión del 3 de noviembre: se mantiene tipo 4 con el contrato OHLCV, reconociendo dependencia residual y causas no identificadas. | El diagnóstico no exige automáticamente rezagos de σ ni demuestra mejora de una red avanzada. |
| P20 | Dependencia del objetivo móvil | Soporte registrado | Si n>D+1, la última etiqueta admisible y validación comparten n−D−1 retornos; 2.196 filas de seis sensibilidades. | Son retornos históricos compartidos, no consulta del retorno futuro; no establecen independencia. |
| P21 | DM, Ljung–Box y contrastes pareados | Unidad y estimando precisados | DM con pérdida cuadrática por fecha, h real 1, HAC y HLN; t/Wilcoxon por corrida y Friedman con bloques completos. | DM no contrasta directamente la media de RMSE entre semillas; h=1 no impone HAC de rezago cero ni σ7 un máximo de seis; métodos pendientes de ejecutar. |
| P22 | BDS y evidencia residual | Atribución preservada | BDS procede de 5.4 de la guía/material específico; 112 ACF y 12 Ljung–Box/Holm ya calculados para cuatro Ridge. | Los doce rechazan bajo referencia asintótica exploratoria; no son resultados de redes ni BDS ejecutado. |
| P23 | Margen, intervalos, baseline y periodos | Cierre con capítulos4–5 | 5% solo frente a persistencia; potencia/MDE y pregunta separadas; baseline con desarrollo antes de test; periodos separados. | Agregado adicional exige regla y predicciones, no promedio de RMSE anuales; no existe baseline DL medido. |
| P24 | Proyección, fases y estado de ejecución | Secuencia explicitada | Diseño/selección/piloto/dimensionamiento/congelación/evaluación con sus dependencias y campos abiertos. | R no se calcula antes de piloto; cambio de procedimiento exige revisar dispersión; mención de diez modelos de E2 no aporta sus guías o configuraciones. |

La numeración interna del libro permanece 6.1.1–6.1.6, 6.2 y 6.3, con una correspondencia explícita a la guía 6.4. La reconsideración ante autocorrelación queda documentada; no se añadieron rezagos de σ, no se cambió el dataset y no se recalcularon las cuatro Ridge. La falta de selección definitiva, piloto, benchmark y test sigue visible.
