# Decisiones, revisión y pendientes de E1

El procedimiento experimental utiliza **validación forward chaining de Filipe** con MLP, LSTM y CNN1D poco profunda. Los índices reales de `timeseries-cv` alimentan entrenamiento, escaladores y evaluación de doce bloques mensuales de 2024. El ciclo registró **cero archivos de test abiertos**; la adquisición previa revisó timestamps, esquema y conteos de los brutos, sin usar sus objetivos para seleccionar modelos. El cumplimiento global de la guía no se declara cerrado.

## Decisiones incorporadas

| Elemento | Criterio o evidencia vigente |
|---|---|
| Activo y fuente | Bitcoin, Binance Spot BTCUSDT, archivos oficiales de un minuto |
| Modelado | Diario UTC, cinco covariables OHLCV |
| Categoría literal de la guía | Tipo 4: regresión temporal, ruta 6.4; sin rezagos explícitos de volatilidad como entrada |
| Historia de entrada | D=7 común para las tres redes; D14/21/28 no se evalúan en esta CV |
| Objetivo principal | Desviación poblacional de n=7 retornos logarítmicos diarios; sin anualizar |
| Horizonte | Un día; próximo valor de volatilidad histórica móvil, no siete días enteramente futuros |
| Sensibilidad del objetivo | n=14/21/28 conservados como sensibilidad; no entrenados en este ciclo |
| Pérdida y métrica | MSE y RMSE principal; MAE/MSE/MAPE/R²/MASE complementarias |
| Relevancia operativa | Reducción relativa mínima del 5 % de RMSE frente a persistencia de las mismas fechas; la mejora puntual de CNN1D relativa en julio–diciembre es aproximadamente 3,10 %, inferior al margen |
| Política de MAPE | Solo objetivos positivos; cobertura y ceros excluidos; sin epsilon arbitrario |
| Política de MASE | Rezago de un día de calendario; denominador del entrenamiento mensual permitido, sin comprimir huecos; no definido si no hay escala positiva |
| Independencia | Una serie BTCUSDT; inferencia por semillas y por fechas diferenciada, con ventanas solapadas y modelos mensuales; no son mercados nuevos |
| Comparaciones por semillas | Familia global de doce hipótesis con Holm-12; las familias componentes de nueve y tres se conservan como trazabilidad |
| Inferencia temporal | DM principal HAC14/HLNh1 con Holm-6 y sensibilidad 7/28; bootstrap circular de fechas con 2.000 réplicas, L14 y sensibilidad 7/28 |
| Residuos y superioridad | Semilla 4101, quince diagnósticos Ljung–Box/BDS con Holm; fracción de victorias por semillas e IC condicional |
| Bloques de validación | Doce bloques mensuales con ajuste real; no son bloques bootstrap ni repeticiones independientes |
| Faltantes | Exclusión estricta del 24/03/2023 y ventanas afectadas, sin interpolar ni comprimir calendario |
| Embargo | D días omitidos antes del origen t; candidatos de ajuste u≤t−D−1, además de calidad y disponibilidad |
| Calendario ejecutado | Entrenamiento inicial 2023; forward chaining 2024. Cada mes se reajusta con pasado admisible, sin usar el mes evaluado para entrenarlo |
| Ajuste final y calendario reservado | Ajuste final previsto con desarrollo 2023–2024 una vez cerradas las configuraciones; todavía no ejecutado. Test enero de 2025–agosto de 2026; septiembre–octubre de 2026 permanece sin asignación automática |
| Reserva bruta de test | Veinte ZIP mensuales del 01/01/2025 al 31/08/2026 identificados por metadatos y hashes; muestras supervisadas y protocolo final pendientes |
| Referencia diagnóstica del bloque C | Cuatro Ridge con intercepto libre y α=1 fijo; 366 predicciones de 2024 por D, sin tuning; diagnóstico de EDA separado del benchmark DL |
| Familias implementadas | MLP de ancho 16 y 593 parámetros; LSTM unidireccional de ancho 32 y 5.025 parámetros; CNN1D causal de dos capas, ancho 16 y 1.057 parámetros |
| Referencia DL de desarrollo | MLP con precios relativos obtiene el menor RMSE anual observado entre las redes. El cierre del pipeline final permanece pendiente |
| Referencia contextual | Persistencia conserva menor RMSE anual y R²=0,759571; no es un cuarto modelo DL |
| CV de Filipe ejecutable | `timeseries-cv` 0.1.5: `split_train_val_forwardChaining(np.arange(731),7,1,1)`; orígenes diarios agrupados en doce bloques mensuales, con índices efectivos de train/inner/validación suministrados a las redes |
| Representación principal y control | Precios relativos al último cierre de la propia ventana, con escalado absoluto como control bajo la misma CV; D7/H/LR/modelos iguales y diez semillas 4101–4110 emparejadas entre variantes |
| Fichas de los modelos implementados | MLP, LSTM y CNN1D en §3.3, con nueve campos por modelo; los perfiles fuente complementarios se archivan por separado y no completan el respaldo aplicado pendiente |

El margen del 5 % es un criterio operativo frente a persistencia, no utilidad financiera calibrada ni margen entre redes, de equivalencia o para modelos avanzados. La sensibilidad del diseño se calculó retrospectivamente con dispersión de las diez semillas; no acredita un piloto previo ni potencia 0,80. El piloto prospectivo y el dimensionamiento definitivo siguen pendientes. El [protocolo de CV](../_static/dl_r2_cv_20261104/method.json) identifica las decisiones de ejecución y sus límites.

## Revisión integral contra la guía

La revisión mantiene separados los requisitos documentales, la evidencia ejecutada y los productos pendientes. Los resultados de CV se apoyan en índices, configuraciones y predicciones verificables.

| Punto revisado | Evidencia actual o límite |
|---|---|
| Definición y categoría | Tipo 4, entradas OHLCV, salida continua, MSE y RMSE explícitos |
| Ventanas y temporalidad | D7/n7/h1 diferenciados; pronóstico diario y evaluación histórica |
| Independencia y margen | Una serie; diez semillas condicionadas al calendario; margen del 5 % limitado a persistencia |
| Particiones | Doce bloques mensuales reales de Filipe, embargo de siete días y validación interna causal; no son 366 reajustes |
| Datos y trazabilidad | CSV de desarrollo y hashes conservados; reserva bruta de veinte ZIP de test intacta |
| EDA | Perfiles, calidad, etiquetas, desplazamiento marginal y bloque C con 1.464 residuos Ridge; mecanismo de faltantes y ruido externo no identificados |
| Literatura | Treinta y cinco referencias, cinco textos aplicados exploratorios y 81 cifras de autores; respaldo de MLP/CNN1D simple todavía 0/5 en el corpus |
| Implementación | **60 trayectorias, 720 segmentos modelo–mes y 1.440 ajustes efectivos**, con 21.960 predicciones; [resumen de ejecución](../_static/dl_r2_cv_20261104/execution_summary.json) |
| Comparación e inferencia | Diez semillas por red/variante; familia global de doce contrastes con Holm-12, Friedman, DM y bootstrap por fechas, con periodos y estimandos identificados |
| Sensibilidad y potencia | MDE frente a cero y R orientativo calculados por dispersión retrospectiva; no son piloto previo ni potencia certificada y los folds no reemplazan semillas |

El cumplimiento global del Entregable 1 **no se declara cerrado**. Ejecutar las redes no completa el respaldo bibliográfico, la potencia requerida, la evaluación reservada ni el margen para modelos avanzados.

El [EDA general](../_static/e1/revision_eda_general_20261103.json), el [EDA específico](../_static/eda_temporal_specific/summary.json) y la [referencia Ridge](../_static/eda_temporal_reference/summary.json) mantienen su alcance exploratorio. Los doce diagnósticos Ljung–Box/Holm del módulo Ridge no corresponden a los residuos de las redes. La dependencia residual no añade automáticamente rezagos de volatilidad ni cambia el tipo 4.

El procedimiento tiene [método de CV](../_static/dl_r2_cv_20261104/method.json), [índices efectivos](../_static/dl_r2_cv_20261104/folds.json), [cortes mensuales](../_static/dl_r2_cv_20261104/folds.csv), [corridas](../_static/dl_r2_cv_20261104/runs.csv) y [resumen de ejecución](../_static/dl_r2_cv_20261104/execution_summary.json). El código se registró antes de los ajustes CV en `1f0ed23`.

El [complemento estadístico](../_static/filipe_statistics_20261104/summary.json) se registró mediante una [enmienda](../_static/filipe_statistics_20261104/protocol.json) después de examinar los descriptivos y antes de calcularlo. Reutiliza exclusivamente las predicciones guardadas, sin nuevos ajustes o lectura de test. La familia global Holm-12 integra los nueve contrastes del complemento y tres controles de representación; los IC por semillas, DM, diagnósticos y bootstrap responden preguntas distintas y no son confirmaciones independientes.

La representación principal usa `log(OHLC)−log(último cierre observado de la propia ventana)`, volumen con `log1p`, escalas aprendidas solo con train y salida en unidades originales. La comparación controlada enfrenta esa representación con el escalado absoluto bajo los mismos folds, configuraciones y semillas. Las configuraciones se eligieron con desarrollo ya observado, incluido enero–junio de 2024: la CV es exploratoria y no una validación anidada de toda la selección.

## Estado de los siete cambios solicitados

| Prioridad | Producto | Estado verificable |
|---|---|---|
| 1 | Fuentes y búsqueda trazable | Tabla de las 35 referencias, enlaces y nivel de evidencia; consultas públicas e historial de acceso conservados |
| 2 | Cribado y flujo | 118 ocurrencias, doce duplicadas, 106 repositorios, veinte inspeccionados, siete textos evaluados y cinco incluidos |
| 3 | Matriz de literatura | Cinco textos, 81 cifras publicadas y evaluación de calidad con ubicaciones; resultados de autores, no propios |
| 4 | Tres modelos básicos | MLP–LSTM–CNN1D implementados con configuraciones de desarrollo; la extracción aplicada MLP/CNN1D simple sigue incompleta |
| 5 | Fichas técnicas | Veintisiete campos para las tres implementaciones propias en §3.3; referencias fundacionales separadas de los apoyos aplicados independientes que aún faltan para MLP/CNN1D simple |
| 6 | Comparación crítica | Diferencias de objetivo/mercado/escala en literatura y resultados propios de CV con límites separados |
| 7 | Definición y coherencia | n7/h1/D7, RMSE, 5 % frente a persistencia, MAPE/MASE, ruta 6.4 y procedencia sincronizados |

Las frecuencias sobre cinco textos completos se conservan: familia LSTM 4/5 incluyendo BiLSTM, unidireccional 3/5, GRU 2/5, TCN 1/5 y encoder Transformer 2/5. MLP independiente y CNN1D simple no tienen apoyo acreditado en ese corpus. La elección curricular y su implementación no se presentan como los tres modelos más frecuentes de toda la literatura ni como complejidad mínima absoluta.

## Límites que permanecen abiertos

- **Cobertura académica:** las doce consultas públicas ejecutadas y la verificación posterior no acreditan búsquedas internas en WoS/Scopus, indexación individual completa ni cuartil Q1. La recuperación es acotada y falta respaldar MLP/CNN1D simple.
- **Perfiles suplementarios de atención:** dos apoyos son manuscritos sin revisión por pares acreditada y uno carece de fecha editorial corroborada. Son límites de esos antecedentes, separados del trío implementado.
- **Extracción de los tres básicos:** la guía pide recurrencia, sencillez y diversidad; el corpus actual no cierra esos criterios conjuntamente para el trío curricular.
- **EDA y referencia contextual:** el diagnóstico Ridge está ejecutado; no se ha incorporado como otro competidor bajo el protocolo DL. El mecanismo de faltantes y el ruido externo de etiquetas siguen sin identificarse.
- **Configuraciones de fuentes:** permanecen límites documentales de `keras-tcn` y S6, sin atribuirlos a las implementaciones PyTorch propias. Los ganadores de S1 están recuperados.
- **Potencia:** se calculó sensibilidad retrospectiva MDE/R, con R orientativo 19/9/28 frente a persistencia bajo la aproximación registrada. No sustituye un piloto previo ni certifica potencia 0,80; la alternativa para demostrar ganancia superior al 5 % y el dimensionamiento definitivo siguen abiertos.
- **Generalización:** 2024 fue expuesto al EDA y a decisiones de configuración. Cada fold respeta el tiempo, pero el resumen no constituye validación anidada de toda la selección ni confirmación independiente. El ajuste final con desarrollo y el test supervisado enero de 2025–agosto de 2026 permanecen pendientes.
- **Alcance de la inferencia:** DM, BDS, Ljung–Box y bootstrap se calcularon, con sus supuestos y correcciones. Los cambios de régimen, el solapamiento y la actualización mensual limitan sus referencias; ningún resultado convierte la CV expuesta en confirmación externa. La familia global de doce comparaciones no demuestra diferencias entre las tres redes a α=0,05.
- **Baseline y margen avanzado:** MLP relativa es la red de menor RMSE anual observada; persistencia mantiene mejor desempeño anual. El pipeline final, el margen de equivalencia y la mejora exigible a modelos avanzados siguen abiertos.

## Proyección a E2

Las futuras etapas heredan código, configuraciones, resultados y limitaciones mediante versiones. Cualquier cambio de arquitectura, entradas, esfuerzo de búsqueda o cadencia requerirá una enmienda identificable y un protocolo coherente con la comparación que se quiera sostener.

Antes de consultar la reserva se deben cerrar sus muestras supervisadas, reglas de actualización, análisis y comparaciones. Las guías completas de E2/E3 no se han aportado; no se inventan requisitos adicionales. El diseño de noviembre de 2026 es posterior a los periodos de mercado: una evaluación de 2025–2026 será un backtest con reserva, no un pronóstico emitido prospectivamente. Si se usa ese test para diseñar un modelo posterior, hará falta otra evaluación independiente para confirmarlo.

## Evidencia y reproducción

`scripts/prepare_dl_r2_cv.py` se ejecuta en el entorno separado de `timeseries-cv` 0.1.5/NumPy 1.26. `scripts/train_dl_r2_cv.py` consume los índices para ajustar redes en PyTorch; `scripts/dl_data_v2.py` define la representación relativa, `scripts/dl_models.py` las arquitecturas y `scripts/export_filipe_book.py` recalcula métricas desde predicciones. `scripts/analyze_filipe_statistics.py` produce el complemento usando la enmienda registrada y los mismos archivos, sin entrenar. Los entornos se fijan en `requirements-cv.txt` y `requirements-training.lock.txt`.

Se publican [métricas anuales](../_static/dl_r2_cv_20261104/analysis/metrics_summary.csv), [julio–diciembre](../_static/dl_r2_cv_20261104/analysis/metrics_common_h2_cv_only.csv), [doce contrastes globales](../_static/filipe_statistics_20261104/global_seed_tests.csv), [runtime](../_static/dl_r2_cv_20261104/runtime.json), [predicciones](../_static/dl_r2_cv_20261104/predictions.csv) y [configuraciones por trayectoria](../_static/dl_r2_cv_20261104/runs.csv). RMSE y R² anuales se calculan agrupando las 366 predicciones de cada trayectoria; no se promedian R² de folds unitarios ni se confunde media de RMSE entre semillas con RMSE de un ensemble. MASE utiliza el denominador permitido de cada entrenamiento mensual.

El suplemento publica [DM](../_static/filipe_statistics_20261104/temporal_dm.csv), [diagnósticos](../_static/filipe_statistics_20261104/diagnostics.csv), [bootstrap](../_static/filipe_statistics_20261104/block_bootstrap.csv), [sensibilidad del diseño](../_static/filipe_statistics_20261104/design_sensitivity.csv) y [superioridad entre semillas](../_static/filipe_statistics_20261104/probability_superiority.csv). Las curvas corresponden a registros reales de parada interna de diciembre, semilla 4101; no son curvas de test ni de tamaño muestral. La inferencia temporal y la aleatorización del entrenamiento conservan alcances separados.

Las instrucciones completas están en `BUILD.md`; construir HTML valida presentación y enlaces, sin volver a ajustar las redes ni certificar superioridad o cumplimiento académico total.

El [inventario de revisión](../_static/e1/revision_items_libro_20261104.csv) identifica los 390 incisos numerados comprobados; su [registro JSON](../_static/e1/revision_items_libro_20261104.json) conserva evidencias, límites y pendientes. Revisar un ítem no equivale a declarar completo el requisito científico.
