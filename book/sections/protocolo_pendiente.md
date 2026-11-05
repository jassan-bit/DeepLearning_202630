# Decisiones, revisión y pendientes de E1

El procedimiento vigente compara **GRU–LSTM–TCN** mediante CV forward chaining de Filipe con **D=7/14/21/28**, objetivo n=7 y horizonte h=1. Se conservan las mismas covariables, fechas, purgas y semillas. El baseline neuronal es **GRU/D7**, seleccionado por RMSE media de CV antes de acceder al contenido del test en este ciclo. El periodo de test ya había sido observado en el proyecto; la comparación mantiene alcance exploratorio.

## Decisiones incorporadas

| Elemento | Criterio y evidencia vigente |
|---|---|
| Activo y fuente | Bitcoin, Binance Spot BTCUSDT; velas oficiales de un minuto agregadas por día UTC |
| Problema e información | Tipo 4, ruta 6.4; cinco canales OHLCV sin historia explícita de σ como entrada |
| Terna implementada | **GRU, LSTM y TCN**, adaptaciones compactas de familias con respaldo aplicado independiente |
| Entradas y objetivo | D=7/14/21/28 días; σ₇ poblacional ddof0, sin anualizar; h=1 |
| Sensibilidad del objetivo | n=14/21/28 explorados en EDA, no entrenados en el benchmark |
| Preprocesamiento | LogOHLC respecto al último cierre observable, log1pV; escalas aprendidas solo con entrenamiento permitido |
| Pérdida y métricas | MSE; RMSE primaria; MAE/MSE/MAPE/R²/MASE en escala original, con cobertura y casos no definidos |
| Calidad y embargo | Calendario completo, exclusión del día incompleto y ventanas afectadas, sin imputación; u≤origen−D−1 |
| CV | Historia inicial 2023; doce bloques mensuales de 2024, entrenamiento creciente, pesos fijos dentro del mes y diez semillas 4101–4110 |
| Ejecución nueva de CV | 120 trayectorias, 1.440 segmentos, 2.880 ajustes y 43.920 predicciones; LSTM reentrenada como control |
| Control de reproducción | 14.640 predicciones LSTM de CV exactamente iguales al control histórico |
| Selección con desarrollo | D=7 en las tres familias y **GRU/D7** como baseline; no se reelige según test |
| Ajuste final | 80 ajustes GRU/TCN nuevos y 40 LSTM idénticos reutilizados; 120 checkpoints y escaladores firmados antes del acceso de este ciclo al derivado de test |
| Test observado | 01/01/2025–31/08/2026; 608 fechas y 72.960 predicciones; 80 inferencias nuevas y 40 trayectorias LSTM firmadas reutilizadas |
| Actualización durante test | Ninguna; veinte intervalos mensuales de reporte de pesos fijos, sin selección o reajuste |
| Periodo fuera del experimento | Septiembre–octubre de 2026 conserva su estado no asignado |
| Parámetros | GRU 3.777; LSTM 5.025; TCN 4.289, constantes con D |
| Inferencia dentro de la terna | Tres pares D7 por fase, con Holm; sensibilidad a doce pares de cuatro historias; análisis **posterior y exploratorio** |
| Diagnósticos | BDS y Ljung–Box por bloque; DM con pérdida coherente, dependencia y HLN; familias y límites documentados |
| Referencias contextuales | Persistencia determinista; cuatro Ridge diagnósticas del EDA, sin ocupar puestos DL |
| Repeticiones y potencia | Diez semillas constituyen el presupuesto fijo; piloto y dimensionamiento prospectivo de esta terna pendientes |
| Criterio académico futuro | Superar el 5 % de RMSE frente a GRU/D7 y persistencia; decisión posterior al test actual y para otra evaluación independiente |
| Independencia | Una serie y un mercado; semillas, fechas, entradas y meses no crean réplicas de mercados |

Los porcentajes **27,26/27,33/45,41** describen 365/366/608 días sobre los 1.339 asignados al protocolo. No son proporciones impuestas por Filipe ni una partición aleatoria. Los recuentos supervisados cambian con historia, purga y calidad, y entrenamiento crece dentro de la CV.

Los [manifiestos de procedencia](../_static/gru_lstm_tcn_20261105/protocols_manifest.json), la [selección con CV](../_static/gru_lstm_tcn_20261105/selected_by_cv.json) y el [resumen de ejecución](../_static/gru_lstm_tcn_20261105/summary.json) identifican la secuencia. La publicación usa resultados guardados; no vuelve a entrenar o abrir brutos.

## Revisión integral contra la guía

Los incisos numerados de los capítulos 1–7 permiten seguir la guía. “Revisado” no equivale a cumplimiento científico total. Las cuatro entradas y las tablas finales tienen evidencia empírica; las sensibilidades n=14/21/28 siguen fuera del alcance entrenado.

La revisión bibliográfica mantiene **40 referencias y siete estudios cripto completos**, con roles de referencia y propuesta separados. Todos incluyen BTC; uno incluye otras diecinueve criptomonedas, de modo que esas categorías se solapan. Las frecuencias estrictas de la terna son **GRU 3/7, LSTM 5/7 y TCN 1/7**. No se interpreta el denominador como siete artículos de revistas indexadas. Los niveles de verificación editorial y lectura se conservan en el [catálogo](../_static/literature_e1/crypto_extension_20261105/references_access_40.csv) y la [matriz](../_static/literature_e1/crypto_extension_20261105/evidence_matrix.json).

Akbiyik et al. (2023) compara las tres arquitecturas independientemente para volatilidad de Bitcoin; Miura et al. (2019) aporta evidencia directa de GRU y LSTM {cite:p}`akbiyik2023,miura2019`. La adaptación compacta, la limitada recurrencia de TCN y la justificación de diferencias entre las recurrentes permanecen explícitas. Las fuentes de otros mercados se identifican como transferencia y no aumentan la frecuencia cripto.

## Matriz de cierre derivada de la auditoría

La auditoría de la guía distingue requisitos, ejecución y límites. Los identificadores A01–A23 conservan la trazabilidad documental; no se convierten en una nota ni un porcentaje de cumplimiento. La actualización de modelos no reescribe la cronología o el alcance de fases anteriores.

| Prioridad | Requisito y origen | Estado actual | Evidencia o criterio de cierre |
|---|---|---|---|
| **A01 · Alta** | Extraer tres modelos base de literatura aplicada; §§3–3.3 de la guía | **Parcialmente resuelto** | Terna GRU–LSTM–TCN respaldada por comparadores independientes y recetas documentadas. Falta reforzar recurrencia de TCN y justificar suficientemente la no redundancia de GRU/LSTM; una métrica favorable no lo sustituye. |
| **A02 · Alta** | Margen de mejora y planificación; §5.3 | **Definido como convención futura, no demostrado retrospectivamente** | [Margen futuro](../_static/gru_lstm_tcn_20261105/future_margin.json): 5 % frente a GRU/D7 y persistencia; ancla neuronal CV δ≈0,00025772842665. Requiere protocolo, piloto y otra evaluación si se desarrolla un avanzado. |
| **A03 · Alta** | Comparación formal; §§5.3–5.4 y ruta 6.4 | **Ejecutada con alcance exploratorio** | Métricas de las cuatro entradas y [pares de la terna en CV](../_static/gru_lstm_tcn_20261105/within_trio_pairs_cv.csv) y [test](../_static/gru_lstm_tcn_20261105/within_trio_pairs_test.csv), con IC, efectos y multiplicidad. Los contrastes son posteriores y no confirmación temporal independiente. |
| **A04 · Alta** | Piloto y tamaño de muestra; §4.1 | **Pendiente específico de GRU–LSTM–TCN** | Las diez semillas fueron un presupuesto. Se necesita piloto prospectivo del mismo procedimiento, δ y alternativa definidos, incertidumbre de dispersión y dimensionamiento; no se trasladan pilotos de otras recetas. |
| **A05 · Alta** | Código, configuración, datos y reproducción | **Ejecución y trazabilidad verificadas; alcance delimitado** | Catálogos de [1.440 segmentos CV](../_static/gru_lstm_tcn_20261105/configuration_catalog_cv.csv) y [120 modelos finales](../_static/gru_lstm_tcn_20261105/configuration_catalog_final.csv), hashes, fuentes y controles LSTM. La reproducción de un backtest no crea otro mercado o test desconocido. |

## Mejoras concretas y evidencia de cierre

- La terna actual tiene implementaciones independientes: GRU unidireccional, LSTM unidireccional y TCN residual causal dilatada. El campo receptivo de TCN cubre las cuatro historias con el mismo número de parámetros.
- La CV se ejecutó para las doce combinaciones modelo–historia y diez semillas. Todas se reportan, incluso las que no mejoran; D=7 fue la historia elegida por CV en cada familia.
- El baseline neuronal **GRU/D7** tiene RMSE media CV 0,005155. En el test histórico obtiene RMSE 0,003775 y R² 0,8542; el periodo ya observado mantiene alcance exploratorio.
- La identidad del control LSTM se comprobó sobre 14.640 predicciones nuevas de CV y 24.320 predicciones de test reutilizadas. La reutilización se distingue de un ajuste o inferencia nuevos.
- Persistencia conserva visibilidad: menor RMSE en CV y menores MAE/MAPE en test. No se afirma superioridad de todas las redes o en todas las métricas.
- Las descargas del [capítulo 8](08_entregables.md) y los agregados de [CV](../_static/gru_lstm_tcn_20261105/metric_summary_cv.csv) y [test](../_static/gru_lstm_tcn_20261105/metric_summary_test.csv) identifican el estado vigente. Los resultados, pesos y protocolos previos permanecen trazables en el historial del repositorio.

## Límites que permanecen abiertos

- **Literatura:** fortalecer TCN y la argumentación conjunta de las dos recurrentes; completar verificaciones académicas pendientes sin inferir indexación o Q1 por editorial.
- **Piloto y dimensionamiento:** ejecutar un piloto prospectivo del procedimiento vigente y una fase dimensionada si se requiere esa afirmación. Diez semillas no prueban potencia 0,80.
- **Confirmación externa:** la nueva terna se diseñó con desarrollo expuesto y un test ya observado. Una propuesta posterior necesita una evaluación independiente, con decisiones fijadas antes de sus resultados.
- **Margen futuro:** la regla del 5 % se registra después del test actual. No demuestra por sí sola una ganancia práctica ni utilidad financiera; su futura comparación requiere incertidumbre y multiplicidad.
- **Equivalencia:** TOST/ROPE y una banda justificada se requieren solo si se pretende afirmar equivalencia. Este libro no la afirma.
- **Generalización:** una serie BTCUSDT no representa otras criptomonedas, proveedores o regímenes; ampliar la literatura no amplía el experimento propio.

## Proyección a E2

El insumo vigente es **GRU/D7**, junto con su protocolo, recetas, variación, métricas e incertidumbre. El margen operativo futuro del 5 % se evalúa frente a GRU/D7 y persistencia, sobre la misma cohorte nueva y con el mismo pipeline. La propuesta avanzada deberá estimar su propia dispersión y dimensionar sus repeticiones; no reutilizará el test observado para ajustar y confirmar sucesivamente. No se entrenó aquí una propuesta avanzada ni se reasignó septiembre–octubre de 2026. No se inventan requisitos de documentos completos de E2/E3 que no se han aportado.
