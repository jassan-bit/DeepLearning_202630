# Identidad y ejecución de la familia congelada

Se implementan 21 sistemas: la candidata `input_only_mse` (identidad científica `input-only-overlap-mse-1`), 20 rivales, 16 redes y cinco trayectorias deterministas. `global_mse` sigue siendo la comparación principal. Los alias no crean sistemas ni oportunidades de búsqueda adicionales.

`engine/recipes.json` conserva los parámetros seleccionados de las 16 filas de `book_e3/_static/e3_bloques_ab_20261115/deployment_master.csv`, SHA-256 `d850dd6487a898fa4f7793a1f4f72acb7be1e9a0950051b8e69eea2be5370ee7` (el valor normativo y verificable es `authority.sha256` dentro del JSON). Contiene identificadores de receta/configuración, LR, decay, dropout, tamaño de lote, warmup, oportunidades históricas, hashes de recibos y épocas seleccionadas históricas. Esas épocas quedan como procedencia; una nueva selección temporal no se presenta como el mismo ajuste histórico. No se copian predicciones ni etiquetas históricas a las recetas.

La fuente propia histórica es el checkout `d6cf7f1af6800ec1ac85a26347bdde52380e362f` del traspaso humano. Se copian constructores existentes, sin volver a escribir versiones reducidas. `vendor/SOURCE_MANIFEST.json` fija cada archivo por SHA-256. Los constructores originales verifican además sus subconjuntos de fuentes de autor y licencias. Los inicializadores de paquete añadidos permiten relocalizar el código; los archivos inventariados conservan sus bytes originales. Los imports absolutos de `src` y del hermano `dl_models` se aíslan en el adaptador sin modificar esos archivos. El bloqueo de construcción protege la importación de repositorios que comparten los nombres `models/layers/utils`.

| Sistema | Constructor ejecutado | Alcance e identidad conservada |
|---|---|---|
| input_only_mse | `vendor/candidate/candidate.py::make_model` | Copia exacta del runtime admitido; 1.262 parámetros; entrada local, salida global, ley de tres átomos y MSE original |
| global_mse | `seoi_v5.model.ScaleOverlapModel('global_mse',...)` | Mismo control histórico, preprocesamiento global, tres átomos; 1.262 parámetros |
| additive_positive_head | `src.e3_innovation_model.InnovationModel` | Variante aditiva histórica, backbone M00 y head FP64; 1.210 parámetros; sin recalibración inicial de bias |
| FCN-1D | `e2_recipe_models.build_model` | FCNRecipe histórico, 268.929 parámetros |
| ResNet-1D | `e2_recipe_models.build_model` | ResNetRecipe histórico, 522.955 parámetros |
| Inception-1D | `e2_recipe_models.build_model` | Una red InceptionRecipe histórica, 490.561 parámetros |
| LSTM-autoencoder con predictor | `ReconstructionAE(5,25)`, `ReconstructionAE(25,26)`, `SAERecipe` | AE1 → AE2 → predictor; 8.739 parámetros de inferencia, 18.751 con los dos AEs conservados en checkpoint |
| TimesNet | `e2_published_periodic.build_periodic` | Cuerpo de autor completo, guardia no-DC histórica y head escalar; 1.173.099 parámetros registrados |
| TSMixer | `e2_final_models_mlp.TSMixerAdapter` | Puerto PyTorch histórico atribuido, 35.044 parámetros; se conserva su puente escalar y sus momentos de la misma ventana |
| TiDE | `e2_published_tide.build_tide` | Puerto semántico histórico source-default R0 v2, 163.384 parámetros; rama vacía retenida |
| TimeMixer | `e2_published_periodic.build_periodic` | Cuerpo de autor completo y head escalar, 9.447 parámetros |
| PatchTST | `e2_published_patchtst.build_patchtst` | Encoder y Flatten_Head de autor, P3/S2 D7, 16.388 parámetros |
| iTransformer | `e2_final_models_attention.AttentionVolatilityAdapter` | Encoder invertido de autor completo, head D7 histórico, 795.403 parámetros |
| MLP16 | `dl_models.SmallMLP` | Referencia E1 reinicializada; 593 parámetros |
| LSTM32 | `dl_models.SingleLayerLSTM` | Referencia E1 reinicializada; 5.025 parámetros |
| GRU32 | `dl_models_gru_lstm_tcn.SingleLayerGRU` | Referencia E1 reinicializada; 3.777 parámetros |
| Persistence | Trayectoria observacional lag1 | Un predictor, sin réplica por semilla |
| SeasonalNaive | Trayectoria observacional lag7 | Un predictor, período seleccionado de siete observaciones |
| ARIMA | `statsmodels.tsa.arima.model.ARIMA` 0.14.5 | Ajuste real (2,0,1), orden seleccionado por CSV; cinco parámetros; filtrado continuo con parámetros congelados |
| floor_only | Fórmula histórica `sqrt(A)` | Un control analítico sin ajuste estocástico |
| floor_plus_mean_excess | Fórmula histórica `sqrt(A)+mean_fit(y-sqrt(A))` | Equivale al nombre histórico `floor_plus_fit_mean`; media estimada sólo en ajuste |

Los puertos FCN/ResNet/Inception y SAE se reutilizan como implementaciones propias históricas; sus diferencias declaradas frente a TensorFlow/Keras (BN poblacional, Conv1D, inicialización, precisión y serialización) siguen vigentes. No se afirma paridad bit a bit con el backend de los artículos. TiDE conserva las ramas y las inicializaciones del puerto histórico, incluidas partes con gradiente cero en R0. TimesNet conserva selección FFT por lote durante entrenamiento y ejecuta un ejemplo a la vez en inferencia, política ya establecida para evitar que otra ventana determine el período. Los parámetros de ramas de autor que el forward no usa mantienen la política histórica de congelamiento.

## Preprocesamiento, información y pérdidas

`fit_preprocessor(raw_fit,y_fit)` recibe exclusivamente filas admitidas de ajuste y devuelve un estado JSON. Ajusta medias/desviaciones poblacionales de cinco canales, escala media del objetivo, escala de innovación, referencia RMS de retornos y escala auxiliar de SAE. El estado registra hashes y número de filas. Transformar selección, planificación o prueba no modifica ese estado.

La candidata usa literalmente `candidate.fit_inputs` y `candidate.transform_known_history`: división local causal de log-precios relativos, volumen `log1p`, estandarización con estadísticas de ajuste, arquitectura y MSE sin cambios. `global_mse` usa las coordenadas globales de su código original. Las demás redes usan el `relative_window` histórico (log-precios relativos al último cierre conocido y `log1p(volume)`), estandarizado sólo en ajuste; su salida `sigma7/y_scale` se invierte una sola vez. La aditiva ya devuelve `sigma7` en unidades originales.

Todas las redes optimizan MSE de `sigma7` cruda dividido por la media de ajuste al cuadrado. Selección usa MSE después de la proyección física común `max(pred,sqrt(A))`. Se exportan ambas salidas y el piso; ninguna fila no finita se elimina ni reemplaza. Se mantiene la FP32 del encoder y la FP64 del head cuando la implementación original lo requiere.

Persistence y SeasonalNaive necesitan `known_sigma_history[N,7]`, ordenada de la observación más antigua a la más reciente. Es información ya conocida al origen, no una etiqueta de la fecha a predecir. Calcular los siete valores requiere el soporte de 15 cierres `[t-14,t]`, incluido el cierre del objetivo. El soporte de entrada de las redes sigue siendo sus siete OHLCV originales. La admisión purga el soporte local común completo; la memoria de un modelo ajustado y el filtrado causal de ARIMA pueden cruzar las particiones.

La admisión entrega un prefijo global de sigmas observadas desde el primer sigma genuino (bar7), bajo la clave de compatibilidad `observed_sigma_since_fit`, más `history_end_positions[N]` exclusivo. ARIMA guarda `fit_history_stop = history_end_positions_fit[-1]+1`, que incorpora la última etiqueta de ajuste, y recorta el prefijo desde allí. Para cada origen primero pronostica y después incorpora la observación; replay local no modifica el estado ajustado. También se admite una API explícita con prefijo verdaderamente posterior al ajuste y `forecast_offsets`. Los diagnósticos condicionales dentro del ajuste requieren `arima_in_sample_positions` y no se confunden con pronósticos fuera de muestra. En un reloj diario con faltantes, `fit_sigma_axis` puede mantener NaN genuinos; sus observaciones finitas deben coincidir exactamente con `fit_y`.

ARIMA reproduce la invocación histórica de statsmodels: tendencia constante cuando d=0, estacionariedad/invertibilidad exigidas, máximo 200 iteraciones, comprobación de convergencia y callback de presupuesto. La diferencia respecto al argumento por defecto (1,0,1) del helper ampliado es explícita: se usa el orden histórico seleccionado (2,0,1) de la autoridad CSV. No hay ajuste adicional por fecha ni sustituto si falla.

## Oportunidades y presupuesto

La candidata y `global_mse` conservan AdamW LR .003, decay0, lote32, máximo40 épocas, paciencia8 y clip1; la aditiva conserva la política histórica LR .001/decay0/lote32/40/8/1. Las redes E2/E1 conservan cada LR/decay/lote/dropout/warmup seleccionado, cap200 y paciencia20 del entrenador E2. El controlador aplica su agrupación histórica de decay y su calendario warmup/coseno. Las fuentes históricas permanecen disponibles para auditar esas reglas. El presupuesto es de sistema completo y los tres pasos SAE comparten una sola asignación; los AEs no reciben tres presupuestos de sistema independientes.

`training_phases`, `phase_parameters`, `phase_loss` y `transition_phase` exponen las fases reales al worker. AE1 reconstruye `X/s`; AE2 recibe siete repeticiones del estado final del encoder1 seleccionado, congelado, y reconstruye el mismo `X/s`. Al finalizar AE2 se copian exactamente ambos encoders al predictor; el supervisado los vuelve a entrenar. Cada fase tiene un optimizador nuevo. Se guardan AEs, predictor, banderas y escala para reconstruir un checkpoint intermedio sin repetir preentrenamiento ya terminado.

Los IDs de semillas del banco se conservan como enteros completos de 63 bits para Python y Torch. Sólo el RNG legado de NumPy requiere la transformación fija `seed % 2**32`; ésta se registra como compatibilidad de ese RNG, no como cambio de banco ni selección de semillas. Una semilla distinta no crea ajustes independientes de los cinco métodos deterministas. Los constructores preservan el RNG Torch del llamador; el worker controla y guarda los flujos de entrenamiento.

## Fuentes externas y derechos

| Implementación de autor | Commit | Licencia conservada |
|---|---|---|
| thuml/Time-Series-Library, TimesNet | 4e938a1767106324dd753b2a44832bf870a0252e | MIT |
| kwuking/TimeMixer | e24610583b36fdd8c76cc17a8df4e65759a5f460 | Apache-2.0 |
| yuqinie98/PatchTST | 204c21efe0b39603ad6e2ca640ef5896646ab1a9 | Apache-2.0 |
| thuml/iTransformer | c2426e68ca13f74aaec08045c5c724d8ad328124 | MIT |
| google-research/google-research, TiDE y TSMixer | d3a4bd8f0eceddda553e47085033b537935b283f | Apache-2.0 |

Los subconjuntos de autor ya existían en el workspace; no se descargan fuentes durante entrenamiento. Se incluyen las licencias originales completas, los archivos ejecutados, sus dependencias y los presets que exige la verificación histórica. TiDE/TSMixer ejecutan los puertos históricos, no TensorFlow. Se conserva también la fuente original de Google como evidencia. El archivo de licencia propia incluido limita expresamente su alcance al código identificado en su entrega original; este paquete no concede una licencia global del repositorio ni relicencia datos o dependencias. El `ORIGIN.json` de la candidata mantiene sus límites originales.

Dependencias de ejecución probadas: Python3.12, Torch2.8.0+cpu, NumPy2.3.5, SciPy1.17.0, pandas2.2.3, statsmodels0.14.5, patsy1.0.3, einops0.8.1 y reformer-pytorch1.4.4 con sus dependencias transitivas fijadas en el lock del paquete. iTransformer importa incondicionalmente `reformer_pytorch` en su fuente, aunque la configuración seleccionada use FullAttention: se instala la dependencia real, sin un módulo ficticio ni un reemplazo de atención. Los utilitarios/presets de autor no ejecutados no convierten TensorFlow, PyWavelets, Mamba u otros backends de sus repositorios completos en dependencias de esta ruta.

## Verificación efectivamente ejecutada

`tests/test_models.py` comprueba 25 contratos con fixtures sintéticos fijados: los 21 sistemas, forward/gradientes y un paso de optimizador por red, restauración exacta de predicciones con una fábrica nueva, equivalencia exacta de pesos/preprocesamiento/pérdida de la candidata, tres fases SAE con transferencia exacta, cinco reglas deterministas, un ajuste ARIMA real y replay causal con parámetros congelados. Se verifican todos los hashes de fuente y los hiperparámetros seleccionados. Las pruebas de causalidad comparan inferencia individual y por lote y no recalculan estadísticas.

Los recibos en `qa/models/` indican el comando, el entorno, los casos y los hashes probados. No se entrenó ningún mercado ni se abrió una prueba final. Estos contratos de software no acreditan potencia estadística, superioridad, robustez entre mercados ni R3. La advertencia de inicialización de tensores de tamaño cero en TiDE es una consecuencia conocida de la rama R0 que se conserva.
