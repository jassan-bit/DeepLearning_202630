# Auditoría ampliada y robustez de E3

Esta etapa prepara una ejecución empírica para el componente de evaluación ampliada y robustez. **Los resultados de mercado nuevos todavía no se han producido.** No se asigna una nota por disponer del código. Se declara asistencia de IA en diseño, implementación, revisión y pruebas.

## Ejecución

Abrir [el cuaderno Colab](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/e3_evaluacion_ampliada_robustez_colab.ipynb), seleccionar GPU y habilitar el secreto `GITHUB_TOKEN` con Contents: Read and write para este repositorio. El token permanece en el controlador; no se envía al proceso científico, se escribe en comandos Git o se incorpora en evidencias.

El cuaderno verifica primero los archivos D01/D02/D03 y guarda la admisión, sin depender de Render. Si no están los tres admitidos, conserva el diagnóstico y se detiene antes de entrenar. Después verifica la historia BTC y el NPZ fijados, prepara Python 3.12.12/Torch 2.8.0+cu126 aislados y crea un **tag anotado real** que contiene el protocolo, hashes y perfil de GPU antes de los ajustes. El kernel original no se reinicia ni cambia su Torch.

Cada tres trabajos completos guarda deltas verificables en ramas `e3-extended-audit-*`. Una nueva sesión recupera los checkpoints y el protocolo de la campaña anterior. Si cambia la GPU o el perfil de software, se detiene: un presupuesto en segundos no permite mezclar libremente hardware distinto. Una campaña ya terminada devuelve su enlace sin repetir ajustes. El cambio explícito `E3_EXTENDED_START_NEW = True` inicia otra auditoría; no es una solución a un error de datos o checkpoint.

## Comparación fijada antes de los resultados

- **19 sistemas:** los 17 históricos —incluido el candidato aditivo de 1210 parámetros— más `floor_only` y `floor_plus_mean_excess`. V3 permanece en una investigación separada. El segundo control estima un parámetro de exceso medio usando sólo ajuste; no cuenta como un optimizador neuronal.
- **Cuatro conjuntos:** BTC conocido y los adicionales WIKI/AAPL RAW, ETHUSDT Spot 2018–2021 y BTCUSD_PERP COIN-M 2020–2024. Las acciones usan sesiones, los criptoactivos días UTC y el perpetuo volumen en contratos. No se afirman tres mercados independientes; dos comparten exposición cripto y uno comparte subyacente BTC y parte de las fechas históricas.
- **Tratamiento:** misma rejilla de tres tasas, criterio de selección por validación, diez semillas finales y presupuesto máximo por ajuste. Los clásicos conservan su contrato causal de historia de etiquetas, más amplio que siete barras, y una sola trayectoria. No se crean réplicas deterministas ficticias.
- **Soporte común:** se guardan pronósticos crudos y `max(pronóstico, sqrt(A))` para todos los sistemas. El resultado principal usa esa misma proyección; el piso no se ofrece exclusivamente al candidato.
- **Separación:** escalas estimadas sólo en ajuste; purga de siete soportes; validación para HPO prefijado; calibración reservada; prueba para evaluación. El BTC conserva la geometría histórica 532/116/53/608 y su hueco de datos, sin rellenarlo. La máscara de diagnóstico de ajuste común se fija por disponibilidad de rezagos, no por errores observados.

Hay 776 ajustes neuronales previstos: 168 HPO, 560 finales y 48 adicionales para curvas. Se añaden cuatro ajustes ARIMA; SAE tiene tres fases que **comparten un presupuesto total**, no tres presupuestos completos. El techo nominal de 780 trabajos a 120 segundos es **26 horas**, además de instalación, inferencia, estadísticas y persistencia. El límite se comprueba en fronteras de lote/iteración; se registran excesos y no se presenta como un timeout duro de proceso. Los checkpoints completos pueden ocupar varios GB y se guardan en partes de tamaño limitado.

## Estudios que producirá

1. Métricas por dataset, modelo, semilla y partición, sin promediar RMSE de escalas distintas. Los paneles se fragmentan para evitar un CSV monolítico demasiado grande.
2. Intervalos por bloques temporales de longitud 7/14/28, ganancias emparejadas y dispersión entre semillas con sus ámbitos separados; rangos entre conjuntos descriptivos por su dependencia.
3. Intervalos nominales del 90 % calibrados sólo con la partición de calibración limpia, anchura, score de intervalo y cobertura temporal. Un punto y un intervalo no se convierten artificialmente en CRPS o PIT.
4. Perturbación plausible y reproducible de OHLCV en tres severidades, con escala de ruido estimada en ajuste, objetivos originales y calibrador congelado. En los clásicos la predicción cruda conserva su historia original; la proyección puede cambiar con el piso de las entradas corruptas. No se afirma que se haya perturbado idénticamente todo su historial.
5. Sensibilidad a las tres tasas ya ajustadas, sobre prueba **después** de guardar la selección por validación; no se cambia el ganador ni se añaden ajustes por el resultado de prueba.
6. Curvas de tamaño con ajustes reales para candidato y TSMixer, preseleccionados: 25/50/100 % y tres semillas. El 100 % reutiliza el ajuste final. Los prefijos de ajuste también cambian la cobertura temporal; no se atribuye causalmente toda diferencia sólo al número de muestras.
7. Tiempos, parámetros, pasos y hardware reales, tablas y figuras con incertidumbre identificada. FLOPs o energía ausentes permanecen ausentes; no se inventan mediciones.

## Alcance y estado actual

Es una **auditoría exploratoria preespecificada**, con fecha y tag reales anteriores a sus resultados. La batería completa de potencia temporal y entre conjuntos todavía no está dimensionada. Esa limitación bloquea una acreditación de R3; no impide obtener y publicar honestamente la evaluación. No se reutiliza como prueba de potencia el plan anterior de 18 sistemas/51 contrastes entre semillas.

Las pruebas de los modelos y análisis ejecutadas aquí usan datos generados. La admisión física local inicial confirma D01 y documenta que el proxy bloquea las dos fuentes Vision. La ejecución Colab `20261119T022547490732Z` admite después D01 y D02: 3270 sesiones y 1461 días, respectivamente; no ejecuta modelos. D03 queda bloqueado porque su ZIP mensual oficial de agosto de 2023 contiene sólo los días 1–27, con checksum correcto. `admission/incident_202308.json` documenta el diagnóstico y la recuperación prevista del mes completo mediante ZIP diarios oficiales, coincidencia del solapamiento y calendario íntegro. La recuperación física diaria y los ajustes reales todavía no han ocurrido aquí. Los entrenamientos, métricas, p-valores, pesos y descargas históricos conservan sus versiones. La calificación deberá revisarse después de inspeccionar los resultados reales y los estudios que efectivamente hayan terminado.
