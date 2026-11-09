# 29. Entrenamiento por etapas, selección temporal y perfilado CPU

**Piloto real posterior:** [§30: entrenamiento de desarrollo en Codex Cloud, costes y dimensionamiento condicionado](30_piloto_nube.md). Esta sección conserva el alcance artificial del corte anterior.

**Integración del 09/11/2026.** Se conectan las fábricas de [§28](28_fabricas_nas.md) con entrenamiento, selección interna, reajuste nuevo y evaluación temporal de los18 sistemas. La ejecución publicada utiliza exclusivamente datos artificiales generados por el programa.

**Resultado del fixture:**14 redes y4 comparadores; **142 ajustes neuronales, 176 etapas, 1.097 pasos AdamW y 142 checkpoints guardados y recargados con seguridad**. Las 3 fechas externas artificiales se predijeron y persistieron antes de revelar sus objetivos. No son resultados de Bitcoin ni un benchmark científico.

[Resumen y hashes](../_static/training_integration_20261109/summary.json) · [Perfiles CPU legibles](../_static/training_integration_20261109/profiles_table.json) · [Fuentes iTransformer verificadas](../_static/training_integration_20261109/author_source_receipt.json).

## 29.1 Registro común de catorce redes

`e2_training_models.py` construye las once candidatas y las tres referencias E1 con pesos nuevos. Mantiene dimensiones y conteos de los cinco cuerpos publicados y las referencias; las seis familias NAS reciben una arquitectura explícita y un trial registrado.

iTransformer usa los ocho archivos de autor fijados por commit y SHA256. El helper `fetch_e2_itransformer_sources.py` sólo descarga esa lista cuando se ejecuta explícitamente; construir un modelo no descarga código ni pesos. La falta o alteración de una fuente detiene la construcción, sin sustitutos. Dropout0,5 se aplica a sus módulos de dropout existentes, sin cambiar dimensiones ni conservar silenciosamente el límite0,3 del wrapper histórico.

La marca de fábrica `training_axes_applied=False` describe construcción. Cada recibo de ajuste registra por separado los hiperparámetros efectivamente aplicados, etapas y pasos; no se utiliza esa marca para negar el entrenamiento posterior.

## 29.2 AdamW y las tres fases SAE

`e2_staged_training.py` implementa:

- Optimizador AdamW nuevo por etapa, con estado inicialmente vacío; weight decay sólo sobre matrices de pesos, sin biases ni normalización.
- FP32, clipping global1, batches16/32/64 sin perder filas; una cola singleton se redistribuye.
- Warmup y coseno con LR positivo en las actualizaciones admitidas y cero al terminar el horizonte. La prueba de una sola actualización verifica cambio real de estado.
- Selección de checkpoint por RMSE validación en unidades originales, mejora estricta y primer checkpoint en empate exacto.
- Ajuste de reconstrucción por MSE de X/s para AE1 y AE2. AE1 queda congelado/en evaluación al generar las entradas de AE2; se transfieren encoders y se descartan decoders.
- Reajuste nuevo sin inputs ni etiquetas de validación. Cada fase usa el techo de la mediana de mejores épocas de las semillas de selección, congelado antes del bloque externo.

Los fallos se codifican como geometría, gradiente ausente, no finitez, recurso o checkpoint. No se confunde una divergencia numérica de un trial con un fallo sistemático de implementación. Los costes de fases SAE completadas y de la fase fallida se suman; fallos de reajuste también conservan run_id, origen/split y coste. No hay semilla ni configuración sustituta.

Los límites internos se comprueban en fronteras de operaciones. El supervisor de perfiles añade un watchdog de tiempo y RSS del grupo de procesos; no se describe como aislamiento de memoria inviolable.

## 29.3 Selección temporal y búsqueda por fases

`e2_temporal_training.py` prepara el escenarioB con entrenamiento interno hasta o−98, validación o−90…o−8 y entrenamiento de reajuste hasta o−8. Cada callback recibe copias del split permitido. Se conserva el calendario y la máscara común, con rezagos1/7/14/28 para las referencias.

`relative_window` mantiene el contrato E1: logprecios respecto al último cierre de cada ventana, log1p de volumen y media/DE poblacional ajustadas sólo al entrenamiento de cada etapa. La escala de y es su media positiva de entrenamiento, sin centrar. Selección interna y refit tienen escaladores independientes.

`e2_training_bindings.py` conecta HPO→cribado NAS→confirmación NAS→reajuste. Conserva todos los intentos, promedia RMSE por semilla, añade la referencia a finalistas sin duplicarla y aplica parsimonia por δ/IC nominales entre semillas. Ese criterio no demuestra equivalencia estadística.

El alcance registrado rechaza reducir las18 configuraciones, las1.000 arquitecturas, las semillas, los máximos200/50 épocas, δ o los límites temporales y de memoria fijados. Los reajustes registrados exigen el pool final separado y R cerrado por el protocolo de potencia. **R y k definitivos siguen pendientes de sus pilotos**; construir este motor no los estima.

Los callbacks son una frontera de API y copias de datos, no una protección frente a código externo que cierre deliberadamente sobre otros arrays.

## 29.4 Cuatro comparadores clásicos integrados

`e2_classical_training.py` selecciona18 valores Ridge,18 órdenes ARIMA y tres periodos naive; persistencia no recibe intentos ficticios. Ridge usa las mismas ventanas y escaladores, con intercepto no penalizado.

ARIMA se ajusta con el backend fijado statsmodels0.14.5 y sólo el prefijo permitido. Filtra el puente y avanza el estado con coeficientes congelados. Cada forecast se persiste antes de incorporar su observación; los huecos avanzan como ausencias, sin comprimir el calendario. Se conservan fallos y truncamientos a cero.

El reajuste clásico es fresco. Los recibos guardan coeficientes Ridge, parámetros/estado y versión ARIMA. No se afirma que esos recibos sean por sí solos un formato completo para restaurar cualquier proceso interno de statsmodels.

## 29.5 Qué se ejecutó realmente

La prueba de software recorre un origen de un calendario artificial del año2001, con una barra marcada como inválida:

| Elemento | Ejecución publicada |
|---|---|
| Entrenamiento interno / validación / reajuste | 108 / 83 / 198 ejemplos |
| HPO neuronal | Dos trials registrados por red y dos semillas |
| NAS de fixture | Dos arquitecturas, una semilla corta y cinco de confirmación; k2 |
| Presupuesto de épocas | Una corta y dos completas por fase |
| Reajuste y replay | Una semilla nueva por red; tres fechas externas artificiales |
| Comparadores | 18Ridge+18ARIMA+3periodos+persistencia:40 candidatos/reglas |
| Ajustes neuronales / etapas | 142 / 176 |
| Pasos AdamW / fallos neuronales | 1097 / 0 |
| Checkpoints recargados de forma segura | 142/142 |

**Las reducciones están declaradas como fixture sintético.** No satisfacen ni reemplazan T18, las1.000 arquitecturas, k20/40, R6–30 o los once orígenes del presupuesto científico. Ningún dato de mercado se cargó y no se reabrió la reserva de608 fechas.

Los checkpoints de predictor se crean de forma exclusiva, se sincronizan a disco y se verifican por hash. Los142 se recargaron con `torch.load(...,weights_only=True)`; los metadatos contienen tipos JSON nativos. No incluyen estado de optimizador ni todos los decoders auxiliares para reanudar una etapa interrumpida.

La evaluación usa el ejecutor fechado previo: comprueba pesos, escaladores, modos train/eval y tasas de dropout, escribe el lote de pronósticos con fsync y después revela el objetivo. Los recibos de métricas artificiales sólo acreditan el flujo del software; no se usan para ordenar modelos científicos.

## 29.6 Perfilado acotado en Codex

Se completaron15 procesos aislados: las14 redes en la referencia correspondiente y un caso TimesNet máximo. Cada referencia usó16 ventanas de entrenamiento y ocho de validación artificiales, dos épocas; SAE contabilizó sus tres etapas. El extremo usó un batch y una época fija. En total hubo **33 pasos AdamW adicionales al fixture integrado**, con cambios de estado en todas las etapas y optimizadores iniciales vacíos.

Entorno observado: cuota4vCPU, límite16GiB y sin GPU visible. Los perfiles usaron cuatro hilos, timeout120s y vigilancia de RSS12GiB, un worker a la vez. No hubo timeout ni OOM en estos casos.

| Modelo/perfil | Parámetros predictor | Pasos AdamW | Construcción+ajuste, s | Pico RSS de proceso, GiB |
|---|---:|---:|---:|---:|
| FCN-1D | 268.929 | 2 | 1.556 | 0.315 |
| ResNet-1D | 522.955 | 2 | 2.581 | 0.325 |
| Inception-1D | 490.561 | 2 | 1.788 | 0.331 |
| LSTM-autoencoder con predictor | 55.681 | 6 | 1.620 | 0.317 |
| TimesNet | 504.107 | 2 | 1.645 | 0.339 |
| TSMixer | 35.044 | 2 | 1.405 | 0.304 |
| TiDE | 18.969 | 2 | 1.327 | 0.298 |
| TimeMixer | 59.991 | 2 | 1.784 | 0.327 |
| PatchTST | 259.221 | 2 | 1.749 | 0.315 |
| iTransformer | 795.403 | 2 | 2.653 | 0.336 |
| CNN-1D + LSTM + atención temporal | 39.233 | 2 | 1.465 | 0.317 |
| MLP16 | 593 | 2 | 1.466 | 0.293 |
| LSTM32 | 5.025 | 2 | 1.448 | 0.307 |
| GRU32 | 3.777 | 2 | 1.358 | 0.297 |
| TimesNet · máximo | 225.075.755 | 1 | 61.273 | 4.642 |

El caso de225.075.755 parámetros consumió **61.273s** de construcción+ajuste y **4.642GiB** de RSS máximo; el tiempo total del worker incluye además arranque/imports y está en el recibo.

Estos tiempos varían entre sesiones. Incluyen costes de inicialización, validación y checkpoints del perfil; no equivalen a latencia pura de un paso ni permiten multiplicar linealmente hasta200 épocas, once orígenes o miles de búsquedas. **Horas y coste monetario de la campaña siguen sin estimar.** El perfil tampoco prueba persistencia de una ejecución larga de Codex ni la memoria necesaria para conservar simultáneamente todas las réplicas finales. La campaña deberá definir carga secuencial de checkpoints o una política equivalente antes de fijar ese presupuesto.

## 29.7 Evidencia, pruebas y reproducción

La batería nueva contiene **59 pruebas**, sin omisiones: etapas, splits y máscaras, parsimonia, fuentes fijadas, cuatro comparadores, fallos, checkpoints, supervisor y correspondencia de los recibos con el código. CI las ejecuta además de las verificaciones NAS y de publicación existentes.

- {download}`Integración completa, incluidos intentos y épocas <../_static/training_integration_20261109/integration_full.json.gz>`.
- {download}`Perfiles CPU completos <../_static/training_integration_20261109/profiles_full.json.gz>`.
- {download}`Diario de predicciones clásicas internas <../_static/training_integration_20261109/classical_inner_events.jsonl.gz>`.
- {download}`Diario externo: predicción→observación <../_static/training_integration_20261109/outer_events.jsonl.gz>`.

Desde el repositorio:

```bash
python -m venv work/training-venv
work/training-venv/bin/python -m pip install -r requirements-e2-training-cpu.lock.txt
work/training-venv/bin/python -m pip check
work/training-venv/bin/python scripts/fetch_e2_itransformer_sources.py --output work/author_sources/iTransformer
E2_ITRANSFORMER_SOURCE_ROOT="$PWD/work/author_sources/iTransformer" \
  work/training-venv/bin/python -m unittest discover -s tests -p 'test_e2_*training*.py' -v
work/training-venv/bin/python scripts/run_e2_training_synthetic.py \
  --output work/integracion-nueva --source-root-itransformer work/author_sources/iTransformer --threads 4
work/training-venv/bin/python scripts/profile_e2_training.py \
  --protocol book_e2/_static/search_protocol_20261109/protocol.json \
  --output work/perfiles-nuevos --source-root-itransformer work/author_sources/iTransformer \
  --epochs 2 --threads 4 --include-largest-timesnet
```

Los directorios de salida deben ser nuevos. Esos dos lanzadores generan sus datos artificiales dentro del programa; no aceptan archivos de mercado. Las funciones de entrenamiento en memoria no conceden autorización para abrir datos protegidos.

## 29.8 Próximo cierre científico

El motor ya enlaza selección, fases SAE, reajuste y replay. Antes de la campaña real corresponde cerrar el preregistro de los pilotos de desarrollo, ejecutar el piloto de ruido/potencia y perfilar cohortes y presupuestos reales; después fijar k/R y completar inferencia, calibración, robustez y controles.

Siguen pendientes la conciliación bibliográfica global, la certificación autorizada del soporte final y el cierre integral de requisitos de [§25](25_matriz_cierre.md). La evaluación histórica conocida conserva alcance exploratorio. No se declara un modelo ganador ni potencia alcanzada a partir de este fixture.
