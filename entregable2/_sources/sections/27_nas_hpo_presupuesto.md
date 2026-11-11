# 27. Decisiones NAS, rangos HPO y presupuesto para Codex

**Vigencia al 11/11/2026:** Se conserva el calendario original de once selecciones, con máximo de 85.074 intentos neuronales. Los conteos de etapas auxiliares de este capítulo son otra unidad. La propuesta de tres no es operativa; R y el presupuesto completo siguen pendientes. [Decisión y ejecutor vigentes](40_calendario_original.md).



**Implementación posterior:** [§28 implementa las seis fábricas NAS y el enlace de configuraciones](28_fabricas_nas.md), con comprobaciones sintéticas y sin entrenamiento. Las condiciones de ejecución que siguen describen el corte de diseño; la integración del ciclo completo y el perfilado siguen pendientes.

**Diseño E2-NAS-HPO-20261109-v1 · 09/11/2026.** Esta decisión técnica resuelve las filas NAS y los dominios HPO del [registro §26](26_registro_recetas.md). Se basa en §§11–14 de la guía aportada y sus fuentes documentadas. Los valores numéricos del ajuste son decisiones metodológicas de este diseño, no valores impuestos por el PDF.

[Protocolo JSON](../_static/search_protocol_20261109/protocol.json) · [Resumen calculado del presupuesto](../_static/search_protocol_20261109/budget_summary.json) · {download}`Espacio común de 1.000 arquitecturas <../_static/search_protocol_20261109/architectures.csv>` · {download}`252 configuraciones HPO de las 14 redes <../_static/search_protocol_20261109/hpo_candidates.json>`.

**Este corte sólo genera planes.** No importa modelos ni lee datos de mercado. No implementa todavía las seis nuevas fábricas NAS, no sustituye el preregistro inferencial completo y no ejecuta ajustes. Las implementaciones, parámetros y recibos de §§23–26 conservan su identidad histórica.

**Vigencia de las configuraciones descargables:** el archivo HPO enlazado arriba se conserva como artefacto histórico de diseño. Para los valores del protocolo ejecutable activo, consultar [la exportación regenerada y verificada](../_static/optimized_progress_20261111_213112/registered_hpo_configs.json) y [los 546 ensayos con su configuración](../_static/optimized_progress_20261111_213112/trials.csv). Se verificó cada payload contra su huella de campaña; no se eligieron nuevos hiperparámetros ni se entrenó para producir esta tabla. Los IDs por sí solos no sustituyen la huella/versionado del payload.

## 27.1 NAS resuelto por receta

La guía distingue un cuerpo definido por la fuente de un diseño propio o sin arquitectura verificable; no exige adjudicación del profesor para documentar esta decisión. Se congelan cinco cuerpos publicados y se asigna NAS a seis variantes mediante una interpretación conservadora explícita. Es una decisión del diseño E2, no una certificación de equivalencia con los autores ni de aceptación institucional.

| Modelo | NAS | Fundamento y condición de la decisión | ID de búsqueda |
|---|---|---|---|
| FCN-1D | **No** | El cuerpo Wang 128/256/128 y kernels8/5/3 queda fijo. Port1D y cabeza σ7 son interfaz de tarea explícita; no se permite cambiar el cuerpo bajo esta decisión. | `fcn1d_wang_d7_v1_proposed__hpo_v1` |
| ResNet-1D | **No** | Tres bloques64/128/128 y convenciones de atajos del snapshot quedan fijos. Interfaz D7/regresión separada del cuerpo. | `resnet1d_wang_d7_v1_proposed__hpo_v1` |
| Inception-1D | **No** | Miembro individual definido por fuente: seis módulos, ramas32 y kernels40/20/10. No se atribuye el ensemble de cinco. | `inception1d_single_d7_v1_proposed__hpo_v1` |
| LSTM-autoencoder con predictor | **Sí** | Ruta conservadora propia: receta exacta sostenida en código, texto completo no verificado y escala auxiliar adaptada. Nueva familia SAE NAS con dos fases AE y tres grupos. | `SAE-D7-NAS-v1` |
| TimesNet | **Sí** | Proyección7→8, frecuencias admisibles y preset mixto requieren una variante propia en lugar de exención por nombre. | `TN-D7-NAS-v1` |
| TSMixer | **No** | Cuerpo publicado de seis bloques y ancho512 fijo. Se ajusta tasa del dropout existente sin cambiar profundidad/ancho. | `TSM-D7-r1__hpo_v1` |
| TiDE | **Sí** | Decoder temporal8 frente a1 y covariables sustitutas cambian estructura; se trata como variante propia. | `TIDE-D7-NAS-v1` |
| TimeMixer | **Sí** | Dos descensos7/3/1 y readout propio difieren del diseño fuente; se trata como variante propia. | `TM-D7-NAS-v1` |
| PatchTST | **Sí** | Decisión conservadora para geometría D7 replicada y cabeza conjunta propias. No se usa la exención de la configuración publicada para esta variante. | `PT-D7-NAS-v1` |
| iTransformer | **No** | Cuerpo publicado L2/D256/FF256/H8 fijo; embedding7 y cabeza escalar se declaran interfaz de tarea. | `IT-D7-r1__hpo_v1` |
| CNN-1D + LSTM + atención temporal | **Sí** | Composición propia sin artículo verificado que defina la receta completa. | `HIB-D7-NAS-v1` |

Los cinco cuerpos sin NAS **no cambian profundidad, ancho, kernels, normalización ni número de cabezas de la fuente**. Esto concilia el mínimo general de §11.2 con la disposición específica de §14: usar el cuerpo publicado como está definido y ajustar los otros hiperparámetros. D7 y el readout de volatilidad se declaran interfaces de tarea. Si se altera un cuerpo congelado, deberá crearse otra receta y revisar su decisión NAS.

Para PatchTST se elige conservadoramente NAS por la geometría D7 replicada y la cabeza propia; no se extiende la exención de la configuración publicada por el solo nombre. Para SAE, el texto exacto sin verificar impide fundar la ruta publicada únicamente en el DOI. Esta decisión no convierte las nuevas variantes en réplicas de los cuerpos originales.

## 27.2 Espacio NAS y nuevas variantes

Cada familia NAS recorre W³×{0,1}³ con W={16,32,64,128,256}: **1.000 configuraciones, 125 sin dropout y 875 con al menos una posición activa**. No se descartan formas crecientes, decrecientes o mixtas. Cada bᵢ activa dropout0,2; los otros dropouts de estas variantes quedan en0. No se busca además una tasa de dropout en HPO.

Las correspondencias siguientes son especificaciones nuevas, amparadas en la correspondencia aproximada de §14.2. Necesitan implementación y validación sintética de las mil construcciones por familia. Los parámetros de §23 no describen estas variantes.

### LSTM-autoencoder con predictor · `SAE-D7-NAS-v1`

Grupos (w₁,w₂,w₃): AE1 encoder width w1; AE2 encoder width w2; new dense predictor width w3.

AE1: LSTM5→w1, repeat7, decoder LSTMw1→w1 y Linearw1→5+tanh. AE2 recibe vector finalAE1 repetido7, LSTMw1→w2, repeat7, decoder LSTMw2→w2 y Linearw2→5+tanh. AE1.eval(), dropout inactivo y sin gradientes al generar las entradas de AE2; pesos AE1 congelados en esa fase. En supervisado los encoders transferidos son entrenables y b1/b2 vuelven a estar activos. Predictor encoders transferidos→Densew2→w3+ReLU→Linear1+Softplus. X/s como target auxiliar; no escalar entrada encoder por s. b1/b2 aplican dropout0.2 a vector encoder y secuencia decoder antes de proyección de reconstrucción; b3 tras ReLU densa; decoders descartados en predictor. En inferencia dropout inactivo.

### TimesNet · `TN-D7-NAS-v1`

Grupos (w₁,w₂,w₃): TimesBlock w1; TimesBlock w2; TimesBlock w3.

Embedding temporal7→8 fijo, top_k y kernels fijados por HPO antes de NAS. Proyectar canales a wi antes de cada bloque, FF=2wi, residual dentro del mismo ancho. Diez momentos sólo al readout final. bi tras salida completa del bloque; no cambiar escalas/tiempos entre arquitecturas.

### TiDE · `TIDE-D7-NAS-v1`

Grupos (w₁,w₂,w₃): global residual encoder width w1; global residual decoder width w2; temporal residual decoder width w3.

Correspondencia aproximada: un bloque residual encoder de ancho w1, decoder de ancho w2 y decoder temporal oculto w3; latente8 y salida temporal8 fijos, featureencoder5→64→4 e identidadvariable16 fijos. Proyecciones de entrada/salida y atajos se dimensionan al grupo; b1,b2,b3 tras la salida completa del bloque residual encoder, decoder y temporal, respectivamente. Sentinel futuro cero y diez momentos fijos.

### TimeMixer · `TM-D7-NAS-v1`

Grupos (w₁,w₂,w₃): PDM block width w1; PDM block width w2; PDM block width w3.

Escalas7/3/1, pooling2 y dos descensos fijos. Tres PDM con proyecciones de canales hacia wi en cada escala antes de cada bloque; FF=2wi. bi tras cada PDM en todas sus escalas. FMM usa última dimensiónw3 y cabeza5*w3+10; diez momentos fijos.

### PatchTST · `PT-D7-NAS-v1`

Grupos (w₁,w₂,w₃): encoder block d_model w1; encoder block d_model w2; encoder block d_model w3.

Tres bloques con proyecciones lineales entre anchos; heads_i=wi/16 y FF_i=8wi. Desactivar uniformemente la acumulación de atención residual entre capas, pues cambia número de cabezas. bi activa dropout0.2 tras atención y FF del bloque; atención probabilística interna con dropout0 fijo. RevIN antes de padding9izquierda/8derecha, patch16 stride8 y dos parches fijos. Readout5*2*w3+10; cabeza sin dropout adicional.

### CNN-1D + LSTM + atención temporal · `HIB-D7-NAS-v1`

Grupos (w₁,w₂,w₃): CNN channels w1; LSTM units w2; additive attention projection w3.

Conv5→w1 causal/kernel elegido por HPO, ReLU y dropoutb1 sobre secuencia; LSTMw1→w2 de una capa con estado reiniciado y dropoutb2 sobre siete estados; atención Linearw2→w3+tanh/vectorw3, dropoutb3 en contexto w2 antes de Linearw2→1+Softplus. No dropout sobre pesos softmax. No momentos añadidos.

### Protocolo de búsqueda compartido

1. Ajustar los restantes hiperparámetros sobre referencia (64,64,64), patrón (0,1,0), con T=18 y dos semillas HPO. Congelar la configuración seleccionada antes de explorar arquitecturas.
2. Fase 1: evaluar las 1.000 arquitecturas con una semilla común y máximo50 épocas por fase de optimización. Es el25% del máximo completo.
3. Fase 2: k finalistas distintos más la referencia si aún no está incluida; cinco semillas distintas de fase1 y máximo200 épocas por fase.
4. Seleccionar por menor media de RMSE validación. Empate técnico: diferencia respecto a la menor media ≤δ o IC nominales95% de la media entre semillas solapados; entre esos candidatos elegir menos parámetros, luego menor media y finalmente ID estable. El solapamiento es regla de parsimonia, no prueba de equivalencia.
5. Reajustar desde cero según el protocolo temporal; nunca reabrir búsqueda por el resultado de prueba.

El piloto de ruido usa doce arquitecturas prefijadas y cinco semillas en cada familia: referencia, extremos y nueve configuraciones adicionales sin reemplazo. Se realiza con origen01/10/2023 y etiquetas hasta23/09/2023, antes de cualquier origen CV2024. Si en alguna familia la mediana de correlaciones Spearman entre rankings de presupuesto completo por semilla es <0,5 o no estimable, se usa **k=40 en todas**; en otro caso k=20. Se congela el k global antes del cribado completo. Ni k20 ni k40 garantizan recuperar el mejor de los mil; esa limitación debe reportarse.

Los pilotos globales de ruido/perfilado usan una receta previa fija: LR=0,001, WD=0,0001, batch32, warmup0 y AdamW común; TimesNet top-k4/kernels6, TimeMixer media móvil7, híbrido kernel3, Ridge α1 y ARIMA(1,0,0). Los dropouts NAS obedecen sólo al patrón0,2; los cuerpos fijos con dropout existente usan0,1 y las CNN fuente0. **No usan ganadores HPO de2024/2025 ni incluyen un HPO previo oculto.** Sirven para ruido y recursos, no para elegir la receta ganadora.

Las semillas de HPO, cribado, confirmación, piloto de ruido, perfilado, piloto de potencia y reajustes pertenecen a conjuntos separados. Su lista exacta está en el JSON.

## 27.3 Dominios HPO y reglas de entrenamiento

Se elige T=18 para permitir el espacio completo de dieciocho órdenes ARIMA sin inflar configuraciones repetidas. Cada red y Ridge recibe dieciocho configuraciones distintas; las redes usan dos semillas por configuración. Naive estacional agota tres periodos y persistencia no tiene parámetros ajustables.

| Factor | Dominio o decisión |
|---|---|
| Método | Muestreo aleatorio estratificado, reproducible; lista versionada antes de entrenar |
| Learning rate | Escala logarítmica [10⁻⁵, 3×10⁻³] |
| Weight decay | Tres configuraciones con0 y quince estratos logarítmicos en [10⁻⁶,10⁻²]; sólo matrices de pesos, sin biases ni normalización |
| Batch | {16,32,64}, seis apariciones por tamaño en las18 configuraciones |
| Optimizador | AdamW; β=(0,9;0,999), ε=10⁻⁸ |
| Pérdida supervisada | MSE de y/a; seleccionar con RMSE en unidades originales |
| Dropout sin NAS | FCN/ResNet/Inception:0, sin añadir módulos; TSMixer:{0;0,1;0,3;0,5;0,7;0,9}; iTransformer:{0;0,1;0,2;0,3;0,5} |
| Dropout con NAS | Fijo p=0,2 en posiciones de patrón; no es eje HPO adicional |
| Máximo completo / corto | 200 / 50 épocas por fase, incluidas por separado AE1 y AE2 |
| Parada supervisada | Paciencia20, mejor RMSE validación, mejora estricta y primer checkpoint en empate |
| Fases auxiliares SAE | MSE de reconstrucción X/s en validación interna, paciencia20; optimizador nuevo por fase |
| Scheduler | Coseno después de warmup; mínimo LR0. Warmup0 salvo eje declarado |
| Precisión / clipping | FP32 / norma global máxima1 |
| Ridge | 18 valores equiespaciados en log₁₀α de−6 a3 |
| ARIMA | p,q∈{0,1,2}, d∈{0,1}:18 órdenes únicos |
| Naive estacional | m∈{7,14,28} |

Los rangos de LR y regularización abarcan varios órdenes de magnitud sin afirmar optimalidad. Se incluye dropout0,9 en TSMixer para cubrir el preset fuente; no se copia ese valor a otros cuerpos. El número18 es un límite de oportunidades, no una garantía de encontrar el óptimo.

Ejes adicionales previos a NAS: TimesNet top-k∈{2,3,4} y número de kernels∈{2,4,6}; TimeMixer ventana de media móvil∈{3,7,25}; híbrido kernel causal∈{2,3,5}. Geometría D7 y grupos de la búsqueda quedan fijos al recorrer las mil arquitecturas. En PatchTST NAS e iTransformer, warmup∈{0;0,05;0,10} del total de pasos del presupuesto correspondiente.

D7×5 y h=1 permanecen heredados; no se alarga contexto. Las cabezas de PatchTST son función de wᵢ y las de iTransformer quedan en8. No hay pesos externos para fine-tuning ni teacher forcing en la salida escalar: esos ejes se justifican como no aplicables.

Si queda un batch final de una observación, se redistribuyen las últimas muestras en un lote de tamaño batch−1 y otro de2; se conservan todas las filas y el tamaño máximo. Con menos de dos muestras válidas se registra fallo del contrato. No se hace selección con datos externos al soporte permitido.

NaN, divergencia, timeout y falta de memoria consumen el intento; no generan una configuración o semilla sustituta. Una causa sistemática de implementación bloquea la campaña y exige corregir y repetir lo afectado mediante enmienda. Se conservan los fallos y el coste.

## 27.4 Selección temporal y repeticiones

Se fija **escenario B**: selección interna y reajuste nuevo en cada origen. Son diez orígenes mensuales enero–octubre2024 más una selección final en01/01/2025: once selecciones presupuestadas, sin añadir una duodécima para el piloto. Se conserva entrenamiento interno hasta o−98, validación o−90…o−8 y reajuste hasta o−8.

Cada configuración HPO se selecciona por menor RMSE media entre sus dos semillas; el ID resuelve únicamente empates numéricos exactos. Para el reajuste se congela, por origen y fase, el techo de la mediana de mejores épocas de las dos semillas HPO sin NAS o de las cinco semillas del finalista NAS ganador. Ese número se usa en todas las semillas nuevas, sin parada sobre etiquetas del bloque externo. Los preentrenamientos SAE tienen sus propias cuentas de épocas.

El piloto de potencia usa TimeMixer, híbrido y GRU32 con cinco semillas y53 fechas de desarrollo08/11–30/12/2024. Las recetas se toman de la selección interna del01/10/2024 y se congelan antes de ajustar el piloto; sus etiquetas no reabren HPO. Su variabilidad puede no representar las recetas finales elegidas en enero y deberá declararse esa limitación.

R final será común, entre6 y30, dimensionado con el piloto y simulación temporal que considere multiplicidad y objetivo de potencia≥0,80. **No se declara potencia por fijar R=30.** Si el tamaño requerido excede el techo, se reportan potencia alcanzable/MDE y el límite de recursos. R=6,10,20,30 son escenarios de coste, no resultados del análisis. La inferencia y las familias definitivas siguen en el preregistro pendiente.

Las 608 fechas conocidas conservan carácter exploratorio; el diseño B no borra su exposición. Su soporte común tampoco se certifica con esta planificación.

## 27.5 Presupuesto contable y entorno observado

El entorno Codex inspeccionado tiene cuota de **4 vCPU**, límite de **16 GiB**, CPU Xeon Platinum8573C y ninguna GPU visible. PyTorch y statsmodels no están instalados en ese entorno. Son observaciones de esta instancia, no garantías para futuras sesiones.

Plan operativo: un ajuste a la vez, cuatro hilos de cómputo, un hilo interop y límite de proceso de12GiB. Topes de pared propuestos:30min por fase corta y2h por completa. Son límites defensivos que el ejecutor deberá imponer, **no estimaciones de tiempo**. Si el coste perfilado no cabe en la sesión, no se poda el espacio obligatorio ni se presenta una campaña parcial como completa.

En el presupuesto, una etapa es un optimizador ajustado desde cero. Un SAE tiene tres etapas por trayectoria; una red supervisada ordinaria tiene una. Una fase corta y una completa tienen máximos distintos y no son unidades de tiempo intercambiables.

| Componente | Etapas cortas | Etapas completas o ajustes |
|---|---:|---:|
| HPO:14 redes×18 configuraciones×2 semillas×11 orígenes, más AE1/AE2 | 0 | **6.336** |
| NAS fase1:6 familias×1.000×11, más auxiliares SAE | **88.000** | 0 |
| NAS fase2, k20: hasta21×5×11 por familia, más auxiliares SAE | 0 | **9.240** |
| NAS fase2, k40: hasta41×5×11 por familia, más auxiliares SAE | 0 | **18.040**, alternativa a9.240 |
| Piloto global de ruido:12 arquitecturas por familia | 96 | 480 |
| Perfilado de cinco cuerpos sin NAS y tres referencias E1 | 0 | 8 |
| Piloto de potencia:3 modelos×5 semillas | 0 | 15 |
| Reajustes seleccionados de14 redes, más fases SAE | 0 | **176R** |
| Ridge+ARIMA: selección y reajustes en11 orígenes; dos perfiles clásicos | — | **420 ajustes clásicos** |

Con k20 y R30 son **88.096 etapas cortas y21.359 completas**, más420 ajustes clásicos. Con k40 y R30 son las mismas cortas y**30.159 completas**, más420 clásicos. El presupuesto reserva referencia adicional en fase2: si ya está entre los finalistas, su coste no se duplica.

**Este es el presupuesto de búsqueda, pilotos y reajustes**, no el total científico del proyecto. Controles de etiquetas permutadas y validación adversarial, calibración que requiera otros refits, robustez, ablaciones, interpretabilidad, inferencia y reproducción independiente se deben presupuestar además antes de su ejecución. No se presentan como gratuitos ni incluidos en estos totales.

Para convertir etapas a horas se medirán tiempos/memoria por familia y arquitectura, incluyendo representaciones y decoders SAE. Los tiempos de las variantes NAS se obtienen del piloto de ruido; las otras ocho redes y los dos clásicos tienen perfiles explícitos. Se sumarán los costes de los once orígenes, con todos sus intentos y fallos. **Horas y coste monetario siguen sin estimar**: ni disponer de CPU en Codex ni multiplicar un smoke por1.000 permite calcularlos.

## 27.6 Reproducir el plan sin entrenar

El generador usa sólo biblioteca estándar; no instala dependencias, carga mercado ni importa frameworks de modelos.

```bash
python -m unittest discover -s tests -p test_e2_search_plan.py -v
mkdir -p work/e2_search_plan
python scripts/e2_search_plan.py \
  --protocol book_e2/_static/search_protocol_20261109/protocol.json \
  --k 20 --final-seeds 30 \
  --output work/e2_search_plan/manifest_k20_r30.json
```

El archivo de salida debe ser nuevo: no se sobrescribe. El generador enumera1000 configuraciones por familia,18 HPO por red, dominios clásicos y cotas de coste. Su éxito prueba coherencia del plan; **no demuestra que todas las redes se construyan ni que puedan entrenarse**.

## 27.7 Condiciones antes del primer ajuste

1. Implementar los seis mapeos NAS y los ejes HPO en fábricas versionadas; probar sintéticamente las1.000 construcciones de cada familia, gradientes y normalización sin pérdida de cardinalidad.
2. Completar la integración causal de los18 sistemas y fijar dependencias reproducibles. Comprobar el entorno Codex de ejecución otra vez.
3. Cerrar el preregistro del piloto/perfilado y su presupuesto; medir recursos antes de lanzar la búsqueda completa. Después fijar k y R según las reglas anteriores.
4. Completar hipótesis, inferencia, calibración y los componentes de presupuesto excluidos; registrar enmiendas con fecha real.
5. Certificar soporte final mediante acceso autorizado y conservar la interpretación exploratoria de los datos conocidos.

Quedan resueltas las decisiones NAS, los dominios y el límite de oportunidades de búsqueda. La ejecución permanece condicionada a estos pasos verificables; no se ha entrenado ningún modelo en esta actualización.
