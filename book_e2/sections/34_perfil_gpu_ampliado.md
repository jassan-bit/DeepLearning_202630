# 34. Perfil GPU ampliado y viabilidad del extremo TimesNet

**Resultado informado: quince perfiles completados en la Tesla T4 de Colab**, con cero fallos, cero trabajos sin admitir y **680 pasos de optimización conocidos**. Son las catorce referencias en la cohorte grande y un perfil adicional del TimesNet máximo. El usuario ejecutó el notebook y aportó el informe; Codex comprobó su correspondencia con el registro, las fases y los contadores. Este informe amplía el [piloto pequeño de §33](33_piloto_gpu.md).

El resultado respalda la viabilidad de los perfiles probados con su receta fija. La calidad predictiva, la búsqueda NAS/HPO, las horas de la campaña completa y el R final siguen pendientes.

## Procedencia y configuración conservada

La evidencia está en el [informe científico ampliado](../_static/gpu_pilot_large_public_20261109/scientific_report.json) y su [manifiesto de importación](../_static/gpu_pilot_large_public_20261109/manifest.json). Se conservan los campos científicos admitidos; los diarios y recibos completos del runtime no acompañan la importación. La validación del contenido recibido no constituye una reproducción independiente de los ajustes originales.

El informe público tiene SHA256 `6d0e410da1204b76b1fe8b686eb9ece38be1028f03f752eeca72585b2806b8e6`; el manifiesto tiene SHA256 `2f35450ad91dbb513672e492ef0973e312937c24ec4b1811984914dec18d250d`. Ambos pertenecen a este segundo perfilado y se conservan separados de la evidencia pequeña de §33.

| Elemento | Configuración registrada o informada |
|---|---|
| Ejecutor congelado | `728277517be653be0d0ef10fb677a68adec22bb6` |
| Notebook corregido | `497088cbbabcdcec39c597d41b120a1ccde9379e` |
| SHA256 del registro previo | `892ad148e1a261d16b178371a8f8c71ea3af352cc4c6ac4fea934df7dc22ba87` |
| GPU y entorno | Tesla T4, PyTorch 2.8.0 + CUDA 12.6, cuDNN 91002 |
| Política numérica | FP32, sin AMP ni TF32, determinismo declarado |
| Cohorte | `large`: 619 ventanas de entrenamiento y 83 de validación |
| Receta | Semilla 274201, batch 32, dos épocas por etapa |
| Extremo TimesNet | Anchuras `[256, 256, 256]`, dropout `[1, 1, 1]` a tasa 0,2 |

Cada etapa realizó cuarenta pasos. SAE completó AE1, AE2 y la fase supervisada: **120 pasos**. Las otras trece referencias realizaron cuarenta cada una y el TimesNet máximo añadió cuarenta: **120 + 13 × 40 + 40 = 680**. No hay contadores desconocidos.

Los dos informes aportados por el usuario reúnen **29 perfiles y 872 pasos conocidos**. Son dos campañas de recursos separadas con referencias repetidas por cohorte; no corresponden a 29 modelos diferentes. El registro previo y el informe pequeño se conservan sin cambios.

## Tiempos y memoria de los quince perfiles

La suma de tiempos de pared de los trabajos es **146,406 s**, la de tiempos de los workers es **133,537 s** y la de ajuste es **83,045 s**. Estas medidas tienen fronteras distintas y no deben sumarse entre sí. No incluyen de forma completa la instalación, la conexión a Colab, la sesión del proveedor ni una campaña NAS/HPO.

Los valores se redondean a tres decimales. La VRAM reservada corresponde al máximo informado por el asignador de PyTorch para cada trabajo; no mide toda la memoria del dispositivo.

| Perfil | Ajuste (s) | Pared del trabajo (s) | VRAM reservada máxima (GiB) |
|---|---:|---:|---:|
| LSTM-autoencoder con predictor | 2,280 | 5,475 | 0,117 |
| TimesNet | 5,933 | 9,642 | 0,410 |
| TiDE | 2,103 | 5,710 | 0,064 |
| TimeMixer | 4,730 | 8,608 | 0,102 |
| PatchTST | 2,520 | 5,746 | 0,086 |
| CNN-1D + LSTM + atención temporal | 1,839 | 5,940 | 0,084 |
| FCN-1D | 1,795 | 5,077 | 0,070 |
| ResNet-1D | 3,668 | 7,125 | 0,094 |
| Inception-1D | 2,944 | 6,320 | 0,098 |
| TSMixer | 3,676 | 7,034 | 0,070 |
| iTransformer | 1,587 | 5,960 | 0,100 |
| MLP16 | 1,954 | 5,642 | 0,064 |
| LSTM32 | 1,446 | 5,035 | 0,084 |
| GRU32 | 1,493 | 4,880 | 0,084 |
| TimesNet · máximo | 45,075 | 58,211 | 5,080 |

El TimesNet máximo tiene **225.075.755 parámetros** y 900.303.916 bytes de estado. Completó el ajuste en **45,075 s** y el trabajo en **58,211 s**. Sus picos fueron **4,287 GiB de VRAM asignada**, **5,080 GiB reservada** y **2,583 GiB de RSS muestreado**. El límite efectivo informado para el asignador CUDA es **11,651 GiB**. El perfil terminó dentro del límite, con el batch y las condiciones registradas; no se han comprobado todos los extremos arquitectónicos ni todos los valores del espacio HPO.

## Comparación con la cohorte pequeña

Para comparar las mismas catorce referencias, el extremo adicional de TimesNet se mantiene aparte:

| Medición agregada de las 14 referencias | Pequeña: 161/83 | Grande: 619/83 |
|---|---:|---:|
| Pasos de optimización | 192 | 640 |
| Suma de ajuste (s) | 25,508 | 37,970 |
| Suma de pared de los trabajos (s) | 75,688 | 88,194 |

La receta, la identidad y el número de parámetros de cada referencia coinciden entre los dos informes. Los pasos aumentan 3,333 veces, mientras que el ajuste agregado medido aumenta aproximadamente 1,489 veces. Algunas mediciones individuales incluso disminuyen. Hay una sola ejecución corta por caso y dos épocas; el arranque, el calentamiento y la variación de medición influyen en estos cocientes. No permiten establecer una ley de escalado ni una comparación de aceleración CPU–GPU.

Los tiempos tampoco miden convergencia, errores de predicción o superioridad del candidato híbrido. El TimesNet máximo modifica anchuras y dropout respecto de la referencia; sus diferencias de coste no estiman aisladamente el efecto de uno de esos factores.

## Siguiente cierre antes de NAS/HPO

La [sección 35](35_busqueda_cuda_reanudable.md) incorpora posteriormente el enlace CUDA para la selección, la admisión por presupuesto y la continuación de campañas en GitHub. El siguiente cierre exige verificar y ejecutar ese orquestador en Colab con sus controles, antes de presentar una búsqueda científica completada. Las decisiones NAS y los dominios HPO de [§27](27_nas_hpo_presupuesto.md) permanecen fijados; estos tiempos de recursos no seleccionan nuevos dominios ni ganadores. Los dos pilotos ya aportados se conservan y no hace falta repetirlos sin cambios.

El adaptador histórico `e2_training_bindings.py` conserva el motor CPU `e2_staged_training`; el nuevo enlace CUDA se documenta por separado en §35. También construye entradas CPU para la inferencia. El enlace posterior de §35 conserva HPO → NAS-1 → NAS-2 y las interfaces de reajuste, las fases SAE, la separación temporal, las semillas, los desempates y los costes de cada intento. El notebook de recursos no implementa esa secuencia al cambiar su número de épocas.

El corte de recursos identificó estas tareas previas al lanzamiento científico; la implementación posterior se documenta en §35:

1. **Verificar el adaptador y su lanzador.** Integrar ajuste, selección de épocas e inferencia CUDA con pruebas artificiales y recorridos pequeños declarados, sin reinterpretarlos como una búsqueda completa.
2. **Separar los límites y admitir un presupuesto global.** Conservar los límites científicos de 1800/7200 segundos por etapa, separados de los 300/360 segundos y 900 segundos globales del piloto de recursos. Fijar un presupuesto por etapas y por sesión Colab antes de arrancar miles de ajustes.
3. **Persistir el estado de la campaña entre trabajos.** Distinguir intentos completos, fallidos e incompletos y evitar repeticiones silenciosas. Los checkpoints actuales son de inferencia y no contienen el estado del optimizador para reanudar un ajuste interrumpido.
4. **Dimensionar el almacenamiento y sus costes.** El estado del TimesNet máximo ocupa aproximadamente 0,839 GiB, y el piloto no guardó pesos ni midió escritura a disco. La retención de candidatos y sus tiempos requieren una política explícita; los originales históricos se conservan.

El estimador existente requiere separar tasas de actualización, evaluación, preparación, finalización y clonación. El exportado GPU no contiene esa descomposición, por lo que ese uso devuelve `timing_decomposition_missing`. Tampoco están medidos todos los extremos HPO ni las posiciones arquitectónicas estratificadas por familia. Una integración acotada con instrumentación adicional puede producir esos componentes; multiplicar directamente estos microperfiles por configuraciones y épocas no cierra las horas GPU ni garantiza completar cada asignación científica.

El informe mantiene `scientific_HPO_or_NAS_executed=false`, `final_market_values_read=false`, `final_R=null` y `full_campaign_wall_hours=null`. Los valores finales de mercado no se han abierto según la evidencia aportada. El R final se resolverá después de seleccionar recetas y comprobar su variación; esa tarea pendiente no impide preparar las etapas anteriores admitidas, y el reajuste final conserva su guarda de R.
