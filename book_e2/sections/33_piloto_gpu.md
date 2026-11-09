# 33. Primer piloto de recursos en GPU

**Resultado observado: catorce referencias completaron el piloto en una Tesla T4 de Colab.** El usuario ejecutó el notebook y compartió el informe de su última celda. El informe registra catorce intentos exitosos, cero fallos, cero trabajos sin admitir y **192 pasos de optimización conocidos**, sin contadores desconocidos. Codex revisó la coherencia del informe recibido; estos ajustes no se ejecutaron en la CPU de Codex ni en GitHub.

Este corte mide recursos de las recetas de referencia sobre desarrollo. No contiene errores de predicción, una búsqueda científica NAS/HPO ni una comparación que permita elegir un ganador. La campaña completa, su presupuesto GPU en horas y el R final permanecen abiertos.

## Procedencia y configuración

La evidencia pública está en el [informe científico](../_static/gpu_pilot_public_20261109/scientific_report.json) y su [manifiesto de importación](../_static/gpu_pilot_public_20261109/manifest.json). Se conserva el contenido numérico del informe; los diarios completos y los archivos locales de la ejecución permanecen en Colab. La comprobación de correspondencia con los hashes declarados aporta trazabilidad, pero no sustituye una reproducción independiente de los ajustes originales.

El informe público tiene SHA256 `5e82e7df7b67047962519f40115bac69a4bb1cb278fac29598c477d83b683d3a` y el manifiesto `1454a67c9a27bcabbbecfbdea079c0bf9b2a2ffdee5e200a6110539c53c2a394`. La importación comprobó la lista de trabajos del ejecutor publicado, las fases, los pasos y la coherencia de tiempos y memoria, sin repetir los ajustes GPU.

| Elemento | Valor registrado o informado |
|---|---|
| Ejecutor congelado | `728277517be653be0d0ef10fb677a68adec22bb6` |
| Notebook corregido | `497088cbbabcdcec39c597d41b120a1ccde9379e` |
| SHA256 del registro previo | `892ad148e1a261d16b178371a8f8c71ea3af352cc4c6ac4fea934df7dc22ba87` |
| GPU y entorno | Tesla T4, PyTorch 2.8.0 + CUDA 12.6, cuDNN 91002 |
| Política numérica | FP32, sin AMP ni TF32, determinismo declarado |
| Hilos del proceso | Dos hilos CPU y un hilo interop |
| Cohorte | `small`: 161 ventanas de entrenamiento y 83 de validación |
| Receta del perfil | Semilla 274201, batch 32, dos épocas por etapa |

La [preparación de §32](32_ejecucion_gpu.md) fija el entorno, las fuentes y las guardas del notebook. El registro conserva su estado previo a los ajustes: `gpu_performance_measured=false` describe el momento de registro. Este informe posterior incorpora la medición sin reescribir aquel artefacto.

Los catorce sistemas son los diez modelos del catálogo, el candidato híbrido y las tres referencias MLP16, LSTM32 y GRU32 de E1.

Las trece referencias supervisadas realizaron doce pasos de optimización cada una. El autoencoder realizó doce en AE1, doce en AE2 y doce en la fase supervisada: **36 + 13 × 12 = 192**. Sus tres etapas completaron dos épocas cada una.

## Tiempos y memoria observados

La suma de tiempos de pared de los catorce trabajos es **75,688 s**; la suma del tiempo de ajuste es **25,508 s** y la suma del tiempo de los workers es **63,222 s**. Son mediciones con límites distintos y no deben sumarse entre sí. Ninguna representa la duración completa de la sesión de Colab, el coste facturado por el proveedor o el tiempo de una campaña NAS/HPO.

Los valores siguientes se redondean a tres decimales. El tiempo de pared corresponde al trabajo completo registrado; la VRAM reservada es el máximo del asignador CUDA para ese trabajo, no toda la memoria del dispositivo.

| Referencia | Ajuste (s) | Pared del trabajo (s) | VRAM reservada máxima (MiB) |
|---|---:|---:|---:|
| LSTM-autoencoder con predictor | 1,667 | 4,661 | 88,000 |
| TimesNet | 3,231 | 7,912 | 420,000 |
| TiDE | 1,552 | 4,727 | 66,000 |
| TimeMixer | 2,625 | 6,343 | 104,000 |
| PatchTST | 1,756 | 4,986 | 88,000 |
| CNN-1D + LSTM + atención temporal | 2,116 | 5,574 | 86,000 |
| FCN-1D | 1,555 | 4,677 | 72,000 |
| ResNet-1D | 1,904 | 5,090 | 98,000 |
| Inception-1D | 2,030 | 6,249 | 100,000 |
| TSMixer | 1,720 | 4,818 | 72,000 |
| iTransformer | 1,182 | 6,364 | 102,000 |
| MLP16 | 1,271 | 4,350 | 66,000 |
| LSTM32 | 1,369 | 4,511 | 86,000 |
| GRU32 | 1,530 | 5,425 | 86,000 |

TimesNet concentra el máximo observado de memoria CUDA: **358,103 MiB asignados** y **420 MiB reservados**. El mayor RSS muestreado del grupo de procesos fue **1,653 GiB**, también en TimesNet. El informe declara 15.637.086.208 bytes de memoria del dispositivo y un límite efectivo de 12.509.668.966 bytes para el asignador. La holgura de este perfil pequeño no demuestra que toda arquitectura o configuración del espacio NAS quepa en la misma GPU.

Los tiempos de estas dos épocas no miden convergencia ni calidad predictiva. Tampoco permiten calcular una aceleración frente a [§31](31_optimizacion_nube.md): aquella medición utiliza otro entorno y otros cortes de perfilado. El informe acredita ajustes completados bajo su configuración; no documenta el resultado individual de las dieciocho pruebas CUDA omitidas en el corte de preparación CPU.

## Ampliación posterior antes de presupuestar la búsqueda

Después de este primer corte se ejecutó el perfil registrado de la cohorte **`large`**, con 619 ventanas de entrenamiento y 83 de validación, incluido el extremo opcional de TimesNet. Sus quince perfiles completos se documentan por separado en [§34](34_perfil_gpu_ampliado.md). La configuración utilizada en la primera celda del notebook fijado fue:

```python
RUN_PILOT = True
MODELS = None
COHORTS = ["large"]
INCLUDE_LARGEST = True
```

La selección admite **quince trabajos**: catorce referencias en la cohorte grande y el extremo de TimesNet. Conserva dos épocas por etapa, semilla 274201 y los límites registrados de 300/360 segundos por etapa/proceso y 900 segundos globales. El mismo workspace permite reutilizar las fuentes y los datos verificados; el notebook crea una salida nueva y conserva el intento pequeño anterior. Las opciones se fijan antes de iniciar el perfilado.

El resultado posterior se conserva en su propio informe y no altera las mediciones pequeñas de este corte. La ampliación permite revisar el tamaño y el extremo observado; las dos épocas y una ejecución por perfil no cierran por sí solas el presupuesto de una campaña NAS/HPO.

Después corresponde contrastar esos recursos con el espacio NAS/HPO de [§27](27_nas_hpo_presupuesto.md), sus límites por etapa, las fases auxiliares y las repeticiones previstas. Multiplicar directamente el tiempo de este perfil pequeño por el número de configuraciones no cierra ese presupuesto. La medición de recursos tampoco fija el número de semillas necesario para la inferencia.

Según el informe, no se ejecutó búsqueda científica NAS/HPO y no se abrieron valores finales de mercado. `final_R` y `full_campaign_wall_hours` son `null`. Las referencias y recetas de este piloto no pasan a ser ganadoras por haber completado sus ajustes.
