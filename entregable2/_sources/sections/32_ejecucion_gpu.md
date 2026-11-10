# 32. Ejecución CUDA preparada y piloto de recursos

**Infraestructura y corte de preparación conservado.** Esta sección documenta el backend CUDA, el notebook reproducible y sus pruebas locales en la CPU de Codex. Las comprobaciones que requieren CUDA real se identifican como omitidas en ese corte. El [primer piloto GPU de §33](33_piloto_gpu.md) y el [perfil ampliado de §34](34_perfil_gpu_ampliado.md), ejecutados por el usuario en Colab, aportan mediciones reales de recursos. El presupuesto total en horas continúa abierto.

La ampliación a la cohorte grande y al extremo TimesNet completó sus quince perfiles en [§34](34_perfil_gpu_ampliado.md). La conexión posterior del backend CUDA a la selección NAS/HPO se documentó en [§35](35_busqueda_cuda_reanudable.md), y el [avance de la campaña en §36](36_avance_campana_cuda.md). Las [estimaciones CPU de §31](31_optimizacion_nube.md) siguen siendo evidencia separada; no se trasladan como una predicción de aceleración en GPU ni se sustituyen los fallos anteriores.

## Abrir y preparar el notebook

La GPU se ejecuta en la plataforma elegida por el usuario. La distribución del trabajo es la siguiente:

| Entorno | Papel en este corte |
|---|---|
| Codex Cloud actual | Preparación del código y pruebas sintéticas en CPU |
| GitHub | Repositorio, notebook y comprobaciones de integración; no aporta una GPU a este piloto |
| Colab o servidor CUDA del usuario | Ejecuta el piloto al conectar una GPU real y activar el lanzamiento; el primer informe T4 está en §33 |

El [notebook de ejecución](https://github.com/jassan-bit/DeepLearning_202630/blob/497088cbbabcdcec39c597d41b120a1ccde9379e/notebooks/entregable2_gpu.ipynb) puede [abrirse en Colab](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/497088cbbabcdcec39c597d41b120a1ccde9379e/notebooks/entregable2_gpu.ipynb). También puede ejecutarse en un servidor Linux x86_64 con una GPU compatible con el entorno fijado. El kernel anfitrión puede usar Python 3.13: el piloto utiliza su propio CPython 3.12.12 aislado. La disponibilidad, cuota y coste de la GPU dependen del proveedor.

La descarga del repositorio público no requiere iniciar sesión ni autorizar una cuenta de GitHub. El botón «Guardar en GitHub» corresponde a guardar cambios propios, una operación separada de la lectura del código.

El enlace fija la versión del notebook por commit; el notebook obtiene el código del commit **`728277517be653be0d0ef10fb677a68adec22bb6`**, fijado como SHA literal, y verifica el registro antes de iniciar ajustes. Usa descarga selectiva de código, dependencias y protocolos; los CSV se obtienen por un helper independiente después de verificar CUDA. No hace checkout de los archivos del período final.

1. Seleccionar una GPU en la plataforma y abrir el notebook. En Colab: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**, y conectar ese entorno.
2. Ejecutar las celdas en orden con `RUN_PILOT=False`. Se comprueba Linux y la GPU NVIDIA, y se prepara CPython 3.12.12 mediante `uv==0.9.5` dentro del workspace. El entorno virtual usa `pip==25.3` y queda separado de los paquetes del kernel de Colab.
3. Verificar la versión del Python aislado y el preflight CUDA. Se muestran por separado las versiones del kernel y del intérprete del piloto. Una máquina sin CUDA no admite el piloto ni activa entrenamiento CPU como sustitución.
4. Elegir los modelos y cohortes antes de cargar observaciones. Al habilitar `RUN_PILOT=True`, se comprueban las catorce arquitecturas con ventanas sintéticas y después se descargan los tres insumos registrados.
5. Ejecutar el piloto y revisar los resultados locales. Una nueva ejecución utiliza un directorio nuevo y conserva cualquier intento anterior.

Abrir el notebook o instalar dependencias no inicia la búsqueda NAS/HPO. La configuración predeterminada mantiene `RUN_PILOT=False`. Al activarla, el piloto admite las catorce referencias de la cohorte pequeña, con dos épocas por etapa. Puede elegirse previamente un subconjunto, por ejemplo GRU32 y TimeMixer; esa selección queda registrada antes del acceso a los valores de mercado.

Las fuentes oficiales de iTransformer se preparan una vez por workspace. Al repetir las celdas, el notebook verifica los SHA256 de los ocho archivos y su recibo de origen antes de reutilizar esa caché. Si falta un archivo, cambia su contenido o aparece un enlace simbólico o código Python ajeno al manifiesto, se detiene y conserva los archivos para revisarlos. Esta caché contiene fuentes de código; cada lanzamiento del piloto mantiene su directorio nuevo y sus recibos independientes. El paso de `RUN_PILOT=False` a `True` puede utilizar la misma sesión y el mismo workspace verificados. La comprobación de integración offline con las ocho fuentes auténticas cubrió una preparación inicial y dos reutilizaciones sin nuevas descargas, sin GPU ni datos de mercado.

## Código y entorno fijados

| Componente | Función |
|---|---|
| [Backend CUDA](https://github.com/jassan-bit/DeepLearning_202630/blob/728277517be653be0d0ef10fb677a68adec22bb6/scripts/e2_cuda_training.py) | Ajuste supervisado, selección de épocas, reajuste fijo y las tres etapas SAE con un dispositivo explícito |
| [Lanzador del piloto](https://github.com/jassan-bit/DeepLearning_202630/blob/728277517be653be0d0ef10fb677a68adec22bb6/scripts/run_e2_gpu_pilot.py) | Preflight, admisión del plan, procesos secuenciales, límites y mediciones de tiempo/memoria |
| [Descarga de desarrollo](https://github.com/jassan-bit/DeepLearning_202630/blob/728277517be653be0d0ef10fb677a68adec22bb6/scripts/fetch_e2_development_sources.py) | Tres payloads públicos inmutables y verificados por SHA256, sin analizar precios |
| [Entorno GPU](https://github.com/jassan-bit/DeepLearning_202630/blob/728277517be653be0d0ef10fb677a68adec22bb6/requirements-e2-training-gpu.lock.txt) | PyTorch 2.8.0 + CUDA 12.6 y versiones de sus dependencias; separado de los locks CPU históricos |
| [Registro del piloto](https://github.com/jassan-bit/DeepLearning_202630/blob/728277517be653be0d0ef10fb677a68adec22bb6/experiments/e2_gpu_pilot_20261109/registration.json) | Dominio, semillas, receta, admisiones y límites fijados antes de los ajustes GPU |

El registro tiene SHA256 `892ad148e1a261d16b178371a8f8c71ea3af352cc4c6ac4fea934df7dc22ba87`. Se creó el 09/11/2026 a las 15:28, hora de Bogotá, con estado `registered_before_gpu_market_fits`. Su publicación precede cualquier ajuste GPU de mercado de este diseño. Este registro no acredita que el piloto ya haya corrido.

El bootstrap fija `uv==0.9.5`, CPython 3.12.12 y `pip==25.3`; los wheels de uv y pip se comprueban por SHA256. El intérprete gestionado, su caché y el entorno virtual se crean dentro del workspace. Se conserva el kernel de Colab y sus paquetes. Las versiones CUDA del lock se contrastaron con los metadatos del wheel oficial Linux x86_64 para Python 3.12. El corte local de preparación no instaló el entorno CUDA completo; el informe posterior de §33 registra los ajustes realizados con PyTorch 2.8.0 + CUDA 12.6 en una Tesla T4. El preflight comprueba CUDA mediante una operación sintética y sincronización antes del acceso al desarrollo.

## Semántica conservada y alcance de las pruebas

El backend mantiene la pérdida **MSE**, AdamW con betas y epsilon fijados, los grupos de weight decay, el calendario coseno con el horizonte declarado, warmup, clip global 1, redistribución de minibatches unitarios y parada temprana estricta. En un reajuste fijo, las etiquetas de evaluación no participan en la parada. SAE conserva sus dos fases auxiliares y la supervisada, sus escalas de entrenamiento y sus estados separados.

Los modelos y tensores admitidos se transfieren al dispositivo CUDA explícito. Los estados seleccionados pueden representarse en CPU para su persistencia segura; el piloto de recursos no guarda archivos de pesos. Se mantienen las formas, parámetros y metadatos de las arquitecturas. Las fuentes CPU históricas permanecen intactas.

La configuración registrada usa **FP32**, sin AMP y sin TF32, con determinismo estricto y `CUBLAS_WORKSPACE_CONFIG=:4096:8` establecido antes de inicializar CUDA. Una operación determinista no soportada debe producir un fallo registrado. Estos controles no afirman identidad bit a bit entre CPU y GPU, igualdad de resultados entre plataformas ni una aceleración universal.

| Verificación de preparación | Evidencia local y límite |
|---|---|
| Descarga de desarrollo | Cinco pruebas pasaron y los tres payloads públicos se descargaron realmente con sus hashes exactos; no se interpretaron valores numéricos |
| Portabilidad de las catorce arquitecturas | Catorce pruebas CPU de predicción y backward pasaron; catorce casos CUDA reales se omitieron por ausencia de GPU |
| Semántica del entrenamiento | Once pruebas pasaron y tres casos CUDA reales se omitieron en Codex. La comparación sintética contra el motor CPU incluye estados, logs, RNG, parada, horizonte del scheduler, SAE y reajustes; el informe de ajustes GPU posterior se presenta por separado en §33 |
| Admisión y recursos | Catorce pruebas pasaron y un caso CUDA real quedó omitido; incluyen configuración, presupuesto global, contabilidad de fallos y rechazo temprano sin CUDA. No constituyen mediciones GPU |
| Notebook | Veintitrés pruebas estructurales y de guardas pasaron, incluidas once regresiones del bootstrap y tres de la caché de fuentes; ocho celdas de código con AST válido y salidas vacías, sin ejecutar instalaciones CUDA ni un piloto de mercado en el servidor CPU |
| Bootstrap Python aislado | Se ejecutó realmente el bootstrap del notebook: descargó CPython 3.12.12 gestionado, creó el venv e instaló pip 25.3 con hash verificado. Se comprobó desde hosts Python 3.12.14 y 3.13.9, sin instalar Torch/CUDA ni descargar mercado |

El corte histórico de preparación reúne **67 pruebas aprobadas** (44 de código y 23 del notebook) y **18 pruebas CUDA reales omitidas** por ausencia de GPU. Los resultados aprobados acreditan contratos de software en CPU y comprobaciones estructurales; no certifican una ejecución GPU. El informe de §33 acredita los ajustes descritos allí, sin aportar los logs de las dieciocho pruebas omitidas en este corte.

El notebook incluye las pruebas sintéticas CUDA antes de descargar los CSV. Estas comprueban dispositivos de parámetros y buffers, geometría, salida FP32, conservación de metadatos, tolerancia numérica de evaluación y gradientes finitos. Una prueba omitida aquí no se presenta como una prueba GPU aprobada.

## Datos admitidos

El helper sólo solicita tres archivos del commit `0b079b50625a936456a70695532853fc13298f31` del repositorio público. Conserva un archivo existente si coincide con el hash y se detiene ante un contenido diferente, en lugar de reemplazarlo silenciosamente.

| Insumo | SHA256 |
|---|---|
| Protocolo E1 `experiments/dl_r2_cv_20261104/protocol.json` | `e8da0f0098acd24a58664b83d0ba499b43aa9d597a36c52da6b3325c2c3c7bec` |
| `daily_market_development.csv` | `bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171` |
| `daily_volatility_development.csv` | `20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a` |

El objetivo sigue siendo **σ₇ poblacional de siete retornos logarítmicos diarios, ddof=0, horizonte de un día y sin anualización**. No se redefine como varianza realizada intradía. La barra incompleta del 24/03/2023 conserva su posición en el calendario. El loader verifica el derivado y admite sólo el prefijo de cada cohorte; no interpola ni comprime fechas.

| Cohorte | Entrenamiento | Validación | Uso |
|---|---|---|---|
| `small`, predeterminada | 161 ventanas, hasta 25/06/2023 | 83 fechas, 03/07…23/09/2023 | Primer perfil de recursos |
| `large`, opcional | 619 ventanas, hasta 25/09/2024 | 83 fechas, 03/10…24/12/2024 | Comprobar transferencia de coste a una cohorte mayor |

La cohorte grande utiliza el 01/01/2025 como marcador de calendario no disponible. No incorpora valores numéricos de 2025. Los archivos de 2025–2026 y las 608 fechas finales permanecen fuera de esta ejecución. Los datos de desarrollo tienen exposición histórica; el piloto no se presenta como un experimento confirmatorio intacto.

## Piloto registrado y presupuesto acotado

Se perfilan las seis referencias NAS —SAE, TimesNet, TiDE, TimeMixer, PatchTST e híbrido— y ocho cuerpos fijos —FCN-1D, ResNet-1D, Inception-1D, TSMixer, iTransformer, MLP16, LSTM32 y GRU32—. Son catorce modelos; la referencia NAS conserva anchos 64/64/64 y bits 010. La semilla del piloto es **274201**, nueva respecto de §§30–31.

La receta fija mantiene LR 0,001, WD 0,0001, batch 32 y warmup 0. TimesNet conserva `top_k=4` y seis kernels; TimeMixer, media móvil 7; el híbrido, kernel 3. No se usan recetas elegidas por HPO previo ni el error predictivo para seleccionar una política de ejecución. El autoencoder ejecuta dos épocas en cada una de sus tres fases; las demás referencias ejecutan dos épocas supervisadas. La validación se evalúa por su coste, no para elegir ganadores.

| Control | Asignación registrada |
|---|---:|
| Ajustes simultáneos | 1 |
| Hilos CPU / interop | 2 / 1 |
| Límite interno por etapa / externo por proceso | 300 / 360 s |
| Presupuesto global de admisión y supervisión | 900 s |
| Tiempo mínimo restante para admitir otro proceso | 5 s |
| RSS del grupo de procesos, observado por supervisor | 12 GiB |
| Asignador PyTorch y pico de memoria CUDA asignada | Menor de 12 GiB y 80% de la memoria total |

El control CUDA combina la fracción del asignador de PyTorch y el pico observado de memoria asignada; no contabiliza todas las asignaciones del driver. Los límites por fronteras de etapa y el muestreo RSS no son barreras físicas de asignación. El supervisor ajusta el tiempo del proceso al presupuesto global restante. El bootstrap de Python, la instalación, descarga, preflight y pruebas sintéticas se realizan antes de ese cronómetro y no están cubiertos por sus 900 segundos. El tiempo de conexión GPU del proveedor y la limpieza de procesos pueden añadir coste fuera de los ajustes; 900 segundos no promete que complete toda la lista ni una duración total exacta.

Los modelos, cohortes y la inclusión opcional de TimesNet extremo se sellan antes de acceder a los valores. Con las dos cohortes y el extremo habilitado, el dominio máximo es de treinta trabajos. Los procesos que no llegan a iniciarse por agotamiento global quedan como `not_attempted_campaign_budget`; no cuentan como ajustes realizados ni como fallos de entrenamiento.

Un timeout, OOM u otro error conserva su intento y coste. No hay sustitución automática de semilla, reducción de batch ni reintento. Si una actualización queda incompleta, el conteo total de pasos permanece desconocido y se conserva por separado el límite inferior conocido. Los perfiles incompletos no se convierten en entrenamientos terminados.

## Revisar y conservar la salida

Cada lanzamiento utiliza un directorio nuevo. El `summary.json` y los recibos completos permanecen en el almacenamiento elegido del entorno. El `scientific_report.json` selecciona explícitamente identidades de modelos, estado, pasos, tiempos, memoria y fases, sin copiar comandos, rutas internas ni diarios completos. No se hace push ni subida automática a GitHub u otra aplicación.

Antes de ejecutar, puede elegirse un volumen persistente que el usuario ya haya montado. En el almacenamiento efímero de Colab, los resultados pueden perderse al reiniciar o eliminar el entorno; conviene conservar el informe antes de cerrarlo. Los estados locales y fallos anteriores no se borran para repetir un lanzamiento.

Los recibos distinguen construcción, ajuste, evaluación, tiempo de proceso y memoria CUDA asignada/reservada. Las operaciones de GPU se sincronizan para medir tiempo de pared. Las horas de búsqueda se estimarán después de ampliar el perfilado y revisar la transferencia entre cohortes; **`gpu_budget_hours` permanece `null`**.

## Ejecución manual desde la fuente congelada

El notebook gestiona su Python 3.12.12 e instala las dependencias en un entorno aislado. En un servidor CUDA que ya disponga de Python 3.12, los siguientes comandos muestran el orden de una primera ejecución. El helper de descarga manual de iTransformer conserva su contrato de directorio nuevo; la verificación y reutilización de la caché al repetir celdas corresponde al notebook. El SHA de código debe coincidir con el publicado arriba; los directorios de salida deben ser nuevos.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3.12 -m venv work/gpu-venv
work/gpu-venv/bin/python -m pip install -r requirements-e2-training-gpu.lock.txt
work/gpu-venv/bin/python -m pip check
work/gpu-venv/bin/python scripts/run_e2_gpu_pilot.py --check-environment

work/gpu-venv/bin/python scripts/prepare_e2_pilot_snapshot.py --root .
work/gpu-venv/bin/python scripts/fetch_e2_itransformer_sources.py \
  --output work/author_sources/iTransformer
E2_ITRANSFORMER_SOURCE_ROOT=work/author_sources/iTransformer \
  work/gpu-venv/bin/python -m unittest discover -s tests \
  -p test_e2_cuda_models.py -v

work/gpu-venv/bin/python scripts/fetch_e2_development_sources.py \
  --output work/development_source
work/gpu-venv/bin/python scripts/run_e2_gpu_pilot.py \
  --data-root work/development_source \
  --registration experiments/e2_gpu_pilot_20261109/registration.json \
  --expected-registration-sha256 892ad148e1a261d16b178371a8f8c71ea3af352cc4c6ac4fea934df7dc22ba87 \
  --registration-commit 728277517be653be0d0ef10fb677a68adec22bb6 \
  --source-root-itransformer work/author_sources/iTransformer \
  --output work/gpu-pilot-nuevo
```

La selección explícita `--models GRU32 TimeMixer` permite un primer subconjunto dentro del dominio registrado. `--cohorts small large` añade la cohorte grande; `--include-largest` exige que TimesNet esté seleccionado y añade su extremo. Esas opciones se fijan antes del lanzamiento y no se adaptan a las puntuaciones observadas.

## Qué cierra esta preparación

Se entrega código CUDA, entorno dedicado, registro y notebook con activación explícita. La validación local acredita contratos de software en CPU. Los informes posteriores de [§33](33_piloto_gpu.md) y [§34](34_perfil_gpu_ampliado.md) incorporan los tiempos, la memoria y la completitud de las referencias y del extremo TimesNet en una Tesla T4 real; las dieciocho pruebas CUDA omitidas en el corte local conservan ese estado histórico.

Este corte de preparación no ejecutó la búsqueda NAS/HPO científica. Su avance posterior se documenta en [§36](36_avance_campana_cuda.md). Los controles, el benchmark y la evaluación final siguen pendientes. Después de seleccionar recetas y comprobar su variación se resolverá el dimensionamiento de R. Se conserva **`final_R=null`**, sin afirmación de potencia final o superioridad del candidato 11. Este cambio no reinterpreta la calibración condicional ni los fallos documentados en §31.
